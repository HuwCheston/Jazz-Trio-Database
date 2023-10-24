#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility classes, functions, and variables used in the onset detection process."""

import warnings
from typing import Generator

import librosa
import numpy as np
import soundfile as sf
from madmom.features import DBNDownBeatTrackingProcessor, RNNDownBeatProcessor
from mir_eval.onset import f_measure
from mir_eval.util import match_events
from scipy import signal as signal

from src import utils


FREQUENCY_BANDS = {
    'piano': dict(
        fmin=110,    # A2, the A lying two octaves below middle C4
        fmax=1760,    # A6, the A lying three octaves above middle C4
    ),
    'bass': dict(
        fmin=41,     # E1, the lowest string on a four-string double bass
        fmax=392,     # G4, two octaves above open G2 on a four-string double bass
    ),
    'drums': dict(
        fmin=2000,    # Approximate upper range of the snare drum, to be filtered out
        fmax=11000,    # Upper frequency range of a cymbal
    ),
    'mix': dict(
        fmin=20,    # Approximately the lower limit of human hearing
        fmax=20000,     # Approximately the upper limit of human hearing
    ),
}


class OnsetMaker:
    """Automatically detect onset and beat positions for each instrument in a single item in the corpus."""
    # I think these are justifiable: Corcoran and Frieler (2021) claim that, in jazz, a 32nd note will be perceived
    # as a grace note (part of) the next beat, rather than as part of the preceding beat.
    detection_note_values = dict(
        left=1/32,
        right=1/16
    )
    silence_threshold = 1 / 3  # Warn when more of a track is silent than this threshold
    # The threshold to use when matching onsets
    window = 0.05
    top_db = dict(
        piano=40,
        bass=30,
        drums=60,
    )
    # Define file paths
    references_dir: str = rf"{utils.get_project_root()}\references"
    data_dir: str = rf"{utils.get_project_root()}\data"
    reports_dir: str = rf"{utils.get_project_root()}\reports"

    def __init__(
            self,
            corpus_name: str = 'corpus_bill_evans',
            item: dict = None,
            **kwargs
    ):
        # Set inputs as class parameters
        self.item = item
        self.corpus_name = corpus_name
        # The sharpness of the filter
        self.order = kwargs.get('order', 30)
        # Define optimised defaults for onset_strength and onset_detect functions, for each instrument
        # These defaults were found through a parameter search against a reference set of onsets, annotated manually
        self.onset_strength_params, self.onset_detect_params = self.return_converged_paramaters()
        # Construct the default file paths where our audio is saved
        self.instrs = {
            'mix': rf'{self.data_dir}\raw\audio\{self.item["fname"]}.{utils.AUDIO_FILE_FMT}',
            'piano': rf'{self.data_dir}\processed\spleeter_audio\{self.item["fname"]}_piano.{utils.AUDIO_FILE_FMT}',
            'bass': rf'{self.data_dir}\processed\demucs_audio\{self.item["fname"]}_bass.{utils.AUDIO_FILE_FMT}',
            'drums': rf'{self.data_dir}\processed\demucs_audio\{self.item["fname"]}_drums.{utils.AUDIO_FILE_FMT}'
        }
        # Dictionary to hold arrays of onset envelopes for each instrument
        self.env = {}
        # Dictionary to hold arrays of detected onsets for each instrument
        self.ons = {}
        # Empty attribute to hold our tempo
        self.tempo = None
        # Empty dictionary to hold the percentage of silence in each track
        self.silent_perc = {}
        # Empty attribute to hold our matched onset dictionary
        self.summary_dict = {}
        # Empty attribute to hold our evaluation with a reference
        self.onset_evaluation: list = []
        # Load our audio file in when we initialise the item: we won't be changing this much
        if self.item is not None:
            self.audio = self._load_audio(**kwargs)

    def __repr__(self):
        """Overrides default method so that summary dictionary is printed when the class is printed"""
        return repr(self.summary_dict)

    def return_converged_paramaters(self) -> tuple[dict, dict]:
        def fmt(val):
            if isinstance(val, bool):
                return val
            elif int(val) == val:
                return int(val)
            else:
                return float(val)

        onset_detect_args = [
            *utils.return_function_kwargs(librosa.util.peak_pick),
            *utils.return_function_kwargs(DBNDownBeatTrackingProcessor.__init__),
            *utils.return_function_kwargs(self.beat_track_rnn),
            *utils.return_function_kwargs(librosa.onset.onset_detect)
        ]
        onset_strength_args = utils.return_function_kwargs(librosa.onset.onset_strength)
        od_fmt, os_fmt = {}, {}
        js = utils.load_json(
            fpath=fr'{self.references_dir}\parameter_optimisation\{self.corpus_name}', fname='converged_parameters'
        )
        for item in js:
            od_fmt[item['instrument']] = {k: fmt(v) for k, v in item.items() if k in onset_detect_args}
            os_fmt[item['instrument']] = {k: fmt(v) for k, v in item.items() if k in onset_strength_args}
            os_fmt[item['instrument']].update(FREQUENCY_BANDS[item['instrument']])
        return os_fmt, od_fmt

    def _load_audio(
            self,
            **kwargs
    ) -> dict:
        """Loads audio as a time-series array for all instruments + the raw mix.

        Wrapper around `librosa.load_audio`, called when class instance is constructed in order to generate audio for
        all instruments in required format. Keyword arguments are passed on to .load_audio

        Arguments:
            **kwargs: passed to `librosa.load_audio`

        Return:
            dict: each key-value pair corresponds to the loaded audio for one instrument, as an array

        Raises:
            UserWarning: when a greater portion of a track than given in OnsetMaker.silence_threshold is silent

        """
        # These arguments are passed in whenever this class is constructed, i.e. to __init__
        duration = kwargs.get('duration', None)
        offset = kwargs.get('offset', 0)
        res_type = kwargs.get('res_type', 'soxr_vhq')
        mono = kwargs.get('mono', True)
        dtype = kwargs.get('dtype', np.float64)
        # Empty dictionary to hold audio
        audio = {}
        # Iterate through all the source separated tracks
        for name, fpath in self.instrs.items():
            # Catch any UserWarnings that might be raised, usually to do with different algorithms being used to load
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                y, _ = librosa.load(
                    path=self._get_channel_override_fpath(name, fpath),
                    sr=utils.SAMPLE_RATE,
                    mono=mono,
                    offset=offset,
                    duration=duration,
                    dtype=dtype,
                    res_type=res_type,
                )
            # We apply the bandpass filter to the audio here in order to be sure that it's applied when detecting
            y = bandpass_filter(
                audio=y,
                lowcut=FREQUENCY_BANDS[name]['fmin'],
                highcut=FREQUENCY_BANDS[name]['fmax'],
                order=self.order
            )
            # Warn if our track exceeds silence threshold
            if name in self.top_db.keys():
                self.silent_perc[name] = self.get_silent_track_percent(y.T, top_db=self.top_db[name])
                if self.silent_perc[name] > self.silence_threshold:
                    warnings.warn(
                        f'item {self.item["fname"]}, track {name} exceeds silence threshold: '
                        f'({round(self.silent_perc[name], 2)} > {round(self.silence_threshold, 2)})'
                    )
            audio[name] = y
        return audio

    def _get_channel_override_fpath(
            self,
            name: str,
            fpath: str
    ) -> str:
        """Gets the filepath for an item, with any channel overrides specified.

        For instance, if we wish to use only the left channel for the double bass (and have specified "bass": "l" in the
        "channel_overrides" dictionary for this item in the corpus), this function will return the correct filepath
        pointing to the source-separated left channel file.

        Arguments:
            name (str): the name of the instrument
            fpath (str): the default filepath for the item (i.e. stereo audio)

        Returns:
            str: the overriden filepath if this is required and present locally, or the default (stereo) filepath if not

        """
        if 'channel_overrides' in self.item.keys():
            if name in self.item['channel_overrides'].keys():
                fp = fpath.replace(f'_{name}', f'-{self.item["channel_overrides"][name]}chan_{name}')
                if utils.check_item_present_locally(fp):
                    return fp
        return fpath

    def beat_track_rnn(
            self,
            starting_min: int = 100,
            # TODO: do we need to set this higher to account for extremely fast tracks, e.g. Peterson Tristeza?!
            starting_max: int = 300,
            use_nonoptimised_defaults: bool = False,
            audio_start: int = 0,
            audio_cutoff: int = 0,
            passes: int = 1,
            **kwargs
    ) -> np.array:
        """Tracks the position of crotchet beats in the full mix of a track using recurrent neural networks.

        Wrapper around `RNNDownBeatProcessor' and 'DBNDownBeatTrackingProcessor` from `madmom.features.downbeat` that
        allows for per-instrument defaults and multiple passes. A 'pass' refers to taking the detected crotchets from
        one run of the network, cleaning the results, extracting features from the cleaned array (e.g. minimum and
        maximum tempi), then creating a new network using these features and repeating the estimation process. This
        narrows down the range of tempo values that can be detected and increases the accuracy of detected crotchets
        substantially over a period of several passes.

        Arguments:
            starting_min (int, optional): the minimum possible tempo (in BPM) to use for the first pass, defaults to 100
            starting_max (int, optional): the maximum possible tempo (in BPM) to use for the first pass, defaults to 300
            use_nonoptimised_defaults (bool, optional): use default parameters over optimised, defaults to False
            audio_start (int, optional): start reading audio from this point (in total seconds)
            audio_cutoff (int, optional): stop reading audio after this point (in total seconds)
            passes (int, optional): the number of passes of the processer to use, defaults to 1
            **kwargs: passed to `madmom.features.downbeat.DBNDownBeatTrackingProcessor`

        Returns:
            np.array: an array of detected crotchet beat positions from the final pass
        """

        # If we're using defaults, set kwargs to an empty dictionary
        kws = self.onset_detect_params['mix'] if not use_nonoptimised_defaults else dict()
        # Update our default parameters with any kwargs we've passed in
        kws.update(**kwargs)
        passes = utils.try_get_kwarg_and_remove('passes', kws)
        # Define the start point for sampling the audio
        start = audio_start * utils.SAMPLE_RATE
        # If we haven't passed a cutoff, our last sample is the very end of the audio file
        if audio_cutoff == 0:
            end = len(self.audio['mix'])
        # Otherwise, create the cutoff by adding it to the start time and multiplying by the sample rate
        else:
            end = (audio_cutoff + audio_start) * utils.SAMPLE_RATE
        # If the cutoff is past the length of the track, rectify this here
        if end > len(self.audio['mix']):
            end = len(self.audio['mix'])
        # Slice the audio file according to the start and end point
        samples = self.audio['mix'][start: end]

        def tracker(
                tempo_min_: int = 100,
                tempo_max_: int = 300,
                **kws_
        ) -> tuple[np.array, np.array]:
            """Wrapper around classes from `madmom.features.downbeat`"""

            # Catch VisibleDeprecationWarnings that appear when creating the processor
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', np.VisibleDeprecationWarning)
                # Create the tracking processor
                proc = DBNDownBeatTrackingProcessor(
                    min_bpm=tempo_min_,
                    max_bpm=tempo_max_,
                    fps=100,
                    **kws_
                )
                # Fit the processor to the audio
                act = RNNDownBeatProcessor()(samples)
                # Return both the detected beat timestamps and the estimated position in a bar
                return proc(act)[:, 0], proc(act)[:, 1]

        # Create the first pass: this is designed to use a very low threshold and wide range of tempo values, enabling
        # the tempo to fluctuate a great deal; we will then use these results to narrow down the tempo in future passes
        timestamps, metre_positions = tracker(
            tempo_min_=starting_min,
            tempo_max_=starting_max,
            observation_lambda=2,
            beats_per_bar=[self.item['time_signature']],
            # We don't pass in our **kwargs here
            **dict(threshold=0, transition_lambda=75)
        )
        # Start creating our passes, progressively cleaning the data from our first pass
        # We count our initial pass as our first pass, so passes=2 will only iterate through this loop once
        for i in range(1, passes):
            # Extract the BPM value for each IOI obtained from our most recent pass
            bpms = np.array([60 / p for p in np.diff(timestamps)])
            # Clean any outliers from our BPMs by removing values +/- 1.5 * IQR
            clean = utils.iqr_filter(bpms)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                # Try and get our new tempo from the lower and upper quantile of the detected BPM values
                try:
                    tempo_min, tempo_max = np.nanpercentile(clean, 25), np.nanpercentile(clean, 75)
                # If we didn't detect any onsets, the above lines will throw an error, so return an empty array
                except ValueError:
                    warnings.warn(f'item {self.item["fname"]}, did not detect any beats in mix')
                    return np.array([0])
            # Create the new pass, using the new maximum and minimum tempo
            timestamps, metre_positions = tracker(
                tempo_min_=tempo_min,
                tempo_max_=tempo_max,
                observation_lambda=16,
                beats_per_bar=[self.item['time_signature']],
                # Now we pass in our keyword arguments
                **kws
            )
        # Set the tempo value using the crotchet beat positions from our previous pass
        self.tempo = calculate_tempo(timestamps)
        return timestamps, metre_positions

    def onset_strength(
            self,
            instr: str,
            aud: np.array = None,
            use_nonoptimised_defaults: bool = False,
            **kwargs
    ) -> np.array:
        """Generates an onset strength envelope for a given instrument

        Wrapper around `librosa.onset.onset_strength` that allows for the use of per-instrument defaults. Any \*\*kwargs
        should be accepted by this function, and can be passed to override optimised per-instrument defaults.

        Arguments:
            instr (str): the name of the instrument to generate an onset strength envelope for
            aud (np.array, optional): an audio time-series array to generate the envelope for
            use_nonoptimised_defaults (bool, optional): whether to use default parameters, defaults to False
            **kwargs: any additional keyword arguments must be accepted by `librosa.onset.onset_strength`

        Returns:
            np.array: the onset strength envelope as an array

        """
        # If we haven't passed any audio in, then construct this using the instrument name that we've passed
        if aud is None:
            aud = self.audio[instr]
            # If the audio is stereo, convert it to mono for the onset strength calculation
            if len(aud.shape) == 2:
                aud = aud.mean(axis=1)

        # If we're using defaults, set kwargs to an empty dictionary
        kws = self.onset_strength_params[instr] if not use_nonoptimised_defaults else dict()
        # Update our default parameters with any kwargs we've passed in
        kws.update(**kwargs)
        # Suppress any user warnings that Librosa might throw
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            # Return the onset strength envelope using our default (i.e. hard-coded) sample rate and hop length
            return librosa.onset.onset_strength(
                y=aud,
                sr=utils.SAMPLE_RATE,
                hop_length=utils.HOP_LENGTH,
                # We pass our minimum and maximum frequencies in to this function
                **kws
            )

    def onset_detect(
            self,
            instr: str,
            aud: np.array = None,
            env: np.array = None,
            units: str = 'time',
            use_nonoptimised_defaults: bool = False,
            **kwargs
    ) -> np.array:
        """Detects onsets in an audio signal.

        Wrapper around `librosa.onset.onset_detect` that enables per-instrument defaults to be used. Arguments passed as
        kwargs should be accepted by `librosa.onset.onset_detect`, except for rms: set this to True to use a custom
        energy function when backtracking detected onsets to local minima. Other keyword arguments overwrite
        current per-instrument defaults.

        Arguments:
            instr (str): the name of the instrument to detect onsets in
            aud (np.array, optional): an audio time-series to detect onsets in
            env (np.array, optional): the envelope to use when detecting onsets
            units (str, optional): the units to return detected onsets in, defaults to 'time',
            use_nonoptimised_defaults (bool, optional): whether to use default parameters, defaults to False
            **kwargs: additional keyword arguments passed to `librosa.onset.onset_detect`

        Returns:
            np.array: the position of detected onsets

        """
        # If we haven't passed any input audio, get this now
        if aud is None:
            aud = self.audio[instr]
            # If the audio is stereo, convert it to mono for the onset strength calculation
            if len(aud.shape) == 2:
                aud = aud.mean(axis=1)

        # If we haven't passed an input onset envelope, get this now
        if env is None:
            env = self.env[instr]

        # Update the default parameters for the input instrument with any kwargs we've passed in
        self.onset_detect_params[instr].update(**kwargs)
        # The RMS argument can't be passed to .onset_detect(). We need to try and get it, then remove it from our dict
        rms = utils.try_get_kwarg_and_remove(
            kwarg='rms',
            kwargs=self.onset_detect_params[instr],
            default_=False
        )
        # If we're using defaults, set kwargs to an empty dictionary
        kws = self.onset_detect_params[instr] if not use_nonoptimised_defaults else dict()
        # Update our default parameters with any kwargs we've passed in
        kws.update(**kwargs)
        # If we're backtracking onsets from the picked peak
        if self.onset_detect_params[instr]['backtrack']:
            # If we want to use RMS values instead of our onset envelope when back tracking onsets
            if rms:
                energy = librosa.feature.rms(S=np.abs(librosa.stft(self.audio[instr].mean(axis=1))))[0]
            else:
                energy = env
            return librosa.onset.onset_detect(
                y=aud,
                sr=utils.SAMPLE_RATE,
                hop_length=utils.HOP_LENGTH,
                units=units,
                energy=energy,
                onset_envelope=env,
                **kws
            )
        # If we're not backtracking, and using the picked peaks themselves
        else:
            return librosa.onset.onset_detect(
                y=aud,
                sr=utils.SAMPLE_RATE,
                hop_length=utils.HOP_LENGTH,
                units=units,
                onset_envelope=env,
                **kws
            )

    def generate_click_track(
            self,
            instr: str,
            *args
    ) -> None:
        """Renders detected onsets to a click sound and outputs, combined with the original audio.

        Arguments:
            instr (str): the name of the instrument to render audio from
            *args (np.array): arrays of detected onsets to render to audio

        Returns:
            None

        """
        # Get our onset list and our filename
        onsets_list = [self.ons[instr], *args]
        fname = rf'{self.reports_dir}\click_tracks\{self.item["fname"]}_{instr}_beats.{utils.AUDIO_FILE_FMT}'
        # Create the click track maker class and then generate the click track using the above variables
        click_track_cls = _ClickTrackMaker(audio=self.audio[instr])
        click_track_audio = click_track_cls.generate_audio(onsets_list)
        # Create the audio file and save into the click tracks directory
        with open(fname, 'wb') as f:
            sf.write(f, click_track_audio, utils.SAMPLE_RATE)

    def compare_onset_detection_accuracy(
            self,
            ref: np.array = None,
            fname: str = None,
            instr: str = None,
            onsets: list[np.array] = None,
            onsets_name: list[str] = None,
            audio_cutoff: int = None,
            window: float = None,
            **kwargs
    ) -> dict:
        """Evaluates onset detection algorithm against reference onsets.

        For every onset detected by an algorithm, attempt to match these to the nearest onset in a reference set
        (usually obtained from manual annotation). Then, construct a summary dictionary, containing summary statistics
        relating to the precision, recall, and accuracy of the detection. For more information on the evaluation
        procedure, see `mir_eval.onset.f_measure`.

        At least one of ref or fname must be passed: ref must be an array of onset times, in seconds; fname must be a
        path to a text file containing onset times, with one onset per line. If both ref and fname are passed (don't do
        this), ref will take priority.

        Arguments:
            ref (np.array): an array of reference onsets (in seconds) to use for evaluation
            fname (str): the file path to a reference set of onsets, one onset per line
            instr (str): the name of an instrument or track
            onsets (list[np.array]): a list of arrays, each array should be the results from one algorithm
            onsets_name (list[str]): a list of names that should match with the algorithm results in onsets
            window (float): the size of the window used for matching each onset to a reference
            audio_cutoff (int, optional): stop reading audio after this point (in total seconds)
            **kwargs: additional key-value pairs passed to the returned summary dictionary

        Yields:
            dict: each dictionary contains summary statistics for one evaluation

        """
        if ref is None and fname is None:
            raise AttributeError('At least one of ref, fname must be provided')
        # If we haven't passed in reference onsets but we have passed in a file path, generate the array from the file
        elif ref is None and fname is not None:
            ref = np.loadtxt(fname, ndmin=1, usecols=0)
        # If we haven't provided any names for our different onset lists, create these now
        if onsets_name is None:
            onsets_name = [None for _ in range(len(onsets))]
        # If we haven't provided a window value, use our default (50 ms)
        if window is None:
            window = self.window
        # Iterate through all the onset detection algorithms passed in
        for (name, estimate) in zip(onsets_name, onsets):
            with warnings.catch_warnings():
                # If we've passed a cutoff value, remove all reference and estimated onsets above this
                if audio_cutoff is not None:
                    ref = np.array([i for i in ref if i < audio_cutoff])
                    estimate = np.array([i for i in estimate if i < audio_cutoff])
                # If we have no onsets, both mir_eval and numpy will throw warnings, so catch them here
                warnings.simplefilter('ignore', RuntimeWarning)
                warnings.simplefilter('ignore', UserWarning)
                # Calculate the mean asynchrony between the reference and estimate onsets
                matched = match_events(ref, estimate, window)
                mean_async = np.nanmean([estimate[e] - ref[r] for r, e in matched])
                # Generate the F, precision, and recall values from mir_eval and yield as a dictionary
                # TODO: I wonder if we could get rid of the dependency on mir_eval here? This is a simple function.
                f, p, r = f_measure(ref, estimate, window=window)
            yield {
                'name': name,
                'instr': instr,
                'f_score': f,
                'precision': p,
                'recall': r,
                'mean_asynchrony': mean_async,
                'track': self.item['track_name'],
                'fraction_matched': len(matched) / len(ref),
                **kwargs
            }

    def match_onsets_and_beats(
            self,
            beats: np.array,
            onsets: np.array = None,
            instr: str = None,
            use_hard_threshold: bool = False,
            detection_note_values: dict = None
    ) -> np.array:
        """Matches event onsets with crotchet beat locations.

        For every beat in the iterable `beats`, find the closest proximate onset in the iterable `onsets`, within a
        given window. If no onset can be found within this window, set the matched onset to NaN. Window type can either
        be a hard, fixed value by setting `use_hard_threshold`, or flexible and dependant on a particular rhythmic value
        within the underlying tempo (set using the `detection_note_value` class attribute). The latter option is
        recommended and used as a default, given that hard thresholds for matching onsets at one tempo may not be
        appropriate for other tempi.

        Examples:
            >>> om = OnsetMaker()
            >>> bea = np.array([0, 0.5, 1.0, 1.5])
            >>> ons = np.array([0.1, 0.6, 1.25, 1.55])
            >>> print(om.match_onsets_and_beats(beats=bea, onsets=ons, use_hard_threshold=True, threshold=0.1))
            np.array([0.1 0.6 nan 1.55])

            >>> om = OnsetMaker()
            >>> om.tempo = 160
            >>> bea = np.array([0, 0.5, 1.0, 1.5])
            >>> ons = np.array([0.1, 0.6, 1.25, 1.55])
            >>> print(om.match_onsets_and_beats(beats=bea, onsets=ons, use_hard_threshold=False))
            np.array([nan nan nan 1.55])

        Arguments:
            beats (np.ndarray): iterable containing crotchet beat positions, typically tracked from the full mix
            onsets (np.ndarray): iterable containing onset positions, typically tracked from a source separated file
            instr (str): the name of an instrument, to be used if onsets is not provided
            use_hard_threshold (bool): whether to use a hard or tempo-dependent (default) threshold for matching onsets
            detection_note_values (dict): dictionary of note values to use either side of crotchet beat, e.g. 1/32, 1/8

        Returns:
            np.array: the matched onset array, with shape == len(beats)

        Raises:
            AttributeError: if neither onsets or instr are provided

        """
        def matcher() -> Generator:
            """Matching function. Matches the closest onset to each beat, within a window, returns a generator"""
            for beat in beats:
                # Subtract our onset array from our beat
                sub = onsets - beat
                re = []
                # Get the 'left onsets'; those played *before* the beat, and threshold them
                left = sub[sub < 0][np.abs(sub[sub < 0]) < l_threshold]
                # If we have left onsets, threshold them and get the one closest to the beat
                if len(left) > 0:
                    re.append(left.max())
                # Get the 'right onsets'; those played *after* the beat, and threshold them
                right = sub[sub >= 0][np.abs(sub[sub >= 0]) < r_threshold]
                # If we have right onsets, append the one closest to the beat to our list
                if len(right) > 0:
                    re.append(right.min())
                # Get the closest match from our left and right array
                try:
                    arr = np.array(re)
                    closest = arr[np.abs(arr).argmin()]
                    yield onsets[np.equal(sub, closest)][0]
                # If our array is empty (no left or right match), we'll get an error, so catch and return nan
                except ValueError:
                    yield np.nan

        # If we haven't passed an onsets list but we have passed an instrument as a string, try and get the onset list
        if onsets is None and instr is not None:
            onsets = self.ons[instr]
        # If we haven't passed an onset list or instrument string, raise an error
        if onsets is None and instr is None:
            raise AttributeError('At least one of onsets, instr must be provided')
        # Define the onset detection threshold: either hard or tempo-adjustable
        if use_hard_threshold:
            l_threshold = self.window
            r_threshold = self.window
        else:
            if detection_note_values is None:
                detection_note_values = self.detection_note_values
            l_threshold = ((60 / self.tempo) * 4) * detection_note_values['left']
            r_threshold = ((60 / self.tempo) * 4) * detection_note_values['right']
        # Return the list of matched onsets below our threshold
        return np.fromiter(matcher(), count=len(beats), dtype=np.float64)

    def generate_matched_onsets_dictionary(
            self,
            beats: np.array,
            onsets_list: list[np.array] = None,
            instrs_list: list = None,
            **kwargs
    ) -> dict:
        """Matches onsets from multiple instruments with crotchet beat positions and returns a dictionary.

        Wrapper function for `OnsetMaker.match_onsets_and_beats`. `onsets_list` should be a list of arrays corresponding
        to onset positions tracked from multiple source-separated instruments. These will then be sent individually to
        `OnsetMaker.match_onsets_and_beats` and matched with the provided `beats` array, then returned as the values in
        a dictionary, where the keys are identifiers passed in `instrs_list` (or numerical values, if this iterable is
        not passed). Any ``**kwargs`` will be passed to `OnsetMaker.match_onsets_and_beats`.

        Examples:
            >>> om = OnsetMaker()
            >>> bea = np.array([0, 0.5, 1.0, 1.5])
            >>> ons = [
            >>>     np.array([0.1, 0.6, 1.25, 1.55]),
            >>>     np.array([0.05, 0.45, 0.95, 1.45]),
            >>> ]
            >>> instrs = ['instr1', 'instr2']
            >>> print(om.generate_matched_onsets_dictionary(
            >>>     beats=bea, onsets_list=ons, instrs_list=instrs, use_hard_threshold=True, threshold=0.1)
            >>> )
            {
                'beats': array([0. , 0.5, 1. , 1.5]),
                'instr1': array([0.1 , 0.6 ,  nan, 1.55]),
                'instr2': array([0.05, 0.45, 0.95, 1.45])
            }

        Arguments:
            beats (np.array): iterable containing crotchet beat positions, typically tracked from the full mix
            onsets_list (list[np.array]): iterable containing arrays of onset positions
            instrs_list (list[str]): iterable containing names of instruments
            **kwargs: arbitrary keyword arguments, passed to `OnsetMaker.match_onsets_and_beats`

        Returns:
            dict: keys are instrument names, values are matched arrays

        Raises:
            AttributeError: if neither onsets_list or instrs_list are passed

        """
        # Get our required arguments
        if onsets_list and instrs_list is None:
            raise AttributeError('At least one of onsets_list and instrs_list must be provided')
        if onsets_list is None:
            onsets_list = [self.ons[ins_] for ins_ in instrs_list]
        if instrs_list is None:
            instrs_list = [i for i in range(len(onsets_list))]
        # Create the dictionary of crotchet beats and matched onsets, then return
        ma: dict = {'beats': beats}
        ma.update({
            name: self.match_onsets_and_beats(beats=beats, onsets=ons_, **kwargs) for ons_, name in
            zip(onsets_list, instrs_list)
        })
        return ma

    @staticmethod
    def get_nonsilent_sections(
            aud: np.array,
            thresh: float = 1,
            **kwargs
    ) -> np.array:
        """Returns the sections of a track which are not silent.

        Wrapper function for `librosa.effects.split` that returns slices of a given audio track that are not silent.
        Slices are only considered not silent if their duration is above a reference threshold, given in seconds: this
        is to prevent the parsing of many small slices of audio.

        Arguments:
            aud (np.array): array of audio, read in during construction of the OnsetMaker class
            thresh (float): value in seconds used when parsing slices
            **kwargs: arbitrary keyword arguments, passed to `librosa.effects.split`

        Returns:
            np.array: rows corresponding to sections of non-silent audio

        """
        # Get the sections of the track that are not silent
        non_silent = librosa.effects.split(
            librosa.util.normalize(aud).T,
            hop_length=utils.HOP_LENGTH,
            **kwargs
        )
        # Convert the non-silent sections (in frames) to time stamps
        to_ts = lambda s: s / utils.SAMPLE_RATE
        li = np.array([(to_ts(se[0]), to_ts(se[1])) for se in non_silent if to_ts(se[1]) - to_ts(se[0]) > thresh])
        # Combine slices of non-silent audio if the distance between the right and left edges is below the threshold
        try:
            roll = np.roll(li, 1)[1:, :]
        except IndexError:
            pass
        else:
            # Check the distance between the last element on one row and the first element on the next row
            for row in roll:
                # This will combine the slices if the distance is below threshold
                if row[1] - row[0] < thresh:
                    li[np.where(li == row[0])] = row[1]
        finally:
            return li

    def get_silent_track_percent(
            self,
            aud: np.array = None,
            silent: np.array = None,
            **kwargs
    ) -> float:
        """Returns the fraction of a track which is silent.

        Arguments:
            aud (np.array): array of audio, read in during construction of the OnsetMaker class
            silent (np.array): array of non-silent audio slices, returned from OnsetMaker.get_nonsilent_sections
            **kwargs: arbitrary keyword arguments, passed to `OnsetMaker.get_nonsilent_sections`

        Returns:
            float: the fraction of a track which is silent, e.g. 1 == a completely silent track

        Raises:
            AttributeError: if neither aud or silent are passed

        """
        # Catch any issues with arguments not being passed in
        if silent is None and aud is None:
            raise AttributeError('At least one of silent and aud must be passed!')
        # If we haven't generated our silent sections from our audio yet, do this now (and pass in any kwargs)
        if silent is None:
            silent = self.get_nonsilent_sections(aud=aud, **kwargs)

        # Get the overall duration of the track
        duration = librosa.get_duration(
            y=aud.T,
            sr=utils.SAMPLE_RATE,
            hop_length=utils.HOP_LENGTH
        )
        # Try to return the fraction of the track which is silent
        try:
            return 1 - (np.sum(silent[:, 1] - silent[:, 0]) / duration)
        # If sil is an empty list, that means the whole track is silent. So catch the resulting error and return 1
        except IndexError:
            return 1

    def remove_onsets_in_silent_passages(
            self,
            onsets: np.array,
            instr: str = None,
            silent: np.array = None,
            **kwargs
    ) -> np.array:
        """Removes onsets if they occurred during a portion of a track which was silent.

        For a given array of event onsets and a given array of non-silent audio slice timestamps, returns only those
        onsets which occurred during a portion of an audio track deemed not to be silent. This prevents any spurious
        onsets detected by Librosa from being included in an analysis.

        Examples:
            >>> om = OnsetMaker()
            >>> non_silent = np.array(
            >>>     [
            >>>         [0, 5],
            >>>         [10, 15]
            >>>     ]
            >>> )
            >>> ons_ = np.array([0.1, 0.6, 5.5, 12.5, 17.5])
            >>> print(om.remove_onsets_in_silent_passages(onsets=ons_, silent=non_silent))
            array([0.1, 0.6, 12.5])


        Arguments:
            onsets (np.array): an array of event onsets
            instr (str): the name of an instrument
            silent (np.array): an array of non-silent audio slices, returned from `OnsetMaker.get_nonsilent_sections`
            **kwargs: arbitrary keyword arguments, passed to `OnsetMaker.get_nonsilent_sections`

        Returns:
            np.array: an array of onset timestamps with those occurring during a silent slice removed

        Raises:
            AttributeError: if neither `silent` or `instr` are passed

        """
        # Catch errors if we haven't passed in the required arguments
        if silent is None and instr is None:
            raise AttributeError('At least one of silent or instr must be passed!')
        # If we haven't passed in our silent sections but have passed in our instrument name, get the silent sections
        if silent is None:
            silent = self.get_nonsilent_sections(
                self.audio[instr],
                top_db=self.top_db[instr],
                **kwargs
            )
        # Remove onsets from our onset list if they occurred during a silent section of the track and return the array
        clean = []
        for slice_ in silent:
            clean.extend([ons for ons in onsets if slice_[0] < ons < slice_[1]])
        return np.array(clean)

    def process_separated_audio(
            self,
            generate_click: bool,
            remove_silence: bool = True,
    ) -> None:
        """Process the separated audio for all of our individual instruments (piano, bass, drums)

        This is the central function for running processing on each source-separated audio file. It will generate an
        onset envelope, detect onsets within it, remove onsets from when the track was silent, compare the detections
        to a reference file (if this exists), generate a click track (if this is required), and match the detected
        onsets to the nearest crotchet beat. This function must be called AFTER `OnsetMaker.process_mixed_audio`, to
        ensure that the crotchet beat positions have been detected correctly in the raw audio mix.

        Parameters:
            generate_click (bool): whether to generate an audio click track
            remove_silence (bool): whether to remove onsets from portions of a track deemed to be silent by librosa

        """
        # Iterate through each instrument name
        for ins in utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys():
            # Get the onset envelope for this instrument
            self.env[ins] = self.onset_strength(ins, use_nonoptimised_defaults=False)
            # Get the onsets
            self.ons[ins] = self.onset_detect(ins, env=self.env[ins], use_nonoptimised_defaults=False)
            # If we're removing onsets when the audio is silent, do that now
            if remove_silence:
                sil = self.get_nonsilent_sections(aud=self.audio[ins], top_db=self.top_db[ins])
                self.ons[ins] = self.remove_onsets_in_silent_passages(onsets=self.ons[ins], silent=sil)
            # If we have manually annotated onsets for this item, try and evaluate the accuracy of detected onsets
            try:
                eval_ = list(self.compare_onset_detection_accuracy(
                    fname=rf'{self.references_dir}\manual_annotation\{self.item["fname"]}_{ins}.txt',
                    onsets=[self.ons[ins]],
                    onsets_name=['optimised_librosa'],
                    instr=ins,
                ))
            except FileNotFoundError:
                pass
            else:
                self.onset_evaluation.append(eval_)
            # Match the detected onsets with our detected crotchet beats from the mad-mom output
            matched = self.match_onsets_and_beats(beats=self.ons['mix'], onsets=self.ons[ins])
            # Output our click track of detected beats + matched onsets
            if generate_click:
                self.generate_click_track(ins, matched)

    def process_mixed_audio(
            self,
            generate_click: bool,
    ) -> None:
        """Process the raw audio mix, i.e. with all tracks together.

        This is the central function for running processing on the mixed audio. It will generate an onset envelope,
        detect crotchets within it using both predominant local pulse estimation and recurrent neural networks,
        compare the detections to a reference file (if this exists), and generate a click track (if this is
        required). This function should be called before `OnsetMaker.process_separated_audio`, to ensure that the
        crotchet beat positions are present before matching these to onsets detected in the source-separated tracks.

        Parameters:
            generate_click (bool): whether to generate an audio click track

        """
        # TODO: can we make the mix methods their own class, created in this function?
        # Track the beats using recurrent neural networks
        timestamps, metre_auto = self.beat_track_rnn(use_nonoptimised_defaults=False)
        self.ons['mix'] = timestamps
        # Estimate the metre automatically using the neural network results
        self.ons['metre_auto'] = metre_auto
        self.ons['downbeats_auto'] = self.extract_downbeats(timestamps, metre_auto)
        # Estimate the metre using a known downbeat and time signature
        self.ons['metre_manual'] = self.metre_from_annotated_downbeat(timestamps)
        self.ons['downbeats_manual'] = self.extract_downbeats(timestamps, self.ons['metre_manual'])
        # Warn if we're getting different results for automatic and manual metre detection
        if not all(self.ons['metre_auto'] == self.ons['metre_manual']):
            warnings.warn(f'item {self.item["fname"]}: manual and automatic metre detection diverge')
        # Try and get manual annotations for our crotchet beats, if we have them
        try:
            eval_ = list(self.compare_onset_detection_accuracy(
                fname=rf'{self.references_dir}\manual_annotation\{self.item["fname"]}_mix.txt',
                onsets=[self.ons['mix']],
                onsets_name=['madmom'],
                instr='mix',
            ))
        except FileNotFoundError:
            pass
        else:
            self.onset_evaluation.append(eval_)
        # Generate the click track for the tracked beats, including the manually-annotated downbeats
        if generate_click:
            self.generate_click_track('mix', self.ons['downbeats_manual'])

    def finalize_output(
            self
    ) -> None:
        """Finalizes the output by cleaning up leftover files and setting any final attributes"""
        # Match the detected onsets together with the detected beats to generate our summary dictionary
        self.summary_dict = self.generate_matched_onsets_dictionary(
            beats=self.ons['mix'],
            onsets_list=[self.ons['piano'], self.ons['bass'], self.ons['drums']],
            instrs_list=['piano', 'bass', 'drums'],
            use_hard_threshold=False
        )
        self.summary_dict.update(dict(metre_auto=self.ons['metre_auto'], metre_manual=self.ons['metre_manual']))
        # Delete the raw audio as it will take up a lot of space when serialised
        del self.audio

    def metre_from_annotated_downbeat(
            self,
            timestamps_arr: np.array,
    ) -> np.array:
        """Constructs an array of metre positions from a known downbeat and time signature"""
        downbeat_position = np.argmin(np.abs(timestamps_arr - self.item['first_downbeat']))
        backwards_metre = []
        # First, we work backwards from the known downbeat
        val = self.item['time_signature']
        for i in reversed(range(downbeat_position)):
            if i >= 0:
                backwards_metre.append(float(val))
                val -= 1
                if val < 1:
                    val = self.item['time_signature']
        # Now, we work forwards from the annotated downbeat
        val = 1
        forwards_metre = []
        for i in range(len(timestamps_arr)):
            if i >= downbeat_position:
                forwards_metre.append(float(val))
                val += 1
                if val > self.item['time_signature']:
                    val = 1
        # Concatenate both arrays together
        return np.array(list(reversed(backwards_metre)) + forwards_metre)

    @staticmethod
    def extract_downbeats(
            beat_timestamps: np.array,
            beat_positions: np.array
    ) -> tuple[np.array, np.array]:
        """Takes in arrays of beat onsets and bar positions and returns the downbeats of each bar"""
        # Combine timestamps and bar positions into one array
        comb = np.array([beat_timestamps, beat_positions]).T
        # Create the boolean mask
        mask = (comb[:, 1] == 1)
        # Subset on the mask to get downbeats only and return
        return comb[mask, 0]


class _ClickTrackMaker:
    width = 200
    start_freq = 750
    volume_threshold = 1/3
    order = 20    # Lower than the value used for the stems as less precise filtering is needed here

    def __init__(self, audio: np.array):
        # Convert the input audio to mono, if we haven't done this already
        if len(audio.shape) == 2:
            audio = audio.mean(axis=1)
        self.audio = audio

    def generate_audio(
            self,
            onsets_list: list[np.array],
    ) -> np.array:
        """Renders detected onsets to a click sound and combines with the original audio.

        Takes in a list of reference onset arrays, converts these to audible clicks, applies a bandpass filter (to make
        telling different onsets apart easier), filters the original audio to the frequencies considered when detecting
        onsets, then combines filtered original audio + click to a new audio track.

        Arguments:
            onsets_list (list[np.array]): a list containing arrays of detected onsets

        Returns:
            np.array: the click audio track that can be rendered to a file using soundfile, Librosa, etc.

        """

        # Iterate through all of our passed onsets, with a counter to increase click output frequency
        clicks = [self.clicks_from_onsets(self.start_freq * n, onsets) for n, onsets in enumerate(onsets_list, 1)]
        # Sum the click signals together and lower the volume by the given threshold
        clicks = sum(clicks) * self.volume_threshold
        # Sum the combined click signal with the audio and return
        return self.audio + clicks

    def clicks_from_onsets(self, freq, onsets, **kwargs) -> np.array:
        """Renders detected onsets to a click sound with a given frequency"""
        return bandpass_filter(
            audio=librosa.clicks(
                times=onsets[~np.isnan(onsets)],  # Remove any NaN values obtained from matching onsets & beats
                sr=utils.SAMPLE_RATE,
                hop_length=utils.HOP_LENGTH,
                length=len(self.audio),
                click_freq=freq,
                click_duration=0.2,
                **kwargs
            ),
            lowcut=freq - self.width,
            highcut=freq + self.width,
            # We can pass in a lower order value here to reduce the amount of time the filtering takes
            order=self.order,
            # We don't need to apply any fading to our click track, it'll just take extra time
            fade_dur=0
        )


def bandpass_filter(
        audio: np.array,
        lowcut: int,
        highcut: int,
        order: int = 30,
        pad_len: float = 1.0,
        fade_dur: float = 0.5,
) -> np.array:
    """Applies a bandpass filter with given low and high cut frequencies to an audio signal.

    Arguments:
        audio (np.array): the audio array to filter
        lowcut (int): the lower frequency to filter
        highcut (int): the higher frequency to filter
        order (int): the sharpness of the filter, defaults to 30
        pad_len (float): the number of seconds to pad the audio by, defaults to 1
        fade_dur (float): the length of time to fade the audio in and out by

    Returns:
        np.array: the filtered audio array

    """
    # Create the filter: we use a second-order butterworth filter here
    filt = signal.butter(
        N=order,
        Wn=[lowcut, highcut],
        output='sos',
        btype='bandpass',
        analog=False,
        fs=utils.SAMPLE_RATE,
    )
    # Apply the filter to the audio (with padding)
    filtered = signal.sosfiltfilt(
        filt,
        audio,
        padtype='constant',
        padlen=int(utils.SAMPLE_RATE * pad_len)
    )
    # If we don't want to apply any fading to the audio, return it straight away
    if fade_dur == 0:
        return filtered
    # Else, create the fade curve: we use a log curve so that the earliest events fade quickest
    dur = int(fade_dur * utils.SAMPLE_RATE)
    fade_curve = 1.0 - np.logspace(0, -2, num=dur)
    # Apply the fade to the start and end of the audio using the fade curve (fliped for a fade out)
    filtered[:dur] *= fade_curve
    filtered[-dur:] *= np.flip(fade_curve)
    # Return the audio with the fade applied
    return filtered


def calculate_tempo(
        pass_: np.ndarray
) -> float:
    """Extract the average tempo from an array of times corresponding to crotchet beat positions"""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(np.array([60 / p for p in np.diff(pass_)]))


if __name__ == '__main__':
    import logging

    # Initialise the logger
    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format=fmt)
    # Load in the first track from the Bill Evans corpus, for demonstration
    corpus = utils.CorpusMaker.from_excel(fname='corpus_bill_evans')
    corpus_item = corpus.tracks[0]
    # Create the OnsetMaker class instance for this item in the corpus
    made = OnsetMaker(corpus_name='corpus_bill_evans', item=corpus_item)
    # Run our processing on the mixed audio
    logger.info(f'processing audio mix for item {corpus_item["mbz_id"]}, track name {corpus_item["track_name"]} ...')
    made.process_mixed_audio(generate_click=False)
    # Run our processing on the separated audio
    logger.info(f'processing audio stems for item {corpus_item["mbz_id"]}, track name {corpus_item["track_name"]} ...')
    made.process_separated_audio(generate_click=False, remove_silence=True)
    # Clean up the results
    made.finalize_output()
    logger.info(f'... processing finished !')
