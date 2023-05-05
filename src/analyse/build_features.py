import json
import os
import sys
import warnings

import basic_pitch.inference as bp
import librosa
import numpy as np
import pandas as pd
import scipy.signal as signal
import scipy.stats as stats
import soundfile as sf
import tensorflow as tf
from basic_pitch import ICASSP_2022_MODEL_PATH
from mir_eval.onset import f_measure
from mir_eval.util import match_events

BASIC_PITCH_MODEL = tf.saved_model.load(str(ICASSP_2022_MODEL_PATH))


class HidePrints:
    """
    Helper class that prevents a function from printing to stdout when used as a context manager
    """

    def __enter__(
            self
    ) -> None:
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(
            self,
            exc_type,
            exc_val,
            exc_tb
    ) -> None:
        sys.stdout.close()
        sys.stdout = self._original_stdout


class OnsetDetectionMaker:
    # These values are hard-coded and used throughout: we probably shouldn't change them
    sample_rate = 44100
    hop_length = 512
    # The threshold to use when matching onsets
    window = 0.05
    # Define optimised defaults for onset_strength and onset_detect functions, for each instrument
    # These defaults were found through a parameter search against a reference set of onsets, annotated manually
    # TODO: we need to expand the number of recordings used in our parameter search
    # These are passed whenever onset_strength is called for this particular instrument's audio
    # onset_strength_params = {
    #     'piano': dict(
    #         fmin=110,  # Minimum frequency to use
    #         fmax=4100,  # Maximum frequency to use
    #         center=False,  # Use left-aligned frame analysis in short term Fourier transform
    #         max_size=1,  # Size of local maximum filter in frequency bins (1 = no filtering)
    #     ),
    #     'bass': dict(
    #         fmin=30,
    #         fmax=500,
    #         center=False,
    #         max_size=167,
    #     ),
    #     'drums': dict(
    #         fmin=3500,
    #         fmax=11000,
    #         center=False,
    #         max_size=1,
    #     ),
    #     'mix': dict(
    #         fmin=10,
    #         fmax=16000
    #     )
    # }
    # # These are passed whenever onset_detect is called for this particular instrument's audio
    # onset_detection_params = {
    #     'piano': dict(
    #         backtrack=False,  # Whether to roll back detected onset to nearest preceding minima, i.e. start of a note
    #         wait=3,  # How many samples must pass from one detected onset to the next
    #         delta=0.06,  # Hard threshold a sample must exceed to be picked as an onset
    #         pre_max=4,  # Number of samples to consider before a sample when computing max of a window
    #         post_max=4,  # Number of samples to consider after a sample when computing max of a window
    #         pre_avg=10,  # Number of samples to consider before a sample when computing average of a window
    #         post_avg=10  # Number of samples to consider after a sample when computing average of a window
    #     ),
    #     'bass': dict(
    #         backtrack=True,
    #         rms=True,
    #         wait=4,
    #         delta=0.04,
    #         pre_max=6,
    #         post_max=6,
    #         pre_avg=15,
    #         post_avg=15
    #     ),
    #     'drums': dict(
    #         backtrack=False,
    #         wait=4,
    #         delta=0.09,
    #         pre_max=6,
    #         post_max=6,
    #         pre_avg=19,
    #         post_avg=19
    #     ),
    #     'mix': dict(
    #         win_length=960
    #     )
    # }
    onset_strength_params = {
        'piano': {
            'fmin': 110,
            'fmax': 4100,
            'center': True,
            'max_size': 1
        },
        'bass': {
            'fmin': 30,
            'fmax': 500,
            'center': True,
            'max_size': 47
        },
        'drums': {
            'fmin': 3500,
            'fmax': 11000,
            'center': True,
            'max_size': 1
        },
        'mix': {
            'fmin': 10,
            'fmax': 16000
        }
    }
    onset_detection_params = {
        'piano': {
            'backtrack': False,
            'delta': 0.06,
            'post_avg': 6,
            'post_max': 5,
            'pre_avg': 1,
            'pre_max': 4,
            'wait': 4
        },
        'bass': {
            'backtrack': False,
            'delta': 0.07,
            'post_avg': 10,
            'post_max': 17,
            'pre_avg': 6,
            'pre_max': 5,
            'wait': 5
        },
        'drums': {
            'backtrack': False,
            'delta': 0.08,
            'post_avg': 13,
            'post_max': 6,
            'pre_avg': 1,
            'pre_max': 4,
            'wait': 5
        },
        'mix': {
            'win_length': 960
        }
    }
    # These are passed whenever polyphonic_onset_detect is called for this particular instrument's audio
    polyphonic_onset_detect_params = {
        'bass': dict(
            minimum_frequency=30,   # Same as passed to onset_strength
            maximum_frequency=500,    # Same as passed to onset_strength
            onset_threshold=0.6,
            frame_threshold=0.4,
            minimum_note_length=90,
            melodia_trick=True,
            multiple_pitch_bends=True
        ),
        'piano': dict(
            minimum_frequency=110,
            maximum_frequency=4100,
            onset_threshold=0.8,
            frame_threshold=0.3,
            minimum_note_length=60,
            melodia_trick=True,
            multiple_pitch_bends=False
        ),
        'drums': dict(
            minimum_frequency=3500,
            maximum_frequency=11000,
        )
    }
    data_dir = r'..\..\data'

    def __init__(
            self,
            item: dict,
            **kwargs
    ):
        self.item = item
        # Load our audio file in when we initialise the item: we won't be changing this much
        self.audio = self._load_audio(**kwargs)
        # Dictionary to hold arrays of onset envelopes for each instrument
        self.env = {}
        # Dictionary to hold arrays of detected onsets for each instrument
        self.ons = {}
        # Empty attribute to hold our tempo
        self.tempo = None

    def _load_audio(
            self,
            **kwargs
    ) -> dict:
        """
        Wrapper around librosa.load_audio, called when class instance is constructed in order to generate audio for
        all instruments in required format. Keyword arguments are passed on to .load_audio
        """

        # These arguments are passed in whenever this class is constructed, i.e. to __init__
        duration = kwargs.get('duration', None)
        offset = kwargs.get('offset', 0)
        res_type = kwargs.get('res_type', 'soxr_vhq')
        mono = kwargs.get('mono', False)
        dtype = kwargs.get('dtype', np.float64)
        # Empty dictionary to hold audio
        audio = {}
        # Iterate through all the tracks and paths in the output key of our JSON item
        for track, fpath in self.item['output'].items():
            # Catch any UserWarnings that might be raised, usually to do with different algorithms being used to load
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                y, sr = librosa.load(
                    path=fpath,
                    sr=self.sample_rate,
                    mono=mono,
                    offset=offset,
                    duration=duration,
                    dtype=dtype,
                    res_type=res_type,
                )
            audio[track] = y.T
        return audio

    @staticmethod
    def _iqr_filter(
            arr: np.ndarray,
            low: int = 25,
            high: int = 75,
            mult: float = 1.5
    ) -> np.ndarray:
        """
        Simple IQR-based range filter that subsets array b by q1(b) - 1.5 * iqr(b) < b[n] < q3(b) + 1.5 * iqr(b)
        """

        # Get our upper and lower bound from the array
        min_ = np.nanpercentile(arr, low)
        max_ = np.nanpercentile(arr, high)
        # Construct the IQR
        iqr = max_ - min_
        # Filter the array between our two bounds and return the result
        return np.array([b for b in arr if (mult * iqr) - min_ < b < (mult * iqr) + max_])

    def beat_track_full_mix(
            self,
            passes: int = 2,
            tempo_min: int = 100,
            tempo_max: int = 300,
            use_uniform: bool = False,
            **kwargs
    ) -> np.ndarray:
        """
        Wrapper function around librosa.beat.plp that allows for per-instrument defaults and multiple passes. This
        allows for the output of predominant local pulse estimation to be passed 'back into' the algorithm several
        times, with the goal of
        """

        def plp(
                tempo_min_: int = 100,
                tempo_max_: int = 300,
                prior_: stats.rv_continuous = None,
        ) -> np.ndarray:
            """
            Wrapper function around librosa.beat.plp that takes in arguments from a current pass and converts the
            output to timestamps automatically.
            """

            # Obtain our predominant local pulse estimation envelope
            pulse = librosa.beat.plp(
                y=self.audio['mix'].mean(axis=1),  # Load in the full mix, transposed as necessary
                sr=self.sample_rate,
                hop_length=self.hop_length,
                win_length=self.onset_detection_params['mix']['win_length'],
                tempo_min=tempo_min_,
                tempo_max=tempo_max_,
                prior=prior_
            )
            # Extract the local maxima from our envelope, flatten the array, and convert from frames to timestamps
            return librosa.frames_to_time(
                np.flatnonzero(
                    librosa.util.localmax(pulse)
                ), sr=made.sample_rate
            )

        # Update our default parameters with any kwargs we've passed in
        self.onset_detection_params['mix'].update(**kwargs)
        # Get our first pass: this always uses a uniform distribution over our starting minimum and maximum tempo
        pass_ = plp(
            prior_=stats.uniform(tempo_min, tempo_max),
            tempo_min_=tempo_min,
            tempo_max_=tempo_max
        )
        # Start creating our passes
        for i in range(1, passes):
            # This extracts the BPM value for each IOI obtained from our most recent pass
            bpms = np.array([60 / p for p in np.diff(pass_)])
            # This cleans any outliers from our BPMs by removing values +/- 1.5 * IQR
            clean = self._iqr_filter(bpms)
            # Now we extract our features from our cleaned BPM array
            min_ = np.nanmin(clean)
            max_ = np.nanmax(clean)
            # Use a uniform distribution over cleaned minimum and maximum
            if use_uniform:
                prior = stats.uniform(min_, max_)
            # Use a truncated normal distribution over cleaned minimum, maximum, mean, and standard deviation
            else:
                mean_ = np.nanmean(clean)
                std_ = np.nanstd(clean)
                prior = stats.truncnorm((min_ - mean_) / std_, (max_ - mean_) / std_, loc=mean_, scale=std_)
            # Construct the next pass using our extracted features and prior distribution, and repeat
            pass_ = plp(
                tempo_min_=min_,
                tempo_max_=max_,
                prior_=prior,
            )
        # Once we've completed all of our passes, set our tempo attribute to the mean bpm from the most recent pass
        self.tempo = np.nanmean(np.array([60 / p for p in np.diff(pass_)]))
        return pass_

    def onset_strength(
            self,
            instr: str,
            aud: np.ndarray = None,
            use_nonoptimised_defaults: bool = False,
            **kwargs
    ) -> np.ndarray:
        """
        Wrapper around librosa.onset.onset_strength that allows for the use of per-instrument defaults. The required
        instrument (instr) must be passed as a string when calling this function. Any other keyword arguments should
        be accepted by librosa.onset.onset_strength, and can be passed to override per-instrument defaults.
        """

        # If we haven't passed any audio in, then construct this using the instrument name that we've passed
        if aud is None:
            aud = self.audio[instr].mean(axis=1)
        # If we're using defaults, set kwargs to an empty dictionary
        kws = self.onset_strength_params[instr] if not use_nonoptimised_defaults else dict()
        # Update our default parameters with any kwargs we've passed in
        kws.update(**kwargs)
        # TODO: check that the correct kwargs are being passed through
        # Suppress any user warnings that Librosa might throw
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            # Return the onset strength envelope using our default (i.e. hard-coded) sample rate and hop length
            return librosa.onset.onset_strength(
                y=aud,
                sr=self.sample_rate,
                hop_length=self.hop_length,
                **kws
            )

    @staticmethod
    def _try_get_kwarg_and_remove(
            kwarg: str,
            kwargs: dict,
            default_=False
    ):
        """
        Simple wrapper function for kwargs.get() that will remove the given kwarg from kwargs after getting. Useful for
        other wrapper functions that take **kwargs as inputs that can be passed onto their parent function.
        """

        # Try and get the keyword argument from our dictionary of keyword arguments, with a default
        got = kwargs.get(kwarg, default_)
        # Attempt to delete the keyword argument from our dictionary of keyword arguments
        try:
            del kwargs[kwarg]
        except KeyError:
            pass
        # Return the keyword argument
        return got

    def onset_detect(
            self,
            instr: str,
            aud: np.ndarray = None,
            env: np.ndarray = None,
            units: str = 'time',
            use_nonoptimised_defaults: bool = False,
            **kwargs
    ) -> np.ndarray:
        """
        Wrapper around librosa.onset.onset_detect that enables per-instrument defaults to be used. Arguments passed as
        kwargs should be accepted by librosa.onset.onset_detect, with the exception of rms: set this to True to use a
        a custom energy function when backtracking detected onsets to local minima. Other keyword arguments overwrite
        current per-instrument defaults.
        """

        # If we haven't passed any input audio, get this now
        if aud is None:
            aud = self.audio[instr].mean(axis=1)
        # If we haven't passed an input onset envelope, get this now
        if env is None:
            env = self.env[instr]

        # Update the default parameters for the input instrument with any kwargs we've passed in
        self.onset_detection_params[instr].update(**kwargs)
        # The RMS argument can't be passed to .onset_detect(). We need to try and get it, then remove it from our dict
        rms = self._try_get_kwarg_and_remove(
            kwarg='rms',
            kwargs=self.onset_detection_params[instr],
            default_=False
        )
        # If we're using defaults, set kwargs to an empty dictionary
        kws = self.onset_detection_params[instr] if not use_nonoptimised_defaults else dict()
        # Update our default parameters with any kwargs we've passed in
        kws.update(**kwargs)
        # TODO: check that the correct kwargs are being passed through
        # TODO: does this code make sense?
        # If we're backtracking onsets from the picked peak
        if self.onset_detection_params[instr]['backtrack']:
            # If we want to use RMS values instead of our onset envelope when back tracking onsets
            if rms:
                energy = librosa.feature.rms(S=np.abs(librosa.stft(self.audio[instr].mean(axis=1))))[0]
            else:
                energy = env
            return librosa.onset.onset_detect(
                y=aud,
                sr=self.sample_rate,
                hop_length=self.hop_length,
                units=units,
                energy=energy,
                onset_envelope=env,
                **kws
            )
        # If we're not backtracking, and using the picked peaks themselves
        else:
            return librosa.onset.onset_detect(
                y=aud,
                sr=self.sample_rate,
                hop_length=self.hop_length,
                units=units,
                onset_envelope=env,
                **kws
            )

    def _clean_polyphonic_midi_output(
            self,
            midi: np.ndarray,
            window: float = None,
    ) -> np.ndarray:
        """
        Cleans the PrettyMidi output from basic_pitch.inference.predict by removing onsets where the distance between
        the preceding onset is below a threshold (usually multiple pitches in one chord).
        """

        # Use the default window unless we've passed one in
        if window is None:
            window = self.window

        def cleaner(
                ons1: float,
                ons2: float
        ) -> float:
            """
            Compares two onsets, ons1 and ons2 (where ons2 >= ons1), and sets ons2 equal to ons1 if the distance
            is below the given window.
            """

            if ons2 - ons1 < window:
                # Little hack: we set the two onsets equal to each other here, then remove non-unique elements later
                ons2 = ons1
            return ons2

        # Define our NumPy function from our Python function
        cleaner_np = np.frompyfunc(cleaner, 2, 1)
        # Remove invalid MIDI notes from our PrettyMidi output, get the onsets, and then sort
        midi.remove_invalid_notes()
        ons = np.sort(midi.get_onsets())
        # Run the cleaner on our onsets and extract only the unique values (removes duplicates we created on purpose)
        return np.unique(
            cleaner_np.accumulate(
                ons,
                dtype=object,
                out=ons
            ).astype(float)
        )

    def polyphonic_onset_detect(
            self,
            instr: str,
            use_nonoptimised_defaults: bool = False,
            **kwargs
    ) -> np.ndarray:
        """
        Wrapper around basic_pitch.inference.predict that enables per-instrument defaults to be used. Arguments passed
        as kwargs should be accepted by this function. Output is a numpy array consisting of predicted onset locations,
        after cleaning directly-adjacent onsets (i.e. multiple notes in the same chord).
        """

        # Update the default parameters for the input instrument with any kwargs we've passed in
        self.polyphonic_onset_detect_params[instr].update(**kwargs)
        # If we're using defaults, set kwargs to an empty dictionary
        kws = self.polyphonic_onset_detect_params[instr] if not use_nonoptimised_defaults else dict()
        # We need to hide printing in basic_pitch and catch any Librosa warnings to keep stdout looking nice
        with HidePrints() as _, warnings.catch_warnings() as __:
            warnings.simplefilter('ignore', UserWarning)
            # Run the polyphonic automatic transcription model over the given audio.
            model_output, midi_data, note_events = bp.predict(
                # basic_pitch doesn't accept objects that have already been loaded in Librosa, so pass in the filename
                self.item['output'][instr],
                BASIC_PITCH_MODEL,
                **kws
            )
        # Clean the MIDI output and return
        return self._clean_polyphonic_midi_output(midi_data)

    def _bandpass_filter(
            self,
            y: np.ndarray,
            lowcut: int,
            highcut: int,
            order: int = 2
    ) -> np.ndarray:
        """
        Applies a bandpass filter with given low and high cut frequencies to an audio signal. Order is set to 2 by
        default as this seems to avoid some weird issues with the audio not rendering properly.
        """

        # Create the filter with the given values
        # Weird bug in PyCharm with signal.butter return here, so we disable checking for this statement
        # noinspection PyTupleAssignmentBalance
        b, a = signal.butter(
            order,
            [lowcut, highcut],
            fs=self.sample_rate,
            btype='band',
            output='ba'
        )
        # Apply the filter to the audio signal
        return signal.lfilter(b, a, y)

    def generate_click_track(
            self,
            instr: str,
            onsets: list = None,
            ext: str = 'wav',
            **kwargs
    ) -> None:
        """
        Outputs a track containing the filtered audio and detected onsets rendered as audio clicks. The onsets list
        should contain a list of arrays: these will be iterated through and converted to audio clicks with increasing
        output frequencies, enabling different onset detection algorithms to be compared. To only test the output
        audio used when detecting onsets, pass onsets as an empty list.
        """

        # Create a default list of onsets if we haven't passed one in ourselves
        if onsets is None:
            onsets = [self.ons[instr]]
        # Create an empty list to store our click track audio arrays
        clicks = []
        # Get the frequencies for each of our clicks
        start_freq = kwargs.get('start_freq', 750)  # The lowest click frequency
        width = kwargs.get('width', 100)  # The width of the click frequency: other frequencies attenuated
        # Iterate through all of our passed onsets, with a counter (used to increase click output frequency)
        for num, times in enumerate(onsets, 1):
            # Render the onsets to clicks, apply the bandpass filter, and append to our list
            clicks.append(
                self._bandpass_filter(
                    y=librosa.clicks(
                        times=times,
                        sr=self.sample_rate,
                        hop_length=self.hop_length,
                        length=len(self.audio[instr].mean(axis=1)),
                        click_freq=(start_freq * num)
                    ),
                    lowcut=(start_freq * num) - width,
                    highcut=(start_freq * num) + width
                )
            )
        # Filter the audio signal to only include the frequencies used in detecting onsets
        audio = self._bandpass_filter(
            y=self.audio[instr].mean(axis=1),
            lowcut=self.onset_strength_params[instr]['fmin'],
            highcut=self.onset_strength_params[instr]['fmax'],
        )
        # Sum the audio and click signals together
        process = audio + sum(clicks)
        # Create the audio file and save into the click tracks directory
        with open(rf'{self.data_dir}\processed\click_tracks\{self.item["fname"]}_{instr}_clicks.{ext}', 'wb') as f:
            sf.write(f, process, self.sample_rate)

    def compare_onset_detection_accuracy(
            self,
            fname: str,
            instr: str = None,
            onsets: list = None,
            onsets_name: list = None,
            window: float = None,
            **kwargs
    ) -> dict:
        """
        Evaluates a given list of onset detection algorithm results, given as onsets, against an array of reference
        onsets. fname should be a filepath to a text file containing the detected onsets, as a single column (i.e. one
        onset per row, the default in Sonic Visualiser). window is the length of time wherein an onset is matched as
        'correct' against the reference.

        Returns a dataframe containing F-score, precision, recall, and mean asynchrony values, defined as:

            - Precision: number of true positive matches / (number of true positives + number of false positives)
            - Recall: number of true positive matches / (number of true positives + number of false negatives)
            - F-score:  (2 * precision * recall) / (precision + recall)
            - Asynchrony: the average time between matched onsets in reference and estimate

        Additional **kwargs will be passed in as key-value pairs to the dictionary yielded by the function
        """

        # Generate the array of reference onsets from our passed file
        ref = np.genfromtxt(fname)[:, 0]
        # If we haven't provided any names for our different onset lists, create these now
        if onsets_name is None:
            onsets_name = [None for _ in range(len(onsets))]
        # If we haven't provided a window value, use our default (50 ms)
        if window is None:
            window = self.window
        # Iterate through all the onset detection algorithms passed in
        for (name, estimate) in zip(onsets_name, onsets):
            with warnings.catch_warnings():
                # If we have no onsets, both mir_eval and numpy will throw warnings, so catch them here
                warnings.simplefilter('ignore', RuntimeWarning)
                warnings.simplefilter('ignore', UserWarning)
                # Calculate the mean asynchrony between the reference and estimate onsets
                mean_async = np.nanmean([estimate[e] - ref[r] for r, e in match_events(ref, estimate, window)])
                # Generate the F, precision, and recall values from mir_eval and yield as a dictionary
                f, p, r = f_measure(
                    ref,
                    estimate,
                    window=window
                )

            yield {
                'name': name,
                'instr': instr,
                'f_score': f,
                'precision': p,
                'recall': r,
                'mean_asynchrony': mean_async,
                **kwargs
            }


if __name__ == '__main__':
    has_manual_annotations = [
        'T.T.T. (Twelve Tone Tune)',
        'T.T.T.T. (Twelve Tone Tune Two)',
        'Autumn Leaves',
        'Israel',
        'Our Love Is Here To Stay'
    ]
    res = []
    with open(r'..\..\data\processed\processing_results.json', "r+") as in_file:
        corpus = json.load(in_file)
        # Iterate through each entry in the corpus, with the index as well
        for corpus_item in corpus:
            if corpus_item['track_name'] in has_manual_annotations:
                for ins in ['drums', 'piano', 'bass']:
                    made = OnsetDetectionMaker(item=corpus_item)
                    made.env[ins] = made.onset_strength(ins, use_nonoptimised_defaults=False)
                    made.ons[ins] = made.onset_detect(ins, use_nonoptimised_defaults=False)
                    made.ons[f'{ins}_poly'] = made.polyphonic_onset_detect(instr=ins)
                    # made.generate_click_track(ins, onsets=[], start_freq=1000)
                    df = pd.DataFrame(made.compare_onset_detection_accuracy(
                        fname=rf'..\..\references\manual_annotation\{corpus_item["fname"]}_{ins}.txt',
                        onsets=[made.ons[ins], made.ons[f'{ins}_poly']],
                        onsets_name=['optimised_librosa', 'optimised_basic-pitch'],
                        instr=ins
                    ))
                    df['track'] = corpus_item['track_name']
                    res.append(df)

                made = OnsetDetectionMaker(item=corpus_item)
                bpm = made.beat_track_full_mix(passes=3, use_uniform=False)
                df = pd.DataFrame(made.compare_onset_detection_accuracy(
                    fname=rf'..\..\references\manual_annotation\{made.item["fname"]}_mix.txt',
                    onsets=[bpm],
                    onsets_name=['mix'],
                ))
                # made.generate_click_track('mix', onsets=[bpm], start_freq=1000)
                df['track'] = corpus_item['track_name']
                res.append(df)
    big = pd.concat(res).reset_index(drop=True)
