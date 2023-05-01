import json
import warnings

import librosa
import numpy as np
import soundfile as sf
import scipy.signal as signal
import pandas as pd
from mir_eval.onset import f_measure


class OnsetDetectionMaker:
    sample_rate = 44100
    hop_length = 512
    dtype = np.float64
    onset_strength_params = {
        'piano': dict(
            fmin=110,
            fmax=4100,
            center=False,
            max_size=1,
        ),
        'bass': dict(
            fmin=30,
            fmax=500,
            center=False,
            max_size=167,
        ),
        'drums': dict(
            fmin=3500,
            fmax=11000,
            center=False,
            max_size=1,
        )
    }
    onset_detection_params = {
        'piano': dict(
            backtrack=False,
            wait=3,
            delta=0.06,
            pre_max=4,
            post_max=4,
            pre_avg=10,
            post_avg=10
        ),
        'bass': dict(
            backtrack=True,
            rms=True,
            wait=4,
            delta=0.04,
            pre_max=6,
            post_max=6,
            pre_avg=15,
            post_avg=15
        ),
        'drums': dict(
            backtrack=False,
            wait=4,
            delta=0.09,
            pre_max=6,
            post_max=6,
            pre_avg=19,
            post_avg=19
        )
    }
    data_dir = r'..\..\data'

    def __init__(
            self,
            item: dict,
            **kwargs
    ):
        self.item = item
        self.duration = kwargs.get('duration', None)
        self.offset = kwargs.get('offset', 0)
        self.audio = self._load_audio()
        # Dictionary to hold arrays of onset envelopes for each instrument
        self.env = {}
        # Dictionary to hold arrays of detected onsets for each instrument
        self.ons = {}

    def _load_audio(
            self,
    ) -> dict:
        audio = {}
        for track, fpath in self.item['output'].items():
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                y, sr = librosa.load(
                    path=fpath,
                    sr=self.sample_rate,
                    mono=False,
                    offset=self.offset,
                    duration=self.duration,
                    dtype=self.dtype,
                    res_type='soxr_vhq',
                )
            audio[track] = y.T
        return audio

    def _beat_track_full_mix(self):
        pass

    def onset_strength(
            self,
            instr: str,
            **kwargs
    ) -> np.ndarray:
        """

        """

        self.onset_strength_params[instr].update(**kwargs)
        return librosa.onset.onset_strength(
            y=self.audio[instr].mean(axis=1),
            sr=self.sample_rate,
            hop_length=self.hop_length,
            **self.onset_strength_params[instr]
        )

    @staticmethod
    def _try_get_kwarg_and_remove(
            kwarg: str,
            kwargs: dict,
            default=False
    ):
        """
        Simple wrapper function for kwargs.get() that will remove the given kwarg from kwargs after getting. Useful for
        other wrapper functions that take **kwargs as inputs that can be passed onto their parent function.
        """

        # Try and get the keyword argument from our dictionary of keyword arguments, with a default
        got = kwargs.get(kwarg, default)
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
            units: str = 'time',
            **kwargs
    ) -> np.ndarray:
        """

        """

        # Update the default parameters for the input instrument with any kwargs we've passed in
        self.onset_detection_params[instr].update(**kwargs)
        # The RMS argument can't be passed to .onset_detect(). We need to try and get it, then remove it from our dict
        rms = self._try_get_kwarg_and_remove(
            kwarg='rms',
            kwargs=self.onset_detection_params[instr],
            default=False
        )
        # If we're backtracking onsets from the picked peak
        if self.onset_detection_params[instr]['backtrack']:
            # If we want to use RMS values instead of our onset envelope when back tracking onsets
            if rms:
                energy = librosa.feature.rms(S=np.abs(librosa.stft(self.audio[instr].mean(axis=1))))[0]
            else:
                energy = self.env[instr]
            return librosa.onset.onset_detect(
                y=self.audio[instr].mean(axis=1),
                sr=self.sample_rate,
                hop_length=self.hop_length,
                units=units,
                energy=energy,
                onset_envelope=self.env[instr],
                **self.onset_detection_params[instr]
            )
        # If we're not backtracking, and using the picked peaks themselves
        else:
            return librosa.onset.onset_detect(
                y=self.audio[instr].mean(axis=1),
                sr=self.sample_rate,
                hop_length=self.hop_length,
                units=units,
                onset_envelope=self.env[instr],
                **self.onset_detection_params[instr]
            )

    def _bandpass_filter(
            self,
            y: np.ndarray,
            lowcut: int,
            highcut: int,
            order: int = 2
    ) -> np.ndarray:
        """
        Applies a bandpass filter with given low and high cut frequencies to an audio signal
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

    def output_click_track(
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
        start_freq = kwargs.get('start_freq', 750)    # The lowest click frequency
        width = kwargs.get('width', 100)    # The width of the click frequency: other frequencies attenuated
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

    @staticmethod
    def compare_onset_detection_accuracy(
            fname: str,
            instr: str = None,
            onsets: list = None,
            onsets_name: list = None,
            window: float = 0.05
    ) -> dict:
        """
        Evaluates a given list of onset detection algorithm results, given as onsets, against an array of reference
        onsets. fname should be a filepath to a text file containing the detected onsets, as a single column (i.e. one
        onset per row, the default in Sonic Visualiser). window is the length of time wherein an onset is matched as
        'correct' against the reference.

        Returns a dataframe containing F-score, precision, and
        recall values, defined as:

            - Precision: number of true positive matches / (number of true positives + number of false positives)
            - Recall: number of true positive matches / (number of true positives + number of false negatives)
            - F-score:  (2 * precision * recall) / (precision + recall)

        Note that, for all metrics, greater values are suggestive of a stronger match between reference and estimated
        onsets, such that a score of 1.0 for any metric indicates equality.
        """

        # Generate the array of reference onsets from our passed file
        ref = np.genfromtxt(fname)[:, 0]
        # If we haven't provided any names for our different onset lists, create these now
        if onsets_name is None:
            onsets_name = [None for _ in range(len(onsets))]
        # Iterate through all the onset detection algorithms
        for (name, estimate) in zip(onsets_name, onsets):
            # Generate the F, precision, and recall values from mir_eval and yield as a dictionary
            f, p, r = f_measure(
                ref,
                estimate,
                window=window
            )
            yield {
                'name': name,
                'f_score': f,
                'precision': p,
                'recall': r,
                'instr': instr
            }


if __name__ == '__main__':
    with open(r'..\..\data\processed\processing_results.json', "r+") as in_file:
        corpus = json.load(in_file)
        # Iterate through each entry in the corpus, with the index as well
        for item in corpus:
            if item['track_name'] == 'Autumn Leaves':
                made = OnsetDetectionMaker(item=item)
                for ins in ['piano', 'bass', 'drums']:
                    made.env[ins] = made.onset_strength(ins)
                    made.ons[ins] = made.onset_detect(ins)
                    made.output_click_track(ins)
                    df = pd.DataFrame(made.compare_onset_detection_accuracy(
                                fname=rf'..\..\references\manual_annotation\{made.item["fname"]}_{ins}.txt',
                                onsets=[made.ons[ins]],
                                instr=ins
                            ))
                    print(df)
                # res = []
                # for max_ in range(1, 50):
                #     print(max_)
                #     for ins in ['piano', 'bass', 'drums']:
                #         made.env[ins] = made.onset_strength(ins)
                #         made.ons[ins] = made.onset_detect(ins, pre_avg=max_, post_avg=max_)
                #         df = pd.DataFrame(made.compare_onset_detection_accuracy(
                #             fname=rf'..\..\references\manual_annotation\{made.item["fname"]}_{ins}.txt',
                #             onsets=[made.ons[ins]],
                #             instr=ins
                #         ))
                #         df['pre_post_avg'] = max_
                #         res.append(df)
                # pass
