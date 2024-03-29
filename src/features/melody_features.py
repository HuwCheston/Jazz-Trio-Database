#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility classes, functions, and variables used in extracting melodic features"""

import math
from collections import Counter

import numpy as np

from src import utils
from src.detect.midi_utils import Note, MelodyMaker
from src.features.features_utils import BaseExtractor

__all__ = [
    'MelodyChunkManager', 'PitchExtractor', 'IntervalExtractor', 'ContourExtractor', 'TonalityExtractor'
]

HURON_CONTOURS = dict(
    # contour-name: function, taking in lowest note, mean inbetween notes, and highest note
    convex=lambda p1, p2, pn: p1 < p2 > pn,
    ascending_horizontal=lambda p1, p2, pn: p1 < p2 == pn,
    ascending=lambda p1, p2, pn: p1 < p2 < pn,
    horizontal=lambda p1, p2, pn: p1 == p2 == pn,
    horizontal_descending=lambda p1, p2, pn: p1 == p2 > pn,
    horizontal_ascending=lambda p1, p2, pn: p1 == p2 < pn,
    descending_horizontal=lambda p1, p2, pn: p1 > p2 == pn,
    descending=lambda p1, p2, pn: p1 > p2 > pn,
    concave=lambda p1, p2, pn: p1 > p2 < pn
)


class MelodyChunkManager(BaseExtractor):
    """For a given `MelodyMaker` instance, applies the given `extractor` to all chunks and averages the results"""
    # A chunk requires at least two `Note` instances in order to be evaluated
    NOTE_LB = 2

    def __init__(self, extractor, mm: MelodyMaker, **kwargs):
        super().__init__()
        mel = list(mm.extract_melody())
        self.chunk_list = [
            extractor(chunk, **kwargs).summary_dict for chunk in mm.chunk_melody(mel) if len(chunk) > self.NOTE_LB
        ]
        if len(self.chunk_list) > 0:
            chunk_dict = {k: [dic[k] for dic in self.chunk_list] for k in self.chunk_list[0]}
            self.update_summary_dict(chunk_dict.keys(), chunk_dict.values())

    def update_summary_dict(self, array_names, arrays, *args, **kwargs):
        """Applies all the functions in `summary_funcs` to each array of values from the base `extractor`"""
        for func_name, func in self.summary_funcs.items():
            for array_name, array in zip(array_names, arrays):
                try:
                    self.summary_dict[f'{array_name}_{func_name}'] = func(array)
                # TODO: integrate some method for getting features from categorical variables e.g. Huron contour
                except TypeError:
                    self.summary_dict[f'{array_name}_{func_name}'] = np.nan


class PitchExtractor(BaseExtractor):
    def __init__(self, my_notes: list[Note]):
        super().__init__()
        pitches = [i.pitch for i in my_notes]
        pitch_classes = [i.pitch_class for i in my_notes]
        self.summary_dict = {
            'pitch_range': max(pitches) - min(pitches) if len(pitches) > 1 else np.nan,
            'pitch_std': np.nanstd(pitches),
            'pitch_class_entropy': normalized_entropy(pitch_classes, 24)
        }


class IntervalExtractor(BaseExtractor):
    def __init__(self, my_notes: list[Note]):
        super().__init__()
        intervals = [i2.pitch - i1.pitch for i2, i1 in zip(my_notes, my_notes[1:])]
        abs_intervals = [abs(i) for i in intervals]
        self.summary_dict = dict(
            abs_interval_range=max(abs_intervals) - min(abs_intervals) if len(abs_intervals) > 1 else np.nan,
            mean_abs_interval=np.nanmean(abs_intervals),
            std_abs_interval=np.nanstd(abs_intervals),
            # Per Mullensiefen (2009), we use a different value for normalizing interval entropy here
            interval_entropy=normalized_entropy(intervals, norm=23),
            modal_interval=self.modal_interval(intervals)
        )

    @staticmethod
    def modal_interval(intervals) -> int:
        # Get the frequency of each interval
        abs_freqs = Counter(intervals)
        # Get the value of the most frequent intervals
        most_frequent = max(abs_freqs.values())
        # Get the numbers of the intervals that appeared most frequently
        top_freqs = [k for k, v in abs_freqs.items() if v == most_frequent]
        # Return the maximum interval: see Mullensiefen (2009)
        return max(top_freqs, key=lambda i: abs(i))


class ContourExtractor(BaseExtractor):
    def __init__(self, my_notes: list[Note]):
        super().__init__()
        pitches = [i.pitch for i in my_notes]
        self.summary_dict = dict(
            huron_contour=self.huron_contour(pitches),
        )

    @staticmethod
    def huron_contour(pitches: list[int]) -> str:
        firstpitch, meanpitch, lastpitch = pitches[0], np.mean(pitches[1:-2]), pitches[-1]
        for name, func in HURON_CONTOURS.items():
            # As soon as we hit a function that returns True, break out of the loop to save time
            if func(firstpitch, meanpitch, lastpitch):
                return name


class TonalityExtractor(BaseExtractor):
    # Krumhansl-Schmuckler weights for major and minor keys
    MAJ_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    MIN_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

    def __init__(self, my_notes: list[Note]):
        super().__init__()
        notes = [i.note for i in my_notes]
        self.corrs = self.krumhansl_schmuckler(notes)
        # Remove any duplicate correlations, sort in ascending order, and get the largest positive correlation
        corrs_set = sorted(set(self.corrs.values()))
        largest_corr = corrs_set[-1]
        self.summary_dict = dict(
            # Simply the largest correlation
            tonalness=largest_corr,
            # The largest correlation divided by the second-largest correlation
            tonal_clarity=largest_corr / corrs_set[-2],
            # The largest correlation divided by the sum of all positive correlations
            tonal_spike=largest_corr / sum(i for i in corrs_set if i > 0),
            mode=[c for c in self.corrs.keys() if self.corrs[c] == largest_corr][0].split(' ')[-1]
        )

    def krumhansl_schmuckler(self, notes: list[str]) -> dict:
        """
        Implements the Krumhansl-Schmuckler key-finding algorithm.

        Returns a dictionary of tonalities and corresponding correlation coefficients for a given list of `notes`.
        """
        len_pitches = len(utils.ALL_PITCHES)
        # Calculate the relative frequency of each note in the input
        relfreqs = {k: v / len(notes) for k, v in Counter(notes).items()}
        # Sort the relative frequencies by the standard C-B pitch class
        keyfreqs = {k: relfreqs[k] if k in relfreqs.keys() else 0 for k in utils.ALL_PITCHES}
        corrs = {}
        # Iterate through each note
        for i in range(len_pitches):
            key_test = [keyfreqs.get(utils.ALL_PITCHES[(i + m) % len_pitches]) for m in range(len_pitches)]
            # Calculate the correlation coefficient against the major and minor profiles for this note
            corrs[utils.ALL_PITCHES[i] + ' major'] = np.corrcoef(self.MAJ_PROFILE, key_test)[1, 0]
            corrs[utils.ALL_PITCHES[i] + ' minor'] = np.corrcoef(self.MIN_PROFILE, key_test)[1, 0]
        return corrs


def normalized_entropy(
        array: list[int],
        norm: int
) -> float:
    """Calculates the entropy of a given `array` and normalizes by `log2(norm)`, per Mulliensiefen (2009)"""
    # Calculate the relative frequency (fi) of each pitch class in the phrase
    #  i.e., the number of occurrences of the given class, divided by the number of occurrences of all classes
    rel_freq = [v / len(array) for _, v in Counter(array).items()]
    # Calculate the pitch class entropy
    #  Equivalent to the sum of fi * log2(fi) for fi in rel_freq,
    #  normalised by the maximum entropy given an upper phrase length limit (defaults to 24 as in Mullensiefen (2009))
    return -(sum(v * math.log(v, 2) for v in rel_freq) / math.log(norm, 2))


if __name__ == '__main__':
    # Extract our melody from a random sample file
    fp = f'{utils.get_project_root()}\data\cambridge-jazz-trio-database-v01\corpus_chronology\evansb-ttttwelvetonetune-gomezemorellm-1971-360d7a67'
    track = utils.load_track_from_files(fp)
    maker = MelodyMaker(fp + '\piano_midi.mid', track)
    # Create our MelodyChunkManager for each feature we want to extract and convert to a single dictionary
    mel_list = [MelodyChunkManager(feature, maker) for feature in __all__]
    mel_features = {k: v for d in mel_list for k, v in d.summary_dict.items()}
    print(mel_features)
