import json
import os
import sys

import dill
import numpy as np
import pandas as pd
import tensorflow as tf
from basic_pitch import ICASSP_2022_MODEL_PATH

# Set options in pandas here so they carry through whenever this file is imported by another
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Define constants used across many files
# TODO: test higher sample rates
SAMPLE_RATE = 88200
FILE_FMT = 'wav'
BASIC_PITCH_MODEL = tf.saved_model.load(str(ICASSP_2022_MODEL_PATH))
N_PLP_PASSES = 3    # This seems to lead to the best results after optimization

# Mapping to turn instrument name into instrument performer, e.g. piano to pianist
INSTRS_TO_PERF = {
    'piano': 'pianist',
    'bass': 'bassist',
    'drums': 'drummer'
}


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


class YtDlpFakeLogger:
    """
    Fake logging class passed to yt-dlp instances to disable overly-verbose logging and unnecessary warnings
    """

    def debug(self, msg=None):
        pass

    def warning(self, msg=None):
        pass

    def error(self, msg=None):
        pass


def serialise_object(
        obj: object,
        fpath: str,
        fname: str
) -> None:
    """
    Simple wrapper around dill.dump that takes in an object, directory, and filename, and creates a serialised object
    """

    with open(rf'{fpath}\{fname}.p', 'wb') as fi:
        dill.dump(obj, fi)


def unserialise_object(
        fpath: str,
        fname: str
) -> object:
    """
    Simple wrapper around dill.load that unserialises an object and returns
    """

    return dill.load(
        open(fr'{fpath}\{fname}.p', 'rb')
    )


def load_json(
        fpath: str = 'r..\..\data\processed',
        fname: str = 'processing_results.json'
) -> dict:
    """
    Simple wrapper around json.load
    """

    with open(rf'{fpath}\{fname}.json', "r+") as in_file:
        return json.load(in_file)


def save_json(
        obj: dict,
        fpath: str,
        fname: str
) -> None:
    """
    Simple wrapper around json.dump
    """

    with open(rf'{fpath}\{fname}.json', "w") as out_file:
        json.dump(obj, out_file)


def try_and_load(
        attempt_func,
        attempt_kwargs,
        backup_func,
        backup_kwargs
):
    """
    Attempts to load an object using attempt_func (with arguments passed as attempt_kwargs dictionary). If this fails
    due to a FileNotFoundError, then attempts to load using backup_func (with arguments passed as backup_kwargs)
    """

    try:
        return attempt_func(**attempt_kwargs)
    except FileNotFoundError:
        return backup_func(**backup_kwargs)


def iqr_filter(
        arr: np.ndarray,
        low: int = 25,
        high: int = 75,
        mult: float = 1.5
) -> np.ndarray:
    """
    Simple IQR-based range filter that subsets array b where q1(b) - 1.5 * iqr(b) < b[n] < q3(b) + 1.5 * iqr(b)
    """

    # Get our upper and lower bound from the array
    min_ = np.nanpercentile(arr, low)
    max_ = np.nanpercentile(arr, high)
    # Construct the IQR
    iqr = max_ - min_
    # Filter the array between our two bounds and return the result
    return np.array(
        [b for b in arr if (mult * iqr) - min_ < b < (mult * iqr) + max_]
    )


def get_tracks_with_manual_annotations(
        annotation_dir: str = r'..\..\references\manual_annotation',
        annotation_ext: str = 'txt',
        all_tracks: tuple = ('bass', 'drums', 'piano', 'mix')
) -> list:
    """
    Returns the filenames of tracks that contain a full set of manual annotation files
    """

    res = {}
    for file in os.listdir(annotation_dir):
        if file.endswith(annotation_ext):
            split = file.split('_')
            try:
                res[split[0]].append(split[1].replace(f'.{annotation_ext}', ''))
            except KeyError:
                res[split[0]] = []
                res[split[0]].append(split[1].replace(f'.{annotation_ext}', ''))
    return [k for k, v in res.items() if sorted(v) == sorted(list(all_tracks))]


def check_item_present_locally(fname: str) -> bool:
    """
    Returns whether a given filepath is present locally or not
    """

    return os.path.isfile(os.path.abspath(fname))
