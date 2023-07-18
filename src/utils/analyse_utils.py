import json
import logging
import os
import pickle
import sys
import warnings
from pathlib import Path
from typing import Any

import audioread
import dill
import numpy as np
import pandas as pd
import tensorflow as tf
from basic_pitch import ICASSP_2022_MODEL_PATH
from tqdm import tqdm

# Set options in pandas and numpy here so they carry through whenever this file is imported by another
# This disables scientific notation and forces all rows/columns to be printed: helps with debugging!
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
np.set_printoptions(suppress=True)

# Define constants used across many files
SAMPLE_RATE = 44100    # TODO: test higher sample rates
HOP_LENGTH = 128
FILE_FMT = 'wav'
BASIC_PITCH_MODEL = tf.saved_model.load(str(ICASSP_2022_MODEL_PATH))
N_PLP_PASSES = 3    # This seems to lead to the best results after optimization

# Mapping to turn instrument name into instrument performer, e.g. piano to pianist
INSTRS_TO_PERF = {
    'piano': 'pianist',
    'bass': 'bassist',
    'drums': 'drummer'
}


def get_project_root() -> Path:
    """Returns the root directory of the project"""
    return Path(__file__).absolute().parent.parent.parent


class HidePrints:
    """Helper class that prevents a function from printing to stdout when used as a context manager"""

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
    """Fake logging class passed to yt-dlp instances to disable overly-verbose logging and unnecessary warnings"""

    def debug(self, msg=None):
        pass

    def warning(self, msg=None):
        pass

    def error(self, msg=None):
        pass


def serialise_object(
        obj: object,
        fpath: str,
        fname: str,
        use_pickle: bool = False
) -> None:
    """Wrapper around dill.dump that takes in an object, directory, and filename, and creates a serialised object"""
    if use_pickle:
        dumper = pickle.dump
    else:
        dumper = dill.dump
    with open(rf'{fpath}\{fname}.p', 'wb') as fi:
        dumper(obj, fi)


def unserialise_object(
        fpath: str,
        fname: str,
        use_pickle: bool = False
) -> object:
    """Simple wrapper around dill.load that unserialises an object and returns it"""
    if use_pickle:
        loader = pickle.load
    else:
        loader = dill.load
    return loader(open(fr'{fpath}\{fname}.p', 'rb'))


def load_json(
        fpath: str = 'r..\..\data\processed',
        fname: str = 'processing_results.json'
) -> dict:
    """Simple wrapper around json.load"""
    with open(rf'{fpath}\{fname}.json', "r+") as in_file:
        return json.load(in_file)


def save_json(
        obj: dict,
        fpath: str,
        fname: str
) -> None:
    """Simple wrapper around json.dump"""
    with open(rf'{fpath}\{fname}.json', "w") as out_file:
        json.dump(obj, out_file, indent=4, default=str, )


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
        arr: np.array,
        low: int = 25,
        high: int = 75,
        mult: float = 1.5,
        fill_nans: bool = False,
) -> np.ndarray:
    """Simple IQR-based range filter that subsets array b where q1(b) - 1.5 * iqr(b) < b[n] < q3(b) + 1.5 * iqr(b)

    Parameters:
        arr (np.array): the array of values to clean
        low (int, optional): the lower quantile to use, defaults to 25
        high (int, optional): the upper quantile to use, defaults to 75
        mult (float, optional): the amount to multiply the IQR by, defaults to 1.5
        fill_nans (bool, optional): replace cleaned values with np.nan, such that the array shape remains the same

    Returns:
        np.array

    """
    # Get our upper and lower bound from the array
    min_ = np.nanpercentile(arr, low)
    max_ = np.nanpercentile(arr, high)
    # Construct the IQR
    iqr = max_ - min_
    # Filter the array between our two bounds and return the result
    if fill_nans:
        return np.array(
            [b if min_ - (mult * iqr) < b < max_ + (mult * iqr) else np.nan for b in arr]
        )
    else:
        return np.array(
            [b for b in arr if min_ - (mult * iqr) < b < max_ + (mult * iqr)]
        )


def get_tracks_with_manual_annotations(
        annotation_dir: str = fr'{get_project_root()}\references\manual_annotation',
        annotation_ext: str = 'txt',
        all_tracks: tuple = ('bass', 'drums', 'piano', 'mix')
) -> list:
    """Returns the filenames of tracks that contain a full set of manual annotation files"""
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
    """Returns whether a given filepath is present locally or not"""
    return os.path.isfile(os.path.abspath(fname))


def calculate_tempo(
        pass_: np.ndarray
) -> float:
    """Extract the average tempo from an array of times corresponding to crotchet beat positions"""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(np.array([60 / p for p in np.diff(pass_)]))


def try_get_kwarg_and_remove(
        kwarg: str,
        kwargs: dict,
        default_: bool = False
) -> Any:
    """Try and get an argument from a kwargs dictionary, remove after getting, and return the value (or a default).

    Arguments:
        kwarg (str): the argument to attempt to get from the kwargs dictionary
        kwargs (dict): the dictionary of keyword arguments
        default_ (bool, optional): the value to return if kwarg is not found in kwargs, defaults to None

    Returns:
        Any: the value returned from kwargs, or a default

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


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


def disable_settingwithcopy_warning(func):
    """Simple decorator that disables the annoying SettingWithCopy warning in Pandas"""
    def wrapper(*args, **kwargs):
        pd.options.mode.chained_assignment = None
        res = func(*args, **kwargs)
        pd.options.mode.chained_assignment = None
        return res
    return wrapper


def get_audio_duration(fpath: str) -> float:
    """Opens a given audio file and returns its duration"""

    try:
        with audioread.audio_open(fpath) as f:
            return float(f.duration)
    except FileNotFoundError:
        return 0.0