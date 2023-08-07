import csv
import json
import inspect
import logging
import os
import pickle
import sys
import time
import warnings
from ast import literal_eval
from functools import wraps
from pathlib import Path
from string import punctuation
from tempfile import NamedTemporaryFile
from typing import Any

import audioread
import dill
import numpy as np
import pandas as pd
import tensorflow as tf
from basic_pitch import ICASSP_2022_MODEL_PATH
from tqdm import tqdm

# TODO: sort out imports here for pickling/unpickling custom classes e.g. OnsetMaker


# Set options in pandas and numpy here so they carry through whenever this file is imported by another
# This disables scientific notation and forces all rows/columns to be printed: helps with debugging!
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
np.set_printoptions(suppress=True)

# Define constants used across many files
SAMPLE_RATE = 44100
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

FREQUENCY_BANDS = {
    'piano': dict(
        fmin=110,  # Minimum frequency to use
        fmax=4100,  # Maximum frequency to use
    ),
    'bass': dict(
        fmin=30,
        fmax=500,
    ),
    'drums': dict(
        fmin=3500,
        fmax=11000,
    ),
    'mix': dict(
        fmin=20,
        fmax=20000,
    ),
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


def retry(exception, tries=4, delay=3, backoff=2):
    """Retry calling the decorated function using an exponential backoff."""
    def deco_retry(f):
        @wraps(f)
        def f_retry(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return f(*args, **kwargs)
                except exception:
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return f(*args, **kwargs)
        return f_retry
    return deco_retry


def serialise_object(
        obj: object,
        fpath: str,
        fname: str,
        use_pickle: bool = False,
        func = None
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


@retry(json.JSONDecodeError)
def load_json(
        fpath: str = 'r..\..\data\processed',
        fname: str = 'processing_results.json'
) -> dict:
    """Simple wrapper around json.load that catches errors when working on the same file in multiple threads"""
    with open(rf'{fpath}\{fname}.json', "r+") as in_file:
        return json.load(in_file)


def save_json(
        obj: dict,
        fpath: str,
        fname: str
) -> None:
    """Simple wrapper around json.dump with protections to assist in multithreaded access"""
    temp_file = NamedTemporaryFile(mode='w', dir=fpath, delete=False, suffix='.json')
    with temp_file as out_file:
        json.dump(obj, out_file, indent=4, default=str, )

    @retry(PermissionError)
    def replacer():
        os.replace(temp_file.name, rf'{fpath}\{fname}.json')

    replacer()


@retry(json.JSONDecodeError)
def load_csv(
        fpath: str = 'r..\..\data\processed',
        fname: str = 'processing_results'
) -> dict:
    """Simple wrapper around json.load that catches errors when working on the same file in multiple threads"""
    def eval_(i):
        try:
            return literal_eval(i)
        except (ValueError, SyntaxError) as _:
            return str(i)


    with open(rf'{fpath}\{fname}.csv', "r+") as in_file:
        return [{k: eval_(v) for k, v in row.items()} for row in csv.DictReader(in_file, skipinitialspace=True)]


def save_csv(
        obj: list | dict,
        fpath: str,
        fname: str
) -> None:
    """Simple wrapper around csv.DictWriter with protections to assist in multithreaded access"""
    # If we have an existing file with the same name, load it in and extend it with our new data
    try:
        existing_file = load_csv(fpath, fname)
    except FileNotFoundError:
        pass
    else:
        if isinstance(obj, dict):
            obj = [obj]
        obj = existing_file + obj

    # Create a new temporary file, in append mode
    temp_file = NamedTemporaryFile(mode='a', newline='', dir=fpath, delete=False, suffix='.csv')
    # Get our CSV header from the keys of the first dictionary, if we've passed in a list of dictionaries
    if isinstance(obj, list):
        keys = obj[0].keys()
    # Otherwise, if we've just passed in a dictionary, get the keys from it directly
    else:
        keys = obj.keys()
    # Open the temporary file and create a new dictionary writer with our given columns
    with temp_file as out_file:
        dict_writer = csv.DictWriter(out_file, keys)
        dict_writer.writeheader()
        # Write all the rows, if we've passed in a list
        if isinstance(obj, list):
            dict_writer.writerows(obj)
        # Alternatively, write a single row, if we've passed in a dictionary
        else:
            dict_writer.writerow(obj)

    @retry(PermissionError)
    def replacer():
        os.replace(temp_file.name, rf'{fpath}\{fname}.csv')

    replacer()


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
    # If the upper and lower bounds are equal, IQR will be 0.0, and our cleaned array will be empty. So don't clean.
    if min_ - max_ == 0:
        return arr
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
        corpus_json: list[dict] = None
) -> list:
    """Returns the IDs of tracks that should be annotated"""
    # return [t.strip('\n') for t in open(rf'{annotation_dir}\tracks_to_annotate.{annotation_ext}', 'r').readlines()]
    res = {}
    track_ids = '\t'.join([track['mbz_id'] for track in corpus_json])
    for file in os.listdir(annotation_dir):
        if file.endswith(annotation_ext):
            split = file.split('_')
            track_id = split[0].split('-')[-1]
            if track_id not in track_ids:
                continue
            try:
                res[split[0]].append(split[1].replace(f'.{annotation_ext}', ''))
            except KeyError:
                res[split[0]] = []
                res[split[0]].append(split[1].replace(f'.{annotation_ext}', ''))
    annotated_with_all_instrs = [k for k, v in res.items() if sorted(v) == sorted([*INSTRS_TO_PERF.keys(), 'mix'])]
    return [t['mbz_id'] for t in corpus_json if t['fname'] in annotated_with_all_instrs]


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

def remove_punctuation(s: str) -> str:
    """Removes punctuation from a string"""
    return s.translate(str.maketrans('', '', punctuation)).replace('’', '')


def return_function_kwargs(func) -> list:
    """Returns a list of keyword arguments accepted by a given function"""
    return [p for p in inspect.signature(func).parameters]
