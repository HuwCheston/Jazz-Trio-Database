#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility classes, functions, and variables used across the entire pipeline"""

import csv
import inspect
import json
import os
import pickle
import re
import subprocess
import time
import warnings
from ast import literal_eval
from datetime import datetime, timedelta
from functools import wraps
from multiprocessing import Manager, Process
from pathlib import Path
from string import punctuation, printable
from tempfile import NamedTemporaryFile
from typing import Generator, Any, Callable

import audioread
import dill
import numpy as np
import pandas as pd

# Set options in pandas and numpy here, so they carry through whenever this file is imported by another
# This disables scientific notation and forces all rows/columns to be printed: helps with debugging!
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
np.set_printoptions(suppress=True)

# Define variables used across many files
ALL_PITCHES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
SAMPLE_RATE = 44100
HOP_LENGTH = 128
AUDIO_FILE_FMT = 'wav'
INSTRUMENTS_TO_PERFORMER_ROLES = {
    'piano': 'pianist',
    'bass': 'bassist',
    'drums': 'drummer'
}
SILENCE_THRESHOLD = 1/3

# These are the underlying categories each predictor belongs to
PREDICTORS_CATEGORIES = {
    'Swing': ['bur_log_mean', 'bur_log_std'],
    'Complexity': ['lz77_mean', 'lz77_std', 'n_onsets_mean', 'n_onsets_std'],
    'Feel': ['bass_prop_async_nanmean', 'drums_prop_async_nanmean', 'bass_prop_async_nanstd', 'drums_prop_async_nanstd'],
    'Interaction': ['self_coupling', 'coupling_drums', 'coupling_bass', 'coupling_piano_drums', 'coupling_piano_bass'],
    'Tempo': ['rolling_std_median', 'tempo', 'tempo_slope',]
}
PREDICTORS = [it for sl in list(PREDICTORS_CATEGORIES.values()) for it in sl]

# Mean values for each predictor in the dataset: use these to impute NaNs when encountering new data
IMPUTE_VALS = {
    "bur_log_mean": 0.3761003068024858,
    "bur_log_std": 0.6596973212959278,
    "lz77_mean": 10.708304075130211,
    "lz77_std": 1.807330355424879,
    "n_onsets_mean": 24.61031148393926,
    "n_onsets_std": 5.269495594070607,
    "bass_prop_async_nanmean": 0.010629296173547409,
    "drums_prop_async_nanmean": 0.0147304515092208,
    "bass_prop_async_nanstd": 0.004531355940149889,
    "drums_prop_async_nanstd": 0.006640026310255122,
    "self_coupling": -0.4955328636051492,
    "coupling_drums": 0.6788343986626387,
    "coupling_bass": 0.4178927218632842,
    "coupling_piano_drums": 0.13752675794042257,
    "coupling_piano_bass": 0.13951901448268583,
    "rolling_std_median": 0.1934943576660537,
    "tempo": 197.379633304093,
    "tempo_slope": 0.026780946855082095
}


def get_project_root() -> Path:
    """Returns the root directory of the project"""
    return Path(__file__).absolute().parent.parent


def check_item_present_locally(fname: str) -> bool:
    """Returns whether a given filepath is present locally or not"""
    return os.path.isfile(os.path.abspath(fname))


def get_audio_duration(fpath: str) -> float:
    """Opens a given audio file and returns its duration"""
    try:
        with audioread.audio_open(fpath) as f:
            return float(f.duration)
    except FileNotFoundError:
        return 0.0


def disable_settingwithcopy_warning(func: Callable) -> Callable:
    """Simple decorator that disables the annoying `SettingWithCopy` warning in Pandas"""
    def wrapper(*args, **kwargs):
        pd.options.mode.chained_assignment = None
        res = func(*args, **kwargs)
        pd.options.mode.chained_assignment = None
        return res
    return wrapper


def remove_punctuation(s: str) -> str:
    """Removes punctuation from a given input string `s`"""
    return ''.join(c for c in str(s).translate(str.maketrans('', '', punctuation)).replace('’', '') if c in printable)


def retry(exception, tries=4, delay=3, backoff=2) -> Callable:
    """Retry calling the decorated function using an exponential backoff."""
    def deco_retry(f: Callable) -> Callable:
        @wraps(f)
        def f_retry(*args, **kwargs) -> Callable:
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
) -> None:
    """Wrapper around `dill.dump` that takes in an object, directory, and filename, and creates a serialised object"""
    if use_pickle:
        dumper = pickle.dump
    else:
        dumper = dill.dump
    with open(rf'{fpath}/{fname}.p', 'wb') as fi:
        dumper(obj, fi)


def unserialise_object(
        fpath: str,
        use_pickle: bool = False,
        _ext: str = 'p'
) -> list:
    """Simple wrapper that unserialises an iterable pickle object using pickle or dill and returns it"""
    if use_pickle:
        loader = pickle.load
    else:
        loader = dill.load
    data = []
    fpath = fpath if fpath.endswith(f'.{_ext}') else f'{fpath}.{_ext}'
    with open(fpath, 'rb') as fr:
        # Iteratively append from our Pickle file until we run out of data
        try:
            while True:
                data.append(loader(fr))
        except EOFError:
            pass
    return data


@retry(json.JSONDecodeError)
def load_json(
        fpath: str = 'r../../data/processed',
        fname: str = 'processing_results.json'
) -> dict:
    """Simple wrapper around `json.load` that catches errors when working on the same file in multiple threads"""
    with open(rf'{fpath}/{fname}.json', "r+") as in_file:
        return json.load(in_file)


def save_json(
        obj: dict,
        fpath: str,
        fname: str
) -> None:
    """Simple wrapper around `json.dump` with protections to assist in multithreaded access"""
    temp_file = NamedTemporaryFile(mode='w', dir=fpath, delete=False, suffix='.json')
    with temp_file as out_file:
        json.dump(obj, out_file, indent=4, default=str, )

    @retry(PermissionError)
    def replacer():
        os.replace(temp_file.name, rf'{fpath}/{fname}.json')

    replacer()


@retry(json.JSONDecodeError)
def load_csv(
        fpath: str = 'r../../data/processed',
        fname: str = 'processing_results'
) -> dict:
    """Simple wrapper around `json.load` that catches errors when working on the same file in multiple threads"""
    def eval_(i):
        try:
            return literal_eval(i)
        except (ValueError, SyntaxError) as _:
            return str(i)

    with open(rf'{fpath}/{fname}.csv', "r+") as in_file:
        return [{k: eval_(v) for k, v in row.items()} for row in csv.DictReader(in_file, skipinitialspace=True)]


def save_csv(
        obj,
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
            try:
                dict_writer.writerows(obj)
            except UnicodeEncodeError:
                for line in obj:
                    line['track_name'] = remove_punctuation(line['track_name'])
                    dict_writer.writerow(line)
        # Alternatively, write a single row, if we've passed in a dictionary
        else:
            dict_writer.writerow(obj)

    @retry(PermissionError, tries=100, delay=5)
    def replacer():
        os.replace(temp_file.name, rf'{fpath}/{fname}.csv')

    replacer()


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


def return_function_kwargs(func) -> list:
    """Returns a list of keyword arguments accepted by a given function"""
    return [p for p in inspect.signature(func).parameters]


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
        fill_nans (bool, optional): replace cleaned values with `np.nan`, such that the array shape remains the same

    Returns:
        `np.array`

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


def serialise_from_queue(item_queue, fpath: str) -> None:
    """Iteratively append items in a queue to a single file. Process dies when `NoneType` added to queue

    Args:
        item_queue: the `multiprocessing.Manager.Queue` instance to draw items from
        fpath (str): the filepath to save items to (file will be created if it does not exist)

    Returns:
        None

    """
    with open(fr'{fpath}.p', 'ab+') as out:
        # Keep getting items from our queue and appending them to our Pickle file
        while True:
            val = item_queue.get()
            # When we receive a NoneType object from the queue, break out and terminate the process
            if val is None:
                break
            pickle.dump(val, out)


def initialise_queue(target_func: Callable = serialise_from_queue, *target_func_args) -> tuple:
    """Initialise the objects we need for caching through multiprocessing

    Args:
        target_func (Callable, optional): target function for the worker process, defaults to `serialise_from_queue`
        *target_func_args: arguments passed to `target_func`

    Returns:
        tuple

    """
    m = Manager()
    q = m.Queue()
    # Initialise our worker for saving completed tracks
    p = Process(target=target_func, args=(q, *target_func_args))
    p.start()
    return p, q


def get_cached_track_ids(fpath: str, **kwargs) -> Generator:
    """Open a pickle file and get the IDs of tracks that have already been processed

    Args:
        fpath (str): filepath to load object from
        **kwargs: passed to `unserialise_object`

    Yields:
        str: the Musicbrainz ID of the processed track

    Returns:
        None: if `fpath` is not found

    """
    try:
        data = unserialise_object(fpath, **kwargs)
    # If we have not created the item yet, return None
    except FileNotFoundError:
        return
    else:
        for track in data:
            yield track.item['mbz_id']


def ignore_warning(*args, **kwargs):
    """Decorator function for suppressing warnings during a function call"""
    def inner(func):
        @wraps(func)
        def wrapper():
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                return func(*args, **kwargs)
        return wrapper
    return inner


def flatten_dict(dd: dict, separator='_', prefix=''):
    """Flattens a dictionary with dictionaries as values, with given separator and prefict"""
    return {
        prefix + separator + k if prefix else k: v
        for kk, vv in dd.items()
        for k, v in flatten_dict(vv, separator, kk).items()
    } if isinstance(dd, dict) else {prefix: dd}


class CorpusMaker:
    """Converts a multi-sheet Excel spreadsheet into the required format for processing"""
    lbz_url_cutoff = 49
    json_indent = 4
    bandleader_instr = 'piano'
    keep_all_tracks = False

    def __init__(self, data: list[dict]):
        self.tracks = list(self.format_track_dict(data))

    @classmethod
    def from_excel(
            cls,
            fname: str,
            ext: str = 'xlsx',
            **kwargs
    ):
        """Construct corpus from an Excel spreadsheet, potentially containing multiple sheets"""
        realdata = []
        # These are the names of sheets that we don't want to process
        sheets_to_skip = ['notes', 'template', 'manual annotation', 'track rating']
        # Open the Excel file
        xl = pd.read_excel(pd.ExcelFile(fr'{get_project_root()}/references/{fname}.{ext}'), None, header=1).items()
        # Iterate through all sheets in the spreadsheet
        for sheet_name, trio in xl:
            if sheet_name.lower() not in sheets_to_skip:
                realdata.extend(cls.format_trio_spreadsheet(cls, trio, **kwargs))
        return cls(realdata)

    @classmethod
    def from_json(
            cls,
            fname: str,
            ext: str = 'json'
    ):
        """Construct corpus from a JSON"""
        # TODO: fill this in
        pass

    def __repr__(self):
        """Sets the string representation of this class to a DataFrame of all processed tracks"""
        return repr(pd.DataFrame(self.tracks))

    @disable_settingwithcopy_warning
    def format_trio_spreadsheet(
            self,
            trio_df: pd.DataFrame,
            **kwargs
    ) -> list[dict]:
        """Formats the spreadsheet for an individual trio and returns a list of dictionaries"""

        # We remove these columns from the dataframe
        to_drop = ['recording_id_for_lbz', 'recording_date_estimate', 'is_acceptable(Y/N)', 'link']
        # We rename these columns
        to_rename = {
            'piano': 'pianist',
            'bass': 'bassist',
            'drums': 'drummer',
            'release_title': 'album_name',
            'recording_title': 'track_name',
            'Unnamed: 19': 'notes',
            'Unnamed: 13': 'link'
        }
        # We keep these columns, in the following order
        to_keep = [
            'track_name',
            'album_name',
            'recording_year',
            'pianist',
            'bassist',
            'drummer',
            'youtube_link',
            'channel_overrides',
            'start_timestamp',
            'end_timestamp',
            'mbz_id',
            'notes',
            'time_signature',
            'first_downbeat',
            "rating_bass_audio",
            "rating_bass_detection",
            "rating_drums_audio",
            "rating_drums_detection",
            "rating_mix",
            "rating_piano_audio",
            "rating_piano_detection",
            "rating_comments",
            "has_annotations"
        ]
        # Remove tracks that did not pass selection criteria
        if kwargs.get('keep_all_tracks', False):
            sheet = trio_df[~(trio_df['recording_id_for_lbz'].isna())]
        else:
            sheet = trio_df[(trio_df['is_acceptable(Y/N)'] == 'Y') & (~trio_df['youtube_link'].isna())]
        # Strip punctuation from album and track name
        sheet['release_title'] = sheet['release_title'].apply(remove_punctuation)
        sheet['recording_title'] = sheet['recording_title'].apply(remove_punctuation)
        # Preserve the unique Musicbrainz ID for a track
        sheet['mbz_id'] = sheet['recording_id_for_lbz'].str[self.lbz_url_cutoff:]
        # Replace NA values in notes column with empty strings
        sheet = sheet.rename(columns={'Unnamed: 19': 'notes'})
        sheet['notes'] = sheet['notes'].fillna('')
        # Get the year the track was recorded in
        sheet['recording_year'] = pd.to_datetime(sheet['recording_date_estimate']).dt.year.astype(str)
        # Convert has annotation column to boolean
        sheet['has_annotations'] = sheet['has_annotations'].map({'Y': True, np.nan: False})
        # Return the formatted dataframe, as a list of dictionaries
        return (
            sheet.rename(columns=to_rename)
            .drop(columns=to_drop)
            [to_keep]
            .reset_index(drop=True)
            .to_dict(orient='records')
        )

    @staticmethod
    def str_to_dict(s: str) -> dict:
        """Converts a string representation of a dictionary to a dictionary"""
        return {i.split(': ')[0]: i.split(': ')[1] for i in s.split(', ')}

    @staticmethod
    def format_timestamp(ts: str, as_string: bool = True):
        """Formats a timestamp string correctly. Returns as either a datetime or string, depending on `as_string`"""
        ts = str(ts)
        if ts == 'nan':
            return pd.NaT

        fmt = '%M:%S' if len(ts) < 6 else '%H:%M:%S'
        if as_string:
            return datetime.strptime(ts, fmt).strftime(fmt)
        else:
            return datetime.strptime(ts, fmt)

    def get_excerpt_duration(self, start, stop) -> str:
        """Returns the total duration of an excerpt, in the format %M:%S"""
        dur = (
                self.format_timestamp(stop, as_string=False) - self.format_timestamp(start, as_string=False)
        ).total_seconds()
        try:
            return str(timedelta(seconds=dur))[2:]
        except ValueError:
            return ''

    @staticmethod
    def construct_filename(item, id_chars: int = 8, desired_words: int = 5) -> str:
        """Constructs the filename for an item in the corpus"""

        def name_formatter(st: str = "track_name") -> str:
            """Formats the name of a track or album by truncating to a given number of words, etc."""
            # Get the item name itself, e.g. album name, track name
            name = item[st].split(" ")
            # Get the number of words we require
            name_length = len(name) if len(name) < desired_words else desired_words
            return re.sub("[\W_]+", "", "".join(char.lower() for char in name[:name_length]))

        def musician_name_formatter(st: str) -> str:
            """Formats the name of a musician into the format: lastnamefirstinitial, e.g. Bill Evans -> evansb"""
            s = remove_punctuation(st).lower().split(" ")
            try:
                return s[1] + s[0][0]
            except IndexError:
                return 'musicianm'

        # Get the names of our musicians in the correct format
        pianist = musician_name_formatter(item["musicians"]["pianist"])
        bassist = musician_name_formatter(item["musicians"]["bassist"])
        drummer = musician_name_formatter(item["musicians"]["drummer"])
        # Get the required number of words of the track title, nicely formatted
        track = name_formatter("track_name")
        # Return our track name formatted nicely
        try:
            return rf"{pianist}-{track}-{bassist}{drummer}-{item['recording_year']}-{item['mbz_id'][:id_chars]}"
        except TypeError:
            return ''

    def format_first_downbeat(
            self,
            start_ts: float,
            first_downbeat: float
    ) -> float:
        """Gets the position of the first downbeat in seconds, from the start of an excerpt"""
        start = self.format_timestamp(start_ts, as_string=False)
        try:
            start_td = timedelta(hours=start.hour, minutes=start.minute, seconds=start.second)
        except ValueError:
            return np.nan
        else:
            return (timedelta(seconds=first_downbeat) - start_td).total_seconds()

    def format_track_dict(
            self,
            track_dict: dict
    ) -> Generator:
        """Formats each dictionary, corresponding to a single track"""

        to_drop = ['youtube_link', 'start_timestamp', 'end_timestamp', 'bassist', 'drummer']
        # Iterate over every track in our list of dictionaries
        for track in track_dict:
            # Format the YouTube links correctly
            track['links'] = {'external': [i for i in [track['youtube_link']]]}
            # Get the total duration of the excerpt
            track['excerpt_duration'] = self.get_excerpt_duration(track['start_timestamp'], track['end_timestamp'])
            # Format our timestamps correctly
            track['timestamps'] = {
                'start': self.format_timestamp(track['start_timestamp']),
                'end': self.format_timestamp(track['end_timestamp'])
            }
            # Format our first downbeat using our start timestamp
            track['first_downbeat'] = self.format_first_downbeat(track['start_timestamp'], track['first_downbeat'])
            # Replace time signature with integer value
            try:
                track['time_signature'] = int(track['time_signature'])
            except ValueError:
                track['time_signature'] = np.nan
            # Add an empty list for our log
            track['log'] = []
            # Format our musician names correctly
            track['musicians'] = {
                'pianist': track['pianist'],
                'bassist': track['bassist'],
                'drummer': track['drummer'],
                'leader': INSTRUMENTS_TO_PERFORMER_ROLES[self.bandleader_instr]
            }
            # Format our musician photos correctly
            track['photos'] = {
                "musicians": {
                    "pianist": None,
                    "bassist": None,
                    "drummer": None
                },
                "album_artwork": None
            }
            # Construct the filename for this track
            track['fname'] = self.construct_filename(track)
            # Format channel overrides as dictionary, or set key to empty dictionary if overrides are not present
            try:
                track['channel_overrides'] = self.str_to_dict(track['channel_overrides'])
            except AttributeError:
                track['channel_overrides'] = {}
            # Remove key-value pairs we no longer need
            for remove in to_drop:
                del track[remove]
            yield track


def save_annotations(track, trackpath):
    """Saves all annotations from a given `OnsetMaker` instance inside `trackpath`"""
    # Iterate through each instrument
    for instr in INSTRUMENTS_TO_PERFORMER_ROLES.keys():
        # Save a `.csv` file of this performer's onsets
        ons = pd.Series(track.ons[instr])
        ons.to_csv(fr"{trackpath}/{track.item['fname']}_{instr}.csv", header=False, index=False)
    # Save a `.csv` file of the matched beats and onsets
    beats = pd.DataFrame(track.summary_dict)
    beats.to_csv(fr"{trackpath}/{track.item['fname']}_beats.csv", header=True, index=True)
    # Save a `.json` file of the track metadata
    save_json(track.item, trackpath, "metadata")


def generate_corpus_files(corpus_fname: str) -> None:
    """Generates the `.csv` and `.json` files for a single `corpus_fname`"""
    def mkdir(path: str):
        try:
            os.mkdir(path)
        except FileExistsError:
            pass

    # Load in the corpus file
    res = unserialise_object(rf'{get_project_root()}/models/matched_onsets_{corpus_fname}')
    # Make a master directory for this corpus
    dirpath = fr"{get_project_root()}/models/{corpus_fname}"
    mkdir(dirpath)
    # Iterate through each track in the corpus
    for track in res:
        # Make a folder for this track
        trackpath = dirpath + fr'/{track.item["fname"]}'
        mkdir(trackpath)
        save_annotations(track, trackpath)


# TODO: think about refactoring below two functions into src.detect.detect_utils
def load_track_from_files(trackpath: str):
    """Loads a single track from loose files generated in `src.utils.generate_corpus_files`"""
    from src.detect.detect_utils import OnsetMaker
    # Load the JSON metadata file
    item = load_json(fpath=trackpath, fname='metadata')
    # Use this to create a new `OnsetMaker`, but skip processing
    om = OnsetMaker(item=item, skip_processing=True)
    # Read the summary dictionary `.csv` file
    sd = pd.read_csv(rf'{trackpath}/beats.csv', index_col=0)
    # Append the requisite columns to our new `OnsetMaker.summary_dict`
    for col in sd.columns:
        om.summary_dict[col] = sd[col].to_numpy()
    # This starts creating the `OnsetMaker.ons` dictionary
    for instr in INSTRUMENTS_TO_PERFORMER_ROLES.keys():
        om.ons[instr] = np.genfromtxt(rf'{trackpath}/{instr}.csv', delimiter=',')
    om.ons['mix'] = sd['beats'].to_numpy()
    # Get both automatically and manually generated downbeats and coerce into correct format
    for var_ in ['auto', 'manual']:
        om.ons[f'metre_{var_}'] = sd[f'metre_{var_}'].to_numpy()
        om.ons[f'downbeats_{var_}'] = om.ons['mix'][np.where(om.ons[f'metre_{var_}'] == 1)]
    # Update this attribute as it won't be present by default
    om.tempo = np.mean(60 / np.diff(om.ons['mix']))
    return om


def load_corpus_from_files(dirpath: str) -> list:
    """Loads corpus from the loose files generated in `src.utils.generate_corpus_files`"""
    # Filter warnings generated when an onset file has no data in it
    warnings.simplefilter('ignore', UserWarning)
    # Iterate through each folder in our directory and return the completed `OnsetMaker` instances
    return [load_track_from_files(dirpath + '/' + track) for track in os.listdir(dirpath)]


def convert_to_mp3(dirpath: str, ext: str = '.wav', delete: bool = False, cutoff: int = False) -> None:
    """Converts all files with target `.wav` in `dirpath` to low bitrate `.mp3`s"""
    # Iterate through all folders in target directory
    for f in os.listdir(dirpath):
        # If file ends with target extension
        if f.endswith(ext):
            # Define the ffmpeg command
            cmd = [
                "ffmpeg", "-y", "-i", fr"{dirpath}/{f}",
                "-vn", "-ar", "44100", "-ac", "2", "-b:a", "120k",
                fr"{dirpath}/{f.replace(ext, '.mp3')}"
            ]
            if cutoff:
                cmd.insert(-1, '-t')
                cmd.insert(-1, str(cutoff))
            # Open the subprocess
            subprocess.Popen(cmd)
            # Delete the file if we want to do this
            if delete:
                os.remove(fr"{dirpath}/{f}")


def construct_audio_fpath_with_channel_overrides(
        root_fname: str,
        channel: str = None,
        instr: str = None
) -> str:
    """From a root file name, optional channel (`"l"` or `"r"`) and instrument, constructs the complete file name"""
    ext = ''
    # Full format is root-{channel}chan_{instr}.{AUDIO_FILE_FMT}
    if channel is not None:
        ext += f'-{channel}chan'
    if instr is not None:
        ext += f'_{instr}'
    return root_fname + ext + f'.{AUDIO_FILE_FMT}'


if __name__ == '__main__':
    pass
