#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Generates the final dataset of separated recordings from the items in the corpus"""

import logging
import os
import re
import subprocess
from datetime import datetime, timedelta
from math import isclose
from pathlib import Path
from shutil import rmtree
from string import punctuation
from time import time

import click
import pandas as pd
import requests
import yt_dlp
from dotenv import find_dotenv, load_dotenv
from joblib import Parallel, delayed
from yt_dlp.utils import download_range_func, DownloadError

from src.utils import analyse_utils as autils


class CorpusMakerFromExcel:
    """Converts a multi-sheet Excel spreadsheet into the required format for processing"""
    excel_ext = 'xlsx'
    lbz_url_cutoff = 49
    json_indent = 4

    def __init__(
            self,
            fname: str,
            bandleader: str,
            dump_json: bool = True,
            **kwargs
    ):
        self.bandleader = bandleader
        self.bandleader_role = kwargs.get('bandleader_role', 'pianist')
        fpath = fr'{autils.get_project_root()}\references\{fname}.{self.excel_ext}'
        self.tracks = []
        for trio in pd.read_excel(pd.ExcelFile(fpath), None, header=1).values():
            self.tracks.extend(self.format_track_dict(self.format_trio_dataframe(trio)))
        if dump_json:
            autils.save_json(self.tracks, fr'{autils.get_project_root()}\references',
                             f'corpus_{self.bandleader.lower().replace(" ", "_")}')

    @autils.disable_settingwithcopy_warning
    def format_trio_dataframe(
            self,
            trio_df: pd.DataFrame
    ) -> list[dict]:
        """Formats the dataframe for an individual trio and returns a list of dictionaries"""

        # We remove these columns from the dataframe
        to_drop = ['recording_id_for_lbz', 'recording_date_estimate', 'is_acceptable(Y/N)', 'link']
        # We rename these columns
        to_rename = {
            'bass': 'bassist',
            'drums': 'drummer',
            'release_title': 'album_name',
            'recording_title': 'track_name'
        }
        # We keep these columns, in the following order
        to_keep = [
            'track_name',
            'album_name',
            'recording_year',
            'bassist',
            'drummer',
            'youtube_link',
            'channel_overrides',
            'start_timestamp',
            'end_timestamp',
            'mbz_id',
            'notes',
        ]
        # Remove tracks that did not pass selection criteria
        sheet = trio_df[(trio_df['is_acceptable(Y/N)'] == 'Y') & (~trio_df['youtube_link'].isna())]
        # Strip punctuation from album and track name
        sheet['release_title'] = sheet['release_title'].apply(self.remove_punctuation)
        sheet['recording_title'] = sheet['recording_title'].apply(self.remove_punctuation)
        # Preserve the unique Musicbrainz ID for a track
        sheet['mbz_id'] = sheet['recording_id_for_lbz'].str[self.lbz_url_cutoff:]
        # Replace NA values in notes column with empty strings
        sheet['notes'] = sheet['notes'].fillna('')
        # Get the year the track was recorded in
        sheet['recording_year'] = pd.to_datetime(sheet['recording_date_estimate']).dt.year.astype(str)
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
    def remove_punctuation(s: str) -> str:
        """Removes punctuation from a string"""
        return s.translate(str.maketrans('', '', punctuation)).replace('’', '')

    @staticmethod
    def format_timestamp(ts: str, as_string: bool = True):
        """Formats a timestamp string correctly. Returns as either a datetime or string, depending on `as_string`"""
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
        return str(timedelta(seconds=dur))[2:]

    def format_track_dict(
            self,
            track_dict: dict
    ):
        """Formats each dictionary, corresponding to a single track"""

        to_drop = ['youtube_link', 'start_timestamp', 'end_timestamp', 'bassist', 'drummer',]
        # Iterate over every track in our list of dictionaries
        for track in track_dict:
            # Format the YouTube links correctly
            track['links'] = {'external': [l for l in [track['youtube_link']]]}
            # Get the total duration of the excerpt
            track['excerpt_duration'] = self.get_excerpt_duration(track['start_timestamp'], track['end_timestamp'])
            # Format our timestamps correctly
            track['timestamps'] = {
                'start': self.format_timestamp(track['start_timestamp']),
                'end': self.format_timestamp(track['end_timestamp'])
            }
            # Add an empty list for our log
            track['log'] = []
            # Format our musician names correctly
            track['musicians'] = {
                'pianist': self.bandleader,
                'bassist': track['bassist'],
                'drummer': track['drummer'],
                'leader': self.bandleader_role
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
            # Format channel overrides as dictionary, or set key to empty dictionary if not present
            try:
                track['channel_overrides'] = self.str_to_dict(track['channel_overrides'])
            except AttributeError:
                track['channel_overrides'] = {}
            # Remove key-value pairs we no longer need
            for remove in to_drop:
                del track[remove]
            yield track


class ItemMaker:
    """Makes a single item in the corpus by downloading from YouTube, splitting audio channels, and separating"""

    # Options JSON to pass to yt_dlp when downloading from YouTube
    ydl_opts = {
        "format": f"{autils.FILE_FMT}/bestaudio[ext={autils.FILE_FMT}]/best",
        "quiet": True,
        "extract_audio": True,
        "overwrites": True,
        "logger": autils.YtDlpFakeLogger,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
    }
    # Tolerance (in seconds) for matching given timestamps with downloaded file
    abs_tol = 0.05    # i.e. 50 milliseconds
    # The instruments we'll conduct source separation on
    instrs = list(autils.INSTRS_TO_PERF.keys())

    def __init__(self, item: dict, output_filepath: str, **kwargs):
        # Directories containing raw and processed (source-separated) audio, respectively
        self.output_filepath=output_filepath
        self.raw_audio_loc = rf"{self.output_filepath}\raw\audio"
        self.spleeter_audio_loc = rf"{self.output_filepath}\processed\spleeter_audio"
        self.demucs_audio_loc = rf"{self.output_filepath}\processed\demucs_audio"
        # The dictionary corresponding to one particular item in our corpus JSON
        self.item = item.copy()
        # Empty attribute to hold valid YouTube links
        self.links = []
        # Model to use in Spleeter
        self.spleeter_model = "spleeter:5stems-16kHz"
        self.demucs_model = "htdemucs_6s"
        # The filename for this item, constructed from the parameters of the JSON
        self.fname: str = self._construct_filename(**kwargs)
        # The complete filepath for this item
        self.in_file: str = rf"{self.raw_audio_loc}\{self.fname}.{autils.FILE_FMT}"
        # Source-separation models to use
        self.use_spleeter: bool = kwargs.get('use_spleeter', True)
        self.use_demucs: bool = kwargs.get('use_demucs', True)
        # Whether to get the left and right channels as separate files (helps with bass separation in some recordings)
        self.get_lr_audio: bool = kwargs.get('get_lr_audio', True)
        # Paths to all the source-separated audio files that we'll create (or load)
        self.out_spleeter = [
            rf"{self.spleeter_audio_loc}\{self.fname}_{i}.{autils.FILE_FMT}"
            if i not in self.item['channel_overrides'].keys()
            else rf"{self.spleeter_audio_loc}\{self.fname}-{self.item['channel_overrides'][i]}chan_{i}.{autils.FILE_FMT}"
            for i in self.instrs
        ]
        self.out_demucs = [
            rf"{self.demucs_audio_loc}\{self.fname}_{i}.{autils.FILE_FMT}"
            if i not in self.item['channel_overrides'].keys()
            else rf"{self.demucs_audio_loc}\{self.fname}-{self.item['channel_overrides'][i]}chan_{i}.{autils.FILE_FMT}"
            for i in self.instrs
        ]
        # Logger object and empty list to hold messages (for saving)
        self.logger = kwargs.get("logger", None)
        self.logging_messages = []
        # Boolean kwargs that will force this item to be downloaded or separated regardless of local presence
        self.force_download = kwargs.get("force_redownload", False)
        self.force_separation = kwargs.get("force_reseparation", False)
        # Starting and ending timestamps, gathered from the corpus JSON
        self.start, self.end = self._return_audio_timestamp("start"), self._return_audio_timestamp("end")
        # Amount to multiply file duration by when calculating source separation timeout value
        self.timeout_multiplier_spleeter = kwargs.get("timeout_multiplier_spleeter", 5)
        self.timeout_multiplier_demucs = kwargs.get("timeout_multiplier_spleeter", 10)

    def _logger_wrapper(self, msg) -> None:
        """Simple wrapper that logs a given message and indexes it for later access"""

        if self.logger is not None:
            self.logger.info(msg)
        self.logging_messages.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]}: {msg}')

    def _construct_filename(self, **kwargs) -> str:
        """Constructs the filename for an item in the corpus"""

        def name_formatter(st: str = "track_name") -> str:
            """Formats the name of a track or album by truncating to a given number of words, etc."""
            # Get the number of words we desire for this item
            desired_words = kwargs.get(f"{st}_len", 5)
            # Get the item name itself, e.g. album name, track name
            name = self.item[st].split(" ")
            # Get the number of words we require
            name_length = len(name) if len(name) < desired_words else desired_words
            return re.sub("[\W_]+", "", "".join(char.lower() for char in name[:name_length]))

        def musician_name_formatter(st: str) -> str:
            """Formats the name of a musician into the format: lastnamefirstinitial, e.g. Bill Evans -> evansb"""
            s = st.lower().split(" ")
            return s[1] + s[0][0]

        # Get the names of our musicians in the correct format
        pianist = musician_name_formatter(self.item["musicians"]["pianist"])
        bassist = musician_name_formatter(self.item["musicians"]["bassist"])
        drummer = musician_name_formatter(self.item["musicians"]["drummer"])
        # Get the required number of words of the track title, nicely formatted
        track = name_formatter("track_name")
        # Get our album recording year
        year = self.item["recording_year"]
        # Get the first 8 characters of our ID, used to separate multiple takes of one track
        i = self.item['mbz_id'][:8]
        return rf"{pianist}-{track}-{bassist}{drummer}-{year}-{i}"

    def _get_valid_links(
        self, bad_pattern: str = '"playabilityStatus":{"status":"ERROR"'
    ) -> list:
        """Returns a list of valid YouTube links from the Corpus JSON"""

        checker = lambda s: bad_pattern not in requests.get(s).text
        return [link for link in self.item["links"]["external"] if "youtube" in link and checker(link)]

    def _return_audio_timestamp(self, timestamp: str = "start", ) -> int:
        """Returns a formatted timestamp from a JSON element"""

        fmt = '%M:%S' if len(self.item['timestamps'][timestamp]) < 6 else '%H:%M:%S'
        try:
            dt = datetime.strptime(self.item["timestamps"][timestamp], fmt)
            return int(timedelta(hours=dt.hour, minutes=dt.minute, seconds=dt.second).total_seconds())
        # TODO: figure out the required exception type to go here
        except (ValueError, TypeError):
            return None

    def get_item(self) -> None:
        """Tries to find a corpus item locally, and downloads it from the internet if not present"""

        # Log start of processing
        self._logger_wrapper(
            f'processing "{self.item["track_name"]}" from {self.item["recording_year"]} album {self.item["album_name"]}, '
            f'leader {self.item["musicians"][self.item["musicians"]["leader"]]} ...'
        )
        # Define our list of checks for whether we need to rebuild the item
        checks = [
            # Is the item actually present locally?
            autils.check_item_present_locally(self.in_file),
            # Are we forcing the corpus to rebuild?
            not self.force_download,
            # Have we changed the timestamps for this item since the last time we built it?
            isclose(float(self.end - self.start), autils.get_audio_duration(self.in_file), abs_tol=self.abs_tol),
        ]
        # If we pass all checks, then go ahead and get the item locally (skip downloading it)
        if all(checks):
            self._logger_wrapper(f"... skipping download, item present locally")
        # Otherwise, rebuild the item
        else:
            # We get the valid links here, so we don't waste time checking links if an item is already present locally
            self.links = self._get_valid_links()
            # Download the item from YouTube
            self._download_audio_excerpt_from_youtube()
            # Separate the audio file into left and right channels and create audio files for each as required
            # This function won't do anything if we haven't specifically called for channel overrides
            self._split_left_right_audio_channels()

    def _split_left_right_audio_channels(self, timeout: int = 10) -> None:
        """Splits audio into left and right channels for independent processing"""

        mappings: dict = {'l': '0.0.0', 'r': '0.0.1'}
        # If we haven't specified channel overrides, the dictionary will be empty, so this loop won't do anything
        for name in self.item['channel_overrides'].values():
            cmd = [
                # Initialise ffmpeg and force overwriting if file is already present
                'ffmpeg', '-y',
                # Specify input file location
                '-i', self.in_file,
                # Specify required channel mapping
                '-map_channel', mappings[name],
                # Specify output location
                rf'{self.raw_audio_loc}\{self.fname}-{name}chan.{autils.FILE_FMT}'
            ]
            # Open the subprocess and kill if it hasn't completed after a given time
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True,)
            try:
                # This will block execution until the above process has completed
                _, __ = p.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                p.kill()
                raise TimeoutError(f"... error when splitting track channels: timed out after {timeout} seconds")

    def _download_audio_excerpt_from_youtube(self) -> None:
        """Downloads an item in the corpus from a YouTube link"""

        # Set our options in yt_dlp
        self.ydl_opts["outtmpl"] = self.in_file
        self.ydl_opts["download_ranges"] = download_range_func(None, [(self.start, self.end)])
        # Iterate through all of our valid YouTube links
        for link_num, link_url in enumerate(self.links):
            # Try and download from each link
            try:
                with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                    ydl.download(link_url)
            # If we get an error, continue on to the next link or retry from the
            except DownloadError as err:
                self._logger_wrapper(f"... error when downloading from {link_url} ({err}), retrying ...")
                self._download_audio_excerpt_from_youtube()
            # If we've downloaded successfully, break out of the loop
            else:
                # Silently try and rename the file, if we've accidentally appended the file format twice
                try:
                    os.rename(self.in_file + f'.{autils.FILE_FMT}', self.in_file)
                except FileNotFoundError:
                    pass
                self._logger_wrapper(f"... downloaded successfully from {link_url}")
                break
        # If, after iterating through all our links, we still haven't been able to save the file, then raise an error
        if not autils.check_item_present_locally(self.in_file):
            raise DownloadError(f'Item could not be downloaded, check input links are working')

    def separate_audio(self) -> None:
        """Checks whether to separate audio, gets requred commands, opens subprocessses, then cleans up"""

        def get_preseparation_checks(out_files: list[str]) -> list[bool]:
            """Conducts checks using given list of filenames for whether an item has already been processed"""
            dur = lambda dur1, dur2: isclose(dur1, dur2, abs_tol=self.abs_tol)
            return [
                # Are all the source separated items present locally?
                all(autils.check_item_present_locally(fn) for fn in out_files),
                # Do all the source-separated items have approximately the same duration as the original file?
                all([dur(autils.get_audio_duration(o), autils.get_audio_duration(self.in_file)) for o in out_files]),
                # Have we changed the timestamps for this item since the last time we built it?
                all([dur(self.end - self.start, autils.get_audio_duration(o)) for o in out_files]),
                # Is the duration of the raw input file *identical* to the duration of our source-separated files?
                all(autils.get_audio_duration(self.in_file) == autils.get_audio_duration(out) for out in out_files)
            ]

        def separation_handler(separation_class: type, separator_name: str) -> None:
            """Handler function for running separation & cleanup using a given separation class"""
            # Raise an error if we no longer have the input file, for whatever reason
            if not autils.check_item_present_locally(self.in_file):
                raise FileNotFoundError(f"Input file {self.in_file} not present, can't proceed to separation")
            # Initialise the separation class: either _SpleeterMaker or _DeezerMaker
            cls = separation_class(item=self.item, output_filepath=self.output_filepath)
            # Get the commands that we'll pipe into our model: the initial command will separate just the stereo track
            cmds = [cls.get_cmd()]
            # These commands call for separation on the individual right and left channels, as desired
            if self.get_lr_audio and 'channel_overrides' in self.item.keys():
                for ch in set(self.item['channel_overrides'].values()):
                    fname = rf'{self.raw_audio_loc}\{self.fname}-{ch}chan.{autils.FILE_FMT}'
                    cmds.append(cls.get_cmd(fname))
            # Run each of our separation commands in parallel, using joblib (set n_jobs to number of commands)
            self._logger_wrapper(f"separating {len(cmds)} tracks with {separator_name} ...")
            with autils.HidePrints() as _:
                Parallel(n_jobs=len(cmds))(delayed(cls.run_separation)(cmd) for cmd in cmds)
            # Clean up after separation by removing any unnecessary files, moving folders etc.
            cls.cleanup_post_separation()

        # Define our list of checks for whether we need to conduct source separation again
        checks = [not self.force_separation]
        if self.use_spleeter:
            checks.extend(get_preseparation_checks(self.out_spleeter))
        if self.use_demucs:
            checks.extend(get_preseparation_checks(self.out_demucs))
        # If we pass all the checks, then we can skip rebuilding the source-separated tracks
        if all(checks):
            self._logger_wrapper(f"... skipping separation, items present locally")
            return
        # Otherwise, we need to build the source-separated items
        else:
            if self.use_spleeter:
                separation_handler(_SpleeterMaker, self.spleeter_model)
            if self.use_demucs:
                separation_handler(_DemucsMaker, self.demucs_model)

    def finalize_output(self, include_log: bool = False) -> None:
        """Finalizes the output by cleaning up leftover files and setting any final attributes"""

        # Try and remove the leftover demucs folder, if it exists
        try:
            rmtree(os.path.abspath(rf"{self.demucs_audio_loc}\{self.demucs_model}"))
        except FileNotFoundError:
            pass
        # Set a few additional variables within the corpus item
        self.item['fname'] = self.fname
        if include_log:
            self.item["log"] = self.logging_messages
        else:
            self.item['log'] = []
        # Log the end of processing
        self._logger_wrapper("... finished processing item")


class _SpleeterMaker(ItemMaker):
    """Internal class called in ItemMaker for separating audio using Deezer's Spleeter model"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_cmd(self, in_file: str = None) -> list:
        """Gets the required command for running Spleeter as a subprocess"""

        if in_file is None:
            in_file = self.in_file
        return [
            # Opens Spleeter in separation mode
            "spleeter", "separate",
            # Specifies the model to use, defaults to 5stems-16kHz
            "-p", self.spleeter_model,
            # Specifies the correct output directory
            "-o", f"{os.path.abspath(self.spleeter_audio_loc)}",
            # Specifies the input filepath for this item
            f"{os.path.abspath(in_file)}",
            # Specifies the output codec, default to m4a
            "-c", f"{autils.FILE_FMT}",
            # This sets the output filename format
            "-f", "{filename}_{instrument}.{codec}",
        ]

    def run_separation(self, cmd: list, good_pattern: str = "written succesfully") -> None:
        """Conducts separation in Spleeter by opening a new subprocess"""

        # TODO: we could check for a pretrained_models folder, as if that isn't present then execution will be longer
        # Get the timeout value: the duration of the item, times the multiplier
        timeout = int((self.end - self.start) * self.timeout_multiplier_spleeter)
        # Open the subprocess. The additional arguments allow us to capture the output
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True,)
        # Open the subprocess and kill if it exceeds the timeout specified
        try:
            # This will block execution until the above process has completed
            out, err = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            raise TimeoutError(f"... error when separating: process timed out after {timeout} seconds")
        else:
            # Check to make sure the expected output is returned by subprocess
            if good_pattern not in out:
                self._logger_wrapper(f"... error when separating: {out}")
            else:
                self._logger_wrapper(f"... item separated successfully")

    def cleanup_post_separation(self) -> None:
        """Cleans up after spleeter by removing unnecessary files and unused files"""

        files_to_keep = []
        all_files = [f for f in os.listdir(self.spleeter_audio_loc) if self.fname in f]
        # Iterate through the instruments we want
        for instr in self.instrs:
            # If we have channel overrides, get the compatible audio file for this instrument
            if instr in self.item['channel_overrides']:
                fs = [f for f in all_files if all([f'-{self.item["channel_overrides"][instr]}chan' in f, instr in f])]
            # Otherwise, get the stereo audio file as a default
            else:
                fs = [f for f in all_files if all([f'-rchan' not in f, '-lchan' not in f, instr in f])]
            files_to_keep.extend(fs)
        # Iterate through all files and remove those which we don't want to keep
        for file in all_files:
            if file not in files_to_keep:
                os.remove(os.path.abspath(rf"{self.spleeter_audio_loc}\{file}"))


class _DemucsMaker(ItemMaker):
    """Internal class called in ItemMaker for separating audio using Facebook's HTDemucs model"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_cmd(self, in_file: str = None) -> list:
        """Gets the required command for running demucs as a subprocess"""

        if in_file is None:
            in_file = self.in_file
        return [
            "demucs",
            # Specify the input file path
            rf"{os.path.abspath(in_file)}",
            # Specify the model to use
            "-n", self.demucs_model,
            # Specify the desired output directory
            "-o", rf"{os.path.abspath(self.demucs_audio_loc)}"
        ]

    def run_separation(self, cmd) -> None:
        """Conducts separation in Demucs by opening a new subprocess"""

        # Get the timeout value: the duration of the item, times the multiplier
        timeout = int((self.end - self.start) * self.timeout_multiplier_demucs)
        # Open the subprocess. The additional arguments hide the print functions from Demucs.
        p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, universal_newlines=True)
        # Keep the subprocess alive until it either finishes or the timeout is hit
        try:
            # This will block execution until the above process has completed
            _, __ = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            raise TimeoutError(f"... error when separating: process timed out after {timeout} seconds")
        else:
            self._logger_wrapper(f"... item separated successfully")

    def cleanup_post_separation(self) -> None:
        """Cleans up after demucs by removing unnecessary files and moving file locations"""

        # Demucs filestructure looks like: model_name/track_name/item_name.
        demucs_fpath = rf"{self.demucs_audio_loc}\{self.demucs_model}"
        # Demucs creates a new folder for each track. If we're using multiple channels for this item, get all folders
        demucs_folders = [os.path.abspath(rf'{demucs_fpath}\{f}') for f in os.listdir(demucs_fpath) if self.fname in f]
        all_files = []
        # Now, iterate through each folder, and get the absolute path of all the tracks within them
        for fp in demucs_folders:
            all_files.extend(os.path.abspath(rf'{fp}\{f}') for f in os.listdir(fp))
        files_to_keep = []
        # Iterate through the instruments in our trio and get the correct filepaths
        for instr in self.instrs:
            if instr in self.item['channel_overrides']:
                fs = [f for f in all_files if all([f'-{self.item["channel_overrides"][instr]}chan' in f, instr in f])]
            else:
                fs = [f for f in all_files if all([f'-rchan' not in f, '-lchan' not in f, instr in f])]
            files_to_keep.extend(fs)
        # Move and rename the files corresponding to the instruments in the trio
        for old_name in files_to_keep:
            li = os.path.normpath(old_name).split(os.path.sep)
            new_name = rf'{self.demucs_audio_loc}\{li[-2]}_{li[-1]}'
            os.rename(old_name, new_name)
        # Remove the demucs folders and all the unwanted files remaining inside them
        for folder in demucs_folders:
            rmtree(folder)


@click.command()
@click.option("-i", "input_filepath", type=click.Path(exists=True), default=rf"{autils.get_project_root()}\references")
@click.option("-o", "output_filepath", type=click.Path(exists=True), default=rf"{autils.get_project_root()}\data")
@click.option("--force-download", "force_download", is_flag=True, default=False, help='Force download from YouTube')
@click.option("--force-separation", "force_separation", is_flag=True, default=False, help='Force source separation')
@click.option("--no-spleeter", "disable_spleeter", is_flag=True, default=False, help='Disable spleeter for separation')
@click.option("--no-demucs", "disable_demucs", is_flag=True, default=False, help='Disable demucs for separation')
def main(
        input_filepath: str,
        output_filepath: str,
        force_download: bool,
        force_separation: bool,
        disable_spleeter: bool,
        disable_demucs: bool
) -> None:
    """Runs processing scripts to turn corpus from (./references) into audio, ready to be analyzed"""

    # Start the timer
    start = time()
    # Set the logger
    logger = logging.getLogger(__name__)
    # Open the corpus Excel file using our custom class
    corpus = CorpusMakerFromExcel(
        fname='corpus_bill_evans',
        bandleader='Bill Evans',
        bandleader_role='pianist',
        dump_json=False
    ).tracks
    js = []
    # Iterate through each item in the corpus and make it
    for corpus_item in corpus:
        im = ItemMaker(
            item=corpus_item,
            logger=logger,
            track_name_len=5,
            output_filepath=output_filepath,
            get_lr_audio=True,
            force_reseparation=force_separation,
            force_redownload=force_download,
            use_spleeter=not disable_spleeter,
            use_demucs=not disable_demucs
        )
        im.get_item()
        im.separate_audio()
        im.finalize_output()
        # Append the item results, log etc. to save
        js.append(im.item)
    # Dump our finalized output to a new json and save in the output directory
    autils.save_json(
        obj=js,
        fpath=input_filepath,
        fname='corpus_bill_evans'
    )
    # Log the total completion time
    logger.info(f"dataset made in {round(time() - start)} secs !")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
