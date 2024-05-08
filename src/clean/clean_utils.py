#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility classes, functions, and variables used in the data cleaning process"""

import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from math import isclose
from pathlib import Path
from shutil import rmtree

import librosa
import numpy as np
import requests
import yt_dlp
import soundfile as sf
from joblib import Parallel, delayed
from scipy.signal import hilbert
from scipy.ndimage import shift
from yt_dlp.utils import download_range_func, DownloadError

from src import utils


class YtDlpFakeLogger:
    """Fake logging class passed to yt-dlp instances to disable overly-verbose logging and unnecessary warnings"""

    def debug(self, msg=None):
        pass

    def warning(self, msg=None):
        pass

    def error(self, msg=None):
        pass


class HidePrints:
    """Helper class that prevents a function from printing to stdout when used as a context manager"""

    def __enter__(
            self
    ) -> None:
        """Overrides default context manager `__enter__` function to disable prints to `stdout`"""
        self._original_stdout = sys.stdout
        # Supress prints
        sys.stdout = open(os.devnull, 'w')

    def __exit__(
            self,
            exc_type,
            exc_val,
            exc_tb
    ) -> None:
        """Overrides default context manager `__exit__` function to reenable prints to `stdout`"""
        sys.stdout.close()
        # Reset prints back to normal
        sys.stdout = self._original_stdout


class ItemMaker:
    """Makes a single item in the corpus by downloading from YouTube, splitting audio channels, and separating"""

    # Audio file format
    fmt = utils.AUDIO_FILE_FMT
    # Options JSON to pass to yt_dlp when downloading from YouTube
    ydl_opts = {
        "format": f"{fmt}/bestaudio[ext={fmt}]/best",
        "quiet": True,
        "extract_audio": True,
        "overwrites": True,
        "logger": YtDlpFakeLogger,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
    }
    # Tolerance (in seconds) for matching given timestamps with downloaded file
    abs_tol = 0.05    # i.e. 50 milliseconds
    # The instruments we'll conduct source separation on
    instrs = list(utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys())

    def __init__(self, item: dict, **kwargs):
        # Directories containing raw and processed (source-separated) audio, respectively
        self.output_filepath = kwargs.get('output_filepath', rf"{utils.get_project_root()}/data")
        self.raw_audio_loc = rf"{self.output_filepath}/raw/audio"
        self.spleeter_audio_loc = rf"{self.output_filepath}/processed/spleeter_audio"
        self.demucs_audio_loc = rf"{self.output_filepath}/processed/demucs_audio"
        self.mvsep_audio_loc = rf"{self.output_filepath}/processed/mvsep_audio"
        # The dictionary corresponding to one particular item in our corpus JSON
        self.item = item.copy()
        # Empty attribute to hold valid YouTube links
        self.links = []
        # Model to use in Spleeter
        self.spleeter_model = "spleeter:5stems-16kHz"
        self.demucs_model = "htdemucs_6s"
        self.mvsep_model = "MVSEP"
        # The filename for this item, constructed from the parameters of the JSON
        self.fname: str = self.item['fname']
        # The complete filepath for this item
        self.in_file: str = rf"{self.raw_audio_loc}/{self.fname}.{self.fmt}"
        # Source-separation models to use
        self.use_spleeter: bool = kwargs.get('use_spleeter', True)
        self.use_demucs: bool = kwargs.get('use_demucs', True)
        self.use_mvsep: bool = kwargs.get('use_mvsep', True)
        # Whether to get the left and right channels as separate files (helps with bass separation in some recordings)
        self.get_lr_audio: bool = kwargs.get('get_lr_audio', True)
        # Paths to all the source-separated audio files that we'll create (or load)
        self.out_spleeter = [
            rf"{self.spleeter_audio_loc}/{self.fname}_{i}.{self.fmt}"
            if i not in self.item['channel_overrides'].keys()
            else rf"{self.spleeter_audio_loc}/{self.fname}-{self.item['channel_overrides'][i]}chan_{i}.{self.fmt}"
            for i in self.instrs
        ]
        self.out_demucs = [
            rf"{self.demucs_audio_loc}/{self.fname}_{i}.{self.fmt}"
            if i not in self.item['channel_overrides'].keys()
            else rf"{self.demucs_audio_loc}/{self.fname}-{self.item['channel_overrides'][i]}chan_{i}.{self.fmt}"
            for i in self.instrs
        ]
        self.out_mvsep = [
            rf"{self.mvsep_audio_loc}/{self.fname}_{i}.{self.fmt}"
            if i not in self.item['channel_overrides'].keys()
            else rf"{self.mvsep_audio_loc}/{self.fname}-{self.item['channel_overrides'][i]}chan_{i}.{self.fmt}"
            for i in self.instrs
        ]
        # Logger object and empty list to hold messages (for saving)
        self.logger = kwargs.get("logger", None)
        self.logging_messages = []
        # Boolean kwargs that will force this item to be downloaded or separated regardless of local presence
        self.force_download = kwargs.get("force_redownload", False)
        self.force_separation = kwargs.get("force_reseparation", False)
        # Starting and ending timestamps, gathered from the corpus JSON
        self.start = return_timestamp(self.item['timestamps']["start"])
        self.end = return_timestamp(self.item['timestamps']["end"])
        # Amount to multiply file duration by when calculating source separation timeout value
        self.timeout_multiplier_spleeter = kwargs.get("timeout_multiplier_spleeter", 50)
        self.timeout_multiplier_demucs = kwargs.get("timeout_multiplier_demucs", 100)
        self.timeout_multiplier_mvsep = kwargs.get("timeout_multiplier_mvsep", 100)

    def _logger_wrapper(self, msg) -> None:
        """Simple wrapper that logs a given message and indexes it for later access"""

        if self.logger is not None:
            self.logger.info(msg)
        self.logging_messages.append(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]}: {msg}')

    def _get_valid_links(
        self, bad_pattern: str = '"playabilityStatus":{"status":"ERROR"'
    ) -> list:
        """Returns a list of valid YouTube links from the Corpus JSON"""

        checker = lambda s: bad_pattern not in requests.get(s).text
        return [link for link in self.item["links"]["external"] if "youtube" in link and checker(link)]

    def get_item(self) -> None:
        """Tries to find a corpus item locally, and downloads it from the internet if not present"""
        log = f'processing "{self.item["track_name"]}"'
        if any([self.item[st] is not None for st in ['recording_year', 'album_name']]):
            log += f' from {self.item["recording_year"]} album {self.item["album_name"]}, ' \
                   f'leader {self.item["musicians"][self.item["musicians"]["leader"]]} ...'
        # Log start of processing
        self._logger_wrapper(log)
        # Define our list of checks for whether we need to rebuild the item
        checks = [
            # Is the item actually present locally?
            utils.check_item_present_locally(self.in_file),
            # Are we forcing the corpus to rebuild?
            not self.force_download,
            # Have we changed the timestamps for this item since the last time we built it?
            # isclose(float(self.end - self.start), utils.get_audio_duration(self.in_file), abs_tol=self.abs_tol),
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
        # We want to do this regardless of if the file is stored locally
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
                rf'{self.raw_audio_loc}/{self.fname}-{name}chan.{self.fmt}'
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
        if self.start is not None and self.end is not None:
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
                    os.replace(self.in_file + f'.{self.fmt}', self.in_file)
                except FileNotFoundError:
                    pass
                self._logger_wrapper(f"... downloaded successfully from {link_url}")
                break
        # If, after iterating through all our links, we still haven't been able to save the file, then raise an error
        if not utils.check_item_present_locally(self.in_file):
            raise DownloadError(f'Item could not be downloaded, check input links are working')

    def separate_audio(self) -> None:
        """Checks whether to separate audio, gets requred commands, opens subprocessses, then cleans up"""

        def get_preseparation_checks(out_files: list[str], abs_tol: float = None) -> list[bool]:
            """Conducts checks using given list of filenames for whether an item has already been processed"""
            if abs_tol is None:
                abs_tol = self.abs_tol
            dur = lambda dur1, dur2: isclose(dur1, dur2, abs_tol=abs_tol)
            return [
                # Are all the source separated items present locally?
                all(utils.check_item_present_locally(fn) for fn in out_files),
                # Do all the source-separated items have approximately the same duration as the original file?
                all([dur(utils.get_audio_duration(o), utils.get_audio_duration(self.in_file)) for o in out_files]),
                # Have we changed the timestamps for this item since the last time we built it?
                # all([dur(self.end - self.start, utils.get_audio_duration(o)) for o in out_files]),
                # Is the duration of the raw input file *identical* to the duration of our source-separated files?
                # all(utils.get_audio_duration(self.in_file) == utils.get_audio_duration(out) for out in out_files)
            ]

        def separation_handler(separation_class: type, separator_name: str, parallel: bool = True) -> None:
            """Handling function for running separation & cleanup using a given separation class"""
            # Raise an error if we no longer have the input file, for whatever reason
            if not utils.check_item_present_locally(self.in_file):
                raise FileNotFoundError(f"Input file {self.in_file} not present, can't proceed to separation")
            # Initialise the separation class: either _SpleeterMaker or _DeezerMaker
            cls = separation_class(item=self.item, output_filepath=self.output_filepath)
            # Get the commands that we'll pipe into our model: the initial command will separate just the stereo track
            cmds = [cls.get_cmd()]
            # These commands call for separation on the individual right and left channels, as desired
            if self.get_lr_audio and 'channel_overrides' in self.item.keys():
                for ch in set(self.item['channel_overrides'].values()):
                    fname = rf'{self.raw_audio_loc}/{self.fname}-{ch}chan.{self.fmt}'
                    cmds.append(cls.get_cmd(fname))
            # Run each of our separation commands in parallel, using joblib (set n_jobs to number of commands)
            self._logger_wrapper(f"... separating {len(cmds)} tracks with {separator_name}")
            # with HidePrints() as _:
            if parallel:
                Parallel(n_jobs=len(cmds))(delayed(cls.run_separation)(cmd) for cmd in cmds)
            else:
                for cmd in cmds:
                    cls.run_separation(cmd)
            # Clean up after separation by removing any unnecessary files, moving folders etc.
            cls.cleanup_post_separation()

        # Define our list of checks for whether we need to conduct source separation again
        checks = [not self.force_separation]
        if self.use_spleeter:
            checks.extend(get_preseparation_checks(self.out_spleeter))
        if self.use_demucs:
            checks.extend(get_preseparation_checks(self.out_demucs))
        if self.use_mvsep:
            # Use a slightly higher absolute tolerance for mvsep, doesn't seem as time-aligned as spleeter and deezer
            checks.extend(get_preseparation_checks(self.out_mvsep, abs_tol=0.1))
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
            if self.use_mvsep:
                # We don't want to process tracks using MVSEP in parallel, as we'll quickly run out of memory
                separation_handler(_MVSEPMaker, self.mvsep_model, parallel=False)

    def finalize_output(self, include_log: bool = False) -> None:
        """Finalizes the output by cleaning up leftover files and setting any final attributes"""

        # Try and remove the leftover demucs folder, if it exists
        try:
            rmtree(os.path.abspath(rf"{self.demucs_audio_loc}/{self.demucs_model}"))
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
            # Specifies the output codec
            "-c", f"{self.fmt}",
            # This sets the output filename format
            "-f", "{filename}_{instrument}.{codec}",
        ]

    def run_separation(self, cmd: list, good_pattern: str = "written succesfully") -> None:
        """Conducts separation in Spleeter by opening a new subprocess"""

        # TODO: we could check for a pretrained_models folder, as if that isn't present then execution will be longer
        # Get the timeout value: the duration of the item, times the multiplier
        timeout = int(utils.get_audio_duration(os.path.abspath(self.in_file))) * self.timeout_multiplier_spleeter
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
                os.remove(os.path.abspath(rf"{self.spleeter_audio_loc}/{file}"))


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
        timeout = int(utils.get_audio_duration(os.path.abspath(self.in_file))) * self.timeout_multiplier_demucs
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
        demucs_fpath = rf"{self.demucs_audio_loc}/{self.demucs_model}"
        # Demucs creates a new folder for each track. If we're using multiple channels for this item, get all folders
        demucs_folders = [os.path.abspath(rf'{demucs_fpath}/{f}') for f in os.listdir(demucs_fpath) if self.fname in f]
        all_files = []
        # Now, iterate through each folder, and get the absolute path of all the tracks within them
        for fp in demucs_folders:
            all_files.extend(os.path.abspath(rf'{fp}/{f}') for f in os.listdir(fp))
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
            new_name = rf'{self.demucs_audio_loc}/{li[-2]}_{li[-1]}'
            os.replace(old_name, new_name)
        # Remove the demucs folders and all the unwanted files remaining inside them
        for folder in demucs_folders:
            rmtree(folder)


class _MVSEPMaker(ItemMaker):
    MVSEP_LOCATION = f'{utils.get_project_root()}/MVSEP-MDX23-music-separation-model'
    # These are the commands we pass into MVSEP
    MVSEP_KWARGS = {
        'cpu': False,    # Force CUDA
        'overlap_large': 0.4,    # This seems to be required to run on the server
        'overlap_small': 0.3,    # This seems to be required to run on the server
        'chunk_size': 500000,
        'single_onnx': False,
        'large_gpu': False,    # If only...!
        'use_kim_model_1': False,
        'only_vocals': False
    }
    # Passed in to librosa.load
    LOAD_KWS = dict(
        sr=utils.SAMPLE_RATE,
        dtype=np.float64,
        offset=0,
        res_type='soxr_vhq'
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_cmd(self, in_file: str = None) -> list:
        """Gets the required command for running MVSEP as a subprocess"""
        if in_file is None:
            in_file = self.in_file
        # if in_file is not None:
        #     self.in_file = in_file
        return {
            'input_audio': [os.path.abspath(in_file)],
            'output_folder': os.path.abspath(self.mvsep_audio_loc),
            **self.MVSEP_KWARGS
        }

    def run_separation(self, cmd: list) -> None:
        """Runs separation in MVSEP"""
        # Having to do this is bad practice, but I'm not sure of any other way
        sys.path.insert(1, self.MVSEP_LOCATION)
        # noinspection PyUnresolvedReferences
        # noinspection PyPackageRequirements
        from inference import predict_with_model
        predict_with_model(cmd)
        # We have to add a short delay in here so that the files have time to save in the correct location
        time.sleep(2)
        self._logger_wrapper(f"... item separated successfully")

    def rename_piano_file(self, root_fpath: str) -> None:
        """Tries to rename the `other` file to `piano`, if it can be found"""
        # Assemble the filepath for the piano, with any channel overrides
        piano_channel = ''
        if 'piano' in self.item['channel_overrides']:
            piano_channel += f"-{self.item['channel_overrides']['piano']}chan"
        piano_src = Path(f'{root_fpath}{piano_channel}_other.{utils.AUDIO_FILE_FMT}')
        # If the piano separated file exists, rename it
        if os.path.isfile(piano_src):
            piano_dst = Path(f'{root_fpath}{piano_channel}_piano.{utils.AUDIO_FILE_FMT}')
            os.rename(piano_src, piano_dst)

    def get_files_to_keep(self, root_fpath: str):
        """Returns a generator of files to keep from the separation"""
        for instr in self.instrs:
            # If we have a channel override for this instrument, get the correct filename
            if instr in self.item['channel_overrides']:
                fp = Path(f'{root_fpath}-{self.item["channel_overrides"][instr]}chan_{instr}.{utils.AUDIO_FILE_FMT}')
            # Otherwise, use the base file name
            else:
                fp = Path(f'{root_fpath}_{instr}.{utils.AUDIO_FILE_FMT}')
            yield fp

    @staticmethod
    def calculate_best_shift(
            raw_env: np.array,
            proc_env: np.array,
            start: float = 5.,
            duration: float = 10.,
            max_shift: float = 2.,
            step: int = 10
    ) -> int:
        """Calculate the number of samples required to shift `proc_env` to maximise `r` vs `proc_env`"""
        def shift_signals(shift_samples: int) -> tuple[int, float]:
            # Shift the demixed audio signal by the correct number of samples
            y = proc_env[int(utils.SAMPLE_RATE * start) + shift_samples: int(utils.SAMPLE_RATE * end) + shift_samples]
            # Return the number of samples and the correlation coefficient
            return shift_samples, np.corrcoef(x, y)[0, 1]

        # Calculate the end timestamp for use in correlation
        end = start + duration
        # Truncate the raw audio envelope between our start and end timestamps
        x = raw_env[int(utils.SAMPLE_RATE * start): int(utils.SAMPLE_RATE * end)]
        # Get the array of sample values to shift the demixed audio by
        shifter = range(int(-utils.SAMPLE_RATE * max_shift), int(utils.SAMPLE_RATE * max_shift), step)
        # In parallel, shift the demixed signal by each sample value
        with Parallel(n_jobs=-1, verbose=1, backend='threading') as par:
            res = par(delayed(shift_signals)(i) for i in shifter)
        # Get the number of samples that resulted in the best positive `r` score and log
        bigr = max(i[1] for i in res)
        print(f'Best r: {round(bigr, 2)}')
        return [i[0] for i in res if i[1] == bigr][0]

    @staticmethod
    def shift_audio_signal(
            audio: np.array,
            n_samples: int,
            shape_to_match: int
    ) -> np.array:
        """Shift a signal `audio` by samples `n_samples` and pad to the same shape as `shape_to_match`"""
        # Shift the audio signal and fill any empty samples with 0 (i.e. silence)
        shifted = shift(audio, (0, -n_samples), cval=0)
        # Add some padding to the end of the signal if it's shorter than the desired size
        if shifted.shape[1] < shape_to_match:
            padding = np.zeros((2, shape_to_match - shifted.shape[1]))
            shifted = np.hstack((shifted, padding))
        # Truncate the end of the signal if it's longer than the desired size
        elif shifted.shape[1] > shape_to_match:
            shifted = shifted[:, :shape_to_match]
        return shifted

    def align_audio_signals(self, files_to_pad: list[str]):
        """For a list of demixes stems `files_to_pad`, align these with their raw audio file"""
        # Load in the raw audio, in mono
        raw, _ = librosa.load(self.in_file, mono=True, **self.LOAD_KWS)
        # Load in the source separated files, in stereo
        audio_to_pad = [librosa.load(f, mono=False, **self.LOAD_KWS)[0] for f in files_to_pad]
        # Sum all the source separated files and convert to mono
        proc = sum(audio_to_pad).mean(axis=0)
        # Compute the audio envelopes for the raw audio and the summed demixed audio
        raw_env = np.abs(hilbert(raw))
        proc_env = np.abs(hilbert(proc))
        # Calculate the number of samples needed to pad the signal by
        self._logger_wrapper(f'... calculating number of seconds to shift separated audio by')
        pad_samples = self.calculate_best_shift(raw_env, proc_env)
        pad_seconds = round(pad_samples / utils.SAMPLE_RATE, 2)
        self._logger_wrapper(f'... shifting by {pad_seconds} secs to align separated audio with raw audio')
        # For each of our source separated stems, pad the audio and return
        for audio in audio_to_pad:
            yield self.shift_audio_signal(audio, pad_samples, raw.shape[0])

    def cleanup_post_separation(self, new_dirpath: str = None) -> None:
        """Cleans up after running MVSEP by renaming files and removing any unnecessary files"""
        # Get the root name of our all our separated files
        mvsep_fpath = os.path.join(
            self.mvsep_audio_loc, os.path.split(self.in_file)[-1].replace('.' + utils.AUDIO_FILE_FMT, '')
        )
        # Rename the `other` file to `piano`, if we have it
        self.rename_piano_file(mvsep_fpath)
        # Get the names of the files we want to keep
        to_keep = list(self.get_files_to_keep(mvsep_fpath))
        # Iterate through each file
        for f in os.listdir(self.mvsep_audio_loc):
            f = os.path.join(self.mvsep_audio_loc, f)
            if mvsep_fpath not in f:
                continue
            # Remove the file if we don't need it
            if Path(f) not in to_keep:
                os.remove(f)
        # Otherwise, pad the audio signal to be the same length as the input audio and save
        padded_audios = self.align_audio_signals(to_keep)
        for audio, fname in zip(padded_audios, to_keep):
            if new_dirpath is not None:
                fname = Path(os.path.join(new_dirpath, fname.parts[-1]))
            with open(fname, 'wb') as fp:
                # We have to transpose the audio signal (i.e. columns -> rows) so that soundfile will accept it
                sf.write(fp, audio.transpose(), utils.SAMPLE_RATE)


def return_timestamp(timestamp: str = "start", ) -> int:
    """Returns a formatted timestamp from a JSON element"""
    try:
        fmt = '%M:%S' if len(timestamp) < 6 else '%H:%M:%S'
        dt = datetime.strptime(timestamp, fmt)
        return int(timedelta(hours=dt.hour, minutes=dt.minute, seconds=dt.second).total_seconds())
    except (ValueError, TypeError):
        return None


if __name__ == '__main__':
    import logging
    logger = logging.getLogger(__name__)
    # Create the `CorpusMaker` object
    cm = utils.CorpusMaker.from_excel('corpus_updated', only_annotated=False, only_30_corpus=False)
    for track in cm.tracks:
        # Create the `ItemMaker` object and process the item
        # im = ItemMaker(item=rb, use_spleeter=False, use_demucs=False, logger=logger)
        # im.get_item()
        # im.separate_audio()
        # im.finalize_output()
        logger.info(f'Now working on ')
        make = _MVSEPMaker(item=track, logger=logger)
        make.cleanup_post_separation(new_dirpath=f'{utils.get_project_root()}/data/processed/mvsep_audio_shifted')
