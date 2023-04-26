#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generates the final dataset of recordings from the items listed in \references\corpus.json
"""

import json
import logging
import os
import re
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from time import time

import audioread
import click
import requests
import yt_dlp
from dotenv import find_dotenv, load_dotenv
from yt_dlp.utils import download_range_func, DownloadError


class _YtDlpFakeLogger:
    """
    Fake logging class passed to yt-dlp instances to disable overly-verbose logging and unnecessary warnings
    """

    def debug(self, msg=None):
        pass

    def warning(self, msg=None):
        pass

    def error(self, msg=None):
        pass


class ItemMaker:
    """
    Makes a single item in the corpus by reading the corresponding JSON entry, attempting to locate the item locally,
    and getting the audio from YouTube if not available (with the required start and stop timestamps).
    """

    # The desired length of our item ID, e.g. 000031 = 6
    id_len = 6
    # The file codec to use when saving
    file_fmt = "m4a"
    sample_rate = 44100
    # Options to pass to yt_dlp when downloading from YouTube
    ydl_opts = {
        "format": f"{file_fmt}/bestaudio[ext={file_fmt}]/best",
        "quiet": True,
        "extract_audio": True,
        "overwrites": True,
        "logger": _YtDlpFakeLogger,
    }
    # Model to use in Spleeter
    model = "spleeter:5stems-16kHz"
    # The instruments we'll conduct source separation on
    instrs = ["piano", "bass", "drums"]

    def __init__(self, item: dict, index: int, output_filepath: str, **kwargs):
        # Directories containing raw and processed (source-separated) audio, respectively
        self.raw_audio_loc = rf"{output_filepath}\raw\audio"
        self.separate_audio_loc = rf"{output_filepath}\processed\audio"
        # The dictionary corresponding to one particular item in our corpus JSON
        self.item = item.copy()
        self.item["id"] = str(index).zfill(self.id_len)
        # Empty attribute to hold valid YouTube links
        self.links = []
        # The filename for this item, constructed from the parameters of the JSON
        self.fname = self._construct_filename(**kwargs)
        # The complete filepath for this item
        self.in_file = rf"{self.raw_audio_loc}\{self.fname}.{self.file_fmt}"
        # Paths to all the source-separated audio files that we'll create (or load)
        self.out_files = [
            rf"{self.separate_audio_loc}\{self.fname}_{i}.{self.file_fmt}"
            for i in self.instrs
        ]
        # Logger object and empty list to hold messages (for saving)
        self.logger = kwargs.get("logger", None)
        self.logging_messages = []
        # Boolean kwargs that will force this item to be downloaded or separated regardless of local presence
        self.force_download = kwargs.get("force_redownload", False)
        self.force_separation = kwargs.get("force_reseparation", False)
        # Starting and ending timestamps, gathered from the corpus JSON
        self.start, self.end = self._return_timestamp("start"), self._return_timestamp(
            "end"
        )
        # Amount to multiply file duration by when calculating source separation timeout value
        self.timeout_multiplier = kwargs.get("timeout_multiplier", 5)
        # Log start of processing
        self._logger_wrapper(
            f'processing item {self.item["id"]}, '
            f'"{self.item["track_name"]}" from {self.item["recording_year"]} album {self.item["album_name"]}, '
            f'leader {self.item["musicians"][self.item["musicians"]["leader"]]} ...'
        )

    def _logger_wrapper(self, msg) -> None:
        """
        Simple wrapper that logs a given message and indexes it for later access
        """

        if self.logger is not None:
            self.logger.info(msg)
        self.logging_messages.append(
            f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]}: {msg}'
        )

    def _construct_filename(self, **kwargs) -> str:
        """
        Constructs the filename for this item using the values in the corpus JSON
        """

        def name_formatter(st: str = "album_name") -> str:
            """
            Applies formatting to a particular element from the JSON
            """
            # Get the number of words we desire for this item
            desired_words = kwargs.get(f"{st}_len", 5)
            # Get the item name itself, e.g. album name, track name
            name = self.item[st].split(" ")
            # Get the number of words we require
            name_length = len(name) if len(name) < desired_words else desired_words
            return re.sub("[\W_]+", "", "".join(i.lower() for i in name[:name_length]))

        # Get the name of the leader and format: lastnamefirstinitial, e.g. evansb
        leader = self.item["musicians"]["leader"]
        musician = self.item["musicians"][leader].lower().split(" ")
        musician = musician[1] + musician[0][0]
        # Get the required number of words of the album + track title, nicely formatted
        album = name_formatter("album_name")
        track = name_formatter("track_name")
        # Try to get the number of our track
        try:
            take = f'{self.item["track_take"]}'
        except KeyError:
            take = "na"
        # Get our album recording year
        year = self.item["recording_year"]
        # Get our unique item id
        id_ = self.item["id"]
        # Return our formatted filename
        return rf"{id_}-{musician}-{album}-{year}-{track}-{take}"

    def _get_valid_links(
        self, bad_pattern: str = '"playabilityStatus":{"status":"ERROR"'
    ) -> list:
        """
        Returns a list of valid YouTube links from the Corpus JSON
        """

        checker = lambda s: bad_pattern not in requests.get(s).text
        return [
            link
            for link in self.item["links"]["external"]
            if "youtube" in link and checker(link)
        ]

    def _return_timestamp(self, timestamp: str = "start", fmt: str = "%M:%S") -> int:
        """
        Returns a formatted timestamp from a JSON element
        """

        try:
            dt = datetime.strptime(self.item["timestamps"][timestamp], fmt)
            return int(
                timedelta(
                    hours=dt.hour, minutes=dt.minute, seconds=dt.second
                ).total_seconds()
            )
        # TODO: figure out the required exception type to go here
        except (ValueError, TypeError):
            return None

    def get_item(self) -> None:
        """
        Tries to find a corpus item locally, and downloads it from the internet if not present
        """

        # Define our list of checks for whether we need to rebuild the item
        checks = [
            # Is the item actually present locally?
            self._check_item_present_locally(self.in_file),
            # Are we forcing the corpus to rebuild?
            not self.force_download,
            # Have we changed the timestamps for this item since the last time we built it?
            float(self.end - self.start) == float(self._get_output_duration(self.in_file)),
        ]
        # If we pass all checks, then go ahead and get the item locally (skip downloading it)
        if all(checks):
            self._logger_wrapper(
                f"... skipping download, item found locally as {os.path.abspath(self.in_file)}"
            )
        # Otherwise, rebuild the item
        else:
            # We get the valid links here so we don't waste time checking links if an item is already present locally
            self.links = self._get_valid_links()
            self._get_item()

    @staticmethod
    def _check_item_present_locally(fname: str) -> bool:
        """
        Returns whether a given filepath is present locally or not
        """

        return os.path.isfile(os.path.abspath(fname))

    def _get_item(self) -> None:
        """
        Downloads from a YouTube link using FFmpeg and yt_dlp, between two timestamps
        """

        # Set our options in yt_dlp
        self.ydl_opts["outtmpl"] = self.in_file
        self.ydl_opts["download_ranges"] = download_range_func(
            None, [(self.start, self.end)]
        )
        # Log start of download process
        self._logger_wrapper(
            f"... found {len(self.links)} valid link(s) to download from"
        )
        # Iterate through all of our valid YouTube links
        for link in self.links:
            # Try and download from each link
            try:
                with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                    ydl.download(link)
            # If we get an error, continue on to the next link
            except DownloadError as err:
                self._logger_wrapper(
                    f"... error when downloading from {link} ({err}), trying next link"
                )
                continue
            # If we've downloaded successfully, break out of the loop
            else:
                self._logger_wrapper(f"... downloaded successfully from {link}")
                break
        # If, after iterating through all our links, we still haven't been able to save the file, then raise an error
        if not self._check_item_present_locally(self.in_file):
            raise DownloadError(
                f'Item {self.item["id"]} could not be downloaded, check input links are working'
            )

    def _separate_audio(self, cmd: list, good_pattern: str = "written succesfully") -> None:
        """
        Conducts the separation process by passing the given cmd through to subprocess.Popen and storing the output.
        The argument good_pattern should be a string contained within the successful output of the subprocess. If this
        string is not contained, it will be assumed that the process has failed, and logged accordingly.
        """

        # Get the timeout value: the duration of the item, times the multiplier
        timeout = int((self.end - self.start) * self.timeout_multiplier)
        # Open the subprocess. The additional arguments allow us to capture the output
        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        # TODO: if this fails, we need some way of killing the overall process? Or setting an error flag?
        try:
            # This will block execution until the above process has completed
            out, err = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            raise TimeoutError(
                f"... error when separating: process timed out after {timeout} seconds"
            )
        else:
            # Check to make sure the expected output is returned by subprocess
            if good_pattern not in out:
                self._logger_wrapper(f"... error when separating: {out}")
            else:
                self._logger_wrapper(f"... item separated successfully")

    def _cleanup_post_separation(self, exts: list = None) -> None:
        """
        Cleans up after source-separation by removing unnecessary files -- defaults to the vocal and other stems
        """

        if exts is None:
            exts = self.instrs
        for file in os.listdir(self.separate_audio_loc):
            if self.fname in file and not any(f"_{i}" for i in exts if i in file):
                os.remove(os.path.abspath(rf"{self.separate_audio_loc}\{file}"))

    def _get_spleeter_cmd(self) -> list:
        """
        Gets the required command for running spleeter using subprocess.Popen
        """

        return [
            "spleeter",
            "separate",  # Opens Spleeter in separation mode
            "-p",
            self.model,  # Defaults to the 5stems-16kHz model
            "-o",
            f"{os.path.abspath(self.separate_audio_loc)}",  # Specifies the correct output directory
            f"{os.path.abspath(self.in_file)}",  # Specifies the input filepath for this item
            "-c",
            f"{self.file_fmt}",  # Specifies the output codec, default to m4a
            "-f",
            "{filename}_{instrument}.{codec}",  # This sets the output filename format
        ]

    @staticmethod
    def _get_output_duration(fpath: str) -> float:
        """
        Opens a given audio file and returns its duration
        """

        try:
            with audioread.audio_open(fpath) as f:
                return f.duration
        except FileNotFoundError:
            return 0

    def separate_audio(self) -> None:
        """
        Tries to find source-separated audio files locally, and builds them if not present/invalid
        """

        # Define our list of checks for whether we need to conduct source separation again
        checks = [
            # Are all the source separated items present locally?
            all(
                self._check_item_present_locally(fname) for fname in self.out_files
            ),
            # Do all the source-separated items have the same duration as the original file?
            all(
                [len(set(self._get_output_duration(fp) for fp in [self.in_file, out])) == 1 for out in self.out_files]
            ),
            # Have we changed the timestamps for this item since the last time we built it?
            all(
                float(self.end - self.start) == float(self._get_output_duration(out)) for out in self.out_files
            ),
            # Are we forcing the corpus to rebuild itself?
            not self.force_separation,
        ]
        # If we pass all the checks, then we can skip rebuilding the source-separated tracks
        if all(checks):
            self._logger_wrapper(
                f"... skipping separation, item found locally in {os.path.abspath(self.separate_audio_loc)}"
            )
        # Otherwise, we need to build the source-separated items
        else:
            # Raise an error if we no longer have the input file, for whatever reason
            if not self._check_item_present_locally(self.in_file):
                raise FileNotFoundError(
                    f"Input file {self.in_file} not present, can't proceed to separation"
                )
            cmd = self._get_spleeter_cmd()
            self._logger_wrapper(f"... separating audio with model {self.model}")
            self._separate_audio(cmd)
            self._cleanup_post_separation()

    def finalize_output(self) -> None:
        """
        Finalizes the output by appending the output information to our item dictionary, ready for saving as a JSON
        """

        self.item["output"] = {}
        self.item["log"] = self.logging_messages
        for name, fpath in zip(
            [*sorted(self.instrs), "mix"], [*sorted(self.out_files), self.in_file]
        ):
            self.item["output"][name] = os.path.abspath(fpath)
        self._logger_wrapper("... finished processing item")


@click.command()
@click.option(
    "-i", "input_filepath", type=click.Path(exists=True), default=r"..\..\references"
)
@click.option(
    "-o", "output_filepath", type=click.Path(exists=True), default="..\..\data"
)
def main(input_filepath, output_filepath):
    """
    Runs clean processing scripts to turn raw clean from (../raw) into cleaned clean ready to be analyzed
    (saved in ../processed)
    """

    start = time()

    logger = logging.getLogger(__name__)
    logger.info("making final clean set from raw clean...")

    js = []
    # Open the corpus json file
    with open(input_filepath + "\corpus.json", "r+") as in_file:
        corpus = json.load(in_file)
        # Iterate through each entry in the corpus, with the index as well
        for index, item in enumerate(corpus, 1):
            # Initialise the ItemMaker instance for this item
            made = ItemMaker(
                item=item,
                album_name_len=5,
                track_name_len=10,
                logger=logger,
                index=index,
                output_filepath=output_filepath,
            )
            # Download the item, separate the audio, and finalize the output
            made.get_item()
            made.separate_audio()
            made.finalize_output()
            # Append our output to the list
            js.append(made.item)

    # Dump our finalized output to a new json and save in the output directory
    output_fname = output_filepath + "\processed\processing_results.json"
    with open(output_fname, "w") as out_file:
        json.dump(js, out_file)

    logger.info(
        f"dataset made in {round(time() - start)} secs, see output JSON at {os.path.abspath(output_fname)}"
    )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
