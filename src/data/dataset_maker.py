from datetime import datetime, timedelta
import json
import os
import re
import logging
import yt_dlp
import requests
from yt_dlp.utils import download_range_func, DownloadError
import subprocess


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


# noinspection PyBroadException
class ItemMaker:
    """
    Makes a single item in the corpus by reading the corresponding JSON entry, attempting to locate the item locally,
    and getting the audio from YouTube if not available (with the required start and stop timestamps).
    """

    file_fmt = 'm4a'
    ydl_opts = {
        'format': f'{file_fmt}/bestaudio[ext={file_fmt}]/best',
        'quiet': True,
        'extract_audio': True,
        'overwrites': True,
        'logger': _YtDlpFakeLogger
    }
    raw_output_loc = r'..\..\data\raw\audio'
    separate_output_loc = r'..\..\data\processed\audio'
    model = 'spleeter:5stems-16kHz'
    instrs = ['piano', 'bass', 'drums']

    def __init__(self, item, **kwargs):
        self.item = item
        self.links = []
        self.fname = fr'{self.raw_output_loc}\{self._construct_filename(**kwargs)}.{self.file_fmt}'
        self.fname_no_dir = self._construct_filename(**kwargs)
        self.ydl_opts['outtmpl'] = self.fname
        self.logger = kwargs.get('logger', None)
        self.force_download = kwargs.get('force_redownload', False)
        self.force_separation = kwargs.get('force_reseparation', False)
        self.start = self._return_timestamp('start')
        self.end = self._return_timestamp('end')
        self.ydl_opts['download_ranges'] = download_range_func(None, [(self.start, self.end)])

    def _construct_filename(
            self, **kwargs
    ) -> str:
        """
        Constructs the filename for this item using the values in the corpus JSON
        """

        def name_formatter(
                st: str = 'album_name'
        ) -> str:
            """
            Applies formatting to a particular element from the JSON
            """
            # Get the number of words we desire for this item
            desired_words = kwargs.get(f'{st}_len', 5)
            # Get the item name itself, e.g. album name, track name
            name = self.item[st].split(' ')
            # Get the number of words we require
            name_length = len(name) if len(name) < desired_words else desired_words
            return re.sub('[\W_]+', '', ''.join(i.lower() for i in name[:name_length]))

        # Get the name of the leader and format: lastnamefirstinitial, e.g. evansb
        leader = self.item['musicians']['leader']
        musician = self.item['musicians'][leader].lower().split(' ')
        musician = musician[1] + musician[0][0]
        # Get the required number of words of the album + track title, nicely formatted
        album = name_formatter('album_name')
        track = name_formatter('track_name')
        # Try to get the number of our track
        try:
            take = f'{self.item["track_take"]}'
        except KeyError:
            take = 'na'
        # Get our album recording year
        year = self.item['recording_year']
        # Get our unique item id
        id_ = self.item['id']
        # Return our formatted filename
        return fr'{id_}-{musician}-{album}-{year}-{track}-{take}'

    def _get_valid_links(
            self, bad_pattern: str = '"playabilityStatus":{"status":"ERROR"'
    ) -> list:
        """
        Returns a list of valid YouTube links from the Corpus JSON
        """

        checker = lambda s: bad_pattern not in requests.get(s).text
        return [link for link in self.item['links']['external'] if 'youtube' in link and checker(link)]

    def _return_timestamp(
            self, timestamp: str = 'start', fmt: str = '%M:%S'
    ) -> int:
        """
        Returns a formatted timestamp from a JSON element
        """

        try:
            dt = datetime.strptime(self.item['timestamps'][timestamp], fmt)
            return int(timedelta(hours=dt.hour, minutes=dt.minute, seconds=dt.second).total_seconds())
        except:
            return None

    def get_item(
            self
    ) -> None:
        """
        Tries to find a corpus item locally, and downloads it from the internet if not present
        """
        if self.logger is not None:
            self.logger.info(f'getting item {self.item["id"]} ...')

        # If the item we want is available locally, and we're not forcing a re-download, skip downloading it
        if self._check_item_present_locally(self.fname) and not self.force_download:
            if self.logger is not None:
                self.logger.info(f'... item {self.item["id"]} was found locally as {self.fname}, skipping download')
        # If we don't have the item locally (or we're forcing a re-download), then download it from YouTube
        else:
            # We get the valid links here so we don't waste time checking links if an item is already found locally
            self.links = self._get_valid_links()
            self._get_item()

    @staticmethod
    def _check_item_present_locally(
            fname
    ) -> bool:
        """
        Returns whether a given filepath is present locally or not
        """

        return os.path.isfile(os.path.abspath(fname))

    def _get_item(
            self
    ) -> None:
        """
        Downloads from a YouTube link using FFmpeg and yt_dlp, between two timestamps
        """

        # Iterate through all of our valid YouTube links
        for link in self.links:
            # Try and download from each link
            try:
                with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                    ydl.download(link)
            # If we get an error, continue on to the next link
            except DownloadError:
                if self.logger is not None:
                    self.logger.info(f'... error when downloading, skipping to next item')
                continue
            # If we've downloaded successfully, break out of the loop
            else:
                if self.logger is not None:
                    self.logger.info(f'... downloaded successfully, saved at {self.fname}')
                break

    def _separate_audio(
            self, cmd: list, good_pattern: str = 'written succesfully'
    ):
        """
        Conducts the separation process by passing the given cmd through to subprocess.Popen and storing the output.
        The argument good_pattern should be a string contained within the successful output of the subprocess. If this
        string is not contained, it will be assumed that the process has failed, and logged accordingly.
        """

        # Open the subprocess. The additional arguments allow us to capture the output
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        out, err = p.communicate()
        # This will block execution until the above process has completed
        p.wait()
        # Check to make sure the expected output is returned by subprocess
        if good_pattern not in out:
            if self.logger is not None:
                self.logger.warn(f'... error when separating item {self.item["id"]}: {out}')
        else:
            if self.logger is not None:
                self.logger.info(f'... item {self.item["id"]} separated successfully')

    def _cleanup_post_separation(
            self, exts: list = None
    ) -> None:
        """
        Cleans up after source-separation by removing unnecessary files -- defaults to the vocal and other stems
        """

        if exts is None:
            exts = self.instrs
        for file in os.listdir(self.separate_output_loc):
            if self.fname_no_dir in file and not any(f'_{i}' for i in exts if i in file):
                os.remove(os.path.abspath(rf'{self.separate_output_loc}\{file}'))

    def _get_spleeter_cmd(
            self
    ) -> list:
        """
        Gets the required command for running spleeter using subprocess.Popen
        """

        return [
            'spleeter', 'separate',    # Opens Spleeter in separation mode
            '-p', self.model,    # Defaults to the 5stems-16kHz model
            '-o', f'{os.path.abspath(self.separate_output_loc)}',    # Specifies the correct output directory
            f'{os.path.abspath(self.fname)}',    # Specifies the input filepath for this item
            '-c', f'{self.file_fmt}',    # Specifies the output codec, default to m4a
            '-f', '{filename}_{instrument}.{codec}'    # This sets the output filename format
        ]

    def separate_audio(
            self,
    ) -> None:
        """
        Separates item audio in Spleeter using default 5stems-16kHz model
        """

        if self.logger is not None:
            self.logger.info(f'separating audio for item {self.item["id"]} with model {self.model} ...')
        # This list includes al of the filenames that we want
        out_fnames = [rf'{self.separate_output_loc}\{self.fname_no_dir}_{i}.{self.file_fmt}' for i in self.instrs]
        # If we have all of the source-separated tracks already and we're not forcing separation, skip creating them
        # TODO: check if local source-separated files are the same length as the local mixed file, and force separation if not
        if all(self._check_item_present_locally(fname) for fname in out_fnames) and not self.force_separation:
            if self.logger is not None:
                self.logger.info(f'... item {self.item["id"]} already separated, skipping')
        # Else, create the required spleeter command, separate the audio using it, then clean up any left over files
        else:
            cmd = self._get_spleeter_cmd()
            self._separate_audio(cmd)
            self._cleanup_post_separation()


def make_dataset(
        logger: logging.Logger = None, fpath: str = r'..\..\references\corpus.json'
) -> None:
    """

    """

    if logger is not None:
        logger.info(f'Making dataset from corpus in {fpath} ...')
    with open(fpath, "r+") as js_file:
        corpus = json.load(js_file)
        for index, item in enumerate(corpus):
            item['id'] = str(index + 1).zfill(6)
            made = ItemMaker(item=item, album_name_len=5, track_name_len=10, logger=logger)
            made.get_item()
            made.separate_audio()
            item['links']['local'] = made.fname
        js_file.seek(0)
        json.dump(corpus, js_file)
        js_file.truncate()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    make_dataset(logger=logger)
