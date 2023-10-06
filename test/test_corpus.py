#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for corpus in references/corpus.json
"""

import json
import math
import os
import unittest
import urllib.error
import urllib.request
from datetime import datetime

import dotenv
import pandas as pd
import requests

from src import utils


class CorpusTest(unittest.TestCase):
    corpus = utils.CorpusMaker.from_excel(rf'{utils.get_project_root()}\references\corpus_bill_evans')
    dotenv.load_dotenv(rf"{utils.get_project_root()}\.env")
    df = pd.DataFrame(corpus)
    links = [link for track in corpus for link in track["links"]["external"]]

    def test_track_metadata(
        self, required_cols: tuple = ("track_name", "album_name", "recording_year")
    ):
        """
        Tests that required metadata (track name, album name, recording year) is present for each item in the corpus.
        Fails when metadata is not present for all items in the corpus.
        """
        self.assertEqual(
            self.df[list(required_cols)].isna().any().sum(),
            0,
            msg=f'Some tracks missing required metadata: {",".join(col for col in required_cols)}',
        )

    def test_all_links_unique(self):
        """
        Tests that all external links in the corpus are unique. Fails when a duplicate link is found.
        """

        df = pd.Series(self.links)
        dupes = df[df.duplicated(keep=False)].to_list()
        self.assertTrue(
            len(dupes) == 0,
            msg=f"Some links in corpus are duplicates {dupes}",
        )

    def test_all_youtube_links_active(
        self, bad_pattern: str = '"playabilityStatus":{"status":"ERROR"'
    ):
        """
        Tests that all external YouTube links given in the corpus are currently active. Fails when bad_pattern is found
        in a GET request from a given link.
        """

        for link in self.links:
            if "youtube" in link:
                self.assertTrue(
                    bad_pattern not in requests.get(link).text,
                    msg=f"YouTube link {link} is dead.",
                )

    def test_take_numbers_present(self):
        """
        Tests that, for albums with multiple takes of one track, these all have take numbers (`track_take`) values.
        Fails when a track with multiple takes on a single album is found not to have take numbers assigned
        """
        grps = ["album_name", "track_name"]
        takes = [
            take
            for idx, grp in self.df.groupby(grps)
            for take in grp["track_take"].to_list()
            if len(grp) > 1
        ]
        self.assertTrue(
            not any(math.isnan(take) for take in takes),
            msg="Some tracks with multiple takes do not have take numbers assigned.",
        )

    def test_timestamp_against_duration(self):
        """
        Tests the end timestamps given for each video against the actual video duration. Fails if a timestamp is longer
        than the actual duration of a video. Make sure YouTube API key is set as an environment variable (YOUTUBE_API)
        """

        def get_youtube_video_duration(li: str) -> datetime:
            """
            Gets the duration of a single YouTube video as a datetime object
            """

            # Get just the video id from our link
            video_id = li.split("/watch?v=")[-1]
            # Get the required URL, using our API key
            search = f"https://www.googleapis.com/youtube/v3/videos?id={video_id}&key={api_key}&part=contentDetails"
            # Read the link and grab the video duration from the json
            try:
                with urllib.request.urlopen(search) as url:
                    response = json.loads(url.read())["items"][0]["contentDetails"][
                        "duration"
                    ]
            except urllib.error.HTTPError:
                self.fail(f"Could not get timestamp for video {li}")
            # Return the ISO-formatted timestamp as a datetime object
            try:
                return datetime.strptime(response, "PT%MM%SS")
            except ValueError:
                return datetime.strptime(response, "PT%MM")

        # Get our YouTube api key from our environment variables
        api_key = os.getenv('YOUTUBE_API')
        self.assertIsNotNone(api_key, msg="YouTube API key not provided as environment variable")
        for track in self.corpus:
            # Get the links from the corpus
            yt_links = [
                yt_link
                for yt_link in track["links"]["external"]
                if "youtube" in yt_link
            ]
            # Get our output timestamp as a datetime object, for comparison
            out_ts = datetime.strptime(track["timestamps"]["end"], "%M:%S")
            # Iterate through each YouTube link we've given
            for yt_link in yt_links:
                # Get our the duration of our link and assert that our output timestamp is below the duration
                yt_ts = get_youtube_video_duration(yt_link)
                self.assertLess(
                    out_ts,
                    yt_ts,
                    msg=f"Output timestamp {out_ts} is greater than video duration {yt_ts} for video {yt_link}",
                )


if __name__ == "__main__":
    unittest.main()
