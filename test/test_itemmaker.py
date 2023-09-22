#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for ItemMaker class in src/clean/make_dataset.py
"""

import json
import os
import unittest
import warnings
from math import isclose

from yt_dlp.utils import DownloadError

import src.clean.clean_utils
import src.global_utils
from src.clean.clean_utils import ItemMaker


class ItemMakerTest(unittest.TestCase):
    # A simple test corpus item with most fields filled out
    test_item = """
        {
            "track_name": "Test Track",
            "track_take": 5,
            "album_name": "Testy McTestFace Rides Again!",
            "recording_year": "2022",
            "musicians": {
                "pianist": "Testy McTestFace",
                "bassist": "Joe LeTesty",
                "drummer": "Test Montgomery",
                "leader": "pianist"
            },
            "timestamps": {
                "start": "02:20",
                "end": "03:00"
            }
        }
        """

    def _prepare_itemmaker_to_get_test_file(self) -> ItemMaker:
        """
        Does some prep work on self.test_item to get it in a position where it could download a file
        """

        # Load in the string
        item = json.loads(self.test_item)
        # Set the links attribute to a known working link
        item["links"] = {}
        item["links"]["external"] = ["https://www.youtube.com/watch?v=qHK09uMsXt4"]
        # Create the item maker instance
        im = ItemMaker(item, index=1, output_filepath=None)
        # Set the duration so we'll download 10 seconds of audio
        im.start = 0
        im.end = 10
        # Set the input file to save in the test folder (will be deleted after tests run)
        im.in_file = "test.m4a"
        return im

    def test_filename_construction(self):
        """
        Tests constructing a filename from input JSON using default arguments
        """

        im = ItemMaker(item=json.loads(self.test_item), index=10, output_filepath=None)
        expected = "mctestfacet-testymctestfaceridesagain-2022-testtrack-5"
        self.assertEqual(im._construct_filename(), expected)

    def test_filename_construction_custom_length_names(self):
        """
        Tests constructing a filename from input JSON using custom arguments
        """

        item = json.loads(self.test_item)
        # Modify a few parameters in the dictionary
        item[
            "track_name"
        ] = "This is a long track name which will be shortened to one word when creating the filename"
        item["album_name"] = "A long album name with !*769! numbers"
        im = ItemMaker(item=item, index=5, output_filepath=None)
        expected = "mctestfacet-alongalbumnamewith769numbers-2022-this-5"
        self.assertEqual(
            im._construct_filename(
                track_name_len=1,
                album_name_len=7,
            ),
            expected,
        )

    def test_invalid_youtube_links(self):
        """
        Tests the validation of working and non-working YouTube links
        """

        # Load the JSON and provide it with three links, two of which are broken
        item = json.loads(self.test_item)
        item["links"] = {}
        item["links"]["external"] = [
            "https://www.youtube.com/watch?v=EyucFAkWUic",  # Working
            "https://www.youtube.com/watch?v=EyucFefwghc",  # Broken
            "https://www.youtube.com/watch?v=EijojoSosSc",  # Broken
        ]
        im = ItemMaker(item=item, index=1, output_filepath=None)
        # We're only expecting one link to work, so the length of our valid links list will be 1
        expected = 1
        self.assertEqual(len(im._get_valid_links()), expected)

    def test_timestamp_return(self):
        """
        Tests the returning of correct timestamps from input JSON
        """

        im = ItemMaker(item=json.loads(self.test_item), index=1, output_filepath=None)
        expected = 40
        self.assertEqual(im.end - im.start, expected)

    def test_invalid_timestamp_return(self):
        """
        Tests correct return when an invalid timestamp is passed in an input JSON
        """

        item = json.loads(self.test_item)
        item["timestamps"]["end"] = "asdf"
        im = ItemMaker(item=item, index=1, output_filepath=None)
        self.assertEqual(im._return_audio_timestamp(timestamp="end"), None)

    def test_checking_for_local_file(self):
        """
        Tests the process for checking whether a file is accessible locally
        """

        # Check that this file is present (should always work)
        self.assertTrue(src.utils.global_utils.check_item_present_locally(fname="test_itemmaker.py"))

    def test_downloading_from_youtube(self):
        """
        Test that downloading from YouTube returns valid audio
        """
        im = self._prepare_itemmaker_to_get_test_file()
        # Try and get the item from YouTube
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            im.get_item()
        # Should return True after a successful download
        self.assertTrue(src.utils.global_utils.check_item_present_locally(im.in_file))
        os.remove(im.in_file)

    def test_error_raised_when_downloading_from_invalid_videos(self):
        """
        Test that the correct error is raised when downloading from an invalid YouTube link
        """

        im = ItemMaker(item=json.loads(self.test_item), index=1, output_filepath=None)
        # This is just a random gibberish link that should always fail
        im.links = ["https://www.youtube.com/watch?v=EyucFefwghc"]
        self.assertRaises(DownloadError, im._download_audio_excerpt_from_youtube)

    def test_output_duration_checking(self):
        """
        Test that the functionality for checking the correct duration of an output file works
        """

        im = self._prepare_itemmaker_to_get_test_file()
        # Get the test file
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            im.get_item()
        # Should be approximately 10 seconds in duration
        self.assertTrue(isclose(im._get_audio_duration(im.in_file), 10, abs_tol=0.01))
        os.remove(im.in_file)

    def test_source_separation_timeout(self):
        """
        Test that the functionality to timeout source separation based on item duration works
        """

        im = self._prepare_itemmaker_to_get_test_file()
        # This will always fail as it makes the timeout duration 10 seconds
        im.timeout_multiplier_spleeter = 0.000001
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            im.get_item()
        self.assertRaises(TimeoutError, im.separate_audio)
        os.remove(im.in_file)


if __name__ == "__main__":
    unittest.main()
