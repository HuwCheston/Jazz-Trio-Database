import unittest
import json
import os
import warnings

from yt_dlp.utils import DownloadError

from src.data.make_dataset import ItemMaker


class ItemMakerTest(unittest.TestCase):
    test_item = '''
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
        '''

    def _prepare_itemmaker_to_get_test_file(self) -> ItemMaker:
        item = json.loads(self.test_item)
        item['links'] = {}
        item['links']['external'] = ['https://www.youtube.com/watch?v=qHK09uMsXt4']
        im = ItemMaker(item, index=1, output_filepath=None)
        im.start = 0
        im.end = 10
        im.in_file = 'test.m4a'
        return im

    def test_filename_construction(self):
        """
        Tests constructing a filename from input JSON using default arguments
        """

        im = ItemMaker(item=json.loads(self.test_item), index=10, output_filepath=None)
        expected = "000010-mctestfacet-testymctestfaceridesagain-2022-testtrack-5"
        self.assertEqual(im._construct_filename(), expected)

    def test_filename_construction_custom_length_names(self):
        """
        Tests constructing a filename from input JSON using custom arguments
        """

        item = json.loads(self.test_item)
        item['track_name'] = "This is a long track name which will be shortened to one word when creating the filename"
        item['album_name'] = "A long album name with !*769! numbers"
        item['recording_year'] = "3000"
        im = ItemMaker(item=item, index=5, output_filepath=None)
        expected = "000005-mctestfacet-alongalbumnamewith769numbers-3000-this-5"
        self.assertEqual(im._construct_filename(track_name_len=1, album_name_len=7,), expected)

    def test_invalid_youtube_links(self):
        """
        Tests the validation of working and non-working YouTube links
        """

        item = json.loads(self.test_item)
        item['links'] = {}
        item['links']['external'] = [
                    "https://www.youtube.com/watch?v=EyucFAkWUic",
                    "https://www.youtube.com/watch?v=EyucFefwghc",
                    "https://www.youtube.com/watch?v=EijojoSosSc"
                ]

        im = ItemMaker(item=item, index=1, output_filepath=None)
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
        item = json.loads(self.test_item)
        item['timestamps']['end'] = 'asdf'
        im = ItemMaker(item=item, index=1, output_filepath=None)
        self.assertEqual(im._return_timestamp(timestamp='end'), None)

    def test_checking_for_local_file(self):
        """
        Tests the process for checking whether a file is accessible locally
        """
        im = ItemMaker(item=json.loads(self.test_item), index=1, output_filepath=None)
        self.assertTrue(im._check_item_present_locally(fname=r'..\README.md'))

    def test_downloading_from_youtube(self):
        im = self._prepare_itemmaker_to_get_test_file()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            im.get_item()
        self.assertTrue(im._check_item_present_locally(im.in_file))
        os.remove(im.in_file)

    def test_error_raised_when_downloading_from_invalid_videos(self):
        im = ItemMaker(item=json.loads(self.test_item), index=1, output_filepath=None)
        # This is just a random gibberish link that should always fail
        im.links = ["https://www.youtube.com/watch?v=EyucFefwghc"]
        self.assertRaises(DownloadError, im._get_item)

    def test_output_duration_checking(self):
        im = self._prepare_itemmaker_to_get_test_file()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            im.get_item()
        self.assertEqual(im._get_output_duration(im.in_file), 10)
        os.remove(im.in_file)

    def test_source_separation_timeout(self):
        im = self._prepare_itemmaker_to_get_test_file()
        im.timeout_multiplier = 0.0001
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            im.get_item()
        self.assertRaises(TimeoutError, im.separate_audio)
        os.remove(im.in_file)


if __name__ == '__main__':
    unittest.main()
