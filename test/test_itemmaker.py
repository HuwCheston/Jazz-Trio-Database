import unittest
import json

from yt_dlp import DownloadError

from src.data.make_dataset import ItemMaker


class ItemMakerTest(unittest.TestCase):

    def test_filename_construction(self):
        """
        Tests constructing a filename from input JSON using default arguments
        """

        item = '''
                {
                    "track_name": "Test Track",
                    "track_take": 5,
                    "album_name": "Testy McTestFace Rides Again!",
                    "recording_year": "2022",
                    "musicians": {
                        "pianist": "Testy McTestFace",
                        "bassist": "Joe LeTesty",
                        "drummer": "Test Me",
                        "leader": "pianist"
                    }
                }
                '''
        im = ItemMaker(item=json.loads(item), index=10, output_filepath=None)
        expected = "000010-mctestfacet-testymctestfaceridesagain-2022-testtrack-5"
        self.assertEqual(im._construct_filename(), expected)

    def test_filename_construction_custom_length_names(self):
        """
        Tests constructing a filename from input JSON using custom arguments
        """

        item = '''
        {
            "track_name": "This is a long track name which will be shortened to one word when creating the filename",
            "album_name": "A long album name with !*769! numbers",
            "recording_year": "3000",
            "musicians": {
                "pianist": "Testy McTest",
                "bassist": "Testy McTest2",
                "drummer": "Testy McTest3",
                "leader": "pianist"
                }
        }
        '''
        im = ItemMaker(item=json.loads(item), index=5, output_filepath=None)
        expected = "000005-mctestt-alongalbumnamewith769numbers-3000-this-na"
        self.assertEqual(im._construct_filename(track_name_len=1, album_name_len=7,), expected)

    def test_invalid_youtube_links(self):
        """
        Tests the validation of working and non-working YouTube links
        """

        item = '''
        {
            "track_name": "Test Track",
            "track_take": 5,
            "album_name": "Testy McTestFace Rides Again!",
            "recording_year": "2022",
            "musicians": {
                "pianist": "Testy McTestFace",
                "bassist": "Joe LeTesty",
                "drummer": "Test Me",
                "leader": "pianist"
            },
            "links": {
                "external": [
                    "https://www.youtube.com/watch?v=EyucFAkWUic",
                    "https://www.youtube.com/watch?v=EyucFefwghc",
                    "https://www.youtube.com/watch?v=EijojoSosSc"
                ]
            }
        }
        '''
        im = ItemMaker(item=json.loads(item), index=1, output_filepath=None)
        expected = 1
        self.assertEqual(len(im._get_valid_links()), expected)

    def test_timestamp_return(self):
        """
        Tests the returning of correct timestamps from input JSON
        """

        item = '''
        {
            "track_name": "Test Track",
            "track_take": 5,
            "album_name": "Testy McTestFace Rides Again!",
            "recording_year": "2022",
            "musicians": {
                "pianist": "Testy McTestFace",
                "bassist": "Joe LeTesty",
                "drummer": "Test Me",
                "leader": "pianist"
            },
            "timestamps": {
                "start": "02:20",
                "end": "03:00"
            }
        }
        '''
        im = ItemMaker(item=json.loads(item), index=1, output_filepath=None)
        expected = 40
        self.assertEqual(im.end - im.start, expected)

    def test_invalid_timestamp_return(self):
        pass

    def test_checking_for_local_file(self):
        """
        Tests the process for checking whether a file is accessible locally
        """

        item = '''
                {
                    "track_name": "Test Track",
                    "track_take": 5,
                    "album_name": "Testy McTestFace Rides Again!",
                    "recording_year": "2022",
                    "musicians": {
                        "pianist": "Testy McTestFace",
                        "bassist": "Joe LeTesty",
                        "drummer": "Test Me",
                        "leader": "pianist"
                    }
                }
                '''
        im = ItemMaker(item=json.loads(item), index=1, output_filepath=None)
        self.assertTrue(im._check_item_present_locally(fname=r'..\README.md'))

    def test_downloading_from_youtube(self):
        pass

    def test_error_raised_when_downloading_from_invalid_videos(self):
        item = '''
        {
            "track_name": "Test Track",
            "track_take": 5,
            "album_name": "Testy McTestFace Rides Again!",
            "recording_year": "2022",
            "musicians": {
                "pianist": "Testy McTestFace",
                "bassist": "Joe LeTesty",
                "drummer": "Test Me",
                "leader": "pianist"
            },
            "timestamps": {
                "start": "02:20",
                "end": "03:00"
            }
        }
        '''
        im = ItemMaker(item=json.loads(item), index=1, output_filepath=None)
        # This is just a random gibberish link that should always fail
        im.links = ["https://www.youtube.com/watch?v=EyucFefwghc"]
        self.assertRaises(DownloadError, im._get_item)

    def test_output_duration_checking(self):
        pass

    def test_source_separation_timeout(self):
        pass


if __name__ == '__main__':
    unittest.main()
