# README

---

This file walks through the process of adding and modifying items in the corpus.

## Structure:

The corpus is saved as a `.json` file, with the default location being `<project-root>\references\corpus.json`. 

Each item in the corpus consists of a dictionary that looks something like:

```
{
    "track_name": "What Is This Thing Called Love",
    "track_take": 2,
    "album_name": "Portrait in Jazz",
    "recording_year": "1959",
    "musicians": {
      "pianist": "Bill Evans",
      "bassist": "Scott LeFaro",
      "drummer": "Paul Motian",
      "leader": "pianist"
    },
    "links": {
      "external": [
        "https://www.youtube.com/watch?v=8WMUZ6nixvQ"
      ]
    },
    "channel_overrides": {
      "bass": "l",
      "piano": "r"
    },
    "photos": {
      "musicians": {
        "pianist": null,
        "bassist": null,
        "drummer": null
      },
      "album_artwork": null
    },
    "timestamps": {
      "start": "01:12",
      "end": "02:48"
    },
    "id": "a2fe3d43cfadbeb2845aad8834b99e76",
    "fname": "evansb-portraitinjazz-1959-whatisthisthingcalledlove-2",
    "log": []
}
```

The meaning of these key-value pairs is explained below. Unless otherwise noted, each pair is mandatory and must be included when adding a new item to prevent errors from arising.

- `track_name`: the name of the track, as a string.
- `track_take`: **optional**. Include this if an album has multiple/alternate takes of a single piece.
  - For instance, set `"track_take": "2"` for the second take of the same piece. 
  - No need to include if a track only has one take.
- `album_name`: the name of the album the recording comes from, as a string.
- `recording_year`: the year of recording, as a string.
- `musicians`: a dictionary, containing the following key-value pairs:
  - `pianist`: the name of the pianist;
  - `bassist`: the name of the bassist;
  - `drummer`: the name of the drummer;
  - `leader`: the *instrument played* by the leader of the group. If the group has no designated leader, set this to `null`.
- `links`: a dictionary, containing the key-value pairs:
  - `external`: a list of YouTube links for this particular item. The script will prioritise links at the start of the list, before iterating to lower-priority links at the end.
- `channel_overrides`: an **optional** dictionary containing the names of instruments and their location in the stereo spectrum. 
  - This will direct the source-separation models only to be applied to one channel at a time, which can help improve separation in tracks with very wide stereo fields.
  - e.g. `"bass": "l"` will use the left audio channel only when separating the bass performance.
- `photos`: an **optional** dictionary, currently not implemented
- `timestamps`: a dictionary, containing the start and stop points for audio to be cut at. 
  - Format is `%M:%S`, or `minutes:seconds`

Several key-value pairs are also created automatically when an item is downloaded, and should not be added by the user:
- `id`: a unique hash assigned to each item after it is downloaded and separated successfully.
- `fname`: the unique filename assigned to the item, in the form `leadersurnameinitial-albumname-recordingyear-trackname-takename`.
- `log`: any logging messages raised when processing the item are stored here.

## Adding a new item to the corpus:

To add a new item, append a new dictionary to the bottom of `corpus.json`. You can use the following code snippet as a template:

```
{
    "track_name": "",
    "album_name": "",
    "recording_year": "",
    "musicians": {
      "pianist": "",
      "bassist": "",
      "drummer": "",
      "leader": ""
    },
    "links": {
      "external": [
        "https://www.youtube.com/watch?v=__________", 
        "https://www.youtube.com/watch?v=__________"
      ]
    },
    "timestamps": {
      "start": "__:__",
      "end": "__:__"
    },
}
```

## Running the tests:

After you've added items into the corpus, it's advisable to run the test suite in order to check for any common errors in the corpus construction (e.g. fields duplicated across multiple items, missing fields, invalid links). This will help prevent errors when running the full scripts.

To run the test suite, you'll need to create a YouTube API key and store it as the environment variable `YOUTUBE_API`. Doing so allows us to scrape information from the included YouTube links in the corpus without having to first download them locally.

> A tutorial on how to generate a YouTube API key is out-of-scope for this document. However, [the official documentation](https://developers.google.com/youtube/v3/getting-started) has good advice. 
> 
> On Windows, you can set your API key as an environment variable by running `set YOUTUBE_API=<api_key>`. Alternatively, on Linux, you can simply append `YOUTUBE_API=<api_key>` in front of the call to `test_corpus.py` given below.

Now, you can test that the corpus is working properly by running (from the project root):
```
python test\test_corpus.py
```
