# Run pipeline on a new corpus
(new-corpus)=

You can set up our data extraction and analysis pipeline to run on your own corpus of musical recordings.

## Setting up

First things first, we'd recommend that you clone our repository, so that you can see our example corpus files:

```
git clone https://github.com/HuwCheston/Cambridge-Jazz-Trio-Database
```

The corpus files will then be located inside `.\references`, with the file extension `.xlsx` (i.e., they should be opened using Microsoft Excel or a similar spreadsheet package).

## Corpus structure

Each corpus file is a multipage spreadsheet, where one page corresponds to one musical ensemble. The structure for each page, however, is always the same. Every row should correspond to an individual recording: each column should contain some metadata about the recording, most of which was compiled by scraping the [MusicBrainz database](https://musicbrainz.org/doc/MusicBrainz_Database). See below for a description of these columns:

:::{dropdown} Column description

| Field                     | Description                                                                                   |
|---------------------------|-----------------------------------------------------------------------------------------------|
| `recording_title`         | Title of the recording                                                                        |
| `release_title`           | Title of the earliest album released that contains the track                                  |
| `recording_date_estimate` | Estimated date of recording                                                                   |
| `piano`, `bass`, `drums`  | Names of the musicians playing the corresponding instruments                                  |
| `recording_position`      | Track number of the recording on `release_title`                                              |
| `recording_length`        | Duration of the recording, in format `%M:%S`                                                  |
| `channel_overrides`       | Key-value pairs relating to panning: `piano: l` means the piano is panned to the left channel |
| `recording_id_for_lbz`    | The estimated URL for the recording on [ListenBrainz](https://listenbrainz.org/)              |
| `link`                    | The `recording_id_for_lbz` column, as a clickable link                                        |
| `is_acceptable(Y/N)`      | Whether or not the track meets the inclusion criteria and should be processed (see below)     |

The following columns are only relevant when `is_acceptable(Y/N) == Y` (see below for descriptions of where this might be the case)

| Field                | Description                                                                                                                                       |
|----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|
| `start_timestamp`    | The beginning of the excerpt to process                                                                                                           |
| `end_timestamp`      | The end of the excerpt to process                                                                                                                 |
| `youtube_link`       | A stable link to the performance on YouTube, to be ripped with `yt_dlp`                                                                           |
| `channel_overrides`  | Whether to use mono left-right outputs for individual instruments: in format `instrument: channel, instrument: channel`, i.e. `piano: l, bass: r` |
| `time_signature`     | The number of quarter note beats per measure for the track                                                                                        |
| `first_downbeat`     | The first clear quarter note downbeat in the track                                                                                                |
| `notes`              | Written comments about the track                                                                                                                  |
| `rating_audio_X`     | Subjective rating (1–3, 3 = best) of source-separation quality for instrument `X`                                                                 |
| `rating_detection_X` | Subjective rating of onset detection quality for instrument `X`                                                                                   |
| `rating_comments`    | Comments given by the rater when making subjective judgements                                                                                     |
| `has_annotations`    | Whether the track has ground truth annotation files: see {ref}`Create new manual annotations <create-annotations>`                                |
:::

## Selecting tracks for inclusion

For each track (row), you'll then need to decide whether it should be processed by listening to it and setting `is_appropriate(Y/N)` to `Y`, for appropriate tracks. Appropriate tracks should also additional metadata: see the dropdown, above.

The decision for whether to include a track should be made according to the requirements of your project. When creating our corpus files, we followed the below criteria:

:::{dropdown} Inclusion criteria
***Instruments:***
- Piano, bass, drums (i.e., Piano trio) only
- Piano:	
  - No [electric/synthesiser/rhodes piano](https://www.youtube.com/watch?v=d1GQZLEnXFs)
- Bass:	
  - Must be acoustic/upright bass, [not electric (listen out for sound of fingers 'slapping' strings)](https://www.youtube.com/watch?v=N5uuC0x5JPk)
  - No [use of the bow](https://www.youtube.com/watch?v=r9D7zdJFLp0)
- Drums: 
  - No [auxiliary percussion (congas, shakers)](https://www.youtube.com/watch?v=4vFSxGhV29M) - whether played by the drummer or someone else 
  - No [mallets](https://www.youtube.com/watch?v=Xl-nblp_SQs)
  - No [brushes](https://www.youtube.com/watch?v=Lr5RiPvxzBQ)

***Tempo:***
- Medium to Up, i.e., approx 100 to 300 beats-per-minute

***Feel:***	
- Swung quavers only 
- No [ballads](https://www.youtube.com/watch?v=a2LFVWB) 
- No ['straight 8s'](https://www.youtube.com/watch?v=DiQagjy5INI): e.g. latin, afro-cuban, rock 
- No [rock-y/rnb-y style straight 8s stuff](https://www.youtube.com/watch?v=y7dpXzDR4Ug)
- No ['free' playing](https://www.youtube.com/watch?v=v9mV_1WSNTw)
- No material that [changes from non-swing to swing feels](https://www.youtube.com/watch?v=L34b0ut8Loc) (and back) during solos quickly

***Section:***
- Piano solo only: everybody needs to be improvising!
  - In recordings with [two pianos solos](https://www.youtube.com/watch?v=jYQupyUOYpo), choose the first
- Avoid material with [solo breaks interspersed](https://www.youtube.com/watch?v=PrEcT2Q51lw) throughout ensemble playing 
  - Material with break leading into solo is ok, as long as this then continues uninterrupted

***Quality:***
- Avoid [obviously bootlegged examples](https://www.youtube.com/watch?v=PFqhZ63PtVY) (e.g. recorded by attendees to concerts)
:::

## Reading the corpus in Python

We've provided a helper class which will convert a `.xlsx` corpus file into the correct format needed for our pipeline. This is located inside `.\src\utils.py` as `CorpusMaker`. Call the class with the `.from_excel()` constructor, passing in the filename of the corpus, as follows:

```
from src import utils

corpus = utils.CorpusMaker.from_excel(fname='...')
```

If the above command works without errors, you should be good to go. You can access the dataframe of tracks (equivalent to the original spreadsheet) by accessing the `tracks` attribute of `utils.CorpusMaker` (type: `pd.DataFrame`).

```{tip}
By default, tracks which did not pass the selection criteria (i.e., those where `is_acceptable(Y/N) != "Y"`) will not be processed by `utils.CorpusMaker`. To suppress this functionality and include all tracks in the output, pass `keep_all_tracks=True` when calling `utils.CorpusMaker.from_excel()`.
```

## Using the corpus with our pipeline

Once you've confirmed that the corpus can be accessed in Python, ensure that this is present inside the `.\references` directory. Then, when running either `.\src\clean\make_dataset.py` or `.\src\detect\detect_onsets.py`, pass the `-corpus` flag, followed by the filename, in order to direct the program to use your corpus file over the defaults. For more information, see {ref}`the respective sections on this page <build-database>` and the {ref}`User guide <user-guide>`.