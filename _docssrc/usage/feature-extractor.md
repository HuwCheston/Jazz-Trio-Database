# Working with `Extractor` classes

```{warning}
Make sure you have access to the database before following the instructions in this tutorial, either by {ref}`downloading it <download-database>` or {ref}`building it from source <build-database>`.
```

The primary method for analyzing the tracks in the database is by using the different `Extractor` classes defined in `.\src\features\features_utils.py`. These classes all follow the same basic principles:

- They inherit from `BaseExtractor`, defined in `.\src\features\features_utils.py`
- They take in arrays of onsets and/or beats and perform analysis on them.
  - An `Extractor` class may take in data related to one performer (referred to in the first-person, as `my_beats`, `my_onsets`, etc.), or from one performer and several other musicians in their ensemble (`my_onsets`, `their_onsets`, for example.)
- They define a `summary_dict` attribute, which contains key-value pairs of summary statistics. 
  - This means that multiple `summary_dicts` can be combined across `Extractor` class instances to create a single dataframe that covers numerous features.
- They define a `summary_funcs` attribute, which contains `callable`s that are applied to an array (or arrays) to populate `summary_dict`.
  - This allows you to very quickly obtain numerous statistics for multiple transformations of a single feature.
  - See {ref}`More on summary_funcs <summary-funcs>` below.

(example-swing)=
## Example: `Beat-Upbeat Ratio`

We're going to start by demonstrating how the `BeatUpbeatRatio` `Extractor` class can be used to extract information relating to a performer's level of "swing". 

If you check the documentation for this class, you'll see it requires two arrays: `my_onsets`, and `my_beats`. This nomenclature is kept consistent throughout all classes: `my_onsets` refers to every onset that a performer has played, and `my_beats` refers to those onsets that marked the quarter note pulse.

Assuming that we want to extract swing for the pianist in a single track, we can apply the `BeatUpbeatRatio` `Extractor` as below:

```
import pandas as pd
from src.detect.detect_utils import OnsetMaker
from src.features.features_utils import BaseExtractor, BeatUpbeatRatio

track = OnsetMaker(...)
my_beats = pd.DataFrame(track.summary_dict)['piano']
my_onsets = track.ons['piano']
bur_extract = BeatUpbeatRatio(my_beats, my_onsets)
```

:::{tip}
Here, we defined `track` as an `OnsetMaker` class instance. The same result could have been obtained by loading the corresponding onset and beat data directly from a pre-compiled build of the database, however.
:::

### `BeatUpbeatRatio` class attributes

We can then access the individual beat-upbeat ratios for the track under the `bur_extract.bur` attribute , or the log transform of these ratios as `bur_extract.bur_log` (both of type: `pd.DataFrame`):

```
bur_extract.bur.head(3)

>>>                              beat      burs
>>> 0   1970-01-01 00:00:00.177052154       NaN
>>> 1   1970-01-01 00:00:00.565986394       NaN
>>> 2   1970-01-01 00:00:00.943310657  1.755102
```

See the documentation for descriptions of the attributes available to each `Extractor`.

### `summary_dict`

The most important attribute for any `Extractor`, however, is the `summary_dict` attribute. In the case of `BeatUpbeatRatio`, this contains the results of applying all the functions defined in `BaseExtractor.summary_funcs` to `BeatUpbeatRatio.bur` and `BeatUpbeatRatio.bur_log`.

```
bur_extract.summary_dict

>>> {
>>>     'bur_mean': ..., 
>>>     'bur_median': ..., 
>>>     ...,
>>> }
```

:::{tip}
By default, any `Extractor` that inherits from `BaseExtractor` will override the default `__repr__` function to display the `summary_dict` whenever the object itself is printed. In practice, this means that the results in the above block of code could simply have been obtained by calling `bur_extract`.
:::

## Example: `EventDensity`

As well as onsets and beats, some `Extractors` need to know a bit more about the metrical structure of a performance. Usually this involves passing some information about where the `downbeats` in a performance occur, i.e., the timestamps corresponding to the first beat of each bar.

One example is the `EventDensity` `Extractor`. This extracts various features relating to the density of an average passage of music, for instance the average number of onsets in a measure. To use this `Extractor`, we can do:

```
import pandas as pd
from src.detect.detect_utils import OnsetMaker
from src.features.features_utils import BaseExtractor, EventDensity

track = OnsetMaker(...)
my_onsets = track.ons['piano']
downbeats = track.ons['downbeats_manual']
density_extract = EventDensity(my_onsets, downbeats)
density_extract.summary_dict

>>> {
>>>     'ed_per_bar_mean': ...,
>>>     'ed_per_bar_median': ...,
>>> }
```

As with `BeatUpbeatRatio`, we can access the individual density scores for every measure with the `EventDensity.per_bar` attribute (type: `pd.DataFrame`).

### The optional `order` argument 

By default, the `EventDensity` `Extractor` calculates the density of every bar, and then applies the every function in `summary_funcs` to the resulting array. To change the window over which density is calculated, we can provide the `order` keyword argument. In the case of `EventDensity`, this parameter refers to the number of measures over which to conduct the density calculation: so, we could set `order=4` to evaluate density over a four measure period. 

:::{tip}
Check the documentation to see what the `order` parameter does in other `Extractor` classes.
:::

## Example: `Asynchrony`

In the first example, the `BeatUpbeatRatio` class only required us to provide arrays relating to one performer as inputs. Other `Extractor` classes require inputs from multiple performers, however. One example is the `Asynchrony` class, which needs the beat positions of one performer (`my_beats`, as above) and the beat positions of any number of other performers (`their_beats`):

```
import pandas as pd
from src.detect.detect_utils import OnsetMaker
from src.features.features_utils import Asynchrony

track = OnsetMaker(...)
beats = pd.DataFrame(track.summary_dict)
my_beats = beats['piano']
their_beats = beats[['bass', 'drums']]
async_extract = Asynchrony(my_beats, their_beats)

>>> {
>>>     'bur_mean': ..., 
>>>     'bur_median': ..., 
>>>     ...,
>>>     'piano_async_mean': ...,
>>>     'piano_async_median': ...,
>>> }   
```
:::{note}
You'll notice that there are many more key-value pairs in the `summary_dict` for `async_extract` versus that from `bur_extract`. This is because `Asynchrony` applies the functions contained in `summary_funcs` to every pairwise combination of performers: i.e., `piano -> bass` and `piano -> drums`.
:::
