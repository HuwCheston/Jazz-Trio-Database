# Combining multiple `Extractor`s together 
(combine-extractors)=

## Combine features in one recording

On the previous page, we explored how we can extract a single feature from one musician's performance. Usually when we're building predictive models, however, we want to work with multiple features at the same time. `Extractor`s can be combined very easily by joining several of their `summary_dict`s together.

Let's assume we've already defined both the `bur_extract` and `async_extract` classes from the code on the previous page. To join these together, we can simply write:

```
big_dict = bur_extract.summary_dict | async_extract.summary_dict
big_dict

>>> {
>>>     'bur_mean': ..., 
>>>     'bur_median': ..., 
>>>     ...,
>>>     'piano_async_mean': ...,
>>>     'piano_async_median': ...,
>>> }   
```

## Combine features across several recordings

Now that we know how to combine several features together for a single recording, the logical next step is to consider how we can combine features across a number of tracks. In the following blocks of code, we'll assume that `res` is a list of `OnsetMaker` classes created by unserialising the output of `.\src\detect\detect_onsets.py`: see {ref}`loading data from source <load-from-src>`.

```
def extract_features(track: OnsetMaker) -> dict:
    """Extracts features for a single track and combines `summary_dict`s"""
    # Extract the necessary timing data from the track
    beats = pd.DataFrame(track.summary_dict)
    my_beats = beats['piano']
    my_onsets = track.ons['piano']
    their_beats = beats[['bass', 'drums']]
    # Extract burs for this track
    bur_extract_one = BeatUpbeatRatio(my_onsets, my_beats)
    # Extract asynchrony for this track
    async_extract_one = Asynchrony(my_beats, their_beats)
    # Combine features and return one dictionary
    return bur_extract_one.summary_dict | 
           async_extract_one.summary_dict


# Iterate through every individual track and build an array
all_features = [extract_features(track) for track in res]
all_features

>>> [
>>>     {
>>>         'bur_mean': ..., 
>>>         'bur_median': ..., 
>>>         ...,
>>>         'piano_async_mean': ...,
>>>         'piano_async_median': ...,
>>>     },
>>>     ...,
>>>     {
>>>         'bur_mean': ..., 
>>>         'bur_median': ..., 
>>>         ...,
>>>         'piano_async_mean': ...,
>>>         'piano_async_median': ...,
>>>     }
>>> ]
```

The output of `all_features` isn't especially useful to us in its current state, however: it'd be better to turn this into a DataFrame by calling `pd.DataFrame(all_features)`.

## Combining features with metadata

In the above example, every row is a track, and every column is a feature. But it can be hard to tell just *which* row corresponds to *which* track. To make this easier, we'd suggest combining your extracted features with additional metadata:

```
def features_with_metadata(track: OnsetMaker) -> dict:
    """Combines `extract_features` results with metadata"""   
    # Define the list of metadata keys we want to extract
    desired_keys = ['track_name', 'album_name', 'recording_year']
    # Create a new dictionary of desired metadata values
    metadata = {k: track.item[k] for k in desired_keys}
    # Combine metadata with the results from `extract_features`
    return metadata | extract_features(track)
    
    
# Iterate through every individual track and build an array
all_features_with_metadata = [extract_features(track) for track in res]

>>> [
>>>     {
>>>         'track_name': ..., 
>>>         'album_name': ..., 
>>>         ...,
>>>         'bur_mean': ..., 
>>>         'bur_median': ..., 
>>>         ...,
>>>         'piano_async_mean': ...,
>>>         'piano_async_median': ...,
>>>     },
>>>     ...,
>>>     {
>>>         'track_name': ..., 
>>>         'album_name': ..., 
>>>         ...,
>>>         'bur_mean': ..., 
>>>         'bur_median': ..., 
>>>         ...,
>>>         'piano_async_mean': ...,
>>>         'piano_async_median': ...,
>>>     },
>>> ]
```

## Combine features for multiple performers

In the above example, we were only interested in the performance of the piano player. But what if, for instance, we wanted to know about the bassist's level of swing as well, alongside their synchronization with the piano and drums? To do so, we need to iterate over the names of individual instruments as well as the data from each track. For instance:


```
all_instrs = ['piano', 'bass', 'drums']


def extract_features_for_instrument(track: OnsetMaker, my_instr: str) -> dict:
    """Extracts features for a single track and combines `summary_dict`s"""
    # Get the names of all other instruments in the ensemble
    their_instrs = [ins for ins in all_instrs if ins != my_instr]
    # Extract the necessary timing data from the track
    beats = pd.DataFrame(track.summary_dict)
    my_beats = beats[my_instr]
    my_onsets = track.ons[my_instr]
    their_beats = beats[their_instrs]
    # Extract burs for this track
    bur_extract_one = BeatUpbeatRatio(my_onsets, my_beats)
    # Extract asynchrony for this track
    async_extract_one = Asynchrony(my_beats, their_beats)
    # Combine features and return one dictionary
    return {'instr': my_instr} | 
           bur_extract_one.summary_dict | 
           async_extract_one.summary_dict


# Iterate through every individual track and build an array
all_features = []
for track in res:
    for instr in all_instrs:
        features = extract_features_for_instrument(track, instr)
        all_features.append(features)
all_features

>>> [
>>>     {
>>>         'instr': 'piano', 
>>>         'bur_mean': ..., 
>>>         'bur_median': ..., 
>>>         ...,
>>>         'piano_async_mean': ...,
>>>         'piano_async_median': ...,
>>>     },
>>>     {
>>>         'instr': 'bass', 
>>>         'bur_mean': ..., 
>>>         'bur_median': ..., 
>>>         ...,
>>>         'piano_async_mean': ...,
>>>         'piano_async_median': ...,
>>>     },
>>>     {
>>>         'instr': 'drums', 
>>>         ...,
>>>     },
>>>     ...,
>>> ]
```