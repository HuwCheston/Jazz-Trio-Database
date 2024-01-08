# Defining your own `Extractor` classes
(define-extractors)=

As noted in the previous pages, `Extractor` classes should inherit from `BaseExtractor` defined in `.\src\detect\detect_utils.py`. They should follow the following logic:

1. Take in some combination of `my_onsets`, `my_beats`, `their_onsets`, and `their_beats` for a given track;
2. Apply processing to the input arrays in order to generate new array(s);
3. Apply the functions inside `summary_funcs` to the new array(s) to populate the `summary_dict` dictionary.

(summary-funcs)=
## More on `summary_funcs`

`summary_funcs` is a dictionary containing keys of type `str` and values of type (function). Keys should generally be the description or name of the function. Values can be any function that takes as input an array (of type `np.ndarray` or `pd.Series`) and returns a numeric value (of type `int` or `float`).

`update_summary_dict` takes in an iterable of `array_names` and an iterable of `arrays`. Each value in `arrays` should be an array resulting from processing the class inputs (e.g., `my_beats`, `their_beats`). Each value in `array_names` should match with an array in `array_names` and provide a description (`str`) of its context in plain text. Ensure that `len(array_names) == len(arrays)`. 

The function will then apply every function in `summary_funcs.values` to every array in `arrays` and join the corresponding strings in `array_names` and `summary_funcs.keys`. The results will be used to populate `summary_dict`.

## An example custom `Extractor`

As an example, let's create a new `Extractor` that takes in `my_beats` and populates `summary_dict` with both the mean and median inter-beat interval and the mean and median beat position:

```
import numpy as np
import pandas as pd
from src.detect.detect_utils import OnsetMaker
from src.features.features_utils import BaseExtractor

class ExtractorExample(BaseExtractor):
    def __init__(self, my_beats: pd.Series, **kwargs):
        super().__init__()
        iois = self.process_iois(my_beats)
        self.summary_funcs = {
            'mean': np.nanmean,
            'median': np.nanmedian
        }
        arrays = [iois, my_beats]
        array_names = ['interbeat_intervals', 'beat_positions']
        self.update_summary_dict(array_names, arrays)
        
      def process_iois(beats: pd.Series) -> pd.Series:
          return beats.astype(np.float64).diff()
          
      # The below code is reproduced from `BaseExtractor` as a reference
      # def update_summary_dict(self, array_names, arrays, *args, **kwargs) -> None:
      #   """Update our summary dictionary with values from this feature. Can be overridden!"""
      #   for name, df in zip(array_names, arrays):
      #       self.summary_dict.update({f'{name}_{func_k}': func_v(df) for func_k, func_v in self.summary_funcs.items()})
          

track = OnsetMaker(...)
my_beats = pd.DataFrame(track.summary_dict)['piano']
ext = ExtractorExample(my_beats)
ext.summary_dict

>>> {
>>>     'interbeat_intervals_mean': ...,
>>>     'interbeat_intervals_median': ...,
>>>     'beat_positions_mean': ...,
>>>     'beat_positions_median': ...,
>>> }

```

As you can see, every function in `self.summary_funcs.values` is applied to each array in `arrays` to create `ext.summary_dict.values`, and the `str`s in `self.summary_funcs.keys` are joined with those in `array_names` to create `ext.summary_dict.keys`.

:::{warning}
The above explanation refers to the default functionality of `update_summary_dict`. However, it's important to mention that several `Extractor` classes override this function. If in doubt, refer to the documentation for the particular `Extractor`. 
:::