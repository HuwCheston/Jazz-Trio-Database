# Visualizing a single performance
(visualise-single-performance)=

There are a few plotting classes defined in `.\src\visualise` that can be used to create plots and visualisations of data from individual performances. Most of these work directly on `OnsetMaker` classes.

## Example: plotting onset position

The following plot class can be used to visualise the onset position of each instrumentalist within a single performance.

```
import matplotlib.pyplot as plt
from src import utils
from src.visualise.asynchrony_plots import ScatterPlotAsynchrony

corpus = utils.load_corpus_from_files('path\to\corpus')
first_track = corpus[0]
plot = ScatterPlotAsynchrony(first_track)
plt.show()
```

![](visualise-performance-scatterplot.svg)

For more plotting classes, check the API reference.