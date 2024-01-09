# General plotting workflow

```{warning}
Make sure you have access to the database before following the instructions in this tutorial, either by {ref}`downloading it <download-database>` or {ref}`building it from source <build-database>`.

You'll also need to be able to generate a list of `OnsetMaker` classes from the database: follow the instructions in {ref}`Loading the database in Python <loading-database-python>`
```

All plotting code follows the same workflow.

## Class inheritance

Plots are classes defined in `.\src\visualise` that all inherit from `BasePlot`, defined in `.\src\visualise\visualise_utils.py`

## Importing plots

The plot class should either be imported from its parent file by name, or via the `import *` syntax (to import all classes in the parent file): e.g.,

```
# Just import one plot
from src.visualise.asynchrony_plots import ScatterPlotAsynchrony
# Import every plot defined in __all__ of the parent file
from src.visualise.asynchrony_plots import *
```

## Constructing a plot

Plots should be constructed by first importing the class, passing in the required inputs, and then calling the `.create_plot()` method on the object. This will coerce the inputs into the correct format, create the plot and format it, then save inside the specified directory (defaults to `.\reports\figures`): e.g.,

```
from src.visualise.asynchrony_plots import ScatterPlotAsynchrony
ScatterPlotAsynchrony(onset_maker=om).create_plot()
```

To view the plot after creating it, use the standard `plt.show()` syntax as with any other plot, e.g.:

```
import matplotlib.pyplot as plt
from src.visualise.asynchrony_plots import ScatterPlotAsynchrony

ScatterPlotAsynchrony(onset_maker=om).create_plot()
plt.show()
```

## Plotting class methods

In general, plotting classes require four methods: `_format_df`, `_create_plot`, `_format_ax`, and `_format_fig`. These are inherited from `BasePlot` and are designed to be overridden in each plot class.

`_format_df` is typically called when the class is initialized (i.e., before calling `.create_plot()`) and coerces the inputs into the correct format, usually via `pandas` code. 

Then, `_create_plot`, `_format_ax`, and `_format_fig` are all called by `.create_plot()`. `_create_plot` (note the leading underscore) creates the graph in `matplotlib.pyplot` code. `_format_ax` formats all `matplotlib.pyplot.Axes` parameters, while `_format_ax` formats `matplotlib.pyplot.Figure` parameters. Note that each of these methods may themselves invoke additional methods and functions that are defined in each plotting class: however, only these four methods are inherited from `BasePlot`.

As suggested by the leading underscore, these methods are protected, and should not be called by themselves: the only method in a plot class called by the user should be `.create_plot()`

## `visualise_utils.py`

Inside `.\src\visualise\visualise_utils.py`, you'll find the constant variables that are referred to in many plotting classes. These are used, for instance, to ensure that the width and height of all plots are kept constant, or that the width of lines is consistent. 

Typically, this module should be imported using the syntax `import src.visualise.visualise_utils as vutils`. Then, the constants can be accessed using the `vutils` alias, e.g. `vutils.WIDTH`, `vutils.ALPHA`, etc.
