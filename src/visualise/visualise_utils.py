#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility classes, functions, and variables used specifically in the visualisation process"""

import functools
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from src import utils

# Ignore annoying matplotlib INFO warnings created even though I'm doing nothing wrong
plt.set_loglevel('WARNING')

# Define constants
WIDTH = 18.8    # This is a full page width: half page plots will need to use 18.8 / 2
FONTSIZE = 18

ALPHA = 0.4
BLACK = '#000000'
WHITE = '#FFFFFF'

RED = '#FF0000'
GREEN = '#008000'
BLUE = '#0000FF'
YELLOW = '#FFFF00'
RGB = [RED, GREEN, BLUE]

LINEWIDTH = 2
LINESTYLE = '-'
TICKWIDTH = 3
MARKERSCALE = 1.6
MARKERS = ['o', 's', 'D']
HATCHES = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']

# Keyword arguments to use when applying a grid to a plot
GRID_KWS = dict(color=BLACK, alpha=ALPHA, lw=LINEWIDTH / 2, ls=LINESTYLE)

N_BOOT = 10000
N_BINS = 50


def plot_decorator(plotter: callable) -> callable:
    """Decorator applied to any plotting function: creates a folder, saves plot, closes the folder cleanly, and exits"""
    @functools.wraps(plotter)
    def wrapper(*args, **kwargs):
        # Define the filetypes we want to save the plot as
        filetypes = ['png', 'svg']
        # Create the output directory to store the plot
        output = kwargs.get('output_dir', None)
        # If we're accessing this decorator from a class, need to get the output by accessing the class attributes
        if output is None:
            output = args[0].output_dir  # Will be None anyway if no output_dir ever passed to class
        # Create the plot and return the figure
        fig, fname = plotter(*args, **kwargs)
        # If we've provided an output directory, create a folder and save the plot within it
        if output is not None:
            # Iterate through all filetypes and save the plot as each type
            for filetype in filetypes:
                try:
                    fig.savefig(f'{fname}.{filetype}', format=filetype, facecolor=WHITE)
                except FileNotFoundError:
                    create_output_folder(str(Path(output).parents[0]))
    return wrapper


def create_output_folder(out: str) -> str:
    """Create a folder to store the plots, with optional subdirectory. Out should be a full system path"""
    Path(out).mkdir(parents=True, exist_ok=True)
    return out


class BasePlot:
    """Base plotting class from which all others inherit"""
    mpl.rcParams.update(mpl.rcParamsDefault)

    output_dir = fr'{utils.get_project_root()}/reports/figures'
    # These variables should all be overridden at will in child classes
    df = None
    fig, ax = None, None
    g = None

    def __init__(self, **kwargs):
        # Set fontsize
        plt.rcParams.update({'font.size': FONTSIZE})
        self.figure_title = kwargs.get('figure_title', 'baseplot')

    @plot_decorator
    def create_plot(self) -> tuple:
        """Calls plot creation, axis formatting, and figure formatting classes, then saves in the decorator"""
        self._create_plot()
        self._format_ax()
        self._format_fig()
        fname = rf'{self.output_dir}/{self.figure_title}'
        return self.fig, fname

    def _create_plot(self) -> None:
        """This function should contain the code for plotting the graph"""
        return

    def _format_ax(self) -> None:
        """This function should contain the code for formatting the `self.ax` objects"""
        return

    def _format_fig(self) -> None:
        """This function should contain the code for formatting the `self.fig` objects"""
        return
