#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility classes, functions, and variables used specifically in the visualisation process"""

import functools
from pathlib import Path

import matplotlib.pyplot as plt

from src import utils

# Ignore annoying matplotlib INFO warnings created even though I'm doing nothing wrong
plt.set_loglevel('WARNING')

# Define constants
WIDTH = 18.8
FONTSIZE = 18

ALPHA = 0.4
BLACK = '#000000'
WHITE = '#FFFFFF'
RED = '#FF0000'

N_BOOT = 10000

def plot_decorator(plotter: callable):
    """
    Decorator applied to any plotting function.
    Used to create a folder, save plot into this, then close it cleanly and exit.
    """

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
                fig.savefig(f'{fname}.{filetype}', format=filetype, facecolor=WHITE)
    return wrapper


def create_output_folder(out):
    """
    Create a folder to store the plots, with optional subdirectory. Out should be a full system path.
    """
    Path(out).mkdir(parents=True, exist_ok=True)
    return out


class BasePlot:
    """
    Base plotting class from which others inherit
    """
    output_dir = fr'{utils.get_project_root()}\reports\figures'
    df = None
    fig, ax = None, None
    g = None

    def __init__(self, **kwargs):
        # Set fontsize
        plt.rcParams.update({'font.size': FONTSIZE})
        self.figure_title = kwargs.get('figure_title', 'baseplot')

    @plot_decorator
    def create_plot(self):
        self._create_plot()
        self._format_ax()
        self._format_fig()
        fname = rf'{self.output_dir}\{self.figure_title}'
        return self.fig, fname

    def _create_plot(self):
        return

    def _format_ax(self):
        return

    def _format_fig(self):
        return
