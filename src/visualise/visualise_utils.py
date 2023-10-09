#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility classes, functions, and variables used specifically in the visualisation process"""

import functools
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
RGB = [RED, GREEN, BLUE]

LINEWIDTH = 2
LINESTYLE = '-'
TICKWIDTH = 3
MARKERSCALE = 1.6

N_BOOT = 10000
N_BINS = 50


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
                try:
                    fig.savefig(f'{fname}.{filetype}', format=filetype, facecolor=WHITE)
                except FileNotFoundError:
                    create_output_folder(str(Path(output).parents[0]))
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


class BarPlotCouplingCoefficients(BasePlot):
    nobs_cutoff = 30
    # TODO: fix this so that it'll work with the Bill Evans corpus, not just the chronology corpus

    def __init__(self, data, **kwargs):
        self.corpus_title = kwargs.get('corpus_title', 'corpus')
        super().__init__(figure_title=fr'phase_correction_plots\barplot_couplingcoefficients_{self.corpus_title}',
                         **kwargs)
        self.df = self._format_df(data)
        self.fig, self.ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True, figsize=(WIDTH, 8))
        self.hand, self.lab = None, None
        if self.df['pianist'].nunique() < len(self.ax.flatten()):
            for band_num in range(len(self.ax.flatten()) - self.df['pianist'].nunique()):
                self.ax.flatten()[len(self.ax.flatten()) - band_num].axis('off')

    @staticmethod
    def _format_df(data):
        return data.melt(id_vars=['instrument', 'performer', 'nobs', 'pianist'],
                         value_vars=['coupling_bass', 'coupling_piano', 'coupling_drums']).reset_index(drop=True)

    def _create_plot(self):
        for ax, (idx, grp) in zip(self.ax.flatten(), self.df.groupby('pianist')):
            grp = grp[grp['nobs'] > self.nobs_cutoff]
            g = sns.barplot(
                data=grp, x='variable', y='value', hue='instrument', ax=ax,
                errorbar=('ci', 95), estimator=np.nanmean, errwidth=2,
                errcolor=BLACK, edgecolor=BLACK, lw=2,
                n_boot=10, seed=1, capsize=0.1, width=0.8
            )
            self.hand, self.lab = g.get_legend_handles_labels()
            g.get_legend().remove()
            g.set(title=idx)

    def _format_ax(self):
        for ax in self.ax.flatten():
            ax.set(ylabel='', xlabel='', xticklabels=['Bass', 'Piano', 'Drums'], ylim=(0, 1))
            plt.setp(ax.spines.values(), linewidth=2)
            ax.tick_params(axis='both', width=3)

    def _format_fig(self):
        self.fig.legend(
            self.hand, [i.title() for i in self.lab], title='Influenced\ninstrument', frameon=False,
            bbox_to_anchor=(1, 0.625), ncol=1, markerscale=1.6, fontsize=FONTSIZE
        )
        self.fig.supylabel('Coupling constant')
        self.fig.supxlabel('Influencer instrument')
        self.fig.subplots_adjust(top=0.95, bottom=0.1, left=0.075, right=0.885)

