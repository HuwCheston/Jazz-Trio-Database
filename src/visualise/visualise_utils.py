#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility classes, functions, and variables used specifically in the visualisation process"""

import functools
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


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


class ViolinPlotBURs(BasePlot):
    # TODO: fix this to plot BUR trends for multiple instruments

    def __init__(self, extracted_features, **kwargs):
        self.corpus_title = kwargs.get('corpus_title', 'corpus')
        super().__init__(figure_title=fr'bur_plots\violinplot_burs_{self.corpus_title}', **kwargs)
        self.df = self._format_df(self._get_burs(extracted_features))
        self.fig, self.ax = plt.subplots(nrows=1, ncols=1, figsize=(WIDTH, WIDTH / 2))

    @staticmethod
    def _get_burs(extracted):
        res = []
        for track in extracted:
            for instr in utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys():
                musician = track.metadata[instr]['performer']
                burs_list = track.BURs[instr].bur['burs'].dropna().to_list()
                for bur in burs_list:
                    res.append(dict(musician=musician, bur=bur, instrument=instr))
        return res

    @staticmethod
    def _format_df(burs: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(burs)
        df = df[(df['instrument'] == 'piano') & (df['bur'] > 0.25) & (df['bur'] < 4)]
        res = df.groupby("musician")["bur"].quantile([0.05, 0.95]).unstack(level=1)
        df = df.loc[((res.loc[df['musician'], 0.05] < df['bur'].values) & (
                    df['bur'].values < res.loc[df['musician'], 0.95])).values]
        df['bur_'] = utils.iqr_filter(df['bur'], low=5, high=95, fill_nans=True)
        return df.sort_values(by='musician')

    def _create_plot(self):
        sns.violinplot(
            data=self.df, x='bur_', y='musician', linecolor=BLACK, density_norm='count', cut=0, hue=True,
            palette='pastel', hue_order=[True, False], split=True, legend=False, inner='quart',
            inner_kws=dict(color=BLACK, lw=2), ax=self.ax
        )

    def _format_ax(self):
        def add_bur_images():
            for x in [0.5, 1, 2, 3]:
                self.ax.axvline(x, ymin=-0.5, ymax=9, color=BLACK, alpha=1, lw=2, ls='dashed', zorder=1)
                img = plt.imread(fr'{utils.get_project_root()}\references\images\bur_notation\bur_{x}.png')
                img = mpl.offsetbox.OffsetImage(img, clip_on=False)
                ab = mpl.offsetbox.AnnotationBbox(
                    img, (x, -1.2), frameon=False, xycoords='data', clip_on=False, annotation_clip=False
                )
                self.ax.add_artist(ab, )

        self.ax.get_legend().remove()
        for collect in self.ax.collections:
            collect.set_edgecolor(BLACK)
            collect.set_linewidth(2)
        new_ticks = []
        for tick in self.ax.get_yticklabels():
            new_ticks.append(
                f'{tick.get_text()} ({len(self.df[self.df["musician"] == tick.get_text()]["bur_"].dropna())})')
        for tick in self.ax.get_yticks():
            self.ax.axhline(tick, 0, 3.25, color=BLACK, alpha=ALPHA, lw=2)
        for line in self.ax.lines:
            line.set_linestyle('-')
            line.set_color('black')
            line.set_alpha(0.8)
        add_bur_images()
        self.ax.set(
            yticklabels=new_ticks, xticks=[0.5, 1, 2, 3], xlim=(0.4, 3.1), ylim=(9, -0.5),
            xlabel='BUR', ylabel='Performer'
        )

    def _format_fig(self):
        plt.setp(self.ax.spines.values(), linewidth=2)
        self.ax.tick_params(axis='both', width=3)
        self.fig.subplots_adjust(left=0.2, right=0.9, top=0.85, bottom=0.1)
