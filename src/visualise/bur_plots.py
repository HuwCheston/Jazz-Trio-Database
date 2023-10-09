#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Classes used for plotting beat-upbeat ratios"""

from typing import Generator

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import src.visualise.visualise_utils as vutils
from src.features.features_utils import FeatureExtractor
from src import utils

# We remove BURs outside this range, these values are taken from Corcoran and Frieler (2021)
LOW_BUR_CUTOFF, HIGH_BUR_CUTOFF = 0.25, 4
# We'll plot these BUR values as musical notation
BURS_WITH_IMAGES = [0.5, 1, 2, 3]
# The folder path we're saving into
FOLDER_PATH = 'bur_plots'


def add_bur_images(ax, y) -> Generator:
    """Adds images for required BUR values"""
    # Iterate through all of our BUR values
    for x in BURS_WITH_IMAGES:
        # Add a dotted vertical line to this BUR value
        ax.axvline(x, ymin=-0.5, ymax=9, color=vutils.BLACK, alpha=1, lw=2, ls='dashed', zorder=1)
        # Try and get the image of the notation type for this BUR value
        try:
            img = plt.imread(fr'{utils.get_project_root()}\references\images\bur_notation\bur_{x}.png')
        except FileNotFoundError:
            pass
        # If we can get the image, then yield it to add to our plot
        else:
            yield mpl.offsetbox.AnnotationBbox(
                mpl.offsetbox.OffsetImage(img, clip_on=False), (x, y),
                frameon=False, xycoords='data', clip_on=False, annotation_clip=False
            )


class ViolinPlotBURs(vutils.BasePlot):
    """Plots the distribution of BUR values obtained for each musician on a specific instrument"""
    # TODO: fix this to plot BUR trends for multiple instruments

    def __init__(self, extracted_features: list[FeatureExtractor], **kwargs):
        self.corpus_title = kwargs.get('corpus_title', 'corpus')
        super().__init__(figure_title=fr'{FOLDER_PATH}\violinplot_burs_{self.corpus_title}', **kwargs)
        self.df = self._format_df(pd.DataFrame(self._get_burs(extracted_features)))
        self.fig, self.ax = plt.subplots(nrows=1, ncols=1, figsize=(vutils.WIDTH, vutils.WIDTH / 2))

    @staticmethod
    def _get_burs(extracted: list[FeatureExtractor]) -> Generator:
        """Gets the raw BUR values from all `FeatureExtractor` instances"""
        # Iterate through every track we've processed
        for track in extracted:
            # Iterate through every instrument
            for instr in utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys():
                # Get the name of the performer
                musician = track.metadata[instr]['performer']
                # Get this performer's raw BUR values for this track, dropping any NaNs
                burs_list = track.BURs[instr].bur['burs'].dropna().to_list()
                # Yield a dictionary for each BUR, including the musician name and instrument
                for bur in burs_list:
                    yield dict(musician=musician, bur=bur, instrument=instr)

    @staticmethod
    def _format_df(df: pd.DataFrame) -> pd.DataFrame:
        """Formats dataframe by removing BURs below/above cutoff and 5th/95th quantile"""
        # Get the instruments we desire and remove BURs below and above our cutoff
        df = df[(df['instrument'] == 'piano') & (df['bur'] > LOW_BUR_CUTOFF) & (df['bur'] < HIGH_BUR_CUTOFF)]
        # Threshold our df to remove values below/above the 5th/95th quantile, respectively
        res = df.groupby("musician")["bur"].quantile([0.05, 0.95]).unstack(level=1)
        return df.loc[((res.loc[df['musician'], 0.05] < df['bur'].values) & (
                    df['bur'].values < res.loc[df['musician'], 0.95])).values].sort_values(by='musician')

    def _create_plot(self) -> None:
        """Creates violinplot in seaborn"""
        # TODO: probably sort out palette here
        self.g = sns.violinplot(
            data=self.df, x='bur', y='musician', linecolor=vutils.BLACK, density_norm='count', cut=0, hue=True,
            palette='pastel', hue_order=[True, False], split=True, legend=False, inner='quart',
            inner_kws=dict(color=vutils.BLACK, lw=2), ax=self.ax
        )

    def _add_nburs_to_tick(self) -> Generator:
        """Add the total number of BURs gathered for each musician next to their name"""
        for tick in self.ax.get_yticklabels():
            tick = tick.get_text()
            yield f'{tick} ({len(self.df[self.df["musician"] == tick]["bur"].dropna())})'

    def _format_ax(self) -> None:
        """Format axis-level properties"""
        # We remove the legend here, as passing `legend=False` to the plot constructor doesn't seem to work
        self.ax.get_legend().remove()
        # Here we set the line styles
        # TODO: is this redundant given the iteration over self.ax.lines?
        for collect in self.ax.collections:
            collect.set_edgecolor(vutils.BLACK)
            collect.set_linewidth(vutils.LINEWIDTH)
        # Add in a horizontal line for each performer on the y-axis
        for tick in self.ax.get_yticks():
            self.ax.axhline(tick, 0, 3.25, color=vutils.BLACK, alpha=vutils.ALPHA, lw=vutils.LINEWIDTH)
        # Set the line styles again, possibly redundant
        for line in self.ax.lines:
            line.set_linestyle(vutils.LINESTYLE)
            line.set_color(vutils.BLACK)
            line.set_alpha(0.8)
        # Add in notation images for each of the BUR values we want to the top of the plot
        for artist in add_bur_images(ax=self.ax, y=-1.2):
            self.ax.add_artist(artist)
        # Set final properties
        self.ax.set(
            yticklabels=list(self._add_nburs_to_tick()), xticks=BURS_WITH_IMAGES,
            xlim=(0.4, 3.1), ylim=(9, -0.5), xlabel='BUR', ylabel='Performer'
        )

    def _format_fig(self) -> None:
        """Format figure-level properties"""
        # Adjust line and tick width
        plt.setp(self.ax.spines.values(), linewidth=vutils.LINEWIDTH)
        self.ax.tick_params(axis='both', width=vutils.TICKWIDTH)
        # Adjust subplot positioning
        self.fig.subplots_adjust(left=0.2, right=0.9, top=0.85, bottom=0.1)


class HistPlotBURByInstrument(vutils.BasePlot):
    def __init__(self, extracted_features: list[FeatureExtractor], **kwargs):
        self.corpus_title = kwargs.get('corpus_title', 'corpus')
        super().__init__(figure_title=fr'{FOLDER_PATH}\histplot_bursbyinstrument_{self.corpus_title}', **kwargs)
        self.df = pd.DataFrame(self._get_burs(extracted_features))
        self.df = self.df[(self.df['bur'] > LOW_BUR_CUTOFF) & (self.df['bur'] < HIGH_BUR_CUTOFF)]
        self.fig, self.ax = plt.subplots(nrows=1, ncols=1, figsize=(vutils.WIDTH, vutils.WIDTH / 2))

    @staticmethod
    def _get_burs(extracted: list[FeatureExtractor]) -> Generator:
        for track in extracted:
            for instr in utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys():
                burs_list = track.BURs[instr].bur['burs'].dropna().to_list()
                for bur in burs_list:
                    yield dict(musician=track.metadata[instr]['performer'], bur=bur, instrument=instr)

    def _create_plot(self) -> None:
        """Creates the histogram and kde plots"""
        self.g = sns.histplot(
            data=self.df, x='bur', stat='density', legend=True, hue='instrument', palette=vutils.RGB,
            bins=vutils.N_BINS, hue_order=utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys(), kde=False, multiple='layer',
            binwidth=0.1, lw=vutils.LINEWIDTH, alpha=vutils.ALPHA
        )
        sns.kdeplot(
            data=self.df, x='bur', hue='instrument', lw=vutils.LINEWIDTH * 2, ls=vutils.LINESTYLE,
            palette=vutils.RGB, hue_order=utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys(),
            clip=(LOW_BUR_CUTOFF, HIGH_BUR_CUTOFF)
        )

    def _format_ticks(self) -> Generator:
        """Formats ticks to add number of BURs"""
        for t in self.ax.get_legend().texts:
            t = t.get_text()
            yield f'{str(t).title()} ({len(self.df[self.df["instrument"] == t])})'

    def _format_ax(self) -> None:
        """Formats axis-level properties"""
        # Add images for each BUR value we want to plot
        for artist in add_bur_images(ax=self.ax, y=self.ax.get_ylim()[1] + 0.03):
            self.ax.add_artist(artist)
        # Create handles objects because seaborn doesn't create this for us, for some reason...
        handles = [
            mpl.patches.Patch(color=c, label=i) for (c, i) in zip(
                vutils.RGB, utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys()
            )
        ]
        # Add the legend in
        self.ax.legend(
            labels=list(self._format_ticks()), handles=handles, title='Instrument ($n$BUR)', fontsize=vutils.FONTSIZE,
            bbox_to_anchor=(1.22, 0.625), frameon=False, ncol=1,  markerscale=vutils.MARKERSCALE
        )
        # Set some additional axis properties
        self.ax.set(
            xticks=BURS_WITH_IMAGES, xlim=(LOW_BUR_CUTOFF, HIGH_BUR_CUTOFF), xlabel='BUR', ylabel='Density',
        )

    def _format_fig(self) -> None:
        """Formats figure-level properties"""
        plt.setp(self.ax.spines.values(), linewidth=vutils.LINEWIDTH)
        self.ax.tick_params(axis='both', width=vutils.TICKWIDTH)
        self.fig.subplots_adjust(left=0.075, bottom=0.1, right=0.83, top=0.85)


if __name__ == '__main__':
    pass
