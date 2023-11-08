#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Classes used for plotting inter-onset interval complexity"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import src.visualise.visualise_utils as vutils
from src import utils

__all__ = ['HistPlotBins', 'BarPlotComplexityDensity']

fracs = [1, 1/2, 5/12, 3/8, 1/3, 1/4, 1/6, 1/8, 1/12, 0]
fracs_s = [r'>$\frac{1}{2}$', r'$\frac{1}{2}$', r'$\frac{5}{12}$', r'$\frac{3}{8}$', r'$\frac{1}{3}$',
           r'$\frac{1}{4}$', r'$\frac{1}{6}$', r'$\frac{1}{8}$', r'$\frac{1}{12}$', r'<$\frac{1}{12}$']


class HistPlotBins(vutils.BasePlot):
    PALETTE = [vutils.BLACK, *reversed(sns.color_palette(None, len(fracs) - 2)), vutils.BLACK]
    HIST_KWS = dict(lw=vutils.LINEWIDTH / 2, ls=vutils.LINESTYLE, zorder=2, align='edge')
    LINE_KWS = dict(linestyle=vutils.LINESTYLE, alpha=1, zorder=3, linewidth=vutils.LINEWIDTH, color=vutils.BLACK)
    VLINE_KWS = dict(
        color=vutils.BLACK, linestyle='dashed', alpha=vutils.ALPHA, zorder=4, linewidth=vutils.LINEWIDTH / 1.5
    )

    def __init__(self, ioi_df: pd.DataFrame, **kwargs):
        self.corpus_title = 'corpus_chronology'
        super().__init__(figure_title=fr'complexity_plots\histplot_ioibins_{self.corpus_title}', **kwargs)
        self.ioi_df = ioi_df
        self.fig, self.ax = plt.subplots(
            nrows=1, ncols=3, figsize=(vutils.WIDTH, vutils.WIDTH / 3), sharex=True, sharey=True
        )

    def _create_plot(self):
        mapping = {f: c for f, c in zip(fracs, self.PALETTE)}
        for ax, (idx, grp), col in zip(self.ax.flatten(), self.ioi_df.groupby('instr', sort=False), vutils.RGB):
            # Normalize the histogram so that the highest bar is 1
            heights, edges = np.histogram(grp['prop_ioi'], bins=300)
            heights = heights / max(heights)
            # Plot the normalized histogram
            self.HIST_KWS.update(dict(x=edges[:-1], height=heights, width=np.diff(edges)))
            b = ax.bar(edgecolor='None', alpha=1, **self.HIST_KWS)
            xs, ys = [], []
            for b_ in b:
                new_color = mapping[min(fracs, key=lambda x: abs(x - b_.xy[0]))]
                b_.set_fc(new_color)
                xs.append(b_.xy[0])
                ys.append(b_.get_height())
            ax.plot(xs, ys, **self.LINE_KWS)
            for frac in fracs[1:-1]:
                ax.axvline(frac, 0, 1, **self.VLINE_KWS)
            ax.set(title=idx.title())

    def _format_ax(self):
        for ax in self.ax.flatten():
            plt.setp(ax.spines.values(), linewidth=vutils.LINEWIDTH)
            ax.tick_params(axis='both', width=vutils.TICKWIDTH)
            ax.set(ylim=(0, 1), xlim=(0, 1))
            ax_t = ax.secondary_xaxis('top')
            ax_t.set_xticks(fracs[1:-1], labels=fracs_s[1:-1])
            ax_t.tick_params(width=vutils.TICKWIDTH)

    def _format_fig(self):
        self.fig.supxlabel(f'Proportional IOI')
        self.fig.supylabel('Density')
        self.fig.subplots_adjust(bottom=0.15, top=0.85, left=0.075, right=0.95, hspace=0.05)


class BarPlotComplexityDensity(vutils.BasePlot):
    BAR_KWS = dict(
        dodge=False, edgecolor=vutils.BLACK, errorbar=('ci', 95),
        lw=vutils.LINEWIDTH, seed=42, capsize=0.1, width=0.8,
        ls=vutils.LINESTYLE, estimator=np.mean,
        errcolor=vutils.BLACK, zorder=3,
        hue_order=utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys(),
        palette=vutils.RGB, alpha=0.8
    )

    def __init__(self, complex_df, **kwargs):
        self.corpus_title = 'corpus_chronology'
        super().__init__(figure_title=fr'complexity_plots\barplot_complexity_density_{self.corpus_title}', **kwargs)
        self.df = (
            complex_df.set_index('instr')
            .loc[utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys()]
            .reset_index(drop=False)
        )
        self.fig, self.ax = plt.subplots(
            nrows=1, ncols=2, figsize=(vutils.WIDTH, vutils.WIDTH / 4), sharex=True, sharey=False
        )

    def _create_plot(self):
        sns.barplot(data=self.df, x='instr', y='lz77 mean', ax=self.ax[0], **self.BAR_KWS)
        sns.barplot(data=self.df, x='instr', y='n_onsets mean', ax=self.ax[1], **self.BAR_KWS)

    def _format_ax(self):
        for ax, lab in zip(self.ax.flatten(), ['Complexity\n(compressed length)', 'Density\n($N$ onsets)']):
            for line in ax.lines:
                line.set_zorder(5)
            for hatch, bar in zip(vutils.HATCHES, ax.patches):
                bar.set_hatch(hatch)
            # Set the width of the edges and ticks
            plt.setp(ax.spines.values(), linewidth=vutils.LINEWIDTH, color=vutils.BLACK)
            ax.tick_params(axis='both', width=vutils.TICKWIDTH, color=vutils.BLACK)
            # Add a vertical grid
            ax.grid(zorder=0, axis='y', **vutils.GRID_KWS)
            ax.set(
                xticklabels=[i.title() for i in utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys()],
                xlabel='Instrument', ylabel=lab,
            )

    def _format_fig(self):
        self.fig.subplots_adjust(top=0.9, bottom=0.135, left=0.065, right=0.975)
