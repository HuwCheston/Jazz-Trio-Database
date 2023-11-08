#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Classes used for plotting asynchrony and rhythmic feel"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as signal
import scipy.stats as stats
import seaborn as sns

import src.visualise.visualise_utils as vutils
from src import utils

__all__ = ['PolarPlotAsynchrony', 'BarPlotProportionalAsynchrony', 'HistPlotProportionalAsynchrony']


class PolarPlotAsynchrony(vutils.BasePlot):
    KDE_BANDWIDTH = 0.0095
    FILL_KWS = dict(step='pre', alpha=0.1, zorder=3,)
    LINE_KWS = dict(drawstyle='steps-pre', linestyle=vutils.LINESTYLE, linewidth=vutils.LINEWIDTH, zorder=3)
    ARROW_KWS = dict(
        coordsA='figure fraction', coordsB='figure fraction', arrowstyle="-|>", color=vutils.BLACK,
        connectionstyle=f"arc3,rad=-{np.radians(22.5)}", linewidth=vutils.LINEWIDTH * 2,
        linestyle=vutils.LINESTYLE, mutation_scale=16
    )
    CIRCLE_FILL_KWS = dict(fc=vutils.WHITE, zorder=5, linewidth=vutils.LINEWIDTH)
    CIRCLE_LINE_KWS = dict(zorder=10, color=vutils.BLACK, markerfacecolor=vutils.WHITE, markersize=1/10)
    CIRCLE_PADDING = 0.5
    CIRCLE_PADDING_RANGE = range(int(CIRCLE_PADDING * 10), int((CIRCLE_PADDING * 10) + 12), 2)

    def __init__(self, async_df: pd.DataFrame, **kwargs):
        """Called when initialising the class"""
        self.corpus_title = 'corpus_chronology'
        # Initialise the base plot with our given kwargs
        super().__init__(figure_title=fr'asynchrony_plots\polarplot_asynchrony_{self.corpus_title}', **kwargs)
        self.fig, self.ax = plt.subplots(
            subplot_kw={'projection': 'polar'}, figsize=(vutils.WIDTH / 2, vutils.WIDTH / 2)
        )
        self.df = async_df

    def _kde(self, data, len_data: int = 1000):
        # Fit the actual KDE to the data, using the default parameters
        kde = stats.gaussian_kde(data.T, bw_method=self.KDE_BANDWIDTH)
        # Create a linear space of integers ranging from our lowest to our highest BUR
        data_plot = np.linspace(data.min(), data.max(), len_data)[:, np.newaxis]
        # Evaluate the KDE on our linear space of integers
        y = kde.evaluate(data_plot.T)
        return data_plot, np.array([(y_ - min(y)) / (max(y) - min(y)) for y_ in y])

    def _create_plot(self):
        for col, (idx, grp) in zip(vutils.RGB, self.df.groupby('instr', sort=False)):
            x, y = self._kde(grp['asynchrony_offset'].values)
            y += self.CIRCLE_PADDING
            x = np.radians(x).T[0]
            self.ax.plot(x, y, color=col, label=idx.title(), **self.LINE_KWS)
            self.ax.fill_between(x, y, color=col, **self.FILL_KWS)

    def _add_center_circle(self):
        # Plot a filled circle at the center with a larger z-order
        ls = np.linspace(0, 2 * np.pi, 10000)
        self.ax.fill_between(ls, 0, self.CIRCLE_PADDING, **self.CIRCLE_FILL_KWS)
        self.ax.plot(ls, np.full_like(ls, self.CIRCLE_PADDING), **self.CIRCLE_LINE_KWS)

    def _format_ticks(self):
        rm = self.ax.get_rmax()
        for t, i, r, in zip(np.deg2rad(np.arange(0, 360, 90)), range(1, 5), [0, 90, 0, 270],):
            self.ax.plot(
                [t, t], [rm, self.CIRCLE_PADDING], color=vutils.BLACK,
                lw=vutils.LINEWIDTH / 2, ls=vutils.LINESTYLE, zorder=0
            )
            self.ax.plot([t, t], [rm * 0.96, rm * 0.99], clip_on=False, color=vutils.BLACK, lw=vutils.LINEWIDTH)
            self.ax.text(t, rm + 0.1, rf'Beat $\frac{{{i}}}{{4}}$', ha='center', va='center', rotation=r)
            t += np.radians(45)
            self.ax.plot(
                [t, t], [rm, self.CIRCLE_PADDING], color=vutils.BLACK,
                alpha=vutils.ALPHA, lw=vutils.LINEWIDTH / 2, ls='--', zorder=0
            )

    def _format_ax(self):
        self._add_center_circle()
        self._format_ticks()
        self.ax.set(xticks=np.radians([0, 90, 180, 270]), xticklabels=['' for _ in range(1, 5)], rlim=(0, 1))
        self.ax.xaxis.grid(False)
        self.ax.yaxis.grid(True, **vutils.GRID_KWS)
        self.ax.set_theta_zero_location('N')
        self.ax.set_theta_direction(-1)
        self.ax.set_rticks([i / 10 for i in self.CIRCLE_PADDING_RANGE], labels=[])
        self.ax.legend(loc='center', title='Instrument', frameon=True, framealpha=1, edgecolor=vutils.BLACK)
        plt.setp(self.ax.spines.values(), linewidth=vutils.LINEWIDTH)
        self.ax.text(np.radians(37.5), 0.75, 'Density', rotation=45)
        for i in self.CIRCLE_PADDING_RANGE:
            i /= 10
            self.ax.text(
                np.radians(45), i * 1.03, round(i - self.CIRCLE_PADDING, 1), ha='left', va='center', rotation=45
            )

    def _format_fig(self):
        # Use ax.transData.transform to convert fractions to figure coordinates
        # Create the ConnectionPatch with figure coordinates
        for xyA, xyB in zip([(0.555, 0.95), (0.45, 0.05)], [(0.955, 0.55), (0.055, 0.45)]):
            curved_line = mpl.patches.ConnectionPatch(xyA=xyA, xyB=xyB, **self.ARROW_KWS)
            self.ax.add_artist(curved_line)
        self.fig.text(0.85, 0.85, 'Time')
        self.fig.text(0.155, 0.9415, 'avg. drums position $=$')
        st = r'     Time, $\frac{4}{4}$ measure' \
             '\n(proportional duration)'
        self.fig.text(0.025, 0.025, st)
        self.fig.tight_layout()


class BarPlotProportionalAsynchrony(vutils.BasePlot):
    BAR_KWS = dict(
        dodge=True, errorbar=None, width=0.8, estimator=np.mean,
        zorder=5, hue_order=utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys(),
    )
    ERROR_KWS = dict(zorder=15, color=vutils.BLACK, ls=vutils.LINESTYLE, lw=vutils.LINEWIDTH)

    def __init__(self, async_df: pd.DataFrame, **kwargs):
        """Called when initialising the class"""
        self.corpus_title = 'corpus_chronology'
        # Initialise the base plot with our given kwargs
        super().__init__(figure_title=fr'asynchrony_plots\barplot_asynchrony_{self.corpus_title}', **kwargs)
        self.df = async_df
        self.fig, self.ax = plt.subplots(nrows=1, ncols=1, figsize=(vutils.WIDTH / 2, vutils.WIDTH / 4))

    def _bootstrap(self):
        for_boot = self.df.groupby(['bandleader', 'instr', 'beat']).mean().reset_index(drop=False)
        for i in range(vutils.N_BOOT):
            print(i)
            # Shuffle the dataset
            sample = for_boot.sample(frac=1, replace=True, random_state=i)
            # Get the value required to shift drummers onto the mean
            shift_value = sample[(sample['instr'] == 'drums') & (sample['beat'] == 1)]['asynchrony_adjusted'].mean()
            # Shift the whole sample
            sample['asynchrony_shifted'] = sample['asynchrony_adjusted'] - shift_value
            # Iterate through each instrument and beat in this shuffled combination
            for i_, grp in sample.groupby(['instr', 'beat'], sort=False):
                # Get the mean value over all bandleaders and yield as a dictionary
                boot_mean = grp['asynchrony_shifted'].mean()
                yield dict(instr=i_[0], beat=i_[1] - 1, mean=boot_mean)

    def _create_plot(self):
        sns.barplot(
            data=self.df, x='beat', y='asynchrony_adjusted_offset', hue='instr', ax=self.ax, palette=vutils.RGB,
            ec=vutils.BLACK, ls=vutils.LINESTYLE, lw=vutils.LINEWIDTH, alpha=1, **self.BAR_KWS
        )
        self._bootstrap_errorbars()

    def _bootstrap_errorbars(self):
        bootstrap_df = (
            pd.DataFrame(self._bootstrap())
            .groupby(['instr', 'beat'], sort=False)
            .agg(dict(mean=[lambda x_: np.percentile(x_, 2.5), lambda x_: np.percentile(x_, 97.5)]))
            .reset_index(drop=False)
            .set_index('instr')
            .loc[utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys()]
            .reset_index(drop=False)
        )
        bootstrap_df.columns = ['instr', 'beat', 'low', 'high']
        for (idx_, grp_), cont, ran in zip(bootstrap_df.groupby('instr', sort=False), self.ax.containers, range(0, 3)):
            for (idx__, grp__), rect in zip(grp_.groupby('beat'), cont.patches):
                rect.set_hatch(vutils.HATCHES[ran])
                if idx_ == 'drums' and idx__ == 0:
                    continue
                x = rect.xy[0] + 0.1333333
                self.ax.plot((x, x), (grp__['low'], grp__['high']), **self.ERROR_KWS)
                for i in ['low', 'high']:
                    self.ax.plot((x - 0.025, x + 0.025), (grp__[i], grp__[i]), **self.ERROR_KWS)

    def _format_ax(self):
        self.ax.set(
            ylim=(-1/64 - 0.01, 1/32 + 0.005), xticklabels=[1, 2, 3, 4], xlabel='Beat',
            ylabel='Proportional position\n($\pm0=$ drums, avg. beat 1)', yticks=[-1/64, 0, 1/64, 1/32],
            yticklabels=[r'–$\frac{1}{64}$', r'$\pm$0', r'+$\frac{1}{64}$', r'+$\frac{1}{32}$']
        )
        hand, _ = self.ax.get_legend_handles_labels()
        self.ax.legend(
            hand, ['Piano', 'Bass', 'Drums'], loc='lower left', title='',
            frameon=True, framealpha=1, edgecolor=vutils.BLACK
        )
        for item in self.ax.get_legend().legend_handles:
            item.set_edgecolor('black')
            item.set_linewidth(vutils.LINEWIDTH)
        self.ax.text(3.875, 0, r'$\pm$0', ha='right', va='center', clip_on=False, zorder=1000)
        for pict, v in zip(self._add_notation_vals(), ['–', '+', '+']):
            self.ax.text(pict.xy[0] - 0.1, pict.xy[1], v, ha='right', va='center', clip_on=False, zorder=1000)
            self.ax.add_artist(pict)
        self.ax.yaxis.grid(True, zorder=0, **vutils.GRID_KWS)
        plt.setp(self.ax.spines.values(), linewidth=vutils.LINEWIDTH)
        self.ax.tick_params(axis='both', bottom=True, right=True, width=vutils.TICKWIDTH)
        self.ax.axhline(0, 0, 3, color=vutils.BLACK, lw=vutils.LINEWIDTH, ls=vutils.LINESTYLE)

    @staticmethod
    def _add_notation_vals():
        for val in [-64, 32, 64]:
            try:
                img = plt.imread(fr'{utils.get_project_root()}\references\images\notation\notation_{abs(val)}.png')
            except FileNotFoundError:
                pass
            # If we can get the image, then yield it to add to our plot
            else:
                yield mpl.offsetbox.AnnotationBbox(
                    mpl.offsetbox.OffsetImage(img, clip_on=False, zoom=0.15), (3.875, 1/val),
                    frameon=False, xycoords='data', clip_on=False, annotation_clip=False
                 )

    def _format_fig(self):
        self.fig.subplots_adjust(right=0.9, top=0.95, bottom=0.15, left=0.15)


class HistPlotProportionalAsynchrony(vutils.BasePlot):
    PLOT_KWS = dict(lw=vutils.LINEWIDTH, ls=vutils.LINESTYLE, zorder=5)
    FILL_KWS = dict(alpha=0.1, zorder=0)
    VLINE_KWS = dict(linestyle='dashed', alpha=1, zorder=4, linewidth=vutils.LINEWIDTH * 1.5)

    def __init__(self, async_df: pd.DataFrame, **kwargs):
        """Called when initialising the class"""
        self.corpus_title = 'corpus_chronology'
        # Initialise the base plot with our given kwargs
        super().__init__(figure_title=fr'asynchrony_plots\histplot_asynchrony_{self.corpus_title}', **kwargs)
        self.df = async_df
        self.fig, self.ax = plt.subplots(nrows=1, ncols=1, figsize=(vutils.WIDTH / 2, vutils.WIDTH / 4))

    @staticmethod
    def _kde(vals):
        # Fit the actual KDE to the data, using the default parameters
        kde = stats.gaussian_kde(vals.T)
        # Create a linear space of integers ranging from our lowest to our highest BUR
        x = np.linspace(vals.min(), vals.max(), 100)[:, np.newaxis].T[0]
        # Evaluate the KDE on our linear space of integers
        y = kde.evaluate(x)
        y = np.array([(y_ - min(y)) / (max(y) - min(y)) for y_ in y])
        return x, y

    @staticmethod
    def _find_peaks(x, y):
        # Find the peaks from our fitted KDE
        peaks, _ = signal.find_peaks(y)
        # Return the sorted peaks from our KDE: this will be an array of BUR values
        return np.sort(x[peaks].flatten())[0]

    def _create_plot(self):
        for (idx, grp), col in zip(self.df.groupby('instr', sort=False), vutils.RGB):
            vals = grp['asynchrony_adjusted_offset'].values
            x, y = self._kde(vals)
            peaks = np.mean(vals)
            self.ax.axvline(peaks, 0, 1, color=col, **self.VLINE_KWS)
            self.ax.plot(x, y, color=col, **self.PLOT_KWS)
            self.ax.fill_between(x, y, color=col, **self.FILL_KWS)

    def _format_ax(self):
        self.ax.xaxis.grid(True, **vutils.GRID_KWS)
        self.ax.set(
            xlim=(-1/32 - 0.01, 1/16 + 0.001), ylim=(0, 1.01), xticks=[-1/32, -1/64, 0, 1/64, 1/32, 1/16],
            ylabel='Density', xlabel='Proportional position ($\pm0=$ drums, avg. beat 1)',
            xticklabels=[
                r'–$\frac{1}{32}$', r'–$\frac{1}{64}$', r'$\pm$0',
                r'+$\frac{1}{64}$', r'+$\frac{1}{32}$', r'+$\frac{1}{16}$'
            ],
        )
        plt.setp(self.ax.spines.values(), linewidth=vutils.LINEWIDTH)
        self.ax.tick_params(axis='both', bottom=True, top=True, width=vutils.TICKWIDTH)
        self.ax.text(0, 1.1, r'$\pm$0', ha='center', va='center', clip_on=False, zorder=1000)
        self._add_images()

    def _add_images(self):
        for val in [-32, -64, 32, 64, 16]:
            try:
                img = plt.imread(fr'{utils.get_project_root()}\references\images\notation\notation_{abs(val)}.png')
            except FileNotFoundError:
                pass
            # If we can get the image, then yield it to add to our plot
            else:
                self.ax.text(
                    1/val - 0.001, 1.1, '–' if val < 0 else '+', ha='right', va='center', clip_on=False, zorder=1000
                )
                self.ax.add_artist(mpl.offsetbox.AnnotationBbox(
                    mpl.offsetbox.OffsetImage(img, clip_on=False, zoom=0.15), (1/val, 1.1),
                    frameon=False, xycoords='data', clip_on=False, annotation_clip=False, zorder=0
                 ))

    def _format_fig(self):
        self.fig.subplots_adjust(left=0.1, right=0.95, top=0.85, bottom=0.175)
