#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Classes used for plotting inter-onset interval complexity"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import src.visualise.visualise_utils as vutils
from src import utils

__all__ = ['HistPlotBins', 'BarPlotComplexityDensity', 'BarPlotTotalBins', 'RegPlotTempoDensityComplexity']

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


class BarPlotTotalBins(vutils.BasePlot):
    img_loc = fr'{utils.get_project_root()}\references\images\notation'

    def __init__(self, ioi_df, **kwargs):
        self.corpus_title = 'corpus_chronology'
        super().__init__(figure_title=fr'complexity_plots\barplot_totalbins_{self.corpus_title}', **kwargs)
        self.df = self._format_df(ioi_df)
        self.fig, self.ax = plt.subplots(1, 1, figsize=(vutils.WIDTH, vutils.WIDTH / 2))

    @staticmethod
    def _format_df(ioi_df):
        return (
            ioi_df.groupby(['instr', 'bin'], as_index=True, sort=False)
            ['prop_ioi']
            .count()
            .reset_index(drop=False)
            .rename(columns={'prop_ioi': 'count'})
            .pivot(index='bin', columns='instr')
            .reindex(columns=utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys(), level=1)
        )

    def _create_plot(self):
        return self.df.plot(
            kind='bar', stacked=True, ax=self.ax, color=vutils.RGB, zorder=10,
            lw=vutils.LINEWIDTH, edgecolor=vutils.BLACK, ylabel='Count',
            xlabel='Bin'
        )

    def _add_notation_images(self, y=155000):
        for tick in self.ax.get_xticklabels()[1:-1]:
            fpath = r'\notation_' + '_'.join(tick.get_text().split('{')[1:])[:-1].replace('}', '') + '.png'
            img = mpl.offsetbox.OffsetImage(
                plt.imread(self.img_loc + fpath), clip_on=False, transform=self.ax.transAxes, zoom=0.5
            )
            ab = mpl.offsetbox.AnnotationBbox(
                img, (tick._x, y), xycoords='data', clip_on=False, transform=self.ax.transAxes,
                annotation_clip=False, bboxprops=dict(edgecolor='none', facecolor='none')
            )
            self.ax.add_artist(ab)

    def _format_ax(self):
        self.ax.set(xticklabels=reversed(fracs_s))
        self.ax.tick_params(axis='both', width=vutils.TICKWIDTH, color=vutils.BLACK, rotation=0)
        plt.setp(self.ax.spines.values(), linewidth=vutils.LINEWIDTH, color=vutils.BLACK)
        self.ax.grid(zorder=0, axis='y', **vutils.GRID_KWS)
        hand, _ = self.ax.get_legend_handles_labels()
        self.ax.legend(
            hand, [i.title() for i in utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys()], title='Instrument',
            loc='upper right', frameon=True, framealpha=1, edgecolor=vutils.BLACK
        )
        self._add_notation_images()
        ax_t = self.ax.secondary_xaxis('top')
        ax_t.set_xticks(self.ax.get_xticks(), labels=[])
        ax_t.tick_params(width=vutils.TICKWIDTH)

    def _format_fig(self):
        self.fig.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.9)


class RegPlotTempoDensityComplexity(vutils.BasePlot):
    # These are keywords that we pass into our given plot types
    REG_KWS = dict(
        scatter=False, ci=95, n_boot=vutils.N_BOOT
    )
    LINE_KWS = dict(
        lw=vutils.LINEWIDTH * 2, ls=vutils.LINESTYLE, zorder=5,
        color=vutils.BLACK
    )
    SCATTER_KWS = dict(
        hue_order=utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys(),
        palette=vutils.RGB, markers=['o', 's', 'D'], s=40,
        edgecolor=vutils.BLACK, zorder=3, alpha=vutils.ALPHA,
        style_order=utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys(),
    )
    HIST_KWS = dict(
        kde=False, color=vutils.BLACK, alpha=vutils.ALPHA,
        lw=vutils.LINEWIDTH, ls=vutils.LINESTYLE
    )
    TEXT_BBOX = dict(
        facecolor=vutils.WHITE, edgecolor=vutils.BLACK,
        lw=vutils.LINEWIDTH, ls=vutils.LINESTYLE,
        boxstyle='round,pad=1'
    )

    def __init__(self, average_df, **kwargs):
        self.corpus_title = 'corpus_chronology'
        # Initialise the base plot with our given kwargs
        super().__init__(figure_title=fr'complexity_plots\regplot_tempo_density_complexity_{self.corpus_title}', **kwargs)
        self.df = average_df
        self.fig, self.ax = plt.subplots(
            nrows=2, ncols=3, figsize=(vutils.WIDTH, vutils.WIDTH / 2), sharey='row', sharex='col',
            gridspec_kw=dict(width_ratios=(1, 1, 0.2), height_ratios=(0.2, 1)),
        )
        # The main ax for plotting the regression/scatter plot
        self.main_ax = np.array(self.ax[[1, 0]])
        # Marginal ax, for plotting histograms
        self.marginal_ax = np.array([self.ax[0, 0], self.ax[0, 1], self.ax[-1, -1]])
        # Top right corner ax, which we can go ahead and disable
        self.ax[0, 2].axis('off')
        self.hand = None

    def _create_plot(self):
        self._create_main_plot()
        self._create_marginal_plot()

    def _create_main_plot(self):
        for ax, var_, xp in zip(self.main_ax.flatten(), ['lz77', 'n_onsets'], [12, 39.5]):
            sns.scatterplot(data=self.df, x=var_, y='tempo', hue='instr', style='instr', ax=ax, **self.SCATTER_KWS)
            sns.regplot(data=self.df, x=var_, y='tempo', ax=ax, line_kws=self.LINE_KWS, **self.REG_KWS)
            self._add_regression_coeff(var_, ax, xp)

    def _create_marginal_plot(self):
        # Top marginal plots
        for num, var in enumerate(['lz77', 'n_onsets']):
            sns.histplot(
                data=self.df, x=var, ax=self.marginal_ax.flatten()[num],
                bins=vutils.N_BINS,  **self.HIST_KWS
            )
        # Right marginal plot
        sns.histplot(
            data=self.df, y='tempo', ax=self.marginal_ax.flatten()[-1],
            bins=vutils.N_BINS,  **self.HIST_KWS
        )

    def _add_regression_coeff(self, var_, ax, xpos):
        r = self.df[['tempo', var_]].corr().iloc[1].iloc[0]
        ax.text(xpos, 300, rf'$r=${round(r, 2)}', bbox=self.TEXT_BBOX)

    def _format_main_ax(self):
        for ax, xl, xlab, ylab in zip(
                self.main_ax.flatten(), [(0, 15), (0, 50)],
                ['Mean complexity (LZ77, windowed)', 'Mean density ($N$ onsets, windowed)'],
                ['Mean tempo (BPM)', '']
        ):
            # Add a grid onto the plot
            ax.grid(visible=True, axis='both', which='major', zorder=0, **vutils.GRID_KWS)
            ax.tick_params(axis='both', bottom=True, right=True, width=vutils.TICKWIDTH)
            plt.setp(ax.spines.values(), linewidth=vutils.LINEWIDTH)
            ax.set(
                xlim=xl, ylim=(90, 320),
                xlabel=xlab, ylabel=ylab,
                # xticks=np.linspace(xl[0], xl[1], 5),
                # yticks=[-0.5, 0, 0.5, 1.0, 1.5]
            )
            self.hand, _ = ax.get_legend_handles_labels()
            ax.get_legend().remove()

    def _format_marginal_ax(self):
        # Remove correct spines from marginal axis
        for spine, ax in zip(['left', 'left', "bottom"], self.marginal_ax.flatten()):
            ax.spines[[spine, 'right', 'top']].set_visible(False)
            ax.tick_params(axis='both', width=vutils.TICKWIDTH)
            plt.setp(ax.spines.values(), linewidth=vutils.LINEWIDTH)
            ax.set(xlabel='', ylabel='')
            if spine == 'left':
                ax.set(yticks=[0], yticklabels=[''])
            else:
                ax.set(xticks=[0], xticklabels=[''])

    def _format_ax(self):
        self._format_main_ax()
        self._format_marginal_ax()

    def _format_fig(self):
        self.fig.tight_layout()
        self.fig.legend(
            self.hand, [i.title() for i in utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys()],
            loc='upper right', title='Instrument', frameon=True, framealpha=1,
            edgecolor=vutils.BLACK
        )
