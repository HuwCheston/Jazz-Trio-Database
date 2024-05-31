#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Classes used for plotting inter-onset interval complexity"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats

import src.visualise.visualise_utils as vutils
from src import utils

__all__ = [
    'HistPlotBins', 'HistPlotBinsTrack', 'BarPlotComplexityDensity', 'BarPlotTotalBins',
    'RegPlotTempoDensityComplexity', 'FRACS', 'FRACS_S', 'ViolinPlotOnsetComplexity'

]

FRACS = [1, 1 / 2, 5 / 12, 3 / 8, 1 / 3, 1 / 4, 1 / 6, 1 / 8, 1 / 12, 0]
FRACS_S = [r'>$\frac{1}{2}$', r'$\frac{1}{2}$', r'$\frac{5}{12}$', r'$\frac{3}{8}$', r'$\frac{1}{3}$',
           r'$\frac{1}{4}$', r'$\frac{1}{6}$', r'$\frac{1}{8}$', r'$\frac{1}{12}$', r'<$\frac{1}{12}$']


class ViolinPlotOnsetComplexity(vutils.BasePlot):
    PAL = sns.cubehelix_palette(dark=1 / 3, gamma=.3, light=2 / 3, start=0, n_colors=20, as_cmap=False)
    img_loc = fr'{utils.get_project_root()}/references/images/musicians'
    KDE_KWS = dict(
        linecolor=vutils.BLACK, density_norm='count', hue=True,
        palette=PAL, hue_order=[True, False], split=True,
        legend=False, inner=None, bw_method='scott', cut=0,
        gridsize=vutils.N_BOOT
    )
    EBAR_KWS = dict(
        ls='none', color=vutils.RED, linewidth=vutils.LINEWIDTH * 1.5, capsize=5,
        capthick=vutils.LINEWIDTH * 1.5, ecolor=vutils.RED, zorder=10
    )
    SCAT_KWS = dict(
        facecolor=vutils.RED, lw=vutils.LINEWIDTH / 2,
        edgecolor=vutils.BLACK, s=100, zorder=15
    )

    def __init__(self, complex_df: pd.DataFrame, **kwargs):
        self.corpus_title = kwargs.get('corpus_title', 'corpus_updated')
        self.include_images = kwargs.get('include_images', True)
        fig_title = fr'complexity_plots/violinplot_complexity_{self.corpus_title}'
        if not self.include_images:
            fig_title += '_no_images'
        super().__init__(figure_title=fig_title, **kwargs)
        self.df = self._format_df(complex_df)
        self.fig, self.ax = plt.subplots(
            nrows=1, ncols=2, sharex=False, sharey=True,
            figsize=(vutils.WIDTH, vutils.WIDTH / 2)
        )

    @staticmethod
    def _format_df(df):
        order = reversed(
            df.groupby('bandleader', as_index=False)
            ['n_onsets']
            .mean()
            .sort_values(by='n_onsets')
            ['bandleader']
            .values
        )
        return (
            df.copy(deep=True)
            .set_index('bandleader')
            .loc[order]
            .reset_index(drop=False)
        )

    def _create_plot(self):
        for a, v in zip(self.ax.flatten(), ['n_onsets', 'lz77']):
            bwa = 1 if v == 'n_onsets' else 1.6
            sns.violinplot(
                data=self.df, x=v, y='bandleader', ax=a, bw_adjust=bwa, **self.KDE_KWS
            )
            for patch, col in zip(a.collections, self.PAL):
                patch.set_facecolor(col)
                patch.set_edgecolor(vutils.BLACK)
                patch.set_linewidth(vutils.LINEWIDTH)
            for line in a.lines:
                line.set_linestyle(vutils.LINESTYLE)
                line.set_color(vutils.BLACK)
                line.set_alpha(0.8)
            med = self.df.groupby('bandleader', as_index=False, sort=False).agg({v: ['mean', 'std']})
            med.columns = [col[0] if col[1] == '' else col[1] for col in med.columns]
            a.scatter(med['mean'], [i + 0.4 for i in range(len(med))], **self.SCAT_KWS)
            for idx, row in med.iterrows():
                a.errorbar(
                    x=[row['mean'], row['mean']], y=[idx + 0.4, idx + 0.4],
                    xerr=[row['std'], row['std']], **self.EBAR_KWS
                )
                a.axhline(
                    idx + 0.4, 0, 1, color=vutils.BLACK,
                    linewidth=vutils.LINEWIDTH, linestyle=vutils.LINESTYLE, zorder=0
                )

    def _add_bandleader_images(self, ax, bl: str, y: float) -> None:
        """Adds images corresponding to a given bandleader `bl` at position `y`"""
        fpath = fr'{self.img_loc}/{bl.replace(" ", "_").lower()}.png'
        img = mpl.offsetbox.OffsetImage(
            plt.imread(fpath), clip_on=False, transform=ax.transAxes, zoom=0.5
        )
        ab = mpl.offsetbox.AnnotationBbox(
            img, (-6, y + 0.2), xycoords='data', clip_on=False, transform=ax.transAxes,
            annotation_clip=False, bboxprops=dict(edgecolor='none', facecolor='none')
        )
        ax.add_artist(ab)

    def _format_ax(self):
        """Format axis-level properties"""
        # Set final properties
        # self._add_nburs_to_tick()
        for ax, xlim, xlab in zip(
                self.ax.flatten(),
                [(0, 64), (4, 20)],
                ['Density ($N$ onsets)', 'Compression (LZ77)']
        ):
            ax.set(
                xticks=np.linspace(xlim[0], xlim[1], 5),
                # xticklabels=[-1, 0, 1, 1.585],
                xlabel=xlab, ylabel='', xlim=xlim, ylim=(9.6, -0.3),
                yticks=np.linspace(0.4, 9.4, 10)
            )
            ax.set_xlabel(xlab, fontsize=vutils.FONTSIZE * 1.1)
            # ax.set_yticklabels([i.get_text() for i in ax.get_yticklabels()], va='bottom')
            ax.grid(axis='x', which='major', **vutils.GRID_KWS)
            ax.tick_params(axis='y', which='major', pad=70)
            # Adjust line and tick width
            plt.setp(ax.spines.values(), linewidth=vutils.LINEWIDTH)
            ax.tick_params(axis='both', width=vutils.TICKWIDTH)
        if self.include_images:
            for num, pi in enumerate(self.df['bandleader'].unique()):
                self._add_bandleader_images(self.ax[0], pi, num)

    def _format_fig(self) -> None:
        """Format figure-level properties"""
        # self.fig.supxlabel('Four measure window', x=0.55)
        self.fig.supylabel('Pianist')
        # Adjust subplot positioning
        self.fig.subplots_adjust(left=0.2, right=0.95, top=0.85, bottom=0.1, hspace=0, wspace=0.05)


class HistPlotBins(vutils.BasePlot):
    """Creates a histogram showing the density of inter-onset intervals across each bin for all instruments"""
    PALETTE = [vutils.BLACK, *reversed(sns.color_palette(None, len(FRACS) - 2)), vutils.BLACK]
    HIST_KWS = dict(lw=vutils.LINEWIDTH / 2, ls=vutils.LINESTYLE, zorder=2, align='edge')
    LINE_KWS = dict(linestyle=vutils.LINESTYLE, alpha=1, zorder=3, linewidth=vutils.LINEWIDTH, color=vutils.BLACK)
    VLINE_KWS = dict(
        color=vutils.BLACK, alpha=vutils.ALPHA, zorder=4, linewidth=vutils.LINEWIDTH / 1.5
    )

    def __init__(self, ioi_df: pd.DataFrame, **kwargs):
        self.corpus_title = 'corpus_chronology'
        fname = kwargs.get('figure_title', fr'complexity_plots/histplot_ioibins_{self.corpus_title}')
        super().__init__(figure_title=fname)
        self.n_bins = kwargs.get('n_bins', 300)
        self.ioi_df = ioi_df
        self.fig, self.ax = plt.subplots(
            nrows=1, ncols=3, figsize=(vutils.WIDTH, vutils.WIDTH / 3), sharex=True, sharey=True
        )

    def _create_plot(self) -> None:
        """Create the main plot"""
        mapping = {f: c for f, c in zip(FRACS, self.PALETTE)}
        for ax, (idx, grp), col in zip(self.ax.flatten(), self.ioi_df.groupby('instr', sort=False), vutils.RGB):
            grp = grp.dropna()
            if len(grp) == 0:
                continue
            # Normalize the histogram so that the highest bar is 1
            heights, edges = np.histogram(grp['prop_ioi'], bins=self.n_bins)
            heights = heights / max(heights)
            # Plot the normalized histogram
            self.HIST_KWS.update(dict(x=edges[:-1], height=heights, width=np.diff(edges)))
            b = ax.bar(edgecolor='None', alpha=1, **self.HIST_KWS)
            xs, ys = [], []
            for b_ in b:
                new_color = mapping[min(FRACS, key=lambda x: abs(x - b_.xy[0]))]
                b_.set_fc(new_color)
                xs.append(b_.xy[0])
                ys.append(b_.get_height())
            ax.plot(xs, ys, **self.LINE_KWS)
            for frac in FRACS[1:-1]:
                ax.axvline(frac, 0, 1, **self.VLINE_KWS)
            ax.set(title=idx.title())

    def _format_ax(self) -> None:
        """Format axis-level parameters"""
        for ax in self.ax.flatten():
            plt.setp(ax.spines.values(), linewidth=vutils.LINEWIDTH)
            ax.tick_params(axis='both', width=vutils.TICKWIDTH)
            ax.set(ylim=(0, 1), xlim=(0, 1))
            ax_t = ax.secondary_xaxis('top')
            ax_t.set_xticks(FRACS[1:-1], labels=FRACS_S[1:-1])
            ax_t.tick_params(width=vutils.TICKWIDTH)

    def _format_fig(self) -> None:
        """Format figure-level parameters"""
        self.fig.supxlabel(f'Inter-onset interval (measures)')
        self.fig.supylabel('Density')
        self.fig.subplots_adjust(bottom=0.15, top=0.85, left=0.075, right=0.95, hspace=0.05)


class BarPlotComplexityDensity(vutils.BasePlot):
    """Createa a barplot showing levels of complexity and density for each instrument"""
    BAR_KWS = dict(
        dodge=False, edgecolor=vutils.BLACK, errorbar=('ci', 95),
        lw=vutils.LINEWIDTH, seed=42, capsize=0.1, width=0.8,
        ls=vutils.LINESTYLE, estimator=np.mean,
        errcolor=vutils.BLACK, zorder=3,
        hue_order=utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys(),
        palette=vutils.RGB, alpha=0.8, n_boot=vutils.N_BOOT
    )

    def __init__(self, complex_df, **kwargs):
        self.corpus_title = 'corpus_chronology'
        super().__init__(figure_title=fr'complexity_plots/barplot_complexity_density_{self.corpus_title}', **kwargs)
        self.df = (
            complex_df.set_index('instr')
            .loc[utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys()]
            .reset_index(drop=False)
            .dropna()
        )
        self.fig, self.ax = plt.subplots(
            nrows=1, ncols=2, figsize=(vutils.WIDTH, vutils.WIDTH / 4), sharex=True, sharey=False
        )

    def _create_plot(self) -> None:
        """Creates plots in seaborn"""
        sns.barplot(data=self.df, x='instr', y='lz77', ax=self.ax[0], **self.BAR_KWS)
        sns.barplot(data=self.df, x='instr', y='n_onsets', ax=self.ax[1], **self.BAR_KWS)

    def _format_ax(self) -> None:
        """Format axis-level parameters"""
        for ax, lab in zip(self.ax.flatten(), ['Complexity\n(compressed length)', 'Onset density\n($N$ onsets)']):
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

    def _format_fig(self) -> None:
        """Formats figure-level parameters"""
        self.fig.subplots_adjust(top=0.9, bottom=0.135, left=0.065, right=0.975)


class BarPlotTotalBins(vutils.BasePlot):
    """Creates a barplot showing the number of inter-onset intervals contained within each bin"""
    img_loc = fr'{utils.get_project_root()}/references/images/notation'
    BAR_KWS = dict(
        color=vutils.RGB, zorder=10, lw=vutils.LINEWIDTH, edgecolor=vutils.BLACK, ylabel='Count', label='Bin',
        kind='bar', stacked=True,
    )

    def __init__(self, ioi_df, **kwargs):
        self.corpus_title = 'corpus_updated'
        super().__init__(figure_title=fr'complexity_plots/barplot_totalbins_{self.corpus_title}', **kwargs)
        # Coerce provided dataframe into correct format for plotting
        self.df = (
            ioi_df.groupby(['instr', 'bin'], as_index=True, sort=False)
            ['prop_ioi']
            .count()
            .reset_index(drop=False)
            .rename(columns={'prop_ioi': 'count'})
            .pivot(index='bin', columns='instr')
            .reindex(columns=utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys(), level=1)
        )
        self.fig, self.ax = plt.subplots(1, 1, figsize=(vutils.WIDTH, vutils.WIDTH / 2))

    def _create_plot(self):
        return self.df.plot(
            kind='bar', stacked=True, ax=self.ax, color=vutils.RED, zorder=10,
            lw=vutils.LINEWIDTH, edgecolor=vutils.BLACK, ylabel='Count',
            xlabel='Binned inter-onset interval'
        )

    def _add_notation_images(self, y: int = 155000) -> None:
        """Adds notation images to plot at given position `y`"""
        for tick in self.ax.get_xticklabels()[1:-1]:
            fpath = r'/notation_' + '_'.join(tick.get_text().split('{')[1:])[:-1].replace('}', '') + '.png'
            img = mpl.offsetbox.OffsetImage(
                plt.imread(self.img_loc + fpath), clip_on=False, transform=self.ax.transAxes, zoom=0.5
            )
            ab = mpl.offsetbox.AnnotationBbox(
                img, (tick._x, y), xycoords='data', clip_on=False, transform=self.ax.transAxes,
                annotation_clip=False, bboxprops=dict(edgecolor='none', facecolor='none')
            )
            self.ax.add_artist(ab)

    def _format_ax(self):
        """Format axis-level parameters"""
        self.ax.set(xticklabels=reversed(FRACS_S))
        self.ax.tick_params(axis='both', width=vutils.TICKWIDTH, color=vutils.BLACK, rotation=0)
        plt.setp(self.ax.spines.values(), linewidth=vutils.LINEWIDTH, color=vutils.BLACK)
        self.ax.grid(zorder=0, axis='y', **vutils.GRID_KWS)
        self.ax.get_legend().remove()
        self._add_notation_images(y=90000)
        ax_t = self.ax.secondary_xaxis('top')
        ax_t.set_xticks(self.ax.get_xticks(), labels=[])
        ax_t.tick_params(width=vutils.TICKWIDTH)

    def _format_fig(self) -> None:
        """Formats figure-level parameters"""
        self.fig.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.9)


class RegPlotTempoDensityComplexity(vutils.BasePlot):
    """Creates a regression plot showing associations between tempo, density, and complexity scores"""
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
        super().__init__(
            figure_title=fr'complexity_plots/regplot_tempo_density_complexity_{self.corpus_title}', **kwargs
        )
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

    def _create_plot(self) -> None:
        """Creates both main and marginal plot"""
        self._create_main_plot()
        self._create_marginal_plot()

    def _create_main_plot(self) -> None:
        """Creates main plotting object (scatter and regression plots)"""
        for ax, var_, xp in zip(self.main_ax.flatten(), ['lz77', 'n_onsets'], [12, 39.5]):
            sns.scatterplot(data=self.df, x=var_, y='tempo', hue='instr', style='instr', ax=ax, **self.SCATTER_KWS)
            sns.regplot(data=self.df, x=var_, y='tempo', ax=ax, line_kws=self.LINE_KWS, **self.REG_KWS)
            self._add_regression_coeff(var_, ax, xp)

    def _create_marginal_plot(self) -> None:
        """Creates marginal plotting objects (density plots)"""
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

    def _add_regression_coeff(self, var_: str, ax: plt.Axes, xpos: float) -> None:
        """Adds regression plot between tempo and given variable `var_` onto axis object `ax` at position `xpos`"""
        r = self.df[['tempo', var_]].corr().iloc[1].iloc[0]
        ax.text(xpos, 300, rf'$r=${round(r, 2)}', bbox=self.TEXT_BBOX)

    def _format_main_ax(self) -> None:
        """Formats axis-level parameters for main plot"""
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

    def _format_marginal_ax(self) -> None:
        """Formats axis-level parameters for marginal ax"""
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

    def _format_ax(self) -> None:
        """Format axis-level parameters for both main and marginal ax"""
        self._format_main_ax()
        self._format_marginal_ax()

    def _format_fig(self) -> None:
        """Format figure level parameters"""
        self.fig.tight_layout()
        self.fig.legend(
            self.hand, [i.title() for i in utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys()],
            loc='upper right', title='Instrument', frameon=True, framealpha=1,
            edgecolor=vutils.BLACK
        )


class HistPlotBinsTrack(HistPlotBins):
    def __init__(self, onset_maker, **kwargs):
        self.df = self._format_df(onset_maker)
        self.fname = rf'onsets_plots/histplot_complexity_{onset_maker.item["mbz_id"]}'
        self.title = onset_maker.item['fname']
        super().__init__(self.df, figure_title=self.fname, n_bins=kwargs.get('n_bins', 10))

    def _format_df(self, om):
        from src.features.rhythm_features import IOIComplexity
        downbeats = om.ons['downbeats_manual']
        time_signature = om.item['time_signature']
        tempo = om.tempo
        cdfs = []
        for instr in utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys():
            my_onsets = om.ons[instr]
            cdf = IOIComplexity(
                my_onsets=my_onsets,
                downbeats=downbeats,
                tempo=tempo,
                time_signature=time_signature
            )
            cdfs.append({'instr': instr, 'prop_ioi': np.nan})
            for ioi in cdf.binned_iois['binned_ioi'].dropna().values:
                cdfs.append({'instr': instr, 'prop_ioi': ioi})
        return pd.DataFrame(cdfs)

    @staticmethod
    def _kde(data, len_data: int = 1000) -> tuple:
        """Fit the KDE to the data and evaluate on a list of y-values, then scale"""
        # Fit the actual KDE to the data, using the default parameters
        kde = stats.gaussian_kde(data.T)
        # Create a linear space of integers ranging from our lowest to our highest BUR
        data_plot = np.linspace(0, 1, len_data)[:, np.newaxis]
        # Evaluate the KDE on our linear space of integers
        y = kde.evaluate(data_plot.T)
        return data_plot, np.array([(y_ - min(y)) / (max(y) - min(y)) for y_ in y])

    def _create_plot(self) -> None:
        """Create the main plot"""
        for ax, (idx, grp) in zip(self.ax.flatten(), self.ioi_df.groupby('instr', sort=False)):
            grp = grp.dropna()
            if len(grp) == 0:
                continue
            # Plot the kde
            xs, ys = self._kde(grp['prop_ioi'])
            ax.plot(xs, ys, **self.LINE_KWS)
            xs = xs.flatten()
            s = np.sort([(FRACS[i] + FRACS[i + 1]) / 2 for i in range(len(FRACS) - 1)]).tolist()
            for previous, current, col in zip(s, s[1:], list(reversed(self.PALETTE))[1:]):
                slicer = np.where((xs <= current) & (xs >= previous))
                xvals = xs[slicer]
                yvals = ys[slicer]
                ax.fill_between(xvals, yvals, color=col)
            for frac in FRACS[1:-1]:
                ax.axvline(frac, 0, 1, **self.VLINE_KWS)

    def _format_ax(self) -> None:
        """Format axis-level parameters"""
        for ax, (idx, grp) in zip(self.ax.flatten(), self.df.groupby('instr', sort=False)):
            plt.setp(ax.spines.values(), linewidth=vutils.LINEWIDTH)
            ax.tick_params(axis='both', width=vutils.TICKWIDTH)
            ax.set(
                ylim=(0, 1), xlim=(0, 1), title=f'{idx.title()} ($N$ = {len(grp.dropna())})',
                yticks=np.linspace(0, 1, 5)
            )
            ax_t = ax.secondary_xaxis('top')
            ax_t.set_xticks(FRACS[1:-1], labels=FRACS_S[1:-1])
            ax_t.tick_params(width=vutils.TICKWIDTH)

    def _format_fig(self) -> None:
        """Formats figure-level properties"""
        self.fig.supxlabel('Proportional inter-onset interval')
        self.fig.suptitle(self.title)
        self.fig.supylabel('Density', x=0.01)
        self.fig.subplots_adjust(left=0.075, bottom=0.12, right=0.95, top=0.765)


if __name__ == '__main__':
    pass
