#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Classes used for plotting asynchrony and rhythmic feel"""

import warnings
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as signal
import scipy.stats as stats
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning

import src.visualise.visualise_utils as vutils
from src import utils

__all__ = [
    'PolarPlotAsynchrony', 'BarPlotProportionalAsynchrony', 'HistPlotProportionalAsynchrony',
    'RegPlotPianistAsynchrony', 'HistPlotProportionalAsynchronyTriosPiano', 'ScatterPlotAsynchronyTrack'
]


class PolarPlotAsynchrony(vutils.BasePlot):
    """Creates the "propeller" plot of asynchrony values across different beat levels"""
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

    def _kde(self, data: np.array, len_data: int = 1000) -> tuple:
        """Fit the KDE to the data and evaluate"""
        # Fit the actual KDE to the data, using the default parameters
        kde = stats.gaussian_kde(data.T, bw_method=self.KDE_BANDWIDTH)
        # Create a linear space of integers ranging from our lowest to our highest BUR
        data_plot = np.linspace(data.min(), data.max(), len_data)[:, np.newaxis]
        # Evaluate the KDE on our linear space of integers
        y = kde.evaluate(data_plot.T)
        return data_plot, np.array([(y_ - min(y)) / (max(y) - min(y)) for y_ in y])

    def _create_plot(self) -> None:
        """Create the main plot"""
        for col, (idx, grp) in zip(vutils.RGB, self.df.groupby('instr', sort=False)):
            x, y = self._kde(grp['asynchrony_offset'].values)
            y += self.CIRCLE_PADDING
            x = np.radians(x).T[0]
            self.ax.plot(x, y, color=col, label=idx.title(), **self.LINE_KWS)
            self.ax.fill_between(x, y, color=col, **self.FILL_KWS)

    def _add_center_circle(self) -> None:
        """Plot a filled circle at the center of the plot"""
        # Plot a filled circle at the center with a larger z-order
        ls = np.linspace(0, 2 * np.pi, 10000)
        self.ax.fill_between(ls, 0, self.CIRCLE_PADDING, **self.CIRCLE_FILL_KWS)
        self.ax.plot(ls, np.full_like(ls, self.CIRCLE_PADDING), **self.CIRCLE_LINE_KWS)

    def _format_ticks(self) -> None:
        """Format ticks on the radial axis"""
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

    def _format_ax(self) -> None:
        """Format axis level parameters"""
        self._add_center_circle()
        self._format_ticks()
        self.ax.set(xticks=np.radians([0, 90, 180, 270]), xticklabels=['' for _ in range(1, 5)], rlim=(0, 1))
        # Add the grids in
        self.ax.xaxis.grid(False)
        self.ax.yaxis.grid(True, **vutils.GRID_KWS)
        # Set polar parameters, e.g. theta location
        self.ax.set_theta_zero_location('N')
        self.ax.set_theta_direction(-1)
        self.ax.set_rticks([i / 10 for i in self.CIRCLE_PADDING_RANGE], labels=[])
        # Adjust the legend positioning
        self.ax.legend(loc='center', title='Instrument', frameon=True, framealpha=1, edgecolor=vutils.BLACK)
        plt.setp(self.ax.spines.values(), linewidth=vutils.LINEWIDTH)
        self.ax.text(np.radians(37.5), 0.75, 'Density', rotation=45)
        for i in self.CIRCLE_PADDING_RANGE:
            i /= 10
            self.ax.text(
                np.radians(45), i * 1.03, round(i - self.CIRCLE_PADDING, 1), ha='left', va='center', rotation=45
            )

    def _format_fig(self) -> None:
        """Adjust figure-level parameters"""
        # Use ax.transData.transform to convert fractions to figure coordinates
        # Create the ConnectionPatch with figure coordinates
        for xyA, xyB in zip([(0.555, 0.95), (0.45, 0.05)], [(0.955, 0.55), (0.055, 0.45)]):
            curved_line = mpl.patches.ConnectionPatch(xyA=xyA, xyB=xyB, **self.ARROW_KWS)
            self.ax.add_artist(curved_line)
        self.fig.text(0.85, 0.85, 'Time')
        self.fig.text(0.135, 0.9415, 'mean drums position $=$')
        st = r' Time, $\frac{4}{4}$ measure' \
             '\n(relative position)'
        self.fig.text(0.025, 0.025, st)
        self.fig.tight_layout()


class BarPlotProportionalAsynchrony(vutils.BasePlot):
    """Create a barplot showing the proportional asynchrony between different instruments at all beat levels"""
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

    def _bootstrap(self) -> list:
        """Bootstrap the asynchrony values"""
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
        """Create the main plot"""
        sns.barplot(
            data=self.df, x='beat', y='asynchrony_adjusted_offset', hue='instr', ax=self.ax, palette=vutils.RGB,
            ec=vutils.BLACK, ls=vutils.LINESTYLE, lw=vutils.LINEWIDTH, alpha=1, **self.BAR_KWS
        )
        self._bootstrap_errorbars()

    def _bootstrap_errorbars(self) -> None:
        """Bootstrap the errorbars for the plot and add them in """
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

    def _format_ax(self) -> None:
        """Format axis-level parameters"""
        self.ax.set(
            ylim=(-1/64 - 0.01, 1/32 + 0.005), xticklabels=[1, 2, 3, 4], xlabel='Beat',
            ylabel='Relative position\n($\pm0=$ drums, mean beat 1)', yticks=[-1/64, 0, 1/64, 1/32],
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
    def _add_notation_vals() -> list:
        """Creates a generator of image objects (consisting of musical notation values) to be added to the plot"""
        for val in [-64, 32, 64]:
            try:
                img = plt.imread(fr'{utils.get_project_root()}\references\images\notation\notation_{abs(val)}.png')
            except FileNotFoundError:
                pass
            # If we can get the image, then yield it to add to our plot
            else:
                yield mpl.offsetbox.AnnotationBbox(
                    mpl.offsetbox.OffsetImage(img, clip_on=False, zoom=0.5), (3.875, 1/val),
                    frameon=False, xycoords='data', clip_on=False, annotation_clip=False
                 )

    def _format_fig(self) -> None:
        """Format figure-level parameters"""
        self.fig.subplots_adjust(right=0.9, top=0.95, bottom=0.15, left=0.15)


class HistPlotProportionalAsynchrony(vutils.BasePlot):
    """Creates a histogram plot of proportional asynchrony values"""
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
    def _kde(vals: np.array) -> tuple:
        """Fit the KDE to the data and evaluate on a linear space of integers, then scale"""
        # Fit the actual KDE to the data, using the default parameters
        kde = stats.gaussian_kde(vals.T)
        # Create a linear space of integers ranging from our lowest to our highest BUR
        x = np.linspace(vals.min(), vals.max(), 100)[:, np.newaxis].T[0]
        # Evaluate the KDE on our linear space of integers
        y = kde.evaluate(x)
        # Scale the KDE between 0 and 1
        y = np.array([(y_ - min(y)) / (max(y) - min(y)) for y_ in y])
        return x, y

    @staticmethod
    def _find_peaks(x, y) -> np.array:
        """Find peaks from a fitted KDE and sort them"""
        # Find the peaks from our fitted KDE
        peaks, _ = signal.find_peaks(y)
        # Return the sorted peaks from our KDE: this will be an array of BUR values
        return np.sort(x[peaks].flatten())[0]

    def _create_plot(self) -> None:
        """Create the main plot"""
        for (idx, grp), col in zip(self.df.groupby('instr', sort=False), vutils.RGB):
            vals = grp['asynchrony_adjusted_offset'].values
            x, y = self._kde(vals)
            peaks = np.mean(vals)
            self.ax.axvline(peaks, 0, 1, color=col, **self.VLINE_KWS)
            self.ax.plot(x, y, color=col, **self.PLOT_KWS)
            self.ax.fill_between(x, y, color=col, **self.FILL_KWS)

    def _format_ax(self) -> None:
        """Format axis-level parameters"""
        self.ax.xaxis.grid(True, **vutils.GRID_KWS)
        self.ax.set(
            xlim=(-1/32 - 0.01, 1/16 + 0.001), ylim=(0, 1.01), xticks=[-1/32, -1/64, 0, 1/64, 1/32, 1/16],
            ylabel='Density', xlabel='Relative position ($\pm0=$ drums, mean beat 1)',
            xticklabels=[
                r'–$\frac{1}{32}$', r'–$\frac{1}{64}$', r'$\pm$0',
                r'+$\frac{1}{64}$', r'+$\frac{1}{32}$', r'+$\frac{1}{16}$'
            ],
        )
        plt.setp(self.ax.spines.values(), linewidth=vutils.LINEWIDTH)
        self.ax.tick_params(axis='both', bottom=True, top=True, width=vutils.TICKWIDTH)
        self.ax.text(0, 1.1, r'$\pm$0', ha='center', va='center', clip_on=False, zorder=1000)
        self._add_images()

    def _add_images(self) -> None:
        """Add notation images into the plot"""
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
                    mpl.offsetbox.OffsetImage(img, clip_on=False, zoom=0.5), (1/val, 1.15),
                    frameon=False, xycoords='data', clip_on=False, annotation_clip=False, zorder=0
                 ))

    def _format_fig(self):
        """Format figure-level parameters"""
        self.fig.subplots_adjust(left=0.1, right=0.95, top=0.85, bottom=0.175)


class RegPlotPianistAsynchrony(vutils.BasePlot):
    """Create a regression plot showing association between accompaniment-soloist asynchrony and tempo"""
    # Disable convergence and user warnings here, raised when the model is created with bootstrapping
    warnings.simplefilter('ignore', ConvergenceWarning)
    warnings.simplefilter('ignore', UserWarning)

    FORMULA = "diff ~ tempo_standard * C(instr, Treatment(reference='bass'))"
    RE_STRUCT = "0 + tempo_standard + C(instr, Treatment(reference='bass'))"
    N_BOOT = vutils.N_BOOT
    BIN_MULTIPLER = 1.5

    # These are keywords that we pass into our given plot types
    LINE_KWS = dict(lw=vutils.LINEWIDTH * 2, ls=vutils.LINESTYLE, zorder=5)
    FILL_KWS = dict(lw=0, ls=vutils.LINESTYLE, alpha=vutils.ALPHA, zorder=5)
    SCATTER_KWS = dict(
        hue_order=list(utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys())[1:],
        palette=vutils.RGB[1:], markers=['s', 'D'], s=40,
        edgecolor=vutils.BLACK, zorder=3, alpha=vutils.ALPHA * 2,
    )
    HIST_KWS = dict(
        kde=False, color=vutils.BLACK, alpha=vutils.ALPHA,
        lw=vutils.LINEWIDTH, ls=vutils.LINESTYLE
    )

    def __init__(self, async_df: pd.DataFrame, **kwargs):
        """Called when initialising the class"""
        self.corpus_title = 'corpus_chronology'
        # Initialise the base plot with our given kwargs
        super().__init__(figure_title=fr'asynchrony_plots\regplot_pianistasync_{self.corpus_title}', **kwargs)
        self.df = async_df
        self.md = self._mixedlm(self.df)
        self.fig, self.ax = plt.subplots(
            nrows=2, ncols=2, figsize=(vutils.WIDTH / 2, vutils.WIDTH / 2),
            gridspec_kw=dict(width_ratios=(11, 1), height_ratios=(1, 5)),
        )
        # The main ax for plotting the regression/scatter plot
        self.main_ax = self.ax[1, 0]
        # Marginal ax, for plotting histograms
        self.marginal_ax = np.array([self.ax[0, 0], self.ax[1, 1]])
        # Top right corner ax, which we can go ahead and disable
        self.ax[0, 1].axis('off')

    def _mixedlm(self, data):
        """Fit the regression model"""
        return smf.mixedlm(
            self.FORMULA,
            data=data,
            groups=data['bandleader'],
            re_formula=self.RE_STRUCT
        ).fit()

    def _get_line(self, md) -> list:
        """Create the line for the regression model, with interaction effects"""
        # Get our intercept and tempo parameters from the model
        intercept = md.params['Intercept']
        tempo = md.params['tempo_standard']
        # Get our bass parameters from the model
        is_bass = 0
        is_bass_tempo = 0
        # Get our drums parameters from the model
        is_drums = md.params["C(instr, Treatment(reference='bass'))[T.drums]"]
        is_drums_tempo = md.params["tempo_standard:C(instr, Treatment(reference='bass'))[T.drums]"]
        # This is the range of values we'll be iterating through, taken from the actual results
        low_ = int(np.floor(self.df['tempo'].min()))
        high_ = int(np.ceil(self.df['tempo'].max()))
        mean_, std_ = np.array(range(low_, high_)).mean(), np.array(range(low_, high_)).std()
        # Iterate through each BPM in our range
        for bpm in range(low_, high_):
            # Standardise the BPM (Z-score) according to the observed values
            bpm_z = (bpm - mean_) / std_
            tempo_coeff = tempo * bpm_z
            # Iterate through each instrument and both coefficients
            for instr_, coeff_, interact_ in zip(
                    ['bass', 'drums'], [is_bass, is_drums], [is_bass_tempo, is_drums_tempo]
            ):
                # Construct the BUR value by following the regression equation
                diff = intercept + tempo_coeff + coeff_ + (interact_ * bpm_z)
                # Yield a dictionary of the results
                yield dict(tempo=bpm, tempo_std=bpm_z, instr=instr_, diff=diff)

    def _get_bootstrapped_sample(self) -> list:
        """Returns bootstrapped samples of the full dataset"""
        def bootstrap(state: int):
            """Bootstrapping function"""
            # Take a random sample of bandleaders and iterate through each
            for _, leader in bandleaders.sample(frac=1, replace=True, random_state=state).items():
                # Get all the data belonging to each bandleader
                yield self.df[self.df['bandleader'] == leader]

        # These are the names of all bandleaders
        bandleaders = pd.Series(self.df['bandleader'].unique())
        for i in range(self.N_BOOT):
            # Print the current iteration to act as a log
            print(i)
            # Return each bootstrapped sample as a single dataframe
            yield pd.concat(bootstrap(i), axis=0)

    def _format_bootstrap_lines(self, boot_models: list) -> list:
        """Formats data from a series of bootstrapped models into one dataframe of errors"""
        # Get a straight line for each bootstrapped model and combine into one dataframe
        big = pd.concat([pd.DataFrame(self._get_line(boot)) for boot in boot_models], axis=1)
        # Iterate through each tempo value
        for idx, row in big.iterrows():
            # Return a dictionary of results
            yield dict(
                tempo=row['tempo'].iloc[0],
                instr=row['instr'].iloc[0],
                low_ci=np.percentile(row['diff'], 2.5),
                high_ci=np.percentile(row['diff'], 97.5)
            )

    def _create_main_plot(self) -> None:
        """Creates the main axis plot"""
        # Get the line for the actual data
        line_df = pd.DataFrame(self._get_line(self.md))
        # Bootstrap to get random samples, replacement unit is bandleader
        boot_samples = self._get_bootstrapped_sample()
        # Create model for each sample of data
        boot_mds = [self._mixedlm(sample) for sample in boot_samples]
        # Convert all bootstrapped models into one single dataframe of errors
        boot_lines = pd.DataFrame(self._format_bootstrap_lines(boot_mds))
        # Iterate through each instrument and line color
        for instr_, col_ in zip(['bass', 'drums'], vutils.RGB[1:]):
            # First temporary dataframe: our actual data for this instrument
            temp_ = line_df[line_df['instr'] == instr_]
            # Plot the actual data
            self.main_ax.plot(temp_['tempo'], temp_['diff'], color=col_, **self.LINE_KWS)
            # Second temporary dataframe: our bootstrapped data for this instrument
            temp_boot_ = boot_lines[boot_lines['instr'] == instr_]
            # Fill between the low and high bounds
            self.main_ax.fill_between(
                temp_boot_['tempo'], temp_boot_['low_ci'], temp_boot_['high_ci'], color=col_, **self.FILL_KWS
            )
        sns.scatterplot(
            data=self.df, x='tempo', y='diff', hue='instr', style='instr', ax=self.main_ax, **self.SCATTER_KWS
        )

    def _create_marginal_plots(self):
        """Plots histograms and density estimates onto the marginal axis"""
        # Top marginal plot
        sns.histplot(
            data=self.df, x='tempo', ax=self.marginal_ax[0],
            bins=int(vutils.N_BINS * self.BIN_MULTIPLER),  **self.HIST_KWS
        )
        # Right marginal plot
        sns.histplot(
            data=self.df, y='diff', ax=self.marginal_ax[1],
            bins=int(vutils.N_BINS / self.BIN_MULTIPLER),  **self.HIST_KWS
        )

    def _create_plot(self) -> None:
        """Creates both main and marginal axis"""
        self._create_main_plot()
        self._create_marginal_plots()

    def _format_main_ax(self):
        """Sets axis-level parameters for the main plot"""
        # Add a grid onto the plot
        self.main_ax.grid(visible=True, axis='both', which='major', zorder=0, **vutils.GRID_KWS)
        # Get our legend handles, and set their edge color to black
        hand, _ = self.main_ax.get_legend_handles_labels()
        for ha in hand:
            ha.set_edgecolor(vutils.BLACK)
        # Remove the old legend, then add the new one on
        self.main_ax.get_legend().remove()
        self.main_ax.legend(
            hand, ['Bass', 'Drums'], loc='upper left', title='Instrument', frameon=True, framealpha=1,
            edgecolor=vutils.BLACK
        )
        # Final attributes to set here
        self.main_ax.set(
            xticks=[100, 150, 200, 250, 300], yticks=[-1/128, 0, 1/128, 1/64, 1/32], xlim=(100, 335),
            yticklabels=[r'–$\frac{1}{128}$', r'$\pm$0', r'+$\frac{1}{128}$', r'+$\frac{1}{64}$', r'+$\frac{1}{32}$'],
            xlabel='Tempo (BPM)', ylim=(-1/128 - 0.003, 1/32 + 0.0065),
            ylabel='Piano asynchrony (proportion of measure)',

        )
        plt.setp(self.main_ax.spines.values(), linewidth=vutils.LINEWIDTH)
        self.main_ax.axhline(0, 0, 1, lw=vutils.LINEWIDTH, ls=vutils.LINESTYLE, color=vutils.BLACK)
        self.main_ax.tick_params(axis='both', bottom=True, right=True, width=vutils.TICKWIDTH)
        self.main_ax.text(330, 0, r'$\pm$0', ha='right', va='center', clip_on=False, zorder=1000)
        for pict, v in zip(self._add_notation_vals(), ['–', '+', '+', '+']):
            self.main_ax.add_artist(pict)
            self.main_ax.text(pict.xy[0] - 1, pict.xy[1], v, ha='right', va='center', clip_on=False, zorder=1000)

    def _format_marginal_ax(self):
        """Formats axis-level properties for marginal axis"""
        # Remove correct spines from marginal axis
        for spine, ax in zip(['left', "bottom"], self.marginal_ax.flatten()):
            ax.spines[[spine, 'right', 'top']].set_visible(False)
            ax.tick_params(axis='both', width=vutils.TICKWIDTH)
            plt.setp(ax.spines.values(), linewidth=vutils.LINEWIDTH)
        # Set other features for the main axis
        self.marginal_ax[0].set(
            xlabel='', ylabel='', yticks=[0], yticklabels=[''], xticklabels=[], xlim=(100, 335),
            xticks=[100, 150, 200, 250, 300]
        )
        self.marginal_ax[1].set(
            xlabel='', ylabel='', xticks=[0], xticklabels=[''], yticklabels=[], ylim=(-1/128 - 0.003, 1/32 + 0.0065),
            yticks=[-1/128, 0, 1/128, 1/64, 1/32],
        )

    def _format_ax(self):
        """Sets axis-level parameters for both main and marginal axis"""
        self._format_main_ax()
        self._format_marginal_ax()

    @staticmethod
    def _add_notation_vals() -> list:
        """Adds notation values into the plot"""
        for val in [-128, 128, 64, 32]:
            try:
                img = plt.imread(fr'{utils.get_project_root()}\references\images\notation\notation_{abs(val)}.png')
            except FileNotFoundError:
                pass
            # If we can get the image, then yield it to add to our plot
            else:
                yield mpl.offsetbox.AnnotationBbox(
                    mpl.offsetbox.OffsetImage(img, clip_on=False, zoom=0.4), (325, 1/val),
                    frameon=False, xycoords='data', clip_on=False, annotation_clip=False
                 )

    def _format_fig(self) -> None:
        """Format figure-level properties"""
        self.fig.subplots_adjust(left=0.11, right=0.99, top=0.99, bottom=0.09, hspace=0.1, wspace=0.05)


class HistPlotProportionalAsynchronyTriosPiano(vutils.BasePlot):
    """Creates density plots for each pianist, showing asynchrony with both accompanying instruments"""
    img_loc = fr'{utils.get_project_root()}/references/images/musicians'
    PLOT_KWS = dict(lw=vutils.LINEWIDTH, ls=vutils.LINESTYLE, zorder=10)
    FILL_KWS = dict(alpha=0.1, zorder=5)
    VLINE_KWS = dict(linestyle='dashed', alpha=1, zorder=1, linewidth=vutils.LINEWIDTH * 1.5)
    XTICKS = [-1 / 16, -1 / 32, -1 / 64, 0, 1 / 64, 1 / 32, 1 / 16]
    XTICKLABELS = [f'{round(i * 400, 1)}%' if i != 0 else r'$\pm$0%' for i in XTICKS]

    def __init__(self, async_df: pd.DataFrame, **kwargs):
        """Called when initialising the class"""
        self.corpus_title = 'corpus_updated'
        self.include_images = kwargs.get('include_images', True)
        fig_title = fr'asynchrony_plots/histplot_asynchronytriospiano_{self.corpus_title}'
        if not self.include_images:
            fig_title += '_no_images'
        # Initialise the base plot with our given kwargs
        super().__init__(figure_title=fig_title, **kwargs)
        self.df = async_df.copy(deep=True)
        order = reversed(
            self.df[self.df['instr'] == 'drums']
            .groupby('bandleader', as_index=False)
            ['asynchrony']
            .mean()
            .sort_values(by='asynchrony')
            ['bandleader']
            .values
        )
        self.df = self.df.set_index('bandleader').loc[order].reset_index(drop=False)
        self.fig, self.ax = plt.subplots(
            async_df['bandleader'].nunique(), 2, figsize=(vutils.WIDTH, vutils.WIDTH / 2), sharex=True, sharey=False,
            gridspec_kw=dict(height_ratios=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        )

    @staticmethod
    def _kde(vals: np.array, bw_adjust: float = 1.2) -> tuple:
        """Fit the KDE to the data and evaluate on a linear space of integers, then scale between 0 and 1"""
        # Fit the actual KDE to the data, using the default parameters
        kde = stats.gaussian_kde(vals.T, bw_method='scott')
        kde.set_bandwidth(kde.factor * bw_adjust)
        # Create a linear space of integers ranging from our lowest to our highest BUR
        x = np.linspace(vals.min(), vals.max(), 100)[:, np.newaxis].T[0]
        # Evaluate the KDE on our linear space of integers
        y = kde.evaluate(x)
        y = np.array([(y_ - min(y)) / (max(y) - min(y)) for y_ in y])
        return x, y

    def _create_plot(self) -> None:
        """Create the main plot"""
        for (idx, grp), ax_row in zip(self.df.groupby('bandleader', sort=False), self.ax):
            zi = zip(grp.groupby('instr', sort=False), [vutils.GREEN, vutils.BLUE], ax_row, ['s', 'D'])
            for (i, g), col, a, mark in zi:
                vals = g['asynchrony'].values
                x, y = self._kde(vals)
                a.plot(x, y, color=col, **self.PLOT_KWS, label=i.title())
                a.fill_between(x, y, color=col, **self.FILL_KWS)
                me = np.mean(vals)
                std = np.std(vals)
                a.scatter(
                    me, 0, color=col, marker=mark, lw=vutils.LINEWIDTH / 2, edgecolor=vutils.BLACK, s=50, zorder=15
                )
                a.errorbar(
                    x=[me, me], y=[0, 0], xerr=std, zorder=10, linewidth=vutils.LINEWIDTH * 1.5,
                    color=col, capsize=5, capthick=vutils.LINEWIDTH * 1.5
                )

    def _add_bandleader_images(self, bl: str, ax: plt.Axes, y: float = 0.5, ) -> None:
        """Adds images corresponding with each bandleader `bl` to the provided axis object `ax`"""
        fpath = fr'{self.img_loc}/{bl.replace(" ", "_").lower()}.png'
        img = mpl.offsetbox.OffsetImage(
            plt.imread(fpath), clip_on=False, transform=ax.transAxes, zoom=0.5
        )
        ab = mpl.offsetbox.AnnotationBbox(
            img, (-1 / 16 - 0.02, y), xycoords='data', clip_on=False, transform=ax.transAxes,
            annotation_clip=False, bboxprops=dict(edgecolor='none', facecolor='none')
        )
        ax.add_artist(ab)

    def _add_notation_images(self, y: float = 5.7) -> None:
        """Adds images corresponding with notation values to the plot"""
        # ax = self.ax[1]
        for ax in self.ax[1]:
            ax.text(0, y - .25, r'$\pm$0', ha='center', va='center', clip_on=False, zorder=1000)
            for val in [-16, -32, -64, 32, 64, 16]:
                try:
                    img = plt.imread(fr'{utils.get_project_root()}/references/images/notation/notation_{abs(val)}.png')
                except FileNotFoundError:
                    pass
                # If we can get the image, then yield it to add to our plot
                else:
                    ax.text(
                        1 / val - 0.001, y, '–' if val < 0 else '+', ha='right', va='center', clip_on=False, zorder=1000
                    )
                    ax.add_artist(mpl.offsetbox.AnnotationBbox(
                        mpl.offsetbox.OffsetImage(img, clip_on=False, zoom=0.75), (1 / val, y + .25),
                        frameon=False, xycoords='data', clip_on=False, annotation_clip=False, zorder=0
                    ))

    def _format_ax(self) -> None:
        """Format axis-level parameters"""
        self._add_notation_images()
        for num, ((i, g), ax_row) in enumerate(zip(self.df.groupby('bandleader', sort=False), self.ax)):
            for (idx, grp), ax, col in zip(g.groupby('instr', sort=False), ax_row, [vutils.GREEN, vutils.BLUE]):
                if num == 0:
                    spines = ['bottom']
                    ax.set_title(f'Piano→{idx.title()}', y=2.3, zorder=10000)
                    yl = (-0.35, 2)
                elif num == 9:
                    spines = ['top']
                    yl = (-0.7, 1.65)
                else:
                    spines = ['bottom', 'top']
                    yl = (-0.35, 2)
                # ax.text(-1/16 + 0.001, yl[1] - 0.2, '$N$ = ', va='top')
                # ax.text(-1/16 + 0.012, yl[1] - 0.2, len(grp), va='top', ha='left', color=col)
                ax.spines[spines].set_visible(False)
                ax.axvline(0, 0, 1, color=vutils.BLACK, linewidth=vutils.LINEWIDTH, linestyle=vutils.LINESTYLE)
                ax.axhline(
                    0, 0, 1, color=vutils.BLACK, linewidth=vutils.LINEWIDTH, linestyle=vutils.LINESTYLE, zorder=0
                )
                plt.setp(ax.spines.values(), linewidth=vutils.LINEWIDTH)
                ax.tick_params(
                    axis='both', bottom=True, right=True, width=vutils.TICKWIDTH,
                    top=True if num == 0 else False, left=True,
                )
                ax.xaxis.grid(True, color=vutils.BLACK, alpha=1, lw=2, ls='dashed', zorder=1)
                ax.tick_params(axis='y', which='major', pad=60)
                lab = ''
                if idx == 'bass':
                    y = 0.5 if num != 9 else 0
                    if self.include_images:
                        self._add_bandleader_images(i, ax, y)
                    lab = i
                ax.set(
                    xlim=(-1 / 16 - 0.01, 1 / 16 + 0.01), ylim=yl, yticks=[0], yticklabels=[lab], xticks=self.XTICKS,
                    xticklabels=self.XTICKLABELS
                )
                if num != 9:
                    plt.setp(ax.yaxis.get_majorticklabels(), va="bottom")

    def _format_fig(self) -> None:
        """Format figure-level parameters"""
        self.fig.supxlabel('Relative position of piano (% of quarter note duration)', x=0.55)
        self.fig.supylabel('Pianist')
        # Adjust subplot positioning
        self.fig.subplots_adjust(left=0.2, right=0.95, top=0.85, bottom=0.1, hspace=0, wspace=0.05)


class ScatterPlotAsynchronyTrack(vutils.BasePlot):
    """Creates a scatter plot for all onset values within a given track, similar to those in `OnsetSync` R package"""
    wraparound = 0.9

    def __init__(self, onset_maker, **kwargs):
        self.onset_maker = onset_maker
        self.fname = kwargs.get('figure_title', rf'onsets_plots\scatterplot_bybeat_{self.onset_maker.item["mbz_id"]}')
        self.time_sig = self.onset_maker.item['time_signature']
        super().__init__(figure_title=self.fname, **kwargs)
        self.df = pd.DataFrame(self.format_df())
        self.cmap = vutils.RGB
        self.fig, self.ax = plt.subplots(
            nrows=1,
            ncols=len(utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys()),
            sharex=True,
            sharey=True,
            figsize=(vutils.WIDTH, vutils.WIDTH / 3)
        )

    def format_df(self) -> list:
        """Formats provided onset maker into correct dataframe format for plotting"""
        for instr in utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys():
            instr_ons = self.onset_maker.ons[instr]
            z = zip(self.onset_maker.ons['mix'], self.onset_maker.ons['mix'][1:], self.onset_maker.ons['metre_manual'])
            for beat1, beat2, beat1pos in z:
                vals = instr_ons[np.logical_and(beat1 <= instr_ons, instr_ons < beat2)]
                for i in vals:
                    pos = ((i - beat1) / (beat2 - beat1)) + beat1pos
                    yield {
                        'instrument': instr,
                        'timestamp': pd.to_datetime(datetime.fromtimestamp(beat1).strftime('%H:%M:%S')),
                        'musical_position': pos if pos < self.time_sig + self.wraparound else pos - self.time_sig
                    }

    def _create_plot(self) -> None:
        """Creates main plot: scatter plot of each instrument"""
        for ax, instr, col in zip(self.ax.flatten(), utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys(), self.cmap):
            g = sns.scatterplot(data=self.df[self.df['instrument'] == instr], x='musical_position', y='timestamp', ax=ax, color=col, s=40, legend=None)
            g.set_title(instr.title())

    def _format_ax(self) -> None:
        """Formats axis-level parameters"""
        minor_ticks = [i + f for i in range(1, 5) for f in (1 / 3, 2 / 3)]
        for ax in self.ax.flatten():
            ax.set(xlim=(0.8, self.time_sig + 0.9), xticks=list(range(1, self.time_sig +1)), xlabel='', ylabel='')
            ax.set_xticks(minor_ticks, labels=[], minor=True)
            ax.grid(which='major', ls='-', lw=1)
            ax.grid(which='minor', ls='--', lw=0.3)
            plt.setp(ax.spines.values(), linewidth=2)
            ax.tick_params(axis='both', width=2)

    def _format_fig(self) -> None:
        """Formats figure-level parameters"""
        self.fig.supxlabel(r'Beat ($\frac{1}{4}$ note)')
        self.fig.supylabel('Time (s)')
        self.fig.suptitle(self.onset_maker.item['fname'])
        self.fig.subplots_adjust(left=0.11, right=0.95, top=0.85, bottom=0.15)


if __name__ == '__main__':
    pass
