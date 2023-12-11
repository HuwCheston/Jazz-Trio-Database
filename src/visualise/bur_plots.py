#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Classes used for plotting beat-upbeat ratios"""

import warnings
from typing import Generator

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning

import src.visualise.visualise_utils as vutils
from src import utils

__all__ = ['BarPlotBUR', 'HistPlotBURByInstrument', 'RegPlotBURTempo', 'ViolinPlotBURs']

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
    BURS_WITH_IMAGES = [0.5, 1, 2, 3]
    img_loc = fr'{utils.get_project_root()}\references\images\musicians'
    PAL = sns.cubehelix_palette(dark=1/3, gamma=.3, light=2/3, start=0, n_colors=20, as_cmap=False)
    VP_KWS = dict(vert=False, showmeans=False, showextrema=False)
    EBAR_KWS = dict(
        ls='none', color=vutils.RED, linewidth=vutils.LINEWIDTH * 1.5, capsize=5,
        capthick=vutils.LINEWIDTH * 1.5, ecolor=vutils.RED, zorder=10
    )
    SCAT_KWS = dict(facecolor=vutils.RED, lw=vutils.LINEWIDTH / 2, edgecolor=vutils.BLACK, s=100, zorder=15)

    def __init__(self, bur_df: pd.DataFrame, **kwargs):
        self.corpus_title = kwargs.get('corpus_title', 'corpus')
        super().__init__(figure_title=fr'bur_plots\violinplot_burs_{self.corpus_title}', **kwargs)
        self.df = bur_df[bur_df['instrument'] == 'piano'].copy()
        order = reversed(
            self.df.groupby('bandleader', as_index=False)
            ['bur']
            .mean()
            .sort_values(by='bur')
            ['bandleader']
            .values
        )
        self.df = self.df.set_index('bandleader').loc[order].reset_index(drop=False)
        self.fig, self.ax = plt.subplots(nrows=1, ncols=1, figsize=(vutils.WIDTH, vutils.WIDTH / 2))

    def add_bur_images(self, y):
        """Adds images for required BUR values"""
        # Iterate through all of our BUR values
        for x in self.BURS_WITH_IMAGES:
            # Add a dotted vertical line to this BUR value
            self.ax.axvline(np.log2(x), ymin=-0.5, ymax=9, color=vutils.BLACK, alpha=1, lw=2, ls='dashed', zorder=1)
            # Try and get the image of the notation type for this BUR value
            try:
                img = plt.imread(fr'{utils.get_project_root()}\references\images\bur_notation\bur_{x}.png')
            except FileNotFoundError:
                pass
            # If we can get the image, then yield it to add to our plot
            else:
                yield mpl.offsetbox.AnnotationBbox(
                    mpl.offsetbox.OffsetImage(img, clip_on=False), (np.log2(x), y),
                    frameon=False, xycoords='data', clip_on=False, annotation_clip=False
                )

    def _add_bandleader_images(self, bl, y):
        fpath = fr'{self.img_loc}\{bl.replace(" ", "_").lower()}.png'
        img = mpl.offsetbox.OffsetImage(
            plt.imread(fpath), clip_on=False, transform=self.ax.transAxes, zoom=0.5
        )
        ab = mpl.offsetbox.AnnotationBbox(
            img, (-2.15, y - 0.05), xycoords='data', clip_on=False, transform=self.ax.transAxes,
            annotation_clip=False, bboxprops=dict(edgecolor='none', facecolor='none')
        )
        self.ax.add_artist(ab)

    def _create_plot(self) -> None:
        """Creates violinplot in seaborn"""
        self.g = sns.violinplot(
            data=self.df, x='bur', y='bandleader', linecolor=vutils.BLACK, density_norm='count', hue=True,
            palette=self.PAL, hue_order=[True, False], split=True, legend=False, inner=None, ax=self.ax, bw=0.1,
        )
        for patch, col in zip(self.ax.collections, self.PAL):
            patch.set_facecolor(col)
        med = self.df.groupby('bandleader', as_index=False, sort=False).agg({'bur': ['mean', 'std']})
        med.columns = [col[0] if col[1] == '' else col[1] for col in med.columns]
        for line in self.ax.lines:
            line.set_linestyle(vutils.LINESTYLE)
            line.set_color(vutils.BLACK)
            line.set_alpha(0.8)
        for collect in self.ax.collections:
            collect.set_edgecolor(vutils.BLACK)
            collect.set_linewidth(vutils.LINEWIDTH)
        self.ax.scatter(med['mean'], med['bandleader'], **self.SCAT_KWS)
        for idx, row in med.iterrows():
            self.ax.errorbar(x=[row['mean'], row['mean']], y=[idx, idx], xerr=[row['std'], row['std']], **self.EBAR_KWS)

    def _add_nburs_to_tick(self):
        """Add the total number of BURs gathered for each musician next to their name"""
        for num, (idx, grp) in enumerate(self.df.groupby('bandleader', sort=False)):
            self.ax.text(-1.95, num - 0.15, f'$N$ = ')
            self.ax.text(-1.805, num - 0.15, len(grp['bur'].dropna()), color=vutils.RED)

    def _format_ax(self) -> None:
        """Format axis-level properties"""
        # Here we set the line styles
        self.ax.get_legend().remove()
        # Add in a horizontal line for each performer on the y-axis
        for tick in self.ax.get_yticks():
            self.ax.axhline(tick, 0, 3.25, color=vutils.BLACK, alpha=vutils.ALPHA, lw=vutils.LINEWIDTH)
        # Add in notation images for each of the BUR values we want to the top of the plot
        for artist in self.add_bur_images(y=-1.3):
            self.ax.add_artist(artist)
        # Set final properties
        self._add_nburs_to_tick()
        self.ax.set(
            xticks=[np.log2(b) for b in self.BURS_WITH_IMAGES], xlabel='', ylabel='',
            xticklabels=[-1, 0, 1, 1.585], xlim=(-2, 2), ylim=(9.5, -0.5),
        )
        self.ax.tick_params(axis='y', which='major', pad=70)
        for num, pi in enumerate(self.df['bandleader'].unique()):
            self._add_bandleader_images(pi, num)

    def _format_fig(self) -> None:
        """Format figure-level properties"""
        self.fig.supxlabel('${Log_2}$ beat-upbeat ratio', x=0.55)
        self.fig.supylabel('Pianist')
        # Adjust line and tick width
        plt.setp(self.ax.spines.values(), linewidth=vutils.LINEWIDTH)
        self.ax.tick_params(axis='both', width=vutils.TICKWIDTH, labeltop=True)
        # Adjust subplot positioning
        self.fig.subplots_adjust(left=0.205, right=0.95, top=0.875, bottom=0.08)


class HistPlotBURByInstrument(vutils.BasePlot):
    BURS_WITH_IMAGES = [0.5, 1, 2, 3]
    HIST_KWS = dict(lw=vutils.LINEWIDTH / 2, ls=vutils.LINESTYLE, zorder=2, align='edge')
    KDE_KWS = dict(linestyle=vutils.LINESTYLE, alpha=1, zorder=3, linewidth=vutils.LINEWIDTH)
    VLINE_KWS = dict(color=vutils.BLACK, linestyle='dashed', alpha=1, zorder=4, linewidth=vutils.LINEWIDTH * 1.5)
    mpl.rcParams.update(mpl.rcParamsDefault)

    def __init__(self, bur: pd.DataFrame, peaks: pd.DataFrame, **kwargs):
        self.corpus_title = 'corpus_chronology'
        super().__init__(figure_title=fr'bur_plots\histplot_bursbyinstrumentgmm_{self.corpus_title}', **kwargs)
        self.bur_df = bur
        self.peak_df = peaks
        self.fig, self.ax = plt.subplots(
            nrows=1, ncols=3, figsize=(vutils.WIDTH, vutils.WIDTH / 3), sharex=True, sharey=True
        )

    def add_bur_images(self, y):
        """Adds images for required BUR values"""
        # Iterate through all of our BUR values
        for x in self.BURS_WITH_IMAGES:
            # Try and get the image of the notation type for this BUR value
            try:
                img = plt.imread(fr'{utils.get_project_root()}\references\images\bur_notation\bur_{x}.png')
            except FileNotFoundError:
                pass
            # If we can get the image, then yield it to add to our plot
            else:
                yield mpl.offsetbox.AnnotationBbox(
                    mpl.offsetbox.OffsetImage(img, clip_on=False, zoom=0.5), (np.log2(x), y),
                    frameon=False, xycoords='data', clip_on=False, annotation_clip=False
                 )

    @staticmethod
    def _kde(data, len_data: int = 1000):
        # Fit the actual KDE to the data, using the default parameters
        kde = stats.gaussian_kde(data.T)
        # Create a linear space of integers ranging from our lowest to our highest BUR
        data_plot = np.linspace(data.min(), data.max(), len_data)[:, np.newaxis]
        # Evaluate the KDE on our linear space of integers
        y = kde.evaluate(data_plot.T)
        return data_plot, np.array([(y_ - min(y)) / (max(y) - min(y)) for y_ in y])

    def _create_plot(self) -> None:
        """Creates the histogram and kde plots"""
        for ax, (idx, grp), col in zip(self.ax.flatten(), self.bur_df.groupby('instrument', sort=False), vutils.RGB):
            ax.set_xscale('symlog', base=2)
            ax.grid(zorder=0, axis='x', **vutils.GRID_KWS)
            heights, edges = np.histogram(grp['bur'], bins=vutils.N_BINS)
            heights = heights / max(heights)
            # Plot the normalized histogram
            self.HIST_KWS.update(dict(x=edges[:-1], height=heights, width=np.diff(edges)))
            ax.bar(fc=col, edgecolor='None', alpha=vutils.ALPHA, **self.HIST_KWS)
            ax.bar(fc='None', edgecolor=vutils.BLACK, alpha=1, **self.HIST_KWS)
            # # Plot the kde
            x, y = self._kde(grp['bur'])
            ax.plot(x, y, color=col, label=f'{idx.title()}\n({len(grp)})', **self.KDE_KWS)
            self._add_peaks(ax, idx)

    def _add_peaks(self, ax, ins):
        ps = self.peak_df[self.peak_df['instrument'] == ins]
        for _, peak in ps.iterrows():
            ax.axvline(peak['peak'], 0, 1, **self.VLINE_KWS)

    def _format_ax(self) -> None:
        """Formats axis-level properties"""
        # Add images for each BUR value we want to plot
        hands, labs = [], []
        for ax, name in zip(self.ax.flatten(), utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys()):
            for artist in self.add_bur_images(y=1.15):
                ax.add_artist(artist)
            ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
            ax.get_xaxis().set_minor_formatter(mpl.ticker.NullFormatter())
            ax.set(
                xticks=[np.log2(i) for i in self.BURS_WITH_IMAGES], xticklabels=[-1, 0, 1, 1.58], xlabel='', ylabel='',
                xlim=(np.log2(LOW_BUR_CUTOFF), np.log2(HIGH_BUR_CUTOFF)), yticks=np.linspace(0, 1, 5)
            )
            ax.set_title(name.title(), y=1.125)
            plt.setp(ax.spines.values(), linewidth=vutils.LINEWIDTH)
            ax.tick_params(axis='both', top=True, bottom=True, labeltop=False, labelbottom=True, width=vutils.TICKWIDTH)
            hand, lab = ax.get_legend_handles_labels()
            hands.extend(hand)
            labs.extend(lab)

    def _format_fig(self) -> None:
        """Formats figure-level properties"""
        self.fig.supxlabel('${Log_2}$ beat-upbeat ratio')
        self.fig.supylabel('Density', x=0.01)
        self.fig.subplots_adjust(left=0.075, bottom=0.12, right=0.95, top=0.825)


class BarPlotBUR(vutils.BasePlot):
    BURS_WITH_IMAGES = [1, 2]
    BAR_KWS = dict(
        dodge=False, edgecolor=vutils.BLACK, errorbar=('ci', 95),
        lw=vutils.LINEWIDTH, seed=42, capsize=0.1, width=0.8,
        ls=vutils.LINESTYLE, estimator=np.mean,
        errcolor=vutils.BLACK, zorder=1,
        hue_order=utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys(),
        palette=vutils.RGB, alpha=0.8
    )

    def __init__(self, bur_df, **kwargs):
        self.corpus_title = 'corpus_chronology'
        super().__init__(figure_title=fr'bur_plots\barplot_mean_bur_{self.corpus_title}', **kwargs)
        self.df = bur_df
        self.fig, self.ax = plt.subplots(nrows=1, ncols=1, figsize=(vutils.WIDTH / 2, vutils.WIDTH / 2))

    def _create_plot(self):
        sns.barplot(data=self.df, x='instrument', y='bur', **self.BAR_KWS)

    def add_bur_images(self, y):
        """Adds images for required BUR values"""
        # Iterate through all of our BUR values
        for x in self.BURS_WITH_IMAGES:
            # Try and get the image of the notation type for this BUR value
            try:
                img = plt.imread(fr'{utils.get_project_root()}\references\images\bur_notation\bur_{x}.png')
            except FileNotFoundError:
                pass
            # If we can get the image, then yield it to add to our plot
            else:
                yield mpl.offsetbox.AnnotationBbox(
                    mpl.offsetbox.OffsetImage(img, clip_on=False, zoom=0.5), (y, np.log2(x)),
                    frameon=False, xycoords='data', clip_on=False, annotation_clip=False
                 )

    def _format_ax(self):
        # Set the width of the edges and ticks
        for artist in self.add_bur_images(y=2.65):
            self.ax.add_artist(artist)
        for hatch, bar in zip(vutils.HATCHES, self.ax.patches):
            bar.set_hatch(hatch)
        self.ax.grid(zorder=0, axis='y', **vutils.GRID_KWS)
        plt.setp(self.ax.spines.values(), linewidth=vutils.LINEWIDTH, color=vutils.BLACK)
        self.ax.tick_params(axis='both', width=vutils.TICKWIDTH, color=vutils.BLACK)
        self.ax.set(
            ylabel='Mean ${Log_2}$ BUR',
            xticklabels=[i.title() for i in utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys()],
            xlabel='Instrument',
            ylim=(-0.1, 1.1),
            yticks=[np.log2(i) for i in self.BURS_WITH_IMAGES]
        )


class RegPlotBURTempo(vutils.BasePlot):
    """Creates a graph showing tempo vs mean BUR, with marginal distributions"""
    # Disable convergence and user warnings here, raised when the model is created with bootstrapping
    warnings.simplefilter('ignore', ConvergenceWarning)
    warnings.simplefilter('ignore', UserWarning)
    # Initial attributes for plotting
    BURS_WITH_IMAGES = [0.5, 1, 2]
    BUR_THRESHOLD = 15
    N_BOOT = 10
    BIN_MULTIPLER = 1.5
    # These are keywords that we pass into our given plot types
    LINE_KWS = dict(lw=vutils.LINEWIDTH * 2, ls=vutils.LINESTYLE)
    FILL_KWS = dict(lw=0, ls=vutils.LINESTYLE, alpha=vutils.ALPHA)
    SCATTER_KWS = dict(
        hue_order=utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys(),
        palette=vutils.RGB, markers=['o', 's', 'D'], s=40,
        edgecolor=vutils.BLACK, zorder=1
    )
    HIST_KWS = dict(
        kde=False, color=vutils.BLACK, alpha=vutils.ALPHA,
        lw=vutils.LINEWIDTH, ls=vutils.LINESTYLE
    )
    # Attributes for our model
    MODEL = "bur_mean ~ tempo_standard * C(instrument_, Treatment(reference='piano'))"
    RE_FORMULA = "0 + tempo_standard + C(instrument_, Treatment(reference='piano'))"

    def __init__(self, bur_df: pd.DataFrame, **kwargs):
        """Called when initialising the class"""
        self.corpus_title = 'corpus_chronology'
        # Initialise the base plot with our given kwargs
        super().__init__(figure_title=fr'bur_plots\regplot_burtempo_{self.corpus_title}', **kwargs)
        # Format the dataframe
        self.average = self._format_df(bur_df)
        # Create our initial model, using the actual data
        self.md = self._mixedlm(self.average)
        # Create our gridded plots
        self.fig, self.ax = plt.subplots(
            nrows=2, ncols=2, figsize=(vutils.WIDTH, vutils.WIDTH / 2),
            gridspec_kw=dict(width_ratios=(11, 1), height_ratios=(1, 5)),
        )
        # The main ax for plotting the regression/scatter plot
        self.main_ax = self.ax[1, 0]
        # Marginal ax, for plotting histograms
        self.marginal_ax = np.array([self.ax[0, 0], self.ax[1, 1]])
        # Top right corner ax, which we can go ahead and disable
        self.ax[0, 1].axis('off')

    def _format_df(self, bur_df: pd.DataFrame) -> pd.DataFrame:
        """Formats the dataframe of raw BUR values"""
        # Group by instrument and track, get the mean BUR value and number of BUR values
        clean = (
            bur_df.groupby(['instrument', 'mbz_id'])
            .agg(dict(bur=['mean', 'count'], tempo='median', bandleader='first'))
            .reset_index(drop=False)
        )
        # This resets the multi index and column names
        clean.columns = ['_'.join(col).strip() for col in clean.columns.values]
        # Drop BURs without enough values
        clean = clean[clean['bur_count'] > self.BUR_THRESHOLD]
        # Standardise the tempo into Z-scores and return
        clean['tempo_standard'] = (clean['tempo_median'] - clean['tempo_median'].mean()) / clean['tempo_median'].std()
        return clean

    def add_bur_images(self, y):
        """Adds images for required BUR values"""
        # Iterate through all of our BUR values
        for x in self.BURS_WITH_IMAGES:
            # Try and get the image of the notation type for this BUR value
            try:
                img = plt.imread(fr'{utils.get_project_root()}\references\images\bur_notation\bur_{x}.png')
            except FileNotFoundError:
                pass
            # If we can get the image, then yield it to add to our plot
            else:
                yield mpl.offsetbox.AnnotationBbox(
                    mpl.offsetbox.OffsetImage(img, clip_on=False, zoom=0.5), (y, np.log2(x)),
                    frameon=False, xycoords='data', clip_on=False, annotation_clip=False
                 )

    def _mixedlm(self, model_data: pd.DataFrame):
        """Creates a mixed effects model with given parameters from a dataset"""
        return smf.mixedlm(
            self.MODEL, data=model_data, groups=model_data['bandleader_first'], re_formula=self.RE_FORMULA
        ).fit()

    def _get_line(self, model):
        """Creates data for a straight line by predicting values from a mixed effects model"""
        # Get our intercept and tempo parameters from the model
        intercept = model.params['Intercept']
        tempo = model.params['tempo_standard']
        # Get our bass parameters from the model
        is_bass = model.params["C(instrument_, Treatment(reference='piano'))[T.bass]"]
        is_bass_tempo = model.params["tempo_standard:C(instrument_, Treatment(reference='piano'))[T.bass]"]
        # Get our drums parameters from the model
        is_drums = model.params["C(instrument_, Treatment(reference='piano'))[T.drums]"]
        is_drums_tempo = model.params["tempo_standard:C(instrument_, Treatment(reference='piano'))[T.drums]"]
        # Get our piano parameters from the model
        is_piano = 0
        is_piano_tempo = 0
        # This is the range of values we'll be iterating through, taken from the actual results
        low_ = int(np.floor(self.average['tempo_median'].min()))
        high_ = int(np.ceil(self.average['tempo_median'].max()))
        mean_, std_ = np.array(range(low_, high_)).mean(), np.array(range(low_, high_)).std()
        # Iterate through each BPM in our range
        for bpm in range(low_, high_):
            # Standardise the BPM (Z-score) according to the observed values
            bpm_z = (bpm - mean_) / std_
            tempo_coeff = tempo * bpm_z
            # Iterate through each instrument and both coefficients
            for instr_, coeff_, interact_ in zip(
                utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys(),
                [is_piano, is_bass, is_drums],
                [is_piano_tempo, is_bass_tempo, is_drums_tempo]
            ):
                # Construct the BUR value by following the regression equation
                bur_ = intercept + tempo_coeff + coeff_ + (interact_ * bpm_z)
                # Yield a dictionary of the results
                yield dict(tempo=bpm, tempo_std=bpm_z, instr=instr_, bur=bur_)

    def _format_bootstrap_lines(self, boot_models: list):
        """Formats data from a series of bootstrapped models into one dataframe of errors"""
        # Get a straight line for each bootstrapped model and combine into one dataframe
        big = pd.concat([pd.DataFrame(self._get_line(boot)) for boot in boot_models], axis=1)
        # Iterate through each tempo value
        for idx, row in big.iterrows():
            sem = stats.sem(row['bur'].to_numpy())
            # Get the standard error of the mean of the row
            # Return a dictionary of results
            yield dict(
                tempo=row['tempo'].iloc[0],
                instr=row['instr'].iloc[0],
                sem=sem,
                low_ci=np.percentile(row['bur'], 2.5),
                high_ci=np.percentile(row['bur'], 97.5)
            )

    def _get_bootstrapped_sample(self):
        """Returns bootstrapped samples of the full dataset"""
        def bootstrap(state: int):
            """Bootstrapping function"""
            # Take a random sample of bandleaders and iterate through each
            for _, leader in bandleaders.sample(frac=1, replace=True, random_state=state).items():
                # Get all the data belonging to each bandleader
                yield self.average[self.average['bandleader_first'] == leader]

        # These are the names of all bandleaders
        bandleaders = pd.Series(self.average['bandleader_first'].unique())
        for i in range(self.N_BOOT):
            # Print the current iteration to act as a log
            print(i)
            # Return each bootstrapped sample as a single dataframe
            yield pd.concat(bootstrap(i), axis=0)

    def _create_main_plot(self):
        """Plots regression and scatter plot onto the main axis, with bootstrapped errorbars"""
        # Get the line for the actual data
        line_df = pd.DataFrame(self._get_line(self.md))
        # Bootstrap to get random samples, replacement unit is bandleader
        boot_samples = self._get_bootstrapped_sample()
        # Create model for each sample of data
        boot_mds = [self._mixedlm(sample) for sample in boot_samples]
        # Convert all bootstrapped models into one single dataframe of errors
        boot_lines = pd.DataFrame(self._format_bootstrap_lines(boot_mds))
        # Iterate through each instrument and line color
        for instr_, col_ in zip(utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys(), vutils.RGB):
            # First temporary dataframe: our actual data for this instrument
            temp_ = line_df[line_df['instr'] == instr_]
            # Plot the actual data
            self.main_ax.plot(temp_['tempo'], temp_['bur'], color=col_, **self.LINE_KWS)
            # Second temporary dataframe: our bootstrapped data for this instrument
            temp_boot_ = boot_lines[boot_lines['instr'] == instr_]
            # Fill between the low and high bounds
            self.main_ax.fill_between(
                temp_boot_['tempo'], temp_boot_['low_ci'], temp_boot_['high_ci'], color=col_, **self.FILL_KWS
            )
        # Create the scatter plot
        sns.scatterplot(
            data=self.average, x='tempo_median', y='bur_mean', style='instrument_',
            ax=self.main_ax, hue='instrument_', **self.SCATTER_KWS
        )

    def _create_marginal_plots(self):
        """Plots histograms and density estimates onto the marginal axis"""
        # Top marginal plot
        sns.histplot(
            data=self.average, x='tempo_median', ax=self.marginal_ax[0],
            bins=int(vutils.N_BINS * self.BIN_MULTIPLER),  **self.HIST_KWS
        )
        # Right marginal plot
        sns.histplot(
            data=self.average, y='bur_mean', ax=self.marginal_ax[1],
            bins=int(vutils.N_BINS / self.BIN_MULTIPLER),  **self.HIST_KWS
        )

    def _create_plot(self):
        """Creates the main and marginal plots"""
        self._create_main_plot()
        self._create_marginal_plots()

    def _format_marginal_ax(self):
        """Formats axis-level properties for marginal axis"""
        # Remove correct spines from marginal axis
        for spine, ax in zip(['left', "bottom"], self.marginal_ax.flatten()):
            ax.spines[[spine, 'right', 'top']].set_visible(False)
        # Set other features for the main axis
        self.marginal_ax[0].set(
            xlabel='', ylabel='', yticks=[0], yticklabels=[''], xticklabels=[], xlim=(100, 310),
            xticks=[100, 150, 200, 250, 300]
        )
        self.marginal_ax[1].set(
            xlabel='', ylabel='', xticks=[0], xticklabels=[''], yticklabels=[], ylim=(-1.35, 1.7),
            yticks=[-1, 0, 1]
        )

    def _format_main_ax(self):
        """Formats axis-level properties for the main axis"""
        # Add BUR images onto the right-hand side of the main plot
        for artist in self.add_bur_images(y=315):
            self.main_ax.add_artist(artist)
        # Add a grid onto the plot
        self.main_ax.grid(visible=True, axis='both', which='major', zorder=0, **vutils.GRID_KWS)
        # Get our legend handles, and set their edge color to black
        hand, _ = self.main_ax.get_legend_handles_labels()
        for ha in hand:
            ha.set_edgecolor(vutils.BLACK)
        # Remove the old legend, then add the new one on
        self.main_ax.get_legend().remove()
        self.main_ax.legend(
            hand, [i.title() for i in utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys()],
            loc='lower left', title='Instrument', frameon=True, framealpha=1,
            edgecolor=vutils.BLACK
        )
        # Final attributes to set here
        self.main_ax.set(
            xticks=[100, 150, 200, 250, 300], yticks=[-1, 0, 1], xlim=(100, 320),
            xlabel='Mean Tempo (BPM)', ylabel='Mean ${Log_2}$ beat-upbeat ratio', ylim=(-1.35, 1.7)
        )

    def _format_ax(self):
        """Formats axis-level properties"""
        # Run code for formatting main and marginal ax separately
        self._format_main_ax()
        self._format_marginal_ax()
        # These lines of code apply to every ax on the plot
        for a in [self.main_ax, *self.marginal_ax.flatten()]:
            plt.setp(a.spines.values(), linewidth=vutils.LINEWIDTH)
            a.tick_params(axis='both', bottom=True, width=vutils.TICKWIDTH)

    def _format_fig(self):
        """Format figure-level properties"""
        self.fig.subplots_adjust(left=0.05, right=0.99, top=0.99, bottom=0.09, hspace=0.1, wspace=0.05)


if __name__ == '__main__':
    pass
