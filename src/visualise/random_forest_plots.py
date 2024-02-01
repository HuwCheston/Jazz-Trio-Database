#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Plotting classes for random forest model, e.g. heatmaps, feature importance."""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import confusion_matrix, RocCurveDisplay
from joblib import Parallel, delayed

from src import utils
import src.visualise.visualise_utils as vutils

__all__ = [
    'BarPlotCategoryImportances', 'BarPlotFeatureImportances', 'HeatMapFeatureCorrelation', 'CountPlotMissingValues',
    'HeatMapPredictionProbDendro', 'RocPlotLogRegression', 'StripPlotLogitCoeffs', 'HistPlotFirstLastP',
    'RegPlotPredictorsCareerProgress', "RegPlotCareerJazzProgress"
]

PREDICTORS_CATEGORIES = {
    'Swing': ['bur_log_mean', 'bur_log_std'],
    'Complexity': ['lz77_mean', 'lz77_std', 'n_onsets_mean', 'n_onsets_std'],
    'Feel': ['bass_prop_async_nanmean', 'drums_prop_async_nanmean',
             'bass_prop_async_nanstd', 'drums_prop_async_nanstd'],
    'Interaction': ['self_coupling', 'coupling_drums', 'coupling_bass', 'coupling_piano_drums', 'coupling_piano_bass'],
    'Tempo': ['rolling_std_median', 'tempo', 'tempo_slope']
}
CATEGORY_CMAP = {
    cat: col for cat, col in zip(PREDICTORS_CATEGORIES.keys(), plt.rcParams['axes.prop_cycle'].by_key()['color'])
}
COL_MAPPING = {
    'bur_log_mean': 'Beat-upbeat ratio, mean',
    'bur_log_std': 'Beat-upbeat ratio, std.',
    'lz77_mean': 'Compression score, mean',
    'lz77_std': 'Compression score, std',
    'n_onsets_mean': 'Onset density, mean',
    'n_onsets_std': 'Onset density, std',
    'bass_prop_async_nanmean': 'Piano→Bass, async mean',
    'bass_prop_async_nanstd': 'Piano→Bass, async std.',
    'drums_prop_async_nanmean': 'Piano→Drums, async mean',
    'drums_prop_async_nanstd': 'Piano→Drums, async std.',
    'coupling_bass': 'Piano→Bass, coupling',
    'coupling_piano_bass': 'Bass→Piano, coupling',
    'coupling_drums': 'Piano→Drums, coupling',
    'coupling_piano_drums': 'Drums→Piano, coupling',
    'self_coupling': 'Piano→Piano, coupling',
    'rolling_std_median': 'Tempo stability',
    'tempo': 'Tempo average',
    'tempo_slope': 'Tempo slope',
}


class CountPlotMissingValues(vutils.BasePlot):
    """Creates a bar plot showing the percentage of missing values for each predictor"""
    BAR_KWS = dict(
        dodge=False, edgecolor=vutils.BLACK, errorbar=None, lw=vutils.LINEWIDTH, zorder=3,
        capsize=0.1, width=0.8, ls=vutils.LINESTYLE, hue_order=PREDICTORS_CATEGORIES.keys()
    )

    def __init__(self, predictors_df, category_mapping, **kwargs):
        self.corpus_title = 'corpus_chronology'
        self.cat_map = category_mapping
        self.df = self._format_df(predictors_df)
        super().__init__(
            figure_title=fr'random_forest_plots\countplot_missing_values_{self.corpus_title}', **kwargs
        )
        self.fig, self.ax = plt.subplots(1, 1, figsize=(vutils.WIDTH / 2, vutils.WIDTH / 2))

    def _format_df(self, pred_df: pd.DataFrame) -> pd.DataFrame:
        """Formats the provided dataframe `pred_df` into the correct format for plotting"""
        d = (
            pred_df.isna()
            .sum()
            .sort_values()
            .reset_index(drop=False)
            .rename(columns={'index': 'variable', 0: 'value'})
        )
        d['category'] = d['variable'].map(self.cat_map)
        d['value'] = (d['value'] / 300) * 100
        return d

    def _create_plot(self) -> plt.Axes:
        """Creates the main axis object: percentage of missing values"""
        return sns.barplot(self.df, y='variable', x='value', hue='category', **self.BAR_KWS)

    def _format_ax(self) -> None:
        """Formats axis-level parameters"""
        self.ax.legend(loc='upper right', title='Category', frameon=True, framealpha=1, edgecolor=vutils.BLACK,)
        plt.setp(self.ax.spines.values(), linewidth=vutils.LINEWIDTH, color=vutils.BLACK)
        self.ax.tick_params(axis='both', width=vutils.TICKWIDTH, color=vutils.BLACK, rotation=0)
        labs = []
        for tick in self.ax.get_yticklabels():
            cat = [k for k, v in PREDICTORS_CATEGORIES.items() if tick.get_text() in v][0]
            tick.set_color(CATEGORY_CMAP[cat])
            labs.append(COL_MAPPING[tick.get_text()])
        self.ax.set(ylabel='Variable', xlabel='% of missing values', yticklabels=labs)

    def _format_fig(self) -> None:
        """Formats figure-level parameters"""
        self.fig.tight_layout()


class HeatMapFeatureCorrelation(vutils.BasePlot):
    """Creates a (triangular) heatmap showing the correlation between different predictor features"""
    LINE_KWS = dict(lw=vutils.LINEWIDTH, ls=vutils.LINESTYLE)
    TEXT_KWS = dict(rotation=0, va='top', ha='right', fontsize=vutils.FONTSIZE * 1.5)
    HEAT_KWS = dict(
        vmin=-1, vmax=1, annot=True, cmap='vlag', fmt='.2f', annot_kws={'size': vutils.FONTSIZE},
        cbar_kws={'label': 'Correlation ($r$)'}, linecolor=vutils.WHITE, linewidths=vutils.LINEWIDTH / 2,
    )

    def __init__(self, features: pd.DataFrame, **kwargs):
        self.corpus_title = 'corpus_chronology'
        self.df = features
        super().__init__(
            figure_title=fr'random_forest_plots\heatmap_feature_correlations_{self.corpus_title}', **kwargs
        )
        self.fig = plt.figure(figsize=(vutils.WIDTH, vutils.WIDTH))
        self.gs0 = mpl.gridspec.GridSpec(1, 2, width_ratios=[40, 1], hspace=0.00)
        self.ax = self.fig.add_subplot(self.gs0[0])
        self.cax = self.fig.add_subplot(self.gs0[1])

    def _create_plot(self) -> plt.Axes:
        """Creates main plotting object"""
        corr = self.df.rename(columns=COL_MAPPING).corr()
        matrix = np.triu(corr)
        g = sns.heatmap(corr, cbar_ax=self.cax, ax=self.ax,  mask=matrix, **self.HEAT_KWS)
        s = len(corr)
        g.plot([0, s, 0, 0], [0, s, s, 0], clip_on=False, color=vutils.BLACK, lw=vutils.LINEWIDTH)
        return g

    def _format_cax(self) -> None:
        """Sets parameters for the color bar"""
        self.cax.tick_params(axis='both', width=vutils.TICKWIDTH, color=vutils.BLACK)
        for spine in self.cax.spines:
            self.cax.spines[spine].set_color(vutils.BLACK)
            self.cax.spines[spine].set_linewidth(vutils.LINEWIDTH)

    def _format_corr_labels(self, lim: float = 1/2) -> None:
        """Hides correlation text if values of `r` below given absolute limit `lim`"""
        for t in self.ax.texts:
            if abs(float(t.get_text())) > lim:
                t.set_text(t.get_text())
            else:
                t.set_text('')

    def _add_text_to_triangle(self) -> None:
        """Adds text to the triangle plot"""
        x = 0
        for c, (cat, predictors) in zip(CATEGORY_CMAP.values(), PREDICTORS_CATEGORIES.items()):
            x += len(predictors)
            # Diagonal
            self.ax.plot((x - len(predictors), x), (x - len(predictors), x), color=c, **self.LINE_KWS)
            # Horizontal
            self.ax.plot((x - len(predictors), x), (x - len(predictors), x - len(predictors)), color=c, **self.LINE_KWS)
            # Vertical
            self.ax.plot((x, x), (x - len(predictors), x), color=c, **self.LINE_KWS)
            self.ax.text(x, x - len(predictors), cat, color=c, **self.TEXT_KWS)

    def _format_tick_labels(self) -> None:
        """Sets colors for predictor tick values to their given category"""
        rev = {v: k for k, v in COL_MAPPING.items()}
        for func in [self.ax.get_xticklabels, self.ax.get_yticklabels]:
            ticks = func()
            for tick in ticks:
                orig = rev[tick.get_text()]
                for cat, preds in PREDICTORS_CATEGORIES.items():
                    if orig in preds:
                        tick.set_color(CATEGORY_CMAP[cat])

    def _format_ax(self) -> None:
        """Formats axis-level parameters"""
        self.ax.tick_params(axis='both', width=vutils.TICKWIDTH, color=vutils.BLACK)
        self._format_cax()
        self._format_corr_labels()
        self._add_text_to_triangle()
        self._format_tick_labels()

    def _format_fig(self) -> None:
        """Formats figure-level parameters"""
        self.fig.tight_layout()


class BarPlotFeatureImportances(vutils.BasePlot):
    """Creates barplot showing importance of all features"""
    # These are the keywords we apply to all bar plots
    BAR_KWS = dict(
        dodge=False, edgecolor=vutils.BLACK, errorbar=None, lw=vutils.LINEWIDTH, capsize=0.1, width=0.8,
        ls=vutils.LINESTYLE, hue_order=PREDICTORS_CATEGORIES.keys(), zorder=3, estimator=np.mean,
    )
    ERROR_KWS = dict(ls='none', color=vutils.BLACK, elinewidth=2, capsize=5, markeredgewidth=2, zorder=5)
    # This creates a new dictionary with a unique hash for each predictor category
    hatches = {k: h for k, h in zip(PREDICTORS_CATEGORIES.keys(), vutils.HATCHES)}

    def __init__(self, importances: pd.DataFrame, **kwargs):
        self.corpus_title = 'corpus_chronology'
        super().__init__(figure_title=fr'random_forest_plots\barplot_feature_importances_{self.corpus_title}', **kwargs)
        # Create both dataframes
        self.importances = (
            importances.copy(deep=True)
            .melt(id_vars=['feature', 'category'], var_name='fold')
            .groupby('feature')
            [['value', 'category']]
            .agg({'value': [np.mean, stats.sem], 'category': 'first'})
            .droplevel(0, axis=1)
            .rename(columns={'first': 'category', 'sem': 'std'})
            .reset_index(drop=False)
            .sort_values(by='mean')
        )
        self.importances['feature'] = self.importances['feature'].map(COL_MAPPING)
        # Create subplot matrix
        self.fig, self.ax = plt.subplots(
            nrows=1, ncols=1, figsize=(vutils.WIDTH / 2, vutils.WIDTH / 3)
        )

    def _create_plot(self) -> plt.Axes:
        """Creates all plots in seaborn with given arguments"""
        self.importances['mean'] *= 100
        sns.barplot(
            data=self.importances, x='mean', y='feature', hue='category',
            ax=self.ax, **self.BAR_KWS
        )

    def _format_ticks(self) -> None:
        """Sets colors for predictor ticks to their category"""
        rev = {v: k for k, v in COL_MAPPING.items()}
        for tick in self.ax.get_yticklabels():
            var = rev[tick.get_text()]
            for cat, vars_ in PREDICTORS_CATEGORIES.items():
                if var in vars_:
                    tick.set_color(CATEGORY_CMAP[cat])

    def _format_ax(self) -> None:
        """Formats axis-level properties"""
        # Set variable labelling
        self.ax.set(ylabel='Variable', xlabel='Feature Importance (%)')
        self._format_ticks()
        # Remove the legend
        self.ax.get_legend().remove()
        # Set the width of the edges and ticks
        plt.setp(self.ax.spines.values(), linewidth=vutils.LINEWIDTH, color=vutils.BLACK)
        self.ax.tick_params(axis='both', width=vutils.TICKWIDTH, color=vutils.BLACK)
        # Add a vertical grid
        self.ax.grid(zorder=0, axis='x', **vutils.GRID_KWS)

    def _format_fig(self) -> None:
        """Formats figure-level attributes"""
        self.fig.tight_layout()


class BarPlotCategoryImportances(vutils.BasePlot):
    """Creates barplot showing average importance of feature category"""
    # These are the keywords we apply to all bar plots
    BAR_KWS = dict(
        dodge=False, edgecolor=vutils.BLACK, errorbar=None, lw=vutils.LINEWIDTH, capsize=0.1, width=0.8,
        ls=vutils.LINESTYLE, hue_order=PREDICTORS_CATEGORIES.keys(), zorder=3, estimator=np.mean,
    )
    ERROR_KWS = dict(ls='none', color=vutils.BLACK, elinewidth=2, capsize=5, markeredgewidth=2, zorder=5)
    # This creates a new dictionary with a unique hash for each predictor category
    hatches = {k: h for k, h in zip(PREDICTORS_CATEGORIES.keys(), vutils.HATCHES)}

    def __init__(self, importances: pd.DataFrame, **kwargs):
        self.corpus_title = 'corpus_chronology'
        super().__init__(
            figure_title=fr'random_forest_plots\barplot_category_importances_{self.corpus_title}', **kwargs
        )
        # Create both dataframes
        self.grouped_importances = (
            importances.copy(deep=True)
            .mean(axis=1)
            .sort_values(ascending=True)
            .reset_index(drop=False)
            .rename(columns={'index': 'category', 0: 'mean'})
        )
        # Create subplot matrix
        self.fig, self.ax = plt.subplots(
            nrows=1, ncols=1, figsize=(vutils.WIDTH / 2, vutils.WIDTH / 3)
        )

    def _create_plot(self) -> plt.Axes:
        """Creates all plots in seaborn with given arguments"""
        self.grouped_importances['mean'] *= 100
        sns.barplot(
            data=self.grouped_importances, x='mean', y='category',
            hue='category', ax=self.ax, **self.BAR_KWS
        )

    def _format_ticks(self) -> None:
        """Sets tick values to the color assigned to each predictor category"""
        for tick in self.ax.get_yticklabels():
            tick.set_color(CATEGORY_CMAP[tick.get_text()])

    def _format_ax(self) -> None:
        """Formats axis-level properties"""
        # Set variable labelling
        self.ax.set(ylabel='Category', xlabel='Feature Importance (%)')
        self._format_ticks()
        # Remove the legend
        self.ax.get_legend().remove()
        # Set the width of the edges and ticks
        plt.setp(self.ax.spines.values(), linewidth=vutils.LINEWIDTH, color=vutils.BLACK)
        self.ax.tick_params(axis='both', width=vutils.TICKWIDTH, color=vutils.BLACK)
        # Add a vertical grid
        self.ax.grid(zorder=0, axis='x', **vutils.GRID_KWS)

    def _format_fig(self) -> None:
        """Formats figure-level attributes"""
        self.fig.tight_layout()


class HeatMapPredictionProbDendro(vutils.BasePlot):
    """Creates a heatmap of probabilities for all pianists in a dataset, with attached clustering dendrogram"""
    img_loc = fr'{utils.get_project_root()}\references\images\musicians'
    PAL = sns.cubehelix_palette(dark=1/3, gamma=.3, light=2/3, start=0, n_colors=20, as_cmap=False)
    MODEL_KWS = dict(n_clusters=None, distance_threshold=0, metric='precomputed', linkage='average')
    DENDRO_KWS = dict(truncate_mode=None, no_labels=False, color_threshold=0, above_threshold_color=vutils.BLACK)

    def __init__(self, prob_df, **kwargs):
        self.corpus_title = 'corpus_chronology'
        self.include_images = kwargs.get('include_images', True)
        fig_title = fr'random_forest_plots\heatmap_prediction_prob_dendro_{self.corpus_title}'
        if not self.include_images:
            fig_title += 'no_images'
        super().__init__(figure_title=fig_title, **kwargs)
        self.hm = pd.DataFrame(confusion_matrix(
            y_true=prob_df['actual'], y_pred=prob_df['prediction'], normalize='true')
        )
        self.md = self._fit_agg()
        self.fig = plt.figure(figsize=(vutils.WIDTH, vutils.WIDTH))
        gs0 = mpl.gridspec.GridSpec(2, 2, width_ratios=[20, 1], hspace=0.0, height_ratios=[4, 20])
        self.ax = self.fig.add_subplot(gs0[2])
        self.dax = self.fig.add_subplot(gs0[0])
        self.mapping = {i: k for i, k in enumerate(prob_df['actual'].unique())}

    def _fit_agg(self) -> AgglomerativeClustering:
        """Fits the agglomerative clustering model with the given parameters"""
        # We use correlation based distances here
        dm = (1 - np.corrcoef(self.hm))
        md = AgglomerativeClustering(**self.MODEL_KWS)
        md.fit(dm)
        return md

    def _create_dendrogram(self) -> None:
        """Creates the dendrogram and attaches to the top of the plot"""
        counts = np.zeros(self.md.children_.shape[0])
        n_samples = len(self.md.labels_)
        for i, merge in enumerate(self.md.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count
        linkage_matrix = np.column_stack(
            [self.md.children_, self.md.distances_, counts]
        ).astype(float)
        # Plot the corresponding dendrogram
        with plt.rc_context({'lines.linewidth': vutils.LINEWIDTH, 'lines.linestyle': vutils.LINESTYLE}):
            dendrogram(linkage_matrix, ax=self.dax, **self.DENDRO_KWS)

    def _create_plot(self) -> None:
        """Creates both the agglomerative clustering dendrogram and the confusion matrix heatmap"""
        self._create_dendrogram()
        self._format_dax()
        self.labs = [int(i.get_text()) for i in self.dax.get_xticklabels()]
        reord = self.hm[self.labs].reindex(reversed(self.labs))
        sns.heatmap(
            reord * 100, ax=self.ax, cmap="Reds", linecolor=vutils.WHITE, square=True, annot=True,
            fmt='.0f', linewidths=vutils.LINEWIDTH/2, vmin=0, vmax=100,
            cbar_kws=dict(
                label='Probability (%)', use_gridspec=False, location="right", pad=0.2, shrink=0.725,
                ticks=[0, 25, 50, 75, 100],
            )
        )

    def _add_pianist_images(self) -> None:
        """Adds images corresponding to each pianist along the plot"""
        for num, mus in enumerate(self.ax.get_xticklabels()):
            for f in os.listdir(self.img_loc):
                if mus.get_text().lower() in f.lower():
                    img = mpl.offsetbox.OffsetImage(
                        plt.imread(fr'{self.img_loc}\{f}'), clip_on=False, transform=self.ax.transAxes, zoom=0.75
                    )
                    ab = mpl.offsetbox.AnnotationBbox(
                        img, (num + 0.5, -0.5), xycoords='data', clip_on=False, transform=self.ax.transAxes,
                        annotation_clip=False, bboxprops=dict(
                            edgecolor=vutils.BLACK, lw=2, boxstyle='sawtooth', clip_on=False,
                        ),
                    )
                    self.ax.add_artist(ab)
                    ab = mpl.offsetbox.AnnotationBbox(
                        img, (10.5, -num + 9.5), xycoords='data', clip_on=False, transform=self.ax.transAxes,
                        annotation_clip=False, bboxprops=dict(
                            edgecolor=vutils.BLACK, lw=2, boxstyle='sawtooth', clip_on=False,
                        )
                    )
                    self.ax.add_artist(ab)

    def _format_annotations(self) -> None:
        """Format annotations to only show those along the diagonal, i.e. 'hits'"""
        texts = np.array(self.ax.texts).reshape(-1, 10)
        masks = np.fliplr(np.eye(texts.shape[0], dtype=bool))
        for text, mask in zip(texts.flatten(), masks.flatten()):
            if mask:
                text.set_fontsize((vutils.FONTSIZE * 2) * (int(text.get_text())) / 80)
                text.set_color(vutils.WHITE)
                text.set_text(text.get_text() + '%')
            else:
                text.set_text('')

    def _format_dax(self) -> None:
        """Sets axis-level parameters for the dendrogram"""
        self.dax.set(ylim=(0, 1.2), yticks=[])
        self.dax.axhline(1.15, 0, 1, color=vutils.BLACK, lw=vutils.LINEWIDTH, ls='dashed', alpha=vutils.ALPHA)
        plt.setp(self.dax.spines.values(), linewidth=vutils.LINEWIDTH, color=vutils.BLACK)
        self.dax.tick_params(axis='both', width=vutils.TICKWIDTH, color=vutils.BLACK, bottom=True)
        for spine in ['top', 'left', 'right']:
            self.dax.spines[spine].set_visible(False)

    def _format_ax_ticks(self) -> None:
        """Sets tick labels for all axis"""
        for ax in [self.ax.xaxis, self.ax.yaxis, self.dax.xaxis]:
            prev_labs = [int(i.get_text()) for i in ax.get_ticklabels()]
            ax.set_ticklabels([self.mapping[li].split(' ')[-1] for li in prev_labs])

    def _format_ax(self) -> None:
        """Sets axis-level parameters for the main heatmap and the colorbar"""
        self._format_ax_ticks()
        if self.include_images:
            self._add_pianist_images()
        self._format_annotations()
        self.ax.set(ylabel='', xlabel='',)
        self.ax.tick_params(axis='both', top=True, left=True, right=True, bottom=True)
        cax = self.ax.figure.axes[-1]
        for a in [self.ax, cax]:
            for spine in a.spines.values():
                spine.set_visible(True)
                spine.set_color(vutils.BLACK)
            plt.setp(a.spines.values(), linewidth=vutils.LINEWIDTH, color=vutils.BLACK)
            a.tick_params(axis='both', width=vutils.TICKWIDTH, color=vutils.BLACK, rotation=0)

    def _format_fig(self) -> None:
        """Sets figure-level parameters"""
        self.fig.supxlabel('Predicted pianist', y=0.01)
        self.fig.supylabel('Actual pianist', y=0.45)
        self.fig.subplots_adjust(top=1, bottom=0.01, left=0.1, right=0.95)
        pos = self.ax.get_position()
        self.ax.set_position([pos.x0, pos.y0, pos.width, pos.height])
        cax = self.ax.figure.axes[-1]
        cax.set_position([0.915, 0.17, 0.2, 0.51])
        cax.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        cax.set_ylim(0, 100)
        cax.set_yticklabels(['0', '25', '50', '75', '100'])
        cax.tick_params(axis='y', which='minor', width=None, right=False)
        pos = self.dax.get_position()
        self.dax.set_position([pos.x0, pos.y0 + 0.03, pos.width, pos.height - 0.035])
        self.dax.set(xticklabels=[])


class RocPlotLogRegression(vutils.BasePlot):
    """Creates a plot showing the receiver-operator curve from true and predicted values from a logistic regression"""
    def __init__(self, y_true: np.array, y_predict: np.array, **kwargs):
        self.corpus_title = 'corpus_chronology'
        super().__init__(figure_title=fr'random_forest_plots\rainplot_algohuman_onsets_{self.corpus_title}', **kwargs)
        self.fig, self.ax = plt.subplots(1, 1, figsize=(vutils.WIDTH / 2, vutils.WIDTH / 2))
        self.y_true = y_true
        self.y_pred = y_predict

    def _create_plot(self) -> plt.Axes:
        """Creates the main plotting object using the function in `sklearn`"""
        self.ax.plot(
            (0, 1.01), (0, 1.01), linestyle='dashed', lw=vutils.LINEWIDTH,
            color=vutils.BLACK, alpha=vutils.ALPHA, label='Random (AUC = 0.5)'
        )
        return RocCurveDisplay.from_predictions(
            self.y_true, self.y_pred, ax=self.ax, c=vutils.BLACK, lw=vutils.LINEWIDTH,
            ls=vutils.LINESTYLE, name='Logistic regression'
        )

    def _format_ax(self) -> None:
        """Sets axis-level parameters"""
        self.ax.legend(loc='lower right', frameon=True, framealpha=1, edgecolor=vutils.BLACK, title='Classifier')
        plt.setp(self.ax.spines.values(), linewidth=vutils.LINEWIDTH, color=vutils.BLACK)
        self.ax.tick_params(axis='both', width=vutils.TICKWIDTH, color=vutils.BLACK, rotation=0)
        self.ax.set(xlim=(0, 1.01), ylim=(0, 1.01), xlabel='False positive rate', ylabel='True positive rate')

    def _format_fig(self) -> None:
        """Sets figure-level parameters"""
        self.fig.tight_layout()


class StripPlotLogitCoeffs(vutils.BasePlot):
    """Creates a 'forest plot' showing coefficients obtained from the logistic regression model"""
    STRIP_KWS = dict(
        edgecolor=vutils.BLACK, linewidth=vutils.LINEWIDTH, zorder=5,
        hue_order=['Feel', 'Tempo', 'Swing', 'Complexity', 'Interaction']
    )
    ERROR_KWS = dict(
        lw=vutils.LINEWIDTH, color=vutils.BLACK, linestyle='none', capsize=5, elinewidth=2, markeredgewidth=2
    )
    LEGEND_KWS = dict(frameon=True, framealpha=1, edgecolor=vutils.BLACK)
    palette = sns.color_palette('tab10')
    palette = [palette[2], palette[4], palette[0], palette[1],   palette[3]]

    def __init__(self, logit_md, category_mapping, **kwargs):
        self.corpus_title = 'corpus_chronology'
        super().__init__(
            figure_title=fr'random_forest_plots\stripplot_logitcoeffs_{self.corpus_title}', **kwargs
        )
        self.cat_map = category_mapping
        self.df = self._format_df(logit_md)
        self.fig, self.ax = plt.subplots(1, 1, figsize=(vutils.WIDTH, vutils.WIDTH / 2))

    @staticmethod
    def _format_p(pval: float) -> str:
        """Returns correct number of asterisks for a given significance level corresponding to input `pval`"""
        if pval < 0.001:
            return '***'
        elif pval < 0.01:
            return '**'
        elif pval < 0.05:
            return '*'
        else:
            return ''

    def _format_df(self, logit_md) -> pd.DataFrame:
        """Coerces logistic regression output into correct format for plotting"""
        coeff = logit_md.params.rename('coeff').apply(np.exp)
        ci = logit_md.conf_int().rename(columns={0: 'low', 1: 'high'}).apply(np.exp)
        pvals = logit_md.pvalues.rename('p').apply(lambda r: self._format_p(r))
        params = (
            pd.concat([coeff, ci, pvals], axis=1)
            .reset_index(drop=False)[1:]
        )
        params['low'] = params['coeff'] - params['low']
        params['high'] -= params['coeff']
        params['category'] = params['index'].map(self.cat_map)
        return params.iloc[::-1]

    def _create_plot(self) -> None:
        """Creates the forest plot"""
        sns.stripplot(
            data=self.df, x='coeff', y='index', s=10, hue='category', ax=self.ax,
            palette=self.palette, **self.STRIP_KWS
        )
        self.ax.errorbar(self.df['coeff'], self.df['index'], **self.ERROR_KWS, xerr=(self.df['low'], self.df['high']))
        for idx, row in self.df.iterrows():
            self._add_pvals(row)

    def _format_ax(self) -> None:
        """Sets axis-level properties"""
        self.ax.grid(axis='y', which='major', **vutils.GRID_KWS)
        self.ax.axvline(1, 0, 1, color=vutils.BLACK, lw=vutils.LINEWIDTH, ls='dashed', zorder=1, )
        for spine in ['left', 'right', 'top']:
            self.ax.spines[spine].set_visible(False)
        self.ax.set_xscale('log')
        plt.setp(self.ax.spines.values(), linewidth=vutils.LINEWIDTH, color=vutils.BLACK)
        self.ax.tick_params(axis='both', width=vutils.TICKWIDTH, color=vutils.BLACK, rotation=0)
        self.ax.set(
            ylim=(17.5, -0.5), xlabel='Odds ratio (95% CI, log scale)', xticks=[0.1, 1, 10],
            xlim=(0.099, 20.5), ylabel='Predictor', xticklabels=[0.1, 1, 10],
            yticklabels=[COL_MAPPING[i.get_text()] for i in self.ax.get_yticklabels()],
        )
        self._format_legend()
        self._format_yticks()
        self.ax.minorticks_off()

    def _add_pvals(self, row):
        """Adds in p-values for a given y-axis value `row`"""
        self.ax.text((row['coeff'] + row['high']) * 0.75, row['index'], str(row['p']))

    def _format_legend(self) -> None:
        """Formats axis legend"""
        handles, labels = self.ax.get_legend_handles_labels()
        for ha in handles:
            ha.set_edgecolor(vutils.BLACK)
            ha.set_linewidth(vutils.LINEWIDTH)
            ha.set_sizes([100])
        self.ax.legend(handles, labels, loc='upper right', title='Category', **self.LEGEND_KWS)

    def _format_yticks(self) -> None:
        """Formats y-axis ticks by setting them to their correct color for a given category"""
        new_pal = [[self.palette[i1] for _ in range(i2)] for i1, i2 in zip(range(5), [4, 3, 2, 4, 5])]
        for tl, tc in zip(self.ax.get_yticklabels(), [item for sublist in new_pal for item in sublist]):
            tl.set_color(tc)

    def _format_fig(self) -> None:
        """Formats figure-level parameters"""
        self.ax.text(0.15, -1, '← More likely to be "Impressionist"', fontsize=vutils.FONTSIZE * 1.1)
        self.ax.text(1.15, -1, 'More likely to be "Blues" →', fontsize=vutils.FONTSIZE * 1.1)
        self.fig.tight_layout()


class HistPlotFirstLastP(vutils.BasePlot):
    """Histogram plot of the probabilities resulting from the Monte Carlo permutation test"""
    KDE_BW = 0.3
    KDE_KWS = dict(lw=vutils.LINEWIDTH, ls=vutils.LINESTYLE, zorder=5, color=vutils.BLACK)
    FILL_KWS = dict(alpha=vutils.ALPHA, edgecolor=vutils.BLACK, lw=vutils.LINEWIDTH, ls=vutils.LINESTYLE)
    TEXT_KWS = dict(bbox=dict(
        facecolor='wheat', boxstyle='round', edgecolor='black', linewidth=vutils.LINEWIDTH)
    )

    def __init__(self, acc_scores, first_acc, last_acc, **kwargs):
        self.corpus_title = 'corpus_chronology'
        self.acc_scores = acc_scores
        self.first_acc, self.last_acc = first_acc, last_acc
        super().__init__(
            figure_title=fr'random_forest_plots\histplot_firstlastp_{self.corpus_title}', **kwargs
        )
        self.fig, self.ax = plt.subplots(
            nrows=1, ncols=2, sharex=True, sharey=True, figsize=(vutils.WIDTH, vutils.WIDTH / 4)
        )

    def _get_kde(self) -> tuple:
        """Creates the kernel density estimate, fits to the data, and standardizes"""
        kde = stats.gaussian_kde(self.acc_scores, bw_method=self.KDE_BW)
        x = np.linspace(np.min(self.acc_scores), np.max(self.acc_scores), 1000)
        y = kde.evaluate(x)
        return x, np.array([(y_ - min(y)) / (max(y) - min(y)) for y_ in y])

    @staticmethod
    def _slice_kde(x, y, acc) -> tuple:
        """Slice the KDE results between the given points"""
        x0 = x[x < acc]
        x1 = x[len(x0):]
        y0, y1 = y[:len(x0)], y[len(x0):]
        return x0, x1, y0, y1

    def _get_pval(self, acc: float) -> float:
        """Returns the proportion of values below the given accuracy score `acc`"""
        return stats.percentileofscore(self.acc_scores, acc, kind='weak') / 100

    def _create_plot(self) -> None:
        """Creates the main plotting object"""
        x, y = self._get_kde()
        for ax, acc in zip(self.ax.flatten(), [self.first_acc, self.last_acc]):
            ax.plot(x, y, **self.KDE_KWS)
            # Slice the arrays
            x0, x1, y0, y1 = self._slice_kde(x, y, acc)
            # Fill the areas in the KDE
            ax.fill_between(x=x0, y1=y0, hatch='X', color=vutils.RED, **self.FILL_KWS)
            ax.fill_between(x=x1, y1=y1, color=vutils.BLUE, **self.FILL_KWS)
            # Add the p-value
            txt = rf'$p$ = {str(round(self._get_pval(acc), 2))[1:]}'
            ax.annotate(txt, (np.mean(x0) - 0.1, np.mean(y0) + 0.1), **self.TEXT_KWS)

    def _format_ax(self) -> None:
        """Sets axis-level parameters"""
        for a, tit in zip(self.ax.flatten(), ['earliest', 'final']):
            a.grid(axis='x', which='major', **vutils.GRID_KWS)
            a.set(
                xlim=(0, 1), ylim=(0, 1.01), yticks=(0, 0.5, 1), xticklabels=[int(i*100) for i in a.get_xticks()],
                title=f'Predicting {tit} recordings'
            )
            plt.setp(a.spines.values(), linewidth=vutils.LINEWIDTH)
            a.tick_params(axis='both', bottom=True, width=vutils.TICKWIDTH)

    def _format_fig(self) -> None:
        """Sets figure-level parameters"""
        self.fig.supxlabel('Accuracy (%)')
        self.fig.supylabel('Density')
        self.fig.subplots_adjust(left=0.065, right=0.965, top=0.9, bottom=0.15, wspace=0.05)


class RegPlotPredictorsCareerProgress(vutils.BasePlot):
    """Creates regression plots showing associations between career progress and individual predictors"""
    palette = sns.color_palette('tab10')
    palette = [palette[2], palette[4], palette[0], palette[1],   palette[3]]
    predictors = ['drums_prop_async_nanmean', 'tempo', 'bur_log_mean', 'n_onsets_mean', 'coupling_piano_drums']
    markers = ['o', 's', 'D', '^', 'p']
    categories = ['Feel', 'Tempo', 'Swing', 'Complexity', 'Interaction']
    REG_KWS = dict(
        scatter=False, color=vutils.BLACK, n_boot=vutils.N_BOOT, ci=95,
        line_kws=dict(linewidth=vutils.LINEWIDTH * 2, ls=vutils.LINESTYLE)
    )
    TEXT_KWS = dict(bbox=dict(
        facecolor='wheat', boxstyle='round', edgecolor='black', linewidth=vutils.LINEWIDTH)
    )

    def __init__(self, model_df, cat_mapping, **kwargs):
        self.corpus_title = 'corpus_chronology'
        self.cat_mapping = cat_mapping
        self.df = model_df
        super().__init__(
            figure_title=fr'random_forest_plots\regplot_careerprogress_{self.corpus_title}', **kwargs
        )
        self.fig, self.ax = plt.subplots(
            nrows=2, ncols=3, figsize=(vutils.WIDTH, vutils.WIDTH / 2), sharex=True, sharey=False
        )
        self.ax.flatten()[-1].axis('off')

    def _create_plot(self) -> None:
        """Creates the main plotting objects: scatter and regression plots for each predictor"""
        for a, predict, col, mark in zip(self.ax.flatten(), self.predictors, self.palette, self.markers):
            data = self.df[['career_progress', predict]].fillna(self.df[['career_progress', predict]].mean())
            sns.scatterplot(
                data=data, x='career_progress', y=predict, label=self.cat_mapping[predict],
                ax=a, color=col, marker=mark, s=100, alpha=vutils.ALPHA
            )
            sns.regplot(data=data, x='career_progress', y=predict, ax=a, **self.REG_KWS)
            r = stats.pearsonr(data['career_progress'], data[predict])[0]
            txt = str(round(r, 2)).replace('0.', '.')
            a.text(0.75, 0.905, f'$r$ = {txt}', transform=a.transAxes, **self.TEXT_KWS)

    def _format_ax(self) -> None:
        """Sets axis-level parameters"""
        leg, hand = [], []
        for a, tit, col in zip(self.ax.flatten(), self.categories, self.palette):
            l, h = a.get_legend_handles_labels()
            leg.extend(l)
            hand.extend(h)
            a.get_legend().remove()
            a.grid(axis='x', which='major', **vutils.GRID_KWS)
            a.set_title(tit, color=col)
            a.set_ylabel(COL_MAPPING[a.get_ylabel()], color=col)
            a.set(xlabel='', xlim=(-0.05, 1.05))
            plt.setp(a.spines.values(), linewidth=vutils.LINEWIDTH)
            a.tick_params(axis='both', bottom=True, width=vutils.TICKWIDTH)
        self.fig.legend(
            leg, hand, loc='lower right', title='Category', frameon=True,
            framealpha=1, edgecolor=vutils.BLACK, bbox_to_anchor=(0.9, 0.15)
        )

    def _format_fig(self) -> None:
        """Sets figure-level parameters"""
        self.fig.supxlabel('Career progress (0 = earliest recording, 1 = final recording)')
        self.fig.subplots_adjust(left=0.07, right=0.99, bottom=0.1, top=0.95, wspace=0.25)


class RegPlotCareerJazzProgress(vutils.BasePlot):
    predictors = ['drums_prop_async_nanmean', 'tempo_slope', 'bur_log_mean', 'n_onsets_mean', 'coupling_piano_drums']
    palette = sns.color_palette('tab10')
    palette = [palette[2], palette[4], palette[0], palette[1],   palette[3]]
    markers = ['o', 's', 'D', '^', 'p']
    categories = ['Feel', 'Tempo', 'Swing', 'Complexity', 'Interaction']
    REG_KWS = dict(color=vutils.BLACK, linewidth=vutils.LINEWIDTH * 2, ls=vutils.LINESTYLE)
    xspace = np.linspace(0, 1, 100)
    N_JOBS = -1

    def __init__(self, model_df, cat_mapping, **kwargs):
        self.corpus_title = 'corpus_chronology'
        self.cat_mapping = cat_mapping
        self.df = model_df.copy(deep=True)
        super().__init__(
            figure_title=fr'random_forest_plots\regplot_careerjazzprogress_{self.corpus_title}', **kwargs
        )
        self.fig, self.ax = plt.subplots(
            nrows=2, ncols=3, figsize=(vutils.WIDTH, vutils.WIDTH / 2), sharex=True, sharey=False
        )
        self.ax.flatten()[-1].axis('off')

    def get_line(self, model, mean):
        params = model.params
        intercept = params['Intercept']
        jp = params['jazz_progress'] * mean
        return [intercept + jp + (i * params['career_progress']) for i in self.xspace]

    @staticmethod
    def fit_model(data_, predict_):
        return (
            smf.mixedlm(
                f'{predict_}~career_progress+jazz_progress',
                groups=data_['pianist'],
                data=data_,
                re_formula='~jazz_progress'
            ).fit(method=['lbfgs'])
        )

    def bootstrap(self, data_, predict_) -> pd.DataFrame:
        def shuffle_data_and_fit(state):
            samp = data_.sample(frac=1, replace=True, random_state=state)
            mode = self.fit_model(samp, predict_)
            return pd.Series(self.get_line(mode, np.mean(samp['jazz_progress'])))

        with Parallel(n_jobs=self.N_JOBS, verbose=3) as par:
            boots = par(delayed(shuffle_data_and_fit)(num) for num in range(vutils.N_BOOT))
        boot_ys = pd.concat(boots, axis=1)
        low = boot_ys.apply(lambda r: np.percentile(r, 2.5), axis=1)
        high = boot_ys.apply(lambda r: np.percentile(r, 97.5), axis=1)
        return pd.DataFrame(dict(x=self.xspace, low=low, high=high))

    def _create_plot(self):
        for ax, predict, col, mark in zip(self.ax.flatten(), self.predictors, self.palette, self.markers):
            data = self.df.copy(deep=True)
            data[predict] = data[predict].fillna(data[predict].mean())
            md = self.fit_model(data, predict)
            sns.scatterplot(
                data=data, x='career_progress', y=predict, label=self.cat_mapping[predict],
                ax=ax, color=col, marker=mark, s=100, alpha=vutils.ALPHA
            )
            results_y = pd.DataFrame(self.get_line(md, np.mean(data['jazz_progress'])))
            ax.plot(self.xspace, results_y, **self.REG_KWS)
            boot_df = self.bootstrap(data, predict)
            ax.fill_between(boot_df['x'], boot_df['low'], boot_df['high'], color=vutils.BLACK, alpha=vutils.ALPHA)

    def _format_ax(self):
        leg, hand = [], []
        for a, tit, col in zip(self.ax.flatten(), self.categories, self.palette):
            l, h = a.get_legend_handles_labels()
            leg.extend(l)
            hand.extend(h)
            a.get_legend().remove()
            a.grid(axis='x', which='major', **vutils.GRID_KWS)
            a.set_title(tit, color=col)
            a.set_ylabel(COL_MAPPING[a.get_ylabel()], color=col)
            a.set(xlabel='', xlim=(-0.05, 1.05))
            plt.setp(a.spines.values(), linewidth=vutils.LINEWIDTH)
            a.tick_params(axis='both', bottom=True, width=vutils.TICKWIDTH)
        self.fig.legend(
            leg, hand, loc='lower right', title='Category', frameon=True,
            framealpha=1, edgecolor=vutils.BLACK, bbox_to_anchor=(0.9, 0.15)
        )

    def _format_fig(self):
        self.fig.supxlabel('Career progress (0 = earliest recording, 1 = final recording)')
        self.fig.subplots_adjust(left=0.07, right=0.99, bottom=0.1, top=0.95, wspace=0.25)


if __name__ == '__main__':
    pass
