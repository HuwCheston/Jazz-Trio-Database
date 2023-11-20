#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Plotting classes for random forest model, e.g. heatmaps, feature importance."""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import confusion_matrix

from src import utils
import src.visualise.visualise_utils as vutils

__all__ = [
    'BarPlotCategoryImportances', 'BarPlotFeatureImportances', 'HeatMapFeatureCorrelation',
    'HeatMapPredictionProbDendro'
]

PREDICTORS_CATEGORIES = {
    'Swing': ['bur_log_mean', 'bur_log_std'],
    'Complexity': ['lz77_mean', 'lz77_std', 'n_onsets_mean', 'n_onsets_std'],
    'Feel': ['piano_bass_prop_async_nanmean', 'piano_drums_prop_async_nanmean',
             'piano_bass_prop_async_nanstd', 'piano_drums_prop_async_nanstd'],
    'Interaction': ['self_coupling', 'coupling_drums', 'coupling_bass', 'coupling_piano_drums', 'coupling_piano_bass'],
    'Tempo': ['rolling_std_median', 'tempo', 'tempo_slope']
}
CATEGORY_CMAP = {
    cat: col for cat, col in zip(PREDICTORS_CATEGORIES.keys(), plt.rcParams['axes.prop_cycle'].by_key()['color'])
}
COL_MAPPING = {
    'bur_log_mean': 'Beat-upbeat ratio, mean',
    'bur_log_std': 'Beat-upbeat ratio, std.',
    'lz77_mean': 'Window LZ77, mean',
    'lz77_std': 'Window LZ77, std',
    'n_onsets_mean': 'Window density, mean',
    'n_onsets_std': 'Window density, std',
    'piano_bass_prop_async_nanmean': 'Piano→Bass, async mean',
    'piano_bass_prop_async_nanstd': 'Piano→Bass, async std.',
    'piano_drums_prop_async_nanmean': 'Piano→Drums, async mean',
    'piano_drums_prop_async_nanstd': 'Piano→Drums, async std.',
    'coupling_bass': 'Piano→Bass, coupling',
    'coupling_piano_bass': 'Bass→Piano, coupling',
    'coupling_drums': 'Piano→Drums, coupling',
    'coupling_piano_drums': 'Drums→Piano, coupling',
    'self_coupling': 'Piano→Piano, coupling',
    'rolling_std_median': 'Tempo stability',
    'tempo': 'Tempo average',
    'tempo_slope': 'Tempo slope',
}


class HeatMapFeatureCorrelation(vutils.BasePlot):
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

    def _create_plot(self):
        corr = self.df.rename(columns=COL_MAPPING).corr()
        matrix = np.triu(corr)
        g = sns.heatmap(corr, cbar_ax=self.cax, ax=self.ax,  mask=matrix, **self.HEAT_KWS)
        s = len(corr)
        g.plot([0, s, 0, 0], [0, s, s, 0], clip_on=False, color=vutils.BLACK, lw=vutils.LINEWIDTH)
        return g

    def _format_cax(self):
        self.cax.tick_params(axis='both', width=vutils.TICKWIDTH, color=vutils.BLACK)
        for spine in self.cax.spines:
            self.cax.spines[spine].set_color(vutils.BLACK)
            self.cax.spines[spine].set_linewidth(vutils.LINEWIDTH)

    def _format_corr_labels(self, lim: float = 1/2):
        for t in self.ax.texts:
            if abs(float(t.get_text())) > lim:
                t.set_text(t.get_text())
            else:
                t.set_text('')

    def _add_text_to_triangle(self):
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

    def _format_tick_labels(self):
        rev = {v: k for k, v in COL_MAPPING.items()}
        for func in [self.ax.get_xticklabels, self.ax.get_yticklabels]:
            ticks = func()
            for tick in ticks:
                orig = rev[tick.get_text()]
                for cat, preds in PREDICTORS_CATEGORIES.items():
                    if orig in preds:
                        tick.set_color(CATEGORY_CMAP[cat])

    def _format_ax(self):
        self.ax.tick_params(axis='both', width=vutils.TICKWIDTH, color=vutils.BLACK)
        self._format_cax()
        self._format_corr_labels()
        self._add_text_to_triangle()
        self._format_tick_labels()

    def _format_fig(self):
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
            .agg({'value': [np.mean, np.std], 'category': 'first'})
            .droplevel(0, axis=1)
            .rename(columns={'first': 'category'})
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
        sns.barplot(
            data=self.importances, x='mean', y='feature', hue='category',
            ax=self.ax, **self.BAR_KWS
        )
        self.ax.errorbar(
            self.importances['mean'], self.importances['feature'],
            xerr=self.importances['std'], **self.ERROR_KWS
        )

    def _format_ticks(self):
        rev = {v: k for k, v in COL_MAPPING.items()}
        for tick in self.ax.get_yticklabels():
            var = rev[tick.get_text()]
            for cat, vars_ in PREDICTORS_CATEGORIES.items():
                if var in vars_:
                    tick.set_color(CATEGORY_CMAP[cat])

    def _format_ax(self) -> None:
        """Formats axis-level properties"""
        # Set variable labelling
        self.ax.set(ylabel='Variable', xlabel='Variable Importance Score')
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
            .melt(id_vars=['feature', 'category'], var_name='fold')
            .groupby('category', as_index=False)
            ['value']
            .apply(self._bootstrap)
            .sort_values(by='mean')
        )
        # Create subplot matrix
        self.fig, self.ax = plt.subplots(
            nrows=1, ncols=1, figsize=(vutils.WIDTH / 2, vutils.WIDTH / 3)
        )

    @staticmethod
    def _bootstrap(vals, n_boot: int = vutils.N_BOOT):
        boots = [vals.sample(frac=1, replace=True, random_state=i).mean() for i in range(n_boot)]
        true_mean = np.mean(vals)
        return pd.Series({
            'low': true_mean - np.percentile(boots, 2.5),
            'mean': true_mean,
            'high': np.percentile(boots, 97.5) - true_mean
        })

    def _create_plot(self) -> plt.Axes:
        """Creates all plots in seaborn with given arguments"""
        sns.barplot(
            data=self.grouped_importances, x='mean', y='category',
            hue='category', ax=self.ax, **self.BAR_KWS
        )
        self.ax.errorbar(
            self.grouped_importances['mean'], self.grouped_importances['category'],
            xerr=(self.grouped_importances['low'], self.grouped_importances['high']),
            **self.ERROR_KWS
        )

    def _format_ticks(self):
        for tick in self.ax.get_yticklabels():
            tick.set_color(CATEGORY_CMAP[tick.get_text()])

    def _format_ax(self) -> None:
        """Formats axis-level properties"""
        # Set variable labelling
        self.ax.set(ylabel='Variable Category', xlabel='Mean Variable Importance Score')
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
    img_loc = fr'{utils.get_project_root()}\references\images\musicians'
    MODEL_KWS = dict(n_clusters=None, distance_threshold=0, metric='precomputed', linkage='average')
    DENDRO_KWS = dict(truncate_mode=None, no_labels=False, color_threshold=0, above_threshold_color=vutils.BLACK)

    def __init__(self, prob_df, **kwargs):
        self.corpus_title = 'corpus_chronology'
        super().__init__(
            figure_title=fr'random_forest_plots\heatmap_prediction_prob_dendro_{self.corpus_title}', **kwargs
        )
        self.hm = self._format_df(prob_df)
        self.md = self._fit_agg()
        self.fig = plt.figure(figsize=(vutils.WIDTH, vutils.WIDTH))
        gs0 = mpl.gridspec.GridSpec(2, 2, width_ratios=[20, 1], hspace=0.0, height_ratios=[4, 20])
        self.ax = self.fig.add_subplot(gs0[2])
        self.dax = self.fig.add_subplot(gs0[0])
        self.mapping = {i: k for i, k in enumerate(prob_df['actual'].unique())}

    def _fit_agg(self):
        dm = (1 - np.corrcoef(self.hm))
        md = AgglomerativeClustering(**self.MODEL_KWS)
        md.fit(dm)
        return md

    @staticmethod
    def _format_df(prob_df):
        return pd.DataFrame(confusion_matrix(
            y_true=prob_df['actual'], y_pred=prob_df['prediction'], normalize='true')
        )

    def _create_dendrogram(self):
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

    def _create_plot(self):
        self._create_dendrogram()
        self._format_dax()
        self.labs = [int(i.get_text()) for i in self.dax.get_xticklabels()]
        reord = self.hm[self.labs].reindex(reversed(self.labs))
        sns.heatmap(
            reord * 100, ax=self.ax, cmap="Purples", linecolor=vutils.WHITE, square=True, annot=True,
            fmt='.0f', linewidths=vutils.LINEWIDTH/2, vmin=0, vmax=100, norm=mpl.colors.LogNorm(0.1, 100),
            cbar_kws=dict(
                label='Probability (%)', use_gridspec=False, location="right", pad=0.2, shrink=0.725,
                ticks=[0.1, 1, 5, 10, 50, 100],
            )
        )

    def _add_pianist_images(self):
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

    def _format_annotations(self):
        texts = np.array(self.ax.texts).reshape(-1, 10)
        masks = np.fliplr(np.eye(texts.shape[0], dtype=bool))
        for text, mask in zip(texts.flatten(), masks.flatten()):
            if mask:
                text.set_fontsize((vutils.FONTSIZE * 2) * (int(text.get_text())) / 80)
                text.set_color(vutils.WHITE)
                text.set_text(text.get_text() + '%')
            else:
                text.set_text('')

    def _format_dax(self):
        self.dax.set(ylim=(0, 1.2), yticks=[])
        self.dax.axhline(1.15, 0, 1, color=vutils.BLACK, lw=vutils.LINEWIDTH * 2, ls='dashed', alpha=vutils.ALPHA)
        plt.setp(self.dax.spines.values(), linewidth=vutils.LINEWIDTH, color=vutils.BLACK)
        self.dax.tick_params(axis='both', width=vutils.TICKWIDTH, color=vutils.BLACK, bottom=True)
        for spine in ['top', 'left', 'right']:
            self.dax.spines[spine].set_visible(False)

    def _format_ax_ticks(self):
        for ax in [self.ax.xaxis, self.ax.yaxis, self.dax.xaxis]:
            prev_labs = [int(i.get_text()) for i in ax.get_ticklabels()]
            ax.set_ticklabels([self.mapping[li].split(' ')[-1] for li in prev_labs])

    def _format_ax(self):
        self._format_ax_ticks()
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

    def _format_fig(self):
        self.fig.supxlabel('Predicted pianist', y=0.03)
        self.fig.supylabel('Actual pianist')
        self.fig.subplots_adjust(top=1, bottom=0.075, left=0.1, right=0.95)
        pos = self.ax.get_position()
        self.ax.set_position([pos.x0, pos.y0, pos.width, pos.height])
        cax = self.ax.figure.axes[-1]
        cax.set_position([0.915, 0.17, 0.2, 0.575])
        cax.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        cax.set_ylim(0.1, 100)
        cax.set_yticklabels(['0.1', '1', '5', '10', '50', '100'])
        cax.tick_params(axis='y', which='minor', width=None, right=False)
        pos = self.dax.get_position()
        self.dax.set_position([pos.x0, pos.y0 + 0.055, pos.width, pos.height - 0.06])
        self.dax.set(xticklabels=[])


if __name__ == '__main__':
    pass
