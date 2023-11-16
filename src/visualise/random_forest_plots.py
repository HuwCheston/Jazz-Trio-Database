#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Plotting classes for random forest model, e.g. heatmaps, feature importance."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

import src.visualise.visualise_utils as vutils

__all__ = [
    'BarPlotCategoryImportances', 'BarPlotFeatureImportances', 'HeatMapFeatureCorrelation', 'HeatMapPredictionProb'
]

PREDICTORS_CATEGORIES = {
    'Swing': ['bur_log_mean', 'bur_log_std'],
    'Complexity': ['lz77_mean', 'lz77_std', 'n_onsets_mean', 'n_onsets_std'],
    'Feel': ['piano_bass_prop_async_nanmean', 'piano_drums_prop_async_nanmean',
             'piano_bass_prop_async_nanstd', 'piano_drums_prop_async_nanstd'],
    'Interaction': ['self_coupling', 'coupling_drums', 'coupling_bass', 'coupling_piano_drums', 'coupling_piano_bass'],
    'Tempo': ['rolling_std_median', 'tempo', 'tempo_slope', 'tempo_drift']
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
    'tempo_drift': 'Tempo drift',
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
        super().__init__(figure_title=fr'random_forest_plots\barplot_category_importances_{self.corpus_title}', **kwargs)
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


class HeatMapPredictionProb(vutils.BasePlot):
    def __init__(self, prob_df, **kwargs):
        self.corpus_title = 'corpus_chronology'
        super().__init__(figure_title=fr'random_forest_plots\heatmap_prediction_prob_{self.corpus_title}', **kwargs)
        self.hm = self._format_df(prob_df)
        self.fig = plt.figure(figsize=(vutils.WIDTH / 2, vutils.WIDTH/2))
        gs0 = mpl.gridspec.GridSpec(1, 2, width_ratios=[20, 1], hspace=0.0, height_ratios=[1])
        self.ax = self.fig.add_subplot(gs0[0])

    @staticmethod
    def _format_df(prob_df):
        labs = [l.split(' ')[-1] for l in prob_df['actual'].unique()]
        hm = pd.DataFrame(confusion_matrix(prob_df['actual'], prob_df['prediction'], normalize='true'))
        hm *= 100
        hm.index = labs
        hm.columns = labs
        return hm.sort_index(ascending=False).reindex(sorted(hm.columns), axis=1)

    def _create_plot(self):
        sns.heatmap(
            self.hm, ax=self.ax, cmap="Purples", linecolor=vutils.WHITE, square=True,
            annot=True, fmt='.0f', linewidths=vutils.LINEWIDTH/2, vmin=0, vmax=100,
            cbar_kws=dict(
                label='Predictions (%)',
                use_gridspec=False, location="right", pad=0.2, shrink=0.725
            )
        )

    def _format_annotations(self):
        for t in self.ax.texts:
            if float(t.get_text()) > 25:
                t.set_text(t.get_text(),)
                t.set_fontsize(vutils.FONTSIZE / 1.25)
            else:
                t.set_text('')

    def _format_ax(self):
        self._format_annotations()
        self.ax.set(ylabel='Actual pianist', xlabel='Predicted pianist')
        cax = self.ax.figure.axes[-1]
        for a in [self.ax, cax]:
            for spine in a.spines.values():
                spine.set_visible(True)
                spine.set_color(vutils.BLACK)
            plt.setp(a.spines.values(), linewidth=vutils.LINEWIDTH, color=vutils.BLACK)
            a.tick_params(axis='both', width=vutils.TICKWIDTH, color=vutils.BLACK)

    def _format_fig(self):
        self.fig.subplots_adjust(top=1, bottom=0.1, left=0.2, right=0.9)
        self.ax.figure.axes[-1].set_position([0.85, 0.245, 0.2, 0.6075])


if __name__ == '__main__':
    pass
