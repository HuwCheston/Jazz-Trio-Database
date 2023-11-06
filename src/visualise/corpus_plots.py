#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Plotting classes for corpus description, e.g. F-measures, API scraping results etc."""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import src.visualise.visualise_utils as vutils
from src import utils

__all__ = [
    'BarPlotFScores', 'TimelinePlotBandleaders', 'BarPlotBandleaderDuration', 'BarPlotLastFMStreams',
    'BarPlotSubjectiveRatings'
]


class BarPlotFScores(vutils.BasePlot):
    def __init__(self, **kwargs):
        self.corpus_title = kwargs.get('corpus_title', 'corpus_chronology')
        super().__init__(figure_title=fr'corpus_plots\barplot_fscores_{self.corpus_title}',
                         **kwargs)
        self.df = (
            pd.concat(self._format_df(), axis=0)
            .reset_index(drop=True)
            .set_index('instrument')
            .loc[['mix', *utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys()]]
            .reset_index(drop=False)
        )
        self.fig, self.ax = plt.subplots(nrows=1, ncols=1, figsize=(vutils.WIDTH / 2, vutils.WIDTH / 3))

    def _format_df(self):
        fpath = rf'{utils.get_project_root()}\references\parameter_optimisation\{self.corpus_title}'
        cols = ['mbz_id', 'instrument', 'f_score']
        for f in os.listdir(fpath):
            if not f.endswith('.csv'):
                continue
            df = pd.read_csv(fr'{fpath}\{f}')
            df = df[df['iterations'] == df['iterations'].max()][cols]
            yield df

    def _create_plot(self):
        sns.barplot(
            data=self.df, x='instrument', hue='instrument', alpha=vutils.ALPHA, dodge=False, y='f_score', ax=self.ax,
            palette=[vutils.WHITE, *vutils.RGB], edgecolor=None, lw=0, n_boot=vutils.N_BOOT, errorbar=('ci', 95),
            seed=1, capsize=0.1, width=0.8, errwidth=2, errcolor=vutils.BLACK, zorder=2, estimator=np.mean
        )
        sns.barplot(
            data=self.df, x='instrument', hue='instrument', errorbar=None, dodge=False, y='f_score', ax=self.ax,
            palette=[vutils.WHITE, vutils.WHITE, vutils.WHITE, vutils.WHITE], edgecolor=vutils.BLACK,
            lw=vutils.LINEWIDTH, ls=vutils.LINESTYLE,
        )
        sns.stripplot(
            data=self.df, x='instrument', y='f_score', s=4, jitter=True, alpha=vutils.ALPHA, marker='o',
            edgecolor=vutils.BLACK, color=vutils.BLACK, ax=self.ax
        )

    def _format_ax(self):
        self.ax.get_legend().remove()
        self.ax.set(xticklabels=['Mixture', 'Piano', 'Bass', 'Drums'], xlabel='Instrument', ylabel='$F$-Measure',
                    ylim=(0, 1.02))
        plt.setp(self.ax.spines.values(), linewidth=vutils.LINEWIDTH)
        self.ax.tick_params(axis='both', width=vutils.TICKWIDTH)

    def _format_fig(self):
        self.fig.tight_layout()


class BarPlotSubjectiveRatings(vutils.BasePlot):
    BAR_KWS = dict(
        zorder=1, edgecolor=vutils.BLACK, lw=vutils.LINEWIDTH, ls=vutils.LINESTYLE,
        n_boot=vutils.N_BOOT, errorbar=('ci', 95), seed=1, capsize=0.1, width=0.8,
        errwidth=2, errcolor=vutils.BLACK, estimator=np.mean,
    )
    COLS = [vutils.WHITE, *vutils.RGB, vutils.WHITE, *vutils.RGB]
    HATCHES = [*['' for _ in range(4)], *['/' for _ in range(4)]]

    def __init__(self, **kwargs):
        self.corpus_title = kwargs.get('corpus_title', 'corpus_chronology')
        super().__init__(figure_title=fr'corpus_plots\barplot_subjective_rating_{self.corpus_title}',
                         **kwargs)
        self.df = self._format_df()
        self.fig, self.ax = plt.subplots(nrows=1, ncols=1, figsize=(vutils.WIDTH / 2, vutils.WIDTH / 3))

    def _format_df(self):
        corp = utils.CorpusMaker.from_excel(self.corpus_title)
        df = pd.DataFrame([track for track in corp.tracks if not np.isnan(track['rating_bass_audio'])])
        columns = [c for c in df.columns if 'rating' in c and 'comments' not in c]
        clean = (
            df.drop(columns=['rating_comments'])
            .melt(id_vars=['mbz_id', ], value_vars=columns, value_name='rating')
        )
        clean['is_audio'] = ~clean['variable'].str.contains('audio')
        for st in ['rating_', '_audio', '_detection']:
            clean['variable'] = clean['variable'].str.replace(st, '')
        return (
            clean.reset_index(drop=True)
            .set_index('variable')
            .loc[['mix', *utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys()]]
            .reset_index(drop=False)
        )

    def _create_plot(self):
        return sns.barplot(data=self.df, x='variable', y='rating', hue='is_audio', ax=self.ax, **self.BAR_KWS)

    @staticmethod
    def _get_color(hex_code: str):
        return [*[round(i / 255, 2) for i in [int(hex_code.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4)]], vutils.ALPHA]

    @staticmethod
    def _get_legend_handles():
        p1 = mpl.patches.Patch(facecolor=vutils.BLACK, alpha=vutils.ALPHA, hatch='', label='Audio')
        p2 = mpl.patches.Patch(facecolor=vutils.BLACK, alpha=vutils.ALPHA, hatch='/', label='Detection')
        return [p1, p2]

    def _format_ax(self):
        for col, patch, hatch in zip(self.COLS, self.ax.patches, self.HATCHES):
            patch.set_facecolor(self._get_color(col))
            patch.set_hatch(hatch)
        self.ax.get_legend().remove()
        self.ax.legend(handles=self._get_legend_handles(), loc='lower left', frameon=True, framealpha=1,
                       edgecolor=vutils.BLACK)
        self.ax.set(
            xticklabels=['Mixture', *[i.title() for i in utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys()]],
            xlabel='Instrument', ylabel='Rating', ylim=(0, 3.06)
        )
        plt.setp(self.ax.spines.values(), linewidth=vutils.LINEWIDTH)
        self.ax.tick_params(axis='both', width=vutils.TICKWIDTH)

    def _format_fig(self):
        self.fig.tight_layout()


class TimelinePlotBandleaders(vutils.BasePlot):
    SCATTER_KWS = dict(s=50, marker='x', color=vutils.BLACK, alpha=1, zorder=1, label='Recording')
    TEXT_KWS = dict(va='center', ha='left', zorder=2, fontsize=vutils.FONTSIZE / 1.2)
    BAR_KWS = dict(edgecolor=vutils.BLACK, zorder=0, label=None)
    PAL = sns.cubehelix_palette(dark=1/3, gamma=.3, light=2/3, start=2, n_colors=10, as_cmap=False)

    def __init__(self, bandleaders_df: pd.DataFrame, **kwargs):
        self.corpus_title = 'corpus_chronology'
        self.timeline_df = self._format_timeline_df(bandleaders_df)
        self.corpus_df = self._format_corpus_df(bandleaders_df)
        super().__init__(figure_title=fr'corpus_plots\timeline_bandleaders_{self.corpus_title}', **kwargs)
        self.fig, self.ax = plt.subplots(nrows=1, ncols=1, figsize=(vutils.WIDTH, vutils.WIDTH / 3))

    @staticmethod
    def _format_timeline_df(bandleaders_df):
        bandleaders_df['date_fmt'] = bandleaders_df['recording_date_estimate'].dt.year + \
                                     (bandleaders_df['recording_date_estimate'].dt.month / 12)
        timeline = bandleaders_df.groupby('bandleader').agg(dict(date_fmt=['min', 'max'], birth='mean', death='median'))
        timeline.columns = timeline.columns.droplevel()
        timeline['alive'] = timeline['median'].apply(lambda x: x.year < 2023)
        timeline['diff'] = timeline['max'] - timeline['min']
        return timeline.sort_values(by='min', ascending=False).reset_index(drop=False)

    def _format_corpus_df(self, bandleaders_df):
        in_corpus = bandleaders_df[bandleaders_df['in_corpus']]
        mapping = {b: i for b, i in zip(self.timeline_df['bandleader'].to_list(), self.timeline_df.index.to_list())}
        in_corpus['mapping'] = in_corpus['bandleader'].map(mapping) - 0.25
        in_corpus['mapping'] += np.random.uniform(-0.1, 0.1, len(in_corpus))
        return in_corpus

    @staticmethod
    def _get_birth_death_range(birth, death, alive):
        if not alive:
            return f'(b. {birth})'
        else:
            return f'({birth}–{death})'

    def _create_plot(self):
        for (idx, row), col in zip(self.timeline_df.iterrows(), self.PAL):
            self.ax.broken_barh([(row['min'], row['max'] - row['min'])], (idx - 0.5, 0.5), color=col, **self.BAR_KWS)
            dates = self._get_birth_death_range(
                row['mean'].year, row['median'].year, True if row['bandleader'] == 'Ahmad Jamal' else row['alive']
            )
            self.ax.text(row['min'], idx + 0.2, f"{row['bandleader']} {dates}", **self.TEXT_KWS)
        sns.scatterplot(data=self.corpus_df, x='date_fmt', y='mapping', ax=self.ax, **self.SCATTER_KWS)

    def _format_ax(self):
        plt.setp(self.ax.spines.values(), linewidth=vutils.LINEWIDTH, color=vutils.BLACK)
        self.ax.tick_params(axis='both', width=vutils.TICKWIDTH, color=vutils.BLACK)
        sns.move_legend(self.ax, loc='upper right', frameon=True, framealpha=1, edgecolor=vutils.BLACK)
        self.ax.get_legend().get_frame().set_linewidth(vutils.LINEWIDTH)
        self.ax.grid(visible=True, which='major', axis='x', **vutils.GRID_KWS)
        self.ax.set(yticks=[], ylabel="", xlabel='Date')
        sns.despine(ax=self.ax, left=True, top=True, right=True, bottom=False)

    def _format_fig(self):
        self.fig.tight_layout()


class BarPlotBandleaderDuration(vutils.BasePlot):
    BAR_KWS = dict(
        edgecolor=vutils.BLACK, lw=vutils.LINEWIDTH, ls=vutils.LINESTYLE, zorder=5, dodge=False,
        palette=[vutils.RED, vutils.GREEN], estimator=np.sum,
    )
    bandleaders = ['Bill Evans', 'Ahmad Jamal', 'Bud Powell', 'Oscar Peterson', 'Keith Jarrett', 'Tommy Flanagan',
                   'Junior Mance', 'Kenny Barron', 'John Hicks', 'McCoy Tyner']

    def __init__(self, cleaned_df: pd.DataFrame, **kwargs):
        self.corpus_title = 'corpus_chronology'
        self.df = self._format_df(cleaned_df)
        super().__init__(figure_title=fr'corpus_plots\barplot_bandleader_duration_{self.corpus_title}', **kwargs)
        self.fig, self.ax = plt.subplots(1, 1, figsize=(vutils.WIDTH / 2, vutils.WIDTH / 2))

    @staticmethod
    def initials(a):
        """Converts a list of strings of arbitrary length to their first initial"""
        if len(a) == 0:
            return ''
        return ''.join(map(lambda li: li[0] + '.', [a[0]]))

    def abbreviate(self, s):
        """Abbreviates a name to surname, first initial"""
        return f'{s.split()[-1]}, {self.initials(s.split()[0:-1])}'

    def _format_df(self, cleaned_df):
        small_df = (
            cleaned_df.groupby('bandleader')
            .agg({'recording_title': 'count', 'recording_length_': 'sum'})
            .reset_index(drop=False)
        )
        small_df['bandleader_'] = (
            small_df['bandleader']
            .apply(self.abbreviate)
            .apply(lambda s: "".join(c for c in s if ord(c) < 128))
        )
        small_df['in_corpus'] = small_df['bandleader'].isin(self.bandleaders)
        small_df['recording_length_'] = (small_df['recording_length_'].dt.total_seconds()) / 3600
        return small_df.sort_values(by='recording_length_', ascending=False)

    def _create_plot(self):
        g = sns.barplot(
            data=self.df, x='bandleader_', y='recording_length_', ax=self.ax, hue='in_corpus', **self.BAR_KWS
        )
        for patch, (idx_, row_) in zip(g.patches, self.df.iterrows()):
            if not row_['in_corpus']:
                patch.set_hatch(vutils.HATCHES[0])

    def _format_ax(self):
        self.ax.set_xticks(
            self.ax.get_xticks(), self.ax.get_xticklabels(), rotation=45, ha='right',
            fontsize=vutils.FONTSIZE / 1.5, rotation_mode="anchor"
        )
        self.ax.set(ylabel='Total recording duration (hours)', xlabel='Bandleader', xlim=(-0.5, self.ax.get_xlim()[-1]))
        self.ax.grid(visible=True, which='major', axis='y', zorder=0, **vutils.GRID_KWS)
        self.ax.tick_params(width=vutils.TICKWIDTH, which='both')
        self.ax.set_yscale('log')
        self.ax.minorticks_off()
        self.ax.set_yticklabels([1, 10, 100])
        self.ax.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        plt.setp(self.ax.spines.values(), linewidth=vutils.LINEWIDTH)
        self.ax.axvspan(
            -0.5, 9, 0, self.ax.get_ylim()[1], alpha=vutils.ALPHA, color=vutils.BLACK,
            lw=vutils.LINEWIDTH, ls=vutils.LINESTYLE
        )
        self.ax.text(9.5, 50, r'$\it{Top}$ $\mathit{10}$', rotation=-90, va='center', ha='center')
        hand, lab = self.ax.get_legend_handles_labels()
        self.ax.get_legend().remove()
        p = mpl.patches.Patch(facecolor=vutils.RED, lw=2, hatch=vutils.HATCHES[0], edgecolor=vutils.BLACK)
        self.ax.legend(
            [p, hand[-1]], lab, loc='upper right', frameon=True, title='Included?', framealpha=1, edgecolor=vutils.BLACK
        )
        self.ax.get_legend().get_frame().set_linewidth(vutils.LINEWIDTH)

    def _format_fig(self):
        self.fig.subplots_adjust(top=0.95, bottom=0.15, left=0.1, right=0.95)


class BarPlotLastFMStreams(vutils.BasePlot):
    BAR_KWS = dict(edgecolor=vutils.BLACK, lw=vutils.LINEWIDTH, ls=vutils.LINESTYLE, zorder=5)

    def __init__(self, streams_df: pd.DataFrame, **kwargs):
        self.corpus_title = 'corpus_chronology'
        super().__init__(figure_title=fr'corpus_plots\batplot_lastfmstreams_{self.corpus_title}', **kwargs)
        self.df = self._format_df(streams_df.copy(deep=True))
        self.fig, self.ax = plt.subplots(nrows=1, ncols=1, figsize=(vutils.WIDTH / 2, vutils.WIDTH / 2))

    @staticmethod
    def _format_df(streams_df):
        # for col in ['track_1_plays', 'track_2_plays', 'track_3_plays']:
        #     streams_df[col] /= 1000
        streams_df['combined_plays'] = streams_df[['track_1_plays', 'track_2_plays', 'track_3_plays']].sum(axis=1)
        return streams_df.sort_values(by='combined_plays', ascending=False).iloc[:20]

    def _create_plot(self):
        self.ax.barh(self.df['name'], self.df['track_1_plays'], label='1st', **self.BAR_KWS)
        self.ax.barh(
            self.df['name'], self.df['track_2_plays'], left=self.df['track_1_plays'], label='2nd', **self.BAR_KWS
        )
        self.ax.barh(
            self.df['name'], self.df['track_3_plays'], left=self.df['track_1_plays'] + self.df['track_2_plays'],
            label='3rd', **self.BAR_KWS
        )

    def _format_ax(self):
        self.ax.grid(visible=True, which='major', axis='x', zorder=0, **vutils.GRID_KWS)
        self.ax.set(
            xlabel='Streams (millions)', ylabel='', xticks=[0, 1000000, 2000000, 3000000, 4000000],
            xticklabels=["0M", '1M', '2M', '3M', '4M']
        )
        self.ax.set_yticklabels(self.ax.get_yticklabels(), fontsize=vutils.FONTSIZE / 1.3)
        self.ax.set_xticklabels(self.ax.get_xticklabels(), fontsize=vutils.FONTSIZE / 1.3)
        self.ax.legend(loc='upper right', frameon=True, framealpha=1, edgecolor=vutils.BLACK, title='Track')
        self.ax.get_legend().get_frame().set_linewidth(vutils.LINEWIDTH)
        plt.setp(self.ax.spines.values(), linewidth=vutils.LINEWIDTH, color=vutils.BLACK)
        self.ax.tick_params(axis='both', width=vutils.TICKWIDTH, color=vutils.BLACK)

    def _format_fig(self):
        self.fig.tight_layout()
