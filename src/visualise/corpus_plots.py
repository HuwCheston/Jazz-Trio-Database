#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Plotting classes for corpus description, e.g. F-measures, API scraping results etc."""

import copy
import os
import time

import librosa
import librosa.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mir_eval.util import match_events

import src.visualise.visualise_utils as vutils
from src import utils
from src.detect.detect_utils import FREQUENCY_BANDS

__all__ = [
    'BarPlotFScores', 'TimelinePlotBandleaders', 'BarPlotBandleaderDuration', 'BarPlotLastFMStreams',
    'BarPlotSubjectiveRatings', 'BoxPlotRecordingLength', 'SpecPlotBands', 'CountPlotPanning',
    'LinePlotOptimizationIterations', 'BarPlotCorpusDuration'
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
    img_loc = fr'{utils.get_project_root()}\references\images\musicians'
    SCATTER_KWS = dict(s=50, marker='x', color=vutils.BLACK, alpha=1, zorder=1, label='Recording')
    TEXT_KWS = dict(va='center', ha='left', zorder=2, fontsize=vutils.FONTSIZE / 1.2)
    BAR_KWS = dict(edgecolor=vutils.BLACK, zorder=0, label=None)
    PAL = sns.cubehelix_palette(dark=1/3, gamma=.3, light=2/3, start=2, n_colors=10, as_cmap=False)

    def __init__(self, bandleaders_df: pd.DataFrame, **kwargs):
        self.corpus_title = 'corpus_chronology'
        self.timeline_df = self._format_timeline_df(bandleaders_df)
        self.corpus_df = self._format_corpus_df(bandleaders_df)
        super().__init__(figure_title=fr'corpus_plots\timeline_bandleaders_{self.corpus_title}', **kwargs)
        self.fig, self.ax = plt.subplots(nrows=1, ncols=1, figsize=(vutils.WIDTH, vutils.WIDTH / 2))

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
            self._add_pianist_image(row['bandleader'], row['min'], idx)
        sns.scatterplot(data=self.corpus_df, x='date_fmt', y='mapping', ax=self.ax, **self.SCATTER_KWS)

    def _add_pianist_image(self, bandleader_name, x, y):
        fpath = fr'{self.img_loc}\{bandleader_name.replace(" ", "_").lower()}.png'
        img = mpl.offsetbox.OffsetImage(
            plt.imread(fpath), clip_on=False, transform=self.ax.transAxes, zoom=0.35
        )
        ab = mpl.offsetbox.AnnotationBbox(
            img, (x - 1.5, y - 0.05), xycoords='data', clip_on=False, transform=self.ax.transAxes,
            annotation_clip=False, bboxprops=dict(edgecolor='none', facecolor='none')
        )
        self.ax.add_artist(ab)

    def _format_ax(self):
        plt.setp(self.ax.spines.values(), linewidth=vutils.LINEWIDTH, color=vutils.BLACK)
        self.ax.tick_params(axis='both', width=vutils.TICKWIDTH, color=vutils.BLACK)
        sns.move_legend(self.ax, loc='upper right', frameon=True, framealpha=1, edgecolor=vutils.BLACK)
        self.ax.get_legend().get_frame().set_linewidth(vutils.LINEWIDTH)
        self.ax.grid(visible=True, which='major', axis='x', **vutils.GRID_KWS)
        self.ax.set(yticks=[], ylabel="", xlabel='Date')
        sns.despine(ax=self.ax, left=True, top=True, right=True, bottom=False)

    def _format_fig(self):
        self.fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1)


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
    PAL = sns.cubehelix_palette(dark=1/3, gamma=.3, light=2/3, start=2, n_colors=20, as_cmap=False)
    BAR_KWS = dict(edgecolor=vutils.BLACK, lw=vutils.LINEWIDTH, ls=vutils.LINESTYLE, zorder=5, palette=reversed(PAL))

    def __init__(self, streams_df: pd.DataFrame, **kwargs):
        self.corpus_title = 'corpus_chronology'
        super().__init__(figure_title=fr'corpus_plots\batplot_lastfmstreams_total_{self.corpus_title}', **kwargs)
        self.df = streams_df.copy(deep=True).iloc[:20]
        self.fig, self.ax = plt.subplots(nrows=1, ncols=1, figsize=(vutils.WIDTH / 2, vutils.WIDTH / 2))

    def _create_plot(self):
        sns.barplot(data=self.df, x='playcount', y='name', ax=self.ax, **self.BAR_KWS)

    def _format_ax(self):
        self.ax.grid(visible=True, which='major', axis='x', zorder=0, **vutils.GRID_KWS)
        self.ax.set(
            xlabel='Streams (millions)', ylabel='', xticks=[1000000, 5000000, 10000000, ],
            xticklabels=['1M', '5M', '10M']
        )
        self.ax.set_yticklabels(self.ax.get_yticklabels(), fontsize=vutils.FONTSIZE / 1.3)
        self.ax.set_xticklabels(self.ax.get_xticklabels(), fontsize=vutils.FONTSIZE / 1.3)
        # self.ax.legend(loc='upper right', frameon=True, framealpha=1, edgecolor=vutils.BLACK, title='Track')
        # self.ax.get_legend().get_frame().set_linewidth(vutils.LINEWIDTH)
        plt.setp(self.ax.spines.values(), linewidth=vutils.LINEWIDTH, color=vutils.BLACK)
        self.ax.tick_params(axis='both', width=vutils.TICKWIDTH, color=vutils.BLACK)

    def _format_fig(self):
        self.fig.tight_layout()


class CountPlotPanning(vutils.BasePlot):
    LEGEND_KWS = dict(frameon=True, framealpha=1, edgecolor=vutils.BLACK)

    def __init__(self, track_df: pd.DataFrame, **kwargs):
        self.corpus_title = 'corpus_chronology'
        super().__init__(figure_title=fr'corpus_plots\countplot_panning_{self.corpus_title}', **kwargs)
        self.df = self._format_df(track_df)
        self.fig, self.ax = plt.subplots(1, 1, figsize=(vutils.WIDTH / 2, vutils.WIDTH / 2))

    @staticmethod
    def _format_df(track_df):
        return (
            track_df['channel_overrides']
            .apply(pd.Series, dtype=object)
            .apply(lambda x: x.fillna('c').value_counts())
            .reset_index(drop=False)
            .melt(id_vars='index')
            .pivot_table(index='index', columns=['variable'])
            .reindex(columns=utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys(), level=1)
            .reindex(index=['l', 'c', 'r'])
        )

    def _create_plot(self):
        self.df.index = self.df.index.str.title()
        return self.df.plot(
            kind='bar', stacked=True, ax=self.ax, xlabel='Audio channel', ylabel='Number of tracks', color=vutils.RGB,
            zorder=10, lw=vutils.LINEWIDTH, edgecolor=vutils.BLACK
        )

    def _add_track_numbers(self):
        patches = np.array(self.ax.patches)[[6, 7, 8]]
        for bar, (idx, row) in zip(patches, self.df.iterrows()):
            x, y = bar.get_xy()
            self.ax.text(x + (bar.get_width() / 2), y + bar.get_height(), f'N = {row.sum()}', ha='center', va='bottom')

    def _format_ax(self):
        hand, _ = self.ax.get_legend_handles_labels()
        self.ax.set_xticklabels(self.ax.get_xticklabels(), rotation=0)
        self.ax.legend(hand, ['Piano', 'Bass', 'Drums'], title='Instrument', loc='upper right', **self.LEGEND_KWS)
        plt.setp(self.ax.spines.values(), linewidth=vutils.LINEWIDTH, color=vutils.BLACK)
        self.ax.tick_params(axis='both', width=vutils.TICKWIDTH, color=vutils.BLACK)
        self.ax.grid(zorder=0, axis='y', **vutils.GRID_KWS)
        self._add_track_numbers()

    def _format_fig(self):
        self.fig.tight_layout()


class SpecPlotBands(vutils.BasePlot):
    TRACK_LEN = 5

    def __init__(self, track: str, **kwargs):
        self.corpus_title = 'corpus_chronology'
        self.track = track

        self.fname = track['fname']
        super().__init__(figure_title=fr'corpus_plots\specplot_bands_{self.corpus_title}', **kwargs)
        self.df = self._format_df()
        self.fig, self.ax = plt.subplots(1, 1, figsize=(vutils.WIDTH / 2, vutils.WIDTH / 2))

    def _format_df(self):
        y, _ = librosa.load(fr"{utils.get_project_root()}\data\raw\audio\{self.fname}.wav", sr=utils.SAMPLE_RATE)
        y = y[:utils.SAMPLE_RATE * self.TRACK_LEN]
        return librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

    def _create_plot(self):
        return librosa.display.specshow(
            self.df, y_axis='log', sr=utils.SAMPLE_RATE, x_axis='s', ax=self.ax,
            auto_aspect=False, cmap=sns.color_palette('Greys', as_cmap=True)
        )

    def _add_bands(self):
        for (instr, bands), col in zip(FREQUENCY_BANDS.items(), vutils.RGB):
            self.ax.axhspan(
                ymin=bands['fmin'], ymax=bands['fmax'], xmin=0, xmax=self.TRACK_LEN,
                alpha=vutils.ALPHA, color=col, label=instr.title()
            )

    def _format_ax(self):
        self._add_bands()
        self.ax.legend(loc='upper right', title='Instrument', frameon=True, framealpha=1, edgecolor=vutils.BLACK)
        self.ax.set(ylim=(0, FREQUENCY_BANDS['drums']['fmax']), xticks=range(self.TRACK_LEN + 1),
                    ylabel='Frequency (Hz)',
                    title=f"{self.track['pianist']} – {self.track['track_name']} ({self.track['recording_year']})")
        plt.setp(self.ax.spines.values(), linewidth=vutils.LINEWIDTH, color=vutils.BLACK)
        self.ax.tick_params(axis='both', width=vutils.TICKWIDTH, color=vutils.BLACK)

    def _format_fig(self):
        self.fig.tight_layout()


class BoxPlotRecordingLength(vutils.BasePlot):
    img_loc = fr'{utils.get_project_root()}\references\images\musicians'
    PAL = sns.cubehelix_palette(dark=1/3, gamma=.3, light=2/3, start=2, n_colors=10, as_cmap=False)
    TICKS = np.linspace(0, 2100, 8)

    def __init__(self, cleaned_df: pd.DataFrame, **kwargs):
        self.corpus_title = 'corpus_chronology'
        super().__init__(figure_title=fr'corpus_plots\boxplot_recording_length_{self.corpus_title}', **kwargs)
        self.df = cleaned_df.sort_values(by='birth')
        self.fig, self.ax = plt.subplots(1, 1, figsize=(vutils.WIDTH, vutils.WIDTH / 2))

    def _create_plot(self):
        sns.boxplot(
            self.df.sort_values(by='birth'), x="duration", y="piano",
            whis=[0, 100], width=.6, palette=self.PAL, ax=self.ax,
            linewidth=vutils.LINEWIDTH, color=vutils.BLACK
        )
        # Add in points to show each observation
        sns.stripplot(self.df, x="duration", y="piano", size=4, color=vutils.BLACK)

    @staticmethod
    def _format_time(nos, fmt: str = '%M:%S'):
        return time.strftime(fmt, time.gmtime(nos))

    def _format_bandleader(self, bl):
        d = self.df[self.df['piano'] == bl].iloc[0]
        if bl == 'Ahmad Jamal' or d['death'].year < 2023:
            return f"{bl}\n({d['birth'].year}–{d['death'].year})"
        else:
            return f"{bl}\n(b. {d['birth'].year})"

    def _add_bandleader_images(self, bl, y):
        fpath = fr'{self.img_loc}\{bl.replace(" ", "_").lower()}.png'
        img = mpl.offsetbox.OffsetImage(
            plt.imread(fpath), clip_on=False, transform=self.ax.transAxes, zoom=0.5
        )
        ab = mpl.offsetbox.AnnotationBbox(
            img, (-75, y - 0.05), xycoords='data', clip_on=False, transform=self.ax.transAxes,
            annotation_clip=False, bboxprops=dict(edgecolor='none', facecolor='none')
        )
        self.ax.add_artist(ab)

    def _add_number_of_tracks(self, bl, y):
        tracks = self.df[self.df['piano'] == bl]
        ti = round(tracks['duration'].sum() / 3600, 2)
        x = (tracks['duration'].max() - tracks['duration'].min()) * 0.85
        self.ax.text(x, y - 0.35, f'Total hours: {ti}', va='center')

    def _format_ax(self):
        # Tweak the visual presentation
        self.ax.xaxis.grid(True)
        self.ax.set(
            ylabel="Pianist", xlabel='Track duration (MM:SS)', xlim=(0, 2200), xticks=self.TICKS,
            xticklabels=[self._format_time(ti) for ti in self.TICKS],
            yticklabels=[self._format_bandleader(bl.get_text()) for bl in self.ax.get_yticklabels()]
        )
        for num, pi in enumerate(self.df['piano'].unique()):
            self._add_bandleader_images(pi, num)
            self._add_number_of_tracks(pi, num)
        self.ax.tick_params(axis='y', which='both', pad=65)
        sns.despine(trim=True, left=True)
        self.ax.tick_params(width=vutils.TICKWIDTH, which='both')
        plt.setp(self.ax.spines.values(), linewidth=vutils.LINEWIDTH)

    def _format_fig(self):
        self.fig.subplots_adjust(top=0.95, bottom=0.1, left=0.2, right=1)


class RainPlotAlgoHumanOnset(vutils.BasePlot):
    BP_KWS = dict(
        patch_artist=True, vert=False, sym='', widths=0.25, manage_ticks=False,
        zorder=5, boxprops=dict(facecolor=vutils.WHITE, alpha=vutils.ALPHA)
    )
    VP_KWS = dict(vert=False, showmeans=False, showextrema=False, showmedians=False)
    SP_KWS = dict(s=1.5, zorder=5)
    INSTRS = ['mix', *utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys()]
    XLABEL = 'Difference between onsets (algorithm – human, ms)'

    def __init__(self, all_onsets, **kwargs):
        self.corpus_title = 'corpus_chronology'
        super().__init__(figure_title=fr'corpus_plots\rainplot_algohuman_onsets_{self.corpus_title}', **kwargs)
        self.df = pd.DataFrame(self._format_df(all_onsets))
        self.vals = [grp['async'].values * 1000 for _, grp in self.df.groupby('instr', sort=False)]
        self.fig, self.ax = plt.subplots(1, 1, figsize=(vutils.WIDTH, vutils.WIDTH / 2))

    @staticmethod
    def _format_df(ons):
        for an in [o for o in ons if o.item['has_annotations']]:
            for instr in ['mix', *utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys()]:
                estimate = an.ons[instr]
                fname = rf'{an.references_dir}\manual_annotation\{an.item["fname"]}_{instr}.txt'
                ref = np.loadtxt(fname, ndmin=1, usecols=0)
                matched = match_events(ref, estimate, window=0.05)
                for asy in np.array([estimate[e] - ref[r] for r, e in matched]):
                    yield {'instr': instr, 'async': asy}

    def _create_boxplot(self):
        bp = self.ax.boxplot(self.vals, positions=[5.75, 3.75, 1.75, -0.25], **self.BP_KWS)
        # Change to the desired color and add transparency
        for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
            try:
                plt.setp(bp[item], edgecolor=vutils.BLACK, linewidth=vutils.LINEWIDTH, alpha=1, zorder=5)
            except AttributeError:
                plt.setp(bp[item], color=vutils.BLACK, linewidth=vutils.LINEWIDTH, alpha=1, zorder=5)
        for median, col in zip(bp['medians'], [vutils.BLACK, *vutils.RGB]):
            median.set_color(col)

    def _create_violinplot(self):
        vp = self.ax.violinplot(self.vals, positions=[6, 4, 2, 0], **self.VP_KWS)
        for b, col, idx in zip(vp['bodies'], [vutils.WHITE, *vutils.RGB], reversed(range(0, 7, 2))):
            # Modify it so we only see the upper half of the violin plot
            b.get_paths()[0].vertices[:, 1] = np.clip(b.get_paths()[0].vertices[:, 1], idx, idx + 2)
            b.set_alpha(1)
            b.set_facecolor(col)
            b.set_edgecolor(vutils.BLACK)
            b.set_linewidth(vutils.LINEWIDTH)
            b.set_zorder(5)

    def _create_scatterplot(self):
        # Scatterplot data
        for features, col, idx in zip(self.vals, [vutils.BLACK, *vutils.RGB], reversed(range(0, 7, 2))):
            # Add jitter effect so the features do not overlap on the y-axis
            y = np.full(len(features), idx - .6)
            idxs = np.arange(len(y))
            out = y.astype(float)
            out.flat[idxs] += np.random.uniform(low=-.05, high=.05, size=len(idxs))
            y = out
            self.ax.scatter(features, y, color=col, **self.SP_KWS)

    def _create_plot(self):
        self._create_boxplot()
        self._create_violinplot()
        self._create_scatterplot()

    def _format_ax(self):
        wlab = r'Window boundaries ($\pm$ 50 ms)'
        for x, ls, lab in zip((-50, 0, 50), (vutils.LINESTYLE, 'dashed', 'dashed'), (None, wlab, None)):
            self.ax.axvline(x, 0, 1, color=vutils.BLACK, lw=vutils.LINEWIDTH, ls=ls, zorder=0, label=lab)
        ylab = [f'{ins.title()}\nN = {len(self.df[self.df["instr"] == ins])}' for ins in self.INSTRS]
        self.ax.set(
            ylim=(-1, 7), xlim=(-55, 55), xticks=[-50, -25, 0, 25, 50], yticks=[-0.25, 1.75, 3.75, 5.75],
            yticklabels=reversed(ylab), xlabel=self.XLABEL, ylabel='Instrument'
        )
        for tick, col in zip(reversed(self.ax.get_yticklabels()), [vutils.BLACK, *vutils.RGB]):
            tick.set_color(col)
        plt.setp(self.ax.spines.values(), linewidth=vutils.LINEWIDTH, color=vutils.BLACK)
        self.ax.tick_params(axis='both', width=vutils.TICKWIDTH, color=vutils.BLACK, rotation=0)
        self.ax.legend(loc='upper right', frameon=True, framealpha=1, edgecolor=vutils.BLACK)
        self.ax.text(25, 7.25, 'Algorithm detects LATER than human →', ha='center', va='center')
        self.ax.text(-25, 7.25, '← Algorithm detects EARLIER than human', ha='center', va='center')

    def _format_fig(self):
        self.fig.tight_layout()


class LinePlotOptimizationIterations(vutils.BasePlot):
    ORDER = ['mix', *utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys()]
    LINE_KWS = dict(
        palette=[vutils.BLACK, *vutils.RGB], hue_order=ORDER,
        lw=vutils.LINEWIDTH * 2, ls=vutils.LINESTYLE, errorbar=None,
        alpha=vutils.ALPHA, zorder=5
    )
    SCATTER_KWS = dict(
        palette=[vutils.BLACK, *vutils.RGB], hue_order=ORDER,
        s=200, edgecolor=vutils.BLACK, linewidth=vutils.LINEWIDTH,
        legend=False, markers=['$M$', '$P$', '$B$', '$D$'], style_order=ORDER
    )

    def __init__(self, opt_fpath: str, **kwargs):
        self.corpus_title = 'corpus_chronology'
        # Initialise the base plot with our given kwargs
        super().__init__(figure_title=fr'corpus_plots\lineplot_optimziationiterations_{self.corpus_title}', **kwargs)
        self.df = pd.concat(self._format_df(opt_fpath))
        self.grp_mean = (
            self.df.groupby(['instrument', 'iterations'], as_index=False)
            .mean()
            .groupby('instrument', as_index=False)
            .agg({'iterations': 'max', 'f_score': 'max'})
        )
        tp = self.df.groupby(['instrument', 'iterations'], as_index=False).std()
        self.grp_std = tp.iloc[tp.groupby('instrument')['iterations'].idxmax()].reset_index(drop=True)
        self.fig, self.ax = plt.subplots(1, 2, sharey=False, sharex=True, figsize=(vutils.WIDTH, vutils.WIDTH / 2))

    @staticmethod
    def _format_df(opt_fpath):
        for f in os.listdir(opt_fpath):
            if '.csv' in f and 'forest' not in f:
                d = pd.DataFrame(utils.load_csv(opt_fpath, f.replace('.csv', '')))
                yield d[['mbz_id', 'instrument', 'f_score', 'iterations']]

    def _create_plot(self):
        for ax, est, grp in zip(self.ax.flatten(), [np.mean, np.std], [self.grp_mean, self.grp_std]):
            sns.lineplot(
                data=self.df, x='iterations', y='f_score', hue='instrument', estimator=est, ax=ax, **self.LINE_KWS
            )
            sns.scatterplot(
                data=grp, x='iterations', y='f_score', hue='instrument', ax=ax, style='instrument', **self.SCATTER_KWS
            )

    @staticmethod
    def _format_legend(ax, loc):
        hand, lab = ax.get_legend_handles_labels()
        # copy the handles
        hand = [copy.copy(ha) for ha in hand]
        # set the linewidths to the copies
        [ha.set_linewidth(vutils.LINEWIDTH * 2) for ha in hand]
        ax.legend(
            hand, [la.title() for la in lab], title='Instrument', loc=loc,
            frameon=True, framealpha=1, edgecolor=vutils.BLACK,
        )

    def _format_ax(self):
        xt = np.linspace(0, 400, 5)
        yt = np.linspace(0, 1, 5)
        n = self.df['mbz_id'].nunique()
        self.ax[0].set(
            xlabel='Iterations', xticks=xt, yticks=yt, ylabel='$F$-Measure',
            xlim=(0, 400), ylim=(0, 1), title=f'Mean $F$-Measure ($N$ = {n})'
        )
        self.ax[1].set(
            xlabel='Iterations', xticks=xt, yticks=yt, xlim=(0, 400), ylim=(0, 1),
            ylabel='$F$-Measure', title=f'Standard Deviation $F$-Measure ($N$ = {n})'
        )
        for a, loc in zip(self.ax.flatten(), ['lower right', 'upper right']):
            self._format_legend(a, loc)
            a.grid(axis='both', which='major', **vutils.GRID_KWS)
            a.tick_params(axis='both', width=vutils.TICKWIDTH)
            plt.setp(a.spines.values(), linewidth=vutils.LINEWIDTH)

    def _format_fig(self):
        self.fig.tight_layout()


class BarPlotCorpusDuration(vutils.BasePlot):
    img_loc = fr'{utils.get_project_root()}\references\images\musicians'
    BAR_KWS = dict(
        stacked=True, color=[vutils.RED, vutils.GREEN],
        zorder=10, lw=vutils.LINEWIDTH, edgecolor=vutils.BLACK,
    )
    PERC_KWS = dict(zorder=10, ha='center', va='center', color=vutils.WHITE, fontsize=vutils.FONTSIZE / 2)

    def __init__(self, corp_df: str, **kwargs):
        self.corpus_title = 'corpus_chronology'
        # Initialise the base plot with our given kwargs
        super().__init__(figure_title=fr'corpus_plots\barplot_corpusduration_{self.corpus_title}', **kwargs)
        self.df = self._format_df(corp_df)
        self.fig, self.ax = plt.subplots(1, 1, figsize=(vutils.WIDTH, vutils.WIDTH / 2))

    @staticmethod
    def _format_df(corp_df_):
        small_ = (
            corp_df_.copy(deep=True)
            .sort_values(by='birth')
            .groupby(['bandleader', 'in_corpus'], as_index=True)
            .agg({'duration': 'sum'})
            .reset_index(drop=False)
            .pivot_table(index='bandleader', columns=['in_corpus'], aggfunc='first')
        )
        small_['sum'] = small_.sum(axis=1)
        return small_.sort_values(by='sum').drop(columns=['sum'])

    def _create_plot(self):
        return self.df.plot(
            kind='barh', ax=self.ax, ylabel='Pianist', **self.BAR_KWS,
            xlabel='Total duration of all recordings (hours)',
        )

    def _add_bandleader_images(self, bl, y):
        fpath = fr'{self.img_loc}\{bl.replace(" ", "_").lower()}.png'
        img = mpl.offsetbox.OffsetImage(
            plt.imread(fpath), clip_on=False, transform=self.ax.transAxes, zoom=0.5
        )
        ab = mpl.offsetbox.AnnotationBbox(
            img, (-12500, y - 0.05), xycoords='data', clip_on=False, transform=self.ax.transAxes,
            annotation_clip=False, bboxprops=dict(edgecolor='none', facecolor='none')
        )
        self.ax.add_artist(ab)

    def _add_percentage(self):
        for (idx, row), rect in zip(self.df.iterrows(), self.ax.patches[10:]):
            perc = str(round((row.values[-1] / row.values.sum()) * 100)) + '%'
            x, y = rect.xy
            height, width = rect.get_height(), rect.get_width()
            self.ax.text(x + (width / 2), y + (height / 2), perc, **self.PERC_KWS)

    def _format_ax(self):
        for num, pi in enumerate(self.df.index):
            self._add_bandleader_images(pi, num)
        self._add_percentage()
        xt = np.linspace(0, 345600, 5)
        self.ax.set(xticks=xt, xticklabels=[int(x / 60 / 60) for x in xt])
        hand, lab = self.ax.get_legend_handles_labels()
        self.ax.legend(
            hand, ['False', 'True'], title='Included?', frameon=True,
            framealpha=1, edgecolor=vutils.BLACK, loc='lower right'
        )
        self.ax.grid(axis='x', which='major', **vutils.GRID_KWS)
        self.ax.tick_params(axis='y', which='both', pad=65)
        self.ax.tick_params(axis='both', width=vutils.TICKWIDTH)
        plt.setp(self.ax.spines.values(), linewidth=vutils.LINEWIDTH)

    def _format_fig(self):
        self.fig.subplots_adjust(top=0.95, bottom=0.1, left=0.2, right=0.95)
