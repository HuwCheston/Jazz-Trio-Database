#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Classes used for creating interactive plots in `plotly`"""

import os
import shutil
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.stats as stats
import seaborn as sns
from PIL import Image
from plotly.subplots import make_subplots

import src.visualise.visualise_utils as vutils
from src import utils
from src.visualise.bur_plots import BURS_WITH_IMAGES
from src.visualise.complexity_plots import FRACS, FRACS_S


class BasePlotPlotly:
    def __init__(self):
        self.fig = make_subplots(
            rows=1, cols=3, shared_xaxes=True, shared_yaxes=True,
            subplot_titles=[i.title() for i in utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys()]
        )

    def create_plot(self):
        self._create_plot()
        self._format_fig()

    def render_html(self, fpath, div_id):
        self.fig.write_html(
            fpath, include_plotlyjs='cdn', full_html=False, include_mathjax='cdn', default_width='100%',
            div_id=div_id, auto_open=False
        )

    def _create_plot(self):
        pass

    def _format_fig(self):
        pass


class ScatterPlotFeelInteractive(BasePlotPlotly):
    """Creates a scatter plot for all onset values within a given track, similar to those in `OnsetSync` R package"""
    wraparound = 0.9

    def __init__(self, onset_maker):
        self.onset_maker = onset_maker
        self.title = self.onset_maker.item['fname']
        self.time_sig = self.onset_maker.item['time_signature']
        self.df = pd.DataFrame(self.format_df())
        super().__init__()

    def format_df(self) -> list:
        """Formats provided onset maker into correct dataframe format for plotting"""
        for instr in utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys():
            instr_ons = self.onset_maker.ons[instr]
            z = zip(self.onset_maker.ons['mix'], self.onset_maker.ons['mix'][1:], self.onset_maker.ons['metre_manual'])
            for beat1, beat2, beat1pos in z:
                vals = instr_ons[np.logical_and(beat1 <= instr_ons, instr_ons < beat2)]
                for i in vals:
                    pos = ((i - beat1) / (beat2 - beat1)) + beat1pos
                    newpos = pos if pos < self.time_sig + self.wraparound else pos - self.time_sig
                    floor = np.floor(newpos)
                    yield {
                        'instrument': instr,
                        'timestamp': pd.to_datetime(datetime.fromtimestamp(beat1).strftime('%H:%M:%S.%f')[:-3]),
                        'musical_position': newpos,
                        'mp_floor': floor if floor > 0 else 1,
                    }

    def create_plot(self):
        self._create_plot()
        self._format_fig()

    def _create_plot(self) -> None:
        for ax, (idx, grp), col in zip(range(1, 4), self.df.groupby('instrument', sort=False), vutils.RGB):
            grp = grp.dropna()
            if len(grp) < 2:
                line = go.Scatter(
                    x=[], y=[], mode='markers', showlegend=False, name=idx.title(),
                )
                self.fig.add_trace(line, row=1, col=ax)
                continue

            scat = go.Scatter(
                x=grp['musical_position'], y=grp['timestamp'], mode='markers', customdata=grp['mp_floor'],
                marker=dict(color=col, size=2.5), hovertemplate='Beat: %{customdata}<br>Timestamp: %{y|%M:%S.%L}',
                showlegend=False, name=idx.title(),
            )
            self.fig.add_trace(scat, row=1, col=ax)

    def _format_fig(self):
        self.fig.update_layout(
            plot_bgcolor="white", title=dict(text=self.title, xanchor='center', yanchor='top'), title_x=0.5,
            yaxis=dict(title=dict(text='Time (minutes:seconds)')), autosize=False, width=700, height=400,
        )
        self.fig.update_xaxes(
            showline=True, linewidth=vutils.LINEWIDTH, linecolor=vutils.BLACK, mirror=True,
            range=[0.8, self.time_sig + 0.9],
            tickmode='array', tickvals=[i for i in range(1, self.time_sig + 1)], showgrid=True, gridcolor='grey'
        )
        self.fig.update_yaxes(
            showline=True, linewidth=vutils.LINEWIDTH, linecolor=vutils.BLACK, mirror=True, fixedrange=True,
            tickformat="%M:%S"
        )
        self.fig['layout']['xaxis2']['title'] = 'Position in bar (quarter note)'


class HistPlotComplexityInteractive(BasePlotPlotly):
    PALETTE = [vutils.BLACK, *reversed(sns.color_palette(None, len(FRACS) - 2)), vutils.BLACK]

    def __init__(self, onset_maker):
        self.df = self._format_df(onset_maker)
        self.title = onset_maker.item['fname']
        super().__init__()

    @staticmethod
    def _format_df(om):
        from src.features.features_utils import IOIComplexity
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
        for ax, (idx, grp) in zip(range(1, 4), self.df.groupby('instr', sort=False)):
            grp = grp.dropna()
            if len(grp) == 0:
                continue
            # Plot the kde
            xs, ys = self._kde(grp['prop_ioi'])
            xs = xs.flatten()
            s = np.sort([(FRACS[i] + FRACS[i + 1]) / 2 for i in range(len(FRACS) - 1)]).tolist()
            for previous, current, col, bi in zip(s, s[1:], list(reversed(self.PALETTE))[1:],
                                                  list(reversed(FRACS))[1:]):
                slicer = np.where((xs <= current) & (xs >= previous))
                xvals = xs[slicer]
                yvals = ys[slicer]
                le = len(grp[(grp['prop_ioi'] <= current) & (grp['prop_ioi'] >= previous)])
                colstr = 'rgb(' + ','.join(str(int(np.floor(i * 255))) for i in col) + ')'
                line = go.Scatter(
                    x=xvals, y=yvals, line=dict(color=colstr), showlegend=False, name=idx.title(),
                    hoverinfo='skip',
                )
                customdata = np.stack([
                    [round(bi, 2) for _ in range(len(xvals))],
                    [le for _ in range(len(xvals))]
                ], axis=1)
                fill = go.Scatter(
                    x=xvals, y=yvals, line=dict(color=colstr), fill='tozeroy', fillcolor=colstr,
                    customdata=customdata, hovertemplate='Bin: %{customdata[0]}<br>IOIs in Bin: %{customdata[1]}',
                    showlegend=False, name=idx.title()
                )
                for gp in [line, fill]:
                    self.fig.add_trace(gp, row=1, col=ax)
            self.fig.add_trace(
                go.Scatter(
                    x=xs, y=ys, line=dict(color=vutils.BLACK), hoverinfo='skip', name=idx.title(),
                    marker_line=dict(width=vutils.LINEWIDTH), showlegend=False,
                ),
                row=1, col=ax
            )

    def _format_fig(self):
        self.fig.update_layout(
            plot_bgcolor="white", title=dict(text=self.title, xanchor='center', yanchor='top'), title_x=0.5,
            yaxis=dict(title=dict(text='Density')), autosize=False, width=700, height=400,
        )
        self.fig.update_xaxes(
            showline=True, linewidth=vutils.LINEWIDTH, linecolor=vutils.BLACK, mirror=True, range=[0, 1],
            tickmode='array', tickvals=[0, 0.25, 0.5, 0.75, 1]
        )
        self.fig.update_yaxes(
            showline=True, linewidth=vutils.LINEWIDTH, linecolor=vutils.BLACK, mirror=True, range=[0, 1.2],
            tickmode='array', fixedrange=True, tickvals=[0, 0.25, 0.5, 0.75, 1]
        )
        for frac, frac_s in zip(FRACS[1:-1], FRACS_S[1:-1]):
            self.fig.add_vline(x=frac, line_width=1, line_dash='solid', line_color='grey')
            for ax in range(1, 4):
                self.fig.add_annotation(
                    go.layout.Annotation(text=frac_s, x=frac + 0.01, y=1.1, showarrow=False, ),
                    row=1, col=ax
                )
        self.fig['layout']['xaxis2']['title'] = 'Proportional inter-onset interval'


class BarPlotCoordinationInteractive(BasePlotPlotly):
    def __init__(self, onset_maker):
        self.df = self._format_df(onset_maker)
        self.title = onset_maker.item['fname']
        super().__init__()
        self.fig = make_subplots(rows=1, cols=1)

    @staticmethod
    def _format_df(om):
        from src.features.features_utils import PhaseCorrection
        sd = pd.DataFrame(om.summary_dict)
        res = []
        for my_instr in utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys():
            their_instrs = [i for i in utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys() if i != my_instr]
            my_beats = sd[my_instr]
            their_beats = sd[their_instrs]
            pc = PhaseCorrection(my_beats=my_beats, their_beats=their_beats)
            res.append({'instrument': my_instr, 'pianist': om.item['pianist'],
                        'performer': om.item['musicians'][
                            utils.INSTRUMENTS_TO_PERFORMER_ROLES[my_instr]]} | pc.summary_dict)
        df = pd.DataFrame(res)
        model_df = (
            df.melt(
                id_vars=['instrument', 'nobs'],
                value_vars=['coupling_piano', 'coupling_bass', 'coupling_drums']
            )
            .dropna()
            .reset_index(drop=False)
        )
        instr = model_df['variable'].str.replace('coupling_', '').str.title()
        model_df['variable'] = instr + '→' + model_df['instrument'].str.title()
        model_df['instrument'] = instr
        model_df['nobs'] = model_df['nobs'].astype(int)
        return (
            model_df.set_index('instrument')
            .loc[[i.title() for i in utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys()]]
            .reset_index(drop=False)
        )

    def _create_plot(self):
        for (idx, grp), col in zip(self.df.groupby('instrument', sort=False), vutils.RGB):
            bar = go.Bar(
                x=grp['variable'], y=grp['value'], customdata=grp['nobs'],
                marker_color=np.repeat(col, 2), textposition="none",
                marker_line=dict(width=vutils.LINEWIDTH, color=vutils.BLACK),
                name=idx.title(), hovertemplate="Coupling %{x}<br>B: %{y:.2f}<br>Obs: %{customdata}"
            )
            self.fig.add_trace(bar, row=1, col=1)

    def _format_fig(self):
        self.fig.update_layout(
            plot_bgcolor="white", title=dict(text=self.title, xanchor='center', yanchor='top'), title_x=0.5,
            yaxis=dict(title=dict(text='Coupling coefficient (B)')), autosize=False, width=700, height=400,
        )
        self.fig.update_xaxes(
            showline=True, linewidth=vutils.LINEWIDTH, linecolor=vutils.BLACK, mirror=True,
            tickmode='array', fixedrange=True,
        )
        self.fig.update_yaxes(
            showline=True, linewidth=vutils.LINEWIDTH, linecolor=vutils.BLACK, mirror=True,
            range=[self.df['value'].min() - 0.1, self.df['value'].max() + 0.1],
            tickmode='array', fixedrange=False,
        )
        self.fig['layout']['xaxis']['title'] = 'Direction of influence'


class HistPlotSwingInteractive(BasePlotPlotly):
    def __init__(self, onset_maker):
        self.bur_df, self.peak_df = self.format_df(onset_maker)
        self.fname = rf'onsets_plots\histplot_bur_{onset_maker.item["mbz_id"]}'
        self.title = onset_maker.item['fname']
        super().__init__()

    @staticmethod
    def format_df(om):
        from src.features.features_utils import BeatUpbeatRatio
        res = []
        for instr in utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys():
            my_beats = om.summary_dict[instr]
            my_ons = om.ons[instr]
            burs = BeatUpbeatRatio(my_beats=my_beats, my_onsets=my_ons).bur_log['burs'].dropna().values
            res.append(dict(instrument=instr, bur=np.nan))
            for bur in burs:
                res.append(dict(instrument=instr, bur=bur))
        bur_df = pd.DataFrame(res)
        peak_df = bur_df.groupby('instrument', as_index=False).mean().rename(columns={'bur': 'peak'})
        return bur_df, peak_df

    @staticmethod
    def _kde(data, len_data: int = 1000) -> tuple:
        """Fit the KDE to the data and evaluate on a list of y-values, then scale"""
        # Fit the actual KDE to the data, using the default parameters
        kde = stats.gaussian_kde(data.T)
        # Create a linear space of integers ranging from our lowest to our highest BUR
        data_plot = np.linspace(data.min(), data.max(), len_data)[:, np.newaxis]
        # Evaluate the KDE on our linear space of integers
        y = kde.evaluate(data_plot.T)
        return data_plot, np.array([(y_ - min(y)) / (max(y) - min(y)) for y_ in y])

    def _create_plot(self):
        for ax, (idx, grp), col in zip(range(1, 4), self.bur_df.groupby('instrument', sort=False), vutils.RGB):
            grp = grp.dropna()
            if len(grp) < 2:
                line = go.Scatter(
                    x=[], y=[], marker=dict(color=vutils.BLACK),
                    showlegend=False, hoverinfo='skip',
                    marker_line=dict(width=vutils.LINEWIDTH, color=vutils.BLACK),
                )
                self.fig.add_trace(line, row=1, col=ax)
                continue
            heights, edges = np.histogram(grp['bur'], bins=10)
            heights_ = heights / max(heights)
            # Plot the normalized histogram
            bar = go.Bar(
                x=edges[:-1], y=heights_, width=np.diff(edges), marker=dict(color=col, opacity=vutils.ALPHA),
                marker_line=dict(width=vutils.LINEWIDTH, color=vutils.BLACK), textposition="none", name=idx.title(),
                hovertemplate="Log2 BUR: %{x:.2f}<br>%{text}", text=['N: {}'.format(i) for i in heights],
                showlegend=False,
            )
            # Plot the kde
            x, y = self._kde(grp['bur'])
            line = go.Scatter(
                x=x.flatten(), y=y, marker=dict(color=col), showlegend=False, hoverinfo='skip',
                marker_line=dict(width=vutils.LINEWIDTH),
            )
            for gp in [bar, line]:
                self.fig.add_trace(gp, row=1, col=ax)

    def _format_fig(self):
        self.fig.update_layout(
            plot_bgcolor="white", title=dict(text=self.title, xanchor='center', yanchor='top'), title_x=0.5,
            yaxis=dict(title=dict(text='Density')), autosize=False, width=700, height=400,
        )
        self.fig.update_xaxes(
            showline=True, linewidth=vutils.LINEWIDTH, linecolor=vutils.BLACK, mirror=True, range=[-2, 2],
            tickmode='array', fixedrange=True, tickvals=[round(np.log2(i), 2) for i in BURS_WITH_IMAGES]
        )
        self.fig.update_yaxes(
            showline=True, linewidth=vutils.LINEWIDTH, linecolor=vutils.BLACK, mirror=True, range=[0, 1.2],
            tickmode='array', fixedrange=True, tickvals=[0, 0.25, 0.5, 0.75, 1]
        )
        for bur in BURS_WITH_IMAGES:
            self.fig.add_vline(x=np.log2(bur), line_width=1, line_dash='solid', line_color='grey')
            fpath = fr'{utils.get_project_root()}\references\images\bur_notation\bur_{bur}.png'
            img = Image.open(fpath)
            for i in range(1, 4):
                self.fig.add_layout_image(
                    source=img, x=np.log2(bur), y=1.1, row=1, col=i, xref="paper", xanchor="center", yanchor="middle",
                    yref="paper", sizex=0.65, sizey=0.65,
                )
        # Update layout to center x-axis label
        self.fig['layout']['xaxis2']['title'] = 'Log<sub>2</sub> beat-upbeat ratio'


def create_interactive_plots_for_one_track(track_om):
    plotters = [
        ScatterPlotFeelInteractive, HistPlotSwingInteractive,
        HistPlotComplexityInteractive, BarPlotCoordinationInteractive
    ]
    names = ['feel', 'swing', 'complexity', 'interaction']
    root = fr'{utils.get_project_root()}\_docssrc\static\data-explorer'
    new_fpath = fr'{root}\{track_om.item["fname"]}'
    try:
        os.mkdir(new_fpath)
    except FileExistsError:
        pass
    shutil.copy(fr'{root}\explorer-template.html', rf'{new_fpath}\display.html')
    for plotter, name in zip(plotters, names):
        p = plotter(track_om)
        p.create_plot()
        p.render_html(fpath=fr'{new_fpath}\{name}.html', div_id=name)
    meta = pd.Series(track_om.item).to_json()
    with open(fr'{new_fpath}\metadata.json', 'w') as f:
        f.write(meta)


if __name__ == '__main__':
    tracks = utils.unserialise_object(fr'{utils.get_project_root()}\models\matched_onsets_corpus_chronology')
    track = tracks[0]

    create_interactive_plots_for_one_track(track)
