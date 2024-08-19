#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Creates the data, to be rendered in HTML using JS"""

import json
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src import utils
from src.features.rhythm_features import *

EXOG_INS = 'piano'


def formatter(item, key):
    val = item.summary_dict[key]
    if np.isnan(val):
        return None
    else:
        return round(val, 2)


def proc(track):
    summary_df = pd.DataFrame(track.summary_dict)
    my_beats = summary_df[EXOG_INS]
    my_onsets = track.ons[EXOG_INS]
    their_beats = summary_df[['bass', 'drums']]
    downbeats = track.ons['downbeats_auto']
    # These are the beat timestamps from the matched onsets
    band_beats = get_beats_from_matched_onsets(track.summary_dict)
    # The tempo and time signature of the track
    tempo = (60 / band_beats.diff()).mean()
    time_signature = track.item['time_signature']
    bur = BeatUpbeatRatio(my_onsets=my_onsets, my_beats=my_beats, clean_outliers=True)
    pcorr = PhaseCorrection(my_beats=my_beats, their_beats=their_beats, order=1)
    pcorr_b = PhaseCorrection(my_beats=summary_df['bass'], their_beats=summary_df[[EXOG_INS, 'drums']], order=1)
    pcorr_d = PhaseCorrection(my_beats=summary_df['drums'], their_beats=summary_df[[EXOG_INS, 'bass']], order=1)
    pasync = ProportionalAsynchrony(summary_df=summary_df, my_instr_name=EXOG_INS, metre_col='metre_auto')
    ioi = IOIComplexity(my_onsets=my_onsets, downbeats=downbeats, tempo=tempo, time_signature=time_signature)
    ts = TempoSlope(my_beats=band_beats)
    tstab = RollingIOISummaryStats(my_onsets=my_beats, downbeats=downbeats, bar_period=4)
    return [
        track.item['fname'],
        track.item['track_name'],
        track.item['recording_year'],
        track.item['bandleader'],
        track.item['musicians'][utils.INSTRUMENTS_TO_PERFORMER_ROLES[EXOG_INS]],
        track.item['in_30_corpus'],
        formatter(bur, 'bur_log_mean'),
        formatter(bur, 'bur_log_std'),
        formatter(ioi, 'lz77_mean'),
        formatter(ioi, 'lz77_std'),
        formatter(ioi, 'n_onsets_mean'),
        formatter(ioi, 'n_onsets_std'),
        round(pasync.summary_dict['bass_prop_async_nanmean'] * (track.item['time_signature'] * 100), 2),
        round(pasync.summary_dict['bass_prop_async_nanstd'] * (track.item['time_signature'] * 100), 2),
        round(pasync.summary_dict['drums_prop_async_nanmean'] * (track.item['time_signature'] * 100), 2),
        round(pasync.summary_dict['drums_prop_async_nanstd'] * (track.item['time_signature'] * 100), 2),
        formatter(pcorr, 'self_coupling'),
        formatter(pcorr, 'coupling_bass'),
        formatter(pcorr, 'coupling_drums'),
        formatter(pcorr_d, 'coupling_piano'),
        formatter(pcorr_b, 'coupling_piano'),
        round(track.item['tempo'], 2),
        formatter(tstab, 'rolling_std_median'),
        formatter(ts, 'tempo_slope'),
    ]


if __name__ == "__main__":
    warnings.simplefilter('ignore', FutureWarning)
    crp = utils.load_corpus_from_files(str(utils.get_project_root()) + '/data/jazz-trio-database-v02')
    allres = {"data": []}
    with Parallel(n_jobs=-1, backend='loky', verbose=10) as par:
        bigdat = par(delayed(proc)(t) for t in crp)
    allres = {"data": list(bigdat)}

    json.dump(
        allres,
        open(fr'{utils.get_project_root()}/_docssrc/_static/data-explorer/table.txt', 'w'),
        indent=4,
        ensure_ascii=False
    )
