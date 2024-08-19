#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Creates the individual popups when clicking on a row in the data explorer"""

import os
import shutil
import warnings

import pandas as pd
from joblib import Parallel, delayed

from src import utils
from src.visualise.interactive_plots import *


def create_interactive_plots_for_one_track(track_om) -> None:
    """Creates all .html files for data visualiser for a single track"""
    plotters = [
        ScatterPlotFeelInteractive, HistPlotSwingInteractive,
        HistPlotComplexityInteractive, BarPlotCoordinationInteractive
    ]
    names = ['feel', 'swing', 'complexity', 'interaction']
    root = fr'{utils.get_project_root()}/_docssrc/_static/data-explorer'
    new_fpath = fr'{root}/{track_om.item["fname"]}'
    try:
        os.mkdir(new_fpath)
    except FileExistsError:
        pass
    shutil.copy(fr'{root}/explorer-template.html', rf'{new_fpath}/display.html')
    for plotter, name in zip(plotters, names):
        try:
            p = plotter(track_om)
        except BaseException as e:
            print(f'Error in {name}, {track_om.item["fname"]}@ {e}')
            with open(fr'{new_fpath}/{name}.html', 'w') as fp:
                fp.write(f"""
                <div>
                {name.title()} plot not found! There was probably not enough data found for this track 
                ({track_om.item["fname"]}) to create the requested plot type.<br>
                If this is urgent, please 
                <a href="mailto:hwc31@cam.ac.uk?subject=Missing plot!&cc=huwcheston@gmail.com">contact us</a> and
                we'll try and fix things.
                </div>
                """)
        else:
            p.create_plot()
            p.render_html(fpath=fr'{new_fpath}/{name}.html', div_id=name)
    meta = pd.Series(track_om.item).to_json()
    with open(fr'{new_fpath}/metadata.json', 'w') as f:
        f.write(meta)


if __name__ == '__main__':
    warnings.simplefilter('ignore', FutureWarning)
    tracks = utils.load_corpus_from_files(fr'{utils.get_project_root()}/data/jazz-trio-database-v02')
    with Parallel(n_jobs=-1, backend='loky', verbose=10) as par:
        _ = par(delayed(create_interactive_plots_for_one_track)(t) for t in tracks)