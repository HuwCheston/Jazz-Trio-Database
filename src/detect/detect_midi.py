#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Convert source separated piano audio to MIDI for each item in the corpus"""

import logging
import os
from pathlib import Path

import click
from dotenv import find_dotenv, load_dotenv
from joblib import Parallel, delayed
from piano_transcription_inference.utilities import write_events_to_midi

from src import utils
from src.detect.midi_utils import MIDIMaker


def proc(track: dict, corpus_filename: str):
    # We need to initialise the logger here again, otherwise it won't work with joblib
    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format=fmt)
    logger.info(f'... now working on {track["mbz_id"]}, track name {track["track_name"]} ...')
    mm = MIDIMaker(track)
    midi = mm.convert_to_midi()
    os.makedirs(f'{utils.get_project_root()}/models/{corpus_filename}/{track["fname"]}', exist_ok=True)
    write_events_to_midi(
        start_time=0,
        note_events=midi['est_note_events'],
        pedal_events=midi['est_pedal_events'],
        midi_path=f'{utils.get_project_root()}/models/{corpus_filename}/{track["fname"]}/piano_midi.mid'
    )
    logger.info(f'... finished working on {track["mbz_id"]}, track name {track["track_name"]} ...')


def get_tracks(track_list: list[dict], corpus_filename):
    for track in track_list:
        if not os.path.isfile(f'{utils.get_project_root()}/models/{corpus_filename}/{track["fname"]}/piano_midi.mid'):
            yield track


@click.command()
@click.option("-corpus", "corpus_filename", type=str, default="corpus_chronology", help='Name of the corpus to use')
@click.option("-n_jobs", "n_jobs", type=click.IntRange(-1, clamp=True), default=-1, help='Number of CPU cores to use')
def main(
        corpus_filename: str,
        n_jobs: int
):
    logger = logging.getLogger(__name__)
    corpus = utils.CorpusMaker.from_excel(corpus_filename)
    to_process = list(get_tracks(corpus.tracks, corpus_filename))
    logger.info(f"extracting midi from {len(to_process)} tracks ({len(corpus.tracks) - len(to_process)} "
                f"loaded from disc) using {n_jobs} CPUs ...")
    with Parallel(n_jobs=-1, prefer='threads') as par:
        par(delayed(proc)(t, corpus_filename) for t in to_process)


if __name__ == '__main__':
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
