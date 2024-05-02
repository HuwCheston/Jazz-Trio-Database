#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Automatically detects note and beat onsets in the source separated tracks for each item in the corpus"""

import logging
from pathlib import Path
from time import time

import click
from dotenv import find_dotenv, load_dotenv
from joblib import Parallel, delayed

from src import utils
from src.detect.detect_utils import OnsetMaker


def process_item(
        corpus_item: dict,
        generate_click: bool,
) -> OnsetMaker:
    """Process one item from the corpus, used in parallel contexts (i.e. called with joblib.Parallel)"""
    # We need to initialise the logger here again, otherwise it won't work with joblib
    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format=fmt)
    # Create the OnsetMaker class instance for this item in the corpus
    made = OnsetMaker(item=corpus_item)
    # Run our processing on the mixed audio
    logger.info(f'processing audio mix for item {corpus_item["mbz_id"]}, track name {corpus_item["track_name"]} ...')
    made.process_mixed_audio(generate_click)
    # Run our processing on the separated audio
    logger.info(f'processing audio stems for item {corpus_item["mbz_id"]}, track name {corpus_item["track_name"]} ...')
    made.process_separated_audio(generate_click, remove_silence=False)
    # Clean up the results
    made.finalize_output()
    logger.info(f'... item {corpus_item["mbz_id"]} done !')
    return made


@click.command()
@click.option("-corpus", "corpus_filename", type=str, default="corpus_updated", help='Name of the corpus to use')
@click.option("-n_jobs", "n_jobs", type=click.IntRange(-1, clamp=True), default=-1, help='Number of CPU cores to use')
@click.option("-no_click", "generate_click", is_flag=True, default=False, help='Suppress click track generation')
@click.option("-ignore-cache", "ignore_cache", is_flag=True, default=False, help='Ignore any cached items')
def main(
        corpus_filename: str,
        n_jobs: int,
        generate_click: bool,
        ignore_cache: bool
) -> list[OnsetMaker]:
    """Runs scripts to detect onsets in audio from (../raw and ../processed) and generate data for modelling"""
    # Start the counter
    start = time()
    # Initialise the logger
    logger = logging.getLogger(__name__)
    corpus = utils.CorpusMaker.from_excel(fname=corpus_filename, only_annotated=True, only_30_corpus=False)
    fname = rf"{utils.get_project_root()}/data/cambridge-jazz-trio-database-v02"
    # Remove any tracks which we've already processed
    from_cache = 0
    if not ignore_cache:
        cached_ids = list(utils.get_cached_track_ids(fname))
        from_cache = len(cached_ids)
        corpus.tracks = [track for track in corpus.tracks if track['mbz_id'] not in cached_ids]
    # Process each item in the corpus, using multiprocessing in job-lib
    logger.info(f"detecting onsets in {len(corpus.tracks)} tracks ({from_cache} from disc) using {n_jobs} CPUs ...")
    res = Parallel(n_jobs=n_jobs)(delayed(process_item)(item, not generate_click) for item in corpus.tracks)
    # Log the completion time
    logger.info(f'onsets detected for all tracks in {corpus_filename} in {round(time() - start)} secs !')
    # Return the class instances
    return res


if __name__ == '__main__':
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
