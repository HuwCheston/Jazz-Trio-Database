#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Extracts the required features for each item in the corpus, using the automatically detected onsets."""

import logging
from pathlib import Path
from time import time

import click
from dotenv import find_dotenv, load_dotenv
from joblib import Parallel, delayed

from src import utils
from src.detect.detect_utils import OnsetMaker
from src.features.features_utils import FeatureExtractor


def process_item(onset_maker: OnsetMaker, item_queue) -> None:
    """Process the data in a single OnsetMaker class, used in parallel contexts (i.e. called with joblib.Parallel)"""
    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format=fmt)
    logger.info(f'processing item {onset_maker.item["mbz_id"]}, track name {onset_maker.item["track_name"]} ...')
    # Create the FeatureExtractor and extract features
    item = FeatureExtractor(om=onset_maker, interpolate=True, interpolation_limit=1)
    item.extract_features()
    # Put the finalized FeatureExtractor into our queue to be saved by the worker process
    item_queue.put(item)


@click.command()
@click.option("-corpus", "corpus_filename", type=str, default="corpus_bill_evans", help='Name of the corpus to use')
@click.option("-n_jobs", "n_jobs", type=click.IntRange(-1, clamp=True), default=-1, help='Number of CPU cores to use')
@click.option("-one-track-only", "one_track_only", is_flag=True, default=False, help='Only process one item')
@click.option("-annotated-only", "annotated_only", is_flag=True, default=False, help='Only use items with annotations')
@click.option("-ignore-cache", "ignore_cache", is_flag=True, default=False, help='Ignore any cached items')
def main(
        corpus_filename: str,
        n_jobs: int,
        one_track_only: bool,
        annotated_only: bool,
        ignore_cache: bool
) -> list[FeatureExtractor]:
    """Runs scripts to extract features and generate models from detected onsets (in `.\models`)"""

    # Start the counter
    start = time()
    # Initialise the logger
    logger = logging.getLogger(__name__)
    # Get input and output filenames
    input_fname = rf'{utils.get_project_root()}\models\matched_onsets_{corpus_filename}'
    output_fname = rf'{utils.get_project_root()}\models\extracted_features_{corpus_filename}'
    # Load in our serialised onset maker classes
    onsets = utils.unserialise_object(input_fname)
    # Remove any tracks which we've already processed in previous runs of this script
    from_cache = 0
    if not ignore_cache:
        cached_ids = list(utils.get_cached_track_ids(output_fname, use_pickle=True))
        from_cache = len(cached_ids)
        onsets = [track for track in onsets if track.item['mbz_id'] not in cached_ids]
    # If we only want to get tracks which have manual annotation files completed for them
    if annotated_only:
        # No need to check MBz ids, just check if the variable for onset evaluations is not None
        onsets = [track for track in onsets if track.onset_evaluation]
    # If we only want to process one track (can be useful for debugging)
    if one_track_only:
        onsets = [onsets[0]]
    # Now we can report the total number of tracks that we'll be extracting features from
    logger.info(f"extracting features from {len(onsets)} tracks ({from_cache} from disc) using {n_jobs} CPUs ...")
    # Initialise the output queue
    p, q = utils.initialise_queue(utils.serialise_from_queue, output_fname)
    # Extract features from all of our serialised onset maker classes, in parallel
    extracted_features = Parallel(n_jobs=n_jobs, backend='loky')(delayed(process_item)(om, q) for om in onsets)
    # extracted_features = [process_item(om, q) for om in onsets]
    # Kill the track saving worker by adding a NoneType object to its queue
    q.put(None)
    p.join()
    # Log the completion time and return the class instances
    logger.info(f'features extracted for all tracks in corpus {corpus_filename} in {round(time() - start)} secs !')
    return extracted_features


if __name__ == '__main__':
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
