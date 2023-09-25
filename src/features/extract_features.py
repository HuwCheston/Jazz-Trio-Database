#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Extracts the required features for each item in the corpus, using the automatically detected onsets."""

import logging
from pathlib import Path
from time import time

import click
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from joblib import Parallel, delayed

from src import utils
from src.detect.detect_utils import OnsetMaker
from src.features.features_utils import FeatureExtractor


def process_item(onset_maker: OnsetMaker) -> FeatureExtractor:
    """Process the data in a single OnsetMaker class, used in parallel contexts (i.e. called with joblib.Parallel)"""
    mm = FeatureExtractor(om=onset_maker, interpolate=True, interpolation_limit=1)
    summary = []
    for ins in utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys():
        mm.models[ins] = mm.generate_model(ins, standardise=False, difference_ioi=True, iqr_clean=False)
        summary.append(mm.create_instrument_dict(endog_ins=ins, md=mm.models[ins]))
    mm.summary_df = pd.DataFrame(summary)
    return mm


@click.command()
@click.option("-corpus", "corpus_filename", type=str, default="corpus_chronology", help='Name of the corpus to use')
@click.option("-n_jobs", "n_jobs", type=click.IntRange(-1, clamp=True), default=-1, help='Number of CPU cores to use')
@click.option("-one-track-only", "one_track_only", is_flag=True, default=False, help='Only process one item')
def main(
        corpus_filename: str,
        n_jobs: int,
        one_track_only: bool
) -> list[FeatureExtractor]:

    # Start the counter
    start = time()
    # Initialise the logger
    logger = logging.getLogger(__name__)
    fname = rf'{utils.get_project_root()}\models\matched_onsets_{corpus_filename}'
    # Load in our serialised onset maker classes
    onsets = utils.unserialise_object(fname)
    # If we only want to process one track, useful for debugging
    if one_track_only: onsets = [onsets[0]]
    logger.info(f"extracting features from {len(onsets)} tracks using {n_jobs} CPUs ...")
    # Extract features from all of our serialised onset maker classes, in parallel
    features = Parallel(n_jobs=n_jobs, backend='loky')(delayed(process_item)(onset_maker) for onset_maker in onsets)
    # Serialise all of our FeatureExtractor classes
    # TODO: implement caching of incoming FeatureExtractor classes here
    utils.serialise_object(features, rf'{utils.get_project_root()}\models', f'extracted_features_{corpus_filename}')
    # Log the completion time and return the class instances
    logger.info(f'features extracted for all tracks in corpus {corpus_filename} in {round(time() - start)} secs !')
    return features


if __name__ == '__main__':
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
