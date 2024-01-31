#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Process a track directly from the command line."""

import logging
import os
import warnings

import click
import numpy as np
import pandas as pd
from joblib import load

from src import utils
from src.clean.clean_utils import ItemMaker
from src.detect.detect_utils import OnsetMaker
from src.features.features_utils import *


def extract_track_features(track: OnsetMaker, exog_ins) -> dict:
    """Processes a single track, extracting all required features, and returns a dictionary"""
    def get_feature_data(feature_cls, cols, extra_str='', **cls_kwargs):
        """Creates a class with given kwargs and returns the desired key-value pairs from its summary dictionary"""
        cls = feature_cls(**cls_kwargs)
        return {k + extra_str: v for k, v in cls.summary_dict.items() if k in cols}

    # Convert the summary dictionary (dictionary of arrays) to a dataframe
    summary_dict = pd.DataFrame(track.summary_dict)
    # These are the positions of downbeats, i.e. the first beat of a measure
    downbeats = track.ons['downbeats_auto']
    # The tempo and time signature of the track
    tempo = track.tempo
    time_signature = track.item['time_signature']
    # Subset to get my onsets and partner onsets as separate dataframes
    my_onsets = track.ons[exog_ins]
    my_beats = summary_dict[exog_ins]
    their_beats = summary_dict[[i for i in utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys() if i != exog_ins]]
    # BEAT-UPBEAT RATIO
    bur = get_feature_data(
        BeatUpbeatRatio, ['bur_log_mean', 'bur_log_std', 'bur_log_count_nonzero'],
        my_onsets=my_onsets, my_beats=my_beats, clean_outliers=True
    )
    # PHASE CORRECTION
    pc = get_feature_data(
        PhaseCorrection, ['self_coupling', 'coupling_bass', 'coupling_drums', 'nobs'],
        my_beats=my_beats, their_beats=their_beats, order=1
    )
    # PHASE CORRECTION - PARTNER
    # In comparison to the 'full' phase correction model, we only need to get a few columns here
    pcb = get_feature_data(
        PhaseCorrection, ['coupling_piano', 'nobs'], extra_str='_bass',
        my_beats=summary_dict['bass'], their_beats=summary_dict[['piano', 'drums']], order=1
    )
    pcd = get_feature_data(
        PhaseCorrection, ['coupling_piano', 'nobs'], extra_str='_drums',
        my_beats=summary_dict['drums'], their_beats=summary_dict[['piano', 'bass']], order=1
    )
    # PROPORTIONAL ASYNCHRONY
    pa = get_feature_data(
        ProportionalAsynchrony, [
            'prop_async_count_nonzero', 'bass_prop_async_nanmean',
            'drums_prop_async_nanmean', 'bass_prop_async_nanstd',
            'drums_prop_async_nanstd'
        ],
        summary_df=summary_dict, my_instr_name=exog_ins, metre_col='metre_auto'
    )
    # IOI COMPLEXITY
    ioi = get_feature_data(
        IOIComplexity, ['lz77_mean', 'lz77_std', 'n_onsets_mean', 'n_onsets_std', 'window_count', 'ioi_count'],
        my_onsets=my_onsets, downbeats=downbeats, tempo=tempo, time_signature=time_signature
    )
    # TEMPO SLOPE
    ts = get_feature_data(
        TempoSlope, ['tempo_slope', 'tempo_drift'],
        my_beats=pd.concat([my_beats, their_beats], axis=1).mean(axis=1)
    )
    # TEMPO STABILITY
    tstab = get_feature_data(
        RollingIOISummaryStats, ['rolling_std_count_nonzero', 'rolling_std_median'],
        my_onsets=my_beats, downbeats=downbeats, bar_period=4
    )
    # Return a single dictionary that combines the summary dictionary for all the features
    return dict(**track.item, **bur, **pc, **pcb, **pcd, **pa, **ioi, **ts, **tstab, tempo=tempo)


def get_track_dictionary(filename: str, start: str, stop: str, json_location: str = '') -> dict:
    """Returns a dictionary from `json_location` or creates a dictionary from scratch"""
    try:
        return utils.load_json(json_location)
    except FileNotFoundError:
        return {
            'track_name': filename,
            'fname': filename,
            'channel_overrides': {},
            'timestamps': {
                'start': start,
                'end': stop
            },
            'recording_year': None,
            'time_signature': 4,
            'first_downbeat': None,
            'album_name': None,
            'musicians': {
                'leader': None,
                'piano': None
            }
        }


def create_output_filestructure(filename: str):
    """Creates necessary folder structure for track with given `filename`"""
    for path in [
        filename,
        f"{filename}/data/raw/audio",
        f"{filename}/data/processed/spleeter_audio",
        f"{filename}/data/processed/demucs_audio",
        f"{filename}/annotations",
        f"{filename}/outputs",
    ]:
        os.makedirs(path, exist_ok=True)


def make_pianist_prediction(feature_dict: dict, model_filepath: str = None):
    """Predicts the pianist for an input track from extracted features using the pre-trained model"""
    # This is the default directory for our model to be saved
    if model_filepath is None:
        model_filepath = f'{utils.get_project_root()}/models/pianist_model.joblib'
    # Load in the pretrained model
    model = load(model_filepath)
    feature_dict = {k: v for k, v in feature_dict.items() if k in utils.PREDICTORS}
    # Impute missing values using the averaged obtained for the training dataset
    # This may invalidate predictions, but it's necessary as the random forest can't handle missing values
    for feature in utils.PREDICTORS:
        if np.isnan(feature_dict[feature]):
            avg = utils.IMPUTE_VALS[feature]
            message = f"Feature {feature} did not extract from the track correctly. " \
                      f"Replacing with average obtained rom dataset ({round(avg, 2)}). " \
                      f"This may invalidate any predictions made for this track!"
            warnings.warn(message)
            feature_dict[feature] = avg
    # Convert the feature dictionary to a dataframe to make it easier to extract the items we want
    feature_df = pd.DataFrame(pd.Series(feature_dict)).transpose()[utils.PREDICTORS]
    # Return the prediction made using the input data
    return format_predictions(model.predict_proba(feature_df), model.classes_)


def format_predictions(predict_proba: np.ndarray, class_names: np.ndarray) -> str:
    """Formats model predictions into a nice human-readable string"""
    # Get the pianist names and probabilities for the top 3 pianists in the dataset
    pred = (
        pd.concat([pd.Series(predict_proba[0]), pd.Series(class_names)], axis=1)
        .rename(columns={0: 'probability', 1: 'pianist'})
        .sort_values(by='probability', ascending=False)
        .head(3)
    )
    # Format the predictions nicely and return the formatted string
    st = ', '.join([f'{row["pianist"]}: {row["probability"] * 100}%' for _, row in pred.iterrows()])
    return f'... predicted pianist is {pred.iloc[0]["pianist"]} (top 3 predictions: {st})'


def proc_inner(
        input: str,
        json: str,
        params: str,
        begin: str,
        end: str,
        exog_ins: str,
        generate_click: bool,
):
    """An inner function for processing that can be imported directly in Python"""
    # Set the logger
    logger = logging.getLogger(__name__)
    # Validate the input address (more checking for if the link actually works happens later)
    if 'youtube' in input.lower() and input.lower().startswith('http'):
        filename = input.split('&')[0].split('?v=')[-1].lower()
    else:
        raise AttributeError('Input file must be a valid YouTube link.')
    # Set parameters for processing
    create_output_filestructure(filename)
    item = get_track_dictionary(filename, begin, end, json)
    item['links'] = {'external': [input]}
    # Download and separate the audio
    logger.info(f"downloading and separating source audio ...")
    im = ItemMaker(
        item=item,
        output_filepath=f"{filename}/data",
        logger=logger,
        get_lr_audio=len(item['channel_overrides'].keys()) > 1,
    )
    im.get_item()
    im.separate_audio()
    im.finalize_output()
    logger.info(f"... the audio can be found in {filename}/data")
    # Detect onsets and beats in the audio
    logger.info(f"running detection on separated audio ...")
    om = OnsetMaker(
        corpus_name=params,
        item=item,
        output_filepath=filename,
        references_filepath=f"{utils.get_project_root()}/references",
        click_track_dir=f'{filename}/outputs',
        generate_click=generate_click
    )
    logger.info(f"... detecting beats")
    om.process_mixed_audio(generate_click)
    logger.info(f"... detecting onsets")
    om.process_separated_audio(generate_click, remove_silence=True)
    om.finalize_output()
    utils.save_annotations(om, f"{filename}/annotations")
    logger.info(f"... the annotations can be found in {filename}/annotations")
    # Extract features from the annotations
    logger.info(f"extracting features from detected annotations ...")
    features = extract_track_features(om, exog_ins)
    utils.save_json(features, f'{filename}/outputs', f'{filename}_features')
    # Make predictions
    predict = make_pianist_prediction(features)
    logger.info(predict)
    logger.info(f"... the features can be found in {filename}/outputs")
    logger.info(f"done !")


@click.command()
@click.option(
    "--input", "-i", type=str, default=None, help='Input to process (either filepath or YouTube link)'
)
@click.option(
    "--json", '-j', type=click.Path(), default='',  help='The file to use to configure track options'
)
@click.option(
    "--params", '-p', type=click.Path(), default='corpus_chronology',
    help='The name of a folder containing parameter settings inside `references/parameter_optimisation`'
)
@click.option(
    "--begin", "-b", type=str, default='00:01', help='Starting timestamp (in %H:%M:%S or %M:%S format)'
)
@click.option(
    "--end", "-e", type=str, default='01:01', help='Stopping timestamp (in %H:%M:%S or %M:%S format)'
)
@click.option(
    "--instr", "exog_ins", default='piano', help='Extract features for this instrument (defaults to piano)'
)
@click.option(
    "--no_click", "generate_click", is_flag=True, default=True, help='Suppress click track generation'
)
def proc(
        input: str,
        json: str,
        params: str,
        begin: str,
        end: str,
        exog_ins: str,
        generate_click: bool,
):
    """Main processing function, to be run from the command line using arguments parsed by `click`"""
    proc_inner(input, json, params, begin, end, exog_ins, generate_click)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    proc()
