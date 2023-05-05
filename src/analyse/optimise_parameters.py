#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Optimises the parameters used in build_features.py by running a large-scale search over multiple items in
the corpus and comparing the accuracy of the detected onsets to a reference (manually-detected onsets)
"""

import json
import logging
import os
from ast import literal_eval

import pandas as pd

from src.analyse.build_features import OnsetDetectionMaker

# These are the already optimised parameters that will be passed in to the relevant function
onset_strength_optimised_params = {
    'piano': dict(
        fmin=110,  # Minimum frequency to use
        fmax=4100,  # Maximum frequency to use
    ),
    'bass': dict(
        fmin=30,
        fmax=500,
    ),
    'drums': dict(
        fmin=3500,
        fmax=11000,
    ),
    'mix': dict(
        fmin=10,
        fmax=16000
    )
}
onset_detect_optimised_params = {
    'piano': dict(),
    'bass': dict(),
    'drums': dict(),
    'mix': dict()
}
polyphonic_onset_detect_optimised_params = {
    'bass': dict(
        minimum_frequency=30,  # Same as passed to onset_strength
        maximum_frequency=500,  # Same as passed to onset_strength
    ),
    'piano': dict(
        minimum_frequency=110,
        maximum_frequency=4100,
    ),
    'drums': dict(
        minimum_frequency=3500,
        maximum_frequency=11000,
    )
}

# These are the parameters that we will seek to optimise. The key of each dictionary should be an argument that is
# accepted by the specific function, and the value should be an iterable that will be cycled during optimisation
onset_strength_test_params = dict(
    center=[True, False],
    max_size=range(1, 200, 2),  # 200
)
onset_detect_test_params = dict(
    backtrack=[True, False],
    wait=range(1, 100),
    delta=[i / 100 for i in range(1, 100)],
    pre_max=range(1, 100),
    post_max=range(1, 100),
    pre_avg=range(1, 100),
    post_avg=range(1, 100)
)
beat_track_full_mix_test_Params = dict(
    win_length=range(10, 1000)
)
polyphonic_onset_detect_test_params = dict(
    onset_threshold=[i / 100 for i in range(1, 10)],
    frame_threshold=[i / 100 for i in range(1, 10)],
    minimum_note_length=range(1, 100),
    melodia_trick=[True, False],
    multiple_pitch_bends=[True, False]
)
# These constants are the extensions used to identify individual tracks
source_sep = ['bass', 'drums', 'piano']
raw_audio = 'mix'
all_tracks = [raw_audio, *source_sep]


def get_tracks_with_manual_annotations(
        annotation_dir: str = r'..\..\references\manual_annotation',
        annotation_ext: str = 'txt',
) -> list:
    """

    """

    res = {}
    for file in os.listdir(annotation_dir):
        if file.endswith(annotation_ext):
            split = file.split('_')
            try:
                res[split[0]].append(split[1].replace(f'.{annotation_ext}', ''))
            except KeyError:
                res[split[0]] = []
                res[split[0]].append(split[1].replace(f'.{annotation_ext}', ''))

    return [k for k, v in res.items() if sorted(v) == sorted(list(all_tracks))]


def set_new_optimised_values(
        res: list,
        dic: dict,
) -> None:
    """

    """

    # Create a dataframe from our passed results list
    df = pd.DataFrame(res)
    # We set the name column to a string as this avoids issues with booleans being converted to integers and vice-versa
    df['name'] = df['name'].astype(str)
    # For each parameter and instrument, this gets the arguments that lead to the best results
    best = (
        df.groupby(['param', 'instr', 'name'])['f_score']
          .mean(numeric_only=True)
          .groupby(level=[0, 1])
          .idxmax()
          .reset_index(drop=True)
    )
    # Wrangle the data into the correct form
    new = pd.DataFrame()
    new['param'], new['instr'], new['val'] = zip(*best)
    # Iterate through each instrument and update the parameters dictionary with the optimised values
    for idx, grp in new.groupby('instr'):
        dic[idx].update(**pd.Series(grp['val'].apply(literal_eval).values, index=grp['param']).to_dict())


def _optimise_onset_strength(
        param: str,
        vals: list,
        made_: OnsetDetectionMaker,
        instr: str,
        **kwargs
):
    """
    This function should be passed in as the optimise_func to a call to optimise_parameters to run optimisation over
    the OnsetDetectionMaker.onset_strength function
    """

    # Iterate through all of the values we're testing for this parameter
    for val in vals:
        made_.env[instr] = made_.onset_strength(
            instr,
            use_nonoptimised_defaults=True,
            **onset_strength_optimised_params[instr],
            **{param: val},
        )
        # Detect the onsets, using default parameters
        yield made_.onset_detect(
            instr,
            env=made_.env[instr],
            use_nonoptimised_defaults=True,
        )


def _optimise_onset_detect(
        param: str,
        vals: list,
        made_: OnsetDetectionMaker,
        instr: str,
        **kwargs
):
    """
    This function should be passed in as the optimise_func to a call to optimise_parameters to run optimisation over
    the OnsetDetectionMaker.onset_detect function
    """

    # Iterate through all of the values we're testing for this parameter
    for val in vals:
        # Create the onset strength envelope
        made_.env[instr] = made_.onset_strength(
            instr,
            use_nonoptimised_defaults=True,
            **onset_strength_optimised_params[instr],
        )
        # Detect the onsets, with our current parameter as a kwarg
        yield made_.onset_detect(
            instr,
            env=made_.env[instr],
            use_nonoptimised_defaults=True,
            **onset_detect_optimised_params[instr],
            **{param: val}
        )


def optimise_parameters(
        annotated: list,
        corpus: list,
        instrs_to_optimise: tuple,
        params_to_test: dict,
        params_to_optimise: dict,
        optimise_func,
        **kwargs
) -> None:
    """
    Run optimisation for one function across all

    Arguments:
        - annotated: a list of track file names with annotated onsets present for all parts, returned from
    get_tracks_with_manual_annotations()
        - corpus: the corpus JSON file read as a dictionary
        - instrs_to_optimise: the names of instrument extensions to run optimisation over, e.g. 'bass', 'drums', 'piano'
        - params_to_test: the parameters we're using for our optimisation
        - params_to_optimise: the dictionary of parameters where our optimised values will be stored
        - optimise_func: a function used to run optimisation
        - **kwargs: passed to optimise_func

    """

    res = []
    # Iterate through every item in the corpus
    for corpus_item in corpus:
        # If we have manually-annotated onsets for this item
        if corpus_item['fname'] in annotated:
            # Create the onset detection maker instance
            made = OnsetDetectionMaker(item=corpus_item)
            # Iterate over all the instruments we want to run optimisation for and log to stdout
            for ins in instrs_to_optimise:
                logger.info(f'Optimising track {corpus_item["fname"]}, {ins} ...')
                # Iterate over the individual parameters we want to test and log some summary info
                for k, v in params_to_test.items():
                    logger.info(f'... {k} ({len(v)} values being tested, range {min(v)} -> {max(v)})')
                    # Run the optimisation function with our required arguments to obtain a list of detected onsets
                    ons = optimise_func(
                        k,
                        v,
                        made,
                        ins,
                        **kwargs
                    )
                    # Calculate the F-score for each of our parameters and append to the list
                    res.append(list(made.compare_onset_detection_accuracy(
                        fname=rf'..\..\references\manual_annotation\{corpus_item["fname"]}_{ins}.txt',
                        instr=ins,
                        onsets=ons,
                        onsets_name=v,
                        param=k,
                        track=corpus_item['track_name']
                    )))
    # Set the dictionary values to our optimised results
    set_new_optimised_values(
        res=[item for sublist in res for item in sublist],
        dic=params_to_optimise
    )


if __name__ == "__main__":
    annotated_tracks = get_tracks_with_manual_annotations()
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    logger = logging.getLogger(__name__)
    logger.info(f"optimising parameters using manual annotations obtained for {len(annotated_tracks)} tracks ...")

    with open(r'..\..\data\processed\processing_results.json', "r+") as in_file:
        corpus_json = json.load(in_file)
    optimise_parameters(
        annotated=annotated_tracks,
        corpus=corpus_json,
        instrs_to_optimise=source_sep,
        params_to_test=onset_strength_test_params,
        params_to_optimise=onset_strength_optimised_params,
        optimise_func=_optimise_onset_strength
    )
    print(onset_strength_optimised_params)
    optimise_parameters(
        annotated=annotated_tracks,
        corpus=corpus_json,
        instrs_to_optimise=source_sep,
        params_to_test=onset_detect_test_params,
        params_to_optimise=onset_detect_optimised_params,
        optimise_func=_optimise_onset_detect
    )
    print(onset_detect_optimised_params)
