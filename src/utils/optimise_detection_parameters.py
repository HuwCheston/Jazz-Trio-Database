#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Optimises the parameters used in detect_onsets.py by running a large-scale search over multiple items in
the corpus and comparing the accuracy of the detected onsets to a reference (manually-detected onsets)
"""

import logging
import warnings
from ast import literal_eval

import pandas as pd

from src.analyse.detect_onsets import OnsetMaker
from src.utils import analyse_utils as autils

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
        fmax=16000,
        max_size=1,
        center=True
    )
}
onset_detect_optimised_params = {
    'piano': dict(),
    'bass': dict(),
    'drums': dict(),
    'mix_plp': dict(),
    'mix_rnn': dict()
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
    ),
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
beat_track_plp_test_params = dict(
    win_length=range(10, 1000, 10),
    passes=range(1, 10)
)
beat_track_rnn_test_params = dict(
    threshold=[i / 100 for i in range(1, 50, 5)],
    transition_lambda=range(10, 500, 50),
    correct=[True, False],
    passes=range(1, 10),
)
polyphonic_onset_detect_test_params = dict(
    onset_threshold=[i / 100 for i in range(1, 100)],
    frame_threshold=[i / 100 for i in range(1, 100)],
    minimum_note_length=range(1, 100),
    melodia_trick=[True, False],
    multiple_pitch_bends=[True, False]
)
# These constants are the extensions used to identify individual tracks
source_sep = ['bass', 'drums', 'piano']
raw_audio = ['mix']


def set_new_optimised_values(
        res: list,
        dic: dict,
) -> None:
    """
    Updates parameters dictionary with arguments that lead to best results in onset detection, i.e. highest F-score
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
        made_: OnsetMaker,
        instr: str,
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
        made_: OnsetMaker,
        instr: str,
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


def _optimise_polyphonic_onset_detect(
        param: str,
        vals: list,
        made_: OnsetMaker,
        instr: str,
):
    """
    This function should be passed in as the optimise_func to a call to optimise_parameters to run optimisation over
    the OnsetDetectionMaker.polyphonic_onset_detect function
    """

    # Iterate through all of the values we're testing for this parameter
    for val in vals:
        # Detect the onsets, with our current parameter as a kwarg
        yield made_.polyphonic_onset_detect(
            instr=instr,
            use_nonoptimised_defaults=True,
            **polyphonic_onset_detect_optimised_params[instr],
            **{param: val}
        )


def _optimise_beat_track_plp(
        param: str,
        vals: list,
        made_: OnsetMaker,
        instr: str,
):
    """
    This function should be passed in as the optimise_func to a call to optimise_parameters to run optimisation over
    the OnsetDetectionMaker.beat_track_full_mix function
    """

    made_.env[instr] = made_.onset_strength(
        instr,
        use_nonoptimised_defaults=True,
        **onset_strength_optimised_params[instr]
    )
    for val in vals:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            yield made_.beat_track_plp(
                env=made_.env[instr],
                **onset_detect_optimised_params[instr + '_plp'],
                **{param: val}
            )


def _optimise_beat_track_rnn(
        param: str,
        vals: list,
        made_: OnsetMaker,
        instr: str,
):
    """
    This function should be passed in as the optimise_func to a call to optimise_parameters to run optimisation over
    the OnsetDetectionMaker.beat_track_full_mix function
    """

    for val in vals:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            yield made_.beat_track_rnn(
                use_nonoptimised_defaults=True,
                **onset_detect_optimised_params[instr + '_rnn'],
                **{param: val}
            )


def optimise_parameters(
        annotated: list,
        corpus: list,
        instrs_to_optimise: tuple,
        params_to_test: dict,
        params_to_optimise: dict,
        optimise_func,
        append_in_progress: bool = True,
) -> None:
    """
    Run optimisation for one function across all reference tracks and instruments

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
    for k, v in params_to_test.items():
        counter = 0
        logger.info(f'optimising parameter {k}: {len(v)} values, range [{min(v)} ... {max(v)}]')
        # Iterate through every item in the corpus
        for corpus_item in corpus:
            # If we have manually-annotated onsets for this item
            if corpus_item['fname'] in annotated:
                counter += 1
                # Create the onset detection maker instance for this item
                made = OnsetMaker(item=corpus_item)
                logger.info(f'... track {counter}/{len(annotated)}, {corpus_item["fname"]}')
                # Iterate over all the instruments we want to run optimisation for and log to stdout
                for ins in instrs_to_optimise:
                    # Run the optimisation function with our required arguments to obtain a list of detected onsets
                    ons = optimise_func(
                        k,    # The name of the parameter to be optimised
                        v,    # The iterable of possible values this parameter can take
                        made,    # The maker class for this track in the corpus
                        ins,    # The instrument name we're optimising
                    )
                    # Remove onsets in silent passages
                    if ins in made.top_db.keys():
                        ons = [made.remove_onsets_in_silent_passages(o, instr=ins) for o in ons]
                    # Calculate the F-score for each of our parameters and append to the list
                    fn = rf'{autils.get_project_root()}\references\manual_annotation\{corpus_item["fname"]}_{ins}.txt'
                    li = list(made.compare_onset_detection_accuracy(
                        fname=fn,
                        instr=ins,
                        onsets=ons,
                        onsets_name=v,
                        param=k,
                        track=corpus_item['track_name']
                    ))
                    res.append(li)
        # If we're updating our parameters during optimisation, do so now
        if append_in_progress:
            set_new_optimised_values(
                res=[item for sublist in res for item in sublist],
                dic=params_to_optimise
            )
            res.clear()
    # Otherwise, update our parameters after completing optimisation across all tracks/instruments/parameters
    if not append_in_progress:
        set_new_optimised_values(
            res=[item for sublist in res for item in sublist],
            dic=params_to_optimise
        )
    # Set the dictionary values to our optimised results
    if not append_in_progress:
        set_new_optimised_values(
            res=[item for sublist in res for item in sublist],
            dic=params_to_optimise
        )


if __name__ == "__main__":
    annotated_tracks = autils.get_tracks_with_manual_annotations()

    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    logger.info(f"optimising parameters using manual annotations obtained for {len(annotated_tracks)} tracks ...")

    corpus_json = autils.load_json(rf'{autils.get_project_root()}\references', 'corpus')

    # Optimise the made.beat_track_full_mix function
    optimise_parameters(
        annotated=annotated_tracks,
        corpus=corpus_json,
        instrs_to_optimise=raw_audio,
        params_to_test=beat_track_rnn_test_params,
        params_to_optimise=onset_detect_optimised_params,
        optimise_func=_optimise_beat_track_rnn
    )
    # Optimise the made.beat_track_full_mix function
    optimise_parameters(
        annotated=annotated_tracks,
        corpus=corpus_json,
        instrs_to_optimise=raw_audio,
        params_to_test=beat_track_plp_test_params,
        params_to_optimise=onset_detect_optimised_params,
        optimise_func=_optimise_beat_track_plp
    )
    # Optimise the made.onset_strength function
    optimise_parameters(
        annotated=annotated_tracks,
        corpus=corpus_json,
        instrs_to_optimise=source_sep,
        params_to_test=onset_strength_test_params,
        params_to_optimise=onset_strength_optimised_params,
        optimise_func=_optimise_onset_strength
    )
    # Optimise the made.onset_detect function
    optimise_parameters(
        annotated=annotated_tracks,
        corpus=corpus_json,
        instrs_to_optimise=source_sep,
        params_to_test=onset_detect_test_params,
        params_to_optimise=onset_detect_optimised_params,
        optimise_func=_optimise_onset_detect
    )
    # Serialise the results
    autils.save_json(
        obj=onset_strength_optimised_params,
        fpath=rf'{autils.get_project_root()}\references\optimised_parameters',
        fname='onset_strength_optimised'
    )
    autils.save_json(
        obj=onset_detect_optimised_params,
        fpath=rf'{autils.get_project_root()}\references\optimised_parameters',
        fname='onset_detect_optimised'
    )
