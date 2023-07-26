#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Optimises the parameters used in detect_onsets.py by running a large-scale search over multiple items in
the corpus and comparing the accuracy of the detected onsets to a reference set of onsets annotated manually
"""

import logging
from datetime import datetime
from itertools import product
from pathlib import Path

import click
import nlopt
import numpy as np
from dotenv import find_dotenv, load_dotenv
from joblib import Parallel, delayed

from src.analyse.detect_onsets import OnsetMaker
from src.utils import analyse_utils as autils

# These are the already optimised parameters that will be passed in to the relevant function
onset_strength_optimised_params = {
    'piano': dict(
        fmin=110,  # Minimum frequency to use
        fmax=4100,  # Maximum frequency to use
        # TODO: think about using center=False/center=True?
    ),
    'bass': dict(
        fmin=30,
        fmax=500,
    ),
    'drums': dict(
        fmin=3500,
        fmax=11000,
    ),
}
# These are the parameters that we will seek to optimise. The key of each dictionary should be an argument that is
# accepted by the specific function, and the value should be an iterable that will be cycled during optimisation
beat_track_rnn_test_params = dict(
    threshold=[i / 100 for i in range(1, 50, 5)],
    transition_lambda=range(10, 500, 50),
    correct=[True, False],
    passes=range(1, 10),
)


class Optimizer:
    """Base class for non-linear optimization of onset detection parameters"""
    # Define the directory to store results from optimization in, and create it if it doesn't already exist
    results_fpath = rf'{autils.get_project_root()}\references\parameter_optimisation'

    def __init__(self, item: dict, instr: str, args: list[tuple], **kwargs):
        self.init_time = datetime.now().strftime("%d-%m-%y_%H-%M-%S")
        # The dictionary from the corpus JSON containing item metadata
        self.item = item
        # The name of the track we're optimizing
        self.instr = instr
        # Create the onset detection maker class
        self.made = OnsetMaker(item=self.item)
        # Initialise the arguments for optimization
        # This should be in the form of a list of tuples, ("arg_name", dtype, lower_bound, upper_bound, initial_guess)
        self.args = args
        self.N = len(self.args)
        # Initialise the optimizer
        self.opt = nlopt.opt(kwargs.get('opt', nlopt.LN_SBPLX), self.N)
        # Set the objective function: we're always trying to maximize the F-score
        self.opt.set_max_objective(self.objective_function)
        # Set upper and lower bounds for each argument
        self.opt.set_lower_bounds([i[2] for i in self.args])
        self.opt.set_upper_bounds([i[3] for i in self.args])
        # If we exceed this F-score, we treat the optimization as having converged
        self.opt.set_stopval(kwargs.get('stopval', 0.99))
        # If the distance between two successive optimization results is less than this value, we've converged
        self.opt.set_ftol_abs(kwargs.get('ftol_abs', 1e-5))
        self.opt.set_ftol_rel(-1)
        # The maximum number of iterations, defaults to -1 (i.e. no maximum)
        self.opt.set_maxeval(kwargs.get('maxeval', -1))
        # The maximum time (in seconds) to optimize for, breaks after exceeding this, defaults to 10 minutes
        self.opt.set_maxtime(kwargs.get('maxtime', 600))
        # Empty dictionary to store results from optimization in
        self.optimization_results = []

    def log_results(self) -> None:
        """Logs optimization results as a JSON file"""
        try:
            loaded = autils.load_json(self.results_fpath, f'{self.item["fname"]}_{self.instr}_{self.init_time}')
        except FileNotFoundError:
            loaded = []
        loaded.extend(self.optimization_results)
        autils.save_json(loaded, self.results_fpath, f'{self.item["fname"]}_{self.instr}_{self.init_time}')

    def return_kwargs(self, x: np.ndarray) -> dict:
        """Formats arguments from NLopt into the required keyword argument format"""
        return {fmt[0]: fmt[1](unfmt) for fmt, unfmt in zip(self.args, x)}

    def objective_function(self, x, grad) -> None:
        """Placeholder for objective function to be optimized; overridden in child classes"""
        return

    def run_optimization(self) -> tuple[dict, float]:
        """Runs optimization in NLopt"""
        x_optimal = self.opt.optimize([i[4] for i in self.args])
        return self.return_kwargs(x_optimal), self.opt.last_optimum_value()


class OptimizeOnsetDetect(Optimizer):
    """Optimizes the `OnsetMaker.onset_detect` and `OnsetMaker.onset_strength` function for a track + instrument"""
    args = [
        ('max_size', int, 1, 200, 5),
        ('wait', int, 0, 100, 5),
        ('delta', float, 0, 1, 0.05),
        ('pre_max', int, 0, 100, 5),
        ('post_max', int, 1, 100, 5),
        ('pre_avg', int, 0, 100, 5),
        ('post_avg', int, 1, 100, 5)
    ]

    def __init__(self, item: dict, instr: str, backtrack: bool, **kwargs):
        super().__init__(item, instr, self.args, **kwargs)
        self.backtrack = backtrack

    def get_onset_envelope(self, max_size: int) -> np.ndarray:
        """Returns onset envelope using given max_size argument from optimizer"""
        return self.made.onset_strength(
            self.instr,
            max_size=max_size,    # This is the only argument we use from our optimizer for this function
            **onset_strength_optimised_params[self.instr]    # These arguments are our frequency bands
        )

    def get_onsets(self, **kwargs) -> np.ndarray:
        """Returns detected onsets using all arguments (barring max_size) from optimizer"""
        return self.made.onset_detect(
            instr=self.instr,
            env=self.made.env[self.instr],
            backtrack=self.backtrack,
            **kwargs    # We pass in all the arguments from our optimizer here, barring max_size
        )

    def get_f_score(self,) -> float:
        """Returns F-score between detected onsets and manual annotation file"""
        fn = rf'{autils.get_project_root()}\references\manual_annotation\{self.item["fname"]}_{self.instr}.txt'
        # TODO: this is gross, fix
        return list(
            self.made.compare_onset_detection_accuracy(fname=fn, instr=self.instr, onsets=[self.made.ons[self.instr]])
        )[0]['f_score']

    def objective_function(self, x: np.ndarray, grad) -> float:
        """Objective function for maximising F-score of detected onsets"""
        # Format our keyword arguments into the required format
        kwargs = self.return_kwargs(x)
        # Get the max size keyword argument from the optimizer and use it to create our onset envelope
        max_size = autils.try_get_kwarg_and_remove('max_size', kwargs)
        self.made.env[self.instr] = self.get_onset_envelope(max_size)
        # Detect onsets using all the remaining keyword arguments
        self.made.ons[self.instr] = self.get_onsets(**kwargs)
        # Get F-score between manual annotation and detected onsets
        f_score = self.get_f_score()
        # Append the results from this iteration and log them in the JSON for this track/instrument combination
        self.optimization_results.append(dict(
            track_name=self.item['track_name'],
            mbz_id=self.item['mbz_id'],
            fname=self.item['fname'],
            instrument=self.instr,
            f_score=f_score,
            backtrack=self.backtrack,
            iterations=self.opt.get_numevals(),
            max_size=max_size,
            **kwargs
        ))
        self.log_results()
        # Return value of the function to use in setting the next set of arguments
        return f_score


class OptimizeBeatTrack(Optimizer):
    """Optimizes the OnsetMaker.beat_track_rnn function for the mixed audio"""
    args = [
        ('threshold', float, 0.01, 1, 0.05),
        ('transition_lambda', float, 10, 1000, 50),
        ('passes', int, 1, 10, 2)
    ]

    def __init__(self, item: dict, **kwargs):
        super().__init__(item, 'mix', self.args, **kwargs)
        self.made.env['mix'] = self.made.onset_strength('mix')

    def objective_function(self, x: np.ndarray, grad) -> float:
        """Objective function for maximising f-score of detected crotchet beat positions"""
        kwargs = self.return_kwargs(x)
        ons = self.made.beat_track_rnn(**kwargs)
        fn = rf'{autils.get_project_root()}\references\manual_annotation\{self.item["fname"]}_{self.instr}.txt'
        f_score = list(self.made.compare_onset_detection_accuracy(fname=fn, instr=self.instr, onsets=[ons]))[0][
            'f_score']
        self.optimization_results.append(dict(
            track_name=self.item['track_name'],
            mbz_id=self.item['mbz_id'],
            fname=self.item['fname'],
            instrument=self.instr,
            f_score=f_score,
            optimized=False
            **kwargs
        ))
        self.log_results()
        return f_score


def optimize_onset_detection(backtrack: bool = False, n_jobs: int = -1, **kwargs) -> list[dict]:
    """Central function for optimizing onset detection across all reference tracks and instrument stems

    Arguments:
        backtrack (bool): whether to backtrack detected onsets, defaults to false
        n_jobs (int): number of CPU cores to use in parallelisation, defaults to all available cores
        **kwargs: passed onto optimization class

    Returns:
        list[dict]: the optimization results, each dictionary corresponding to one track/stem

    """
    fp = rf'{autils.get_project_root()}\references\parameter_optimisation'
    def open_optimized_tracks_file():
        try:
            optimized_res = autils.load_json(fp, 'onset_detect_optimised')
        except FileNotFoundError:
            optimized_res = []
        return optimized_res

    def optimize_(track_: dict, instrument_: str) -> list[dict]:
        """Optimize f-score for one track and instrument, run in parallel"""
        # We need to initialise the logger here again, otherwise it won't work with joblib
        fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        logger_ = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, format=fmt)
        # Log the track we're optimizing
        msg = f'... now optimizing item {track_["mbz_id"]}, track name {track_["track_name"]}, instrument {instrument_}'
        logger_.info(msg)
        # Create the optimizer object and run optimization, then get the optimizer arguments and corresponding f-score
        o = OptimizeOnsetDetect(track_, instrument_, backtrack=backtrack, **kwargs)
        optimized_args, optimized_f_score = o.run_optimization()
        # Try and load results from previous optimizer tracks, otherwise create an empty list
        # Store the results from this optimized track and return
        optimized_results = open_optimized_tracks_file()
        optimized_results.append(dict(
            track_name=track_['track_name'],
            mbz_id=track_['mbz_id'],
            fname=track_['fname'],
            instrument=instrument_,
            f_score=optimized_f_score,
            backtrack=False,
            iterations=o.opt.get_numevals(),
            **optimized_args
        ))
        autils.save_json(optimized_results, fp, 'onset_detect_optimised')
        return optimized_results

    def get_tracks_to_optimise() -> list[tuple]:
        """Get the combinations of tracks and instruments to optimize, with caching"""
        # Get the IDs of tracks with annotations and combine with the names of instruments
        annotated_tracks = set(product(autils.get_tracks_with_manual_annotations(), autils.INSTRS_TO_PERF.keys()))
        # Get the names and instruments of tracks we've already optimised
        cached_tracks = {(i['mbz_id'], i['instrument']) for i in open_optimized_tracks_file()}
        # Get the details of tracks which require optimisation
        requires_optimisation = list(annotated_tracks - cached_tracks)
        # Load in the corpus: we need to do this to get track metadata
        # TODO: this could probably be improved
        corpus_json = autils.load_json(rf'{autils.get_project_root()}\references', 'corpus_bilL_evans')
        to_optimise = []
        for id_, instrument in requires_optimisation:
            to_optimise.extend([(t, instrument) for t in corpus_json if t['mbz_id'] == id_])
        return to_optimise

    # Load in the results for tracks which have already been optimized
    tracks_to_optimise = get_tracks_to_optimise()
    # Log the number of tracks we're optimizing
    logger = logging.getLogger(__name__)
    logger.info(f"optimising parameters across {len(tracks_to_optimise)} track/instrument combinations ...")
    res = Parallel(n_jobs=n_jobs)(delayed(optimize_)(track, instrument) for track, instrument in tracks_to_optimise)
    return res


@click.command()
@click.option(
    "-optimize_stems", "optimize_stems", is_flag=True, default=True,
    help='Optimize onset detection in stems (bass, drums)'
)
@click.option(
    "-optimize_mix", "optimize_mix", is_flag=True, default=False, help='Optimize beat detection in mixed audio'
)
@click.option(
    "-maxeval", "maxeval", type=click.IntRange(-1, clamp=True), default=-1,
    help='Maximum number of iterations to use when optimizing before forcing convergence (-1 default = no limit)'
)
@click.option(
    "-maxtime", "maxtime", type=click.IntRange(1, clamp=True), default=600,
    help='Maximum number of seconds to wait when optimizing before forcing convergence (600 default = 10 minutes)'
)
@click.option(
    "-n_jobs", "n_jobs", type=click.IntRange(-1, clamp=True), default=-1,
    help='Number of CPU cores to use in parallel processing, defaults to maximum available'
)
def main(
        optimize_stems: bool,
        optimize_mix: bool,
        maxeval: int,
        maxtime: int,
        n_jobs: int
):
    # Configure the logger here
    logger = logging.getLogger(__name__)
    logger.addHandler(autils.TqdmLoggingHandler())
    if optimize_stems:
        logger.info(f'optimizing onset detection for {", ".join(i for i in autils.INSTRS_TO_PERF.keys())} ...')
        optimize_onset_detection(n_jobs=n_jobs, maxtime=maxtime, maxeval=maxeval)
        logger.info(f"... finished optimizing onset detection !")
    if optimize_mix:
        logger.info(f'optimizing beat detection for raw audio ...')


if __name__ == '__main__':
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
