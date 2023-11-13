#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Classes used for creating ensemble coordination simulations from the phase correction model"""


from datetime import timedelta

import numba as nb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src import utils
from src.features.features_utils import Asynchrony


__all__ = ['Simulation', 'SimulationManager']


class Simulation:
    """Creates a single simulated performance with given `params_dict`"""
    starting_onset = 0

    def __init__(self, params_dict, n_beats: int = 100, tempo: int = 120):
        # Get our desired number of beats and tempo
        self.n_beats = n_beats
        # These are our starting inter-onset intervals, taken from our tempo
        self.starting_iois = (60 / tempo, 60 / tempo)
        # Populate our data dictionaries with starter data
        self._piano_params = self._format_dict(params_dict['piano'])
        self._bass_params = self._format_dict(params_dict['bass'])
        self._drums_params = self._format_dict(params_dict['drums'])
        # Get initial empty data
        self.piano = self._get_initial_data('piano')
        self.bass = self._get_initial_data('bass')
        self.drums = self._get_initial_data('drums')
        # Empty variables to hold simulation data and `src.features.features_utils.Asynchrony` classes
        self.sim_df = None
        self.async_cls = None
        self.async_rms = np.nan
        self.bpm = None

    def __repr__(self) -> str:
        """Overloads representation function to string representation of simulation dataframe"""
        return str(self.sim_df)

    @staticmethod
    def _format_dict(python_dict: dict) -> nb.typed.Dict:
        """Converts a Python dictionary into a type that can be utilised by Numba"""
        # Create the empty dictionary
        nb_dict = nb.typed.Dict.empty(key_type=nb.types.unicode_type, value_type=nb.types.float64,)
        # Iterate through our dictionary
        for k, v in python_dict.items():
            # If the type is compatible with numba floats, set it in the numba dictionary
            if type(v) != str:
                nb_dict[k] = v
        return nb_dict

    def _get_initial_data(self, init_instr: str) -> nb.typed.Dict:
        """Gets initial starter data for use when creating the simulation"""
        # Create our empty Numba dictionary
        init_dict = nb.typed.Dict.empty(
            key_type=nb.types.unicode_type,
            value_type=nb.types.float64[:],
        )
        # Get the keys for all the columns we need
        others = [i for i in utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys() if i != init_instr]
        needed = [
            'my_onset', 'my_next_ioi', 'my_prev_ioi',
            'my_next_ioi_diff', 'my_prev_ioi_diff',
            *[f'{i}_asynchrony' for i in others]
        ]
        # Assign empty arrays for each variable we need as values in the dictionary
        for s in needed:
            init_dict[s] = np.zeros(shape=self.n_beats)
        # Fill the dictionary arrays with our starting values
        # My onset
        init_dict['my_onset'][0] = self.starting_onset
        init_dict['my_onset'][1] = self.starting_onset + self.starting_iois[0]
        # My next ioi
        init_dict['my_next_ioi'][0] = self.starting_iois[0]
        init_dict['my_next_ioi'][1] = self.starting_iois[1]
        # My previous ioi
        init_dict['my_prev_ioi'][0] = np.nan
        init_dict['my_prev_ioi'][1] = self.starting_iois[0]
        # My next ioi diff
        init_dict['my_next_ioi_diff'][0] = np.nan
        init_dict['my_next_ioi_diff'][1] = self.starting_iois[1] - self.starting_iois[0]  # This will always be 0
        # My previous ioi diff
        init_dict['my_prev_ioi_diff'][0] = np.nan
        init_dict['my_prev_ioi_diff'][1] = np.nan
        return init_dict

    def _get_async_cls(self) -> Asynchrony:
        """Gets all `src.features.features_utils.Asynchrony` classes for all instruments"""
        get_other = lambda me: [i_ for i_ in utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys() if i_ != me]
        for i in utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys():
            yield Asynchrony(my_beats=self.sim_df[i], their_beats=self.sim_df[get_other(i)])

    def _get_async_rms(self) -> float:
        """Gets root-mean-square of all pairwise asynchrony values"""
        def __get_pwa():
            for pwa in self.async_cls:
                for key, val in pwa.summary_dict.items():
                    if 'pairwise_asynchrony' in key and not np.isnan(val):
                        yield val

        values = list(__get_pwa())
        # Square the asynchrony values
        # Making this a set drops duplicate values
        squares = set([x ** 2 for x in values])
        # Get the mean of all squared values
        mean = np.mean(list(squares))
        # Return the square root of the mean
        return np.sqrt(mean)

    def _get_bpm_values(self):
        bpms = (60 / self.sim_df.diff())
        idxs = pd.to_timedelta([timedelta(seconds=val) for val in self.sim_df.mean(axis=1)])
        offset = -timedelta(seconds=idxs[0].total_seconds() - 1)
        return (
            bpms.set_index(idxs)
            .resample('1s', offset=offset)
            .apply(np.nanmean)
            .mean(axis=1)
        )

    def run_simulation(self):
        """Dispatcher function for a single simulation"""
        # Dispatch the simulation
        all_data = (self.piano, self.bass, self.drums)
        self.piano, self.bass, self.drums = self._simulation_dispatcher(
            data_=all_data,
            params_=(self._piano_params, self._bass_params, self._drums_params)
        )
        # Format a dataframe from our results
        self.sim_df = (
            pd.concat([pd.Series(arr['my_onset']) for arr in all_data], axis=1)
            .rename(columns={0: 'piano', 1: 'bass', 2: 'drums'})
            .iloc[len(self.starting_iois) + 1:]
            .reset_index(drop=True)
        )
        # Construct all of our asynchrony classes, then get RMS from this
        self.async_cls = self._get_async_cls()
        self.async_rms = self._get_async_rms()
        self.bpm = self._get_bpm_values()
        # Return the class instance for storing in the simulation manager
        return self

    @staticmethod
    @nb.jit(nopython=True)
    def _simulation_dispatcher(data_: tuple, params_: tuple) -> tuple:
        """Creates one simulated performance, optimized with numba"""
        def _compute_next_diff(my_data: dict, my_params, other_instrs: list, num) -> float:
            """Estimates the next onset difference by running the regression equation"""
            return (
                my_params['intercept'] +
                (my_data['my_prev_ioi_diff'][num] * my_params['self_coupling']) +
                sum([my_data[f'{o}_asynchrony'][num] * my_params[f'coupling_{o}'] for o in other_instrs]) +
                np.random.normal(0, my_params['resid_std'], 1)[0]
            )

        # Unpack our input tuples
        piano, bass, drums = data_
        piano_params, bass_params, drums_params = params_
        # Iterate over all the beats that we need to get
        for i in range(2, len(piano['my_prev_ioi_diff'])):
            # Shift difference
            piano['my_prev_ioi_diff'][i] = piano['my_next_ioi_diff'][i - 1]
            bass['my_prev_ioi_diff'][i] = bass['my_next_ioi_diff'][i - 1]
            drums['my_prev_ioi_diff'][i] = drums['my_next_ioi_diff'][i - 1]
            # Shift IOI
            piano['my_prev_ioi'][i] = piano['my_next_ioi'][i - 1]
            bass['my_prev_ioi'][i] = bass['my_next_ioi'][i - 1]
            drums['my_prev_ioi'][i] = drums['my_next_ioi'][i - 1]
            # Get next onset by adding previous onset to predicted IOI
            piano['my_onset'][i] = piano['my_onset'][i - 1] + piano['my_prev_ioi'][i]
            bass['my_onset'][i] = bass['my_onset'][i - 1] + bass['my_prev_ioi'][i]
            drums['my_onset'][i] = drums['my_onset'][i - 1] + drums['my_prev_ioi'][i]
            # Get async values for piano
            piano['bass_asynchrony'][i] = bass['my_onset'][i] - piano['my_onset'][i]
            piano['drums_asynchrony'][i] = drums['my_onset'][i] - piano['my_onset'][i]
            # Get async values for bass
            bass['piano_asynchrony'][i] = piano['my_onset'][i] - bass['my_onset'][i]
            bass['drums_asynchrony'][i] = drums['my_onset'][i] - bass['my_onset'][i]
            # Get async values for drums
            drums['piano_asynchrony'][i] = piano['my_onset'][i] - drums['my_onset'][i]
            drums['bass_asynchrony'][i] = bass['my_onset'][i] - drums['my_onset'][i]
            # Predict next IOI diff
            piano['my_next_ioi_diff'][i] = _compute_next_diff(piano, piano_params, ['bass', 'drums'], i)
            bass['my_next_ioi_diff'][i] = _compute_next_diff(bass, bass_params, ['piano', 'drums'], i)
            drums['my_next_ioi_diff'][i] = _compute_next_diff(drums, drums_params, ['piano', 'bass'], i)
            # Use predicted difference between IOIs to get next actual IOI
            piano['my_next_ioi'][i] = piano['my_next_ioi_diff'][i] + piano['my_prev_ioi'][i]
            bass['my_next_ioi'][i] = bass['my_next_ioi_diff'][i] + bass['my_prev_ioi'][i]
            drums['my_next_ioi'][i] = drums['my_next_ioi_diff'][i] + drums['my_prev_ioi'][i]
            # If we've accelerated to a ridiculous extent (due to noise), we need to break.
            if piano['my_next_ioi'][i] < 0 or bass['my_next_ioi'][i] < 0 or drums['my_next_ioi'][i] < 0:
                break
        # Return the complete Numba dictionary after
        return piano, bass, drums


class SimulationManager:
    """Manager for creating and handling multiple `Simulation` instances."""

    def __init__(
            self,
            coupling_params,
            tempo: int = 120,
            n_sims: int = 500,
            n_beats: int = 100,
            n_jobs: int = -1
    ):
        self.params = coupling_params
        self.n_sims = n_sims
        self.n_jobs = n_jobs
        args = dict(tempo=tempo, n_beats=n_beats)
        self.simulations = [Simulation(self.params, **args) for _ in range(self.n_sims)]

    def get_mean_bpm(self) -> pd.Series:
        return pd.concat([sim.bpm for sim in self.simulations], axis=1).mean(axis=1)

    def get_mean_rms(self) -> float:
        return np.nanmean([sim.async_rms for sim in self.simulations])

    def get_rms_values(self) -> np.array:
        return np.array([sim.async_rms for sim in self.simulations if not np.isnan(sim.async_rms)])

    def run_simulations(self):
        with Parallel(n_jobs=self.n_jobs, prefer='threads', verbose=5) as parallel:
            self.simulations = parallel(delayed(sim.run_simulation)() for sim in self.simulations)
        return self
