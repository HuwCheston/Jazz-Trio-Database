#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility classes, functions, and variables used in extracting rhythmic features"""

import string
import warnings
from functools import reduce
from itertools import pairwise
from typing import Generator

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResultsWrapper

from src import utils
from src.features.features_utils import BaseExtractor


__all__ = [
    "PhaseCorrection", "BeatUpbeatRatio", "IOIComplexity", "TempoSlope",
    "ProportionalAsynchrony", "RollingIOISummaryStats", "Asynchrony",
    "get_beats_from_matched_onsets"
]


def get_beats_from_matched_onsets(summary_dict: dict) -> pd.DataFrame:
    """Gets mean beat timestamps from a `summary_dict` marked by more than two instruments in the trio"""
    # Convert the summary dictionary to a dataframe and subset to get beats played by the trio
    df = pd.DataFrame(summary_dict)
    tdf = df[['piano', 'bass', 'drums']]
    # Get rows where we have more than one value set to missing
    miss = np.where(tdf.isnull().sum(1) > 1, True, False)
    # Calculate mean onset position for all rows other than missing ones
    return pd.Series(np.where(miss, np.nan, tdf.mean(axis=1)))


class IOISummaryStats(BaseExtractor):
    """Extracts various baseline summary statistics from an array of IOIs

    Args:
        my_onsets (pd.Series): onsets to compute summary statistics for
        use_bpms (bool, optional): convert IOIs into beat-per-minute values, i.e. 60 / IOI (defaults to False)
        iqr_filter (bool, optional): apply IQR range filtering to IOI/BPM values (defaults to False)

    """
    def __init__(self, my_onsets, **kwargs):
        super().__init__()
        if isinstance(my_onsets, np.ndarray):
            my_onsets = pd.Series(my_onsets)
        iois = my_onsets.diff()
        name = 'iois'
        # Divide 60 / IOI if we want to use BPM values instead
        if kwargs.get('use_bpms', False):
            iois = 60 / iois
            name = 'bpms'
        # Filter our IOIs using an IQR filter, if required
        if kwargs.get('iqr_filter', False):
            iois = utils.iqr_filter(iois, fill_nans=True)
            name += '_filter'
        # Add in some extra functions to our summary functions dictionary
        self.summary_funcs['binary_entropy'] = self.binary_entropy
        self.summary_funcs['npvi'] = self.npvi
        self.summary_funcs['lempel_ziv_complexity'] = self.lempel_ziv_complexity
        # Update the summary dictionary by obtaining results for every function in `self.summary_funcs`
        self.update_summary_dict([name], [iois])

    @staticmethod
    def binary_entropy(iois: pd.Series) -> float:
        """Extract the Shannon entropy from an iterable"""
        # We convert our IOIs into milliseconds here to prevent floating point numbers
        ms_arr = (iois * 1000).dropna().astype(int).to_numpy()
        # Get the counts and probabilities of our individual IOIs
        _, counts = np.unique(ms_arr, return_counts=True)
        probabilities = counts / len(ms_arr)
        # Calculate the entropy and return
        return -np.sum(probabilities * np.log2(probabilities))
        # Alternative method using SciPy, should yield identical results
        # return stats.entropy((ioi * 1000).dropna().astype(int).value_counts().squeeze(), base=2)

    @staticmethod
    def npvi(iois: pd.Series) -> float:
        """Extract the normalised pairwise variability index (nPVI) from an iterable"""
        # Drop NaN values and convert array to Numpy
        dat = iois.dropna().to_numpy()
        # If we only have one element in our array after dropping NaN values, we can't calculate nPVI, so return NaN
        if len(dat) <= 1:
            return np.nan
        # Otherwise, we can go ahead and return the nPVI value for the array
        return sum([abs((k - k1) / ((k + k1) / 2)) for (k, k1) in zip(dat, dat[1:])]) * 100 / (sum(1 for _ in dat) - 1)

    @staticmethod
    def lempel_ziv_complexity(iois: pd.Series) -> float:
        """Extract complexity from a binary sequence using Lempel-Ziv compression algorithm,"""
        def lz(binary: np.array) -> int:
            """Function code for Lempel-Ziv compression algorithm"""
            # Convert our sequence into binary: values below mean = 0, above mean = 1
            # Set starting values for complexity calculation
            u, v, w = 0, 1, 1
            v_max, complexity = 1, 1
            # Begin calculating LZ complexity
            while True:
                if binary[u + v - 1] == binary[w + v - 1]:
                    v += 1
                    if w + v >= len(binary):
                        complexity += 1
                        break
                else:
                    if v > v_max:
                        v_max = v
                    u += 1
                    if u == w:
                        complexity += 1
                        w += v_max
                        if w > len(binary):
                            break
                        else:
                            u = 0
                            v = 1
                            v_max = 1
                    else:
                        v = 1
            return complexity

        # Try and convert our sequence to binary
        try:
            binary_sequence = np.vectorize(lambda x: int(x > np.nanmean(iois)))(iois[~np.isnan(iois)])
        # If we only have NaNs in our array we'll raise an error, so catch this and return NaN
        except ValueError:
            return np.nan
        # We need a sequence with at least 3 items in to calculate LZ complexity, so catch this and return NaN
        else:
            if len(binary_sequence) < 3:
                return np.nan
            # If we've passed all these checks, we should be able to calculate LZ complexity; do so now and return
            else:
                return lz(binary_sequence)


class RollingIOISummaryStats(IOISummaryStats):
    """Extracts the statistics in `IOISummaryStatsExtractor` on a rolling basis, window defaults to 4 bars length"""

    def __init__(self, my_onsets: pd.Series, downbeats, order: int = 4, **kwargs):
        super().__init__(my_onsets=my_onsets, **kwargs)
        if isinstance(my_onsets, np.ndarray):
            my_onsets = pd.Series(my_onsets)
        self.summary_dict.clear()
        # TODO: implement a time-based window here!
        self.bar_period = order
        # We get our raw rolling statistics here
        self.rolling_statistics = self.extract_rolling_statistics(my_onsets, downbeats, **kwargs)
        # We redefine summary_funcs here, as we want to remove extra functions added in `IOISummaryStatsExtractor`
        self.summary_funcs = BaseExtractor().summary_funcs
        # Update the summary dictionary
        self.summary_dict['bar_period'] = order
        self.update_summary_dict(self.rolling_statistics.keys(), self.rolling_statistics.values())

    def extract_rolling_statistics(self, my_onsets: pd.Series, downbeats: np.array, **kwargs) -> dict:
        """Extract rolling summary statistics across the given bar period"""
        results = {f'rolling_{func_k}': [] for func_k in self.summary_funcs.keys()}
        for bar_num, (i1, i2) in enumerate(zip(downbeats, downbeats[self.bar_period:]), 1):
            iois_between = pd.Series(self.get_between(my_onsets.values, i1, i2)).diff()
            # Divide 60 / IOI if we want to use BPM values instead
            if kwargs.get('use_bpms', False):
                iois_between = 60 / iois_between
            # Filter our IOIs using an IQR filter, if required
            if kwargs.get('iqr_filter', False):
                iois_between = utils.iqr_filter(iois_between, fill_nans=True)
            # Iterate through each of our summary functions
            for func_k, func_v in self.summary_funcs.items():
                # Try and apply the summary function to the IOIs, and return NaN on an error
                try:
                    results[f'rolling_{func_k}'].append(func_v(iois_between))
                # These are all the errors that can result from our summary functions with NaN arrays
                except (IndexError, ValueError, ZeroDivisionError):
                    results[f'rolling_{func_k}'].append(np.nan)
        return results


class EventDensity(BaseExtractor):
    """Extract various features related to event density, on both a per-bar and per-second basis.

    Args:
        my_onsets (pd.series): onsets to calculate event density for
        downbeats (np.array): array of times corresponding to the first beat of each bar
        time_period (str, optional): the timeframe to calculate event density over, defaults to '1s' (one second)
        bar_period (int, optional): the number of bars to calculate event density over, defaults to 1 (bar)

    """
    def __init__(
            self,
            my_onsets: pd.Series,
            downbeats: np.array,
            time_period: int = 1,
            bar_period: int = 1
    ):
        super().__init__()
        if isinstance(my_onsets, np.ndarray):
            my_onsets = pd.Series(my_onsets)
        # Set attributes
        self.time_period = f'{time_period}s'
        self.bar_period = bar_period
        # Extract event density
        self.per_second = self.extract_ed_per_second(my_onsets)
        self.per_bar = self.extract_ed_per_bar(my_onsets, downbeats)
        # Update our summary dictionary
        self.summary_dict['time_period'] = time_period
        self.summary_dict['bar_period'] = bar_period
        self.update_summary_dict(['ed_per_second', 'ed_per_bar'], [self.per_second['density'], self.per_bar['density']])

    def extract_ed_per_second(self, my_onsets) -> pd.DataFrame:
        """For every second in a performance, extract the number of notes played"""
        return (
            pd.DataFrame({'ts': pd.to_datetime(my_onsets, unit='s'), 'density': my_onsets})
            .set_index('ts')
            .resample(self.time_period, label='left')
            .count()
            .reset_index(drop=False)
        )

    def extract_ed_per_bar(self, my_onsets, quarter_note_downbeats) -> pd.DataFrame:
        """Extract the number of notes played within each specified bar period"""
        sequential_downbeats = zip(quarter_note_downbeats, quarter_note_downbeats[self.bar_period:])
        my_onsets_arr = my_onsets.to_numpy()
        matches = [
            {f'bars': f'{bar_num}-{bar_num + self.bar_period}', 'density': len(self.get_between(my_onsets_arr, i1, i2))}
            for bar_num, (i1, i2) in enumerate(sequential_downbeats, 1)
        ]
        return pd.DataFrame(matches)


class BeatUpbeatRatio(BaseExtractor):
    LOW_THRESH, HIGH_THRESH = 0.25, 4

    """Extract various features related to beat-upbeat ratios (BURs)"""
    def __init__(self, my_onsets, my_beats, clean_outliers: bool = True):
        super().__init__()
        if isinstance(my_onsets, np.ndarray):
            my_onsets = pd.Series(my_onsets)
        self.clean_outliers = clean_outliers
        # Extract our burs here, so we can access them as instance properties
        self.bur = self.extract_burs(my_onsets, my_beats, use_log_burs=False)
        self.bur_log = self.extract_burs(my_onsets, my_beats, use_log_burs=True)
        # Update our summary dictionary
        self.update_summary_dict(['bur', 'bur_log'], [self.bur['burs'], self.bur_log['burs']])

    def extract_burs(
            self,
            my_onsets: np.array,
            my_beats: np.array,
            use_log_burs: bool = False
    ) -> pd.DataFrame:
        """Extracts beat-upbeat ratio (BUR) values from an array of onsets.

        The beat-upbeat ratio is introduced in [1] as a concept for analyzing the individual amount of 'swing' in two
        consecutive eighth note beat durations. It is calculated simply by dividing the duration of the first, 'long'
        eighth note beat by the second, 'short' beat. A BUR value of 2 indicates 'perfect' swing, i.e. a triplet quarter
        note followed by a triplet eighth note, while a BUR of 1 indicates 'even' eighth note durations.

        Arguments:
            my_onsets (np.array, optional): the array of raw onsets.
            my_beats (np.array, optional): the array of crotchet beat positions.
            use_log_burs (bool, optional): whether to use the log^2 of inter-onset intervals to calculate BURs,
                as employed in [2]. Defaults to False.

        Returns:
            np.array: the calculated BUR values

        References:
            [1]: Benadon, F. (2006). Slicing the Beat: Jazz Eighth-Notes as Expressive Microrhythm. Ethnomusicology,
                50/1 (pp. 73-98).
            [2]: Corcoran, C., & Frieler, K. (2021). Playing It Straight: Analyzing Jazz Soloists’ Swing Eighth-Note
                Distributions with the Weimar Jazz Database. Music Perception, 38(4), 372–385.

        """

        # Use log2 burs if specified
        func = lambda a: a
        if use_log_burs:
            from math import log2 as func

        def bur(a: float, b: float) -> float:
            """BUR calculation function"""
            # Get the onsets between our first and second beat (a and b)
            match = self.get_between(my_onsets, a, b)
            # If we have a group of three notes (i.e. three quavers), including a and b
            if len(match) == 3:
                bur_val = func((match[1] - match[0]) / (match[2] - match[1]))
                # If we're cleaning outliers
                if self.clean_outliers:
                    # If the BUR is above or high threshold, or below our low threshold, return NaN
                    if bur_val > func(self.HIGH_THRESH) or bur_val < func(self.LOW_THRESH):
                        return np.nan
                    # Otherwise, return the BUR
                    else:
                        return bur_val
                # If we're not cleaning outliers, just return the BUR
                else:
                    return bur_val
            # If we don't have a match in the first place, return NaN
            else:
                return np.nan

        # If we're using Pandas, convert our onsets array to numpy for processing
        if isinstance(my_onsets, pd.Series):
            my_onsets = my_onsets.to_numpy()
        # Iterate through consecutive pairs of beats and get the BUR
        burs = [bur(i1, i2) for i1, i2 in zip(my_beats, my_beats[1:])]
        # We can't know the BUR for the final beat, so append NaN
        burs.append(np.nan)
        return pd.DataFrame({'beat': pd.to_datetime(my_beats, unit='s'), 'burs': burs})


class TempoSlope(BaseExtractor):
    """Extract features related to tempo slope, i.e. instantaneous tempo change (in beats-per-minute) per second"""
    def __init__(self, my_beats: pd.Series):
        super().__init__()
        my_bpms = 60 / my_beats.diff()
        self.model = self.extract_tempo_slope(my_beats, my_bpms)
        self.update_summary_dict([], [])

    @staticmethod
    def extract_tempo_slope(my_beats: np.array, my_bpms: np.array) -> RegressionResultsWrapper | None:
        """Create the tempo slope regression model"""
        # Dependent variable: the BPM measurements
        y = my_bpms
        # Predictor variable: the onset time (with an added intercept
        x = sm.add_constant(my_beats)
        # Fit the model and return
        try:
            return sm.OLS(y, x, missing='drop').fit()
        # These are all the different error types that can emerge when fitting to data with too many NaNs
        except (ValueError, IndexError, KeyError):
            return None

    def update_summary_dict(self, array_names, arrays, *args, **kwargs) -> None:
        """Update the summary dictionary with tempo slope and drift coefficients"""
        self.summary_dict.update({
            f'tempo_slope': self.model.params.iloc[1] if self.model is not None else np.nan,
            f'tempo_drift': self.model.bse.iloc[1] if self.model is not None else np.nan
        })


class Asynchrony(BaseExtractor):
    """Extracts various features relating to asynchrony of onsets.

    Many of these features rely on the definitions established in the `onsetsync` package (Eerola & Clayton, 2023),
    and are ported to Python here with minimal changes.

    """
    # TODO: implement some sort of way of calculating circular statistics here?

    def __init__(self, my_beats: pd.Series, their_beats: pd.DataFrame | pd.Series):
        super().__init__()
        # For many summary functions, we just need the asynchrony columns themselves
        self.summary_funcs.update(dict(
            pairwise_asynchronization=self.pairwise_asynchronization,
            groupwise_asynchronization=self.groupwise_asynchronization,
            mean_absolute_asynchrony=self.mean_absolute_asynchrony,
            mean_pairwise_asynchrony=self.mean_pairwise_asynchrony
        ))
        self.extract_asynchronies(my_beats, their_beats)
        # We calculate mean relative asynchrony slightly differently to other variables
        mra = self.mean_relative_asynchrony(my_beats, their_beats)
        self.summary_dict.update({'mean_relative_asynchrony': mra})

    @staticmethod
    def pairwise_asynchronization(asynchronies: pd.Series) -> float:
        """Extract the standard deviation of the asynchronies of a pair of instruments.

        Eerola & Clayton (2023) use the sample standard deviation rather than the population standard deviation, so we
        are required to set the correction term `ddof` in `np.nanstd` to 1 to correct this.

        Parameters:
            asynchronies (np.array): the onset time differences between two instruments

        Returns:
            float

        """
        return np.nanstd(asynchronies.dropna(), ddof=1)

    @staticmethod
    def groupwise_asynchronization(asynchronies: pd.Series) -> float:
        """Extract the root-mean-square (RMS) of the pairwise asynchronizations."""
        # Convert to a list for use in functools
        asynchronies_ = asynchronies.dropna().to_list()
        # We define the function (d^i/n)^2 here, for asynchrony d at the ith time point, with n total asynchrony values
        func = lambda total, asy: total + ((asy / len(asynchronies_)) ** 2)
        # Calculate all function values, then take the square root
        return np.sqrt(reduce(func, [0] + asynchronies_))

    @staticmethod
    def mean_absolute_asynchrony(asynchronies: pd.Series) -> float:
        """Extract the mean of all unsigned asynchrony values."""
        return asynchronies.dropna().abs().mean()
        # Alternative, should lead to identical results
        # return (1 / len(asynchronies)) * sum([abs(a) for a in asynchronies])

    @staticmethod
    def mean_pairwise_asynchrony(asynchronies: pd.Series) -> float:
        """Extract the mean of all signed asynchrony values."""
        return asynchronies.dropna().mean()
        # Alternative, should lead to identical results
        # return (1 / len(asynchronies)) * sum(asynchronies)

    def mean_relative_asynchrony(self, my_beats, their_beats: pd.Series | pd.DataFrame) -> float:
        """Extract the mean position of an instrument's onsets relative to the average position of the group"""
        # TODO: if one instrument is all NaN, this seems to result in a NaN result even if the other instrument is there
        # Get the average position of the whole group
        average_group_position = pd.concat([my_beats, their_beats], axis=1).dropna().mean(axis=1)
        # Get the relative position in comparison to the average: my_onset - average_onsets
        my_relative_asynchrony = my_beats - average_group_position
        # Return the mean asynchrony from our relative asynchrony
        return self.mean_pairwise_asynchrony(my_relative_asynchrony)

    def extract_asynchronies(self, my_beats: pd.Series, their_beats: pd.DataFrame | pd.Series) -> dict:
        """Extract asynchrony between an instrument of interest and all other instruments and calculate functions"""
        if isinstance(their_beats, pd.Series):
            their_beats = pd.DataFrame
        # Iterate through all instruments in the ensemble
        for partner_instrument in utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys():
            # We can't have asynchrony to our own performance, so append NaN in these cases
            if partner_instrument == my_beats.name:
                for func_k, func_v in self.summary_funcs.items():
                    self.summary_dict[f'{partner_instrument}_async_{func_k}'] = np.nan
            # Otherwise, calculate the asynchrony to this instrument
            else:
                partner_beats = their_beats[partner_instrument]
                # Calculate asynchrony: my_onset - partner_onset, then drop NaN values
                asynchronies = my_beats - partner_beats
                # Update our summary dictionary
                for func_k, func_v in self.summary_funcs.items():
                    self.summary_dict[f'{partner_instrument}_async_{func_k}'] = func_v(asynchronies)


class PhaseCorrection(BaseExtractor):
    """Extract various features related to phase correction

    Args:
        my_beats (pd.Series): onsets of instrument to model
        their_beats (pd.DataFrame | pd.Series, optional): onsets of other instrument(s), defaults to None
        order (int, optional): the order of the model to create, defaults to 1 (i.e. 1st-order model, no lagged terms)
        iqr_filter (bool, optional): whether to apply an iqr filter to data, defaults to False
        difference_iois (bool, optional): whether to take the first difference of IOI values, defaults to True

    """
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    def __init__(
            self,
            my_beats: pd.Series,
            their_beats: pd.DataFrame | pd.Series = None,
            order: int = 1,
            **kwargs,
    ):
        super().__init__()
        self.order = order
        self.iqr_filter = kwargs.get('iqr_filter', False)
        self.difference_iois = kwargs.get('difference_iois', True)
        self.standardize = kwargs.get('standardize', False)
        # Threshold dataframe based on provided low and high threshold
        self.low_threshold, self.high_threshold = kwargs.get('low_threshold', None), kwargs.get('high_threshold', None)
        # Create an empty variable to hold the data actually going into the model
        self.model_data = None
        # Create the model
        self.model = self.generate_model(my_beats, their_beats)
        # Create the dataframe and model summary information
        self.df = pd.DataFrame(self.extract_model_coefficients())
        # TODO: why are we subsetting here?
        self.summary_dict = self.df.to_dict(orient='records')[0]

    def truncate(self, my_beats, their_beats) -> tuple:
        """Truncates our input data between given low and high thresholds"""
        # If we haven't set a lower and upper threshold, don't threshold the data
        if self.low_threshold is None and self.high_threshold is None:
            return my_beats, their_beats
        threshold = self.truncate_df(
            pd.concat([my_beats, their_beats], axis=1),
            col=my_beats.name,
            # If we haven't provided a low or high threshold, we want to use all the data
            low=self.low_threshold if self.low_threshold is not None else my_beats.min(numeric_only=True),
            high=self.high_threshold if self.high_threshold is not None else my_beats.max(numeric_only=True)
        )
        # Apply the threshold to `my_beats`
        my_beats = threshold[my_beats.name]
        # If we haven't passed in any data as `their_beats`, then break here
        if their_beats is None:
            return my_beats, their_beats
        # Otherwise, go ahead and threshold every column in `their_beats`
        their_beats = threshold[their_beats.columns if isinstance(their_beats, pd.DataFrame) else their_beats.name]
        return my_beats, their_beats

    def format_async_arrays(self, their_beats: pd.Series | pd.DataFrame | None, my_beats: pd.Series) -> pd.DataFrame:
        """Format our asynchrony columns"""
        # If we haven't specified any asynchrony terms, i.e. we want a restricted model
        if their_beats is None:
            return pd.DataFrame([], [])
        # If we've only passed in one asynchrony term as a series, convert it to a dataframe
        elif isinstance(their_beats, pd.Series):
            their_beats = pd.DataFrame(their_beats)
        results = []
        for partner_instrument in their_beats.columns:
            partner_onsets = their_beats[partner_instrument]
            # In the phase correction model, the asynchrony terms are our partner's asynchrony with relation to us,
            # i.e. their_onset - my_onset (from their perspective, this is my_onset - their_onset).
            # Normally we would calculate asynchrony instead as my_onset - their_onset.
            asynchronies = partner_onsets - my_beats
            # Format our asynchrony array by adding IQR filter, etc.; we don't want to difference them, though
            asynchronies_fmt = self.format_array(asynchronies, difference_iois=False)
            asynchronies_fmt.name = f'{my_beats.name}_{partner_instrument}_asynchrony'
            # Shift our formatted asynchronies variable by the correct amount and extend the list
            results.extend(list(self.shifter(asynchronies_fmt)))
        return pd.concat(results, axis=1)

    def format_array(
            self,
            arr: np.array,
            iqr_filter: bool = None,
            difference_iois: bool = None,
            standardize: bool = None
    ) -> pd.Series:
        """Applies formatting to a single array used in creating the model"""
        # Use the default settings, if we haven't overridden them
        if difference_iois is None:
            difference_iois = self.difference_iois
        if iqr_filter is None:
            iqr_filter = self.iqr_filter
        if standardize is None:
            standardize = self.standardize
        if arr is None:
            return
        # Save the name of the array here
        name = arr.name
        # Apply differencing to the column (only for inter-onset intervals)
        if difference_iois:
            arr = arr.diff()
        # Apply the IQR filter, preserving the position of NaN values
        if iqr_filter:
            arr = pd.Series(utils.iqr_filter(arr, fill_nans=True))
        # Convert the score to standardized values (Z-score)
        if standardize:
            arr = stats.zscore(arr)
        # Restore the name of the array and return
        arr.name = name
        return arr

    def shifter(self, arr: np.array) -> Generator:
        """Shift an input array by the required number of beats and return a generator"""
        for i in range(self.order):
            pi = arr.shift(i)
            # Update the name of the array
            pi.name = f"{str(arr.name)}_lag{i}"
            yield pi

    def generate_model(
            self,
            my_beats: pd.Series,
            their_beats: pd.DataFrame | pd.Series | None
    ) -> RegressionResultsWrapper:
        """Generate the phase correction linear regression model"""
        # Truncate incoming data based on set thresholds
        my_beats, their_beats = self.truncate(my_beats, their_beats)
        # Get my previous inter-onset intervals from my onsets and format
        my_prev_iois = my_beats.diff()
        my_prev_iois = self.format_array(my_prev_iois)
        my_prev_iois.name = f'{my_beats.name}_prev_ioi'
        # Get my next inter-onset intervals by shifting my previous intervals (dependent variable)
        y = my_prev_iois.shift(-1)
        y.name = f'{my_beats.name}_next_ioi'
        # Get array of 'previous' inter-onset intervals (independent variable #1)
        my_prev_iois = pd.concat(list(self.shifter(my_prev_iois)), axis=1)
        # Get arrays of asynchrony values (independent variables #2, #3)
        async_arrs = self.format_async_arrays(their_beats, my_beats)
        # Combine independent variables into one dataframe and add constant term
        x = pd.concat([my_prev_iois, async_arrs], axis=1)
        x = sm.add_constant(x)
        # Update our instance attribute here, so we can debug the data going into the model, if needed
        self.model_data = pd.concat([my_beats, their_beats, x, y], axis=1)
        # Fit the regression model and return
        try:
            return sm.OLS(y, x, missing='drop').fit()
        # These are all the different error types that can emerge when fitting to data with too many NaNs
        except (ValueError, KeyError, IndexError):
            return None

    def extract_model_coefficients(self) -> Generator:
        """Extracts coefficients from linear phase correction model and format them correctly"""
        def extract_endog_instrument() -> str:
            """Returns name of instrument used in dependent variable of the model"""
            ei = self.model.model.endog_names.split('_')[0].lower()
            # Check that the value we've returned is contained within our list of possible instruments
            assert ei in utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys()
            return ei

        def getter(st: str) -> float:
            """Tries to get a coupling value from a given string"""
            try:
                return model_params[st]
            except KeyError:
                return np.nan

        # These are all basic statsmodels attributes we can extract easily from the model
        attributes = ['nobs', 'rsquared', 'rsquared_adj', 'aic', 'bic', 'llf']
        # If the model did not compile, return a dictionary filled with NaNs for every variable
        if self.model is None:
            extra_vars = ['intercept', 'resid_std', 'resid_len', 'self_coupling']
            instrs = utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys()
            for lagterm in range(self.order):
                yield {
                    'phase_correction_order': self.order,
                    'phase_correction_lag': lagterm,
                    **{ke: np.nan for ke in attributes + extra_vars + [f'coupling_{i}' for i in instrs]}
                }
        # Otherwise, get basic model fit attributes from the model class
        else:
            # Extract the name of the endog instrument from our model
            endog_ins = extract_endog_instrument()
            # Convert our model parameters to a dictionary
            model_params = self.model.params.to_dict()
            # Iterate through every lag term
            for lagterm in range(self.order):
                yield {
                    'phase_correction_order': self.order,
                    'phase_correction_lag': lagterm,
                    'coupling_piano': getter(f'{endog_ins}_piano_asynchrony_lag{lagterm}'),
                    'coupling_bass': getter(f'{endog_ins}_bass_asynchrony_lag{lagterm}'),
                    'coupling_drums': getter(f'{endog_ins}_drums_asynchrony_lag{lagterm}'),
                    'self_coupling': model_params[f'{endog_ins}_prev_ioi_lag{lagterm}'],
                    'intercept': model_params['const'],
                    f'resid_std': np.std(self.model.resid),
                    f'resid_len': len(self.model.resid),
                    **{attribute: getattr(self.model, attribute) for attribute in attributes}
                }


class GrangerCausality(BaseExtractor):
    """Extracts various features related to Granger causality.

    Args:
        my_beats (pd.Series): onsets of instrument to model
        their_beats (pd.DataFrame | pd.Series): onsets of remaining instrument(s)
        order (int, optional): the order of the model to create, defaults to 1 (i.e. 1st-order model, no lagged terms)
        **kwargs: keyword arguments passed to `PhaseCorrectionExtractor`

    """

    def __init__(
            self,
            my_beats: pd.Series,
            their_beats: pd.DataFrame | pd.Series,
            order: int = 1,
            **kwargs
    ):
        super().__init__()
        self.order = order
        # Update the summary dictionary
        self.summary_dict = self.compute_granger_indexes(my_beats, their_beats, **kwargs)

    def compute_fisher_test(self, var_restricted: float, var_unrestricted: float, n: int) -> float:
        """Evaluate statistical significance of Granger test with Fisher test"""
        # Calculate degrees of freedom for the F-test
        df1 = self.order
        df2 = n - 2 * self.order
        # Calculate the F-statistic and associated p-value for the F-test
        f_statistic = ((var_restricted - var_unrestricted) / df1) / (var_unrestricted / df2)
        return float(1 - stats.f.cdf(f_statistic, df1, df2))

    def compute_granger_index(self, my_beats, their_beats, **kwargs) -> tuple[float, float]:
        """Compute the Granger index between a restricted (self) and unrestricted (joint) model"""
        # TODO: think about whether we want to compute GCI at every lag UP TO self.order and use smallest,
        #  or just at self.order (current)
        # Create the restricted (self) model: just the self-coupling term(s)
        restricted_model = PhaseCorrection(my_beats, order=self.order, **kwargs).model
        # Create the unrestricted (joint) model: the self-coupling and partner-coupling terms
        unrestricted_model = PhaseCorrection(my_beats, their_beats, order=self.order, **kwargs).model
        # In the case of either model breaking (i.e. if we have no values), return NaN for both GCI and p
        if restricted_model is None or unrestricted_model is None:
            return np.nan, np.nan
        # Otherwise, extract the variance from both models
        var_restricted = np.nanvar(restricted_model.resid)
        var_unrestricted = np.nanvar(unrestricted_model.resid)
        # Calculate the Granger-causality index: the log of the ratio between the variance of the model residuals
        gci = np.log(var_restricted / var_unrestricted)
        # Carry out the Fisher test and obtain a p-value
        p = self.compute_fisher_test(var_restricted, var_unrestricted, restricted_model.nobs)
        return gci, p

    def compute_granger_indexes(self, my_beats, their_beats: pd.DataFrame, **kwargs) -> dict:
        """Compute Granger indexes for given input array and all async arrays, i.e. for both possible leaders in trio"""
        di = {'granger_causality_order': self.order}
        for instrument in utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys():
            # We can't compute Granger causality in relation to our own performance, so we yield an empty dictionary
            if instrument == my_beats.name:
                di.update({
                    f'granger_causality_{my_beats.name}_i': np.nan,
                    f'granger_causality_{my_beats.name}_p': np.nan,
                })
            else:
                gci, p = self.compute_granger_index(my_beats, their_beats[instrument], **kwargs)
                di.update({
                    f'granger_causality_{instrument}_i': gci,
                    f'granger_causality_{instrument}_p': p,
                })
        return di


class PartialCorrelation(BaseExtractor):
    """Extracts various features related to partial correlation between inter-onset intervals and onset asynchrony.

    This class calculates the partial correlation between (differenced) inter-onset intervals by musician `X` and
    lagged asynchronies between `X` and musician `Y`, controlling for prior (differenced) inter-onset intervals by `X`,
    i.e. accounting for the possibility of autocorrelated beat durations by `X`; see [1].

    Args:
        my_beats (pd.Series): onsets of instrument to model
        their_beats (pd.DataFrame | pd.Series): onsets of remaining instrument(s)
        order (int, optional): number of lag terms to calculate, defaults to 1
        iqr_filter (bool, optional): apply an iqr filter to inter-onset intervals, defaults to False
        difference_iois (bool, optional): whether to detrend inter-onset intervals via differencing, defaults to True

    References:
        [1]: Cheston, H. (2022). ‘Turning the beat around’: Time, temporality, and participation in the jazz solo break.
            Proceedings of the Conference on Interdisciplinary Musicology 2022: Participation, Edinburgh, UK.

    """
    iqr_filter = False
    difference_iois = True

    def __init__(self, my_beats: pd.Series, their_beats: pd.DataFrame | pd.Series, order: int = 1, **kwargs):
        super().__init__()
        self.order = order
        self.summary_dict = self.extract_partial_correlations(my_beats, their_beats, **kwargs)

    @staticmethod
    def partial_correlation(x: pd.Series, y: pd.Series, z: pd.Series):
        """Calculates partial correlation between arrays X and Y, controlling for the effect of Z

        Args:
            x (pd.Series): dependent variable
            y (pd.Series): independent variable
            z (pd.Series): control variable

        Returns:
            float

        """
        xy = x.corr(y, method='pearson')
        xz = x.corr(z, method='pearson')
        yz = y.corr(z, method='pearson')
        return (xy - (xz * yz)) / np.sqrt((1 - xz ** 2) * (1 - yz ** 2))

    @staticmethod
    def pvalue(n: int, k: int, r: float) -> float:
        """Extracts p-value from degrees of freedom and regression coefficient"""
        dof = n - k - 2
        tval = r * np.sqrt(dof / (1 - r ** 2))
        return 2 * stats.t.sf(np.abs(tval), dof)

    def extract_partial_correlations(
            self,
            my_beats: pd.Series,
            their_beats: pd.DataFrame | pd.Series,
            **kwargs
    ) -> dict:
        """Extracts partial correlation between inter-onset intervals and onset asynchrony at required lags"""
        # Get our initial inter-onset interval values
        my_differenced_iois = my_beats.diff()
        # Apply any filtering and further differencing as required
        if kwargs.get('difference_iois', self.difference_iois):
            my_differenced_iois = my_differenced_iois.diff()
        if kwargs.get('iqr_filter', self.iqr_filter):
            my_differenced_iois = pd.Series(utils.iqr_filter(my_differenced_iois, fill_nans=True))
        # Get our next inter-onset intervals: this is what we'll be predicting
        my_next_iois = my_differenced_iois.shift(-1)
        di = {'partial_corr_order': self.order}
        # Iterate through all instruments played by the other instruments in our group
        for instrument in utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys():
            if instrument == my_beats.name:
                di.update({
                    f'partial_corr_{instrument}_r': np.nan,
                    f'partial_corr_{instrument}_p': np.nan,
                    f'partial_corr_{instrument}_n': np.nan,
                })
            else:
                # Get the asynchrony values between that instrument and ours
                my_asynchronies = their_beats[instrument] - my_beats
                # TODO: think about labelling of lag terms: is lag 0 really lag 0?
                # Shift our asynchronies and interval variables by the required lag term
                my_prev_asynchronies = my_asynchronies.shift(self.order)    # Independent variable
                my_prev_iois = my_differenced_iois.shift(self.order)    # Control variable
                # Construct the dataframe, drop NaN values, and set column titles
                df = pd.concat([my_next_iois, my_prev_asynchronies, my_prev_iois], axis=1).dropna()
                df.columns = ['my_next_iois', 'my_prev_asynchronies', 'my_prev_iois']
                # Create the partial correlation matrix and extract p-value
                # The results here should be identical to those given by the `pingouin.partial_corr` function
                pcorr = self.partial_correlation(x=df.my_next_iois, y=df.my_prev_asynchronies, z=df.my_prev_iois)
                pval = self.pvalue(df.shape[0], df.shape[1] - 2, pcorr)
                # Yield the results in a nice dictionary format
                di.update({
                    f'partial_corr_{instrument}_r': pcorr,
                    f'partial_corr_{instrument}_p': pval,
                    f'partial_corr_{instrument}_n': df.shape[0],
                })
        return di


class CrossCorrelation(BaseExtractor):
    """Extract features related to the cross-correlation of inter-onset intervals and onset asynchrony"""
    difference_iois = True
    iqr_filter = False

    def __init__(self, my_beats: pd.Series, their_beats: pd.DataFrame, order: int = 1, **kwargs):
        super().__init__()
        if not isinstance(their_beats, pd.DataFrame):
            their_beats = pd.DataFrame(their_beats)
        self.order = order
        self.summary_dict = self.extract_cross_correlations(my_beats, their_beats, **kwargs)

    def extract_cross_correlations(self, my_beats: pd.Series, their_beats: pd.DataFrame, **kwargs) -> dict:
        """Extract cross correlation coefficients at all lags up to `self.order`"""
        # Get inter-onset intervals from onsets and apply any additional filtering needed
        my_iois = my_beats.diff()
        if kwargs.get('difference_iois', self.difference_iois):
            my_iois = my_iois.diff()
        if kwargs.get('iqr_filter', self.iqr_filter):
            my_iois = pd.Series(utils.iqr_filter(my_iois, fill_nans=True))
        di = {'cross_corr_order': self.order}
        # Iterate through each instrument
        for instrument in utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys():
            # We can't have cross-correlation with ourselves
            # TODO: investigate whether we want to provide auto-correlation here
            if instrument == my_beats.name:
                di.update({
                    f'cross_corr_{instrument}_r': np.nan,
                    f'cross_corr_{instrument}_p': np.nan,
                    f'cross_corr_{instrument}_n': np.nan,
                })
            else:
                # Get the asynchronies between us and this instrument
                asynchronies = their_beats[instrument] - my_beats
                # Lag the asynchronies, concatenate with IOIs, and drop NaN values
                combined = pd.concat([my_iois, asynchronies.shift(self.order)], axis=1).dropna()
                combined.columns = ['iois', 'asynchronies']
                # If, after dropping NaN values, we have fewer than 2 values, we can't calculate r, so return NaN
                if len(combined) < 2:
                    r, p = np.nan, np.nan
                # Otherwise, compute the correlation and return the necessary statistics
                else:
                    r, p = stats.pearsonr(combined['iois'], combined['asynchronies'])
                di.update({
                    f'cross_corr_{instrument}_r': r,
                    f'cross_corr_{instrument}_p': p,
                    f'cross_corr_{instrument}_n': int(combined.shape[0]),
                })
        return di


class IOIComplexity(BaseExtractor):
    """Extracts features relating to the complexity and density of inter-onset intervals."""
    col_names = ['bar_range', 'lz77', 'n_onsets']
    fracs = [1, 1 / 2, 5 / 12, 3 / 8, 1 / 3, 1 / 4, 1 / 6, 1 / 8, 1 / 12, 0]
    alphabet = [list(string.ascii_lowercase)[i] for i in range(len(fracs))]

    def __init__(
            self,
            my_onsets: np.array,
            downbeats: np.array,
            tempo: float,
            time_signature: int,
            bar_period: int = 4,
    ):
        super().__init__()
        # Set attributes
        self.bar_period = bar_period
        self.quarter_note = 60 / tempo
        self.time_signature = time_signature
        # Extract event density
        self.binned_iois = pd.DataFrame(self.bin_iois(my_onsets, downbeats))
        self.complexity_df = pd.DataFrame(self.extract_complexity(self.binned_iois), columns=self.col_names)
        # Update our summary dictionary
        self.summary_dict['bar_period'] = bar_period
        self.summary_dict['window_count'] = len(self.complexity_df)
        self.summary_dict['ioi_count'] = len(np.diff(my_onsets))
        self.summary_dict.update(**self._get_summary_dict())

    def _get_summary_dict(self) -> dict:
        """Gets summary variables for this feature"""
        return utils.flatten_dict(self.complexity_df[['lz77', 'n_onsets']].agg(['mean', 'std']).to_dict())

    def _bin_ioi(self, ioi: float) -> float:
        """Bins an IOI as a proportion of a quarter note at the given time signature"""
        proportional_ioi = (ioi / self.quarter_note) / self.time_signature
        # If somehow the IOI is greater than one measure, return missing value
        if proportional_ioi > 1:
            return np.nan
        # Otherwise, return the nearest binned IOI
        else:
            return min(self.fracs, key=lambda x: abs(x - proportional_ioi))

    def bin_iois(self, my_onsets: np.array, downbeats: np.array) -> list:
        """Bins all IOIs within `my_onsets` according to the beats in `downbeats`"""
        # Iterate over
        for i in range(len(downbeats) - self.bar_period):
            first_bar = downbeats[i]
            last_bar = downbeats[i + self.bar_period]
            iois_bar = np.ediff1d(self.get_between(my_onsets, first_bar, last_bar))
            binned_iois = np.array([self._bin_ioi(i) for i in iois_bar])
            binned_iois_clean = binned_iois[~np.isnan(binned_iois)]
            for binned_ioi in binned_iois_clean:
                yield dict(
                    bar_range=f'{i + 1}_{i + self.bar_period + 1}',
                    binned_ioi=binned_ioi,
                    binned_ascii=self.alphabet[self.fracs.index(binned_ioi)]
                )

    @staticmethod
    def lz77_compress(data: np.array, window_size: int = 4096) -> list:
        """Runs the LZ77 compression algorithm over the input `data`, with given `window_size`"""
        compressed = []
        index = 0
        while index < len(data):
            best_offset = -1
            best_length = -1
            best_match = ''
            # Search for the longest match in the sliding window
            for length in range(1, min(len(data) - index, window_size)):
                substring = data[index:index + length]
                offset = data.rfind(substring, max(0, index - window_size), index)
                if offset != -1 and length > best_length:
                    best_offset = index - offset
                    best_length = length
                    best_match = substring
            if best_match:
                # Add the (offset, length, next_character) tuple to the compressed data
                compressed.append((best_offset, best_length, data[index + best_length]))
                index += best_length + 1
            else:
                # No match found, add a zero-offset tuple
                compressed.append((0, 0, data[index]))
                index += 1
        return compressed

    def extract_complexity(self, binned_iois: np.array) -> Generator:
        """Extracts complexity scores for all inter-onset intervals in `binned_iois`"""
        # If we don't have any inter-onset intervals, return empty lists
        if len(binned_iois) == 0:
            return [], [], [], []
        # Otherwise, calculate the LZ77 and density scores for each of our windows
        for idx, grp in binned_iois.groupby('bar_range', sort=False):
            # This converts all the ascii representations to a single string
            ascii_ = ''.join(grp['binned_ascii'].to_list())
            # lz77 compression
            compressed = self.lz77_compress(ascii_)
            yield idx, len(compressed), len(ascii_)


class ProportionalAsynchrony(BaseExtractor):
    """Extracts features relating to the proportional asynchrony between performers."""
    UPPER_BOUND = 1/16
    LOWER_BOUND = 1/32
    REF_INSTR = 'drums'

    def __init__(self, summary_df: pd.DataFrame, my_instr_name: str, metre_col: str = 'metre_manual'):
        super().__init__()
        self.metre_col = metre_col
        asy = pd.DataFrame(self._extract_proportional_durations(summary_df))
        self.asynchronies = self._format_async_df(asy)
        mean_async = self.asynchronies.groupby('instr')['asynchrony_adjusted_offset'].agg([np.nanmean, np.nanstd])
        async_count = len(self.asynchronies[self.asynchronies['instr'] == my_instr_name].dropna())
        self.summary_dict = {
            f'prop_async_count': len(self.asynchronies[self.asynchronies['instr'] == my_instr_name]),
            f'prop_async_count_nonzero': async_count,
            **self._extract_async_stats(mean_async, my_instr_name)
        }

    @staticmethod
    def _extract_async_stats(mean_async: np.array, my_instr_name: str) -> dict:
        """Extracts asynchrony stats from all pairwise combinations of instruments and returns a dictionary"""
        loc = lambda name, c: mean_async[c].loc[name]
        res = {}
        for col in mean_async.columns:
            for instr in utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys():
                if instr == my_instr_name:
                    res[f'{instr}_prop_async_{col}'] = np.nan
                else:
                    res[f'{instr}_prop_async_{col}'] = loc(my_instr_name, col) - loc(instr, col)
            # for i1, i2 in combinations(list(utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys()), 2):
            #     res[f'{i1}_{i2}_prop_async_{col}'] = loc(i1, col) - loc(i2, col)
        return res

    def _format_async_df(self, async_df: pd.DataFrame) -> pd.DataFrame:
        """Coerces asynchrony dataframe into correct format"""
        mean_reference = async_df[(async_df['instr'] == self.REF_INSTR) & (async_df['beat'] == 1)]['asynchrony'].mean()
        # Offset the asynchrony column so that drums average beat 1 is shifted to 0
        async_df['asynchrony_offset'] = async_df['asynchrony'] - mean_reference
        # Adjust the asynchrony values so that asynchrony is independent of beat location
        async_df['asynchrony_adjusted'] = (async_df['asynchrony'] / 360) - ((async_df['beat'] - 1) * 1/4)
        # Adjust the offset beat values
        async_df['asynchrony_adjusted_offset'] = (async_df['asynchrony_offset'] / 360) - ((async_df['beat'] - 1) * 1/4)
        return async_df

    def _extract_proportional_durations(self, summary_df: pd.DataFrame) -> Generator:
        """Extracts proportional beat values for all instruments"""
        idx = summary_df[summary_df[self.metre_col] == 1].index
        for downbeat1, downbeat2 in pairwise(idx):
            # Get all the beats marked between our two downbeats (beat 1 bar 1, beat 1 bar 2)
            bw = summary_df[(downbeat1 <= summary_df.index) & (summary_df.index < downbeat2)]
            sub = bw[utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys()]
            # Get the first downbeat of the first bar, and the last downbeat of the second
            first = summary_df[summary_df.index == downbeat1]['beats'].iloc[0]
            last = summary_df[summary_df.index == downbeat2]['beats'].iloc[0]
            # Scale our onsets to be proportional with our first and last values
            prop = (sub - first) / (last - first)
            # Drop values after 1/16th note or before 1/32nd note
            upper_bound = (((bw[self.metre_col] - 1) * 1/4) + self.UPPER_BOUND)
            lower_bound = ((bw[self.metre_col] - 1) * 1/4) - self.LOWER_BOUND
            # Set values below upper and lower bound to NaN
            for col in prop.columns:
                prop[col][(prop[col] < lower_bound) | (prop[col] > upper_bound)] = np.nan
            # Convert values to degrees
            prop *= 360
            prop = pd.concat([prop, bw[self.metre_col]], axis=1)
            # Iterate through all instruments
            for instr in utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys():
                for _, val in prop[[instr, self.metre_col]].iterrows():
                    yield dict(instr=instr, asynchrony=val[instr], beat=val[self.metre_col])
