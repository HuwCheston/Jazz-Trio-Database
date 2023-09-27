#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility classes, functions, and variables used specifically in the analysis and feature extraction process"""
import json
import warnings
from functools import reduce

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResultsWrapper

from src import utils
from src.detect.detect_utils import OnsetMaker


class FeatureExtractor:
    instrs = list(utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys())

    def __init__(
            self,
            om: OnsetMaker,
            **kwargs
    ):
        self.om = om
        self.item = om.item
        self.interpolate: bool = kwargs.get('interpolate', True)
        self.num_interpolated: dict = {k: 0 for k in self.instrs}
        self.interpolation_limit: int = kwargs.get('interpolation_limit', 1)
        self.df = pd.DataFrame(om.summary_dict)
        if self.interpolate:
            for instr in utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys():
                self.df[instr] = self.interpolate_missing_onsets(self.df[instr], instr=instr)
        self.features = {}

    def interpolate_missing_onsets(
            self,
            onset_arr,
            instr: str = None,
            interpolation_limit: int = None
    ) -> np.array:
        """Interpolate between observed values to fill missing values in an array of onsets.

        If an onset is missing (set to `np.nan`), we can try to fill it in by looking for onsets that have been found
        on either side (i.e. before and after). We can interpolate up to a particular depth, given by
        interpolation_limit, to fill in a given number of consecutive missing values.

        Examples:
            >>> make = FeatureExtractor()
            >>> bea = np.array([0, np.nan, 1.0, 1.5, np.nan, np.nan, 3.0])
            >>> interp = make.interpolate_missing_onsets(bea, interpolation_limit=1)
            >>> print(interp)
            np.array([0 0.5 1.0 1.5 np.nan np.nan 3.0])

            >>> make = FeatureExtractor()
            >>> bea = np.array([0, np.nan, 1.0, 1.5, np.nan, np.nan, 3.0])
            >>> interp = make.interpolate_missing_onsets(bea, interpolation_limit=2)
            >>> print(interp)
            np.array([0 0.5 1.0 1.5 2.0 2.5 3.0])

        Arguments:
            onset_arr (np.array | pd.Series): the array of onsets to inteprolate
            instr (str, optional): the name of the instrument, used to update num_interpolated counter
            interpolation_limit (int, optional): the depth of missing notes to interpolate up to

        Returns:
            np.array: the onset array with missing onsets interpolated, up to a certain depth.

        """
        # If we haven't passed in an interpolation depth, use the default
        if interpolation_limit is None:
            interpolation_limit = self.interpolation_limit
        # If we've passed in a series, convert this to a numpy array
        if not isinstance(onset_arr, np.ndarray):
            onset_arr = onset_arr.to_numpy()

        # Slice the array to get consecutive missing values
        consecutive = lambda data: np.split(data, np.where(np.diff(data) != 1)[0] + 1)
        cons = consecutive(np.argwhere(np.isnan(onset_arr)).flatten())
        # Iterate through our slices of consecutive missing values
        for con in cons:
            if all([
                len(con) <= interpolation_limit,    # If the number of missing values is below our interpolation limit
                0 not in con,    # If one of the missing values isn't our first onset
                len(onset_arr) not in con    # If one of our missing values isn't our last onset
            ]):
                try:
                    # Get the onsets before and after our missing onsets
                    first, last = onset_arr[con[0] - 1], onset_arr[con[-1] + 1]
                except (IndexError, KeyError):
                    # Skip over missing onsets at the start and end of our array
                    pass
                else:
                    # Fill in the missing onsets with a linear space between the start and end of the array
                    onset_arr[con] = np.linspace(first, last, len(con) + 2)[1:-1]
                    # Increase the number of interpolated onsets
                    if instr is not None:
                        self.num_interpolated[instr] += len(con)
        # Return the interpolated array
        return onset_arr

    def create_instrument_dict(
            self,
            endog_ins: str,
            # md: RegressionResultsWrapper,
    ) -> dict:
        """Creates summary dictionary for a single instrument, containing all extracted features.

        Arguments:
            endog_ins (str): the name of the instrument to create the dictionary for

        Returns:
            dict: summary dictionary containing extracted features as key-value pairs

        """
        # TODO: this should combine all of summary dictionaries defined in all the BaseExtractor classes into a df

        return {
            # Item metadata
            **self.item,
            'tempo': self.om.tempo,
            'instrument': endog_ins,
            'performer': self.item['musicians'][utils.INSTRUMENTS_TO_PERFORMER_ROLES[endog_ins]],
            # 'recording_tempo_slope': self.recording_tempo_slope,
            # Raw beats
            'raw_beats': self.df[endog_ins],
            'interpolated_beats': self.num_interpolated[endog_ins],
            'observed_beats': self.df[endog_ins].notna().sum(),
            'missing_beats': self.df[endog_ins].isna().sum(),
            # Performance summary statistics
            # 'ioi_mean': self.df[f'{endog_ins}_prev_ioi'].mean(),
            # 'ioi_median': self.df[f'{endog_ins}_prev_ioi'].median(),
            # 'ioi_std': self.df[f'{endog_ins}_prev_ioi'].std(),
            # Cleaning metadata, e.g. missing beats
            'fraction_silent': self.om.silent_perc[endog_ins],
            'missing_beats_fraction': self.df[endog_ins].isna().sum() / self.df.shape[0],
            'total_beats': self.df.shape[0],
            # 'model_compiled': md is not None,
            # Event density functions
            # 'event_density': self.extract_event_density(endog_ins=endog_ins).mean(),
            # Tempo slopes
            # 'instrument_tempo_slope': self.extract_tempo_slope(f'{endog_ins}_onset', ),
            # 'instrument_tempo_drift': self.extract_tempo_slope(f'{endog_ins}_onset', drift=True),
            # Pairwise asynchrony
            # Grainger causality model
            # Kuramoto coupling model?
            # Model goodness-of-fit
            # **self._extract_model_goodness_of_fit(md=md),
            # Model coefficients
            # **self._extract_model_coefs(endog_ins=endog_ins, md=md),
            # Beat-upbeat ratio stats
            # **self.extract_bur_summary(endog_ins=endog_ins)
        }

    def extract_features(self):
        # TODO: This function should iterate through all instruments and populate the self.features dictionary with the
        #  summary_dict attribute inside the classes defined below that inherit from BaseExtractor
        pass


class BaseExtractor:
    """Base feature extraction class, with some methods that are useful for all classes"""
    # These are the default functions we'll call on any array to populate our summary statistics dictionary
    summary_funcs = dict(
        mean=np.nanmean,
        median=np.nanmedian,
        std=np.nanstd,
        var=np.nanvar,
        quantile25=lambda x: np.nanquantile(x, 0.25),
        quantile75=lambda x: np.nanquantile(x, 0.75),
        count=len,
        count_nonzero=lambda x: np.count_nonzero(~np.isnan(x))
    )

    def __init__(self):
        self.summary_dict = {}

    def __repr__(self) -> dict:
        """Overrides default string representation to print a dictionary of summary stats"""
        return json.dumps(self.summary_dict)

    def update_summary_dict(self, array_names, arrays) -> None:
        """Update our summary dictionary with values from this feature. Can be overridden!"""
        for name, df in zip(array_names, arrays):
            self.summary_dict.update({f'{name}_{func_k}': func_v(df) for func_k, func_v in self.summary_funcs.items()})

    @staticmethod
    def get_between(arr, i1, i2) -> np.array:
        """From an array `arr`, get all onsets between an upper and lower bound `i1` and `i2` respectively"""
        return arr[np.where(np.logical_and(arr >= i1, arr <= i2))]

    @staticmethod
    def truncate_series(arr: pd.Series, low: float, high: float, fill_nans: bool = False) -> pd.Series:
        """Truncate a series between a low and high threshold.

        Args:
            arr (pd.DataFrame): dataframe to truncate
            low (float): lower boundary for truncating
            high (float): upper boundary for truncating. Must be greater than `low`.
            fill_nans (bool, optional): whether to replace values outside `low` and `high` with `np.nan`

        Raises:
            AssertionError: if `high` < `low`

        Returns:
            pd.Series

        """
        assert low < high
        if fill_nans:
            return arr.mask(~arr.between(low, high))
        else:
            return arr[lambda x: (low <= x) & (x <= high)]

    @staticmethod
    def truncate_df(df: pd.DataFrame, col: str, low: float, high: float, fill_nans: bool = False) -> pd.DataFrame:
        """Truncate a dataframe between a low and high threshold.

        Args:
            df (pd.DataFrame): dataframe to truncate
            col (str): array to use when truncating
            low (float): lower boundary for truncating
            high (float): upper boundary for truncating. Must be greater than `low`.
            fill_nans (bool, optional): whether to replace values outside `low` and `high` with `np.nan`

        Raises:
            AssertionError: if `high` < `low`

        Returns:
            pd.DataFrame

        """
        assert low < high
        mask = (low <= df[col]) & (df[col] <= high)
        if fill_nans:
            return df.mask(~mask)
        else:
            return df[mask]


class IOISummaryStatsExtractor(BaseExtractor):
    """Extracts various baseline summary statistics from an array of IOIs"""
    # TODO: think about IOI entropy? Rolling IOI complexity (entropy every 4 bars)
    # TODO: compression algorithms for measuring IOI complexity?
    def __init__(self, my_onsets: pd.Series, iqr_filter: bool = False):
        super().__init__()
        iois = my_onsets.diff()
        if iqr_filter:
            iois = utils.iqr_filter(iois, fill_nans=True)
        self.summary_funcs['binary_entropy'] = self.binary_entropy
        self.summary_funcs['npvi'] = self.npvi
        self.update_summary_dict(['ioi'], [iois])

    @staticmethod
    def binary_entropy(iois: pd.Series) -> float:
        """Extract the Shannon entropy from an iterable"""
        # We convert our IOIs into milliseconds here to prevent floating point numbers
        ms_arr = (iois * 1000).dropna().astype(int).to_numpy()
        _, counts = np.unique(ms_arr, return_counts=True)
        probabilities = counts / len(ms_arr)
        return -np.sum(probabilities * np.log2(probabilities))
        # Alternative method using SciPy, should yield identical results
        # return stats.entropy((iois * 1000).dropna().astype(int).value_counts().squeeze(), base=2)

    @staticmethod
    def npvi(iois) -> float:
        """Extract the normalised pairwise variability index (nPVI) from an iterable"""
        dat = iois.dropna().to_numpy()
        return sum([abs((k - k1) / ((k + k1) / 2)) for (k, k1) in zip(dat, dat[1:])]) * 100 / (sum(1 for _ in dat) - 1)


class BPMSummaryStatsExtractor(BaseExtractor):
    """Extracts various baseline summary statistics from an array of BPMs (60 / IOI)"""
    def __init__(self, my_onsets: pd.Series, iqr_filter: bool = False):
        super().__init__()
        bpms = 60 / my_onsets.diff()
        if iqr_filter:
            bpms = utils.iqr_filter(bpms, fill_nans=True)
        self.update_summary_dict(['bpm'], [bpms])


class EventDensityExtractor(BaseExtractor):
    """Extract various features related to event density, on both a per-bar and per-second basis"""
    def __init__(self, my_onsets: pd.Series, quarter_note_downbeats: np.array):
        super().__init__()
        self.per_second = self.extract_ed_per_second(my_onsets)
        self.per_bar = self.extract_ed_per_bar(my_onsets, quarter_note_downbeats)
        # Update our summary dictionary
        self.update_summary_dict(['ed_per_second', 'ed_per_bar'], [self.per_second['density'], self.per_bar['density']])

    @staticmethod
    def extract_ed_per_second(my_onsets) -> pd.DataFrame:
        """For every second in a performance, extract the number of notes played"""
        return (
            pd.DataFrame({'ts': pd.to_datetime(my_onsets, unit='s'), 'density': my_onsets})
            .set_index('ts')
            .resample('1s', label='left')
            .count()
            .reset_index(drop=False)
        )

    def extract_ed_per_bar(self, my_onsets, quarter_note_downbeats) -> pd.DataFrame:
        """For every complete bar in a performance, extract the number of notes"""
        sequential_downbeats = zip(quarter_note_downbeats, quarter_note_downbeats[1:])
        my_onsets_arr = my_onsets.to_numpy()
        matches = [len(self.get_between(my_onsets_arr, i1, i2)) for i1, i2 in sequential_downbeats]
        matches.append(np.nan)
        return pd.DataFrame({'downbeat': pd.to_datetime(quarter_note_downbeats, unit='s'), 'density': matches})


class BeatUpbeatRatioExtractor(BaseExtractor):
    """Extract various features related to beat-upbeat ratios (BURs)"""

    def __init__(self, my_onsets, quarter_note_beats):
        super().__init__()
        # Extract our burs here, so we can access them as instance properties
        self.bur = self.extract_burs(my_onsets, quarter_note_beats, use_log_burs=False)
        self.bur_log = self.extract_burs(my_onsets, quarter_note_beats, use_log_burs=True)
        # Update our summary dictionary
        self.update_summary_dict(['bur', 'bur_log'], [self.bur['burs'], self.bur_log['burs']])

    def extract_burs(
            self,
            my_onsets: np.array,
            quarter_note_beats: np.array,
            use_log_burs: bool = False
    ) -> pd.DataFrame:
        """Extracts beat-upbeat ratio (BUR) values from an array of onsets.

        The beat-upbeat ratio is introduced in [1] as a concept for analyzing the individual amount of 'swing' in two
        consecutive eighth note beat durations. It is calculated simply by dividing the duration of the first, 'long'
        eighth note beat by the second, 'short' beat. A BUR value of 2 indicates 'perfect' swing, i.e. a triplet quarter
        note followed by a triplet eighth note, while a BUR of 1 indicates 'even' eighth note durations.

        Arguments:
            my_onsets (np.array, optional): the array of raw onsets.
            quarter_note_beats (np.array, optional): the array of crotchet beat positions.
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
                # Then return the BUR (otherwise, we return None, which is converted to NaN)
                return func((match[1] - match[0]) / (match[2] - match[1]))
            else:
                return np.nan

        # If we're using Pandas, convert our onsets array to numpy for processing
        if isinstance(my_onsets, pd.Series):
            my_onsets = my_onsets.to_numpy()
        # Iterate through consecutive pairs of beats and get the BUR
        burs = [bur(i1, i2) for i1, i2 in zip(quarter_note_beats, quarter_note_beats[1:])]
        # We can't know the BUR for the final beat, so append None
        burs.append(None)
        return pd.DataFrame({'beat': pd.to_datetime(quarter_note_beats, unit='s'), 'burs': burs})


class TempoSlopeExtractor(BaseExtractor):
    """Extract features related to tempo slope, i.e. instantaneous tempo change (in beats-per-minute) per second"""
    def __init__(self, my_onsets: pd.Series):
        super().__init__()
        bpms = 60 / my_onsets.diff()
        self.model = self.extract_tempo_slope(my_onsets, bpms)
        self.update_summary_dict([], [])

    @staticmethod
    def extract_tempo_slope(my_onsets: np.array, bpms: np.array) -> RegressionResultsWrapper:
        """Create the tempo slope regression model"""
        # Dependent variable: the BPM measurements
        y = bpms
        # Predictor variable: the onset time (with an added intercept
        x = sm.add_constant(my_onsets)
        # Fit the model and return
        return sm.OLS(y, x, missing='drop').fit()

    def update_summary_dict(self, array_names, arrays):
        """Update the summary dictionary with tempo slope and drift coefficients"""
        self.summary_dict.update({
            f'tempo_slope': self.model.params[1],
            f'tempo_drift': self.model.bse[1]
        })


class AsynchronyExtractor(BaseExtractor):
    """Extracts various features relating to asynchrony of onsets.

    Many of these features rely on the definitions established in the `onsetsync` package (Eerola & Clayton, 2023),
    and are ported to Python here with minimal changes.

    """
    # TODO: implement some sort of way of calculating circular statistics here?

    def __init__(self, my_onsets: pd.Series, their_onsets: pd.DataFrame | pd.Series):
        super().__init__()
        # Extract onset asynchronies with respect to the performance of my partners
        asynchronies = self.extract_asynchronies(my_onsets, their_onsets)
        # For many summary functions, we just need the asynchrony columns themselves
        self.summary_funcs.update(dict(
            pairwise_asynchronization=self.pairwise_asynchronization,
            groupwise_asynchronization=self.groupwise_asynchronization,
            mean_absolute_asynchrony=self.mean_absolute_asynchrony,
            mean_pairwise_asynchrony=self.mean_pairwise_asynchrony
        ))
        self.update_summary_dict(array_names=asynchronies.keys(), arrays=asynchronies.values())
        # For mean relative asynchrony, we need to create a new series
        # This is because we need to refer to the entire group, not just one partner
        mra = self.mean_relative_asynchrony(my_onsets, their_onsets)
        self.summary_dict.update({f'{my_onsets.name}_mean_relative_asynchrony': mra})

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

    def mean_relative_asynchrony(self, my_onsets, their_onsets: pd.Series | pd.DataFrame) -> float:
        """Extract the mean position of an instrument's onsets relative to the average position of the group"""
        # Get the average position of the whole group
        average_group_position = pd.concat([my_onsets, their_onsets], axis=1).dropna().mean(axis=1)
        # Get the relative position in comparison to the average: my_onset - average_onsets
        my_relative_asynchrony = my_onsets - average_group_position
        # Return the mean asynchrony from our relative asynchrony
        return self.mean_pairwise_asynchrony(my_relative_asynchrony)

    @staticmethod
    def extract_asynchronies(my_onsets: pd.Series, their_onsets: pd.DataFrame | pd.Series) -> dict:
        """Extract asynchrony between an instrument of interest and all other instruments, then return a dictionary"""
        if isinstance(their_onsets, pd.Series):
            their_onsets = pd.DataFrame
        results = {}
        for partner_instrument in their_onsets.columns:
            partner_onsets = their_onsets[partner_instrument]
            # Calculate asynchrony: my_onset - partner_onset, then drop NaN values
            asynchronies = my_onsets - partner_onsets
            asynchronies.name = f'{my_onsets.name}_{partner_instrument}'
            results[asynchronies.name] = asynchronies
        return results


class PhaseCorrectionExtractor(BaseExtractor):
    """Extract various features related to phase correction

    Args:
        my_onsets (pd.Series): onsets of instrument to model
        their_onsets (pd.DataFrame | pd.Series, optional): onsets of other instrument(s), defaults to None
        order (int, optional): the order of the model to create, defaults to 1 (i.e. 1st-order model, no lagged terms)
        iqr_filter (bool, optional): whether to apply an iqr filter to data, defaults to False
        difference_iois (bool, optional): whether to take the first difference of IOI values, defaults to True

    """

    def __init__(
            self,
            my_onsets: pd.Series,
            their_onsets: pd.DataFrame | pd.Series = None,
            order: int = 1,
            iqr_filter: bool = False,
            difference_iois: bool = True,
            low_threshold: float = None,
            high_threshold: float = None
    ):
        super().__init__()
        self.order = order
        self.iqr_filter = iqr_filter
        self.difference_iois = difference_iois
        # Threshold dataframe based on provided low and high threshold
        self.low_threshold, self.high_threshold = low_threshold, high_threshold
        # Create an empty variable to hold the data actually going into the model
        self.df = None
        # Create the model
        self.model = self.generate_model(my_onsets, their_onsets)
        self.update_summary_dict([], [])

    def truncate(self, my_onsets, their_onsets) -> tuple:
        """Truncates our input data between low and high thresholds, based on """
        threshold = self.truncate_df(
            self.df,
            col=my_onsets.name,
            # If we haven't provided a low or high threshold, we want to use all the data
            low=self.low_threshold if self.low_threshold is not None else my_onsets.min(),
            high=self.high_threshold if self.high_threshold is not None else my_onsets.max()
        )
        my_onsets = threshold[my_onsets.name]
        their_onsets = threshold[their_onsets.columns if isinstance(their_onsets, pd.DataFrame) else their_onsets.name]
        return my_onsets, their_onsets

    def format_async_arrays(self, their_onsets: pd.Series | pd.DataFrame | None, my_onsets: pd.Series) -> pd.DataFrame:
        """Format our asynchrony columns"""
        # If we haven't specified any asynchrony terms, i.e. we want a restricted model
        if their_onsets is None:
            return pd.DataFrame([], [])
        # If we've only passed in one asynchrony term as a series, convert it to a dataframe
        elif isinstance(their_onsets, pd.Series):
            their_onsets = pd.DataFrame(their_onsets)
        results = []
        for partner_instrument in their_onsets.columns:
            partner_onsets = their_onsets[partner_instrument]
            # In the phase correction model, the asynchrony terms are our partner's asynchrony with relation to us,
            # i.e. their_onset - my_onset (from their perspective, this is my_onset - their_onset).
            # Normally we would calculate asynchrony instead as my_onset - their_onset.
            asynchronies = partner_onsets - my_onsets
            # Format our asynchrony array by adding IQR filter, etc.; we don't want to difference them, though
            asynchronies_fmt = self.format_array(asynchronies, difference_iois=False)
            asynchronies_fmt.name = f'{my_onsets.name}_{partner_instrument}_asynchrony'
            # Shift our formatted asynchronies variable by the correct amount and extend the list
            results.extend(list(self.shifter(asynchronies_fmt)))
        return pd.concat(results, axis=1)

    def format_array(self, arr: np.array, iqr_filter: bool = None, difference_iois: bool = None):
        """Applies formatting to a single array used in creating the model"""
        # Use the default settings, if we haven't overridden them
        # TODO: implement standardisation using z-scores here (see earlier git commits)
        if iqr_filter is None:
            iqr_filter = self.iqr_filter
        if difference_iois is None:
            difference_iois = self.difference_iois
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
        # Restore the name of the array and return
        arr.name = name
        return arr

    def shifter(self, arr: np.array):
        """Shift an input array by the required number of beats and return a generator"""
        for i in range(self.order):
            pi = arr.shift(i)
            # Update the name of the array
            pi.name = f"{str(arr.name)}_lag{i}"
            yield pi

    def generate_model(
            self,
            my_onsets: pd.Series,
            their_onsets: pd.DataFrame | pd.Series | None
    ) -> RegressionResultsWrapper:
        """Generate the phase correction linear regression model"""
        # Truncate incoming data based on set thresholds
        my_onsets, their_onsets = self.truncate(my_onsets, their_onsets)
        # Get my previous inter-onset intervals from my onsets and format
        my_prev_iois = my_onsets.diff()
        my_prev_iois = self.format_array(my_prev_iois)
        my_prev_iois.name = f'{my_onsets.name}_prev_ioi'
        # Get my next inter-onset intervals by shifting my previous intervals (dependent variable)
        y = my_prev_iois.shift(-1)
        y.name = f'{my_onsets.name}_next_ioi'
        # Get array of 'previous' inter-onset intervals (independent variable #1)
        my_prev_iois = pd.concat(list(self.shifter(my_prev_iois)), axis=1)
        # Get arrays of asynchrony values (independent variables #2, #3)
        async_arrs = self.format_async_arrays(their_onsets, my_onsets)
        # Combine independent variables into one dataframe and add constant term
        x = pd.concat([my_prev_iois, async_arrs], axis=1)
        x = sm.add_constant(x)
        # Update our instance attribute here, so we can debug the data going into the model, if needed
        self.df = pd.concat([my_onsets, their_onsets, x, y], axis=1)
        # Fit the regression model and return
        try:
            return sm.OLS(y, x, missing='drop').fit()
        except (ValueError, IndexError):
            return None

    def extract_model_coefficients(self) -> dict:
        """Extracts coefficients from linear phase correction model and format them correctly"""

        def extract_endog_instrument() -> str:
            """Returns name of instrument used in dependent variable of the model"""
            ei = self.model.model.endog_names.split('_')[0].lower()
            # Check that the value we've returned is contained within our list of possible instruments
            assert ei in utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys()
            return ei

        def getter(exog_ins: str, order: int) -> dict:
            """Gets a coupling value between endog_ins and exog_ins with lag order"""
            for name, coef in self.model.params.to_dict().items():
                if f'{endog_ins}_{exog_ins}_asynchrony_lag{order}' in name:
                    return coef
            return np.nan

        endog_ins = extract_endog_instrument()
        # Start creating our results dictionary
        results = {'intercept': self.model.params.to_dict()['const']}
        # Iterate through every instrument
        for instr in utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys():
            # Iterate through every lag term
            for lagterm in range(self.order):
                # Append the coupling coefficient to our results dictionary
                results.update({f'coupling_{instr}_lag{lagterm}': getter(instr, lagterm)})
        return results

    def update_summary_dict(self, array_names, arrays):
        """Update summary dictionary with parameters taken from model"""
        if self.model is None:
            warnings.warn(f'model failed to compile !', UserWarning)
            return {}
        # Get basic model fit attributes
        attributes = ['nobs', 'rsquared', 'rsquared_adj', 'aic', 'bic', 'llf']
        for attribute in attributes:
            self.summary_dict.update({attribute: getattr(self.model, attribute)})
        # Get model fit attributes related to residuals
        self.summary_dict.update({
            'order': self.order,
            f'resid_std': np.std(self.model.resid),
            f'resid_len': len(self.model.resid),
        })
        # Get model coefficients
        self.summary_dict.update(**self.extract_model_coefficients())


class GrangerExtractor(BaseExtractor):
    """Extracts various features related to Granger causality.

    Args:
        my_onsets (pd.Series): onsets of instrument to model
        their_onsets (pd.DataFrame | pd.Series): onsets of remaining instrument(s)
        **kwargs: keyword arguments passed to `PhaseCorrectionExtractor`

    """

    def __init__(
            self,
            my_onsets: pd.Series,
            their_onsets: pd.DataFrame | pd.Series,
            **kwargs
    ):
        super().__init__()
        self.order = kwargs.get("order", 1)
        self.grangers = self.compute_granger_indexes(my_onsets, their_onsets, **kwargs)
        # Update the summary dictionary
        self.summary_dict.update(self.grangers)
        self.summary_dict.update(kwargs)

    def compute_fisher_test(self, var_restricted: float, var_unrestricted: float, n: int) -> float:
        """Evaluate statistical significance of Granger test with Fisher test"""
        # Calculate degrees of freedom for the F-test
        df1 = self.order
        df2 = n - 2 * self.order
        # Calculate the F-statistic and associated p-value for the F-test
        f_statistic = ((var_restricted - var_unrestricted) / df1) / (var_unrestricted / df2)
        return float(1 - stats.f.cdf(f_statistic, df1, df2))

    def compute_granger_index(self, my_onsets, their_onsets, **kwargs) -> tuple[float, float]:
        """Compute the Granger index between a restricted (self) and unrestricted (joint) model"""
        # TODO: think about whether we want to compute GCI at every lag UP TO self.order and use smallest,
        #  or just at self.order (current)
        # Create the restricted (self) model: just the self-coupling term(s)
        restricted_model = PhaseCorrectionExtractor(my_onsets, **kwargs).model
        # Create the unrestricted (joint) model: the self-coupling and partner-coupling terms
        unrestricted_model = PhaseCorrectionExtractor(my_onsets, their_onsets, **kwargs).model
        # In the case of either model breaking (i.e. if we have no values), return NaN for both GCI and p
        if restricted_model is None or unrestricted_model is None:
            return np.nan, np.nan
        # Extract the variance from both models
        var_restricted = np.var(restricted_model.resid)
        var_unrestricted = np.var(unrestricted_model.resid)
        # Calculate the Granger-causality index: the log of the ratio between the variance of the model residuals
        gci = np.log(var_restricted / var_unrestricted)
        # Carry out the Fisher test and obtain a p-value
        p = self.compute_fisher_test(var_restricted, var_unrestricted, restricted_model.nobs)
        return gci, p

    def compute_granger_indexes(self, my_onsets, their_onsets: pd.DataFrame, **kwargs) -> dict:
        """Compute Granger indexes for given input array and all async arrays, i.e. for both possible leaders in trio"""
        results = {}
        for col in their_onsets.columns:
            gci, p = self.compute_granger_index(my_onsets, their_onsets[col], **kwargs)
            results[f'{col}_gci'] = gci
            results[f'{col}_p'] = p
        return results


class CrossCorrelationExtractor(BaseExtractor):
    pass


class PeriodCorrectionExtractor(BaseExtractor):
    pass


class KuramotoExtractor(BaseExtractor):
    pass


if __name__ == '__main__':
    pass
