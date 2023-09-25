#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility classes, functions, and variables used specifically in the analysis and feature extraction process"""
import json
import warnings

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
        self.df: pd.DataFrame = self.format_df(om.summary_dict)
        # self.recording_tempo_slope = self.extract_tempo_slope()
        self.models = {}
        self.summary_df = None

        self.features = {}

    def format_df(
            self,
            summary_dict: dict
    ) -> pd.DataFrame:
        """Formats the dictionary contained in OnsetMaker.summary_dict in a format that can be used for modelling.

        Formatting involves converting the dictionary to a dataframe and creating new columns. These columns detail the
        inter-onset intervals between successive crotchet beats, alongside asynchrony values at that beat. These columns
        are created for every instrument: thus, for piano, bass and drums, create these columns:

            - {ins}_prev_ioi: the inter-onset interval between the current and the previous onset
            - {ins}_next_ioi: the inter-onset interval between the current and the *next* onset
            - {ins}_{exog_ins}_asynchrony: the asynchrony between one instrument and another at the current onset

        Attributes:
            summary_dict (dict)

        Returns:
            pd.DataFrame: a dataframe containing the required columns

        """

        def formatter(
                endog_ins: str,
                exog_instrs: list[str],
        ) -> pd.DataFrame:
            """Helper function that creates the required columns for one instrument.

            Arguments:
                endog_ins (str): the name of an instrument to create columns for, e.g. "piano", "drums"
                exog_instrs (list[str]): a list of names of instruments in the ensemble that are *not* endog_ins

            Returns:
                pd.DataFrame

            """
            if self.interpolate:
                df[endog_ins] = self.interpolate_missing_onsets(df[endog_ins], instr=endog_ins)
            # Compile the inter-onset intervals
            dic = {
                f'{endog_ins}_onset': df[endog_ins],
                f'{endog_ins}_prev_ioi': df[endog_ins].diff(),
                f'{endog_ins}_next_ioi': df[endog_ins].diff().shift(-1),
                f'{endog_ins}_prev_bpm': 60 / df[endog_ins].diff(),
                f'{endog_ins}_next_bpm': 60 / df[endog_ins].diff().shift(-1),
            }
            # Update the dictionary with our asynchrony values
            dic.update({f'{endog_ins}_{exog}_asynchrony': df[exog] - df[endog_ins] for exog in exog_instrs})
            # Return the dictionary as a dataframe
            return pd.DataFrame(dic)

        # Convert our summary dictionary to a dataframe
        df = pd.DataFrame(summary_dict).rename(columns={'beats': 'mix_onset'})
        conc = pd.concat([df, *[formatter(in_, [i for i in self.instrs if i != in_]) for in_ in self.instrs]], axis=1)
        conc['mix_bpm'] = 60 / conc['mix_onset'].diff()
        # Calculate average statistics, across all instruments (i.e. average IOI, average onset time)
        for col in ['onset', 'prev_ioi', 'next_ioi', 'prev_bpm', 'next_bpm']:
            conc[f'avg_{col}'] = conc[[f'{i}_{col}' for i in self.instrs]].mean(axis=1)
        return conc

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
            md: RegressionResultsWrapper,
    ) -> dict:
        """Creates summary dictionary for a single instrument, containing all extracted features.

        Arguments:
            endog_ins (str): the name of the instrument to create the dictionary for
            md (RegressionResultsWrapper): the statsmodels regression instance

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
            'recording_tempo_slope': self.recording_tempo_slope,
            # Raw beats
            'raw_beats': self.df[endog_ins],
            'interpolated_beats': self.num_interpolated[endog_ins],
            'observed_beats': self.df[endog_ins].notna().sum(),
            'missing_beats': self.df[endog_ins].isna().sum(),
            # Performance summary statistics
            'ioi_mean': self.df[f'{endog_ins}_prev_ioi'].mean(),
            'ioi_median': self.df[f'{endog_ins}_prev_ioi'].median(),
            'ioi_std': self.df[f'{endog_ins}_prev_ioi'].std(),
            # Cleaning metadata, e.g. missing beats
            'fraction_silent': self.om.silent_perc[endog_ins],
            'missing_beats_fraction': self.df[endog_ins].isna().sum() / self.df.shape[0],
            'total_beats': self.df.shape[0],
            'model_compiled': md is not None,
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
    # These are the default functions we'll call on any array to populate our summary stats dictionary for this feature
    summary_funcs = dict(
        mean=np.nanmean,
        median=np.nanmedian,
        std=np.nanstd,
        var=np.nanvar,
        quantile25=lambda x: np.nanquantile(x, 0.25),
        quantile75=lambda x: np.nanquantile(x, 0.75),
        count=len
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
    def get_onsets_between_vals(arr, i1, i2) -> np.array:
        """From an array `arr`, get all onsets between an upper and lower bound `i1` and `i2` respectively"""
        return arr[np.where(np.logical_and(arr >= i1, arr <= i2))]

class IOISummaryStatsExtractor(BaseExtractor):
    """Extracts various baseline summary statistics from an array of IOIs"""
    def __init__(self, iois: np.array):
        super().__init__()
        self.update_summary_dict('ioi', iois)


class EventDensityExtractor(BaseExtractor):
    """Extract various features related to event density, on both a per-bar and per-second basis"""
    def __init__(self, onsets: np.array, downbeats: np.array):
        super().__init__()
        self.per_second = self.extract_ed_per_second(onsets)
        self.per_bar = self.extract_ed_per_bar(onsets, downbeats)
        # Update our summary dictionary
        self.update_summary_dict(['ed_per_second', 'ed_per_bar'], [self.per_second['density'], self.per_bar['density']])

    @staticmethod
    def extract_ed_per_second(onsets) -> pd.DataFrame:
        """For every second in a performance, extract the number of notes played"""
        return (
            pd.DataFrame({'ts': pd.to_datetime(onsets, unit='s'), 'density': onsets})
            .set_index('ts')
            .resample('1s', label='left')
            .count()
            .reset_index(drop=False)
        )

    def extract_ed_per_bar(self, onsets, downbeats) -> pd.DataFrame:
        """For every complete bar in a performance, extract the number of notes"""
        matches = [len(self.get_onsets_between_vals(onsets, i1, i2)) for i1, i2 in zip(downbeats, downbeats[1:])]
        matches.append(np.nan)
        return pd.DataFrame({'downbeat': pd.to_datetime(downbeats, unit='s'), 'density': matches})


class BeatUpbeatRatioExtractor(BaseExtractor):
    """Extract various features related to beat-upbeat ratios (BURs)"""

    def __init__(self, onsets, beats):
        super().__init__()
        # Extract our burs here, so we can access them as instance properties
        self.bur = self.extract_burs(onsets, beats, use_log_burs=False)
        self.bur_log = self.extract_burs(onsets, beats, use_log_burs=True)
        # Update our summary dictionary
        self.update_summary_dict(['bur', 'bur_log'], [self.bur['burs'], self.bur_log['burs']])

    def extract_burs(
            self,
            onsets: np.array,
            beats: np.array,
            use_log_burs: bool = False
    ) -> pd.DataFrame:
        """Extracts beat-upbeat ratio (BUR) values from an array of onsets.

        The beat-upbeat ratio is introduced in [1] as a concept for analyzing the individual amount of 'swing' in two
        consecutive eighth note beat durations. It is calculated simply by dividing the duration of the first, 'long'
        eighth note beat by the second, 'short' beat. A BUR value of 2 indicates 'perfect' swing, i.e. a triplet quarter
        note followed by a triplet eighth note, while a BUR of 1 indicates 'even' eighth note durations.

        Arguments:
            onsets (np.array, optional): the array of raw onsets.
            beats (np.array, optional): the array of crotchet beat positions.
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
            match = self.get_onsets_between_vals(onsets, a, b)
            # If we have a group of three notes (i.e. three quavers), including a and b
            if len(match) == 3:
                # Then return the BUR (otherwise, we return None, which is converted to NaN)
                return func((match[1] - match[0]) / (match[2] - match[1]))

        # Iterate through consecutive pairs of beats and get the BUR
        burs = [bur(i1, i2) for i1, i2 in zip(beats, beats[1:])]
        # We can't know the BUR for the final beat, so append None
        burs.append(None)
        return pd.DataFrame({'beat': pd.to_datetime(beats, unit='s'), 'burs': burs})


class TempoSlopeExtractor(BaseExtractor):
    """Extract features related to tempo slope, i.e. instantaneous tempo change (in beats-per-minute) per second"""
    def __init__(self, onset_arr, bpm_arr):
        super().__init__()
        self.model = self.extract_tempo_slope(onset_arr, bpm_arr)
        self.update_summary_dict([], [])

    @staticmethod
    def extract_tempo_slope(onset_arr: np.array, bpm_arr: np.array) -> RegressionResultsWrapper:
        """Create the tempo slope regression model"""
        # Dependent variable: the BPM measurements
        y = bpm_arr
        # Predictor variable: the onset time (with an added intercept
        x = sm.add_constant(onset_arr)
        # Fit the model and return
        return sm.OLS(y, x, missing='drop').fit()

    def update_summary_dict(self, array_names, arrays):
        """Update the summary dictionary with tempo slope and drift coefficients"""
        self.summary_dict.update({
            f'tempo_slope': self.model.params[1],
            f'tempo_drift': self.model.bse[1]
        })


class PhaseCorrectionExtractor(BaseExtractor):
    """Extract various features related to phase correction at a given `order`"""
    iqr_filter = True
    difference_iois = True

    def __init__(self, ioi_arr: pd.Series, async_arrs: pd.Series | pd.DataFrame = None, order: int = 1):
        super().__init__()
        self.order = order
        self.model = self.generate_model(ioi_arr, async_arrs)
        self.update_summary_dict([], [])

    def format_array(self, arr: np.array, iqr_filter: bool = None, difference_iois: bool = None):
        """Applies formatting to a single array used in creating the model"""
        # Use the default settings, if we haven't overridden them
        # TODO: implement standardisation using z-scores here (see earlier git commits)
        if iqr_filter is None: iqr_filter = self.iqr_filter
        if difference_iois is None: difference_iois = self.difference_iois
        if arr is None: return
        # Save the name of the array here
        name = arr.name
        # Apply the IQR filter, preserving the position of NaN values
        if iqr_filter: arr = pd.Series(utils.iqr_filter(arr, fill_nans=True))
        # Apply differencing to the column (only for inter-onset intervals)
        if difference_iois: arr = arr.diff()
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

    def get_asynchrony_variables(self, async_arrs: pd.Series | pd.DataFrame) -> pd.DataFrame:
        # If we haven't specified any asynchrony terms, i.e. we want a restricted model
        if async_arrs is None: return pd.DataFrame([], [])
        # If we've only passed in one asynchrony term as a series, convert it to a dataframe
        elif isinstance(async_arrs, pd.Series): async_arrs = pd.DataFrame(async_arrs)

        arrays = []
        # For each of our asynchrony terms in our dataframe, format it and lag if needed
        for col in async_arrs.columns:
            async_var = self.format_array(async_arrs[col], difference_iois=False)
            arrays.extend(list(self.shifter(async_var)))
        # Combine all of our asynchrony terms into one dataframe
        return pd.concat(arrays, axis=1)

    def generate_model(self, ioi_arr: pd.Series, async_arrs: pd.DataFrame) -> RegressionResultsWrapper:
        """Generate the phase correction linear regression model"""
        ioi_arr_fmt = self.format_array(ioi_arr)
        # Get array of 'next' intervals, which we're trying to predict (dependent variable)
        y = ioi_arr_fmt.shift(-1)
        y.name = str(ioi_arr.name).replace('prev', 'next')  # Little trick to get the correct name here
        # Get array of 'previous' inter-onset intervals (independent variable #1)
        prev_iois = pd.concat(list(self.shifter(ioi_arr_fmt)), axis=1)
        # Get arrays of asynchrony values (independent variables #2, #3)
        async_vars = self.get_asynchrony_variables(async_arrs)
        # Combine independent variables into one dataframe and add constant
        x = pd.concat([prev_iois, async_vars], axis=1)
        x = sm.add_constant(x)
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
    def __init__(self, ioi_arr: pd.Series, async_arrs: pd.DataFrame, order: int):
        super().__init__()
        self.order = order
        self.grangers = self.compute_granger_indexes(ioi_arr, async_arrs)
        self.summary_dict.update(self.grangers)
        self.summary_dict['order'] = self.order

    def compute_fisher_test(self, var_restricted, var_unrestricted, n) -> float:
        """Evaluate statistical significance of Granger test with Fisher test"""
        # Calculate degrees of freedom for the F-test
        df1 = self.order
        df2 = n - 2 * self.order
        # Calculate the F-statistic and associated p-value for the F-test
        f_statistic = ((var_restricted - var_unrestricted) / df1) / (var_unrestricted / df2)
        return float(1 - stats.f.cdf(f_statistic, df1, df2))

    def compute_granger_index(self, ioi_arr, async_arr) -> tuple[float, float]:
        """Compute the Granger index between a restricted (self) and unrestricted (joint) model"""
        # TODO: think about whether we want to compute GCI at every lag UP TO self.order and use smallest,
        #  or just at self.order (current)
        # Create the restricted (self) model: just the self-coupling term(s)
        restricted_model = PhaseCorrectionExtractor(ioi_arr, order=self.order).model
        var_restricted = np.var(restricted_model.resid)
        # Create the unrestricted (joint) model: the self-coupling and partner-coupling terms
        unrestricted_model = PhaseCorrectionExtractor(ioi_arr, async_arr, order=self.order).model
        var_unrestricted = np.var(unrestricted_model.resid)
        # Calculate the Granger-causality index: the log of the ratio between the variance of the model residuals
        gci = np.log(var_restricted / var_unrestricted)
        # Carry out the Fisher test and obtain a P-value
        p = self.compute_fisher_test(var_restricted, var_unrestricted, restricted_model.nobs)
        return gci, p

    def compute_granger_indexes(self, ioi_arr: np.array, async_arrs: pd.DataFrame) -> dict:
        """Compute Granger indexes for given input array and all async arrays, i.e. for both possible leaders in trio"""
        results = {}
        for col in async_arrs.columns:
            gci, p = self.compute_granger_index(ioi_arr, async_arrs[col])
            results[f'{col}_gci'] = gci
            results[f'{col}_p'] = p
        return results

    def update_summary_dict(self, array_names, arrays):
        """Update summary dictionary with parameters taken from model"""
        self.summary_dict.update(self.grangers)
        self.summary_dict['order'] = self.order


class PeriodCorrectionExtractor(BaseExtractor):
    def __init__(self):
        super().__init__()


if __name__ == '__main__':
    pass