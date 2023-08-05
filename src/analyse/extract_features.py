#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Extracts the required features for each item in the corpus, using the automatically detected onsets.
"""

import logging
import warnings
from math import isnan

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.formula.api as smf
from statsmodels.regression.linear_model import RegressionResultsWrapper
from tqdm import tqdm

import src.utils.analyse_utils as autils
from src.analyse.detect_onsets import OnsetMaker


class ModelMaker:
    sample_rate = autils.SAMPLE_RATE
    instrs = list(autils.INSTRS_TO_PERF.keys())

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
        self.df: pd.DataFrame = self._format_df(om.summary_dict)
        self.models = {}
        self.bayes_models = {}
        self.summary_df = None

    def _format_df(
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
                f'{endog_ins}_prev_ioi': df[endog_ins].diff(),
                f'{endog_ins}_next_ioi': df[endog_ins].diff().shift(-1),
            }
            # Update the dictionary with our asynchrony values
            dic.update({f'{endog_ins}_{exog}_asynchrony': df[exog] - df[endog_ins] for exog in exog_instrs})
            # Return the dictionary as a dataframe
            return pd.DataFrame(dic)

        # Convert our summary dictionary to a dataframe
        df = pd.DataFrame(summary_dict)
        return pd.concat([df, *[formatter(in_, [i for i in self.instrs if i != in_]) for in_ in self.instrs]], axis=1)

    def interpolate_missing_onsets(
            self,
            onset_arr,
            instr: str = None,
            interpolation_limit: int = None
    ) -> np.array:
        """Interpolate between observed values to fill missing values in an array of onsets.

        If an onset is missing (set to np.nan), we can try to fill it in by looking for onsets that have been found
        on either side (i.e. before and after). We can interpolate up to a particular depth, given by
        interpolation_limit, to fill in a given number of consecutive missing values.

        Examples:
            >>> make = ModelMaker()
            >>> bea = np.array([0, np.nan, 1.0, 1.5, np.nan, np.nan, 3.0])
            >>> interp = make.interpolate_missing_onsets(bea, interpolation_limit=1)
            >>> print(interp)
            np.array([0 0.5 1.0 1.5 np.nan np.nan 3.0])

            >>> make = ModelMaker()
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

    def generate_model(
            self,
            endog_ins: str,
            standardise: bool = False,
            iqr_clean: bool = True,
            difference_ioi: bool = True,
    ) -> RegressionResultsWrapper | None:
        """Generates the phase correction model for one instrument.

        Arguments:
            endog_ins (str): the name of the 'dependent variable' instrument, whose IOIs we are predicting
            standardise (bool, optional): use standard ('z') scores when computing regression, defaults to False
            iqr_clean (bool, optional): run an IQR filter across all data before modelling, defaults to True
            difference_ioi (bool, optional): take the first difference of IOI data before modelling, defaults to True

        Returns:
            RegressionResultsWrapper: the fitted regression model

        """
        exog = [ex for ex in self.instrs if ex != endog_ins]
        # Create our asynchrony (coupling) terms in the model
        async_cols = [f'{endog_ins}_{instr}_asynchrony' for instr in exog]
        # Create the rest of our model
        md = f'{endog_ins}_next_ioi~{endog_ins}_prev_ioi+' + '+'.join(async_cols)
        df = self.df.copy(deep=True)
        # If we're standardising, convert our columns to Z-scores
        if standardise:
            try:
                df = (
                    df.select_dtypes(include=[np.number])
                      .dropna()
                      .apply(stats.zscore)
                )
            except ValueError:
                return None
        # If we're cleaning our columns to remove values +/- 1.5 * IQR below upper/lower bounds
        if iqr_clean:
            for col in [*async_cols, f"{endog_ins}_prev_ioi", f'{endog_ins}_next_ioi']:
                df[col] = autils.iqr_filter(df[col], fill_nans=True)
        # If we're using the first difference of our inter-onset interval columns
        if difference_ioi:
            for col in [f"{endog_ins}_prev_ioi", f'{endog_ins}_next_ioi']:
                df[col] = df[col].diff()
        # Create the regression model, fit to the data, and return
        try:
            return smf.ols(md, data=df, missing='drop').fit()
        except (ValueError, IndexError):
            return None

    def generate_bayes_model(
            self,
            endog_ins: str
    ):
        """Generates the Bayesian phase correction model for one instrument.

        Arguments:
            endog_ins (str): the name of the 'dependent variable' instrument

        Returns:
            az.InferenceData: the ArviZ InferenceData instance

        """
        # Import bambi here so we don't waste time doing this if we're not generating Bayesian models
        import bambi as bmb
        exog = [ex for ex in self.instrs if ex != endog_ins]
        # Create our asynchrony (coupling) terms in the model
        async_cols = [f'{endog_ins}_{instr}_asynchrony' for instr in exog]
        # Create the rest of our model
        form = f'{endog_ins}_next_ioi~{endog_ins}_prev_ioi+' + '+'.join(async_cols)
        df = self.df.copy(deep=True)
        # Create the dictionary of priors
        priors = {
            'Intercept': bmb.Prior("Normal", mu=0.003, sigma=0.003),
            f'{endog_ins}_prev_ioi': bmb.Prior("Normal", mu=-0.55, sigma=0.06),
            **{col: bmb.Prior("Uniform", lower=0, upper=1) for col in async_cols}
        }
        # Create the model, fit, and return the InferenceData object
        md = bmb.Model(form, df, dropna=True, priors=priors)
        res = md.fit(cores=1)
        return res

    def _extract_bayes_coefs(
            self,
            endog_ins: str,
            bayes_md=None,
            coupling_var: str = 'asynchrony',
            ioi_var: str = 'prev_ioi'
    ) -> dict:
        """Extracts coefficients from a Bayesian model returned by Bambi"""
        def getter(exog_ins) -> float:
            """Gets a particular coupling coefficient between endog_ins and exog_ins from the model"""
            try:
                coef = md_summary.loc[f'{endog_ins}_{exog_ins}_{coupling_var}']
                if coef < 0:
                    warnings.warn(
                        f'track {self.item["track_name"]}, coupling {endog_ins}/{exog_ins} < 0 ({coef})', UserWarning
                    )
                return coef
            except KeyError:
                return np.nan

        cols = ['intercept_bay', 'self_coupling_bay', 'coupling_piano_bay', 'coupling_bass_bay', 'coupling_drums_bay']
        if bayes_md is None:
            return {col: np.nan for col in cols}
        else:
            import arviz as az
            md_summary = az.summary(bayes_md)['mean']
            try:
                vals = [
                    md_summary.loc['Intercept'], md_summary.loc[f'{endog_ins}_{ioi_var}'],
                    getter('piano'), getter('bass'), getter('drums')
                ]
                return {c: v for c, v in zip(cols, vals)}
            except KeyError:
                return {col: np.nan for col in cols}

    def _extract_model_coefs(
            self,
            md: RegressionResultsWrapper | None,
            endog_ins: str,
            coupling_var: str = 'asynchrony',
            ioi_var: str = 'prev_ioi'
    ) -> dict:
        """Extracts coefficients from a regression model returned by statsmodels"""
        def getter(exog_ins: str) -> float:
            """Gets a particular coupling coefficient between endog_ins and exog_ins from the model"""
            try:
                coef = di[f'{endog_ins}_{exog_ins}_{coupling_var}']
                if coef < 0:
                    warnings.warn(
                        f'track {self.item["track_name"]}, coupling {endog_ins}/{exog_ins} < 0 ({coef})', UserWarning
                    )
                return coef
            except KeyError:
                return np.nan

        cols = ['intercept', 'self_coupling', 'coupling_piano', 'coupling_bass', 'coupling_drums']
        if md is None:
            return {col: np.nan for col in cols}
        else:
            di = md.params.to_dict()
            vals = [di['Intercept'], di[f'{endog_ins}_{ioi_var}'], getter('piano'), getter('bass'), getter('drums')]
            return {c: v for c, v in zip(cols, vals)}

    @staticmethod
    def _extract_model_goodness_of_fit(
            md: RegressionResultsWrapper | None
    ) -> dict:
        """Extracts goodness-of-fit parameters from a statsmodels regression."""
        cols = ['n_observations', 'resid_std', 'resid_len', 'rsquared', 'rsquared_adj', 'aic', 'bic', 'log-likelihood']
        if md is None:
            return {col: np.nan for col in cols}
        else:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                return {
                    'n_observations': int(md.nobs),
                    'resid_std': np.std(md.resid),
                    'resid_len': len(md.resid),
                    'rsquared': md.rsquared,
                    'rsquared_adj': md.rsquared_adj,
                    'aic': md.aic,
                    'bic': md.bic,
                    'log-likelihood': md.llf,
                }

    @staticmethod
    def extract_npvi(
            a: np.array
    ) -> float:
        """Extracts normalised pairwise variability index from an array of inter-onset intervals"""
        # TODO: fix this to account for nan
        return sum(
            [abs((k - k1) / ((k + k1) / 2)) if not any([isnan(k), isnan(k1)]) else np.nan for (k, k1) in zip(a, a[1:])]
        ) * 100 / (sum(1 for _ in a) - 1)

    def extract_burs(
            self,
            endog_ins: str = None,
            onset_array: np.array = None,
            beat_positions: np.array = None,
            use_log_burs: bool = False
    ) -> np.array:
        """Extracts beat-upbeat ratio (BUR) values from an array of onsets.

        The beat-upbeat ratio is introduced in [1] as a concept for analyzing the individual amount of 'swing' in two
        consecutive eighth note beat durations. It is calculated simply by dividing the duration of the first, 'long'
        eighth note beat by the second, 'short' beat. A BUR value of 2 indicates 'perfect' swing, i.e. a triplet quarter
        note followed by a triplet eighth note, while a BUR of 1 indicates 'even' eighth note durations.

        Arguments:
            endog_ins (str, optional): the name of the instrument to calculate BURs for. Must be provided if onset_array
                and beat_positions are not provided.
            onset_array (np.array, optional): the array of raw onsets. Must be provided if endog_ins is not given.
            beat_positions (np.array, optional): the array of crotchet beat positions. Must be provided if endog_ins is
                not given
            use_log_burs (bool, optional): whether or not to use the log^2 of inter-onset intervals to calculate BURs,
                as employed in [2]. Defaults to False.

        Returns:
            np.array: the calculated BUR values

        References:
            [1]: Benadon, F. (2006). Slicing the Beat: Jazz Eighth-Notes as Expressive Microrhythm. Ethnomusicology,
                50/1 (pp. 73-98).
            [2]: Corcoran, C., & Frieler, K. (2021). Playing It Straight: Analyzing Jazz Soloists’ Swing Eighth-Note
                Distributions with the Weimar Jazz Database. Music Perception, 38(4), 372–385.

        """
        # Get attributes, using instrument name if required
        if onset_array is None and beat_positions is None and endog_ins is None:
            raise AttributeError('Either endog_ins, or onset_array and beat_positions, must be provided.')
        if onset_array is None and endog_ins is not None:
            onset_array = self.om.summary_dict[endog_ins]
        if beat_positions is None and endog_ins is not None:
            beat_positions = self.om.summary_dict[endog_ins]
        # Use log2 burs if specified
        if use_log_burs:
            from math import log2 as func
        else:
            func = lambda a: a
        # Iterate through consecutive pairs of onsets
        for i1, i2 in zip(beat_positions, beat_positions[1:]):
            # Get the onsets between
            match = onset_array[np.where(np.logical_and(onset_array >= i1, onset_array <= i2))]
            # If we have a group of three consecutive eighth notes that start on a beat
            # TODO: investigate using the FlexQ algorithm instead for matching onsets to metrical positions
            if len(match) == 3:
                yield func((match[1] - match[0]) / (match[2] - match[1]))

    def extract_event_density(
            self,
            endog_ins: str = None,
            onset_arr: np.array = None
    ) -> np.array:
        """ED = notes per second"""
        # Get required attributes
        if endog_ins is None and onset_arr is None:
            raise AttributeError('At least one of endog_ins or onset_arr must be provided.')
        if onset_arr is None and endog_ins is not None:
            onset_arr = self.om.ons[endog_ins]
        # Extract average event density
        return (
            pd.DataFrame({'ts': pd.to_datetime(onset_arr, unit='s'), 'onset': onset_arr})
              .set_index('ts')
              .resample('1s')
              .count()
              .to_numpy()
        )

    def extract_metrical_event_density(self):
        """MED = average notes per (4/4) bar"""
        # TODO: finish this
        beats = self.om.summary_dict[ins]
        onsets = self.om.ons[ins]
        matches = []
        for i1, i2 in zip(beats, beats[1:]):
            matches.append(onsets[np.where(np.logical_and(onsets >= i1, onsets <= i2))])
        matches.append([])
        print(len(matches), len(beats))
        res = pd.DataFrame({'beats': beats, 'onsets': matches})

    def extract_bur_summary(
            self,
            endog_ins: str,
    ) -> dict:
        """Helper function to extract summary statistics from an array of BUR values"""
        burs = np.fromiter(self.extract_burs(endog_ins), dtype=float)
        return {
            'bur_mean': np.nanmean(burs),
            'bur_median': np.nanmedian(burs),
            'bur_std': np.nanstd(burs),
            'bur_count': len(burs),
            'burs': burs
        }

    def create_instrument_dict(
            self,
            endog_ins: str,
            md: RegressionResultsWrapper,
            bayes_md=None
    ) -> dict:
        """Creates summary dictionary for a single instrument, containing all extracted features.

        Arguments:
            endog_ins (str): the name of the instrument to create the dictionary for
            md (RegressionResultsWrapper): the statsmodels regression instance
            bayes_md (optional): the Bayes model returned from Bambi, will be skipped if not present

        Returns:
            dict: summary dictionary containing extracted features as key-value pairs

        """
        return {
            # Item metadata
            'track': self.item['track_name'],
            'album': self.item['album_name'],
            'bandleader': self.item['musicians'][self.item['musicians']['leader']],
            'year': self.item['recording_year'],
            'tempo': self.om.tempo,
            'instrument': endog_ins,
            'performer': self.item['musicians'][autils.INSTRS_TO_PERF[endog_ins]],
            # Raw beats
            'raw_beats': self.df[endog_ins],
            'interpolated_beats': self.num_interpolated[endog_ins],
            'observed_beats': self.df[endog_ins].notna().sum(),
            'missing_beats': self.df[endog_ins].isna().sum(),
            # Performance summary statistics
            'ioi_mean': self.df[f'{endog_ins}_prev_ioi'].mean(),
            'ioi_median': self.df[f'{endog_ins}_prev_ioi'].median(),
            'ioi_std': self.df[f'{endog_ins}_prev_ioi'].std(),
            # 'ioi_npvi': self.extract_npvi(self.df[f'{endog_ins}_prev_ioi']),
            # Cleaning metadata, e.g. missing beats
            'fraction_silent': self.om.silent_perc[endog_ins],
            'missing_beats_fraction': self.df[endog_ins].isna().sum() / self.df.shape[0],
            'total_beats': self.df.shape[0],
            'model_compiled': md is not None,
            # Event density functions
            'event_density': self.extract_event_density(endog_ins=endog_ins).mean(),
            # Model goodness-of-fit
            **self._extract_model_goodness_of_fit(md=md),
            # Model coefficients
            **self._extract_model_coefs(endog_ins=endog_ins, md=md),
            # Bayes model coefficients
            **self._extract_bayes_coefs(endog_ins=endog_ins, bayes_md=bayes_md),
            # Beat-upbeat ratio stats
            **self.extract_bur_summary(endog_ins=endog_ins)
        }


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    logger.addHandler(autils.TqdmLoggingHandler())

    onsets = autils.unserialise_object(rf'{autils.get_project_root()}\models', 'matched_onsets')
    dfs = []
    features = []
    for ons in tqdm(onsets):
        mm = ModelMaker(om=ons, interpolate=True, interpolation_limit=1)
        summary = []
        for ins in autils.INSTRS_TO_PERF.keys():
            mm.models[ins] = mm.generate_model(ins, standardise=False, difference_ioi=True, iqr_clean=False)
            summary.append(mm.create_instrument_dict(endog_ins=ins, md=mm.models[ins]))
        mm.summary_df = pd.DataFrame(summary)
        dfs.append(mm.summary_df)
        features.append(mm)
    big = pd.concat(dfs).reset_index(drop=True)
    autils.serialise_object(features, rf'{autils.get_project_root()}\models', 'extracted_features')
