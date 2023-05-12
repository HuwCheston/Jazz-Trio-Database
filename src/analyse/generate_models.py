#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate the phase correction models for each item in the corpus
"""

from math import isnan

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.formula.api as smf
from statsmodels.regression.linear_model import RegressionResultsWrapper

import src.utils.analyse_utils as autils
from src.analyse.detect_onsets import OnsetMaker


class ModelMaker:
    sample_rate = autils.SAMPLE_RATE
    instrs = ['piano', 'bass', 'drums']

    def __init__(
            self,
            om: OnsetMaker
    ):
        self.om = om
        self.item = om.item
        self.df = self._format_df(om.summary_dict)
        self.models = {}
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

    def generate_model(
            self,
            endog_ins: str,
            standardise: bool = False,
    ) -> RegressionResultsWrapper | None:
        """Generates the phase correction model for one instrument.

        Arguments:
            endog_ins (str): the name of the 'dependent variable' instrument, whose IOIs we are predicting
            standardise (bool, optional): use standard ('z') scores when computing regression, defaults to True

        Returns:
            RegressionResultsWrapper: the fitted regression model

        """
        exog = [ex for ex in self.instrs if ex != endog_ins]
        # Create our asynchrony (coupling) terms in the model
        exog_ins = '+'.join(f'{endog_ins}_{instr}_asynchrony' for instr in exog)
        # Create the rest of our model
        md = f'{endog_ins}_next_ioi~{endog_ins}_prev_ioi+' + exog_ins
        if standardise:
            df = self.df.select_dtypes(include=[np.number]).dropna().apply(stats.zscore)
        else:
            df = self.df
        # Create the regression model, fit to the data, and return
        try:
            return smf.ols(md, data=df, missing='drop').fit()
        except (ValueError, IndexError):
            return None

    @staticmethod
    def _extract_model_coefs(
            md: RegressionResultsWrapper | None,
            endog_ins: str,
            coupling_var: str = 'asynchrony',
            timekeeper_var: str = 'prev_ioi'
    ) -> dict:
        """

        """

        def getter(
                exog_ins: str
        ) -> float:
            try:
                return di[f'{endog_ins}_{exog_ins}_{coupling_var}']
            except KeyError:
                return np.nan

        if md is None:
            return {
                col: np.nan for col in [
                    'intercept', 'self_coupling', 'coupling_piano', 'coupling_drums', 'coupling_bass'
                ]
            }
        else:
            di = md.params.to_dict()
            return {
                'intercept': di['Intercept'],
                'self_coupling': di[f'{endog_ins}_{timekeeper_var}'],
                'coupling_piano': getter('piano'),
                'coupling_bass': getter('bass'),
                'coupling_drums': getter('drums'),
            }

    @staticmethod
    def _extract_model_goodness_of_fit(
            md: RegressionResultsWrapper | None
    ) -> dict:
        """

        """

        if md is None:
            return {
                col: np.nan for col in [
                    'resid_std', 'resid_len', 'rsquared', 'rsquared_adj', 'aic', 'bic', 'log-likelihood'
                ]
            }
        else:
            return {
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
            arr: np.array
    ) -> float:
        """

        """

        # TODO: fix this to account for nan
        return sum(
            [abs((k - k1) / ((k + k1) / 2)) if not any([isnan(k), isnan(k1)]) else np.nan for (k, k1) in
             zip(arr, arr[1:])]) * 100 / (sum(1 for _ in arr) - 1)

    def create_instrument_dict(
            self,
            endog_ins: str,
            md: RegressionResultsWrapper,
    ):
        """

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
            # Performance summary statistics
            'ioi_mean': self.df[f'{endog_ins}_prev_ioi'].mean(),
            'ioi_median': self.df[f'{endog_ins}_prev_ioi'].median(),
            'ioi_std': self.df[f'{endog_ins}_prev_ioi'].std(),
            'ioi_npvi': self.extract_npvi(self.df[f'{endog_ins}_prev_ioi']),
            # Cleaning metadata, e.g. missing beats
            'fraction_silent': self.om.silent_perc[endog_ins],
            'missing_beats': self.df[endog_ins].isna().sum(),
            'missing_beats_fraction': self.df[endog_ins].isna().sum() / self.df.shape[0],
            'total_beats': self.df.shape[0],
            'model_compiled': md is not None,
            # Model goodness-of-fit
            **self._extract_model_goodness_of_fit(md=md),
            # Model coefficients
            **self._extract_model_coefs(endog_ins=endog_ins, md=md)
        }


if __name__ == "__main__":
    onsets = autils.unserialise_object(r'..\..\models', 'matched_onsets')
    dfs = []
    for ons in onsets:
        mm = ModelMaker(om=ons)
        summary = []
        for ins in ['piano', 'bass', 'drums']:
            mm.models[ins] = mm.generate_model(ins)
            summary.append(mm.create_instrument_dict(endog_ins=ins, md=mm.models[ins]))
        mm.summary_df = pd.DataFrame(summary)
        dfs.append(mm.summary_df)
    big = pd.concat(dfs)
    pass

# import seaborn as sns
# import matplotlib.pyplot as plt
# from matplotlib.offsetbox import AnchoredText
# plt.rcParams.update({'font.size': 18})
# def f(row):
#     if int(row['year']) < 1962:
#         val = 'P: Evans\nB: LeFaro\nD: Motian'
#     elif int(row['year']) < 1978:
#         val = 'P: Evans\nB: Gomez\nD: Morell'
#     else:
#         val = 'P: Evans\nB: Johnson\nD: LeBarbera'
#     return val
# big = pd.concat(dfs)
# df = big[(big['missing_beats_fraction'] < 0.5) & (big['rsquared'] < 0.99999999) & (big['rsquared'] > 0)].reset_index(
#     drop=True)
# df['trio'] = df.apply(f, axis=1)
# fig, ax = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(24, 8))
#
# for a, (idx, grp) in zip(ax.flatten(), df.groupby('instrument')):
#     melt = pd.melt(grp, id_vars='trio', value_vars=['coupling_piano', 'coupling_bass', 'coupling_drums'],
#                    var_name='influenced', value_name='coupling')
#     melt['influenced'] = melt['influenced'].str.replace('coupling_', '')
#     g = sns.barplot(
#         data=melt, x='influenced', y='coupling', ax=a, hue='trio', estimator=np.nanmean, errorbar='ci', n_boot=1000,
#         errcolor='black', errwidth=3, edgecolor='black', lw=3, capsize=0.1, width=0.8, palette='tab10'
#
#     )
#     g.set(title=idx.title(), xlabel='', ylabel='', xticklabels=[str(i.get_text()).title() for i in g.get_xticklabels()],
#           ylim=(0, 0.85))
#     a.add_artist(AnchoredText(f"$N$ = {(grp['total_beats'] - grp['missing_beats']).sum()}", loc=1, frameon=False))
#     hand, lab = g.get_legend_handles_labels()
#     g.get_legend().remove()
#     plt.setp(a.spines.values(), linewidth=3)
#     a.tick_params(axis='both', width=3)
# leg = fig.legend(hand, lab, title='Trio', loc='right', edgecolor='black', frameon=False)
# leg = plt.setp(leg.get_title(), fontsize=20)
# fig.supxlabel('Influencer instrument')
# fig.supylabel('Coupling')
# fig.suptitle(f'Influenced instrument')
# fig.subplots_adjust(top=0.875, bottom=0.1, right=0.875, left=0.065, wspace=0.1)
# plt.show()
