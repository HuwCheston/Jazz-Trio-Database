#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility classes, functions, and variables used in the feature extraction process"""

import json

import numpy as np
import pandas as pd


__all__ = ["BaseExtractor"]


# TODO: can we separate this class from the rhythm features classes?
class BaseExtractor:
    """Base feature extraction class, with some methods that are useful for all classes"""

    def __init__(self):
        # These are the default functions we'll call on any array to populate our summary statistics dictionary
        # We have to define these inside __init__ otherwise they'll be overwritten in the child classes
        self.summary_funcs = dict(
            mean=np.nanmean,
            median=np.nanmedian,
            std=np.nanstd,
            var=np.nanvar,
            quantile25=self.quantile25,
            quantile75=self.quantile75,
            count=len,
            count_nonzero=self.count_nonzero,
        )
        self.summary_dict = {}

    @staticmethod
    def count_nonzero(x) -> int:
        """Simple wrapper around `np.count_nonzero` that removes NaN values from an array"""
        return np.count_nonzero(~np.isnan(x))

    @staticmethod
    def quantile25(x) -> float:
        """Simple wrapper around `np.nanquantile` with arguments set"""
        return np.nanquantile(x, 0.25)

    @staticmethod
    def quantile75(x) -> float:
        """Simple wrapper around `np.nanquantile` with arguments set"""
        return np.nanquantile(x, 0.75)

    def __bool__(self):
        """Overrides built-in boolean method to return whether the summary dictionary has been populated"""
        return len(self.summary_dict.keys()) < 0

    def __contains__(self, item: str):
        """Overrides built-in method to return item from summary dictionary by key"""
        return item in self.summary_dict.keys()

    def __iter__(self):
        """Overrides built-in method to return iterable of key-value pairs from summary dictionary"""
        return self.summary_dict.items()

    def __len__(self):
        """Overrides built-in method to return length of summary dictionary"""
        return len(self.summary_dict.keys())

    def __repr__(self) -> dict:
        """Overrides default string representation to print a dictionary of summary stats"""
        return json.dumps(self.summary_dict)

    def update_summary_dict(self, array_names, arrays, *args, **kwargs) -> None:
        """Update our summary dictionary with values from this feature. Can be overridden!"""
        for name, df in zip(array_names, arrays):
            self.summary_dict.update({f'{name}_{func_k}': func_v(df) for func_k, func_v in self.summary_funcs.items()})

    @staticmethod
    def get_between(arr, i1, i2) -> np.array:
        """From an array `arr`, get all onsets between an upper and lower bound `i1` and `i2` respectively"""
        return arr[np.where(np.logical_and(arr >= i1, arr <= i2))]

    @staticmethod
    def truncate_df(
            arr: pd.DataFrame | pd.Series,
            low: float,
            high: float,
            col: str = None,
            fill_nans: bool = False
    ) -> pd.DataFrame:
        """Truncate a dataframe or series between a low and high threshold.

        Args:
            arr (pd.DataFrame | pd.Series): dataframe to truncate
            low (float): lower boundary for truncating
            high (float): upper boundary for truncating. Must be greater than `low`.
            col (str): array to use when truncating. Must be provided if `isinstance(arr, pd.DataFrame)`
            fill_nans (bool, optional): whether to replace values outside `low` and `high` with `np.nan`

        Raises:
            AssertionError: if `high` < `low`

        Returns:
            pd.DataFrame

        """
        # If both our lower and higher thresholds are NaN, return the array without masking
        if all([np.isnan(low), np.isnan(high)]):
            return arr
        # If we only have one value in our dataframe, or every value is NaN, low = high: this doesn't affect things
        assert low <= high
        # If we've passed in a series, we have to deal with it in a slightly different way
        if isinstance(arr, pd.Series):
            if fill_nans:
                return arr.mask(~arr.between(low, high))
            else:
                return arr[lambda x: (low <= x) & (x <= high)]
        # Otherwise, if we've passed in a dataframe, we have to deal with it in a different way
        elif isinstance(arr, pd.DataFrame):
            # We must provide a column to use for truncating in this case
            if col is None:
                raise AttributeError('Must provide argument `col` with `isinstance(arr, pd.DataFrame)`')
            mask = (low <= arr[col]) & (arr[col] <= high)
            if fill_nans:
                return arr.mask(~mask)
            else:
                return arr[mask]
