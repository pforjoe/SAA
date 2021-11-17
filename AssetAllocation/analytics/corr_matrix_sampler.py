"""
Created on Sun Oct 10 12:27:19 2021

@author: Antoine Tan
"""

import numpy as np
import pandas as pd
from .import ts_analytics as ts

# Summary: Samples correlation matrices.
class CorrelationMatrixSampler:

    # Summary: Creates a new correlation matrix sampler.
    # Params:
    #   return_ts:              A dataframe containing the returns to use in sampling.
    #   seed:                   The seed to use in random number generation.
    #   reset_seed_at_each_run: Whether or not to reset the seed at each run. Defaults to true.
    def __init__(self, return_ts: pd.DataFrame, seed: int, reset_seed_at_each_run: bool = False):
        self._returnTimeSeries = return_ts
        self._seed = seed
        self._resetSeedAtEachRun = reset_seed_at_each_run
        self._rng = np.random.RandomState(self._seed)

    # Summary: Computes the base correlation.
    def base_correlation(self):
        return ts.compute_ewcorr_matrix(self._returnTimeSeries)

    # Summary: Randomly generates a series of correlation matrices
    # Params:
    #   number_of_samples:  The number of samples to draw.
    # Returns: An array of arrays containing the indices to sample.
    def randomly_sample_correlation_matrices(self, number_of_samples: int) -> []:
        random_returns = self._sample_returns(number_of_samples)
        correlation_matrices = []

        for return_set in random_returns:
            correlation_matrices.append(ts.compute_ewcorr_matrix(return_set))

        return correlation_matrices

    # Summary: Randomly generates a series of returns to use in sampling returns
    # Params:
    #   number_of_samples:  The number of samples to draw.
    # Returns: An array of arrays containing the indices to sample.
    def _sample_returns(self, number_of_samples: int) -> []:
        if self._resetSeedAtEachRun:
            self._rng = np.random.RandomState(self._seed)

        random_returns = []
        for x in range(0, number_of_samples):
            random_returns.append(self._returnTimeSeries.sample(frac=1, replace=True, random_state=self._rng))

        return random_returns

