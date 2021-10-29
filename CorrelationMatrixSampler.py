import random
import pandas as pd
from AssetAllocation.analytics import ts_analytics

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
        self._isFirstRun = True

    # Summary: Computes the base correlation.
    def base_correlation(self):
        return ts_analytics.compute_ewcorr_matrix(self._returnTimeSeries)

    # Summary: Randomly generates a series of correlation matrices
    # Params:
    #   number_of_samples:  The number of samples to draw.
    # Returns: An array of arrays containing the indices to sample.
    def randomly_sample_correlation_matrices(self, number_of_samples: int) -> []:
        random_returns = self._sample_returns(number_of_samples)
        correlation_matrices = []

        for return_set in random_returns:
            correlation_matrices.append(ts_analytics.compute_ewcorr_matrix(return_set).copy())

        return correlation_matrices

    # Summary: Randomly generates a series of returns to use in sampling returns
    # Params:
    #   number_of_samples:  The number of samples to draw.
    # Returns: An array of arrays containing the indices to sample.
    def _sample_returns(self, number_of_samples: int) -> []:
        random_indices = self._generate_random_indices(number_of_samples)

        random_returns = []
        for index_set in random_indices:
            count = 0
            df = self._returnTimeSeries.iloc[0:0, :].copy()
            for index in index_set:
                row = self._returnTimeSeries.iloc[index].copy()
                date = self._returnTimeSeries.index[count]
                row.name = date
                df = df.append(row.copy())
                count = count + 1

            random_returns.append(df.copy())

        return random_returns

    # Summary: Randomly generates a series of indices to use in sampling returns
    # Params:
    #   number_of_samples:  The number of samples to draw.
    # Returns: An array of arrays containing the indices to sample.
    def _generate_random_indices(self, number_of_samples: int) -> []:

        if self._isFirstRun or self._resetSeedAtEachRun:
            random.seed(self._seed)

        result_matrix = []

        for x in range(0, number_of_samples):
            random_numbers = random.choices(range(0, len(self._returnTimeSeries.index)), k=len(self._returnTimeSeries.index))
            result_matrix.append(random_numbers)

        self._isFirstRun = False
        return result_matrix
