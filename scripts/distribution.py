import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from AssetAllocation.datamanager import datamanager as dm
from copulas.multivariate.gaussian import GaussianMultivariate
import itertools
import seaborn as sns
from copulas.univariate import (
    BetaUnivariate,
    GammaUnivariate,
    GaussianKDE,
    GaussianUnivariate,
    TruncatedGaussian
)

returns_data = dm.get_asset_returns()

distributions = [stats.norm, stats.lognorm, stats.t, stats.gamma, stats.weibull_min,
                 stats.cauchy, stats.beta, stats.pareto, stats.chi2, stats.logistic]

best_distributions = {}
for column in returns_data.columns:
    best_distribution = None
    best_bic = np.inf
    best_params = None

    for distribution in distributions:

        params = distribution.fit(returns_data[column])

        log_likelihood = distribution.logpdf(returns_data[column], *params).sum()

        # Computing BIC
        k = len(params)
        n = len(returns_data[column])
        bic = -2 * log_likelihood + k * np.log(n)

        if bic < best_bic:
            best_distribution = distribution
            best_bic = bic
            best_params = params

    best_distributions[column] = {
        'distribution': best_distribution,
        'params': best_params,
        'bic': best_bic
    }

fitted_data = {}
for asset, dist_info in best_distributions.items():
    distribution = dist_info['distribution']
    params = dist_info['params']
    fitted_data[asset] = distribution.rvs(*params, size=len(returns_data))

best_copulas = {}
for asset, data in fitted_data.items():
    best_copula = None
    best_bic = np.inf

    copula_classes = [BetaUnivariate, GammaUnivariate, GaussianUnivariate, TruncatedGaussian]

    for copula_class in copula_classes:
        copula = copula_class()
        copula.fit(data)

        neg_log_likelihood = -np.sum(np.log(copula.pdf(data)))
        k = len(copula._params)
        n = len(data)
        bic = 2 * neg_log_likelihood + k * np.log(n)

        if bic < best_bic:
            best_copula = copula
            best_bic = bic

    best_copulas[asset] = best_copula

# copula = GaussianMultivariate()
# copula.fit(pd.DataFrame({asset: best_copula.sample(len(returns_data)) for asset, best_copula in best_copulas.items()}))
#
# copula_samples = copula.sample(len(returns_data))
