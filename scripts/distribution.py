import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from AssetAllocation.datamanager import datamanager as dm
from copulas.multivariate.gaussian import GaussianMultivariate
import itertools
import seaborn as sns
from scipy.special import logit, expit
from copulas.multivariate.tree import Tree

from copulas.bivariate import Bivariate
from copulas.bivariate.frank import Frank
from copulas.bivariate.clayton import Clayton
from copulas.bivariate.gumbel import Gumbel
from copulas.bivariate.independence import Independence

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

copula_classes= {
        'independence': Independence(),
        'clayton': Clayton(),
        'frank': Frank(),
        'gumbel': Gumbel()
    }

transformed_data = {}
for asset, data in fitted_data.items():
    if asset != 'Cash':
        transformed_data[asset] = (data - np.min(data)) / (np.max(data) - np.min(data))
asset_copulas = {}
best_likelihood = float('-inf')
for asset_pair in itertools.combinations(transformed_data.keys(), 2):
    asset1, asset2 = asset_pair
    asset1_data = transformed_data[asset1]
    asset2_data = transformed_data[asset2]

    samples = np.column_stack([asset1_data, asset2_data])

    # best_likelihood = float('-inf')
    best_copula = None

    for copula_name, copula_class in copula_classes.items():
        try:
            copula_class.fit(samples)
            log_likelihood = copula_class.log_probability_density(samples)
            max_log_likelihood = np.max(log_likelihood)

            if max_log_likelihood > best_likelihood:
                best_likelihood = max_log_likelihood
                best_copula_name = copula_name
                best_copula = copula_class

        except ValueError as e:
            if 'out of limits for the given' in str(e) or 'value' in str(e):
                print(f"Warning: {e}. Skipping this copula for {asset_pair}.")
                continue
            else:
                raise e

    asset_copulas[asset_pair] = (best_copula_name, best_copula)

# for asset_pair, (best_copula_name, best_copula) in asset_copulas.items():
#     print(f"For assets {asset_pair}, the best bivariate copula is {best_copula_name} with maximum likelihood: {best_likelihood}")
print(f"The best bivariate copula is {best_copula_name} with maximum likelihood: {best_likelihood}")


#Plot
for asset_pair, (best_copula_name, best_copula) in asset_copulas.items():
    asset1, asset2 = asset_pair
    asset1_data = transformed_data[asset1]
    asset2_data = transformed_data[asset2]
    plt.subplot(1, 2, 2)
    u = np.linspace(0, 1, 100)
    v = np.linspace(0, 1, 100)
    U, V = np.meshgrid(u, v)
    uv = np.column_stack([U.ravel(), V.ravel()])
    copula_density = best_copula.pdf(uv).reshape(U.shape)
    plt.contourf(U, V, copula_density, cmap='viridis')
    plt.colorbar(label='Copula Density')
    plt.title(f'Contour Plot of Copula Density ({best_copula_name})')
    plt.xlabel('U')
    plt.ylabel('V')
    plt.grid(True)

    plt.tight_layout()
    plt.show()
