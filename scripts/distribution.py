import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from AssetAllocation.datamanager import datamanager as dm

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
        'distribution': best_distribution.name,
        'params': best_params,
        'bic': best_bic
    }

# Print the results
for asset_class, result in best_distributions.items():
    print(f"Asset Class: {asset_class}")
    print(f"Best Distribution: {result['distribution']}")
    print(f"Parameters: {result['params']}")
    print(f"BIC: {result['bic']}")
    print()


for asset_class, result in best_distributions.items():
    plt.figure(figsize=(8, 6))
    plt.hist(returns_data[asset_class], bins=20, density=True, alpha=0.6, color='g', label='Histogram')

    x = np.linspace(min(returns_data[asset_class]), max(returns_data[asset_class]), 1000)
    fitted_distribution = getattr(stats, result['distribution'])(*result['params'])
    plt.plot(x, fitted_distribution.pdf(x), 'r-', lw=2, label= result['distribution'])

    plt.title(f'Fitted Distribution for {asset_class}')
    plt.xlabel('Returns')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True)
    plt.show()