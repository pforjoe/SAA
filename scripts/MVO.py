# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 14:05:57 2023

@author: PCR7FJW
"""
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from AssetAllocation.datamanager import datamanager as dm
import os
CWD = os.getcwd()


def calculate_covariance_matrix(correlation_matrix, volatility):
    return np.outer(volatility, volatility) * correlation_matrix

    # Bootstrap resampling logic
def bootstrap_resample(data):
    n = len(data)
    resample_indices = np.random.choice(n, n, replace=True)
    return data.iloc[resample_indices].reset_index(drop=True)

def calculate_cagr(returns_series, years_lookback = 3):
    years = returns_series.index.year.unique().tolist()[-years_lookback:]
    returns = returns_series.loc[f'{years[0]}-01-01':f'{years[-1]}-12-31']
    cum_returns = (1 + returns).cumprod() - 1
    total_return = cum_returns.iloc[-1]
    num_years = len(returns) / 252
    cagr = (1 + total_return) ** (1 / num_years) - 1
    return cagr

def calculate_max_drawdown(returns_series):
    cum_returns = (1 + returns_series).cumprod()
    peak = cum_returns.cummax()
    drawdown = (cum_returns/peak)-1
    max_drawdown = drawdown.min()
    return max_drawdown

def calculate_average_annual_drawdowns(returns_series, years_lookback = 3):
    avg_max_dd = 0
    for year in returns_series.index.year.unique().tolist()[-years_lookback:]:
        returns_year = returns_series.loc[f'{year}-01-01':f'{year}-12-31']
        max_dd_year = calculate_max_drawdown(returns_year)
        avg_max_dd += max_dd_year
    avg_max_dd /= years_lookback
    return avg_max_dd

def asset_class_bounds():
    asset_class_bounds = {
        'Private Equity': (0.05, 0.15),
        'Credit': (0.05, 0.15),
        'Real Estate': (0.05, 0.15),
    }
    return asset_class_bounds



def mean_variance_optimization_V1(returns):
    """
    Perform mean-variance optimization and return the optimal weights
    Objective using sharpe
    
    Parameters:
    returns -- DataFrame

    Returns:
    optimal_weights -- array
    """
    
    # Calculate the mean and covariance of returns
    mean_returns = returns.mean() * 12
    cov_matrix = returns.cov()

    # Define the objective function for mean-variance optimization - minimizing the negative sharpe ratio
    def objective(weights):
        portfolio_return = np.sum(mean_returns * weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -portfolio_return / portfolio_volatility

    # Define constraints and bounds
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
                   {'type': 'eq', 'fun': lambda weights: weights[returns.columns.get_loc('Cash')] - 0.02})
    bounds = [asset_class_bounds().get(asset, (0, 0.5)) for asset in returns.columns.tolist()]

    # Initialize equal weights for each asset
    initial_weights = np.ones(len(returns.columns.tolist())) / len(returns.columns.tolist())

    # Run the optimization
    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

    # Get the optimal weights
    optimal_weights = result.x

    return optimal_weights


#including bootstraping resampling correlation
def mean_variance_optimization_V2(returns, num_iterations=1000):
    """
    Perform mean-variance optimization and return the optimal weights with resmapling correlation matrix
    Objective using sharpe
    
    Parameters:
    returns -- DataFrame
    num_iterations -- float

    Returns:
    final_optimal_weights -- array
    """
    
    # Calculate the mean and volatility of returns
    mean_returns = returns.mean() * 12
    volatility = returns.std(axis=0)

    # Resample correlation matrices using bootstrap
    resampled_matrices = []

    for _ in range(num_iterations):
        # Bootstrap resampling for returns data
        resampled_returns = bootstrap_resample(returns)

        # Calculate the correlation matrix for the resampled dataset
        resampled_matrix = np.corrcoef(resampled_returns, rowvar=False)
        
        # Append the resampled matrix to the list
        resampled_matrices.append(resampled_matrix)

    # Define the mean-variance optimization objective function
    def objective(weights, expected_returns, covariance_matrix):
        portfolio_return = np.dot(expected_returns, weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
        return -portfolio_return / portfolio_volatility

    # Perform mean-variance optimization for each resampled matrix
    all_optimal_weights = []

    for correlation_matrix in resampled_matrices:
        covariance_matrix = calculate_covariance_matrix(correlation_matrix, volatility)
        
        # Define constraints
        constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
                       {'type': 'eq', 'fun': lambda weights: weights[returns.columns.get_loc('Cash')] - 0.02})

        # Define bounds
        bounds = [asset_class_bounds().get(asset, (0, 0.5)) for asset in returns.columns.tolist()]

        # Initial guess
        initial_weights = np.ones(len(returns.columns.tolist())) / len(returns.columns.tolist())

        # Perform optimization
        result = minimize(objective, initial_weights, args=(mean_returns, covariance_matrix),
                          method='SLSQP', bounds=bounds, constraints=constraints)

        # Collect optimal weights
        optimal_weights = result.x
        all_optimal_weights.append(optimal_weights)

    # Aggregate results (e.g., take the mean or median)
    final_optimal_weights = np.mean(all_optimal_weights, axis=0)

    return final_optimal_weights


def mean_variance_optimization_V3(returns, num_iterations=1000):
    """
    Perform mean-variance optimizatio and return the optimal weights with resmapling correlation matrix
    Objective using calmar
    
    Parameters:
    returns -- DataFrame
    num_iterations -- float

    Returns:
    final_optimal_weights -- array
    """
    
    # Calculate the mean and volatility of returns
    mean_returns = returns.mean()*12
    volatility = returns.std(axis=0)

    # Resample correlation matrices using bootstrap
    resampled_matrices = []

    for _ in range(num_iterations):
        # Bootstrap resampling for returns data
        resampled_returns = bootstrap_resample(returns)

        # Calculate the correlation matrix for the resampled dataset
        resampled_matrix = np.corrcoef(resampled_returns, rowvar=False)
        
        # Append the resampled matrix to the list
        resampled_matrices.append(resampled_matrix)
    
    # objective function
    def objective(weights, expected_returns, covariance_matrix):
        portfolio_return = np.dot(expected_returns, weights)
        max_drawdown = calculate_max_drawdown(returns @ weights)
        calmar_ratio = portfolio_return / abs(max_drawdown)
        return -calmar_ratio

    # optimization for each sampled matrix
    all_optimal_weights = []

    for correlation_matrix in resampled_matrices:
        covariance_matrix = calculate_covariance_matrix(correlation_matrix, volatility)
        
        # constraints and bounds
        constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
                       {'type': 'eq', 'fun': lambda weights: weights[returns.columns.get_loc('Cash')] - 0.02})
        bounds = [asset_class_bounds().get(asset, (0, 0.5)) for asset in returns.columns.tolist()]

        # Initial guess
        initial_weights = np.ones(len(returns.columns.tolist())) / len(returns.columns.tolist())

        # Perform optimization
        result = minimize(objective, initial_weights, args=(mean_returns, covariance_matrix),
                          method='SLSQP', bounds=bounds, constraints=constraints)

        # Collect optimal weights
        optimal_weights = result.x
        all_optimal_weights.append(optimal_weights)

    # Aggregate results (e.g., take the mean or median)
    final_optimal_weights = np.mean(all_optimal_weights, axis=0)

    return final_optimal_weights


def mean_variance_optimization_V4(returns, num_iterations=1000):
    """
    Perform mean-variance optimizatio and return the optimal weights with resmapling correlation matrix
    Objective using calmar and sharpe
    
    Parameters:
    returns -- DataFrame
    num_iterations -- float

    Returns:
    final_optimal_weights -- array
    """
    
    # Calculate the mean and volatility of returns
    mean_returns = returns.mean()*252
    volatility = returns.std(axis=0)

    # Resample correlation matrices using bootstrap
    resampled_matrices = []

    for _ in range(num_iterations):
        # Bootstrap resampling for returns data
        resampled_returns = bootstrap_resample(returns)

        # Calculate the correlation matrix for the resampled dataset
        resampled_matrix = np.corrcoef(resampled_returns, rowvar=False)
        
        # Append the resampled matrix to the list
        resampled_matrices.append(resampled_matrix)

    # define objective
    def composite_objective(weights, expected_returns, covariance_matrix, alpha=0.3):
        # Objective function that combines Sharpe ratio and Calmar ratio
        portfolio_return = np.dot(expected_returns, weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
        max_drawdown = calculate_max_drawdown(returns @ weights)
        
        sharpe_ratio = portfolio_return / portfolio_volatility
        calmar_ratio = portfolio_return / abs(max_drawdown)
        
        # Combine Sharpe and Calmar ratios using a weighted sum
        composite_objective_value = alpha * sharpe_ratio + (1 - alpha) * calmar_ratio
        
        return -composite_objective_value  # Minimize negative composite objective


    # optimization for each sampled matrix
    all_optimal_weights = []

    for correlation_matrix in resampled_matrices:
        covariance_matrix = calculate_covariance_matrix(correlation_matrix, volatility)
        
        # constraints and bounds
        constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
                       {'type': 'eq', 'fun': lambda weights: weights[returns.columns.get_loc('Cash')] - 0.02})
        bounds = [asset_class_bounds().get(asset, (0, 0.5)) for asset in returns.columns.tolist()]

        # Initial guess
        initial_weights = np.ones(len(returns.columns.tolist())) / len(returns.columns.tolist())

        # Perform optimization
        result = minimize(composite_objective, initial_weights, args=(mean_returns, covariance_matrix),
                          method='SLSQP', bounds=bounds, constraints=constraints)

        # Collect optimal weights
        optimal_weights = result.x
        all_optimal_weights.append(optimal_weights)

    # Aggregate results (e.g., take the mean or median)
    final_optimal_weights = np.mean(all_optimal_weights, axis=0)

    return final_optimal_weights


def mean_variance_optimization_V5(returns, num_iterations=1000, years_lookback = 3, rf_rate = 0.1):
    """
    Perform mean-variance optimization and return the optimal weights with resampled correlation matrix
    Objective using CAGR divided by average annual maximum drawdown
    
    Parameters:
    returns -- DataFrame
    num_iterations -- int

    Returns:
    final_optimal_weights -- array
    """
    
    # Calculate the mean and volatility of returns
    mean_returns = returns.mean() * 252
    volatility = returns.std(axis=0)

    # Resample correlation matrices using bootstrap
    resampled_matrices = []

    for _ in range(num_iterations):
        # Bootstrap resampling for returns data
        resampled_returns = bootstrap_resample(returns)

        # Calculate the correlation matrix for the resampled dataset
        resampled_matrix = np.corrcoef(resampled_returns, rowvar=False)
        
        # Append the resampled matrix to the list
        resampled_matrices.append(resampled_matrix)
    
    # Objective function using CAGR divided by average annual maximum drawdown
    def objective(weights, expected_returns, covariance_matrix):
        cagr = calculate_cagr(returns @ weights, years_lookback)
        avg_max_drawdown = calculate_average_annual_drawdowns(returns @ weights, years_lookback)
        sterling_ratio = cagr / (abs(avg_max_drawdown) + rf_rate)
        return -sterling_ratio

    # Optimization for each sampled matrix
    all_optimal_weights = []

    for correlation_matrix in resampled_matrices:
        covariance_matrix = calculate_covariance_matrix(correlation_matrix, volatility)
        
        # Constraints and bounds
        constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
                       {'type': 'eq', 'fun': lambda weights: weights[returns.columns.get_loc('Cash')] - 0.02})
        bounds = [asset_class_bounds().get(asset, (0, 0.5)) for asset in returns.columns.tolist()]

        # Initial guess
        initial_weights = np.ones(len(returns.columns.tolist())) / len(returns.columns.tolist())

        # Perform optimization
        result = minimize(objective, initial_weights, args=(mean_returns, covariance_matrix),
                          method='SLSQP', bounds=bounds, constraints=constraints)

        # Collect optimal weights
        optimal_weights = result.x
        all_optimal_weights.append(optimal_weights)

    # Aggregate results (e.g., take the mean or median)
    final_optimal_weights = np.mean(all_optimal_weights, axis=0)

    return final_optimal_weights


#==================================================================================================================


#calculated returns off of price data
returns = dm.get_ts_data(plan='IBT')['returns']
returns = dm.get_ts_data(plan='Retirement')['returns']
returns = dm.get_ts_data(plan='Pension')['returns']

drop_list = ['Ultra 30Y Futures', 'Hedges']
returns = returns.drop(columns = drop_list)

mean_var_weights_V1 = pd.DataFrame({'Optimal Weight V1': mean_variance_optimization_V1(returns).tolist()}, index=returns.columns.tolist())
mean_var_weights_V2 = pd.DataFrame({'Optimal Weight V2': mean_variance_optimization_V2(returns).tolist()}, index=returns.columns.tolist())
mean_var_weights_V3 = pd.DataFrame({'Optimal Weight V3': mean_variance_optimization_V3(returns).tolist()}, index=returns.columns.tolist())

program_mean_var_weights = pd.merge(mean_var_weights_V1, mean_var_weights_V2, left_index=True, right_index=True, how='inner').merge(mean_var_weights_V3, left_index=True, right_index=True, how='inner')



