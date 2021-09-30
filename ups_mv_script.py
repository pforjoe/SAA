# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 10:55:23 2021

@author: NVG9HXP
"""

# Import pandas
import pandas as pd

# Import numpy
import numpy as np
from numpy import *
from numpy.linalg import multi_dot

# Plot settings
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.figsize'] = 16, 8

import itertools

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

dataset = pd.read_excel("UPS Mean-Variance.xlsx",sheet_name = "Summarized Output", skiprows = 1, usecols=[2,5,8,9,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26], na_values=[""])
dataset = dataset.set_index('Asset Class or Liability')

# Portfolio Weights
policy_wts = dataset['FS AdjWeights'].to_numpy()[:,np.newaxis]

ret = dataset['Return']

vol = dataset['Vol'].to_numpy()[:,np.newaxis]

corr = dataset.iloc[:,3:21].to_numpy()

retmc= ret.to_numpy()[:,np.newaxis]

# Portfolio Return
policy_wts.T @ ret.to_numpy()[:,np.newaxis]

# Portfolio Covariance
cov = (vol @ vol.T)*corr
cov.shape

# Portfolio Variance
var = multi_dot([policy_wts.T, cov, policy_wts])
var

# Portfolio FSV
FSV = np.sqrt(var)
FSV

numofassets = len(policy_wts)-1
numofassets

symbols = ['Liability', '15+ STRIPS', 'Long Corp', 'Int Corp', 'Ultra 30 UST FUT', 'SP500', 'Russell 2000', 'MSCI EAFE', 'MSCI EM', 'MSCI ACWI','PE', 'RE', 'HY', 'HF', 'Commodities', 'Cash']

def portfolio_stats(weights):
    
    weights= np.array(weights)[:,np.newaxis]
    port_rets = weights.T @ ret[:,np.newaxis]    
    port_vols = np.sqrt(multi_dot([weights.T, cov, weights])) 
    
    return np.array([port_rets, port_vols, port_rets/port_vols]).flatten()

w = random.random(15)

# Set weights such that sum of weights equals 1.02
w /= sum(w)*(1/1.02)
w

w = np.insert(w,0,-1)[:,newaxis]

# Initialize the lists
rets = []; vols = []; wts = []

# Simulate 5,000 portfolios
for i in range (5000):
    
    # Generate random weights
    weights = random.random(numofassets)
    
    # Set weights such that sum of weights equals 1.02
    weights /= sum(weights)*(1/1.02)
    
    # Add the constant Liability
    weights = np.insert(weights,0,-1)[:,newaxis]
    
    # Portfolio statistics
    rets.append(weights.T @ retmc)        
    vols.append(sqrt(multi_dot([weights.T, cov, weights])))
    wts.append(weights.flatten())

# Record values     
port_rets = array(rets).flatten()
port_vols = array(vols).flatten()
port_wts = array(wts)

# Create a dataframe for analysis
msrp_df = pd.DataFrame({'returns': port_rets,
                      'volatility': port_vols,
                      'sharpe_ratio': port_rets/port_vols,
                      'weights': list(port_wts)})
msrp_df.tail(15)

# Maximum Sharpe Ratio
# Max sharpe ratio portfolio 
msrp = msrp_df.iloc[msrp_df['sharpe_ratio'].idxmax()]
msrp

# Max sharpe ratio portfolio weights
max_sharpe_port_wts = msrp_df['weights'][msrp_df['sharpe_ratio'].idxmax()]

# Allocation to achieve max sharpe ratio portfolio
dict(zip(symbols,np.around(max_sharpe_port_wts*100,2)))

# Visualize the simulated portfolio for risk and return
fig = plt.figure()
ax = plt.axes()
matplotlib.rcParams['figure.figsize'] = 16, 8

ax.set_title('Monte Carlo Simulated Allocation')

# Simulated portfolios
fig.colorbar(ax.scatter(port_vols, port_rets, c=port_rets / port_vols, 
                        marker='o', cmap='RdYlGn', edgecolors='black'), label='Sharpe Ratio') 

# Maximum sharpe ratio portfolio
ax.scatter(msrp['volatility'], msrp['returns'], c='red', marker='*', s = 300, label='Max Sharpe Ratio')

ax.set_xlabel('Expected Volatility')
ax.set_ylabel('Expected Return')
ax.grid(True)


# Import optimization module from scipy
import scipy.optimize as sco

# Maximizing sharpe ratio
def min_sharpe_ratio(weights):
    return -portfolio_stats(weights)[2]

numofassets = len(symbols)

#TODO:FUT (.33,.33)
bnds = ((-1.000000000001,-.99999999999999),)+((0,1.02),)*15
bnds

cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - .02})
bnds = bnds
init_weights = policy_wts.copy()

# Optimizing for maximum sharpe ratio
opt_sharpe = sco.minimize(min_sharpe_ratio, init_weights, method= 'SLSQP', bounds=bnds, constraints=cons)

# Portfolio weights
list(zip(symbols,np.around(opt_sharpe['x']*100,2)))

# Portfolio stats
stats = ['Returns', 'Volatility', 'Sharpe Ratio']
list(zip(stats,np.around(portfolio_stats(opt_sharpe['x']),4)))


# Minimize the variance
def min_variance(weights):
    return portfolio_stats(weights)[1]**2

# Optimizing for minimum variance
initial_wts = policy_wts.copy()
opt_var = sco.minimize(min_variance, initial_wts, method='SLSQP', bounds=bnds, constraints=cons)

# Portfolio weights
list(zip(symbols,np.around(opt_var['x']*100,2)))

# Portfolio stats
list(zip(stats,np.around(portfolio_stats(opt_var['x']),4)))

# Minimize the volatility
def min_volatility(weights):
    return portfolio_stats(weights)[1]

targetrets = linspace(-.0049,0.04,100)
tvols = []

for tr in targetrets:
    
    ef_cons = ({'type': 'eq', 'fun': lambda x: portfolio_stats(x)[0] - tr},
               {'type': 'eq', 'fun': lambda x: np.sum(x) - .02})
    
    opt_ef = sco.minimize(min_volatility, initial_wts, method='SLSQP', bounds=bnds, constraints=ef_cons)
    
    tvols.append(opt_ef['fun'])

targetvols = array(tvols)
opt_ef

# Visualize the simulated portfolio for risk and return
fig = plt.figure()
ax = plt.axes()

ax.set_title('Efficient Frontier Portfolio')

# Efficient Frontier
fig.colorbar(ax.scatter(targetvols, targetrets, c=targetrets / targetvols, 
                        marker='x', cmap='RdYlGn', edgecolors='black'), label='Sharpe Ratio') 

# Maximum Sharpe Portfolio
ax.plot(portfolio_stats(opt_sharpe['x'])[1], portfolio_stats(opt_sharpe['x'])[0], 'r*', markersize =15.0)

# Minimum Variance Portfolio
ax.plot(portfolio_stats(opt_var['x'])[1], portfolio_stats(opt_var['x'])[0], 'b*', markersize =15.0)

# Minimum Variance Portfolio
ax.plot(FSV,policy_return, 'y*', markersize =15.0)

ax.set_xlabel('Expected Volatility')
ax.set_ylabel('Expected Return')
ax.grid(True)