# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 17:10:52 2021

@author: Bloomberg
"""

import plan_params as pp
import util
import plots
import numpy as np

data_dict = util.get_data('inputs.xlsx')
plan = pp.plan_params(data_dict['policy_weights'], data_dict['ret'], data_dict['vol'], data_dict['corr'], data_dict['symbols'])

sim_ports_df = util.run_mc_simulation(len(plan), 1.02, plan.ret,plan.cov)

# Max sharpe ratio portfolio 
max_sharpe_port = util.get_max_sharpe_port(sim_ports_df)

# Max sharpe ratio portfolio weights
max_sharpe_port_wts = util.get_max_sharpe_weights(sim_ports_df)

plots.plot_mc_ports(sim_ports_df, max_sharpe_port)


bnds = ((-1.000000000001,-.99999999999999),)+((0,1.02),)+((0,1.02),)*2+((0,.6),)+((0,1.02),)*10+((.01,1.02),)

cons = ({'type': 'ineq', 'fun': lambda x: np.sum(x[1:3]) - 0.5},
        {'type': 'eq', 'fun': lambda x: np.sum(x[0:len(plan)]) - np.sum(x[4]) - .02})

opt_sharpe = plan.optimize(plan.min_sharpe_ratio,bnds, cons)

list(zip(plan.symbols,np.around(opt_sharpe['x']*100,2)))

opt_var = plan.optimize(plan.min_variance,bnds, cons)

targetrets = np.linspace(-.01,0.04,100)
tvols = []
tweights = []

for tr in targetrets:
    
    ef_cons = ({'type': 'eq', 'fun': lambda x: plan.portfolio_stats(x)[0] - tr},
               {'type': 'ineq', 'fun': lambda x: np.sum(x[1]+x[2])-.5},
               {'type': 'eq', 'fun': lambda x: np.sum(x[0]+x[1]+x[2]+x[3]+x[5]+x[6]+x[7]+x[8]+x[9]+x[10]+x[11]+x[12]+x[13]+x[14]+x[15]) - .02})
               
#                {'type': 'eq', 'fun': lambda x: np.sum(x) - .02})
    
    opt_ef = plan.optimize(plan.min_volatility,bnds, ef_cons)
    
    tvols.append(opt_ef['fun'])
    tweights.append(opt_ef['x'])

targetvols = np.array(tvols)
targetweights = np.array(tweights)
optimazed_weights = np.transpose(targetweights)
opt_ef

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.figsize'] = 16, 8


# Visualize the simulated portfolio for risk and return
fig = plt.figure()
ax = plt.axes()

import matplotlib.ticker as mtick

ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0, symbol='%', is_latex=False))
ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0, symbol='%', is_latex=False))

ax.set_title('Efficient Frontier Portfolio')

# Efficient Frontier
fig.colorbar(ax.scatter(targetvols, targetrets, c=targetrets / targetvols, 
                        marker='x', cmap='RdYlGn', edgecolors='black'), label='Sharpe Ratio') 

# Maximum Sharpe Portfolio
ax.plot(plan.portfolio_stats(opt_sharpe['x'])[1], plan.portfolio_stats(opt_sharpe['x'])[0], 'r*', markersize =15.0)

# Minimum Variance Portfolio
ax.plot(plan.portfolio_stats(opt_var['x'])[1], plan.portfolio_stats(opt_var['x'])[0], 'b*', markersize =15.0)

# Policy Portfolio
ax.plot(plan.fsv,plan.policy_rets, 'k*', markersize =15.0)

ax.set_xlabel('Expected Volatility')
ax.set_ylabel('Expected Return')
ax.grid(True)