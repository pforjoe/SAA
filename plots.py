# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 17:33:37 2021

@authors: Roxton McNeal, Matt Johnston, Powis Forjoe
"""

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.figsize'] = 16, 8
import matplotlib.ticker as mtick
    

def plot_mc_ports(sim_ports_df, max_sharpe_port):
    # Visualize the simulated portfolio for risk and return
    fig = plt.figure()
    ax = plt.axes()
    # matplotlib.rcParams['figure.figsize'] = 16, 8
    
    ax.set_title('Monte Carlo Simulated Allocation')
    
    # Simulated portfolios
    fig.colorbar(ax.scatter(sim_ports_df['volatility'],sim_ports_df['returns'], c=sim_ports_df['sharpe_ratio'], 
                        marker='o', cmap='RdYlGn', edgecolors='black'), label='Sharpe Ratio') 
    
    # Maximum sharpe ratio portfolio
    ax.scatter(max_sharpe_port['volatility'], max_sharpe_port['returns'], c='red', marker='*', s = 300, label='Max Sharpe Ratio')
    
    ax.set_xlabel('Expected Volatility')
    ax.set_ylabel('Expected Return')
    ax.grid(True)
    
def plot_risk_ret(targetrets,targetvols,plan,opt_sharpe, opt_var):
    
    fig = plt.figure()
    ax = plt.axes()
    # matplotlib.rcParams['figure.figsize'] = 16, 8
    
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
