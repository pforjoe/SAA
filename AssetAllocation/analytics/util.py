# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 16:22:39 2021

@authors: Roxton McNeal, Matt Johnston, Powis Forjoe
"""

# Import pandas
import pandas as pd
import numpy as np
from AssetAllocation.datamanager import datamanager as dm
from datetime import datetime
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

def add_dimension(data):
    return data.to_numpy()[:,np.newaxis]

def add_sharpe_col(ret_vol_df):
    ret_vol_df['Sharpe'] = ret_vol_df['Return']/ret_vol_df['Volatility']
    return ret_vol_df

def get_max_sharpe_port(sim_ports_df):
    
    # Maximum Sharpe Ratio
    # Max sharpe ratio portfolio 
    return sim_ports_df.iloc[sim_ports_df['sharpe_ratio'].idxmax()]
    
def get_max_sharpe_weights(sim_ports_df):
    
    # Max sharpe ratio portfolio weights
    return sim_ports_df['weights'][sim_ports_df['sharpe_ratio'].idxmax()]
    
    
def ceil(number, bound=5):
    return bound *np.ceil(number/bound)

def calculate_ytd_returns(month_ret_df, year = 2022):
    
    #make index a column
    month_ret_df = month_ret_df.reset_index()
    
    #get series of the years in the returns dataframe
    yrs = month_ret_df['index'].dt.year
    
    #find which years match current year
    current_year = yrs[yrs == year]
    #get returns that are in current year
    current_yr_ret = month_ret_df.loc[list(current_year.index)]
    #set index column back to index
    current_yr_ret.set_index('index', inplace = True)
    #current_yr_ret.drop(columns = 'level_0', inplace = True)
    
    #get prices of returns
    prices = dm.get_prices_df(current_yr_ret)
    
    #calculate year to date 
    ytd = pd.DataFrame(prices.iloc[-1,:]-1)
    ytd = ytd.transpose()
    
    return ytd
    

    