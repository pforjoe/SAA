# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 16:22:39 2021

@authors: Roxton McNeal, Matt Johnston, Powis Forjoe
"""

# Import pandas
# import pandas as pd
import numpy as np
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