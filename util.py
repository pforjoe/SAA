# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 16:22:39 2021

@authors: Roxton McNeal, Matt Johnston, Powis Forjoe
"""

# Import pandas
# import pandas as pd
import numpy as np
from inputs_for_mv import get_summarized_output
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

def add_dimension(data):
    return data.to_numpy()[:,np.newaxis]

def get_data(filename):
    dataset = get_summarized_output(filename)
    
    policy_wts = add_dimension(dataset['FS AdjWeights'])
    
    ret = dataset['Return']

    vol = add_dimension(dataset['Vol'])
    
    corr = dataset.iloc[:,3:].to_numpy()
    
    symbols =  list(dataset.index.values)
    
    return {'policy_weights': policy_wts,'ret':ret, 'vol':vol, 'corr':corr,'symbols':symbols}

def get_data_dict(dataset):
    policy_wts = add_dimension(dataset['FS AdjWeights'])
    
    ret = dataset['Return']

    vol = add_dimension(dataset['Vol'])
    
    corr = dataset.iloc[:,3:].to_numpy()
    
    symbols =  list(dataset.index.values)
    
    return {'policy_weights': policy_wts,'ret':ret, 'vol':vol, 'corr':corr,'symbols':symbols}


def get_max_sharpe_port(sim_ports_df):
    
    # Maximum Sharpe Ratio
    # Max sharpe ratio portfolio 
    return sim_ports_df.iloc[sim_ports_df['sharpe_ratio'].idxmax()]
    
def get_max_sharpe_weights(sim_ports_df):
    
    # Max sharpe ratio portfolio weights
    return sim_ports_df['weights'][sim_ports_df['sharpe_ratio'].idxmax()]
    
    
