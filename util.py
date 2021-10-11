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

def clean_data(filename, year= '2011'):
    ret_data = pd.read_excel(data_fp+ filename, sheet_name=year, index_col=0)
    
    returns_df = ret_data.copy()
    
    returns_df['Credit'] = 0.5*returns_df['CS LL'] + 0.5*returns_df['BOA HY']
    returns_df['Liquid Alternatives'] = 0.4*returns_df['HF MACRO'] + 0.3*returns_df['HFRI MACRO'] + 0.1*returns_df['TREND'] + 0.2*returns_df['ALT RISK']
    returns_df = returns_df[['Liability', '15+ STRIPS', 'Long Corps', 'Total EQ Unhedged','Liquid Alternatives','Total Private Equity','Credit', 'Total Real Estate', 'Total UPS Cash','ULTRA 30Y FUTURES','Equity Hedges' ]]
    returns_df.columns = ['Liability', '15+ STRIPS', 'Long Corporate','Equity','Liquid Alternatives','Private Equity','Credit', 'Real Estate','Cash' , 'Ultra 30-Year UST Futures','Equity Hedges']
    return returns_df

def get_max_sharpe_port(sim_ports_df):
    
    # Maximum Sharpe Ratio
    # Max sharpe ratio portfolio 
    return sim_ports_df.iloc[sim_ports_df['sharpe_ratio'].idxmax()]
    
def get_max_sharpe_weights(sim_ports_df):
    
    # Max sharpe ratio portfolio weights
    return sim_ports_df['weights'][sim_ports_df['sharpe_ratio'].idxmax()]
    
    
