# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 17:05:16 2021

@author: Bloomberg
"""

import pandas as pd
import os
import numpy as np

#set filepath
cwd = os.getcwd()
data_fp = cwd + '\\data\\'


def compute_ewcov(df, x, y, decay_factor=.98,t=1):
    """
    return the sample exponential-weighted covariance between x & y at time t

    Parameters
    ----------
    df : Dataframe
        returns dataframe.
    x : string
        column in dataframe.
    y : string
        column in dataframe.
    decay_factor : float, optional
        Decay factor. The default is .98.
    t : int, optional
        time. The default is 1.

    Returns
    -------
    double
        exponential-weighted covariance.

    """
    if t <= 0:
        return decay_factor * (df.cov()[x][y]) + (1 - decay_factor) * (df[x][-1:][0]*df[y][-1:][0])
    else:
        return decay_factor * (df[:-t].cov()[x][y]) + (1 - decay_factor) * (df[x][-(t+1):-t][0] * df[y][-(t+1):-t][0])

def compute_ewvol(df,x, decay_factor=.98, t=1):
    """
    return the sample exponential-weighted volatility for time series x at time t

    Parameters
    ----------
    df : Dataframe
        returns dataframe.
    x : string
        column in dataframe.
    decay_factor : float, optional
        Decay factor. The default is .98.
    t : int, optional
        time. The default is 1.

    Returns
    -------
    float
        exponential-weighted volatility.

    """
    if t <= 0:
        return np.sqrt(decay_factor * (df[x].var()) + (1 - decay_factor) * (df[x][-1:][0]**2))
    else:
        return np.sqrt(decay_factor * (df[x][:-t].var()) + (1 - decay_factor) * (df[x][-(t+1):-t][0]**2))

def compute_ewcorr(df, x, y, decay_factor=.98,t=1):
    """
    return the sample exponential-weighted correlation between x & y at time t

    Parameters
    ----------
    df : Dataframe
        returns dataframe.
    x : string
        column in dataframe.
    y : string
        column in dataframe.
    decay_factor : float, optional
        Decay factor. The default is .98.
    t : int, optional
        time. The default is 1.

    Returns
    -------
    double
        exponential-weighted correlation.

    """
    #compute sample ew cov between x and y at t
    temp_ewcov = compute_ewcov(df, x, y, decay_factor, t)
    #compute sample ew var of x and y at t
    temp_ewvar_x = compute_ewcov(df, x, x, decay_factor, t)
    temp_ewvar_y = compute_ewcov(df, y, y, decay_factor, t)
    #update dataframe with ew corr value
    return temp_ewcov / np.sqrt(temp_ewvar_x * temp_ewvar_y)

def compute_ewcorr_matrix(df, decay_factor=.98, t=1):
    """
    return the sample exponential-weighted pairwise correlation of columnsin df at time t

    Parameters
    ----------
    df : Dataframe
        returns dataframe.
    decay_factor : float, optional
        Decay factor. The default is .98.
    t : int, optional
        time. The default is 1.

    Returns
    -------
    Dataframe
        exponential-weighted correlation matrix.

    """
    #create correlation matrix
    ewcorr_df = df.corr()
    
    # loop through values in correlation matrix to update values to ewcorr values
    for col in ewcorr_df.columns:
        for index in ewcorr_df.index:
            ewcorr_df[col][index] = compute_ewcorr(df, col, index, decay_factor, t)
    return ewcorr_df

def get_ann_return(return_series):
    """
    Return annualized return for a montly return series.

    Parameters
    ----------
    return_series : series
        returns series.
    
    Returns
    -------
    double
        Annualized return.

    """
    #compute the annualized return
    d = len(return_series)
    return return_series.add(1).prod()**(12/d)-1

def get_ann_vol(return_series):
    """
    Return annualized volatility for a monthly return series.

    Parameters
    ----------
    return_series : series
        returns series.
    
    Returns
    -------
    double
        Annualized volatility.

    """
    #compute the annualized volatility
    return np.std(return_series, ddof=1)*np.sqrt(12)

#import and clean data
ret_data = pd.read_excel(data_fp+'return_data.xlsx', sheet_name='data_2010', index_col=0)
ret_data.index = pd.to_datetime(ret_data.index)
ret_data = ret_data.resample('1M').ffill()
ret_data = ret_data.pct_change()
ret_data.dropna(inplace=True)

df = ret_data.copy()

df['Credit'] = 0.5*df['CS LL'] + 0.5*df['BOA HY']
df['Liq Alts'] = 0.4*df['HF MACRO'] + 0.3*df['HFRI MACRO'] + 0.1*df['TREND'] + 0.2*df['ALT RISK']


ret_vol_dict = {}
for col in df.columns:
    ann_ret = get_ann_return(df[col])
    ann_vol = get_ann_vol(df[col])
    ret_vol_dict[col] = [ann_ret, ann_vol]
    
#Converts hedge_dict to a data grame
df_returns_stats = pd.DataFrame(ret_vol_dict, index = ['Return', 'Vol']).transpose()
# return df_returns_stats
#compute ew corr

corr_df = compute_ewcorr_matrix(df)

