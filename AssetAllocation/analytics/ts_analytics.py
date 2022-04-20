# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 17:05:16 2021

@author: Powis Forjoe
"""

import pandas as pd
import numpy as np
from ..datamanger import datamanger as dm
from .util import add_sharpe_col


def compute_ewcov(returns_df, x, y, decay_factor=.98,t=1):
    """
    return the sample exponential-weighted covariance between x & y at time t

    Parameters
    ----------
    returns_df : Dataframe
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
        return decay_factor * (returns_df.cov()[x][y]) + (1 - decay_factor) * (returns_df[x][-1:][0]*returns_df[y][-1:][0])
    else:
        return decay_factor * (returns_df[:-t].cov()[x][y]) + (1 - decay_factor) * (returns_df[x][-(t+1):-t][0] * returns_df[y][-(t+1):-t][0])

def compute_ewcorr(returns_df, x, y, decay_factor=.98,t=1):
    """
    return the sample exponential-weighted correlation between x & y at time t

    Parameters
    ----------
    returns_df : Dataframe
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
    temp_ewcov = compute_ewcov(returns_df, x, y, decay_factor, t)
    #compute sample ew var of x and y at t
    temp_ewvar_x = compute_ewcov(returns_df, x, x, decay_factor, t)
    temp_ewvar_y = compute_ewcov(returns_df, y, y, decay_factor, t)
    #update dataframe with ew corr value
    return temp_ewcov / np.sqrt(temp_ewvar_x * temp_ewvar_y)

def compute_ewcorr_matrix(returns_df, decay_factor=.98, t=1):
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
    ewcorr_df = returns_df.corr()
    
    # loop through values in correlation matrix to update values to ewcorr values
    for col in ewcorr_df.columns:
        for index in ewcorr_df.index:
            ewcorr_df[col][index] = compute_ewcorr(returns_df, col, index, decay_factor, t)
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

def get_ret_vol_df(returns_df):
    """
    Get a dataframe containig annualized return and volatility analytics of returns

    Parameters
    ----------
    returns_df : Dataframe
        returns dataframe.

    Returns
    -------
    ret_vol_df : Dataframe
        Dataframe of ret and vol for each column in returns_df.

    """
    ret_vol_dict = {}
    for col in returns_df.columns:
        ann_ret = get_ann_return(returns_df[col])
        ann_vol = get_ann_vol(returns_df[col])
        ret_vol_dict[col] = [ann_ret, ann_vol]
    ret_vol_df = pd.DataFrame(ret_vol_dict, index = ['Return', 'Volatility']).transpose()
    return add_sharpe_col(ret_vol_df)

 
