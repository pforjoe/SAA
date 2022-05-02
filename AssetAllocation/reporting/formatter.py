# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 21:26:55 2021

@author: Powis Forjoe
"""

import pandas as pd
from  ..datamanager import datamanager as dm

def highlight_max(s):
    """
    Highlight the maximum in a Series yellow

    Parameters
    ----------
    s : series

    Returns
    -------
    list

    """
    
    is_max = s == s.max()
    return ['background-color: yellow' if v else '' for v in is_max]

def highlight_min(s):
    """
    Highlight the maximum in a Series yellow

    Parameters
    ----------
    s : series

    Returns
    -------
    list

    """
    
    is_min = s == s.min()
    return ['background-color: yellow' if v else '' for v in is_min]

def get_port_styler(port_df):
    """
    Returns styler for historical selloffs dataframe

    Parameters
    ----------
    df_hist : dataframe
    Returns
    -------
    styler

    """
    
    #define formatter
    col_list = list(port_df.columns)
    max_list = ['Asset Return','Excess Return', 'Sharpe']
    min_list = ['Surplus Volatility']
    
    formatter = {}
    for col in col_list:
        if col == 'Sharpe':
            formatter[col] = "{:.4f}"
        else:
            formatter[col] = "{:.2%}"
    
    #return styler
    return port_df.style.\
        applymap(color_neg_red, subset = pd.IndexSlice[:,col_list]).\
        apply(highlight_max,subset = pd.IndexSlice[:,max_list]).\
        apply(highlight_min,subset = pd.IndexSlice[:,min_list]).\
        format(formatter)

def color_neg_red(val):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise.

    Parameters
    ----------
    val : float

    Returns
    -------
    string

    """
    color = 'red' if val < 0 else 'black'
    return 'color: %s' % color

def get_plan_styler(df, returns = True):
    
    
    try:
        df.index = pd.to_datetime(df.index, format = '%m/%d/%Y').strftime('%Y-%m-%d')
    except ValueError:
        pass
    #define formatter
    col_list = list(df.columns)
    formatter = {}
    for col in col_list:
        if col == 'Sharpe':
            formatter[col] = "{:.4f}"
        if returns:
            formatter[col] = "{:.2%}"
        else:
            formatter[col] = "${:,.2f}"
    
    #return styler
    return df.style.\
        applymap(color_neg_red, subset = pd.IndexSlice[:,col_list]).\
        format(formatter)

def reset_bnds(df_bnds, plan):
    new_bnds = dm.get_bounds(plan='IBT')
    
    for asset in df_bnds.index:
        df_bnds['Lower'][asset] = new_bnds['Lower'][asset]
        df_bnds['Upper'][asset] = new_bnds['Upper'][asset]
    return None

def reset_asset_bnds(df_bnds, asset, plan):
    new_bnds = dm.get_bounds(plan='IBT')
    #View bounds
    df_bnds['Lower'][asset] = new_bnds['Lower'][asset]
    df_bnds['Upper'][asset] = new_bnds['Upper'][asset]
    return None


def update_upper_bnds(df_bnds, asset, upper, plan):
    upper_value = (float(upper.split("%")[0])/100)*plan.funded_status
    df_bnds['Upper'][asset] = upper_value
    return None

def update_lower_bnds(df_bnds, asset, lower, plan):
    lower_value = (float(lower.split("%")[0])/100)*plan.funded_status
    if (lower_value) < df_bnds['Upper'][asset]:
        df_bnds['Lower'][asset] = lower_value
    return None


def get_fs_data_styler(df):
    try:
        df.index = pd.to_datetime(df.index, format = '%m/%d/%Y').strftime('%Y-%m-%d')
    except ValueError:
        pass
    #define formatter
    col_list = list(df.columns)
    formatter = {}
    formatter[col_list[0]] = "${:,.2f}"
    formatter[col_list[1]] = "${:,.2f}" 
    formatter[col_list[2]] = "{:.2%}"
    formatter[col_list[3]] = "${:,.2f}"
    formatter[col_list[4]] = "{:.2%}"
    formatter[col_list[5]] = "{:.2%}"

    #return styler
    return df.style.\
        applymap(color_neg_red, subset = pd.IndexSlice[:,col_list]).\
        format(formatter)
        
        
def get_pv_irr_styler(df):
    try:
        df.index = pd.to_datetime(df.index, format = '%m/%d/%Y').strftime('%Y-%m-%d')
    except ValueError:
        pass
    #define formatter
    col_list = list(df.columns)
    formatter = {}
    formatter[col_list[0]] = "${:,.2f}" 
    formatter[col_list[1]] = "{:.2%}"

    #return styler
    return df.style.\
        applymap(color_neg_red, subset = pd.IndexSlice[:,col_list]).\
        format(formatter)