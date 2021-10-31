# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 21:26:55 2021

@author: Powis Forjoe
"""

import pandas as pd

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
    min_list = ['Volatility']
    
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

def get_plan_styler(df):
    
    
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
        else:
            formatter[col] = "{:.2%}"
    
    #return styler
    return df.style.\
        applymap(color_neg_red, subset = pd.IndexSlice[:,col_list]).\
        format(formatter)