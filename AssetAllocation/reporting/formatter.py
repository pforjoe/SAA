# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 21:26:55 2021

@author: NVG9HXP
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
    max_list = ['Return', 'Sharpe']
    min_list = ['Volatility']
    
    formatter = {}
    for col in col_list:
        if col == 'Sharpe':
            formatter[col] = "{:.4f}"
        else:
            formatter[col] = "{:.2%}"
    
    #return styler
    return port_df.style.\
        apply(highlight_max,subset = pd.IndexSlice[:,max_list]).\
        apply(highlight_min,subset = pd.IndexSlice[:,min_list]).\
        format(formatter)

def get_ret_vol_styler(df):
    
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
        format(formatter)