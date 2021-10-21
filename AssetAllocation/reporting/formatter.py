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
    col_list.remove('Sharpe')
    col_list_1 = list(port_df.columns)[0:3]
    
    formatter = {}
    for col in col_list:
        if col == 'Sharpe':
            formatter[col] = "{:.4f}"
        else:
            formatter[col] = "{:.2%}"
    
    #return styler
    return port_df.style.\
        apply(highlight_max,subset = pd.IndexSlice[:,col_list_1]).\
        format(formatter)

def get_percent_styler(df):
    
    #define formatter
    col_list = list(df.columns)
    formatter = {}
    for strat in col_list:
        formatter[strat] = "{:.2%}"
    
    #return styler
    return df.style.\
        format(formatter)