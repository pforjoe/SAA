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


def compute_sample_ewcov(df):
    return df[:-1].cov()

def compute_sample_cov(df, x, y, decay_factor=.98,t=1):
    if t == 0:
        return (decay_factor * df.cov()[x][y]) + (1 - decay_factor)*(df[x][-1:][0]*df[y][-1:][0])
    else:
        return (decay_factor * df[:-t].cov()[x][y]) + (1 - decay_factor)*(df[x][-(t+1):-t][0]*df[y][-(t+1):-t][0])

def compute_sample_vol(df,x, decay_factor=.98, t=1):
    if t == 0:
        return np.sqrt((decay_factor * df[x].var()) + (1 - decay_factor)*(df[x][-1:][0]**2))
    else:
        return np.sqrt((decay_factor * df[x][:-t].var()) + (1 - decay_factor)*(df[x][-(t+1):-t][0]**2))

def compute_ew_corr(df, decay_factor=.98):
    corr_df = df.corr()
    for col in corr_df.columns:
        for index in corr_df.index:
            temp_ew_cov = compute_sample_cov(df, col, index)
            temp_ew_vol_x = compute_sample_vol(df, col)
            temp_ew_vol_y = compute_sample_vol(df, index)
            corr_df[col][index] = temp_ew_cov / (temp_ew_vol_x*temp_ew_vol_y)
    return corr_df


#import and clean data
ret_data = pd.read_excel(data_fp+'index_data.xlsx', sheet_name='data', index_col=0)
ret_data.index = pd.to_datetime(ret_data.index)
ret_data = pd.resample('1M').ffill()
ret_data = ret_data.resample('1M').ffill()
ret_data = ret_data.pct_change()
ret_data.dropna(inplace=True)

df = ret_data.copy()

#compute ew corr

corr_df = compute_ew_corr(df)
