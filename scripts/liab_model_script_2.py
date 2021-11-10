# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 21:05:08 2021

@author: NVG9HXP
"""

############################################################################################################################################################
# IMPORT LIBRARIES                                                            
############################################################################################################################################################
from AssetAllocation.datamanger import datamanger as dm
from itertools import count, takewhile

import scipy as sp
from scipy.optimize import fsolve
import pandas as pd
import numpy as np

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# IBT or Pension or Retirement
PLAN = 'IBT'

UPS_Contr_Pctg = 0
############################################################################################################################################################
def frange(start, stop, step):
    return takewhile(lambda x: x< stop, count(start, step))

def set_cfs_time_col(df_cfs):
    df_cfs['Time'] = list(frange(1/12, (len(df_cfs)+.9)/12, 1/12))

def get_cf_data(cf_type='PBO',plan='IBT'):
    df_cfs = pd.read_excel(dm.TS_FP+'annual_cashflows_data.xlsx', sheet_name='PBO', index_col=0)/12
    df_cfs = dm.reindex_to_monthly_data(df_cfs)
    temp_cfs = pd.read_excel(dm.TS_FP+'monthly_cashflows_data.xlsx', sheet_name='PBO', index_col=0)
    df_cfs = temp_cfs.append(df_cfs)
    set_cfs_time_col(df_cfs)
    return df_cfs

def get_ftse_data():
    df_old_ftse = pd.read_excel(dm.TS_FP+'ftse_data.xlsx',sheet_name='old_data', index_col=0)
    df_new_ftse = pd.read_excel(dm.TS_FP+'ftse_data.xlsx',sheet_name='new_data', index_col=0)
    df_ftse = dm.merge_dfs(df_new_ftse,df_old_ftse)
    df_ftse.reset_index(inplace=True)
    return df_ftse

def generate_liab_curve(df_ftse, cfs):
    liab_curve_dict = {}
    dates = df_ftse['Date']
    range_list =  list(frange(0.5, dates[len(dates)-1]+.9/12, (1/12)))
    
    for col in df_ftse.columns:
        y = []
        interp = sp.interpolate.interp1d(dates, df_ftse[col], bounds_error=False,
                                         fill_value=sp.nan)
        for step in range_list:
                value = float(interp(step))
                if not sp.isnan(value): # Don't include out-of-range values
                    y.append(value)
                    end_rate = [y[-1]] * (len(cfs) - len(range_list)-5)
                    start_rate = [y[0]] * 5
                liab_curve_dict[col] = start_rate + y + end_rate
    liab_curve_dict.pop('Date')
    liab_curve = pd.DataFrame(liab_curve_dict)
    liab_curve = liab_curve.iloc[:, ::-1]
    return liab_curve

def compute_pvs(cfs,dfs,liab_curve):
    pv_dict={}
    for col in liab_curve.columns:
        temp_pv = 0
        for j in range (0,len(cfs)):
            temp_pv += (cfs[j]/((1+liab_curve[col][j]/100)**dfs[j]))
        pv_dict[col] = temp_pv
        
    return pd.DataFrame(pv_dict, index = ['Present Value']).transpose()

def compute_liab_ret(pv_df):
    liab_ret = np.zeros(len(pv_df))
    for i in range (0,len(pv_df)-1):
        liab_ret[i+1] = ((pv_df['Present Value'][i+1])/pv_df['Present Value'][i])-1
    
    return pd.DataFrame(liab_ret, columns=['Liability'], index=pv_df.index)

def compute_pvs_w_dr(cfs,dfs,df_dr):
    pv_dr_list = np.zeros(len(df_dr))
    for i in range(len(df_dr)):
        for j in range (0,len(cfs)):
            pv_dr_list[i] += (cfs[j]/((1+df_dr['IRR'][i])**dfs[j]))
        
    return pd.DataFrame(pv_dr_list, columns=['Present Value'], index=df_dr.index)

def npv(irr, cfs, yrs):  
    return np.sum(cfs / (1. + irr) ** yrs)

def irr(cfs, yrs, x0, **kwargs):
    return np.asscalar(fsolve(npv, x0=x0, args=(cfs, yrs), **kwargs))

def compute_irr(cfs,dfs,df_pv_dr):
    irr_list = np.zeros(len(df_pv_dr))
    for j in range (len(df_pv_dr)):
        cashflows = np.append(np.negative(df_pv_dr['Present Value'][j]),cfs)
        yrs = np.append(0, dfs)
        irr_list[j] += irr(cashflows, yrs, .03)
    return irr_list

def full_solve(initial_mv, df_dr, cfs,dfs,yrs_to_ff, ff_ratio,x0):
    return np.asscalar(fsolve(fullfill_solve, 
                              x0=x0, 
                              args=(initial_mv, df_dr, cfs,dfs,yrs_to_ff, ff_ratio)))


def fullfill_solve(fullfill_return, initial_mv, df_dr, cfs,dfs,yrs_to_ff, ff_ratio):
    #fullfillment_return = .045
    # MV_Assets = 13203547000
    pv_df_erf = np.zeros(len(dfs))
    asset_mv = np.zeros(len(dfs))
    x = yrs_to_ff/dfs[0]
    x = x.astype(int)
    
    for j in range(len(dfs)):
        if (j == 0):
            for i in range(j,len(cfs)):
                pv_df_erf[j] += (cfs[i]/((1+df_dr['IRR'][-1])**dfs[i-j]))
                asset_mv[j] = initial_mv
    
        else:
            for i in range(j,len(cfs)):
                pv_df_erf[j] += (cfs[i]/((1+df_dr['IRR'][-1])**dfs[i-j]))
                asset_mv[j] = (asset_mv[j-1]*(1+fullfill_return)**dfs[0].tolist())-cfs[j-1]
    
    return asset_mv[x] - pv_df_erf[x]*ff_ratio

############################################################################################################################################################
# IMPORT PBO CASHFLOWS                                                            
############################################################################################################################################################
df_pbo_cfs = get_cf_data('PBO')
pbo_cfs = np.array(df_pbo_cfs[PLAN])
dfs = np.array(df_pbo_cfs['Time'])

############################################################################################################################################################
# IMPORT FTSE CURVES AND INTERPOLATE                                                            
############################################################################################################################################################
df_ftse = get_ftse_data()

liab_curve = generate_liab_curve(df_ftse, pbo_cfs)

############################################################################################################################################################
# COMPUTE PVS && LIAB RET                                                            
############################################################################################################################################################
df_pv = compute_pvs(pbo_cfs, dfs, liab_curve)
liab_ret_df = compute_liab_ret(df_pv)
liab_ret_df = dm.merge_dfs(liab_ret_df, df_pv)

############################################################################################################################################################
# IMPORT SC CASHFLOWS                                                            
############################################################################################################################################################
df_sc_cfs = get_cf_data('SC')
sc_cfs = np.array(df_sc_cfs[PLAN])

total_cfs = UPS_Contr_Pctg*sc_cfs + pbo_cfs
############################################################################################################################################################
# COMPUTE PVS && LIAB RET                                                            
############################################################################################################################################################
df_total_pv = compute_pvs(total_cfs, dfs, liab_curve)
total_liab_ret_df = compute_liab_ret(df_total_pv)
total_liab_ret_df = dm.merge_dfs(total_liab_ret_df, df_total_pv)
############################################################################################################################################################
df_dr = pd.read_excel(dm.TS_FP+"discount_rate_data.xlsx",sheet_name=PLAN ,index_col=0)
df_pv_dr = compute_pvs_w_dr(pbo_cfs, dfs, df_dr)
############################################################################################################################################################
irr_list = compute_irr(pbo_cfs, dfs, df_pv_dr)
############################################################################################################################################################

fullfill_irr = full_solve(13203547000,df_dr,pbo_cfs,dfs,20,1.05,.01)