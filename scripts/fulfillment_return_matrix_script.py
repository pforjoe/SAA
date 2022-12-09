# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 21:05:08 2021

@author: NVG9HXP
"""

############################################################################################################################################################
# IMPORT LIBRARIES                                                            
############################################################################################################################################################
from AssetAllocation.datamanager import datamanager as dm
from AssetAllocation.analytics.liability_model import liabilityModel
from AssetAllocation.reporting import reports as rp
import numpy as np
import pandas as pd
import os
PLAN = 'IBT'
############################################################################################################################################################
# IMPORT CASHFLOW AND DISC RATE DATA                                                             
############################################################################################################################################################
df_pbo_cfs = dm.get_cf_data('PBO')
df_pvfb_cfs = dm.get_cf_data('PVFB')
df_sc_cfs = df_pvfb_cfs - df_pbo_cfs
df_ftse = dm.get_ftse_data()
disc_rates = pd.read_excel(dm.TS_FP+"discount_rate_data.xlsx",sheet_name=PLAN ,usecols=[0,1],index_col=0)
    
############################################################################################################################################################
# TRANSFORM DATA TO LIABILITY MODEL INPUTS                                                           
############################################################################################################################################################
pbo_cashflows = df_pbo_cfs[PLAN]
disc_factors = df_pbo_cfs['Time']
sc_cashflows = df_sc_cfs[PLAN]
liab_curve = dm.generate_liab_curve(df_ftse, pbo_cashflows)
asset_mv = dm.get_plan_asset_mv(PLAN)
              
plan_list = ['IBT', 'Pension', 'Retirement']
contrb_pct_list = [0.0,0.05,0.1,0.5,1.0]
yrs_to_ff_list = list(range(10,35,10))
ff_ratio_list = [1.0,1.05, 1.1]

fulfill_ret_dict = {}
for plan in plan_list:
    print('Plan: {}'.format(plan))
    ff_ratio_dict = {}
    pbo_cashflows = np.array(df_pbo_cfs[plan])
    sc_cashflows = np.array(df_sc_cfs[plan])
    disc_rates = pd.read_excel(dm.TS_FP+"discount_rate_data.xlsx",sheet_name=plan ,usecols=[0,1],index_col=0)
    asset_mv = dm.get_plan_asset_mv(plan)
    for ff_ratio in ff_ratio_list:
        print('Full funded ratio: {}'.format(str(ff_ratio)))
        ff_ratio_df = dm.pd.DataFrame(index = yrs_to_ff_list, columns=contrb_pct_list)
        ff_ratio_df.index.names = ['Years to Fully Funded']
        for contrb_pct in contrb_pct_list:
            print('Contrib Pct: {}'.format(contrb_pct))
            liab_model = liabilityModel(pbo_cashflows, disc_factors, sc_cashflows,
                                              liab_curve,disc_rates,contrb_pct, asset_mv)
            for yrs_to_ff in yrs_to_ff_list:
                print('Year to ff: {}'.format(yrs_to_ff))
                liab_model.compute_fulfill_ret(yrs_to_ff, ff_ratio)
                ff_ratio_df[contrb_pct][yrs_to_ff] = liab_model.fulfill_irr
        ff_ratio_df.columns = [str(np.around(x,2)) for x in contrb_pct_list]
        ff_ratio_dict[str(np.around(ff_ratio*100,0))+'%'] = ff_ratio_df
    fulfill_ret_dict[plan] = ff_ratio_dict
            
rp.get_ff_report('fulfill_matrix', fulfill_ret_dict, plan_list)             
