# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 21:05:08 2021

@author: Powis Forjoe
"""
############################################################################################################################################################
# RUN NEXT 2 LINES IF YOU WANT TO RUN THE WHOLE SCRIPT AT ONCE                                                           
############################################################################################################################################################
import os
os.chdir("..")

############################################################################################################################################################
# IMPORT LIBRARIES                                                            
############################################################################################################################################################
from AssetAllocation.datamanager import datamanager as dm
from AssetAllocation.analytics.liability_model import liabilityModel
from AssetAllocation.analytics.liability_model_new import liabilityModelNew
from AssetAllocation.reporting import reports as rp
import numpy as np
############################################################################################################################################################
# IMPORT CASHFLOW & FTSE DATA REQUIRED FOR CREATING LIABILITY DATA                                                          
############################################################################################################################################################
liab_inputs_data_dict = dm.get_liab_data_inputs()
df_pbo_cfs = liab_inputs_data_dict['df_pbo_cfs']
df_sc_cfs = liab_inputs_data_dict['df_sc_cfs']
liab_curve = liab_inputs_data_dict['liab_curve']
plan_asset_data = liab_inputs_data_dict['asset_data']
# plan_mv_cfs_dict = dm.get_plan_mv_cfs_dict()
plan_pbo_cfs_dict = liab_inputs_data_dict['pbo_cfs_dict']
plan_sc_cfs_dict = liab_inputs_data_dict['sc_cfs_dict']

############################################################################################################################################################
# SET VARIABLES TO LOOP THROUGH (PLAN, CONTRIBUTION_PCT, YRS_TO_FF, FF_RATIO)                                                           
############################################################################################################################################################

plan_list = ['IBT', 'Pension', 'Retirement']
# contrb_pct_list = [0.0,0.05,0.1,0.5,1.0]
contrb_pct_list = [0.0,0.5,1.0]
# yrs_to_ff_list = list(range(10,35,10))
yrs_to_ff_list = [10,20]
# ff_ratio_list = [1.0, 1.05, 1.1]
ff_ratio_list = [1.0,1.1]

############################################################################################################################################################
# LOOP THROUGH VARIABLES TO CREATE FULFILLMEMT RETURN MATRIX                                                           
############################################################################################################################################################
disc_factors = df_pbo_cfs['Time']

fulfill_ret_dict = {}
for plan in plan_list:
    print('Plan: {}'.format(plan))
    ff_ratio_dict = {}
    pbo_cashflows = df_pbo_cfs[plan]
    sc_cashflows = df_sc_cfs[plan]
    # liab_curve = dm.generate_liab_curve(df_ftse, pbo_cashflows)
    asset_mv = dm.get_plan_asset_mv(plan_asset_data, plan)
    # liab_mv_cfs = dm.offset(plan_mv_cfs_dict[plan])
    asset_ret = dm.get_plan_asset_returns(plan_asset_data, plan)
    plan_pbo_cfs = plan_pbo_cfs_dict[plan]
    plan_sc_cfs = plan_sc_cfs_dict[plan]
    for ff_ratio in ff_ratio_list:
        print('Full funded ratio: {}'.format(str(ff_ratio)))
        ff_ratio_df = dm.pd.DataFrame(index = yrs_to_ff_list, columns=contrb_pct_list)
        ff_ratio_df.index.names = ['Years to Fully Funded']
        for contrb_pct in contrb_pct_list:
            print('Contrib Pct: {}'.format(contrb_pct))
            liab_model = liabilityModelNew(plan, pbo_cashflows, disc_factors, sc_cashflows, contrb_pct, 
                                        plan_pbo_cfs,plan_sc_cfs,asset_mv,asset_ret, liab_curve)
            for yrs_to_ff in yrs_to_ff_list:
                print('Year to ff: {}'.format(yrs_to_ff))
                liab_model.compute_fulfill_ret(yrs_to_ff, ff_ratio)
                ff_ratio_df[contrb_pct][yrs_to_ff] = liab_model.fulfill_irr
        ff_ratio_df.columns = [str(np.around(x,2)) for x in contrb_pct_list]
        ff_ratio_dict[str(np.around(ff_ratio*100,0))+'%'] = ff_ratio_df
    fulfill_ret_dict[plan] = ff_ratio_dict
            
##################################################################


##########################################################################################
# GENERATE FULLFILLMENT RETURN MATRIX REPORT                                                           
############################################################################################################################################################
rp.get_ff_report('fulfill_matrix-2', fulfill_ret_dict, plan_list)             


