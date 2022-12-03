# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 21:05:08 2021

@author: Powis Forjoe
"""

############################################################################################################################################################
# IMPORT LIBRARIES                                                            
############################################################################################################################################################
from AssetAllocation.datamanager import datamanager as dm
from AssetAllocation.analytics.liability_model import liabilityModel

############################################################################################################################################################
# IMPORT CASHFLOW & FTSE DATA REQUIRED FOR CREATING LIABILITY DATA                                                          
############################################################################################################################################################
df_pbo_cfs = dm.get_cf_data('PBO')
df_sc_cfs = dm.get_cf_data('Service Cost')
df_ftse = dm.get_ftse_data()
plan_asset_data = dm.get_plan_asset_data()
plan_mv_cfs_dict = dm.get_plan_mv_cfs_dict()

    
############################################################################################################################################################
# TRANSFORM DATA TO LIABILITY MODEL INPUTS                                                           
############################################################################################################################################################
plan = 'IBT'
pbo_cashflows = df_pbo_cfs[plan]
disc_factors = df_pbo_cfs['Time']
sc_cashflows = df_sc_cfs[plan]
liab_curve = dm.generate_liab_curve(df_ftse, pbo_cashflows)
contrb_pct=.05
asset_mv = dm.get_plan_asset_mv(plan_asset_data, plan)
liab_mv_cfs = dm.offset(plan_mv_cfs_dict[plan])
asset_ret = dm.get_plan_asset_returns(plan_asset_data, plan)

############################################################################################################################################################
# INITIALIZE LIAB MODEL USING CURVE AND DISC RATES
############################################################################################################################################################
liab_model = liabilityModel(pbo_cashflows, disc_factors, sc_cashflows, contrb_pct, asset_mv, liab_mv_cfs,asset_ret, liab_curve)

############################################################################################################################################################
# GET LIABILITY DATA DICT THAT CONTAINS PLAN DATA (ASSET&LIABILITY MVS & RETURNS, IRR, PVS, FUNDED STATUS)
############################################################################################################################################################
liab_data_dict = liab_model.get_liab_data_dict(pbo_cashflows, sc_cashflows)
