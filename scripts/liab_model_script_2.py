# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 21:05:08 2021

@author: Powis Forjoe
"""

############################################################################################################################################################
# IMPORT LIBRARIES                                                            
############################################################################################################################################################
from AssetAllocation.datamanger import datamanger as dm
from AssetAllocation.analytics.liability_model import liabilityModel
import pandas as pd
PLAN = 'IBT'
############################################################################################################################################################
# IMPORT CASHFLOW AND DISC RATE DATA                                                             
############################################################################################################################################################
df_pbo_cfs = dm.get_cf_data('PBO')
df_pvfb_cfs = dm.get_cf_data('PVFB')
df_sc_cfs = df_pvfb_cfs - df_pbo_cfs
df_ftse = dm.get_ftse_data(False)
disc_rates = pd.read_excel(dm.TS_FP+"discount_rate_data.xlsx",sheet_name=PLAN ,usecols=[0,1],index_col=0)
    
############################################################################################################################################################
# TRANSFORM DATA TO LIABILITY MODEL INPUTS                                                           
############################################################################################################################################################
pbo_cashflows = df_pbo_cfs[PLAN]
disc_factors = df_pbo_cfs['Time']
sc_cashflows = df_sc_cfs[PLAN]
liab_curve = dm.generate_liab_curve(df_ftse, pbo_cashflows)
asset_mv = dm.get_plan_asset_mv(PLAN)
contrb_pct = 0.00
liab_model = liabilityModel(pbo_cashflows, disc_factors, sc_cashflows, 
                                  contrb_pct, asset_mv,liab_curve)

yrs_to_ff = 20
ff_ratio = 1.05                    
liab_model.compute_fulfill_ret(yrs_to_ff, ff_ratio)
