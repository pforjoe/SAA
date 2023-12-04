# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 11:44:59 2022

@author: maddi
"""

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
import AssetAllocation.reporting.sheets as sheets
import AssetAllocation.reporting.reports as rp

import pandas as pd
import numpy as np
from scipy.optimize import fsolve

############################################################################################################################################################
# FUNCTIONS FOR COMPUTING LIAB RETURN                                                         
############################################################################################################################################################
def compute_pvs(pbo_cashflows, disc_factors, liab_curve=pd.DataFrame,disc_rates=pd.DataFrame):
    if disc_rates.empty:
        pv_dict={}
        for col in liab_curve.columns:
            temp_pv = 0
            for j in range (0,len(pbo_cashflows)):
                temp_pv += (pbo_cashflows[j]/((1+liab_curve[col][j]/100)**disc_factors[j]))
            pv_dict[col] = temp_pv
        return pd.DataFrame(pv_dict, index = ['Present Value']).transpose()
    else:
        disc_rates_pv_array = np.zeros(len(disc_rates))
        for i in range(len(disc_rates)):
            for j in range (0,len(pbo_cashflows)):
                disc_rates_pv_array[i] += (pbo_cashflows[j]/((1+disc_rates['IRR'][i])**disc_factors[j]))
            
        return pd.DataFrame(disc_rates_pv_array, columns=['Present Value'], index=disc_rates.index)
    
def npv(irr, cfs, yrs):  
    return np.sum(cfs / (1. + irr) ** yrs)

def irr(cfs, yrs, x0, **kwargs):
    return np.ndarray.item(fsolve(npv, x0=x0,args=(cfs,yrs), **kwargs))

def compute_irr(present_values,pbo_cashflows, disc_factors):
    irr_array = np.zeros(len(present_values))
    for j in range (len(present_values)):
        cashflows = np.append(np.negative(present_values['Present Value'][j]),pbo_cashflows)
        yrs = np.append(0, disc_factors)
        irr_array[j] += irr(cashflows, yrs, .03)
    return pd.DataFrame(irr_array, columns=['IRR'], index=present_values.index)

def compute_liab_ret(present_values, irr_df):
    liab_ret = np.zeros(len(present_values))

    for i in range (0,len(present_values)-1):
        liab_ret[i+1] += irr_df['IRR'][i]/12 + ((present_values['Present Value'][i+1])/present_values['Present Value'][i])-1
        
    return pd.DataFrame(liab_ret, columns=['Liability'], index=present_values.index)
    

############################################################################################################################################################
# IMPORT AND TRANSFORM CASHFLOW                                                            
############################################################################################################################################################

plans = ['Retirement', "Pension","IBT","Total"]
df_pbo_cfs = dm.get_cf_data('PBO')
df_pbo_cfs["Total"] =  df_pbo_cfs["IBT"] + df_pbo_cfs["Retirement"] + df_pbo_cfs["Pension"]

df_pvfb_cfs = dm.get_cf_data('PVFB')
df_pvfb_cfs["Total"] =  df_pvfb_cfs["IBT"] + df_pvfb_cfs["Retirement"] + df_pvfb_cfs["Pension"]

df_sc_cfs = df_pvfb_cfs - df_pbo_cfs
df_ftse = dm.get_ftse_data()

#disc_rates_dict = {}
#for PLAN in plans:
    #disc_rates_dict[PLAN] = pd.read_excel(dm.TS_FP+"discount_rate_data_towers.xlsx",sheet_name=PLAN ,usecols=[0,1],index_col=0)

############################################################################################################################################################
# TRANSFORM DATA TO LIABILITY MODEL INPUTS                                                           
############################################################################################################################################################

tables = {}

for PLAN in plans:
    pbo_cashflows = df_pbo_cfs[PLAN]
    disc_factors = df_pbo_cfs['Time']
    sc_cashflows = df_sc_cfs[PLAN]
    liab_curve = dm.generate_liab_curve(df_ftse, pbo_cashflows)
    #disc_rates = disc_rates_dict[PLAN] 
    asset_mv = dm.get_plan_asset_mv(dm.get_plan_asset_data(),PLAN)
    contrb_pct = 1.00
    
    ############################################################################################################################################################
    # COMPUTE PV, IRR, LIAB RET USING CURVE AND DISC RATES                                                         
    ############################################################################################################################################################
    pv_curve = compute_pvs(pbo_cashflows, disc_factors, liab_curve)
    irr_curve = compute_irr(pv_curve, pbo_cashflows, disc_factors)
    liab_ret_curve = compute_liab_ret(pv_curve, irr_curve)
    
    # pv_disc_rates = compute_pvs(pbo_cashflows, disc_factors, disc_rates=disc_rates)
    # irr_disc_rates = compute_irr(pv_disc_rates, pbo_cashflows, disc_factors)
    # liab_ret_disc_rates = compute_liab_ret(pv_disc_rates, irr_disc_rates)
    
    #
    if PLAN == "Total":
        tables[PLAN] = liab_ret_curve
    else:
        plan_returns = pd.read_excel(dm.TS_FP+"plan_return_data.xlsx",sheet_name = PLAN ,usecols=[0,1],index_col=0)
        tables[PLAN] = liab_ret_curve.merge(plan_returns, how = "left", left_index= True, right_index = True)

filepath = rp.get_reportpath("test")
writer = pd.ExcelWriter(filepath, engine='xlsxwriter')
for plan in plans: 
    sheets.set_asset_liability_charts_sheet(writer, tables, sheet_name = "charts_ftse")
writer.save()  

############################################################################################################################################################
# INITIALIZE LIAB MODEL USING CURVE AND DISC RATES
############################################################################################################################################################
liab_model_curve = liabilityModel(pbo_cashflows, disc_factors, sc_cashflows, contrb_pct, asset_mv,liab_curve)

#liab_model_disc_rates = liabilityModel(pbo_cashflows, disc_factors, sc_cashflows, contrb_pct, asset_mv,disc_rates=disc_rates)
