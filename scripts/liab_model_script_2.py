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
PLAN = 'Retirement'
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
contrb_pct = 0.05
liab_model = liabilityModel(pbo_cashflows, disc_factors, sc_cashflows, 
                                  liab_curve,disc_rates,contrb_pct, asset_mv)

yrs_to_ff = 20
ff_ratio = 1.05                    
liab_model.compute_fulfill_ret(yrs_to_ff, ff_ratio)

import numpy as np

irr_list = np.zeros(len(liab_model.present_values))

for j in range (len(liab_model.disc_rates_pvs)):
    print(j)
    cashflows = np.append(np.negative(liab_model.disc_rates_pvs['Present Value'][j]),liab_model.total_cashflows)
    yrs = np.append(0, liab_model.disc_factors)
    irr_list[j+171] += liab_model.irr(cashflows, yrs, .03)/12
    
from scipy.optimize import fsolve
# import pandas as pd
import numpy as np

def compute_pvs(liab_curve,total_cashflows,disc_factors):
    pv_dict={}
    for col in liab_curve.columns:
        temp_pv = 0
        for j in range (0,len(total_cashflows)):
            temp_pv += (total_cashflows[j]/((1+liab_curve[col][j]/100)**disc_factors[j]))
        pv_dict[col] = temp_pv
    return pd.DataFrame(pv_dict, index = ['Present Value']).transpose()

def compute_liab_ret(present_values,irr_array):
    liab_ret = np.zeros(len(present_values))
    irr_ret = transform_irr_array(present_values, irr_array)
    for i in range (0,len(present_values)-1):
        liab_ret[i+1] += irr_ret[i+1]+((present_values['Present Value'][i+1])/present_values['Present Value'][i])-1
    
    return pd.DataFrame(liab_ret, columns=['Liability'], index=present_values.index)

def transform_irr_array(present_values, irr_array):
    jump = len(present_values) - len(irr_array)
    return np.append(np.zeros(jump), irr_array)

def compute_disc_rates_pvs(disc_rates,total_cashflows,disc_factors):
    disc_rates_pv_array = np.zeros(len(disc_rates))
    for i in range(len(disc_rates)):
        for j in range (0,len(total_cashflows)):
            disc_rates_pv_array[i] += (total_cashflows[j]/((1+disc_rates['IRR'][i])**disc_factors[j]))
            
    return pd.DataFrame(disc_rates_pv_array, columns=['Present Value'], index=disc_rates.index)   

def npv(irr, cfs, yrs):  
    return np.sum(cfs / (1. + irr) ** yrs)
    
def irr(cfs, yrs, x0, **kwargs):
    return np.asscalar(fsolve(npv, x0=x0,args=(cfs,yrs), **kwargs))

def compute_irr(disc_rates_pvs, total_cashflows, disc_factors):
    irr_array = np.zeros(len(disc_rates_pvs))
    for j in range (len(disc_rates_pvs)):
        cashflows = np.append(np.negative(disc_rates_pvs['Present Value'][j]),total_cashflows)
        yrs = np.append(0, disc_factors)
        irr_array[j] += irr(cashflows, yrs, .03)
    return irr_array