# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 14:10:47 2022

@author: NVG9HXP
"""

import pandas as pd
from AssetAllocation.datamanger import datamanger as dm
from AssetAllocation.analytics.liability_model import liabilityModel
import pandas as pd
import numpy as np
from scipy.optimize import fsolve
import AssetAllocation.reporting.sheets as sheets
from AssetAllocation.reporting import reports as rp

# df = pd.read_excel('plan_return_data.xlsx', sheet_name='data')
# df_1 = df.pivot_table(values='Return', index='Date', columns='Plan')
# df_1.columns = ['Pension', 'Retirement', 'IBT']
# df_2 = df.pivot_table(values='MV', index='Date', columns='Plan')
# df_2.columns = ['Pension', 'Retirement', 'IBT']

# writer = pd.ExcelWriter('plan_level_data.xlsx', engine='xlsxwriter')
# df_1.to_excel(writer, sheet_name='ret_data')
# df_2.to_excel(writer, sheet_name='mv_data')
# writer.save()

############################################################################################################################################################
# IMPORT AND TRANSFORM CASHFLOW                                                            
############################################################################################################################################################

df_pbo_cfs = dm.get_cf_data('PBO')
df_pbo_cfs["Total"] =  df_pbo_cfs["IBT"] + df_pbo_cfs["Retirement"] + df_pbo_cfs["Pension"]

df_pvfb_cfs = dm.get_cf_data('PBO')
df_pvfb_cfs["Total"] =  df_pvfb_cfs["IBT"] + df_pvfb_cfs["Retirement"] + df_pvfb_cfs["Pension"]

df_sc_cfs = df_pvfb_cfs - df_pbo_cfs
df_ftse = dm.get_ftse_data(False)
plan_list =['Retirement', 'Pension', 'IBT',"Total"]

############################################################################################################################################################
# TRANSFORM DATA TO LIABILITY MODEL INPUTS AND GET LIABILITY MODEL                                             
############################################################################################################################################################

liab_model_dict={}

for pension_plan in plan_list:
    
    pbo_cashflows = df_pbo_cfs[pension_plan]
    disc_factors = df_pbo_cfs['Time']
    sc_cashflows = df_sc_cfs[pension_plan]
    liab_curve = dm.generate_liab_curve(df_ftse, pbo_cashflows)
    asset_mv = dm.get_plan_asset_mv(pension_plan)
    if pension_plan != "Total":
        asset_returns = pd.read_excel(dm.TS_FP+"plan_return_data.xlsx",sheet_name = pension_plan ,usecols=[0,1],index_col=0)
    contrb_pct = 0.0
    liab_model = liabilityModel(pbo_cashflows, disc_factors, sc_cashflows, contrb_pct, asset_mv, asset_returns, liab_curve)
    liab_model_dict[pension_plan] = liab_model

############################################################################################################################################################
# MERGE DATA FRAMES FOR XCEL WRITER                                                          
############################################################################################################################################################

asset_liab_ret_dict = {}
for plan in plan_list:
    if plan == "Total":
        asset_liab_ret_dict[plan] = liab_model_dict[plan].returns_ts
    else:
        asset_liab_ret_df = dm.merge_dfs( liab_model_dict[plan].asset_returns, liab_model_dict[plan].returns_ts)
        asset_liab_ret_df.columns = ["Asset","Liability"]
        asset_liab_ret_dict[plan] = asset_liab_ret_df

plan_list =['Retirement', 'Pension', 'IBT']
df_return = liab_model_dict['Retirement'].returns_ts.copy()
df_return = dm.merge_dfs(df_return, liab_model_dict['Pension'].returns_ts.copy())
df_return = dm.merge_dfs(df_return, liab_model_dict['IBT'].returns_ts.copy())
df_return.columns = plan_list

df_return.columns = plan_list
df_pvs = liab_model_dict['Retirement'].present_values.copy()
df_pvs = dm.merge_dfs(df_pvs,liab_model_dict['Pension'].present_values)
df_pvs = dm.merge_dfs(df_pvs,liab_model_dict['IBT'].present_values)
df_pvs.columns = plan_list

df_irr = liab_model_dict['Retirement'].irr_df.copy()
df_irr = dm.merge_dfs(df_irr,liab_model_dict['Pension'].irr_df)
df_irr = dm.merge_dfs(df_irr,liab_model_dict['IBT'].irr_df)
df_irr.columns = plan_list

df_asset_mv =  liab_model_dict['Retirement'].asset_mv.copy()
df_asset_mv = dm.merge_dfs(df_asset_mv, liab_model_dict['Pension'].asset_mv.copy())
df_asset_mv = dm.merge_dfs(df_asset_mv, liab_model_dict['IBT'].asset_mv.copy())
df_asset_mv.columns = plan_list

############################################################################################################################################################
# CREATE DATA DICTIONARY WITH DATA FRAMES AND GENERATE REPORT                                                         
############################################################################################################################################################

report_dict = {"df_return": df_return, "df_pvs": df_pvs, "df_irr": df_irr, "df_asset_mv": df_asset_mv, "asset_liab_ret_dict": asset_liab_ret_dict}
rp.get_liability_returns_report(report_dict,report_name = "liability_returns")

#df_return.to_excel(writer, sheet_name='liability_returns')
#df_pvs.to_excel(writer, sheet_name='present_values')
#df_irr.to_excel(writer, sheet_name='irr')\
    
    
# sheets.set_return_sheet(writer, df_return, sheet_name = "liability_returns", sample_ret = False)
# sheets.set_present_values_sheet(writer, df_pvs) 
# sheets.set_return_sheet(writer, df_irr, sheet_name = "IRR", sample_ret = False)
# sheets.set_asset_mv_sheet(writer, df_asset_mv)
# sheets.set_asset_liability_charts_sheet(writer, asset_liab_ret_dict)
    
# writer.save()