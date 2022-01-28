# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 14:10:47 2022

@author: NVG9HXP
"""

import pandas as pd
from AssetAllocation.datamanger import datamanger as dm
from AssetAllocation.analytics.liability_model import liabilityModel

######################################################################################################
# IGNORE
######################################################################################################

df = pd.read_excel('plan_return_data.xlsx', sheet_name='data')
df_1 = df.pivot_table(values='Return', index='Date', columns='Plan')
df_1.columns = ['Pension', 'Retirement', 'IBT']
df_2 = df.pivot_table(values='MV', index='Date', columns='Plan')
df_2.columns = ['Pension', 'Retirement', 'IBT']

writer = pd.ExcelWriter('plan_level_data.xlsx', engine='xlsxwriter')
df_1.to_excel(writer, sheet_name='ret_data')
df_2.to_excel(writer, sheet_name='mv_data')
writer.save()

######################################################################################################
#
######################################################################################################


df_pbo_cfs = dm.get_cf_data('PBO')
df_pvfb_cfs = dm.get_cf_data('PBO')
df_sc_cfs = df_pvfb_cfs - df_pbo_cfs
df_ftse = dm.get_ftse_data(False)
plan_list = ['IBT', 'Pension', 'Retirement']

liab_model_dict={}

for pension_plan in plan_list:
    
    pbo_cashflows = df_pbo_cfs[pension_plan]
    disc_factors = df_pbo_cfs['Time']
    sc_cashflows = df_sc_cfs[pension_plan]
    liab_curve = dm.generate_liab_curve(df_ftse, pbo_cashflows)
    asset_mv = dm.get_plan_asset_mv(pension_plan)
    contrb_pct = 0.0
    liab_model = liabilityModel(pbo_cashflows, disc_factors, sc_cashflows, contrb_pct, asset_mv,liab_curve)
    
    liab_model_dict[pension_plan] = liab_model



df_return = liab_model_dict['Retirement'].returns_ts.copy()
df_return = dm.merge_dfs(df_return, liab_model_dict['Pension'].returns_ts.copy())
df_return = dm.merge_dfs(df_return, liab_model_dict['IBT'].returns_ts.copy())
df_return.columns = plan_list
plan_list = ['Retirement', 'Pension', 'IBT']
df_return.columns = plan_list
df_pvs = liab_model_dict['Retirement'].present_values.copy()
df_pvs = dm.merge_dfs(df_pvs,liab_model_dict['Pension'].present_values)
df_pvs = dm.merge_dfs(df_pvs,liab_model_dict['IBT'].present_values)
df_pvs.columns = plan_list
df_irr = liab_model_dict['Retirement'].irr_df.copy()
df_irr = dm.merge_dfs(df_irr,liab_model_dict['Pension'].irr_df)
df_irr = dm.merge_dfs(df_irr,liab_model_dict['IBT'].irr_df)
df_irr.columns = plan_list
writer = pd.ExcelWriter('liability_returns.xlsx', engine='xlsxwriter')
df_return.to_excel(writer, sheet_name='liability_returns')
df_pvs.to_excel(writer, sheet_name='present_values')
df_irr.to_excel(writer, sheet_name='irr')
writer.save()