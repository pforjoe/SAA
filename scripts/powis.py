# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 21:57:33 2023

@author: NVG9HXP
"""

import os
import pandas as pd
import numpy as np
from AssetAllocation.datamanager import ldi_dm as ldi
from AssetAllocation.datamanager import datamanager as dm
from AssetAllocation.reporting import reports as rp

# 
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

cfs_dict = ldi.generate_liab_cf_dict()
plan_pbo_cfs_dict = ldi.get_plan_cfs_dict(cfs_dict['PBO'])
plan_sc_cfs_dict = ldi.get_plan_cfs_dict(cfs_dict['SC'])

ldi_data_dict = {}
for plan in plan_pbo_cfs_dict:
    plan_pbo_cfs = ldi.offset(plan_pbo_cfs_dict[plan])
    plan_sc_cfs = ldi.offset(plan_sc_cfs_dict[plan])
    plan_cfs = plan_pbo_cfs + plan_sc_cfs
    plan_asset_data = dm.get_plan_asset_data()
    
    asset_mv = dm.get_plan_asset_mv(plan_asset_data,plan)
    asset_ret = dm.get_plan_asset_returns(plan_asset_data,plan)
    
    disc_factors = dm.get_cfs_time_col(plan_cfs)
    
    df_ftse = dm.get_ftse_data()
    liab_curve = dm.generate_liab_curve(df_ftse, plan_cfs)
    
    df_pv = ldi.compute_pvs(plan_cfs,liab_curve, disc_factors)
    df_irr = ldi.compute_irr(df_pv, plan_cfs, disc_factors)
    
    liab_ret_df = ldi.compute_liab_ret(df_pv, plan_cfs)
    liab_mv = df_pv.copy()
    liab_mv.columns = ['Market Value']
    
    
    fs = ldi.compute_funded_status(asset_mv, liab_mv)
    ldi_data_dict[plan] = {'pbo_cfs': plan_pbo_cfs, 'sc_cfs':plan_sc_cfs,
                           'disc_factors':disc_factors, 'asset_mv':asset_mv, 'irr': df_irr,
                           'liab_pv':df_pv, 'asset_ret':asset_ret,
                           'liab_ret':liab_ret_df,'fs':fs}

for plan in ldi_data_dict:
    report_name = f'{plan}_report_122023'
    filepath = rp.get_reportpath(report_name)
    writer = pd.ExcelWriter(filepath, engine = 'xlsxwriter')
    
    for key in ldi_data_dict[plan]:
        try:
            ldi_data_dict[plan][key].to_excel(writer, sheet_name=key)
        except AttributeError:
            pass
        #save file
    rp.print_report_info(report_name, filepath)

    writer.save()


#YTD    
present_values = df_pv.copy()
sc_accruals = plan_sc_cfs.iloc[0:1].transpose()
sc_accruals.columns = ['SC accruals']
sc_accruals = dm.merge_dfs(present_values, sc_accruals)
sc_accruals['YTD'] = sc_accruals['Present Value'].shift()
sc_accruals['QTD'] = sc_accruals['Present Value'].shift()
sc_accruals['MTD'] = sc_accruals['Present Value'].shift()
   
sc_accruals['YTD'][1] = (sc_accruals['Present Value'][1]+sc_accruals['SC accruals'].iloc[0:1].sum())/sc_accruals['Present Value'][0]-1
sc_accruals['YTD'][2] = (sc_accruals['Present Value'][2]+sc_accruals['SC accruals'].iloc[0:2].sum())/sc_accruals['Present Value'][0]-1
sc_accruals['YTD'][3] = (sc_accruals['Present Value'][3]+sc_accruals['SC accruals'].iloc[0:3].sum())/sc_accruals['Present Value'][0]-1
sc_accruals['YTD'][4] = (sc_accruals['Present Value'][4]+sc_accruals['SC accruals'].iloc[0:4].sum())/sc_accruals['Present Value'][0]-1
sc_accruals['YTD'][5] = (sc_accruals['Present Value'][5]+sc_accruals['SC accruals'].iloc[0:5].sum())/sc_accruals['Present Value'][0]-1
sc_accruals['YTD'][6] = (sc_accruals['Present Value'][6]+sc_accruals['SC accruals'].iloc[0:6].sum())/sc_accruals['Present Value'][0]-1
sc_accruals['YTD'][7] = (sc_accruals['Present Value'][7]+sc_accruals['SC accruals'].iloc[0:7].sum())/sc_accruals['Present Value'][0]-1
sc_accruals['YTD'][8] = (sc_accruals['Present Value'][8]+sc_accruals['SC accruals'].iloc[0:8].sum())/sc_accruals['Present Value'][0]-1
sc_accruals['YTD'][9] = (sc_accruals['Present Value'][9]+sc_accruals['SC accruals'].iloc[0:9].sum())/sc_accruals['Present Value'][0]-1
sc_accruals['YTD'][10] = (sc_accruals['Present Value'][10]+sc_accruals['SC accruals'].iloc[0:10].sum())/sc_accruals['Present Value'][0]-1
sc_accruals['YTD'][11] = (sc_accruals['Present Value'][11]+sc_accruals['SC accruals'].iloc[0:11].sum())/sc_accruals['Present Value'][0]-1
sc_accruals['YTD'][12] = (sc_accruals['Present Value'][12]+sc_accruals['SC accruals'].iloc[0:12].sum())/sc_accruals['Present Value'][0]-1
sc_accruals['YTD'][13] = (sc_accruals['Present Value'][13]+sc_accruals['SC accruals'].iloc[12:13].sum())/sc_accruals['Present Value'][12]-1




sc_accruals['QTD'][1] = (sc_accruals['Present Value'][1]+sc_accruals['SC accruals'].iloc[0:1].sum())/sc_accruals['Present Value'][0]-1
sc_accruals['QTD'][2] = (sc_accruals['Present Value'][2]+sc_accruals['SC accruals'].iloc[0:2].sum())/sc_accruals['Present Value'][0]-1
sc_accruals['QTD'][3] = (sc_accruals['Present Value'][3]+sc_accruals['SC accruals'].iloc[0:3].sum())/sc_accruals['Present Value'][0]-1
sc_accruals['QTD'][4] = (sc_accruals['Present Value'][4]+sc_accruals['SC accruals'].iloc[3:4].sum())/sc_accruals['Present Value'][3]-1
sc_accruals['QTD'][5] = (sc_accruals['Present Value'][5]+sc_accruals['SC accruals'].iloc[3:5].sum())/sc_accruals['Present Value'][3]-1
sc_accruals['QTD'][6] = (sc_accruals['Present Value'][6]+sc_accruals['SC accruals'].iloc[3:6].sum())/sc_accruals['Present Value'][3]-1
sc_accruals['QTD'][7] = (sc_accruals['Present Value'][7]+sc_accruals['SC accruals'].iloc[6:7].sum())/sc_accruals['Present Value'][6]-1
sc_accruals['QTD'][8] = (sc_accruals['Present Value'][8]+sc_accruals['SC accruals'].iloc[6:8].sum())/sc_accruals['Present Value'][6]-1
sc_accruals['QTD'][9] = (sc_accruals['Present Value'][9]+sc_accruals['SC accruals'].iloc[6:9].sum())/sc_accruals['Present Value'][6]-1
sc_accruals['QTD'][10] = (sc_accruals['Present Value'][10]+sc_accruals['SC accruals'].iloc[9:10].sum())/sc_accruals['Present Value'][9]-1
sc_accruals['QTD'][11] = (sc_accruals['Present Value'][11]+sc_accruals['SC accruals'].iloc[9:11].sum())/sc_accruals['Present Value'][9]-1
sc_accruals['QTD'][12] = (sc_accruals['Present Value'][12]+sc_accruals['SC accruals'].iloc[9:12].sum())/sc_accruals['Present Value'][9]-1
sc_accruals['QTD'][13] = (sc_accruals['Present Value'][13]+sc_accruals['SC accruals'].iloc[12:13].sum())/sc_accruals['Present Value'][12]-1


sc_accruals['MTD'][1] = (sc_accruals['Present Value'][1]+sc_accruals['SC accruals'].iloc[0:1].sum())/sc_accruals['Present Value'][0]-1
sc_accruals['MTD'][2] = (sc_accruals['Present Value'][2]+sc_accruals['SC accruals'].iloc[1:2].sum())/sc_accruals['Present Value'][1]-1
sc_accruals['MTD'][3] = (sc_accruals['Present Value'][3]+sc_accruals['SC accruals'].iloc[2:3].sum())/sc_accruals['Present Value'][2]-1
sc_accruals['MTD'][4] = (sc_accruals['Present Value'][4]+sc_accruals['SC accruals'].iloc[3:4].sum())/sc_accruals['Present Value'][3]-1
sc_accruals['MTD'][5] = (sc_accruals['Present Value'][5]+sc_accruals['SC accruals'].iloc[4:5].sum())/sc_accruals['Present Value'][4]-1
sc_accruals['MTD'][6] = (sc_accruals['Present Value'][6]+sc_accruals['SC accruals'].iloc[5:6].sum())/sc_accruals['Present Value'][5]-1
sc_accruals['MTD'][7] = (sc_accruals['Present Value'][7]+sc_accruals['SC accruals'].iloc[6:7].sum())/sc_accruals['Present Value'][6]-1
sc_accruals['MTD'][8] = (sc_accruals['Present Value'][8]+sc_accruals['SC accruals'].iloc[7:8].sum())/sc_accruals['Present Value'][7]-1
sc_accruals['MTD'][9] = (sc_accruals['Present Value'][9]+sc_accruals['SC accruals'].iloc[8:9].sum())/sc_accruals['Present Value'][8]-1
sc_accruals['MTD'][10] = (sc_accruals['Present Value'][10]+sc_accruals['SC accruals'].iloc[9:10].sum())/sc_accruals['Present Value'][9]-1
sc_accruals['MTD'][11] = (sc_accruals['Present Value'][11]+sc_accruals['SC accruals'].iloc[10:11].sum())/sc_accruals['Present Value'][10]-1
sc_accruals['MTD'][12] = (sc_accruals['Present Value'][12]+sc_accruals['SC accruals'].iloc[11:12].sum())/sc_accruals['Present Value'][11]-1
sc_accruals['MTD'][13] = (sc_accruals['Present Value'][13]+sc_accruals['SC accruals'].iloc[12:13].sum())/sc_accruals['Present Value'][12]-1

