# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 01:56:30 2022

@author: NVG9HXP
"""

from AssetAllocation.datamanger import datamanger as dm
import pandas as pd


def switch_int(arg,n):

    switcher = {
            "2020": 12-n,
            "2020_1": n,
    }
    return switcher.get(arg, 12)


def fill_dict_df_na(dict_df, value=0):
    for key in dict_df:
        dict_df[key].fillna(value, inplace=True)

def create_empty_liab_mv_cf_df(plan_pbo_dict, plan):
    #create empty liab_mv_cf df
    df_ftse = dm.get_ftse_data(False)
    dates = df_ftse.transpose().iloc[1:,]
    dates.sort_index(inplace=True)
    
    col_list = list(range(0,(len(dates.index)-109)))

    return pd.DataFrame(columns=dates.index[109:], index=plan_pbo_dict[plan].index)

sheet_list = ['2018', '2019','2020','2020_1', '2021']
plan_list = ['IBT','Pension', 'Retirement']

#import past pbo data by year
past_pbo_dict = {}
for sheet in sheet_list[:-1]:
    df_pbo_cfs = pd.read_excel(dm.TS_FP+'past_pbo_cashflow_data.xlsx', sheet_name=sheet, index_col=0)/12
    df_pbo_cfs = dm.reindex_to_monthly_data(df_pbo_cfs)
    #transform this pbo due to change mid year
    if sheet == '2020_1':
        df_pbo_cfs_1 = df_pbo_cfs.iloc[0:12]
        df_pbo_cfs_2 = df_pbo_cfs.iloc[12:,]
        for col in df_pbo_cfs_1.columns:
            n = 9 if col == 'IBT' else 8
            df_pbo_cfs_1[col] = df_pbo_cfs_1[col]*(12/n)
        df_pbo_cfs = df_pbo_cfs_1.append(df_pbo_cfs_2)
    past_pbo_dict[sheet] = df_pbo_cfs
        
#create pbo_dict by plan by merging years together    
plan_pbo_dict = {}
for plan in plan_list:
    pbo_df = pd.DataFrame()
    #merge past years pbos
    for key in past_pbo_dict:
        # pbo_df = pd.merge(pbo_df, past_pbo_dict[key][plan], left_index = True, right_index = True, how = 'outer')
        pbo_df = dm.merge_dfs(pbo_df, past_pbo_dict[key][plan],drop = False)
    
    #merge current years pbos
    pbo_df = dm.merge_dfs(pbo_df, dm.get_cf_data('PBO')[plan], drop = False)
    
    #rename dataframe
    pbo_df.columns = sheet_list
    plan_pbo_dict[plan] = pbo_df

#create sc_dict by plan by merging years together    
plan_sc_dict = {}
for plan in plan_list:
    n = 9 if plan == 'IBT' else 8
    sc_df = plan_pbo_dict[plan].copy()
    for x in range(1,len(sheet_list)):
        sc_df[sheet_list[x-1]] = (plan_pbo_dict[plan][sheet_list[x]]- plan_pbo_dict[plan][sheet_list[x-1]])/switch_int(sheet_list[x-1],n)
    sc_df[sheet_list[x]] = 0
    if plan == 'IBT':
        sc_df.drop(['2020'], axis=1, inplace=True)
        df_sc_cfs = pd.read_excel(dm.TS_FP+'past_sc_cashflow_data.xlsx', sheet_name='2020', index_col=0)/12
        df_sc_cfs = dm.reindex_to_monthly_data(df_sc_cfs)[[plan]]/12
        df_sc_cfs.columns = ['2020']
        sc_df = dm.merge_dfs(sc_df, df_sc_cfs, drop= False)
    plan_sc_dict[plan] = sc_df[sheet_list]

fill_dict_df_na(plan_pbo_dict)
fill_dict_df_na(plan_sc_dict)
    
plan_mv_cfs_dict = {}
for plan in plan_list:
    plan_mv_cfs_dict[plan] = create_empty_liab_mv_cf_df(plan_pbo_dict, plan)
    
#create empty liab_mv_cf df
df_ftse = dm.get_ftse_data(False)
dates = df_ftse.transpose().iloc[1:,]
dates.sort_index(inplace=True)

col_list = list(range(0,(len(dates.index)-109)))

liab_mv_cfs = pd.DataFrame(columns=dates.index[109:], index=plan_pbo_dict[plan].index)
