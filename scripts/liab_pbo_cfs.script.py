# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 01:56:30 2022

@author: NVG9HXP
"""

from AssetAllocation.datamanger import datamanger as dm
from AssetAllocation.reporting import reports as rp
import pandas as pd


SHEET_LIST = ['2019','2020','2021','2021_1', '2022']
PLAN_LIST = ['IBT','Pension', 'Retirement']

def switch_int(arg,n):

    switcher = {
            "2021": 12-n,
            "2021_1": n,
    }
    return switcher.get(arg, 12)


def fill_dict_df_na(dict_df, value=0):
    for key in dict_df:
        dict_df[key].fillna(value, inplace=True)

def get_liab_mv_cf_cols():
    #create empty liab_mv_cf df
    df_ftse = dm.get_ftse_data(False)
    dates = df_ftse.transpose().iloc[1:,]
    dates.sort_index(inplace=True)
    return list(dates.index[109:])

def transform_pbo_df(pbo_df):
    temp_df_list = [pbo_df.iloc[0:12* len(SHEET_LIST)],pbo_df.iloc[12*len(SHEET_LIST):,]]
    temp_df_list[1].fillna(0, inplace=True)
    temp_df = temp_df_list[0].append(temp_df_list[1])
    return temp_df

#################################################################################################################################################        
def get_past_pbo_data(filename = 'past_pbo_cashflow_data.xlsx'):
    past_pbo_dict = {}
    for sheet in SHEET_LIST[:-1]:
        df_pbo_cfs = pd.read_excel(dm.TS_FP+'past_pbo_cashflow_data.xlsx', 
                                   sheet_name=sheet, index_col=0)/12
        df_pbo_cfs = dm.reindex_to_monthly_data(df_pbo_cfs)
        #transform this pbo due to change mid year
        if sheet == '2021_1':
            df_pbo_cfs_1 = df_pbo_cfs.iloc[0:12]
            df_pbo_cfs_2 = df_pbo_cfs.iloc[12:,]
            for col in df_pbo_cfs_1.columns:
                n = 9 if col == 'IBT' else 8
                df_pbo_cfs_1[col] = df_pbo_cfs_1[col]*(12/n)
            df_pbo_cfs = df_pbo_cfs_1.append(df_pbo_cfs_2)
        past_pbo_dict[sheet] = df_pbo_cfs
    return past_pbo_dict

def get_plan_pbo_dict():
    past_pbo_dict = get_past_pbo_data()
    plan_pbo_dict = {}
    for plan in PLAN_LIST:
        temp_pbo_df = pd.DataFrame()
        #merge past years pbos
        for key in past_pbo_dict:
            temp_pbo_df = dm.merge_dfs(temp_pbo_df, past_pbo_dict[key][plan],drop = False)
        
        #merge current years pbos
        temp_pbo_df = dm.merge_dfs(temp_pbo_df, dm.get_cf_data()[plan], drop = False)
        
        #rename dataframe
        temp_pbo_df.columns = SHEET_LIST
        plan_pbo_dict[plan] = temp_pbo_df
    return plan_pbo_dict

def get_plan_sc_dict(plan_pbo_dict, filename='past_sc_cashflow_data.xlsx'):
    plan_sc_dict = {}
    for plan in PLAN_LIST:
        n = 9 if plan == 'IBT' else 8
        sc_df = plan_pbo_dict[plan].copy()
        # sc_df.fillna(0, inplace=True)
        for x in range(1,len(SHEET_LIST)):
            temp_df_list = [sc_df.iloc[0:12*x],sc_df.iloc[12*x:,]]
            temp_df_list[1].fillna(0, inplace=True)
            for df in temp_df_list:
                df[SHEET_LIST[x-1]] = (df[SHEET_LIST[x]] - 
                                      df[SHEET_LIST[x-1]])/switch_int(SHEET_LIST[x-1],n)
            temp_df = temp_df_list[0].append(temp_df_list[1])
            sc_df[SHEET_LIST[x-1]] = temp_df[SHEET_LIST[x-1]]
        sc_df[SHEET_LIST[x]] = dm.get_cf_data('Service Cost')[plan]/12
        # sc_df[SHEET_LIST[x]] = 0
        if plan == 'IBT':
            sc_df.drop(['2021'], axis=1, inplace=True)
            df_sc_cfs = pd.read_excel(dm.TS_FP+filename, 
                                      sheet_name='2021', index_col=0)/12
            df_sc_cfs = dm.reindex_to_monthly_data(df_sc_cfs)[[plan]]/12
            df_sc_cfs.columns = ['2021']
            sc_df = dm.merge_dfs(sc_df, df_sc_cfs, drop= False)
        plan_sc_dict[plan] = sc_df[SHEET_LIST]
    return plan_sc_dict
    
def generate_liab_mv_dict():
    plan_pbo_dict = get_plan_pbo_dict()
    plan_sc_dict = get_plan_sc_dict(plan_pbo_dict)
    
    liab_mv_dict = {}
    for key in plan_pbo_dict:
        year_dict = {}
        n = 9 if key == 'IBT' else 8
        
        for year in plan_pbo_dict[key].columns:
            temp_pbo_df = transform_pbo_df(plan_pbo_dict[key][year])
            #Add below to transform function and make it for temp_cfs
            temp_sc_df = plan_sc_dict[key][year]
            temp_sc_df.fillna(0, inplace=True)
            temp_cfs_df = dm.merge_dfs(temp_pbo_df, temp_sc_df)
            temp_cfs_df.columns = ['PBO', 'SC']
            if year == SHEET_LIST[-1]:
                no_of_cols = len(get_liab_mv_cf_cols())%12 
            else:
                no_of_cols = switch_int(year, n)
            liab_cfs = pd.DataFrame(columns=list(range(1,no_of_cols+1)), index=temp_cfs_df.index)
            
            for col in liab_cfs.columns:
                if key == 'IBT' and year == '2021' and col == 3:
                    liab_cfs[col] = transform_pbo_df(plan_pbo_dict[key]['2021_1'])
                else:    
                    liab_cfs[col] = temp_cfs_df['PBO'] + temp_cfs_df['SC']*col
                jump = 12-no_of_cols if year == '2021_1'else 0
            
                
                liab_cfs.loc[:col+jump,col] = 0
            year_dict[year] = liab_cfs
        liab_mv_dict[key] = year_dict
    return liab_mv_dict

def get_plan_mv_cfs_dict():
    liab_mv_dict = generate_liab_mv_dict()
    plan_mv_cfs_dict = {}
    for plan in liab_mv_dict:
        mv_cfs_df = pd.DataFrame()
        #merge past years pbos
        for year in liab_mv_dict[plan]:
            mv_cfs_df = dm.merge_dfs(mv_cfs_df, liab_mv_dict[plan][year],drop = False)
        mv_cfs_df.columns = get_liab_mv_cf_cols()
        mv_cfs_df.fillna(0, inplace=True)
        plan_mv_cfs_dict[plan] = mv_cfs_df
    plan_mv_cfs_dict['Total'] = aggregate_mv_cfs(plan_mv_cfs_dict)
    return plan_mv_cfs_dict

def aggregate_mv_cfs(plan_mv_cfs_dict):
    agg_df = pd.DataFrame(index=plan_mv_cfs_dict[PLAN_LIST[0]].index, columns=plan_mv_cfs_dict[PLAN_LIST[0]].columns)
    agg_df.fillna(0, inplace=True)
    for plan in PLAN_LIST:
        agg_df = plan_mv_cfs_dict[plan] + agg_df
    return agg_df

###################################################################################################################################################
#import past pbo data by year
past_pbo_dict = {}
for sheet in SHEET_LIST[:-1]:
    df_pbo_cfs = pd.read_excel(dm.TS_FP+'past_pbo_cashflow_data.xlsx', 
                               sheet_name=sheet, index_col=0)/12
    df_pbo_cfs = dm.reindex_to_monthly_data(df_pbo_cfs)
    #transform this pbo due to change mid year
    if sheet == '2021_1':
        df_pbo_cfs_1 = df_pbo_cfs.iloc[0:12]
        df_pbo_cfs_2 = df_pbo_cfs.iloc[12:,]
        for col in df_pbo_cfs_1.columns:
            n = 9 if col == 'IBT' else 8
            df_pbo_cfs_1[col] = df_pbo_cfs_1[col]*(12/n)
        df_pbo_cfs = df_pbo_cfs_1.append(df_pbo_cfs_2)
    past_pbo_dict[sheet] = df_pbo_cfs

    

#create pbo_dict by plan by merging years together    
plan_pbo_dict = {}
for plan in PLAN_LIST:
    pbo_df = pd.DataFrame()
    #merge past years pbos
    for key in past_pbo_dict:
        pbo_df = dm.merge_dfs(pbo_df, past_pbo_dict[key][plan],dropna = False)
    
    #merge current years pbos
    pbo_df = dm.merge_dfs(pbo_df, dm.get_cf_data()[plan], dropna = False)
    
    #rename dataframe
    pbo_df.columns = SHEET_LIST
    plan_pbo_dict[plan] = pbo_df

#create sc_dict by plan by merging years together    
plan_sc_dict = {}
for plan in PLAN_LIST:
    n = 9 if plan == 'IBT' else 8
    sc_df = plan_pbo_dict[plan].copy()
    # sc_df.fillna(0, inplace=True)
    for x in range(1,len(SHEET_LIST)):
        temp_df_list = [sc_df.iloc[0:12*x],sc_df.iloc[12*x:,]]
        temp_df_list[1].fillna(0, inplace=True)
        for df in temp_df_list:
            df[SHEET_LIST[x-1]] = (df[SHEET_LIST[x]] - 
                                  df[SHEET_LIST[x-1]])/switch_int(SHEET_LIST[x-1],n)
        temp_df = temp_df_list[0].append(temp_df_list[1])
        sc_df[SHEET_LIST[x-1]] = temp_df[SHEET_LIST[x-1]]
    sc_df[SHEET_LIST[x]] = dm.get_cf_data('Service Cost')[plan]/12
    # sc_df[SHEET_LIST[x]] = 0
    if plan == 'IBT':
        sc_df.drop(['2021'], axis=1, inplace=True)
        df_sc_cfs = pd.read_excel(dm.TS_FP+'past_sc_cashflow_data.xlsx', 
                                  sheet_name='2021', index_col=0)/12
        df_sc_cfs = dm.reindex_to_monthly_data(df_sc_cfs)[[plan]]/12
        df_sc_cfs.columns = ['2021']
        sc_df = dm.merge_dfs(sc_df, df_sc_cfs, dropna = False)
    plan_sc_dict[plan] = sc_df[SHEET_LIST]


#create liab cfs dict  
liab_mv_dict = {}
for key in plan_pbo_dict:
    year_dict = {}
    n = 9 if key == 'IBT' else 8
        
    
    for year in plan_pbo_dict[key].columns:
        
        temp_pbo_df = transform_pbo_df(plan_pbo_dict[key][year])
        #Add below to transform function and make it for temp_cfs
        temp_sc_df = plan_sc_dict[key][year]
        temp_sc_df.fillna(0, inplace=True)
        temp_cfs_df = dm.merge_dfs(temp_pbo_df, temp_sc_df)
        temp_cfs_df.columns = ['PBO', 'SC']
        if year == SHEET_LIST[-1]:
            no_of_cols = len(get_liab_mv_cf_cols())%12 
        else:
            no_of_cols = switch_int(year, n)
        liab_cfs = pd.DataFrame(columns=list(range(1,no_of_cols+1)), index=temp_cfs_df.index)
        
        for col in liab_cfs.columns:
            if key == 'IBT' and year == '2021' and col == 3:
                liab_cfs[col] = transform_pbo_df(plan_pbo_dict[key]['2021_1'])
            else:    
                liab_cfs[col] = temp_cfs_df['PBO'] + temp_cfs_df['SC']*col
            jump = 12-no_of_cols if year == '2021_1'else 0
            
            liab_cfs.loc[:col+jump,col] = 0
        year_dict[year] = liab_cfs
    liab_mv_dict[key] = year_dict
    
#merge yearly cfs together
plan_mv_cfs_dict = {}
for plan in liab_mv_dict:
    mv_cfs_df = pd.DataFrame()
    #merge past years pbos
    for year in liab_mv_dict[plan]:
        mv_cfs_df = dm.merge_dfs(mv_cfs_df, liab_mv_dict[plan][year],dropna = False)
    mv_cfs_df.columns = get_liab_mv_cf_cols()
    mv_cfs_df.fillna(0, inplace=True)
    plan_mv_cfs_dict[plan] = mv_cfs_df
plan_mv_cfs_dict['Total'] = aggregate_mv_cfs(plan_mv_cfs_dict)
    


filepath = rp.get_reportpath('new_mv_cfs_2')
writer = pd.ExcelWriter(filepath,engine='xlsxwriter')
for key in plan_mv_cfs_dict:
    
    plan_mv_cfs_dict[key].to_excel(writer, key)
    
writer.save()