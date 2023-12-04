# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 21:57:33 2023

@author: NVG9HXP
"""

import os
import pandas as pd
import numpy as np
from scipy.optimize import fsolve
from AssetAllocation.datamanager import datamanager as dm
from AssetAllocation.analytics.ts_analytics import get_ann_vol
# 
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')


# set filepath
CWD = os.getcwd()
DATA_FP = CWD + '\\data\\'
MV_INPUTS_FP = DATA_FP + 'mv_inputs\\'
TS_FP = DATA_FP + 'time_series\\'
PLAN_INPUTS_FP = DATA_FP + 'plan_inputs\\'
UPDATE_FP = DATA_FP + 'update_files\\'

SHEET_LIST_1 = ['2021','2021_1', '2022','2023']
SHEET_LIST_2 = ['2021', '2022','2023']
PLAN_LIST = ['IBT','Pension', 'Retirement']



def get_past_cf_data(past_cf_filename = 'past_pbo_cashflow_data.xlsx'):
    """
    

    Parameters
    ----------
    filename : TYPE, optional
        DESCRIPTION. The default is 'past_pbo_cashflow_data.xlsx'.

    Returns
    -------
    past_cf_dict : TYPE
        DESCRIPTION.

    """
    past_cf_dict = {}
    for sheet in SHEET_LIST_2[:-1]:
        df_cfs = pd.read_excel(TS_FP+past_cf_filename, 
                                   sheet_name=sheet, index_col=0)/12
        df_cfs = dm.reindex_to_monthly_data(df_cfs)
        past_cf_dict[sheet] = df_cfs
    return past_cf_dict

def get_plan_cf_dict(past_cf_filename = 'past_pbo_cashflow_data.xlsx', cf_type = 'PBO'):
    """
    

    Parameters
    ----------
    filename : TYPE, optional
        DESCRIPTION. The default is 'past_pbo_cashflow_data.xlsx'.
    cf_type : TYPE, optional
        DESCRIPTION. The default is 'PBO'.

    Returns
    -------
    plan_cf_dict : TYPE
        DESCRIPTION.

    """
    past_cf_dict = get_past_cf_data(past_cf_filename)
    plan_cf_dict = {}
    for plan in PLAN_LIST:
        temp_cf_df = pd.DataFrame()
        #merge past years cfs
        for key in past_cf_dict:
            temp_cf_df = dm.merge_dfs(temp_cf_df, past_cf_dict[key][plan],dropna = False)
        
        #merge current years cfs
        df_cfs = dm.get_cf_data(cf_type)[[plan]]
        temp_cf_df = dm.merge_dfs(temp_cf_df, df_cfs, dropna = False)
        
        #rename dataframe
        temp_cf_df.columns = SHEET_LIST_2
        plan_cf_dict[plan] = temp_cf_df
    return plan_cf_dict

def generate_liab_cf_dict(past_pbo_filename = 'past_pbo_cashflow_data.xlsx', past_sc_filename='past_sc_cashflow_data.xlsx',
                          ftse_filename = 'ftse_data.xlsx'):
    """
    

    Parameters
    ----------
    past_pbo_filename : TYPE, optional
        DESCRIPTION. The default is 'past_pbo_cashflow_data.xlsx'.
    past_sc_filename : TYPE, optional
        DESCRIPTION. The default is 'past_sc_cashflow_data.xlsx'.
    ftse_filename : TYPE, optional
        DESCRIPTION. The default is 'ftse_data.xlsx'.

    Returns
    -------
    liab_mv_dict : TYPE
        DESCRIPTION.

    """
    plan_pbo_dict = get_plan_cf_dict(past_pbo_filename, 'PBO')
    plan_sc_dict = get_plan_cf_dict(past_sc_filename, 'Service Cost')
    
    liab_pbo_cf_dict = {}
    liab_sc_cf_dict = {}
    for key in plan_pbo_dict:
        pbo_year_dict = {}
        sc_year_dict = {}
        
        for year in plan_pbo_dict[key].columns:
            temp_pbo_df = dm.transform_pbo_df(plan_pbo_dict[key][year])
            #Add below to transform function and make it for temp_cfs
            temp_sc_df = plan_sc_dict[key][year]
            temp_sc_df.fillna(0, inplace=True)
            temp_cfs_df = dm.merge_dfs(temp_pbo_df, temp_sc_df)
            temp_cfs_df.columns = ['PBO', 'SC']
            if year == SHEET_LIST_1[-1]:
                no_of_cols = len(dm.get_liab_mv_cf_cols(ftse_filename))%12-1
            else:
                no_of_cols = 11
            liab_cfs = pd.DataFrame(columns=list(range(0,no_of_cols+1)), index=temp_cfs_df.index)
            liab_pbo_cfs = pd.DataFrame(columns=list(range(0,no_of_cols+1)), index=temp_cfs_df.index)
            liab_sc_cfs = liab_cfs.copy()
            
            for col in liab_pbo_cfs.columns:
                liab_pbo_cfs[col] = temp_cfs_df['PBO']
                liab_sc_cfs[col] = temp_cfs_df['SC']
                
                jump = 11-no_of_cols if year == '2021_1'else 0
        
                liab_pbo_cfs.loc[:col+jump,col] = 0
                liab_sc_cfs.loc[:col+jump,col] = 0
            pbo_year_dict[year] = liab_pbo_cfs
            sc_year_dict[year] = liab_sc_cfs
        liab_pbo_cf_dict[key] = pbo_year_dict
        liab_sc_cf_dict[key] = sc_year_dict
    return {'PBO':liab_pbo_cf_dict, 'SC':liab_sc_cf_dict}

# def get_past_pbo_data(past_pbo_filename = 'past_pbo_cashflow_data.xlsx'):
#     """
    

#     Parameters
#     ----------
#     filename : TYPE, optional
#         DESCRIPTION. The default is 'past_pbo_cashflow_data.xlsx'.

#     Returns
#     -------
#     past_pbo_dict : TYPE
#         DESCRIPTION.

#     """
#     past_pbo_dict = {}
#     for sheet in SHEET_LIST_1[:-1]:
#         df_pbo_cfs = pd.read_excel(TS_FP+past_pbo_filename, 
#                                    sheet_name=sheet, index_col=0)/12
#         df_pbo_cfs = dm.reindex_to_monthly_data(df_pbo_cfs)
#         #transform this pbo due to change mid year
#         if sheet == '2021_1':
#             df_pbo_cfs_1 = df_pbo_cfs.iloc[0:12]
#             df_pbo_cfs_2 = df_pbo_cfs.iloc[12:,]
#             for col in df_pbo_cfs_1.columns:
#                 n = 9 if col == 'IBT' else 8
#                 df_pbo_cfs_1[col] = df_pbo_cfs_1[col]*(12/n)
#             df_pbo_cfs = df_pbo_cfs_1.append(df_pbo_cfs_2)
#         past_pbo_dict[sheet] = df_pbo_cfs
#     return past_pbo_dict

# def get_past_sc_data(past_sc_filename = 'past_sc_cashflow_data.xlsx'):
#     """
    

#     Parameters
#     ----------
#     filename : TYPE, optional
#         DESCRIPTION. The default is 'past_sc_cashflow_data.xlsx'.

#     Returns
#     -------
#     past_sc_dict : TYPE
#         DESCRIPTION.

#     """
#     past_sc_dict = {}
#     for sheet in SHEET_LIST_1[:-1]:
#         if sheet=='2021_1':
#             df_sc_cfs = pd.read_excel(TS_FP+past_sc_filename, 
#                                    sheet_name='2021', index_col=0)/12
#         else:
#             df_sc_cfs = pd.read_excel(TS_FP+past_sc_filename, 
#                                    sheet_name=sheet, index_col=0)/12
#         df_sc_cfs = dm.reindex_to_monthly_data(df_sc_cfs)
#         #transform this pbo due to change mid year
#         if sheet == '2021_1' or sheet =='2021':
#             df_sc_cfs_1 = df_sc_cfs.iloc[0:12]
#             df_sc_cfs_2 = df_sc_cfs.iloc[12:,]
#             for col in df_sc_cfs_1.columns:
#                 n = 9 if col == 'IBT' else 8
#                 df_sc_cfs_1[col] = df_sc_cfs_1[col]*(12/dm.switch_int(sheet,n))
#             df_sc_cfs = df_sc_cfs_1.append(df_sc_cfs_2)
#         past_sc_dict[sheet] = df_sc_cfs
#     return past_sc_dict

# def get_plan_pbo_dict(past_pbo_filename = 'past_pbo_cashflow_data.xlsx', cf_type = 'PBO'):
#     """
    

#     Parameters
#     ----------
#     filename : TYPE, optional
#         DESCRIPTION. The default is 'past_pbo_cashflow_data.xlsx'.

#     Returns
#     -------
#     plan_pbo_dict : TYPE
#         DESCRIPTION.

#     """
#     past_pbo_dict = get_past_pbo_data(past_pbo_filename)
#     plan_pbo_dict = {}
#     for plan in PLAN_LIST:
#         temp_pbo_df = pd.DataFrame()
#         #merge past years pbos
#         for key in past_pbo_dict:
#             temp_pbo_df = dm.merge_dfs(temp_pbo_df, past_pbo_dict[key][plan],dropna = False)
        
#         #merge current years pbos
#         df_pbo_cfs = dm.get_cf_data(cf_type)[[plan]]
#         temp_pbo_df = dm.merge_dfs(temp_pbo_df, df_pbo_cfs, dropna = False)
        
#         #rename dataframe
#         temp_pbo_df.columns = SHEET_LIST_1
#         plan_pbo_dict[plan] = temp_pbo_df
#     return plan_pbo_dict


# def get_plan_sc_dict(past_sc_filename = 'past_sc_cashflow_data.xlsx', cf_type = 'Service Cost'):
#     """
    

#     Parameters
#     ----------
#     filename : TYPE, optional
#         DESCRIPTION. The default is 'past_pbo_cashflow_data.xlsx'.

#     Returns
#     -------
#     plan_pbo_dict : TYPE
#         DESCRIPTION.

#     """
#     past_sc_dict = get_past_sc_data(past_sc_filename)
#     plan_sc_dict = {}
#     for plan in PLAN_LIST:
#         temp_sc_df = pd.DataFrame()
#         #merge past years scs
#         for key in past_sc_dict:
#             temp_sc_df = dm.merge_dfs(temp_sc_df, past_sc_dict[key][plan],dropna = False)
        
#         #merge current years scs
#         df_sc_cfs = dm.get_cf_data(cf_type)[[plan]]
#         temp_sc_df = dm.merge_dfs(temp_sc_df, df_sc_cfs, dropna = False)
        
#         #rename dataframe
#         temp_sc_df.columns = SHEET_LIST_1
#         plan_sc_dict[plan] = temp_sc_df
#     return plan_sc_dict


# def get_plan_sc_dict_old(plan_pbo_dict, filename='past_sc_cashflow_data.xlsx'):
#     """
    

#     Parameters
#     ----------
#     plan_pbo_dict : TYPE
#         DESCRIPTION.
#     filename : TYPE, optional
#         DESCRIPTION. The default is 'past_sc_cashflow_data.xlsx'.

#     Returns
#     -------
#     plan_sc_dict : TYPE
#         DESCRIPTION.

#     """
#     plan_sc_dict = {}
#     for plan in PLAN_LIST:
#         n = 9 if plan == 'IBT' else 8
#         sc_df = plan_pbo_dict[plan].copy()
#         # Compute service cost for past years by subtracting year_x pbo from year_x-1
#         for x in range(1,len(SHEET_LIST_1)):
#             temp_df_list = [sc_df.iloc[0:12*x],sc_df.iloc[12*x:,]]
#             temp_df_list[1].fillna(0, inplace=True)
#             for df in temp_df_list:
#                 df[SHEET_LIST_1[x-1]] = (df[SHEET_LIST_1[x]] - 
#                                       df[SHEET_LIST_1[x-1]])/dm.switch_int(SHEET_LIST_1[x-1],n)
#             temp_df = temp_df_list[0].append(temp_df_list[1])
#             sc_df[SHEET_LIST_1[x-1]] = temp_df[SHEET_LIST_1[x-1]]
#         # get service cost for current year
#         sc_df[SHEET_LIST_1[x]] = dm.get_cf_data('Service Cost')[plan]/12
#         # sc_df[SHEET_LIST[x]] *= 12/5
#         # Replace service cost data for 2021 for IBT plan
#         if plan == 'IBT':
#             sc_df.drop(['2021'], axis=1, inplace=True)
#             df_sc_cfs = pd.read_excel(TS_FP+filename, 
#                                       sheet_name='2021', index_col=0)/12
#             df_sc_cfs = dm.reindex_to_monthly_data(df_sc_cfs)[[plan]]/12
#             df_sc_cfs.columns = ['2021']
#             sc_df = dm.merge_dfs(sc_df, df_sc_cfs, dropna= False)
#         plan_sc_dict[plan] = sc_df[SHEET_LIST_1]
#     return plan_sc_dict


# def generate_liab_mv_dict(past_pbo_filename = 'past_pbo_cashflow_data.xlsx', past_sc_filename='past_sc_cashflow_data.xlsx',
#                           ftse_filename = 'ftse_data.xlsx'):
#     """
    

#     Parameters
#     ----------
#     past_pbo_filename : TYPE, optional
#         DESCRIPTION. The default is 'past_pbo_cashflow_data.xlsx'.
#     past_sc_filename : TYPE, optional
#         DESCRIPTION. The default is 'past_sc_cashflow_data.xlsx'.
#     ftse_filename : TYPE, optional
#         DESCRIPTION. The default is 'ftse_data.xlsx'.

#     Returns
#     -------
#     liab_mv_dict : TYPE
#         DESCRIPTION.

#     """
#     plan_pbo_dict = get_plan_pbo_dict(past_pbo_filename)
#     plan_sc_dict = get_plan_sc_dict(past_sc_filename)
    
#     liab_mv_dict = {}
#     for key in plan_pbo_dict:
#         year_dict = {}
#         n = 9 if key == 'IBT' else 8
        
#         for year in plan_pbo_dict[key].columns:
#             temp_pbo_df = dm.transform_pbo_df(plan_pbo_dict[key][year])
#             #Add below to transform function and make it for temp_cfs
#             temp_sc_df = plan_sc_dict[key][year]
#             temp_sc_df.fillna(0, inplace=True)
#             temp_cfs_df = dm.merge_dfs(temp_pbo_df, temp_sc_df)
#             temp_cfs_df.columns = ['PBO', 'SC']
#             if year == SHEET_LIST_1[-1]:
#                 no_of_cols = len(dm.get_liab_mv_cf_cols(ftse_filename))%12-1
#                 # if no_of_cols == 0:
#                 #     no_of_cols = 12
#             else:
#                 no_of_cols = dm.switch_int(year, n)-1
#             liab_cfs = pd.DataFrame(columns=list(range(0,no_of_cols+1)), index=temp_cfs_df.index)
            
#             for col in liab_cfs.columns:
#                 # if key == 'IBT' and year == '2021' and col == 2:
#                 #     liab_cfs[col] = dm.transform_pbo_df(plan_pbo_dict[key]['2021_1'])
#                 # else:    
#                     # liab_cfs[col] = temp_cfs_df['PBO'] + temp_cfs_df['SC']*col
#                 liab_cfs[col] = temp_cfs_df['PBO'] + temp_cfs_df['SC']
#                 jump = 11-no_of_cols if year == '2021_1'else 0
                
#                 liab_cfs.loc[:col+jump,col] = 0
#             year_dict[year] = liab_cfs
#         liab_mv_dict[key] = year_dict
#     return liab_mv_dict

# def generate_liab_cf_dict(past_pbo_filename = 'past_pbo_cashflow_data.xlsx', past_sc_filename='past_sc_cashflow_data.xlsx',
#                           ftse_filename = 'ftse_data.xlsx'):
#     """
    

#     Parameters
#     ----------
#     past_pbo_filename : TYPE, optional
#         DESCRIPTION. The default is 'past_pbo_cashflow_data.xlsx'.
#     past_sc_filename : TYPE, optional
#         DESCRIPTION. The default is 'past_sc_cashflow_data.xlsx'.
#     ftse_filename : TYPE, optional
#         DESCRIPTION. The default is 'ftse_data.xlsx'.

#     Returns
#     -------
#     liab_mv_dict : TYPE
#         DESCRIPTION.

#     """
#     plan_pbo_dict = get_plan_pbo_dict(past_pbo_filename)
#     plan_sc_dict = get_plan_sc_dict(past_sc_filename)
    
#     # liab_cf_dict = {}
#     liab_pbo_cf_dict = {}
#     liab_sc_cf_dict = {}
#     for key in plan_pbo_dict:
#         # year_dict = {}
#         pbo_year_dict = {}
#         sc_year_dict = {}
        
#         n = 9 if key == 'IBT' else 8
        
#         for year in plan_pbo_dict[key].columns:
#             temp_pbo_df = dm.transform_pbo_df(plan_pbo_dict[key][year])
#             #Add below to transform function and make it for temp_cfs
#             temp_sc_df = plan_sc_dict[key][year]
#             temp_sc_df.fillna(0, inplace=True)
#             temp_cfs_df = dm.merge_dfs(temp_pbo_df, temp_sc_df)
#             temp_cfs_df.columns = ['PBO', 'SC']
#             if year == SHEET_LIST_1[-1]:
#                 no_of_cols = len(dm.get_liab_mv_cf_cols(ftse_filename))%12-1
#                 # if no_of_cols == 0:
#                 #     no_of_cols = 12
#             else:
#                 no_of_cols = dm.switch_int(year, n)-1
#             liab_cfs = pd.DataFrame(columns=list(range(0,no_of_cols+1)), index=temp_cfs_df.index)
#             liab_pbo_cfs = pd.DataFrame(columns=list(range(0,no_of_cols+1)), index=temp_cfs_df.index)
#             liab_sc_cfs = liab_cfs.copy()
            
#             for col in liab_pbo_cfs.columns:
#                 # if key == 'IBT' and year == '2021' and col == 2:
#                 #     liab_cfs[col] = dm.transform_pbo_df(plan_pbo_dict[key]['2021_1'])
#                 # else:    
#                     # liab_cfs[col] = temp_cfs_df['PBO'] + temp_cfs_df['SC']*col
#                 # liab_cfs[col] = temp_cfs_df['PBO'] + temp_cfs_df['SC']
#                 liab_pbo_cfs[col] = temp_cfs_df['PBO']
#                 liab_sc_cfs[col] = temp_cfs_df['SC']
                
#                 jump = 11-no_of_cols if year == '2021_1'else 0
                
#                 # liab_cfs.loc[:col+jump,col] = 0
#                 liab_pbo_cfs.loc[:col+jump,col] = 0
#                 liab_sc_cfs.loc[:col+jump,col] = 0
#             # year_dict[year] = liab_cfs
#             pbo_year_dict[year] = liab_pbo_cfs
#             sc_year_dict[year] = liab_sc_cfs
#         # liab_cf_dict[key] = year_dict
#         liab_pbo_cf_dict[key] = pbo_year_dict
#         liab_sc_cf_dict[key] = sc_year_dict
#     return {
#         # 'cfs':liab_cf_dict,
#         'PBO':liab_pbo_cf_dict, 'SC':liab_sc_cf_dict}

# def get_plan_mv_cfs_dict(past_pbo_filename = 'past_pbo_cashflow_data.xlsx', past_sc_filename='past_sc_cashflow_data.xlsx',
#                          ftse_filename='ftse_data.xlsx'):
#     liab_mv_dict = generate_liab_mv_dict(past_pbo_filename, past_sc_filename,ftse_filename)
#     plan_mv_cfs_dict = {}
#     for plan in liab_mv_dict:
#         mv_cfs_df = pd.DataFrame()
#         #merge past years pbos
#         for year in liab_mv_dict[plan]:
#             mv_cfs_df = dm.merge_dfs(mv_cfs_df, liab_mv_dict[plan][year],dropna = False)
#         mv_cfs_df.columns = dm.get_liab_mv_cf_cols(ftse_filename, len(mv_cfs_df.columns))
#         mv_cfs_df.fillna(0, inplace=True)
#         plan_mv_cfs_dict[plan] = mv_cfs_df
#     plan_mv_cfs_dict['Total'] = dm.aggregate_mv_cfs(plan_mv_cfs_dict)
    # return plan_mv_cfs_dict

def get_plan_cfs_dict(cf_dict,ftse_filename='ftse_data.xlsx'):
    plan_mv_cfs_dict = {}
    for plan in cf_dict:
        mv_cfs_df = pd.DataFrame()
        #merge past years pbos
        for year in cf_dict[plan]:
            mv_cfs_df = dm.merge_dfs(mv_cfs_df, cf_dict[plan][year],dropna = False)
        mv_cfs_df.columns = dm.get_liab_mv_cf_cols(ftse_filename, len(mv_cfs_df.columns))
        mv_cfs_df.fillna(0, inplace=True)
        plan_mv_cfs_dict[plan] = mv_cfs_df
    plan_mv_cfs_dict['Total'] = dm.aggregate_mv_cfs(plan_mv_cfs_dict)
    return plan_mv_cfs_dict

def offset(pbo_cfs):
    #make a copy of the data
    data = pbo_cfs.copy()

    #loop through each period and offset first n rows of 0's to the end
    for i in range(1,len(data.columns)):
        #get discount factor for the period
        disc_rate = i
        #make a list of the cashflows
        cfs = list(data.iloc[:,i])
        #removes top discount amount of rows and adds to the bottom of the list
        cfs = cfs[disc_rate:] + cfs[:disc_rate] 
        #replaces column with new offset data
        data.iloc[:,i] = cfs
    return(data)

def compute_pvs(plan_cfs,liab_curve, disc_factors):
    pv_dict={}
    for col in plan_cfs:
        # print(col)
        temp_pv = 0
        for j in range (0,len(plan_cfs)):
            temp_pv += (plan_cfs[col][j]/((1+liab_curve[col][j]/100)**disc_factors[j]))
        pv_dict[col] = temp_pv
    return pd.DataFrame.from_dict(pv_dict, orient='index', columns=['Present Value'])


def npv(irr, cfs, yrs):  
    return np.sum(cfs / (1. + irr) ** yrs)

def irr(cfs, yrs, x0, **kwargs):
    return np.ndarray.item(fsolve(npv, x0=x0,args=(cfs,yrs), **kwargs))

def compute_irr(present_values,plan_cfs, disc_factors):
    irr_dict = {}
    for col in plan_cfs:
        cfs = np.append(np.negative(present_values['Present Value'][col]),plan_cfs[col])
        yrs = np.append(0, disc_factors)
        irr_dict[col] = irr(cfs, yrs, .03)
    return pd.DataFrame.from_dict(irr_dict, orient='index', columns=['IRR'])

def get_freq_factor(arg):

    switcher = {
            "Q": 3,
            "Y": 12,
    }
    return switcher.get(arg, 1)

def get_liab_ret(present_values, plan_sc_cfs, freq='M'):
    ret_string =freq+'TD'
    temp_ret = plan_sc_cfs.iloc[0:1].transpose()
    temp_ret.columns = ['SC accruals']
    temp_ret = dm.merge_dfs(present_values, temp_ret)
    temp_ret[ret_string] = temp_ret['Present Value'].shift()
    
    for i in range(1, len(temp_ret)):
        j = i-1 if freq=='M' else int(i/get_freq_factor(freq))*get_freq_factor(freq)
        j = j-get_freq_factor(freq) if j==i else j
        temp_ret[ret_string][i] = (temp_ret['Present Value'][i]+temp_ret['SC accruals'].iloc[j:i].sum())/temp_ret['Present Value'][j]-1
    temp_ret.dropna(inplace=True)
    return temp_ret[[ret_string]]


def compute_liab_ret(present_values, plan_sc_cfs, freq_list = ['M', 'Q', 'Y']):
    liab_ret_df = pd.DataFrame(index=present_values.index)
    for freq in freq_list:
        liab_ret_df = dm.merge_dfs(liab_ret_df, get_liab_ret(present_values, plan_sc_cfs, freq))
    return liab_ret_df

def compute_funded_status(asset_mv, liab_mv):
    #compute funded status: asset/liability
    fs_df = dm.merge_dfs(asset_mv,liab_mv)
    fs_df['Funded Status'] = asset_mv / liab_mv
    fs_df.dropna(inplace=True)
    fs_df.columns = ['Asset MV','Liability MV','Funded Status']
    
    #compute funded status gap: liability(i.e PBO) - asset
    fs_df['Funded Status Gap'] = liab_mv - asset_mv   
    fs_df.dropna(inplace=True)
    
    #compute funded status difference between each date
    gap_diff = fs_df['Funded Status Gap'].diff()
       
    #compute funded status gap difference percent: funded status gap/liability
    gap_diff_percent = gap_diff/liab_mv['Market Value']
    # gap_diff_percent = gap_diff/liab_mv
       
    #compute fs vol 
    gap_diff_percent.dropna(inplace=True)
    fs_df['1Y FSV'] = gap_diff_percent.rolling(window = 12).apply(get_ann_vol)
    fs_df['6mo FSV'] = gap_diff_percent.rolling(window = 6).apply(get_ann_vol)
      
    return fs_df