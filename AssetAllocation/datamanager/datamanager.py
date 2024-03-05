# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 12:01:39 2021

@authors: Powis Forjoe, Maddie Choi
"""

import os
import pandas as pd
import numpy as np
from AssetAllocation.reporting import reports as rp
from AssetAllocation.analytics import ts_analytics as ts
# import statistics as stat

from itertools import count, takewhile
import scipy as sp
from scipy import interpolate


# Ignore warnings
import warnings
warnings.filterwarnings('ignore')
from AssetAllocation.analytics import util


# set filepath
CWD = os.getcwd()
DATA_FP = CWD + '\\data\\'
MV_INPUTS_FP = DATA_FP + 'mv_inputs\\'
TS_FP = DATA_FP + 'time_series\\'
PLAN_INPUTS_FP = DATA_FP + 'plan_inputs\\'
UPDATE_FP = DATA_FP + 'update_files\\'


UPPER_BND_LIST = ['15+ STRIPS', 'Long Corporate', 'Equity', 'Liquid Alternatives']
SHEET_LIST = ['2019','2020','2021','2021_1','2022','2023','2024']
PLAN_LIST = ['IBT','Pension', 'Retirement']
SHEET_LIST_LDI = ['2021','2022','2023','2024']


def get_fi_data(filename):
    """


    Parameters
    ----------
    filename : TYPE
        DESCRIPTION.

    Returns
    -------
    fi_df : TYPE
        DESCRIPTION.

    """
    fi_df = pd.read_excel(filename, sheet_name='fi', index_col=0)
    fi_df['Current Vol'] = fi_df['Historical Vol'] / \
        fi_df['Historical Spread'] * fi_df['Current Spread']
    fi_df = fi_df.fillna(0)
    return fi_df

def get_rv_data(filename):
    """


    Parameters
    ----------
    filename : TYPE
        DESCRIPTION.

    Returns
    -------
    rv_df : TYPE
        DESCRIPTION.

    """
    rv_df = pd.read_excel(filename, sheet_name='rates_vol', index_col=0)
    rv_df['Pros Rate Vol'] = rv_df['3mo Tsy Futures Options Vol'] * \
        rv_df['12mo expiry'] / rv_df['3mo expiry']
    return rv_df

def get_mv_inputs_data(filename='inputs_test.xlsx', plan='IBT'):
    
    filepath = MV_INPUTS_FP + filename
    
    ret_assump = get_ret_assump(filepath)

    mkt_factor_prem = get_mkt_factor_premiums(filepath)

    # Get FI inputs
    fi_data = get_fi_data(filepath)

    # Get RSA inputs
    rsa_data = pd.read_excel(filepath, sheet_name='rsa', index_col=0)

    # Get Rates Vol inputs
    rv_data = get_rv_data(filepath)

    # Get weights
    weights = format_weights_index(get_weights(plan=plan), list(mkt_factor_prem.keys()))
    
    # get vol definitions
    vol_defs = pd.read_excel(filepath, sheet_name='vol_defs')

    # get corr
    corr_data = pd.read_excel(filepath, sheet_name='corr', index_col=0)

    return {'fi_data': fi_data, 'rsa_data': rsa_data, 'rv_data': rv_data,
            'vol_defs': vol_defs, 'weights': weights, 'corr_data': corr_data,
            'ret_assump': ret_assump, 'mkt_factor_prem': mkt_factor_prem}

def format_weights_index(weights_df, index_list):
    weights_df_t = weights_df.transpose()
    weights_df_t = weights_df_t[index_list]
    new_weights_df = weights_df_t.transpose()
    return new_weights_df
    
def get_ret_assump(filename):
    ret_assump = pd.read_excel(filename, sheet_name='ret_assump', index_col=0)
    return ret_assump['Return'].to_dict()

def get_mkt_factor_premiums(filename, sheet_name = 'mkt_factor_prem'):
    mkt_factor_prem = pd.read_excel(filename, sheet_name,  index_col=0)
    mkt_factor_prem.fillna(0,inplace=True)
    # mkt_factor_prem.dropna(inplace=True)
    return mkt_factor_prem['Market Factor Premium'].to_dict()


def merge_dfs(main_df, new_df, dropna=True):
    merged_df = pd.merge(main_df, new_df, left_index = True, right_index = True, how = 'outer')
    if dropna:
        merged_df = merged_df.dropna()
    return merged_df
    
def get_plan_data(filename):
    
    plan_list = ['ret_vol', 'corr', 'weights']
    plan_dict = {}
    for sheet in plan_list:
        plan_dict[sheet] = pd.read_excel(PLAN_INPUTS_FP+filename,
                                           sheet_name=sheet,
                                           index_col=0)
    return plan_dict

def get_returns_df(plan='IBT', year='2011'):
    """


    Parameters
    ----------
    filename : TYPE
        DESCRIPTION.
    year : TYPE, optional
        DESCRIPTION. The default is '2011'.

    Returns
    -------
    returns_df : TYPE
        DESCRIPTION.

    """
    asset_ret_df = get_asset_returns(year=year)
    liab_ret_df = get_liab_returns(plan=plan)
    returns_df = merge_dfs(liab_ret_df, asset_ret_df)
    return returns_df

def get_asset_returns(filename='asset_return_data.xlsx', sheet_name = 'Monthly Historical Returns'):
    """


    Parameters
    ----------
    filename : TYPE
        DESCRIPTION.
    year : TYPE, optional
        DESCRIPTION. The default is '2011'.

    Returns
    -------
    returns_df : TYPE
        DESCRIPTION.

    """
    filepath = TS_FP+filename
    asset_ret_df = pd.read_excel(filepath,
                             sheet_name=sheet_name, index_col=0)
    # returns_df['Credit'] = 0.2*returns_df['CS LL'] + 0.3*returns_df['BOA HY'] + 0.5*returns_df['CDLI']
    # returns_df['Liquid Alternatives'] = 0.33*returns_df['HF MACRO'] + 0.33*returns_df['HFRI MACRO'] + 0.34*returns_df['TREND']
    asset_ret_df = asset_ret_df[['15+ STRIPS', 'Long Corps', 'WN1 COMB Comdty', 'Total EQ w/o Derivatives', 'Total Liquid Alts',
                             'Total Private Equity', 'Total Credit', 'Total Real Estate', 'Cash', 'Equity Hedges']]
    asset_ret_df.columns = ['15+ STRIPS', 'Long Corporate','Ultra 30Y Futures', 'Equity', 'Liquid Alternatives',
                          'Private Equity', 'Credit', 'Real Estate', 'Cash', 'Hedges']
    asset_ret_df = asset_ret_df.dropna()
    return asset_ret_df

def get_liab_returns(filename='liability_return_data.xlsx', plan='IBT'):
    filepath = TS_FP+filename
    liab_ret_df = pd.read_excel(filepath, sheet_name=plan, usecols=[0,1], index_col=0)
    liab_ret_df.columns = ['Liability']
    return liab_ret_df
    
def get_weights(filename = 'weights.xlsx', plan='IBT'):
    filepath = PLAN_INPUTS_FP+filename
    weights_df = pd.read_excel(filepath, sheet_name=plan,index_col=0)
    # weights_df = add_fs_load_col(weights_df,plan)
    # weights_df['FS AdjWeights'] = weights_df['Weights'] * weights_df['FS Loadings']
    # weights_df = weights_df[['FS AdjWeights']]
    return weights_df

def get_ts_data(plan='IBT'):
    # returns_df = get_returns_df(plan=plan, year=year)
    returns_df = get_asset_returns()
    weights_df = get_weights(plan=plan)
    return {'returns': returns_df,
            'weights': weights_df}

def get_bounds(funded_status, filename='bounds.xlsx', plan='IBT', unconstrained = False):
    filepath=PLAN_INPUTS_FP+filename
    if unconstrained:
        plan = 'Unbounded'
    bnds = pd.read_excel(filepath, sheet_name=plan, index_col=0)
    update_bnds_with_fs(bnds,funded_status)
    return bnds

def transform_bnds(bnds):
    return tuple(zip(bnds.Lower, bnds.Upper))

def update_bnds_with_fs(bnds, funded_status):
    bnds *= funded_status
    bnds['Upper']['Liability'] /= funded_status
    bnds['Lower']['Liability'] /= funded_status
    return None

def get_ports_df(rets, vols, asset_vols, weights, symbols, raw=True):
    if raw:
        return pd.DataFrame(np.column_stack([rets, vols, rets/vols, asset_vols, weights]),columns=['Return', 'Volatility', 'Sharpe', 'Asset Vol'] + symbols).rename_axis('Portfolio')
    else:
        return pd.DataFrame(np.column_stack([100*np.around(rets,6), 100*np.around(vols,6), np.around(rets/vols,6),100*np.around(weights,6)]),
                        columns=['Return', 'Volatility', 'Sharpe', 'Asset Vol'] + symbols).rename_axis('Portfolio')
    
def format_ports_df(ports_df, ret_df):
    #rename Return & Volatility column to Excess Return & Surplus Volatility
    ports_df.columns = [col.replace('Return', 'Excess Return') for col in ports_df.columns]
    ports_df.columns = [col.replace('Volatility', 'Surplus Volatility') for col in ports_df.columns]
    
    col_list = list(ports_df.columns)
    col_list.remove('Liability')
    
    #create Asset Return column
    ports_df['Asset Return'] = ret_df['Liability'] + ports_df['Excess Return']
    
    return ports_df[['Asset Return']+col_list]
    
def reindex_to_monthly_data(df):
    #set start date and end date
    start_date = df.index.min() - pd.DateOffset(day=0)
    end_date = df.index.max() + pd.DateOffset(months=11)

    #create new dataframe monthly index
    dates = pd.date_range(start_date, end_date, freq='M')
    dates.name = 'Date'

    #reindex dataframe to monthly dataframe
    df = df.reindex(dates, method='ffill')

    return df
    
def get_prices_df(df_returns):
    """"
    Converts returns dataframe to index level dataframe
    
    Parameters:
    df_returns -- returns dataframe
    
    Returns:
    index price level - dataframe
    """
    
    df_prices = df_returns.copy()
    
    for col in df_returns.columns:
        df_prices[col][0] = df_returns[col][0] + 1
    
    for i in range(1, len(df_returns)):
        for col in df_returns.columns:
            df_prices[col][i] = (df_returns[col][i] + 1) * df_prices[col][i-1]
    return df_prices

def format_data(df_index, freq="1M"):
    """
    Format dataframe, by freq, to return dataframe
    
    Parameters:
    df_index -- dataframe
    freq -- string ('1M', '1W', '1D')
    
    Returns:
    dataframe
    """
    data = df_index.copy()
    data.index = pd.to_datetime(data.index)
    if not(freq == '1D'):
       data = data.resample(freq).ffill()
    data = data.pct_change(1)
    # data = data.dropna()
    data = data.loc[(data!=0).any(1)]
    return data

#TODO:remove
def compute_fs(plan='IBT'):
    liab_pv = pd.read_excel(TS_FP+'liability_return_data.xlsx', sheet_name=plan,index_col=0,usecols=[0,2])
    asset_mv = pd.read_excel(TS_FP+'plan_mkt_value_data.xlsx', sheet_name=plan, index_col=0)
    fs_df = merge_dfs(asset_mv, liab_pv)
    return fs_df.iloc[-1:]['Market Value'][0]/fs_df.iloc[-1:]['Present Value'][0]

def get_plan_asset_mv(plan_data, plan='IBT'):
    asset_mv = pd.DataFrame(plan_data['mkt_value'][plan])
    asset_mv.columns = ['Market Value']
    return asset_mv

def get_plan_asset_returns(plan_data, plan='IBT'):
    asset_return = pd.DataFrame(plan_data['return'][plan])
    asset_return.columns = ['Return']
    return asset_return

def get_plan_asset_data(filename = 'plan_data.xlsx'):
    plan_mv_df = pd.read_excel(TS_FP+filename, sheet_name='mkt_value', index_col=0)
    plan_ret_df = pd.read_excel(TS_FP+filename, sheet_name='return', index_col=0)
    return {'mkt_value': plan_mv_df,
            'return': plan_ret_df}

def add_fs_load_col(weights_df, plan='IBT'):
    fs = compute_fs(plan)
    weights_df['FS Loadings'] = np.nan
    for ind in weights_df.index:
        if ind == 'Liability':
            weights_df['FS Loadings'][ind] = 1
        else:
            weights_df['FS Loadings'][ind] = fs
    return weights_df

def frange(start, stop, step):
    return takewhile(lambda x: x< stop, count(start, step))

def set_cfs_time_col(df_cfs):
    df_cfs['Time'] = list(frange(1/12, (len(df_cfs)+.9)/12, 1/12))

def get_cf_data(cf_type='PBO',annual_cf_filename ='annual_cashflows_data.xlsx',
                monthly_cf_filename = 'monthly_cashflows_data.xlsx'):
    df_cfs = monthize_cf_data(cf_type, annual_cf_filename)
    temp_cfs = pd.read_excel(TS_FP+monthly_cf_filename, sheet_name=cf_type, index_col=0)
    df_cfs = temp_cfs.append(df_cfs)
    set_cfs_time_col(df_cfs)
    return df_cfs

def monthize_cf_data(cf_type = 'PBO',filename = 'annual_cashflows_data.xlsx'):
    df_cfs = pd.read_excel(TS_FP+filename, sheet_name=cf_type, index_col=0)/12
    df_cfs = reindex_to_monthly_data(df_cfs)
    return df_cfs

def get_ftse_data(include_old = False, filename = 'ftse_data.xlsx'):
    df_ftse = pd.read_excel(TS_FP+filename,sheet_name='new_data', index_col=0)
    
    if include_old:
        df_old_ftse = pd.read_excel(TS_FP+filename,sheet_name='old_data', index_col=0)
        df_ftse = merge_dfs(df_ftse,df_old_ftse)
    df_ftse.reset_index(inplace=True)
    return df_ftse

def generate_liab_curve(df_ftse, cfs):
    liab_curve_dict = {}
    dates = df_ftse['Date']
    range_list =  list(frange(0.5, dates[len(dates)-1]+.9/12, (1/12)))
    
    for col in df_ftse.columns:
        y = []
        interp = sp.interpolate.interp1d(dates, df_ftse[col], bounds_error=False,
                                         fill_value=sp.nan)
        for step in range_list:
                value = float(interp(step))
                if not sp.isnan(value): # Don't include out-of-range values
                    y.append(value)
                    end_rate = [y[-1]] * (len(cfs) - len(range_list)-5)
                    start_rate = [y[0]] * 5
                liab_curve_dict[col] = start_rate + y + end_rate
    liab_curve_dict.pop('Date')
    liab_curve = pd.DataFrame(liab_curve_dict)
    liab_curve = liab_curve.iloc[:, ::-1]
    return liab_curve

def get_liab_model_data(plan='IBT', contrb_pct=.05):
    plan_asset_data = get_plan_asset_data()
    asset_mv = get_plan_asset_mv(plan_asset_data, plan)
    asset_ret = get_plan_asset_returns(plan_asset_data, plan)
    #only need total consolidation for asset data
    if plan == "Total Consolidation":
        plan = "Total"
    df_pbo_cfs = get_cf_data('PBO')
    df_sc_cfs = get_cf_data('Service Cost')
    df_ftse = get_ftse_data()
    liab_curve = generate_liab_curve(df_ftse, df_pbo_cfs[plan])
    plan_mv_cfs_dict = get_plan_mv_cfs_dict()
    return {'pbo_cashflows': df_pbo_cfs[plan], 'disc_factors':df_pbo_cfs['Time'], 'sc_cashflows': df_sc_cfs[plan],
            'liab_curve': liab_curve, 'contrb_pct':contrb_pct, 'asset_mv': asset_mv,
            'liab_mv_cfs':offset(plan_mv_cfs_dict[plan]),'asset_ret': asset_ret}




    
    
def get_n_year_df(liab_plan_data_dict, data='returns', n=3):
    
    if data == 'fs_data':
        n_year_df = liab_plan_data_dict['Funded Status'].copy()
    else:
        df_data_dict = switch_liab_dict(data)

        n_year_df = pd.DataFrame()

        for i in list(df_data_dict.keys())[0:-1]:
            n_year_df = merge_dfs(n_year_df,liab_plan_data_dict[df_data_dict[i]])

        #rename columns
        n_year_df.columns = df_data_dict['col_names']

    #returns most recent n years
    return n_year_df.iloc[-(n*12):,]

def switch_liab_dict(arg):

    switcher = {
            "returns": {'df1':'Asset Returns', 'df2':'Liability Returns', 'col_names':['Asset','Liability']},
            "mv_pv_irr": {'df1':'Asset Market Values', 'df2':'Present Values', 'df3': 'IRR', 'col_names':['Asset MV','Present Values', 'IRR']},
            "pv_irr": {'df1':'Present Values', 'df2':'IRR', 'col_names':['Present Values','IRR']},
            "ytd_returns": {'df1':'Asset YTD Returns', 'df2':'Liability YTD Returns', 'col_names':['Asset','Liability']},
            "qtd_returns": {'df1':'Asset QTD Returns', 'df2':'Liability QTD Returns', 'col_names':['Asset','Liability']},
    }
    return switcher.get(arg)

def offset(pbo_cfs):
    #make a copy of the data
    data = pbo_cfs.copy()

    #loop through each period and offset first n rows of 0's to the end
    for i in range(0,len(data.columns)):
        #get discount factor for the period
        disc_rate = i+1
        #make a list of the cashflows
        cfs = list(data.iloc[:,i])
        #removes top discount amount of rows and adds to the bottom of the list
        cfs = cfs[disc_rate:] + cfs[:disc_rate] 
        #replaces column with new offset data
        data.iloc[:,i] = cfs
    return(data)

def update_plan_data(report_name = 'Plan level Historical Returns.xls', sheet_name = 'Plan level Historical Returns'):
    '''
    

    Parameters
    ----------
    report_name : String
        DESCRIPTION. The default is 'monthly_plan_data.xlsx'.
    sheet_name : String
        DESCRIPTION. The default is 'data'.

    Returns
    -------
        Dict:
            Updated plan market values and returns 

    '''
    print('updating plan_data.xlsx')
    #read in monthly plan data
    plan_data = pd.read_excel(UPDATE_FP + report_name, sheet_name = sheet_name)
    
    #rename columns
    plan_data.columns = ["Account Name","Account Id","Return Type","Date", "Market Value","Monthly Return"]
    
    #rename each plan 
    plan_data["Account Name"].replace({"Total Retirement":"Retirement", "Total Pension":"Pension", "Total UPS/IBT FT Emp Pension":"IBT", 
                                       "LDI ONLY-TotUSPenMinus401H":"Total", "UPS GT Total Consolidation" : "Total Consolidation"}, inplace = True)
    
    #pivot table so that plans are in the columns and the rows are the market value/returns for each date
    mv_df = plan_data.pivot_table(values = 'Market Value', index='Date', columns='Account Name')
    
    #add back misc recievables
    add_misc_receiv(mv_df)
    
    ret_df = plan_data.pivot_table(values = 'Monthly Return', index='Date', columns='Account Name')
    #divide returns by 100
    ret_df /= 100
    
    plan_data_dict = {"mkt_value" : mv_df, "return":ret_df}
    
    rp.get_plan_data_report(plan_data_dict)

    # return(plan_data_dict)

def add_misc_receiv(mv_df, filename='misc_receivables_data.xlsx'):
    misc_receiv_df = pd.read_excel(TS_FP+filename, sheet_name='misc_receiv', index_col=0)
    mv_df['Total'] += misc_receiv_df['Misc Receivable']
    mv_df['Retirement'] += misc_receiv_df['Misc Receivable']

def get_new_ftse_data(file_name = 'ftse-pension-discount-curve.xlsx'):
    #read in new ftse data
    new_ftse = pd.read_excel(UPDATE_FP + file_name , sheet_name = 'Data - Current', skiprows= 25,header = 1)
    
    #get list of the rows needed to drop
    drop_rows = list(range(61, len(new_ftse))) + [0]
    
    #drop rows to only get ftse curve
    new_ftse.drop(labels = drop_rows ,axis = 0, inplace = True)
    
    #set index 
    new_ftse.set_index('Date', inplace = True)
    
    return(new_ftse)

def update_ftse_data(file_name = "ftse_data.xlsx"):
    print('updating {}'.format(file_name))
    #read in current ftse data
    prev_ftse = pd.read_excel(TS_FP + file_name, sheet_name = ['new_data','old_data'],index_col=0)

    #get new ftse data
    new_ftse = get_new_ftse_data()

    #create ftse dict for report
    ftse_dict = {'new_data' : new_ftse, 'old_data' :  prev_ftse['old_data']}
    
    #export report
    rp.get_ftse_data_report(ftse_dict, "ftse_data")
    
    return ftse_dict

def group_asset_liab_data(liab_data_dict, data='returns'):
    
    #create asset_liab_dict
    asset_liab_data_dict = {}
    
    #loop through each plan and get asset/liability table
    for key in liab_data_dict:
        asset_liab_data_dict[key] = get_n_year_df(liab_data_dict[key], data)
        
    return asset_liab_data_dict

def merge_liab_data_df(liab_data_dict, plan_list):
    
    #get report dict for first plan in plan list
    report_dict = liab_data_dict[plan_list[0]]
    
    #go through each plan and merge each data frames 
    for plan in plan_list[1:]:
        for key in liab_data_dict[plan]:
            report_dict[key] = merge_dfs(report_dict[key], liab_data_dict[plan][key])
    
    return report_dict

def transform_report_dict(report_dict, plan_list):
    report_dict_tx = {}
    for plan in plan_list:
        temp_dict = {}
        for key in report_dict:
            temp_dict[key] = report_dict[key][plan]
        report_dict_tx[plan] = temp_dict
    return report_dict_tx
    
def switch_int(arg,n):

    switcher = {
            "2021": 12-n,
            "2021_1": n,
    }
    return switcher.get(arg, 12)


def fill_dict_df_na(dict_df, value=0):
    for key in dict_df:
        dict_df[key].fillna(value, inplace=True)

def get_liab_mv_cf_cols(ftse_filename='ftse_data.xlsx'):
    #create empty liab_mv_cf df
    df_ftse = get_ftse_data(False,ftse_filename)
    dates = df_ftse.transpose().iloc[1:,]
    dates.sort_index(inplace=True)
    return list(dates.index[109:])

def transform_pbo_df(pbo_df):
    temp_df_list = [pbo_df.iloc[0:12* len(SHEET_LIST)],pbo_df.iloc[12*len(SHEET_LIST):,]]
    temp_df_list[1].fillna(0, inplace=True)
    temp_df = temp_df_list[0].append(temp_df_list[1])
    return temp_df

#TODO: combine past pbo cashflow for ldi w past pbo cashflow data
def get_past_pbo_data(filename = 'past_pbo_cashflow_data.xlsx'):
    past_pbo_dict = {}
    for sheet in SHEET_LIST[:-1]:
        df_pbo_cfs = pd.read_excel(TS_FP+'past_pbo_cashflow_data.xlsx', 
                                   sheet_name=sheet, index_col=0)/12
        df_pbo_cfs = reindex_to_monthly_data(df_pbo_cfs)
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

def get_plan_pbo_dict(filename = 'past_pbo_cashflow_data.xlsx'):
    past_pbo_dict = get_past_pbo_data(filename)
    plan_pbo_dict = {}
    for plan in PLAN_LIST:
        temp_pbo_df = pd.DataFrame()
        #merge past years pbos
        for key in past_pbo_dict:
            temp_pbo_df = merge_dfs(temp_pbo_df, past_pbo_dict[key][plan],dropna = False)

        #merge current years pbos
        temp_pbo_df = merge_dfs(temp_pbo_df, get_cf_data()[plan], dropna = False)

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
        sc_df[SHEET_LIST[x]] = get_cf_data('Service Cost')[plan]/12
        # sc_df[SHEET_LIST[x]] = 0
        if plan == 'IBT':
            sc_df.drop(['2021'], axis=1, inplace=True)
            df_sc_cfs = pd.read_excel(TS_FP+filename,
                                      sheet_name='2021', index_col=0)/12
            df_sc_cfs = reindex_to_monthly_data(df_sc_cfs)[[plan]]/12
            df_sc_cfs.columns = ['2021']
            sc_df = merge_dfs(sc_df, df_sc_cfs, dropna= False)
        plan_sc_dict[plan] = sc_df[SHEET_LIST]
    return plan_sc_dict

def generate_liab_mv_dict(past_pbo_filename = 'past_pbo_cashflow_data.xlsx', past_sc_filename='past_sc_cashflow_data.xlsx',
                          ftse_filename = 'ftse_data.xlsx'):
    plan_pbo_dict = get_plan_pbo_dict(past_pbo_filename)
    plan_sc_dict = get_plan_sc_dict(plan_pbo_dict, past_sc_filename)

    liab_mv_dict = {}
    for key in plan_pbo_dict:
        year_dict = {}
        n = 9 if key == 'IBT' else 8

        for year in plan_pbo_dict[key].columns:
            temp_pbo_df = transform_pbo_df(plan_pbo_dict[key][year])
            #Add below to transform function and make it for temp_cfs
            temp_sc_df = plan_sc_dict[key][year]
            temp_sc_df.fillna(0, inplace=True)
            temp_cfs_df = merge_dfs(temp_pbo_df, temp_sc_df)
            temp_cfs_df.columns = ['PBO', 'SC']
            if year == SHEET_LIST[-1]:
                #make no of cols 12 if using old pbos
                no_of_cols = len(get_liab_mv_cf_cols(ftse_filename))%12+1

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


def get_plan_mv_cfs_dict(past_pbo_filename = 'past_pbo_cashflow_data.xlsx', past_sc_filename='past_sc_cashflow_data.xlsx',
                         ftse_filename='ftse_data.xlsx'):
    liab_mv_dict = generate_liab_mv_dict(past_pbo_filename, past_sc_filename,ftse_filename)
    plan_mv_cfs_dict = {}
    for plan in liab_mv_dict:
        mv_cfs_df = pd.DataFrame()
        #change to 12 if want to use old pbos
        liab_mv_dict[plan]['2023'] =  liab_mv_dict[plan]['2023'].iloc[:,:11]
        #merge past years pbos
        for year in liab_mv_dict[plan]:
            mv_cfs_df = merge_dfs(mv_cfs_df, liab_mv_dict[plan][year],dropna = False)
        mv_cfs_df.columns = get_liab_mv_cf_cols(ftse_filename)
        mv_cfs_df.fillna(0, inplace=True)
        plan_mv_cfs_dict[plan] = mv_cfs_df
    plan_mv_cfs_dict['Total'] = aggregate_mv_cfs(plan_mv_cfs_dict)
    return plan_mv_cfs_dict

def get_plan_mv_file(plan_list = ['Retirement','IBT','Pension','Total'], file_name = "liab_mv_cfs.xlsx"):
    plan_mv_dict = {}
    for plan in plan_list:
        plan_mv_dict[plan] = pd.read_excel(TS_FP + file_name, sheet_name = plan, index_col=0)
        
    return plan_mv_dict 
        
def aggregate_mv_cfs(plan_mv_cfs_dict):
    agg_df = pd.DataFrame(index=plan_mv_cfs_dict[PLAN_LIST[0]].index, columns=plan_mv_cfs_dict[PLAN_LIST[0]].columns)
    agg_df.fillna(0, inplace=True)
    for plan in PLAN_LIST:
        agg_df = plan_mv_cfs_dict[plan] + agg_df
    return agg_df


def update_plan_mv():
    print('updating liab_mv_cfs.xlsx')
    #update plan liability market values
    plan_mv_cfs_dict = get_plan_mv_cfs_dict()
    
    #export report
    rp.get_liab_mv_cf_report(plan_mv_cfs_dict)

def update_ldi_data(update_plan_market_val = False):
    update_plan_data()
    if update_plan_market_val:
        update_plan_mv()
    update_ftse_data()
   
def transform_asset_returns(file_name = 'Historical Asset Class Returns.xls',
                            sheet_name = 'Historical Asset Class Returns'):
    hist_ret = pd.read_excel(DATA_FP + file_name, sheet_name = sheet_name)
   
    #rename columns
    hist_ret.columns = ["Account Name","Account Id","Return Type","Date", "Market Value","Monthly Return"]
    
    
    #pivot table so that plans are in the columns and the rows are the market value/returns for each date
    hist_ret_df = hist_ret.pivot_table(values = 'Monthly Return', index='Date', columns='Account Name')
    
    # hist_ret_df  = hist_ret_df.dropna()

    #divide returns by 100
    hist_ret_df /= 100
    return hist_ret_df

    # rp.get_monthly_returns_report(hist_ret_df, report_name = 'historical_asset_class_returns')
    
def transform_index_data(file_name = 'index_data.xlsx', sheet_name = 'data'):
    index_data = pd.read_excel(DATA_FP + file_name, sheet_name = sheet_name,index_col=0)
     
    #calculate returns
    index_returns = format_data(index_data)
    
    #rename columns
    index_returns.columns = ['15+ STRIPS', 'Long Corps', 'BNP 30Y ULTRA FUT', 'SP500',
                             'MSCI ACWI', 'RUSS2000', 'MSCI EAFE', 'MSCI EM', 'CS LL',
                             'BOA HY', 'HF MACRO', 'HFRI MACRO', 'TREND',
                             'ALT RISK', 'DW REIT', 'BXIIU3MC Index', 'USGB090Y Index',
                             'USBMMY3M Index', 'CDLI', 'MSCI ACWI IMI', 'RUSS3000',
                             'WN1 COMB Comdty', 'BARCLAYS ULTRA LONG FUT']
    
    
    return index_returns
    # rp.get_monthly_returns_report(index_returns, report_name = 'index_returns')

def transform_eq_hedges():
    #read in new hedge data
    eq_hedge_df = pd.read_excel(TS_FP+'equity_hedge_data.xlsx', sheet_name = 'Monthly Historical Returns', usecols=['Date','Weighted Hedges'], index_col=0)
    #rename columns
    eq_hedge_df.columns = ['Equity Hedges']
        
    return eq_hedge_df


def get_new_asset_returns():
    hist_ret_df = transform_asset_returns()
    index_returns = transform_index_data()
    eq_hedges = transform_eq_hedges()
    new_asset_ret_df = merge_dfs(index_returns, hist_ret_df)
    new_asset_ret_df['Cash'] = 1/600
    new_asset_ret_df = merge_dfs(new_asset_ret_df, eq_hedges)
    new_asset_ret_df.index.names = ['Date']
    return new_asset_ret_df

def update_asset_ret_data(file_name='asset_return_data.xlsx',sheet_name='Monthly Historical Returns'):
    asset_ret_data = pd.read_excel(TS_FP+file_name, sheet_name = sheet_name, index_col=0)
    new_asset_ret_df = get_new_asset_returns()
    new_asset_ret_df = new_asset_ret_df[list(asset_ret_data.columns)]
    
    new_asset_returns = update_ret_data_dates(asset_ret_data, new_asset_ret_df)
    
    try:
        asset_ret_data =  asset_ret_data.append(new_asset_returns)
    except KeyError:
        pass
    rp.get_monthly_returns_report(asset_ret_data, 'asset_return_data')
    

def update_ret_data_dates(ret_df, new_ret_df):
    #reset both data frames index
    new_ret_df.reset_index(inplace = True)
    ret_df.reset_index(inplace=True)
    
    #find difference in dates
    difference = set(new_ret_df.Date).difference(ret_df.Date)
    #find which dates in the new returns are not in the current returns data
    difference_dates = new_ret_df['Date'].isin(difference)
    
    #isolate
    new_ret_df = new_ret_df[difference_dates]
    
    #set index for both data frames
    new_ret_df.set_index('Date', inplace = True)
    ret_df.set_index('Date', inplace = True)
    
    return new_ret_df

def get_disc_factors(pbo_cf_data):
    disc_factor = [1/12]
    for i in list(range(1,len(pbo_cf_data.index))):
        disc_factor += [disc_factor[i-1]+1/12]
    return disc_factor

#consolidate 
def get_cf_dict_by_plan(filename = 'past_pbo_cashflow_data_for_ldi.xlsx' ):
    temp_data_dict = {}
    for year in SHEET_LIST_LDI:
        temp_data_dict[year] = monthize_cf_data(cf_type = year ,filename = filename)
        
    cf_data_dict = {}
    for plan in PLAN_LIST + ['Total']:
        temp_df = {}
        for year in SHEET_LIST_LDI:
            temp_df[year] = ( temp_data_dict[year][plan])

        cf_data_dict[plan] = temp_df
        
    return cf_data_dict
    
#TODO: fix/ split into multiple methods
def get_ldi_data(contrb_pct = 1):
    plan_asset_data = get_plan_asset_data()
    #only need total consolidation for asset data
    
    pbo_data_dict = get_cf_dict_by_plan(filename = 'pbo_cashflow_data_for_ldi.xlsx' )
    sc_data_dict= get_cf_dict_by_plan(filename = 'sc_cashflow_data_for_ldi.xlsx' )
    
    disc_factors = get_disc_factors(pbo_data_dict['IBT']['2021'])

    df_ftse = get_ftse_data()
    liab_curve = generate_liab_curve(df_ftse, pbo_data_dict['IBT'][SHEET_LIST_LDI[-1]])
    plan_mv_cfs_dict = get_plan_mv_cfs_dict()
    

    return {'pbo_cfs_dict': pbo_data_dict, 'sc_cfs_dict': sc_data_dict, 'disc_factors': disc_factors,
            'liab_curve': liab_curve, 'asset_mv': plan_asset_data['mkt_value'],'asset_ret': plan_asset_data['return'],
            'contrb_pct': contrb_pct, 'liab_mv_cfs_dict':plan_mv_cfs_dict}

def offset_df(pbo_cfs):
    #make a copy of the data
    data = pbo_cfs.copy()

    #loop through each period and offset first n rows of 0's to the end
    for i in range(0,len(data.columns)):
        #get discount factor for the period
        disc_rate = i
        #make a list of the cashflows
        cfs = list(data.iloc[:,i])
        #removes top discount amount of rows and adds to the bottom of the list
        cfs = cfs[disc_rate:] + cfs[:disc_rate] 
        #replaces column with new offset data
        data.iloc[:,i] = cfs
    return data


def switch_freq_int(arg):
    """
    Return an integer equivalent to frequency in years
    
    Parameters:
    arg -- string ('1D', '1W', '1M')
    
    Returns:
    int of frequency in years
    """

    switcher = {
            "1M": [1]*12,
            "1Q": [1,2,3]*4,
            "1Y": list(range(1,13)),
            }
    return switcher.get(arg, 1)


    

def get_no_cols(year):
    if year == SHEET_LIST_LDI[-1]:
        #if at december and want to use old PBOS, set no of cols to 12
        no_of_cols = len(get_liab_mv_cf_cols())%12
    else: 
        no_of_cols = 11
    return no_of_cols


def get_lookback_windows(df, freq):
    n = switch_freq_int(freq)

    date_df = pd.DataFrame(df.index.month, columns = ['Date'])
    date_df['Roll Window'] = 0
    
    #loop through each date in the pv_df
    for date in list(range(0,len(date_df))):
        #loop each month to see if it matches that in pv_df
        for month in list(range(1,13)):
            #if given date matches the month
            if date_df['Date'].loc[date] == month:
                #if given date matches the month, assign corresponding roll window
                date_df['Roll Window'].loc[date] = n[month-1]
    
    return date_df['Roll Window']


def get_future_sc(sc, n_years, contrib_pct = [], growth_factor = []):
    #get current service costs
    sc_df = sc.to_frame()
    # multiply previous years service cost by growth factor
    sc_df = sc_df * (1 + growth_factor[0])
    sc_df = sc_df * (1 - contrib_pct[0])

    for i in list(range(1, n_years+1)):
        #get future year
        year = str(int(SHEET_LIST_LDI[-1])+i)
        
        #get future year sc
        temp_df = pd.DataFrame()
        #multiply previous years service cost by growth factor
        temp_df[year] = sc_df.iloc[:,i-1].shift(12)*(1+growth_factor[i])
        #subtract contributions
        temp_df[year] = temp_df[year]*( 1- contrib_pct[i])
        temp_df.fillna(0, inplace = True)
        
        sc_df = sc_df.merge(temp_df, how = "outer", left_index=True, right_index=True)

    return sc_df.sum(axis = 1)



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    