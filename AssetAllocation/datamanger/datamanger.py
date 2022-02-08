# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 12:01:39 2021

@author: Powis Forjoe
"""

import os
import pandas as pd
import numpy as np

from itertools import count, takewhile
import scipy as sp
from scipy import interpolate

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')


# set filepath
CWD = os.getcwd()
DATA_FP = CWD + '\\data\\'
MV_INPUTS_FP = DATA_FP + 'mv_inputs\\'
TS_FP = DATA_FP + 'time_series\\'
PLAN_INPUTS_FP = DATA_FP + 'plan_inputs\\'
UPPER_BND_LIST = ['15+ STRIPS', 'Long Corporate', 'Equity', 'Liquid Alternatives']

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

def get_mkt_factor_premiums(filename):
    mkt_factor_prem = pd.read_excel(filename, sheet_name='mkt_factor_prem', index_col=0)
    mkt_factor_prem.fillna(0,inplace=True)
    # mkt_factor_prem.dropna(inplace=True)
    return mkt_factor_prem['Market Factor Premium'].to_dict()

def merge_dfs(main_df, new_df):
    merged_df = pd.merge(main_df, new_df,left_index=True, right_index=True, how='outer')
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

def get_asset_returns(filename='return_data.xlsx', year='2010'):
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
                             sheet_name=year, index_col=0)
    # returns_df['Credit'] = 0.2*returns_df['CS LL'] + 0.3*returns_df['BOA HY'] + 0.5*returns_df['CDLI']
    # returns_df['Liquid Alternatives'] = 0.33*returns_df['HF MACRO'] + 0.33*returns_df['HFRI MACRO'] + 0.34*returns_df['TREND']
    asset_ret_df = asset_ret_df[['15+ STRIPS', 'Long Corps', 'WN1 COMB Comdty', 'Total Dom Eq w/o Derivatives', 'Total Liquid Alts',
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

def get_ts_data(plan='IBT', year='2010'):
    # returns_df = get_returns_df(plan=plan, year=year)
    returns_df = get_asset_returns(year=year)
    weights_df = get_weights(plan=plan)
    return {'returns': returns_df,
            'weights': weights_df}

def get_bounds(liab_model, filename='bounds.xlsx', plan='IBT'):
    filepath=PLAN_INPUTS_FP+filename
    bnds = pd.read_excel(filepath, sheet_name=plan, index_col=0)
    update_bnds_with_fs(bnds,liab_model)
    return bnds

def transform_bnds(bnds):
    return tuple(zip(bnds.Lower, bnds.Upper))

def update_bnds_with_fs(bnds, liab_model):
    bnds *= liab_model.funded_status
    bnds['Upper']['Liability'] /= liab_model.funded_status
    bnds['Lower']['Liability'] /= liab_model.funded_status
    return None

def get_ports_df(rets, vols, weights, symbols, raw=True):
    if raw:
        return pd.DataFrame(np.column_stack([rets, vols, rets/vols,weights]),columns=['Return', 'Volatility', 'Sharpe'] + symbols).rename_axis('Portfolio')
    else:
        return pd.DataFrame(np.column_stack([100*np.around(rets,6), 100*np.around(vols,6), np.around(rets/vols,6),100*np.around(weights,6)]),
                        columns=['Return', 'Volatility', 'Sharpe'] + symbols).rename_axis('Portfolio')
    
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
    data = data.dropna()
    data = data.loc[(data!=0).any(1)]
    return data

def compute_fs(plan='IBT'):
    liab_pv = pd.read_excel(TS_FP+'liability_return_data.xlsx', sheet_name=plan,index_col=0,usecols=[0,2])
    asset_mv = pd.read_excel(TS_FP+'plan_mkt_value_data.xlsx', sheet_name=plan, index_col=0)
    fs_df = merge_dfs(asset_mv, liab_pv)
    return fs_df.iloc[-1:]['Market Value'][0]/fs_df.iloc[-1:]['Present Value'][0]

def get_plan_asset_mv(plan='IBT'):
    asset_mv = pd.read_excel(TS_FP+'plan_data.xlsx', sheet_name='mkt_value', index_col=0)
    return asset_mv.iloc[-1:][plan][0]

#TODO use this method to get all plan data
def get_plan_data():
    plan_mv_df = pd.read_excel(TS_FP+'plan_data.xlsx', sheet_name='mkt_value', index_col=0)
    plan_ret_df = pd.read_excel(TS_FP+'plan_data.xlsx', sheet_name='return', index_col=0)
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

def get_cf_data(cf_type='PBO'):
    df_cfs = pd.read_excel(TS_FP+'annual_cashflows_data.xlsx', sheet_name=cf_type, index_col=0)/12
    df_cfs = reindex_to_monthly_data(df_cfs)
    temp_cfs = pd.read_excel(TS_FP+'monthly_cashflows_data.xlsx', sheet_name=cf_type, index_col=0)
    df_cfs = temp_cfs.append(df_cfs)
    set_cfs_time_col(df_cfs)
    return df_cfs

def get_ftse_data(include_old=True):
    df_ftse = pd.read_excel(TS_FP+'ftse_data.xlsx',sheet_name='new_data', index_col=0)
    
    if include_old:
        df_old_ftse = pd.read_excel(TS_FP+'ftse_data.xlsx',sheet_name='old_data', index_col=0)
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

#TODO: have option to compute using disc_rates
def get_liab_model_data(plan='IBT', contrb_pct=.05):
    df_pbo_cfs = get_cf_data('PBO')
    # df_pvfb_cfs = get_cf_data('PVFB')
    # df_sc_cfs = df_pvfb_cfs - df_pbo_cfs
    df_sc_cfs = get_cf_data('Service Cost')
    df_ftse = get_ftse_data()
    # disc_rates = pd.read_excel(TS_FP+"discount_rate_data.xlsx",sheet_name=plan ,usecols=[0,1],index_col=0)
    disc_rates = pd.DataFrame()
    liab_curve = generate_liab_curve(df_ftse, np.array(df_pbo_cfs[plan]))
    asset_mv = get_plan_asset_mv(plan)
    return {'pbo_cashflows': df_pbo_cfs[plan], 'disc_factors':df_pbo_cfs['Time'], 'sc_cashflows': df_sc_cfs[plan],
            'liab_curve': liab_curve, 'disc_rates':disc_rates, 'contrb_pct':contrb_pct, 'asset_mv': asset_mv}
    
