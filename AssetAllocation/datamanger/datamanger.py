# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 12:01:39 2021

@author: NVG9HXP
"""

# Import pandas
import os
import pandas as pd
import numpy as np
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')


# set filepath
CWD = os.getcwd()
DATA_FP = CWD + '\\data\\'
MV_INPUTS_FP = DATA_FP + 'mv_inputs\\'
TS_FP = DATA_FP + 'time_series\\'
PLAN_INPUTS_FP = DATA_FP + 'plan_inputs\\'


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

def get_mv_inputs_data(filename, plan='IBT'):
    
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
    weights_df = add_fs_load_col(pd.read_excel(filepath, sheet_name=plan,index_col=0),
                                 plan)
    weights_df['FS AdjWeights'] = weights_df['Weights'] * weights_df['FS Loadings']
    # weights_df = weights_df[['FS AdjWeights']]
    return weights_df

def get_ts_data(plan='IBT', year='2010'):
    returns_df = get_returns_df(plan=plan, year=year)
    weights_df = get_weights(plan=plan)
    return {'returns': returns_df,
            'weights': weights_df}

def get_bounds(filename='bounds.xlsx', plan='IBT'):
    filepath=PLAN_INPUTS_FP+filename
    bnds = pd.read_excel(filepath, sheet_name=plan, index_col=0)
    return tuple(zip(bnds.Lower, bnds.Upper))

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
    # return ports_df
    
def monthize_data(df):
    #set start date and end date
    start_date = df.index.min() - pd.DateOffset(day=0)
    end_date = df.index.max() + pd.DateOffset(day=31)

    #create new dataframe monthly index
    dates = pd.date_range(start_date, end_date, freq='M')
    dates.name = 'Date'

    #reindex yearly dataframe to monthly dataframe
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

def compute_fs(plan='IBT'):
    liab_pv = pd.read_excel(TS_FP+'liability_return_data.xlsx', sheet_name=plan,index_col=0,usecols=[0,2])
    asset_mv = pd.read_excel(TS_FP+'plan_mkt_value_data.xlsx', sheet_name=plan, index_col=0)
    fs_df = merge_dfs(asset_mv, liab_pv)
    return fs_df.iloc[-1:]['Market Value'][0]/fs_df.iloc[-1:]['Present Value'][0]

def add_fs_load_col(weights_df, plan='IBT'):
    fs = compute_fs(plan)
    weights_df['FS Loadings'] = np.nan
    for ind in weights_df.index:
        if ind == 'Liability':
            weights_df['FS Loadings'][ind] = 1
        else:
            weights_df['FS Loadings'][ind] = fs
    return weights_df