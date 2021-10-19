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
OUTPUTS_FP = DATA_FP + 'outputs\\'


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

def get_mv_inputs_data(filename):
    
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
    weights = pd.read_excel(filepath, sheet_name='weights', index_col=0)

    # get vol definitions
    vol_defs = pd.read_excel(filepath, sheet_name='vol_defs')

    # get corr
    corr_data = pd.read_excel(filepath, sheet_name='corr', index_col=0)

    return {'fi_data': fi_data, 'rsa_data': rsa_data, 'rv_data': rv_data,
            'vol_defs': vol_defs, 'weights': weights, 'corr_data': corr_data,
            'ret_assump': ret_assump, 'mkt_factor_prem': mkt_factor_prem}

def get_ret_assump(filename):
    ret_assump = pd.read_excel(filename, sheet_name='ret_assump', index_col=0)
    return ret_assump['Return'].to_dict()

def get_mkt_factor_premiums(filename):
    mkt_factor_prem = pd.read_excel(filename, sheet_name='mkt_factor_prem', index_col=0)
    mkt_factor_prem.dropna(inplace=True)
    return mkt_factor_prem['Market Factor Premium'].to_dict()

def merge_dfs(main_df, new_df):
    return pd.merge(main_df, new_df,left_index=True, right_index=True, how='outer')

def get_output_data(filename):
    
    output_list = ['ret_vol', 'corr', 'weights']
    output_dict = {}
    for sheet in output_list:
        output_dict[sheet] = pd.read_excel(OUTPUTS_FP+filename,
                                           sheet_name=sheet,
                                           index_col=0)
    return output_dict

def get_returns_df(filename, year='2011'):
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
    returns_df = pd.read_excel(filename,
                             sheet_name=year, index_col=0)
    returns_df['Credit'] = 0.2*returns_df['CS LL'] + 0.3*returns_df['BOA HY'] + 0.5*returns_df['CDLI']
    returns_df['Liquid Alternatives'] = 0.33*returns_df['HF MACRO'] + 0.33*returns_df['HFRI MACRO'] + 0.34*returns_df['TREND']
    returns_df = returns_df[['Liability', '15+ STRIPS', 'Long Corps', 'ULTRA 30Y FUTURES', 'Total EQ Unhedged', 'Liquid Alternatives',
                             'Total Private Equity', 'Credit', 'Total Real Estate', 'Total UPS Cash', 'Equity Hedges']]
    returns_df.columns = ['Liability', '15+ STRIPS', 'Long Corporate','Ultra 30-Year UST Futures', 'Equity', 'Liquid Alternatives',
                          'Private Equity', 'Credit', 'Real Estate', 'Cash', 'Equity Hedges']
    return returns_df

def get_weights(filename = 'weights.xlsx'):
    weights_df = pd.read_excel(filename, sheet_name='weights',index_col=0)
    weights_df['FS AdjWeights'] = weights_df['Weights'] * weights_df['Factor Loadings']
    # weights_df = weights_df[['FS AdjWeights']]
    return weights_df

def get_ts_data(filename):
    filepath = TS_FP + filename
    returns_df = get_returns_df(filepath)
    weights_df = get_weights(filepath)
    return {'returns': returns_df,
            'weights': weights_df}

def get_bounds(filename):
    bnds = pd.read_excel(filename, sheet_name='bnds', index_col=0)
    return tuple(zip(bnds.Lower, bnds.Upper))

def get_ports_df(rets, vols, weights, symbols, raw=True):
    if raw:
        return pd.DataFrame(np.column_stack([rets, vols, rets/vols,weights]),columns=['Return', 'Volatility', 'Sharpe'] + symbols).rename_axis('Portfolio')
    else:
        return pd.DataFrame(np.column_stack([100*np.around(rets,6), 100*np.around(vols,6), np.around(rets/vols,6),100*np.around(weights,6)]),
                        columns=['Return', 'Volatility', 'Sharpe'] + symbols).rename_axis('Portfolio')