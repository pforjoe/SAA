# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 22:22:20 2021

@author: NVG9HXP
"""

import pandas as pd
import scipy.interpolate
import numpy as np
from numpy.linalg import multi_dot
import os

CWD = os.getcwd()
DATA_FP = CWD + '\\data\\'

#INPUTS
#broaad mkt ret assumptions
TSY_MKT_RET = .02
MKT_RISK_PREM = .04
FI_TERM_PREM = 0
BOND_OAS_CR = .75
MKT = 'S&P 500'
MKT_FACTOR_PREM_DICT = {'Ultra 30-Year UST Futures':-TSY_MKT_RET,
                      'MSCI Emerging Markets':.015,
                      'Global HF': .02}

ACWI_WEIGHTS_DICT = {'S&P 500':0.615206,
                'MSCI EAFE':0.265242,
                'MSCI Emerging Markets':0.119552}


def create_factor_wgts(filename,fi_df, asset_list):
    #Get vol definitions
    vol_df = pd.read_excel(DATA_FP+filename, sheet_name='vol_defs')
    
    #Add Factor Weightings
    for asset in fi_df.index:
        if asset == '15+ STRIPS':
            vol_df.loc[(vol_df['Factor Description'] == asset) & (vol_df['Fundamental Factor Group'] == 'FI Rates'), asset] = fi_df['Duration'][asset]  
            vol_df.loc[(vol_df['Factor Description'] != asset) | (vol_df['Fundamental Factor Group'] != 'FI Rates'), asset] = 0
        else:
            vol_df[asset]=vol_df['Factor Description'].apply(lambda x: fi_df['Duration'][asset] 
                                                                     if x.startswith(asset) else 0)        
    
    for asset in asset_list[len(fi_df.index):]:
        vol_df[asset] = vol_df['Factor Description'].apply(lambda x: 1 if x == asset else 0)
    
    return vol_df

def compute_vol_of_rate(asset, fi_df, rates_vol_df):
    x = list(rates_vol_df['Maturity'])
    y = list(rates_vol_df['Pros Rate Vol'])
    y_interp = scipy.interpolate.interp1d(x, y)
    return y_interp(fi_df['Duration'][asset])/10000
    
def compute_vol_of_spread(asset, fi_df):
    return fi_df['Current Vol'][asset]/10000
    
def get_rsa_vol(asset,rsa_df):
    return rsa_df['Prospective Vol'][asset]

def compute_fi_return(oas, oas_ratio, duration):
    return TSY_MKT_RET + oas*oas_ratio + FI_TERM_PREM*duration

def compute_rsa_return(beta):
    return TSY_MKT_RET + MKT_RISK_PREM*beta

def compute_beta(asset, vol_ret_df, corr_df, mkt = MKT):
    return (vol_ret_df['Vol'][asset]/vol_ret_df['Vol'][mkt]) * corr_df[mkt][asset]

def compute_vol_assump(vol_df, fi_df, rv_df, rsa_df):
    vol_assump = []
    for ind in vol_df.index:
        asset = vol_df['Factor Description'][ind]
        risk_unit = vol_df['Risk Unit'][ind]
        if vol_df['Fundamental Factor Group'][ind] == 'Liability':
            if risk_unit == 'Vol of Rate':
                vol_assump.append(compute_vol_of_rate('Liability', fi_df, rv_df))
            elif risk_unit == 'Vol of Spread':
                vol_assump.append(compute_vol_of_spread('Liability', fi_df))
        elif vol_df['Fundamental Factor Group'][ind] == 'Cash':
            vol_assump.append(0.000001)
        else:
            if risk_unit == 'Vol of Rate':
                vol_assump.append(compute_vol_of_rate(asset, fi_df, rv_df))
            elif risk_unit == 'Vol of Spread':
                vol_assump.append(compute_vol_of_spread(asset, fi_df))
            else:
                vol_assump.append(get_rsa_vol(asset, rsa_df))
    return vol_assump

def compute_cov(vol_assump, corr):
    vol = np.array(vol_assump)[:,np.newaxis]
    return (vol @ vol.T)*corr

def compute_plan_vol(vol_df, cov, asset_list):
    vol_list=[]
    for asset in asset_list:
        x = vol_df[asset].to_numpy()[:,np.newaxis]
        asset_vol = np.sqrt(multi_dot([x.T,cov,x]))
        vol_list.append(asset_vol[0][0])
    return vol_list

def compute_plan_corr(vol_list, cov,comp_weights):
    vol_array = np.array(vol_list)[:,np.newaxis]
    return multi_dot([comp_weights.T,cov,comp_weights]) / (vol_array*vol_array.T)

def compute_plan_return(fi_df,corr_df, output_df):
    ret_list=[]
    for asset in fi_df.index:
        oas = fi_df['Spread'][asset]
        duration = fi_df['Duration'][asset]
        oas_ratio = 1 if asset == 'Liability' else BOND_OAS_CR
        ret_list.append(compute_fi_return(oas, oas_ratio, duration))
    
    for asset in output_df.index[len(fi_df.index):]:
        beta = compute_beta(asset, output_df, corr_df)
        ret_list.append(compute_rsa_return(beta))
    
    return ret_list

def adjust_returns(output_df):
    for key in MKT_FACTOR_PREM_DICT:
        output_df['Return'][key] += MKT_FACTOR_PREM_DICT[key]
        
    new_acwi_ret = 0
    for key in ACWI_WEIGHTS_DICT:
        new_acwi_ret += output_df['Return'][key]*ACWI_WEIGHTS_DICT[key]
    
    output_df['Return']['MSCI ACWI'] = new_acwi_ret

def compute_sum_output(vol_df, weights_df, cov,fi_df):
    
    asset_list = list(weights_df.index)
    vol_list=compute_plan_vol(vol_df, cov,asset_list)
    
    comp_weights = vol_df.iloc[:,3:].to_numpy()
    
    #Compute plan correlations
    corr_array = compute_plan_corr(vol_list, cov,comp_weights)
    
    corr_df = pd.DataFrame(corr_array, columns=asset_list, index=asset_list)
    
    output_df = pd.DataFrame(vol_list, columns=['Vol'], index=weights_df.index)
    
    #Compute plan Return
    output_df['FS AdjWeights'] = weights_df['Weights'] * weights_df['Factor Loadings']
    output_df['Return'] = compute_plan_return(fi_df,corr_df, output_df)
    
    adjust_returns(output_df)
    
    output_df = pd.merge(output_df, corr_df,left_index=True, right_index=True, how='outer')
    return output_df

def get_summarized_output(filename):
    #Get FI inputs
    fi_df = pd.read_excel(DATA_FP+filename, sheet_name='fi', index_col=0)
    fi_df['Current Vol'] = fi_df['Historical Vol'] / fi_df['Historical Spread'] * fi_df['Current Spread']
    fi_df = fi_df.fillna(0)
    
    #Get RSA inputs
    rsa_df = pd.read_excel(DATA_FP+filename, sheet_name='rsa', index_col=0)
    
    #Get Rates Vol inputs
    rv_df = pd.read_excel(DATA_FP+filename, sheet_name='rates_vol', index_col=0)
    rv_df['Pros Rate Vol'] = rv_df['3mo Tsy Futures Options Vol'] * rv_df['12mo expiry'] / rv_df['3mo expiry']
    
    weights_df = pd.read_excel(DATA_FP+filename, sheet_name='weights', index_col=0)
    
    vol_df = create_factor_wgts(filename,fi_df, weights_df.index)
    
    
    #Compute Vol Assumptions
    vol_assump = compute_vol_assump(vol_df, fi_df, rv_df, rsa_df)
    
    #Compute Correlations (EWMA)
    corr = pd.read_excel(DATA_FP+filename, sheet_name='corr', index_col=0).to_numpy()
    
    #Compute Covariance
    cov = compute_cov(vol_assump, corr)
    
    #Compute plan Volatility
    return compute_sum_output(vol_df, weights_df, cov,fi_df)
    
