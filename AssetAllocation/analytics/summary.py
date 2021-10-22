  # -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 12:27:19 2021

@author: NVG9HXP
"""
from .mv_inputs import mv_inputs
from .plan_params import plan_params as pp
from  ..datamanger import datamanger as dm
from .import ts_analytics as ts
from .util import add_dimension

def get_mv_inputs(mv_inputs_dict):
    return mv_inputs(mv_inputs_dict['ret_assump'], 
                     mv_inputs_dict['mkt_factor_prem'],
                     mv_inputs_dict['fi_data'], 
                     mv_inputs_dict['rsa_data'], 
                     mv_inputs_dict['rv_data'],
                     mv_inputs_dict['vol_defs'], 
                     mv_inputs_dict['corr_data'], 
                     mv_inputs_dict['weights'])

def get_mv_output(mv_inputs, mkt='Equity'):
    output_dict = mv_inputs.get_output(mkt)
    ret_vol_df = dm.merge_dfs(output_dict['Return'], output_dict['Vol'])
    ret_vol_df['Sharpe'] = ret_vol_df['Return']/ret_vol_df['Vol']
    return {'ret_vol': ret_vol_df, 
            'corr': output_dict['corr'],
            'weights': output_dict['weights']}

def get_ts_output(ts_dict,decay_factor=0.98, t=1):
    ret_vol_df = ts.get_ret_vol_df(ts_dict['returns'])
    corr_df = ts.compute_ewcorr_matrix(ts_dict['returns'],decay_factor, t)
    return {'ret_vol':ret_vol_df, 
            'corr':corr_df,
            'weights':ts_dict['weights']}

def get_pp_inputs(mv_inputs, ts_dict, mkt='Equity'):
    #compute analytics using historical data
    pp_inputs = get_ts_output(ts_dict)
    
    #change cash correlations to 0
    for asset in pp_inputs['corr'].columns:
        if asset != 'Cash':
            pp_inputs['corr'][asset]['Cash'] = 0
            pp_inputs['corr']['Cash'][asset] = 0
    
    #compute returns using buiild up approach
    ret_df = mv_inputs.compute_plan_return('Equity')
    
    #add return to pp_inputs
    for asset in ret_df.index:
        pp_inputs['ret_vol']['Return'][asset] = ret_df['Return'][asset]
    
    pp_inputs['ret_vol']['Sharpe'] = pp_inputs['ret_vol']['Return']/pp_inputs['ret_vol']['Vol']
    return pp_inputs

def get_data_dict(dataset):
    policy_wgts = add_dimension(dataset['FS AdjWeights'])

    ret = dataset['Return']

    vol = add_dimension(dataset['Vol'])

    corr = dataset.iloc[:, 3:].to_numpy()

    symbols = list(dataset.index.values)

    return {'policy_weights': policy_wgts, 'ret': ret, 'vol': vol, 'corr': corr, 'symbols': symbols}

def get_plan_params(output_dict):
    
    policy_wgts = add_dimension(output_dict['weights']['FS AdjWeights'])

    ret = output_dict['ret_vol']['Return']

    vol = add_dimension(output_dict['ret_vol']['Vol'])

    corr = output_dict['corr'].to_numpy()

    symbols = list(ret.index.values)
    return pp(policy_wgts, ret, vol, corr, symbols)

def get_pp_dict(plan):
    return {'Policy Weights':dm.pd.DataFrame(plan.policy_wgts, index=plan.symbols, columns=['Weights']),
            'Asset/Liability Returns':dm.pd.DataFrame(plan.ret),
            'Asset/Liability Vol':dm.pd.DataFrame(plan.vol, index=plan.symbols, columns=['Volatility']),
            'Corr':dm.pd.DataFrame(plan.corr, index=plan.symbols, columns=plan.symbols),
            'Cov':dm.pd.DataFrame(plan.cov, index=plan.symbols, columns=plan.symbols),
        } 