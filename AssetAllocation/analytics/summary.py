  # -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 12:27:19 2021

@author: Powis Forjoe
"""
from .mv_inputs import mv_inputs
from .plan_params import planParams
from .liability_model import liabilityModel
from ..datamanger import datamanger as dm
from .import ts_analytics as ts
from .import util

def get_mv_inputs(mv_inputs_dict, liab_model):
    weights_df = add_fs_load_col(mv_inputs_dict['weights'],liab_model)
    weights_df['FS AdjWeights'] = weights_df['Weights'] * weights_df['FS Loadings']
    return mv_inputs(mv_inputs_dict['ret_assump'],mv_inputs_dict['mkt_factor_prem'],
                     mv_inputs_dict['fi_data'],mv_inputs_dict['rsa_data'], 
                     mv_inputs_dict['rv_data'],mv_inputs_dict['vol_defs'], 
                     mv_inputs_dict['corr_data'],weights_df)

def get_mv_output(mv_inputs, mkt='Equity'):
    output_dict = mv_inputs.get_output(mkt)
    ret_vol_df = dm.merge_dfs(output_dict['Return'], output_dict['Volatility'])
    ret_vol_df = util.add_sharpe_col(ret_vol_df)
    return {'ret_vol': ret_vol_df, 
            'corr': output_dict['corr'],
            'weights': output_dict['weights']}

def get_ts_output(ts_dict,liab_model,decay_factor=0.98, t=1):
    returns_df = dm.merge_dfs(liab_model.returns_ts, ts_dict['returns'])
    ret_vol_df = ts.get_ret_vol_df(returns_df)
    corr_df = ts.compute_ewcorr_matrix(returns_df,decay_factor, t)
    weights_df = add_fs_load_col(ts_dict['weights'],liab_model)
    weights_df['FS AdjWeights'] = weights_df['Weights'] * weights_df['FS Loadings']
    return {'ret_vol':ret_vol_df, 
            'corr':corr_df,
            'weights':weights_df,
            'returns':returns_df, 
            }

def get_liab_model(plan='IBT', contrb_pct=.05):
    liab_input_dict = dm.get_liab_model_data(plan, contrb_pct)
    return liabilityModel(liab_input_dict['pbo_cashflows'], liab_input_dict['disc_factors'], 
                          liab_input_dict['sc_cashflows'], liab_input_dict['contrb_pct'], 
                          liab_input_dict['asset_mv'], liab_input_dict['liab_curve'], 
                          liab_input_dict['disc_rates'])

def get_pp_inputs(liab_model, plan='IBT', mkt='Equity'):
    #get return
    mv_inputs = get_mv_inputs(dm.get_mv_inputs_data(plan=plan), liab_model)
   
    #compute analytics using historical data
    pp_inputs = get_ts_output(dm.get_ts_data(plan=plan), liab_model)
    
    #change cash correlations to 0
    for asset in pp_inputs['corr'].columns:
        if asset != 'Cash':
            pp_inputs['corr'][asset]['Cash'] = 0
            pp_inputs['corr']['Cash'][asset] = 0
    
    #compute returns using buiild up approach
    ret_df = mv_inputs.compute_plan_return('Equity')
    ret_df['Return']['Liability'] = liab_model.ret
    
    #add return to pp_inputs
    for asset in ret_df.index:
        pp_inputs['ret_vol']['Return'][asset] = ret_df['Return'][asset]
    
    pp_inputs['ret_vol']=util.add_sharpe_col(pp_inputs['ret_vol'])
    return pp_inputs

def get_data_dict(dataset):
    policy_wgts = util.add_dimension(dataset['FS AdjWeights'])

    ret = dataset['Return']

    vol = util.add_dimension(dataset['Volatility'])

    corr = dataset.iloc[:, 3:].to_numpy()

    symbols = list(dataset.index.values)

    return {'policy_weights': policy_wgts, 'ret': ret, 'vol': vol, 'corr': corr, 'symbols': symbols}

def get_plan_params(output_dict):
    
    policy_wgts = util.add_dimension(output_dict['weights']['FS AdjWeights'])

    ret = output_dict['ret_vol']['Return']

    vol = util.add_dimension(output_dict['ret_vol']['Volatility'])

    corr = output_dict['corr'].to_numpy()

    symbols = list(ret.index.values)
    
    funded_status = output_dict['weights']['FS Loadings'][1]
    try:
        ret_df = output_dict['returns']
    except KeyError:
        ret_df=None
    return planParams(policy_wgts, ret, vol, corr, symbols, funded_status,ret_df)

def get_pp_dict(plan):
    return {'Policy Weights':dm.pd.DataFrame(plan.policy_wgts, index=plan.symbols, columns=['Weights']),
            'Asset/Liability Returns':dm.pd.DataFrame(plan.ret),
            'Asset/Liability Vol':dm.pd.DataFrame(plan.vol, index=plan.symbols, columns=['Volatility']),
            'Corr':dm.pd.DataFrame(plan.corr, index=plan.symbols, columns=plan.symbols),
            'Cov':dm.pd.DataFrame(plan.cov, index=plan.symbols, columns=plan.symbols),
        }

def add_fs_load_col(weights_df, liab_model):
    
    weights_df['FS Loadings'] = dm.np.nan
    for ind in weights_df.index:
        if ind == 'Liability':
            weights_df['FS Loadings'][ind] = 1
        else:
            weights_df['FS Loadings'][ind] = liab_model.funded_status
    return weights_df

def get_liab_model_dict(plan_list = ['Retirement', 'Pension', 'IBT',"Total"]):
    
    df_pbo_cfs = dm.get_cf_data('PBO')
    df_sc_cfs = dm.get_cf_data('Service Cost')
    
    df_ftse = dm.get_ftse_data(False)
    
    plan_data = dm.get_plan_data()
    disc_factors = df_pbo_cfs['Time']
    liab_curve = dm.generate_liab_curve(df_ftse, df_pbo_cfs["IBT"])
    
        
    liab_model_dict={}
    
    for pension_plan in plan_list:
        pbo_cashflows = df_pbo_cfs[pension_plan]
        sc_cashflows = df_sc_cfs[pension_plan]
        asset_mv = plan_data['mkt_value'][pension_plan]
        asset_returns = plan_data['return'][pension_plan]
        contrb_pct = 0.0
        liab_model = liabilityModel(pbo_cashflows, disc_factors, sc_cashflows, contrb_pct, asset_mv, asset_returns, liab_curve)
        del liab_model.data_dict['Cashflows']
        liab_model.data_dict['Asset/Liability Returns'] = dm.get_n_year_ret(liab_model)
        liab_model_dict[pension_plan] = liab_model.data_dict

    return liab_model_dict


def get_report_dict(plan_list = ['Retirement', 'Pension', 'IBT',"Total"]):
    liab_model_dict = get_liab_model_dict(plan_list)
    
    #set report dict keys by using first data_dict in liab_model_dict
    report_dict = liab_model_dict[plan_list[0]]
    
    #create asset_liab_dict
    asset_liab_ret_dict = {}
    asset_liab_ret_dict[plan_list[0]] = report_dict['Asset/Liability Returns']
    
    #loop through rest of plan_list and merge dataframes
    for plan in plan_list[1:]:
        for key in liab_model_dict[plan]:
            report_dict[key] = dm.merge_dfs(report_dict[key], liab_model_dict[plan][key])
        
        #add to asset_liab_dict
        asset_liab_ret_dict[plan] = liab_model_dict[plan]['Asset/Liability Returns']
            
    #delete 'Asset/Liability Returns' key from report_dict    
    del report_dict['Asset/Liability Returns']
    
    #rename columns in report_dict
    for key in report_dict:
        report_dict[key].columns = plan_list
    
    #add asset_liab_dict to report_dict
    report_dict['asset_liab_ret_dict'] = asset_liab_ret_dict   
    
    return report_dict