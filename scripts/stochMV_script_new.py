# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 21:26:55 2023

@author: RRQ1FYQ
"""


###############################################################################
# IMPORT LIBRARIES                                                            #
###############################################################################
import os
os.chdir("..")
from AssetAllocation.datamanager import datamanager as dm
from AssetAllocation.analytics import summary
from AssetAllocation.analytics.stoch_mv import stochMV
from AssetAllocation.reporting import plots, reports as rp
import numpy as np
import pandas as pd

###############################################################################
#                                                                             #
###############################################################################
# unconstrained True -->  all assets except cash has no bounds
unconstrained = False
# corp True --> STRIPS constrained to 10% lower than corporates
corp = False
# priv_mrp0 True --> private markets market risk premium set to 0
priv_mrp0 = False
# future_sc True --> account for future service cost assumptions
future_sc = True
# priv_vol_multiplier True --> multiply private market vols by x
priv_vol_multiplier = False

###############################################################################
#   Get LDI input dict                                                        #
###############################################################################

ldi_input_dict = dm.get_ldi_data()

for PLAN in ['IBT']:

    if future_sc:
        n_years = 3
        contrib_pct = [0]+[1]*(n_years)
        growth_factor = [0] * (n_years+1)
        ldi_input_dict['sc_cfs_dict'][PLAN][dm.SHEET_LIST_LDI[-1]] = dm.get_future_sc( ldi_input_dict['sc_cfs_dict'][PLAN][dm.SHEET_LIST_LDI[-1]] ,n_years,
                                                            contrib_pct, growth_factor)

    ###############################################################################
    # COMPUTE LIABILITY DATA                                                      #
    ###############################################################################
    liab_model = summary.get_liab_model_new(ldi_input_dict,PLAN, sc_accrual=False)

    asset_mv_pv = dm.merge_dfs(liab_model.asset_mv, liab_model.present_values)
    fs_data = {}
    fs_data[PLAN] = dm.merge_dfs(asset_mv_pv, liab_model.funded_status)
    fs_data[PLAN] = fs_data[PLAN].iloc[-37:,]
    ###############################################################################
    # COMPUTE PLAN INPUTS                                                         #
    ###############################################################################
    pp_inputs = summary.get_pp_inputs(liab_model,PLAN, priv_mrp0 = priv_mrp0)

    if priv_vol_multiplier:
        #multiplier on private asset class vols
        multiplier = 1
        pp_inputs['ret_vol']['Volatility'].loc['Private Equity'] = pp_inputs['ret_vol']['Volatility'].loc['Private Equity']*multiplier
        pp_inputs['ret_vol']['Volatility'].loc['Credit'] =  pp_inputs['ret_vol']['Volatility'].loc['Credit']*multiplier
        pp_inputs['ret_vol']['Volatility'].loc['Real Estate'] = pp_inputs['ret_vol']['Volatility'].loc['Real Estate']*multiplier

    ###############################################################################
    # INITIALIZE PLAN                                                             #
    ###############################################################################
    plan = summary.get_plan_params(pp_inputs)
    pp_dict2 = plan.get_pp_dict()
    
    ###############################################################################
    # INITIALIZE STOCHMV                                                          #
    ###############################################################################
    #initialize the stochastic mean variance
    s = stochMV(plan, 20)


    #generate the random returns Aand sample corr
    s.generate_plans()
    s.generate_resamp_corr_dict()
    ###############################################################################
    # VIEW CORRELATIONS                                                           #
    ###############################################################################
    # for key in s.resamp_corr_dict:
    #     resamp_corr_fig = plots.get_resamp_corr_fig(s.resamp_corr_dict[key], key)
    #     resamp_corr_fig.show()
        
    # ###############################################################################
    # # VIEW  RETURNS                                                               #
    # ###############################################################################
    # #visualize the simulated returns
    # plots.get_sim_return_fig(s)
    
    ###############################################################################
    # DEFINE BOUNDS                                                               #
    ###############################################################################
    bnds = dm.get_bounds(plan.funded_status,plan=PLAN, unconstrained = unconstrained)
    
    ###############################################################################
    # DEFINE CONSTRAINTS TO OPTIMIZE FOR MIN AND MAX RETURN                       #
    ###############################################################################

    if PLAN == "Retirement":
        lb = 0.49
        ub = 0.51
    else: 
        lb = 0.39
        ub = 0.41
        
        
    if corp:
        cons = (
                # STRIPS constrained to 30%
                {'type': 'ineq', 'fun': lambda x: np.sum(x[1]) - 0.29*s.init_plan.funded_status},
                {'type': 'ineq', 'fun': lambda x: 0.31*s.init_plan.funded_status - np.sum(x[1])},
               
                #Corporates constrained to 10% less than STRIPS
                {'type': 'eq', 'fun': lambda x: np.sum(x[2]) - (np.sum(x[1])-0.1)},
                
                #sum of all plan assets (excluding Futures and Hedges) = Funded Status Difference    
                {'type': 'eq', 'fun': lambda x: np.sum(x[0:len(s.init_plan)-1]) - x[3] + (1-s.init_plan.funded_status)},
                # 50% of Equity and PE >= Hedges
                {'type': 'ineq', 'fun': lambda x: (x[4]+x[6])*.5 - x[len(s.init_plan)-1]},
                # 15+ STRIPS >= sum(50% of Futures and 25% of Hedges weights)
                {'type': 'ineq', 'fun': lambda x: x[1] - (x[3]/2+x[len(s.init_plan)-1]/4)}
                )
    else:
        cons = (
             # 45% <= sum of Fixed Income (strips and 30y) Assets <= 55%
             {'type': 'ineq', 'fun': lambda x: np.sum(x[1:3]) - lb*s.init_plan.funded_status},
             {'type': 'ineq', 'fun': lambda x: ub*s.init_plan.funded_status - np.sum(x[1:3])},
             #sum of all plan assets (excluding Futures and Hedges) = Funded Status Difference    
             {'type': 'eq', 'fun': lambda x: np.sum(x[0:len(s.init_plan)-1]) - x[3] + (1-s.init_plan.funded_status)},
             # 50% of Equity and PE >= Hedges
             {'type': 'ineq', 'fun': lambda x: (x[4]+x[6])*.5 - x[len(s.init_plan)-1]},
             # 15+ STRIPS >= sum(50% of Futures and 25% of Hedges weights)
             {'type': 'ineq', 'fun': lambda x: x[1] - (x[3]/2+x[len(s.init_plan)-1]/4)}
             )
 
    ###############################################################################
    # COMPUTE MV EFFICIENT FRONTIER PORTFOLIOS                                    #
    ###############################################################################
    #Get data for MV efficient frontier portfolios
    s.generate_efficient_frontiers(bnds, cons,num_ports=100)

    s.get_adjusted_weights()
    s.get_max_sharpe_weights()
    ###############################################################################
    # DISPLAY MV ASSET ALLOCATION                                                 #
    ###############################################################################
    # #Asset Allocation Plot
    # aa_fig = plots.get_aa_fig(s.opt_ports_df)
    # aa_fig.show()
    
    # ###############################################################################
    # # DISPLAY MV EFFICIENT FRONTIER                                               #
    # ###############################################################################
    # #Plotly version of the Efficient Frontier plot
    # ef_fig = plots.get_ef_fig(s.opt_ports_df)
    # ef_fig.show()
    
    ###############################################################################
    # Find max sharpe and adjusted weights                                        #
    ###############################################################################

    
   ###############################################################################
    # EXPORT DATA TO EXCEL                                                        #
    ###############################################################################
    #Export Efficient Frontier portfoio data to excel
    filename = ' stochmv_ef_report0.1'
    if unconstrained:
        filename = ' stochmv_ef_report_unconstraine'
        
    rp.get_stochmv_ef_portfolios_report(PLAN + filename, s, fs_data)
    
    
