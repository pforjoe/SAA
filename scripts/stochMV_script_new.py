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
ldi_input_dict = dm.get_ldi_data()

max_sharpe_dict = {}
max_sharpe_dict['Retirement'] = pd.DataFrame()
max_sharpe_dict['Pension'] = pd.DataFrame()
max_sharpe_dict['IBT'] = pd.DataFrame()

for p in ['Retirement','Pension','IBT']:
        
    
    PLAN = p
    unbounded = True
    ###############################################################################
    # COMPUTE LIABILITY DATA                                                      #
    ###############################################################################
    liab_model = summary.get_liab_model_new(ldi_input_dict,PLAN)
    
    ###############################################################################
    # COMPUTE PLAN INPUTS                                                         #
    ###############################################################################
    pp_inputs = summary.get_pp_inputs(liab_model,PLAN)
    
    ###############################################################################
    # INITIALIZE PLAN                                                             #
    ###############################################################################
    plan = summary.get_plan_params(pp_inputs)
    pp_dict = plan.get_pp_dict()
    
    ###############################################################################
    # INITIALIZE STOCHMV                                                          #
    ###############################################################################
    #initialize the stochastic mean variance
    s = stochMV(plan, 10)
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
    bnds = dm.get_bounds(plan.funded_status,plan=PLAN, unbounded = unbounded)
    
    ###############################################################################
    # DEFINE CONSTRAINTS TO OPTIMIZE FOR MIN AND MAX RETURN                       #
    ###############################################################################
    corp = False
    if p == "Retirement":
        lb = 0.50
        ub = 0.8999999999999999
    else: 
        lb = 0.40
        ub = 0.8999999999999999
        
        
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
    s.generate_efficient_frontiers(bnds, cons,num_ports=20)
    
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
    #Export Efficient Frontier portfoio data to excel
    ef_df = s.opt_ports_df.copy()
    ef_df['Total'] = ef_df['15+ STRIPS'] + ef_df['Long Corporate'] + ef_df['Equity'] + ef_df['Liquid Alternatives'] + ef_df['Private Equity'] +ef_df['Credit'] + ef_df['Real Estate'] + ef_df['Cash']
    
    new_ef_df = ef_df.copy()
    
    assets = ['15+ STRIPS', 'Long Corporate', 'Equity', 'Liquid Alternatives', 'Private Equity', 'Credit', 'Real Estate', 'Cash']
    for col in assets:
        new_ef_df[col] = new_ef_df[col] / ef_df['Total']
    new_ef_df['Total'] =  new_ef_df['15+ STRIPS'] + new_ef_df['Long Corporate'] + new_ef_df['Equity'] + new_ef_df['Liquid Alternatives'] + new_ef_df['Private Equity'] +new_ef_df['Credit'] + new_ef_df['Real Estate'] + new_ef_df['Cash']   
    max_sharpe_weights = new_ef_df.loc[new_ef_df['Sharpe'].idxmax()]
    
    
   
        
   ###############################################################################
    # EXPORT DATA TO EXCEL                                                        #
    ###############################################################################

    max_sharpe_weights = max_sharpe_weights.to_frame()
    max_sharpe_dict[PLAN] = dm.merge_dfs(max_sharpe_dict[PLAN] , max_sharpe_weights)
    
    
   
    
   ###############################################################################
    # EXPORT DATA TO EXCEL                                                        #
    ###############################################################################
    #Export Efficient Frontier portfoio data to excel
    filename = ' stochmv_ef_report_corp_FI_inf'
    if unbounded: 
        filename = ' stochmv_ef_report_unconstrained_corp_FI_inf'
        
    rp.get_stochmv_ef_portfolios_report(PLAN + filename, s, new_ef_df, max_sharpe_weights)
    
    
