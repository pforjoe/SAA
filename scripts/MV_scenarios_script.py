# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 14:08:59 2021

@author: Powis Forjoe
"""

from AssetAllocation.datamanager import datamanager as dm
from AssetAllocation.analytics import summary
from AssetAllocation.reporting import reports as rp, plots
import numpy as np

plan_list = ['IBT', 'Pension', 'Retirement']
bounds_list = ['unbounded', 'ips', 'q32021_privates']
fi_cons_dict = {'IBT':{'min':.2, 'max':.7},
                'Pension':{'min':.2, 'max':.45},
                'Retirement':{'min':.4, 'max':.75}}

for pension_plan in plan_list:
    pp_inputs = summary.get_pp_inputs(plan=pension_plan)

    ###############################################################################
    # COMPUTE LIABILITY DATA                                                      #
    ###############################################################################
    liab_model = summary.get_liab_model(pension_plan, .05, False)
    
    ###############################################################################
    # COMPUTE PLAN INPUTS                                                         #
    ###############################################################################
    pp_inputs = summary.get_pp_inputs(liab_model,pension_plan)
    ###############################################################################
    # CREATE PLAN OBJECT                                                          #
    ###############################################################################
    plan = summary.get_plan_params(pp_inputs)
    pp_dict = plan.get_pp_dict()
    
    for bounds in bounds_list:
        print(pension_plan + ' ' + bounds)
        ###############################################################################
        # DEFINE BOUNDS                                                               #
        ###############################################################################
        bnds = dm.get_bounds(plan.funded_status,filename=bounds+'_bounds.xlsx',plan=pension_plan)
        
        ###############################################################################
        # DEFINE CONSTRAINTS TO OPTIMIZE FOR MIN AND MAX RETURN                       #
        ###############################################################################
        fi_min = .45
        fi_max = .55
        if bounds =='ips':
            fi_min = fi_cons_dict[pension_plan]['min']
            fi_max = fi_cons_dict[pension_plan]['max']
        fi_cons = (
            # fi_min <= sum of Fixed Income Assets <= fi_max
            {'type': 'ineq', 'fun': lambda x: np.sum(x[1:3]) - fi_min*plan.funded_status},
            {'type': 'ineq', 'fun': lambda x: fi_max*plan.funded_status - np.sum(x[1:3])},
        )
        
        cons = (
            #sum of all plan assets (excluding Futures and Hedges) = Funded Status Difference    
            {'type': 'eq', 'fun': lambda x: np.sum(x[0:len(plan)-1]) - x[3] + (1-plan.funded_status)},
            # Hedges <= 50% of Equity & PE
            {'type': 'ineq', 'fun': lambda x: (x[4]+x[6])*.5 - x[len(plan)-1]},
            # 15+ STRIPS >= sum(50% of Futures and 25% of Hedges weights)
            {'type': 'ineq', 'fun': lambda x: x[1] - (x[3]/2+x[len(plan)-1]/4)}
        )
        
        if bounds != 'unbounded':
            cons = fi_cons+cons
        
        ###############################################################################
        # COMPUTE MV EFFICIENT FRONTIER PORTFOLIOS                                    #
        ###############################################################################
        #Get data for MV efficient frontier portfolios
        plan.compute_eff_frontier(bnds,cons,num_ports=100)
        
        ###############################################################################
        # DISPLAY MV ASSET ALLOCATION                                                 #
        ###############################################################################
        #Asset Allocation Plot
        aa_fig = plots.get_aa_fig(plan.ports_df)
        aa_fig.show()
        
        ###############################################################################
        # DISPLAY MV EFFICIENT FRONTIER                                               #
        ###############################################################################
        #Plotly version of the Efficient Frontier plot
        ef_fig = plots.get_ef_fig(plan.ports_df)
        ef_fig.show()
        
        ###############################################################################
        # EXPORT DATA TO EXCEL                                                        #
        ###############################################################################
        #Export Efficient Frontier portfoio data to excel
        rp.get_ef_portfolios_report(pension_plan + '_' + bounds + '_ef_portfolios', plan, bnds)