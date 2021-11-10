# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 14:08:59 2021

@author: NVG9HXP
"""

from AssetAllocation.datamanger import datamanger as dm
from AssetAllocation.analytics import summary
from AssetAllocation.reporting import reports as rp, plots
import numpy as np
import stochMV as stMV

plan_list = ['IBT', 'Pension', 'Retirement']
bounds_list = ['q32021_privates']
# bounds_list = ['unbounded', 'ips', 'q32021_privates']
fi_cons_dict = {'IBT':{'min':.2, 'max':.7},
                'Pension':{'min':.2, 'max':.45},
                'Retirement':{'min':.4, 'max':.75}}
for pension_plan in plan_list:
    pp_inputs = summary.get_pp_inputs(plan=pension_plan)

    ###############################################################################
    # CREATE PLAN OBJECT                                                          #
    ###############################################################################
    plan = summary.get_plan_params(pp_inputs)
    pp_dict = plan.get_pp_dict()
    
    # INITIALIZE STOCHMV                                                          #
    ###############################################################################
    #initialize the stochastic mean variance
    s = stMV.stochMV(plan, 200)
    #generate the random returns Aand sample corr
    s.generate_plans()
    s.generate_resamp_corr_dict()
    plots.get_sim_return_fig(s)

    for bounds in bounds_list:
        print(pension_plan + ' ' + bounds)
        ###############################################################################
        # DEFINE BOUNDS                                                               #
        ###############################################################################
        bnds = dm.get_bounds(filename=bounds+'_bounds.xlsx',plan=pension_plan)
        
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
            {'type': 'ineq', 'fun': lambda x: np.sum(x[1:3]) - fi_min*s.init_plan.funded_status},
            {'type': 'ineq', 'fun': lambda x: fi_max*s.init_plan.funded_status - np.sum(x[1:3])},
        )
        cons = (
            #sum of all plan assets (excluding Futures and Hedges) = Funded Status Difference    
            {'type': 'eq', 'fun': lambda x: np.sum(x[0:len(s.init_plan)-1]) - x[3] + (1-s.init_plan.funded_status)},
            # 50% of Equity and PE >= Hedges
            {'type': 'ineq', 'fun': lambda x: (x[4]+x[6])*.5 - x[len(s.init_plan)-1]},
            # 15+ STRIPS >= sum(50% of Futures and 25% of Hedges weights)
            {'type': 'ineq', 'fun': lambda x: x[1] - (x[3]/2+x[len(s.init_plan)-1]/4)}
        )
        
        if bounds != 'unbounded':
            cons = fi_cons+cons
        
        ###############################################################################
        # COMPUTE MV EFFICIENT FRONTIER PORTFOLIOS                                    #
        ###############################################################################
        #Get data for MV efficient frontier portfolios
        s.generate_efficient_frontiers(bnds, cons,num_ports=100)
        
        ###############################################################################
        # EXPORT DATA TO EXCEL                                                        #
        ###############################################################################
        #Export Efficient Frontier portfoio data to excel
        rp.get_stochmv_ef_portfolios_report(pension_plan+'_'+bounds+'_robustmv_ef_report', s,bnds)