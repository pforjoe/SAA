# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 11:38:48 2021

@author: Powis Forjoe
"""

###############################################################################
# IMPORT LIBRARIES                                                            #
###############################################################################
import os
os.chdir("..")
from AssetAllocation.datamanger import datamanger as dm
from AssetAllocation.analytics import summary
from AssetAllocation.analytics.stoch_mv import stochMV
from AssetAllocation.reporting import plots, reports as rp
import numpy as np

PLAN = 'IBT'

###############################################################################
# COMPUTE LIABILITY DATA                                                      #
###############################################################################
liab_model = summary.get_liab_model(PLAN)

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
for key in s.resamp_corr_dict:
    resamp_corr_fig = plots.get_resamp_corr_fig(s.resamp_corr_dict[key], key)
    resamp_corr_fig.show()
    
###############################################################################
# VIEW  RETURNS                                                               #
###############################################################################
#visualize the simulated returns
plots.get_sim_return_fig(s)

###############################################################################
# DEFINE BOUNDS                                                               #
###############################################################################
bnds = dm.get_bounds(plan.funded_status,plan=PLAN)

###############################################################################
# DEFINE CONSTRAINTS TO OPTIMIZE FOR MIN AND MAX RETURN                       #
###############################################################################
cons = (
        # 45% <= sum of Fixed Income Assets <= 55%
        {'type': 'ineq', 'fun': lambda x: np.sum(x[1:3]) - 0.45*s.init_plan.funded_status},
        {'type': 'ineq', 'fun': lambda x: .55*s.init_plan.funded_status - np.sum(x[1:3])},
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
#Asset Allocation Plot
aa_fig = plots.get_aa_fig(s.opt_ports_df)
aa_fig.show()

###############################################################################
# DISPLAY MV EFFICIENT FRONTIER                                               #
###############################################################################
#Plotly version of the Efficient Frontier plot
ef_fig = plots.get_ef_fig(s.opt_ports_df)
ef_fig.show()

###############################################################################
# EXPORT DATA TO EXCEL                                                        #
###############################################################################
#Export Efficient Frontier portfoio data to excel
rp.get_stochmv_ef_portfolios_report(PLAN+' stochmv_ef_report', s)