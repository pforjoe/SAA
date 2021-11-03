# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 11:38:48 2021

@author: Powis Forjoe
"""
import os
os.chdir("..")

###############################################################################
# IMPORT LIBRARIES                                                            #
###############################################################################
from AssetAllocation.datamanger import datamanger as dm
from AssetAllocation.analytics import summary
from AssetAllocation.reporting import reports as rp, plots
import numpy as np

PLAN = 'IBT'

###############################################################################
# COMPUTE PLAN INPUTS                                                         #
###############################################################################
pp_inputs = summary.get_pp_inputs(plan=PLAN)

###############################################################################
# CREATE PLAN OBJECT                                                          #
###############################################################################
plan = summary.get_plan_params(pp_inputs)
pp_dict = plan.get_pp_dict()

###############################################################################
# DEFINE BOUNDS                                                               #
###############################################################################
bnds = dm.get_bounds(plan=PLAN)

###############################################################################
# DEFINE CONSTRAINTS TO OPTIMIZE FOR MIN AND MAX RETURN                       #
###############################################################################
cons = (
    # sum of Fixed Income Assets >= 50%
    {'type': 'ineq', 'fun': lambda x: np.sum(x[1:3]) - 0.5},
    #sum of all plan assets (excluding Futures and Hedges) = 2%    
    {'type': 'eq', 'fun': lambda x: np.sum(x[0:len(plan)-1]) - x[3] - (1-plan.funded_status)},
    # Hedges <= 50% of Equity & PE
    {'type': 'ineq', 'fun': lambda x: (x[4]+x[6])*.5 - x[len(plan)-1]},
    # STRIPS*4 >= sum(Futures and Hedges)
    {'type': 'ineq', 'fun': lambda x: x[1]*4 - (x[3]+x[len(plan)-1])}
)

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
rp.get_ef_portfolios_report(PLAN+'_ef_portfolios', plan)