# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 11:38:48 2021

@author: Powis Forjoe
"""

###############################################################################
# IMPORT LIBRARIES                                                            #
###############################################################################
from AssetAllocation.datamanger import datamanger as dm
from AssetAllocation.analytics import summary
from AssetAllocation.reporting import plots
# import matplotlib.pyplot as plt
import numpy as np
import stochMV as stMV
import seaborn as sns

PLAN = 'IBT'

###############################################################################
# COMPUTE PLAN INPUTS                                                         #
###############################################################################
#get return
mv_inputs_dict = dm.get_mv_inputs_data('inputs_test.xlsx', plan=PLAN)
mv_inputs = summary.get_mv_inputs(mv_inputs_dict)
#get historical vol and correlation
ts_dict = dm.get_ts_data(plan=PLAN)
pp_inputs = summary.get_pp_inputs(mv_inputs, ts_dict)

###############################################################################
# INITIALIZE PLAN                                                             #
###############################################################################
plan = summary.get_plan_params(pp_inputs)
pp_dict = plan.get_pp_dict()

###############################################################################
# INITIALIZE STOCHMV                                                          #
###############################################################################
#initialize the stochastic mean variance
s = stMV.stochMV(plan, 5)
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
sns.pairplot(s.returns_df, corner=True)
# plt.savefig("sampling.jpeg")

###############################################################################
# DEFINE BOUNDS                                                               #
###############################################################################
bnds = dm.get_bounds(plan=PLAN)

#View bounds
df_bnds = dm.pd.DataFrame(bnds, index=plan.symbols, columns=['Lower', 'Upper'])

###############################################################################
# DEFINE CONSTRAINTS TO OPTIMIZE FOR MIN AND MAX RETURN                       #
###############################################################################
cons = (
    # sum of Fixed Income Assets >= 50%
    {'type': 'ineq', 'fun': lambda x: np.sum(x[1:3]) - 0.5},
    #sum of all plan assets (excluding Futures and Hedges) = 2%    
    {'type': 'eq', 'fun': lambda x: np.sum(x[0:len(plan)-1]) - x[3] - .02},
    # Hedges <= 50% of Equity & PE
    {'type': 'ineq', 'fun': lambda x: (x[4]+x[6])*.5 - x[len(plan)-1]},
    # STRIPS*4 >= sum(Futures and Hedges)
    {'type': 'ineq', 'fun': lambda x: x[1]*4 - (x[3]+x[len(plan)-1])}
)

###############################################################################
# COMPUTE MV EFFICIENT FRONTIER PORTFOLIOS                                    #
###############################################################################
#Get data for MV efficient frontier portfolios
s.generate_efficient_frontiers(bnds, cons,num_ports=25)

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
