# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 12:57:51 2022

@author: RRQ1FYQ
"""

import pandas as pd
from AssetAllocation.datamanger import datamanger as dm
from AssetAllocation.analytics.liability_model import liabilityModel
import pandas as pd
import numpy as np
from scipy.optimize import fsolve
import AssetAllocation.reporting.sheets as sheets
from AssetAllocation.reporting import reports as rp
import AssetAllocation.analytics.summary as summary

import matplotlib as plt


################################################################################################################################################
################################################################################################################################################

#run report dict
report_dict = summary.get_report_dict()

#get asset market values and PBO values table (i.e. asset/liability market values table)
asset_liab_mkt_val_dict = report_dict["asset_liab_mkt_val_dict"]

#get funded status for each plan
asset_liab_mkt_val_dict = dm.get_funded_status(asset_liab_mkt_val_dict)

# 1y vol
one_yr_vol = dm.get_funded_status_vol(asset_liab_mkt_val_dict,n = 12)

#6m vol
six_mo_vol = dm.get_funded_status_vol(asset_liab_mkt_val_dict,n = 6)

#merge 1y and 6m dfs to plot
graph_df = dm.merge_dfs(one_yr_vol,six_mo_vol)
graph_df = graph_df.tail(n = 12)
#multiply by 100 to get percentages
graph_df = graph_df*100


######################
#Plot for Retirement
######################
plt.style.use('default')
retirement = graph_df[['Retirement_x','Retirement_y']].plot(title = "Retirement Realized Funded Status Volatility",
                                                            use_index = True,
                                                            color = ["b","0.5"])
retirement.yaxis.set_major_formatter(plt.ticker.PercentFormatter())
retirement.legend(['1yr FSV','6mth FSV'])


######################
#Plot for IBT
######################
plt.style.use('default')
retirement = graph_df[['IBT_x','IBT_y']].plot(title = "IBT Realized Funded Status Volatility",
                                                            use_index = True,
                                                            color = ["b","0.5"])
retirement.yaxis.set_major_formatter(plt.ticker.PercentFormatter())
retirement.legend(['1yr FSV','6mth FSV'])


######################
#Plot for Pension
######################
plt.style.use('default')
retirement = graph_df[['Pension_x','Pension_y']].plot(title = "Pension Realized Funded Status Volatility",
                                                            use_index = True,
                                                            color = ["b","0.5"])
retirement.yaxis.set_major_formatter(plt.ticker.PercentFormatter())
retirement.legend(['1yr FSV','6mth FSV'])


