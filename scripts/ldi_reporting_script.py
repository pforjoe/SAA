# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 14:10:47 2022

@author: NVG9HXP
"""

from AssetAllocation.reporting import reports as rp
from AssetAllocation.analytics import summary
from AssetAllocation.datamanger import datamanger as dm
# import time


# start = time.time()
# liab_data_dict = summary.get_liab_data_dict()
dm.update_ldi_data()
report_dict = summary.get_report_dict()

# end = time.time()
#print(end - start)
# temp_dict = {}
# for key in liab_model_dict:
#     temp_df = dm.merge_dfs(liab_model_dict[key]['Present Values'], liab_model_dict[key]['IRR'])
#     temp_dict[key] = temp_df

# report_dict['pv_irr_dict'] = temp_dict

#rp.get_liability_returns_report(report_dict,report_name = "test")

#new ldi report format
rp.get_ldi_report(report_dict)
