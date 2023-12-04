# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 14:10:47 2022

@author: NVG9HXP
"""

from AssetAllocation.reporting import reports as rp
from AssetAllocation.analytics import summary
from AssetAllocation.datamanager import datamanager as dm
# import time

#update ldi data
dm.update_ldi_data()

dm.update_plan_data(report_name = 'plan_data_062023')
dm.update_ftse_data(file_name='ftse_data_062023.xlsx',
                    ftse_file='ftse-pension-discount-curve-06-30-2023.xlsx')
dm.update_plan_mv(ftse_filename='ftse_data_062023.xlsx',report_name = "liab_mv_cfs")

#compute ldi stats
report_dict = summary.get_report_dict(n=3,contrb_pct = 1.0, plan_filename='plan_data.xlsx',
                                      ftse_filename='ftse_data.xlsx')

#generate ldi report
rp.get_ldi_report(report_dict,report_name='ldi_report_092023-sc', dashboard_graphs=True)
