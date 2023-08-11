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

#compute ldi stats
report_dict = summary.get_report_dict(n=3)

#generate ldi report
rp.get_ldi_report(report_dict, dashboard_graphs=True)
