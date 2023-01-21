# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 14:10:47 2022

@author: NVG9HXP
"""

from AssetAllocation.reporting import reports as rp
from AssetAllocation.analytics import summary
from AssetAllocation.datamanager import datamanager as dm
# import time

# start = time.time()
# liab_data_dict = summary.get_liab_data_dict()

dm.update_ldi_data()
report_dict = summary.get_report_dict()

# end = time.time()
#print(end - start)

#new ldi report format
rp.get_ldi_report(report_dict)
