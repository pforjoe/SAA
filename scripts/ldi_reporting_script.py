# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 14:10:47 2022

@author: NVG9HXP
"""

from AssetAllocation.reporting import reports as rp
from AssetAllocation.analytics import summary
#import time


#start = time.time()
report_dict = summary.get_report_dict()
#end = time.time()
#print(end - start)

rp.get_liability_returns_report(report_dict,report_name = "ldi_reporting")


