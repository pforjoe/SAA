# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 14:10:47 2022

@author: NVG9HXP
"""

from AssetAllocation.reporting import reports as rp
from AssetAllocation.analytics import summary
from AssetAllocation.datamanager import datamanager as dm
# import time

update_ldi_data = True
update_asset_ret_data = True


if update_ldi_data:
    dm.update_ldi_data()
    
if update_asset_ret_data:
    dm.update_asset_ret_data()
report_dict = summary.get_report_dict()

#new ldi report format
rp.get_ldi_report(report_dict)

