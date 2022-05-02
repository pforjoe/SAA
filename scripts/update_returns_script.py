# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 11:25:44 2022

@author: NVG9HXP
"""

import pandas as pd
from AssetAllocation.datamanager import datamanager as dm
import AssetAllocation.reporting.sheets as sheets
from AssetAllocation.reporting import reports as rp

# TODO: automate updating ftse curve
plan_data_dict = dm.update_plan_data('Plan level Historical Returns.xls', 'Plan level Historical Returns')

ftse_data_dict = dm.update_ftse_data()

liab_mv_cfs_dict = dm.update_plan_mv()

dm.update_ldi_data()
