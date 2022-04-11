# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 14:20:20 2022

@author: Maddie Choi
"""

import pandas as pd
from AssetAllocation.datamanger import datamanger as dm
import AssetAllocation.reporting.sheets as sheets
from AssetAllocation.reporting import reports as rp

#read in current ftse data
prev_ftse = pd.read_excel(dm.TS_FP + "ftse_data.xlsx", sheet_name = ['new_data','old_data'],index_col=0)

#get new ftse data
new_ftse = dm.get_new_ftse_data()

#create ftse dict for report
ftse_dict = {'new_data' : new_ftse, 'old_data' :  prev_ftse['old_data']}

#generate new ftse report
rp.get_ftse_data_report(ftse_dict, "ftse_data")
