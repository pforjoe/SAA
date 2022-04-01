# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 14:20:20 2022

@author: RRQ1FYQ
"""

import pandas as pd
from AssetAllocation.datamanger import datamanger as dm
import AssetAllocation.reporting.sheets as sheets
from AssetAllocation.reporting import reports as rp

#read in current ftse data
prev_ftse = pd.read_excel(dm.TS_FP + "ftse_data.xlsx", sheet_name = ['new_data','old_data'],index_col=0)

#get new ftse data
new_ftse = dm.get_new_ftse_data()

#drop n columns 
n = 146
new_ftse.drop(columns = new_ftse.columns[-n:], axis = 1, inplace = True)

#merge new ftse with previous ftse data
updated_ftse = pd.merge(new_ftse, prev_ftse['new_data'], how = "outer", left_index= True, right_index = True)

#create ftse dict for report
ftse_dict = {'new_data' : updated_ftse, 'old_data' :  prev_ftse['old_data']}

#generate new ftse report
rp.get_ftse_data_report(ftse_dict)
