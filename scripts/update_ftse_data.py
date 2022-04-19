# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 14:20:20 2022

@author: Maddie Choi
"""

import pandas as pd
from AssetAllocation.datamanger import datamanger as dm
from AssetAllocation.reporting import reports as rp

ftse_dict = dm.update_ftse_data()

#generate new ftse report
rp.get_ftse_data_report(ftse_dict, "ftse_data")
