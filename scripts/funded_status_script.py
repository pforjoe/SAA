# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 12:57:51 2022

@author: RRQ1FYQ
"""

import pandas as pd
from AssetAllocation.datamanger import datamanger as dm
from AssetAllocation.analytics.liability_model import liabilityModel
import pandas as pd
import numpy as np
from scipy.optimize import fsolve
import AssetAllocation.reporting.sheets as sheets
from AssetAllocation.reporting import reports as rp
import AssetAllocation.analytics.summary as summary

import matplotlib as plt


################################################################################################################################################
################################################################################################################################################

#run report dict
report_dict = summary.get_report_dict()

mkt_val_dict = report_dict['asset_liab_mkt_val_dict']

#get fs vol
mkt_val_dict = dm.get_fs_data(mkt_val_dict, n = 6)

