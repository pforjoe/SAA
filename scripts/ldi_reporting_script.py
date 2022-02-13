# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 14:10:47 2022

@author: NVG9HXP
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

report_dict = summary.get_report_dict()

rp.get_liability_returns_report(report_dict,report_name = "liability_returns")


