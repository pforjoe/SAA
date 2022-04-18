# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 10:37:01 2022

@author: RRQ1FYQ
"""
from AssetAllocation.reporting import reports as rp
from AssetAllocation.analytics import summary
from AssetAllocation.datamanger import datamanger as dm
import pandas as pd
start_yr = 2003
end_yr = 2028
filename = 'Benefit Pmt Overpmt Monthly JE (BNYM detail).xlsx'
    
#get list of which sheets to read in
years = list(range(start_yr, end_yr+1, 1))

#define empty dict
pmt = pd.DataFrame()

for yr in years:
    #try reading in sheet by year 
    try:
        overpmt = pd.read_excel(dm.MV_INPUTS_FP + filename, sheet_name = str(yr), skiprows=4, header = 1, usecols=[0,1,2])
    
    #if error then read in sheeat by year + " " 
    except:
        overpmt = pd.read_excel(dm.MV_INPUTS_FP + filename, sheet_name = str(yr) + " ", skiprows=4, header = 1, usecols=[0,1,2])
    
    #drop first row by row number since index is not consistent
    overpmt.drop(0, axis = 0, inplace = True)
    
    #set dates as index
    overpmt.set_index(overpmt.columns[0], inplace = True)
    
    #drop na values
    overpmt.dropna(axis = 0, inplace = True)
    
    #define dict
    pmt = pmt.append(overpmt)
    
#seet to end of month
pmt.index = pmt.index.to_period('M').to_timestamp('M')


