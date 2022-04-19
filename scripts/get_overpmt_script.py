# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 10:37:01 2022

@author: Maddie Choi
"""

from AssetAllocation.reporting import reports as rp
from AssetAllocation.datamanger import datamanger as dm
import pandas as pd
start_yr = 2003
end_yr = 2028
filename = 'Benefit Pmt Overpmt Monthly JE (BNYM detail).xlsx'
    
#get list of which sheets to read in
years = list(range(start_yr, end_yr+1, 1))

#define empty dict
pmt_df = pd.DataFrame()

for yr in years:
    #try reading in sheet by year 
    try:
        overpmt_df = pd.read_excel(dm.UPDATE_FP + filename, sheet_name = str(yr), skiprows=4, header = 1, usecols=[0,1,2])
    
    #if error then read in sheet by year + " " 
    except:
        overpmt_df = pd.read_excel(dm.UPDATE_FP + filename, sheet_name = str(yr) + " ", skiprows=4, header = 1, usecols=[0,1,2])
    
    #drop first row by row number since index is not consistent
    overpmt_df.drop(0, axis = 0, inplace = True)
    
    #set dates as index
    overpmt_df.set_index(overpmt_df.columns[0], inplace = True)
    
    #drop na values
    overpmt_df.dropna(axis = 0, inplace = True)
    
    #define dict
    pmt_df = pmt_df.append(overpmt_df)
    
#rename columns & index
pmt_df.columns = ['Monthly Benefit adj', 'Misc Receivable']
pmt_df = pmt_df[['Misc Receivable']]
pmt_df.index.names = ['Date']

#seet to end of month
pmt_df.index = pmt_df.index.to_period('M').to_timestamp('M')

#create excel file
report_name = 'misc_receivables_data'
filepath = rp.get_ts_path(report_name)
writer = pd.ExcelWriter(filepath, engine = 'xlsxwriter')
rp.sheets.set_dollar_values_sheet(writer, pmt_df, 'misc_receiv')
rp.print_report_info(report_name, filepath)
writer.save()