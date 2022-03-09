# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 11:25:44 2022

@author: NVG9HXP
"""

import pandas as pd
from AssetAllocation.datamanger import datamanger as dm
import AssetAllocation.reporting.sheets as sheets
from AssetAllocation.reporting import reports as rp


    
def update_plan_data():
    plan_data=pd.read_excel(dm.TS_FP+'monthly_plan_data.xlsx', sheet_name='data')
    plan_data.columns = ["Account Name","Account Id","Return Type","Date", "Market Value","Monthly Return"]
    plan_data["Account Name"].replace({"Total Retirement":"Retirement", "Total Pension":"Pension", "Total UPS/IBT FT Emp Pension":"IBT", "LDI ONLY-TotUSPenMinus401H":"Total"}, inplace = True)
    
    mv_df = plan_data.pivot_table(values = 'Market Value', index='Date', columns='Account Name')
    
    ret_df = plan_data.pivot_table(values = 'Monthly Return', index='Date', columns='Account Name')
    ret_df /= 100
    
    plan_data_dict = {"mkt_value" : mv_df, "return":ret_df}
    return(plan_data_dict)

plan_data_dict = update_plan_data()

def get_plan_data_report(plan_data_dict, file_name = "new_plan_data"):
    filepath = rp.get_reportpath(file_name)
    writer = pd.ExcelWriter(filepath, engine = 'xlsxwriter')
    sheets.set_dollar_values_sheet(writer, plan_data_dict['mkt_value'], sheet_name='mkt_value')
    sheets.set_return_sheet(writer, plan_data_dict["return"], sheet_name='return', set_neg_value_format= False)

    writer.save()

get_plan_data_report(plan_data_dict)
