# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 11:25:44 2022

@author: NVG9HXP
"""

import pandas as pd
from AssetAllocation.datamanger import datamanger as dm
import AssetAllocation.reporting.sheets as sheets
from AssetAllocation.reporting import reports as rp

plan_data=pd.read_excel(dm.DATA_FP+'monthly_plan_data.xlsx', sheet_name='data')
plan_data.columns
mv_df = plan_data.pivot_table(values = 'Market Value\n', index='As Of Date\n', columns='Account Name\n')
mv_df.columns = ['Total', 'Pension', 'Retirement', 'IBT']
ret_df = plan_data.pivot_table(values = 'Account Monthly Return\n', index='As Of Date\n', columns='Account Name\n')
ret_df.columns = ['Total', 'Pension', 'Retirement', 'IBT']
ret_df /= 100
filepath = rp.get_reportpath('new_plan_data')
writer = pd.ExcelWriter(filepath, engine = 'xlsxwriter')
sheets.set_return_sheet(writer, ret_df, sheet_name='return')
sheets.set_dollar_values_sheet(writer, mv_df, sheet_name='mkt_value')
writer.save()