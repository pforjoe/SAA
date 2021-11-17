# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 15:13:59 2021

@author: Powis Forjoe
"""

from AssetAllocation.datamanger import datamanger as dm
import numpy as np
import pandas as pd
from AssetAllocation.analytics import ts_analytics as ts
from AssetAllocation.reporting import reports as rp

quarterly_re_pi = pd.read_excel(dm.TS_FP+'re_pi.xlsx', sheet_name='data', index_col=0)
quarterly_re_pi = quarterly_re_pi/100
monthly_re_pi = quarterly_re_pi.copy()
monthly_re_pi = dm.reindex_to_monthly_data(monthly_re_pi)
monthly_re_pi['RE-PI_monthly'] = np.nan

monthly_re_pi['RE-PI_monthly'][0] = (1 + monthly_re_pi['RE-PI'][0])**(1/3) - 1

count=3
for i in range(1, len(monthly_re_pi)):    
    if i<=count:
        monthly_re_pi['RE-PI_monthly'][i] = (1 + monthly_re_pi['RE-PI'][count])**(1/3) - 1
    else:
        count +=3
        monthly_re_pi['RE-PI_monthly'][i] = (1 + monthly_re_pi['RE-PI'][count])**(1/3) - 1

asset_ret_df = pd.read_excel(dm.TS_FP+'return_data.xlsx', sheet_name='2010', index_col=0)
asset_ret_df = asset_ret_df[['15+ STRIPS', 'Long Corps', 'WN1 COMB Comdty', 'Total Dom Eq w/o Derivatives', 
                             'Total Liquid Alts','Total Private Equity', 'Total Credit', 'Total Real Estate',
                             'Cash', 'Equity Hedges']]

asset_ret_df.columns = ['15+ STRIPS', 'Long Corporate','Ultra 30Y Futures','Equity','Liquid Alternatives',
                        'Private Equity', 'Credit','RE-BNY','Cash', 'Hedges']

liab_ret_df = dm.get_liab_returns()
monthly_ret = dm.merge_dfs(liab_ret_df, asset_ret_df)

quarterly_ret = dm.get_prices_df(monthly_ret)
quarterly_ret = dm.format_data(quarterly_ret,'1Q')


monthly_ret = dm.merge_dfs(monthly_ret, monthly_re_pi)
monthly_ret = monthly_ret[['Liability','15+ STRIPS', 'Long Corporate','Ultra 30Y Futures','Equity','Liquid Alternatives',
                        'Private Equity', 'Credit','RE-BNY','RE-PI_monthly','Cash', 'Hedges']]

monthly_corr = ts.compute_ewcorr_matrix(monthly_ret)

quarterly_ret = dm.merge_dfs(quarterly_ret, quarterly_re_pi)
quarterly_ret = quarterly_ret[['Liability','15+ STRIPS', 'Long Corporate','Ultra 30Y Futures','Equity','Liquid Alternatives',
                        'Private Equity', 'Credit','RE-BNY','RE-PI','Cash', 'Hedges']]

quarterly_corr = ts.compute_ewcorr_matrix(quarterly_ret)

filepath = rp.get_reportpath('re-pi_corr_quarterly_monthly')
writer = pd.ExcelWriter(filepath, engine='xlsxwriter')
rp.sheets.set_corr_sheet(writer, monthly_corr, sheet_name='Monthly Correlations')
rp.sheets.set_return_sheet(writer, monthly_ret)
rp.sheets.set_corr_sheet(writer, quarterly_corr,sheet_name='Quarterly Correlations')
rp.sheets.set_return_sheet(writer, quarterly_ret, sheet_name='Historical Quarterly Returns')
writer.save()

