# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 19:47:10 2023

@author: rrq1fyq
"""
from AssetAllocation.reporting import reports as rp
from AssetAllocation.analytics import summary
from AssetAllocation.datamanager import datamanager as dm
import pandas as pd
import numpy as np
from scipy.optimize import fsolve

def offset(pbo_cfs):
    #make a copy of the data
    data = pbo_cfs.copy()

    #loop through each period and offset first n rows of 0's to the end
    for i in range(0,len(data.columns)):
        #get discount factor for the period
        disc_rate = i
        #make a list of the cashflows
        cfs = list(data.iloc[:,i])
        #removes top discount amount of rows and adds to the bottom of the list
        cfs = cfs[disc_rate:] + cfs[:disc_rate] 
        #replaces column with new offset data
        data.iloc[:,i] = cfs
    return(data)

def npv(irr, cfs, yrs):
    """
    Returns net present value of cashflows given an irr

    Parameters
    ----------
    irr : double
        IRR.
    cfs : array
        cashflows.
    yrs : array
        periods.

    Returns
    -------
    double

    """
    return np.sum(cfs / (1. + irr) ** yrs)

   
def irr(cfs, yrs, x0, **kwargs):
    """
    Compute internal rate of return(IRR)

    Parameters
    ----------
    cfs : array
        cashflows.
    yrs : array
        periods.
    x0 : double
        guess.
    
    Returns
    -------
    double
        IRR.

    """
    return np.ndarray.item(fsolve(npv, x0=x0,args=(cfs,yrs), **kwargs))



liab_input_dict = dm.get_liab_model_data(plan='Retirement', contrb_pct = 1 )

total_cashflows = liab_input_dict['sc_cashflows']+liab_input_dict['pbo_cashflows']

liab_curve = liab_input_dict['liab_curve']

liab_curve = liab_input_dict['disc_factors']

liab_curve = liab_input_dict['liab_curve']

disc_factors = liab_input_dict['disc_factors']


###############################################################################################################
#For current Year

plan_list = ['Retirement', 'IBT', 'Pension','Total']
contrb_pct = 1
no_of_cols = len(dm.get_liab_mv_cf_cols())%12

pv_df = pd.DataFrame()
irr_df = pd.DataFrame()
cf_dict = {}

for plan in plan_list:
    
    liab_input_dict = dm.get_liab_model_data(plan=plan, contrb_pct = 1 )

    pbo_cfs = liab_input_dict['pbo_cashflows']
    sc_cfs = liab_input_dict['sc_cashflows']
    total_cf = pbo_cfs+sc_cfs*contrb_pct
    

    pbo_table = pd.DataFrame(columns=list(range(0,no_of_cols+1)), index = total_cf.index)

    for col in pbo_table.columns:
        pbo_table[col] = total_cf
        pbo_table.loc[:col,col] = 0
    pbo_offset= offset(pbo_table)
    pbo_offset.columns = list(liab_curve.columns[-no_of_cols-1:])
    
    pv = pd.DataFrame()

    temp_irr =  pd.DataFrame()
    cf = pd.DataFrame()
    for col in pbo_offset.columns:
        temp_pv = pbo_offset[col].values/((1+liab_curve[col].values/100)**disc_factors.values)
        pv[col] = [temp_pv.sum()]

        #get irr
        
        cashflows = np.append(-pv[col], pbo_offset[col])
        cf[col] = cashflows
        yrs = np.append(0, disc_factors)
        
        temp_irr[col] = [irr(cashflows, yrs, 0.03)]
        
            
    pv_df[plan + ' Present Value'] = pv.transpose()
    irr_df[plan + ' Present Value'] = temp_irr.transpose()
    cf_dict[plan] = cf
    
    #get returns

with pd.ExcelWriter('output2023.xlsx', engine='xlsxwriter') as writer:
    # Write each data frame to a different sheet
    pv_df.to_excel(writer, sheet_name='pv', index=True)
    irr_df.to_excel(writer, sheet_name='irr', index=True)

    for i in cf_dict:
        cf_dict[i].to_excel(writer, sheet_name= i+' cf ts.xlsx', index= True)


###############################################################################################################

plan_list = ['Retirement', 'IBT','Pension']
pbo_dict = {}
for sheet in ['2021','2021_1']:
    df_pbocfs = pd.read_excel(dm.TS_FP+'past_pbo_cashflow_data.xlsx', 
                               sheet_name=sheet, index_col=0)/12
    df_pbocfs = dm.reindex_to_monthly_data(df_pbocfs)
    
    pbo_dict[sheet] = df_pbocfs

#keep rp and pp same as before rmsmt
pbo_dict['2021_1'][['Retirement','Pension']] = pbo_dict['2021'][['Retirement','Pension']]
pbo_dict['2021_1']['Total'] =  pbo_dict['2021_1'].sum(axis = 1)
pbo_dict['2021']['Total'] =  pbo_dict['2021_1'].sum(axis = 1)
    
sc_dict = {}
for sheet in ['2021']:
    df_sc_cfs = pd.read_excel(dm.TS_FP+'past_sc_cashflow_data.xlsx', 
                               sheet_name=sheet, index_col=0)/12
    df_sc_cfs = dm.reindex_to_monthly_data(df_sc_cfs)
    
    sc_dict[sheet] = df_sc_cfs
sc_dict['2021']['Total'] =  sc_dict['2021'].sum(axis = 1)  

contrb_pct = 1
no_of_cols = 12


pv_df = pd.DataFrame()
irr_df = pd.DataFrame()
cf_dict = {}

plan_list = ['Retirement', 'IBT','Pension', 'Total']
for plan in plan_list:
    
    pbo_cfs = pbo_dict['2021'][plan]
    sc_cfs = sc_dict['2021'][plan]
    total_cf = pbo_cfs+sc_cfs*contrb_pct
    

    pbo_table = pd.DataFrame(columns=list(range(0,no_of_cols+1)), index = total_cf.index)

    for col in pbo_table.columns:
        pbo_table[col] = total_cf
        pbo_table.loc[:col,col] = 0
        
    pbo_offset= offset(pbo_table)
    pbo_offset.columns = list(pbo_cfs.index[0:no_of_cols+1])
    
    pv = pd.DataFrame()
    temp_irr =  pd.DataFrame()
    cf = pd.DataFrame()
    for col in pbo_offset.columns:
        temp_pv = pbo_offset[col].values/((1+liab_curve[col].values/100)**disc_factors.values)
        pv[col] = [temp_pv.sum()]

        #get irr
        
        cashflows = np.append(-pv[col], pbo_offset[col])
        cf[col] = cashflows
        yrs = np.append(0, disc_factors)
        
        temp_irr[col] = [irr(cashflows, yrs, 0.03)]
        
            
    pv_df[plan + ' Present Value'] = pv.transpose()
    irr_df[plan + ' Present Value'] = temp_irr.transpose()
    cf_dict[plan] = cf
    

  
with pd.ExcelWriter('output2021.xlsx', engine='xlsxwriter') as writer:
    # Write each data frame to a different sheet
    pv_df.to_excel(writer, sheet_name='pv', index=True)
    irr_df.to_excel(writer, sheet_name='irr', index=True)

    for i in cf_dict:
        cf_dict[i].to_excel(writer, sheet_name= i+' cf ts.xlsx', index= True)