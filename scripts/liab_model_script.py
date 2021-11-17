# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 23:46:51 2021

@author: Roxton McNeal, Powis Forjoe
"""

############################################################################################################################################################
# IMPORT LIBRARIES                                                            
############################################################################################################################################################
from AssetAllocation.datamanger import datamanger as dm
# from itertools import count, takewhile

import scipy as sp
# from scipy import optimize
from scipy.optimize import fsolve

# Import pandas
import pandas as pd

# Import numpy
import numpy as np
# from numpy import * 
# from numpy.linalg import multi_dot
# import numpy_financial as npf

# Plot settings
# import matplotlib
# import matplotlib.pyplot as plt
# matplotlib.rcParams['figure.figsize'] = 16, 8

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# IBT or Pension or Retirement
PLAN = 'Pension'

UPS_Contr_Pctg = 0

############################################################################################################################################################
# IMPORT CASHFLOWS                                                            
############################################################################################################################################################
df_CF = pd.read_excel("2021_PBO_CF.xlsx",sheet_name = "PBO Cashflows Summary",
                      skiprows = [0,1,2], usecols=[0,1,2,3,4], na_values=[""],
                      index_col=0)
df_CF = df_CF.divide(12)
df_CF = dm.reindex_to_monthly_data(df_CF)
df_CF_add = pd.read_excel("2021 Liability Cash Flows.xlsx",sheet_name = "CF", 
                          usecols=[0,1,2,3,4], na_values=[""],index_col=0)
df_CF  = df_CF_add.append(df_CF)

from itertools import count, takewhile
def frange(start, stop, step):
        return takewhile(lambda x: x< stop, count(start, step))
t = list(frange((1/12), 79.41666666666, (1/12)))
# df_t=pd.DataFrame(t, columns=['Time'])
df_CF['Time'] = t
CF = np.array(df_CF[PLAN])
Time_CF = df_CF.index
DF = np.array(df_CF['Time'])


############################################################################################################################################################
Raw_FTSE_YC = pd.read_excel("Raw FTSE Data.xlsx",skiprows = [0,2],
                            usecols = [*range(0, 145)], na_values=[""])

raw_liab_ibt_dict={}
tr = Raw_FTSE_YC['Date']
# list of raw (not interpolated) times to maturity
# yr = Raw_FTSE_YC[col] 
# list of raw (not interpolated) yields
t = list(frange(0.5, 30.08, (1/12))) # interpolating in range 1..30 years

for col in Raw_FTSE_YC.columns:
    y = []
    interp = sp.interpolate.interp1d(tr, Raw_FTSE_YC[col], bounds_error=False,
                                     fill_value=sp.nan)
    for i in t:
            value = float(interp(i))
            if not sp.isnan(value): # Don't include out-of-range values
                y.append(value)
                End_Rate = [y[-1]] * 592
                Beg_Rate = [y[0]] * 5
            raw_liab_ibt_dict[col] = Beg_Rate + y + End_Rate


int_df = pd.DataFrame(raw_liab_ibt_dict)
int_df.drop(['Date'], axis=1, inplace=True)
int_df = int_df.iloc[:, ::-1]

############################################################################################################################################################
liab_plan_dict={}
for col in int_df.columns:
    temp_pv = 0
    for j in range (0,len(Time_CF)):
        temp_pv += (CF[j]/((1+int_df[col][j]/100)**DF[j]))
    liab_plan_dict[col] = temp_pv
    
pv_df = pd.DataFrame(liab_plan_dict, index = ['Present Values']).transpose()


############################################################################################################################################################
Plan_Return_YC = np.zeros(len(liab_plan_dict))
for i in range (0,len(liab_plan_dict)-1):
    Plan_Return_YC[i+1] = ((pv_df['Present Values'][i+1])/pv_df['Present Values'][i])-1

Plan_Return_YC = pd.DataFrame(Plan_Return_YC, columns=['Liability'], index=pv_df.index)
liab_ret_df =dm.merge_dfs(Plan_Return_YC, pv_df)
#export monthly dataframe to excel
filepath = (dm.TS_FP+PLAN+ ' Liability Returns & PV.xlsx')
writer = pd.ExcelWriter(filepath, engine='xlsxwriter')
liab_ret_df.to_excel(writer, sheet_name=PLAN)
writer.save()

############################################################################################################################################################
df_SC = pd.read_excel("2021_PBO_CF.xlsx",sheet_name = "Service Cost Cashflows Summary",
                      skiprows = [0,1,2], usecols=[0,1,2,3], na_values=[""],index_col=0)
df_SC = df_SC.divide(12)
df_SC = dm.reindex_to_monthly_data(df_SC)
df_SC_add = pd.read_excel("2021 Liability Cash Flows.xlsx",sheet_name = "SC", usecols=[0,1,2,3,4], na_values=[""],index_col=0)
df_SC  = df_SC_add.append(df_SC)

df_CF_Tot = UPS_Contr_Pctg*df_SC + df_CF
df_CF_Tot['Time'] = df_CF['Time']


CF_Tot = df_CF_Tot[PLAN]
Time_CF_Tot = df_CF_Tot.index
DF_Tot=df_CF_Tot['Time']

############################################################################################################################################################
liab_plan_dict_tot={}
for col in int_df.columns:
    temp_pv=0
    for j in range (0,len(Time_CF)):
        temp_pv += (CF_Tot[j]/((1+int_df[col][j]/100)**DF_Tot[j]))
    liab_plan_dict_tot[col] = temp_pv

pv_df_tot = pd.DataFrame(liab_plan_dict_tot, index = ['Present Values']).transpose()

############################################################################################################################################################
Plan_Return_YC_Tot = np.zeros(len(liab_plan_dict_tot))
for i in range (0,len(liab_plan_dict_tot)-1):
    Plan_Return_YC_Tot[i+1] = ((pv_df_tot['Present Values'][i+1])/pv_df_tot['Present Values'][i])-1
    
Plan_Return_YC_Tot = pd.DataFrame(Plan_Return_YC_Tot, columns=['Liability'], index=pv_df_tot.index)
liab_ret_tot_df = dm.merge_dfs(Plan_Return_YC_Tot, pv_df_tot)
#export monthly dataframe to excel
filepath = (dm.TS_FP+PLAN+ ' Liability & SC Returns & PV.xlsx')
writer = pd.ExcelWriter(filepath, engine='xlsxwriter')
liab_ret_tot_df.to_excel(writer, sheet_name=PLAN)
writer.save()


############################################################################################################################################################
df_DR_M = pd.read_excel("UPS Pension - Historical Liability Data - 9.30.21.xlsx",
                        sheet_name = PLAN ,skiprows = [0,1,2,3],usecols=[1,2],
                        na_values=[""],index_col=0)

Plan_PV_DR = np.zeros(len(df_DR_M))
for j in range (len(df_DR_M)):
    for i in range (len(Time_CF)):
        Plan_PV_DR[j] += (CF[i]/((1+df_DR_M['IRR'][j])**DF[i]))

pv_df_dr = pd.DataFrame(Plan_PV_DR, columns=['Present Values'], index=pv_df.index)
filepath = (dm.TS_FP+PLAN+ ' Liability PV Using DR.xlsx')
writer = pd.ExcelWriter(filepath, engine='xlsxwriter')
pv_df_dr.to_excel(writer, sheet_name='Present Values')
writer.save()

############################################################################################################################################################

def npv(irr, cfs, yrs):  
    return np.sum(cfs / (1. + irr) ** yrs)

def irr(cfs, yrs, x0, **kwargs):
    return np.asscalar(fsolve(npv, x0=x0, args=(cfs, yrs), **kwargs))


IRR = np.zeros(len(df_DR_M))
for j in range (len(df_DR_M)):
    cfs = np.append(np.negative(pv_df_dr['Present Values'][j]),CF)
    yrs = np.append(0, DF)
    IRR[j] += irr(cfs, yrs, .03)


############################################################################################################################################################
df_PVFB = pd.read_excel("YE2020 PBO SC Cashflows_QP.xlsx",sheet_name = "PVFB Cashflows Summary", skiprows = [0,1,2,3,4,5,6,7,8], usecols=[0,1,2,3,4], na_values=[""],index_col=0)
df_PVFB = df_PVFB.divide(12)

############################################################################################################################################################
