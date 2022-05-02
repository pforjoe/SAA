# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 21:51:38 2021

@author: Powis Forjoe, Maddie Choi
"""
#TODO: rewrite code to make OOP-too many repeated code

import pandas as pd
from  ..datamanager import datamanager as dm
from .import sheets
import os


def get_reportpath(reportname):
    """
    Gets the file path where the report will be stored

    Parameters
    ----------
    reportname : String
        Name of report

    Returns
    -------
    string
        File path

    """
    
    cwd = os.getcwd()
    reports_fp = '\\reports\\'
    filename = reportname +'.xlsx'
    return cwd + reports_fp + filename

def get_ts_path(reportname):
    """
    Gets the file path where the report will be stored

    Parameters
    ----------
    reportname : String
        Name of report

    Returns
    -------
    string
        File path

    """
    
    cwd = os.getcwd()
    ts_fp = '\\data\\time_series\\'
    filename = reportname +'.xlsx'
    return cwd + ts_fp + filename

def get_plan_inputpath(inputname):
    """
    Gets the file path where the output report will be stored

    Parameters
    ----------
    reportname : String
        Name of report

    Returns
    -------
    string
        File path

    """
    
    filename = inputname +'.xlsx'
    return dm.PLAN_INPUTS_FP + filename

def get_output_report(reportname, output_dict):
    """
    Generates output report

    Parameters
    ----------
    reportname : string
        Name of report.
    output_dict : dict
        dictionary containing ret_vol, weights and corr analytics.
    
    Returns
    -------
    None. An excel report called [reportname].xlsx is created 

    """
    #get file path and create excel writer
    filepath = get_reportpath(reportname)
    writer = pd.ExcelWriter(filepath, engine='xlsxwriter')
    
    
    sheets.set_ret_vol_sheet(writer, output_dict['ret_vol'])
    sheets.set_corr_sheet(writer, output_dict['corr'])
    sheets.set_wgts_sheet(writer, output_dict['weights'])
    try:
        sheets.set_return_sheet(writer, output_dict['returns'])
    except KeyError:
        pass
    #save file
    print_report_info(reportname, filepath)
    writer.save()

def get_ef_portfolios_report(reportname, plan, bnds=pd.DataFrame):
    """
    Generates output report

    Parameters
    ----------
    reportname : string
        Name of report.
    output_dict : dict
        dictionary containing ret_vol, weights and corr analytics.
    
    Returns
    -------
    None. An excel report called [reportname].xlsx is created 

    """
    #get file path and create excel writer
    filepath = get_reportpath(reportname)
    writer = pd.ExcelWriter(filepath, engine='xlsxwriter')
    
    pp_dict = plan.get_pp_dict()
    
    sheets.set_ret_vol_sheet(writer, pp_dict['Asset/Liability Returns/Vol'])
    sheets.set_corr_sheet(writer, pp_dict['Corr'])
    if not(bnds.empty):
        sheets.set_ret_vol_sheet(writer, bnds, 'bounds')
    try:
        ports_df = plan.ports_df
        sheets.set_ef_port_sheet(writer, ports_df)
        
    except TypeError:
        print('efficient frontier sheet not added\nRun plan.compute_eff_frontier(bnd, cons, num_ports) function')
        pass
    
    try:
        sheets.set_return_sheet(writer, pp_dict['Historical Returns'])
    except AttributeError:
        pass
    
    #save file
    print_report_info(reportname, filepath)
    writer.save()

def get_stochmv_ef_portfolios_report(reportname, stochmv, bnds=pd.DataFrame):
    """
    Generates output report

    Parameters
    ----------
    reportname : string
        Name of report.
    output_dict : dict
        dictionary containing ret_vol, weights and corr analytics.
    
    Returns
    -------
    None. An excel report called [reportname].xlsx is created 

    """
    #get file path and create excel writer
    filepath = get_reportpath(reportname)
    writer = pd.ExcelWriter(filepath, engine='xlsxwriter')
    
    pp_dict = stochmv.init_plan.get_pp_dict()
    
    sheets.set_ret_vol_sheet(writer, pp_dict['Asset/Liability Returns/Vol'])
    sheets.set_corr_sheet(writer, pp_dict['Corr'])
    if not(bnds.empty):
        sheets.set_ret_vol_sheet(writer, bnds, 'bounds')
    try:
        ports_df = stochmv.opt_ports_df
        sheets.set_ef_port_sheet(writer, ports_df)
        
    except TypeError:
        print('efficient frontier sheet not added\nRun plan.compute_eff_frontier(bnd, cons, num_ports) function')
        pass
    
    try:
        sheets.set_return_sheet(writer, pp_dict['Historical Returns'])
    except AttributeError:
        pass
    sheets.set_return_sheet(writer, stochmv.returns_df,sheet_name='Simulated Returns',sample_ret=True)
    
    for key in stochmv.resamp_corr_dict:
        sheets.set_resamp_corr_sheet(writer, stochmv.resamp_corr_dict[key], sheet_name= key + ' Resamp Corr')
        
    #save file
    print_report_info(reportname, filepath)
    writer.save()
    if os.path.exists('simulated_returns.png'):
            os.remove('simulated_returns.png')

def get_ff_report(reportname, fulfill_ret_dict,plan_list):
    filepath = get_reportpath(reportname)
    writer = pd.ExcelWriter(filepath, engine='xlsxwriter')
    for plan in plan_list:
        sheets.set_ff_ratio_matrix_sheet(writer, plan, fulfill_ret_dict)
    writer.save()  

def print_report_info(reportname, filepath):
    """
    Print name of report and location

    Parameters
    ----------
    reportname : string
        Name of report.
    filepath : string
        flie location.

    Returns
    -------
    None.

    """
    folder_location = filepath.replace(reportname+'.xlsx', '')
    print('"{}.xlsx" report generated in "{}" folder'.format(reportname,folder_location))

def get_liability_returns_report(report_dict,report_name = "liability_returns"):
    '''
    

    Parameters
    ----------
    report_dict : dictionary
        report dictionary.
    report_name : string
        name of excel eport

    Returns
    -------
    None.

    '''
    
    #get file path and create excel writer
    filepath = get_reportpath(report_name)
    writer = pd.ExcelWriter(filepath, engine='xlsxwriter')
    
    #set up sheets for report
    sheets.set_return_sheet(writer, report_dict["Liability Returns"], sheet_name = "Liability Returns")
    sheets.set_dollar_values_sheet(writer, report_dict["Liability Market Values"], sheet_name = "Liability Market Values")
    sheets.set_return_sheet(writer, report_dict["Asset Returns"], sheet_name = "Asset Returns")
    sheets.set_dollar_values_sheet(writer, report_dict["Asset Market Values"], sheet_name = "Asset Market Values")
    sheets.set_dollar_values_sheet(writer, report_dict["Present Values"], sheet_name = "Present Values") 
    sheets.set_return_sheet(writer, report_dict["IRR"], sheet_name = "IRR")
    sheets.set_asset_liability_sheet(writer, report_dict["asset_liab_ret_dict"])
    sheets.set_asset_liability_sheet(writer, report_dict["asset_liab_mkt_val_dict"], sheet_name = "Asset-Liability Mkt Values", num_values = True)
    sheets.set_fs_data_sheet(writer, report_dict["fs_data"])

    #save file
    print_report_info(report_name, filepath)
    writer.save()

def get_plan_data_report(plan_data_dict, report_name = "plan_data"):
    '''
    

    Parameters
    ----------
    plan_data_dict : dictionary
        plan data dictionary (includes plan market values and returns)
    file_name : string
        name of excel report. The default is "plan_data".

    Returns
    -------
    None.

    '''
    #creates excel report with updated plan market vallues and returns
    filepath = get_ts_path(report_name)
    writer = pd.ExcelWriter(filepath, engine = 'xlsxwriter')
    sheets.set_dollar_values_sheet(writer, plan_data_dict['mkt_value'], sheet_name='mkt_value')
    sheets.set_return_sheet(writer, plan_data_dict["return"], sheet_name='return', set_neg_value_format= True)

    #save file
    print_report_info(report_name, filepath)
    writer.save()
    
def get_ftse_data_report(ftse_dict, report_name = "ftse_data"):
    '''
    

    Parameters
    ----------
    ftse_dict : dictionary

    file_name : string
        name of excel report. The default is "plan_data".

    Returns
    -------
    None.

    '''
    #creates excel report with updated ftse data
    filepath = get_ts_path(report_name)
    
    
    writer = pd.ExcelWriter(filepath, engine = 'xlsxwriter')
    sheets.set_ftse_data_sheet(writer, ftse_dict['new_data'], sheet_name='new_data')
    sheets.set_ftse_data_sheet(writer, ftse_dict['old_data'], sheet_name='old_data')

    #save file
    print_report_info(report_name, filepath)
    writer.save()
    
def get_ldi_report(report_dict, report_name = "ldi_report"):
    #creates excel report with updated ftse data
    filepath = get_reportpath(report_name)
    writer = pd.ExcelWriter(filepath, engine = 'xlsxwriter')
  
    for key in report_dict:
        temp_mv_fs_df = dm.merge_dfs(report_dict[key]['market_values'], report_dict[key]['fs_data'], dropna = False)
        sheets.set_plan_ldi_sheet(writer, report_dict[key]['returns'], 
                                  report_dict[key]['pv_irr'], 
                                  temp_mv_fs_df,
                                  sheet_name = key)
       
    #save file
    print_report_info(report_name, filepath)
    writer.save()
    
def get_liab_mv_cf_report(plan_mv_cfs_dict, report_name = "liab_mv_cfs"):
    '''
    

    Parameters
    ----------
    plan_mv_cfs_dict : TYPE
        DESCRIPTION.
    report_name : TYPE, optional
        DESCRIPTION. The default is "liab_mv_cfs".

    Returns
    -------
    None.

    '''
    
    filepath = get_ts_path(report_name)
    writer = pd.ExcelWriter(filepath, engine = 'xlsxwriter')
    
    for plan in plan_mv_cfs_dict:
        sheets.set_liab_mv_cf_sheet(writer, plan_mv_cfs_dict[plan], plan)
    
    #save file
    print_report_info(report_name, filepath)
    writer.save()

pd.merge