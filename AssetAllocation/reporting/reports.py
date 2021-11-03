# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 21:51:38 2021

@author: Powis Forjoe
"""

import pandas as pd
from  ..datamanger import datamanger as dm
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

def get_ef_portfolios_report(reportname, plan):
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

def get_stochmv_ef_portfolios_report(reportname, stochmv):
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