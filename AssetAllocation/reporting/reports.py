# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 21:51:38 2021

@author: Powis Forjoe
"""

import pandas as pd
from  ..datamanger import datamanger as dm
from ..analytics import summary
from ..analytics.util import add_sharpe_col
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

def get_outputpath(outputname):
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
    
    filename = outputname +'.xlsx'
    return dm.OUTPUTS_FP + filename

    
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
    filepath = get_outputpath(reportname)
    writer = pd.ExcelWriter(filepath, engine='xlsxwriter')
    
    
    sheets.set_ret_vol_sheet(writer, output_dict['ret_vol'])
    sheets.set_corr_sheet(writer, output_dict['corr'])
    sheets.set_wgts_sheet(writer, output_dict['weights'])
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
    
    pp_dict = summary.get_pp_dict(plan)
    ret_vol_df = dm.merge_dfs(pp_dict['Asset/Liability Returns'], pp_dict[ 'Asset/Liability Vol'])
    
    sheets.set_ret_vol_sheet(writer, add_sharpe_col(ret_vol_df))
    sheets.set_corr_sheet(writer, pp_dict['Corr'])
    sheets.set_wgts_sheet(writer, pp_dict['Policy Weights'])
    
    try:
        ports_df = dm.get_ports_df(plan.eff_frontier_trets,
                               plan.eff_frontier_tvols,
                               plan.eff_frontier_tweights,
                               plan.symbols)
        sheets.set_ef_port_sheet(writer, ports_df)
        
    except TypeError:
        print('efficient frontier sheet not added\nRun plan.compute_eff_frontier(bnd, cons, num_ports) function')    
    #save file
    print_report_info(reportname, filepath)
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