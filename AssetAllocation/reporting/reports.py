# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 21:51:38 2021

@author: NVG9HXP
"""

import pandas as pd
from ..analytics import summary
# from ...analytics import summary
# from ...analytics import util
# from ...analytics.corr_stats import get_corr_rank_data
# from ...analytics.historical_selloffs import get_hist_sim_table
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

def get_ef_portfolios_report(reportname, ports_df, pp_inputs):
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
    
    
    sheets.set_ret_vol_sheet(writer, pp_inputs['ret_vol'])
    sheets.set_corr_sheet(writer, pp_inputs['corr'])
    sheets.set_wgts_sheet(writer, pp_inputs['weights'])
    sheets.set_ef_port_sheet(writer, ports_df)
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