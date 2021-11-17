# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 21:41:41 2021

@author: Powis Forjoe
"""

import pandas as pd
from .import formats
from .import plots

def set_return_sheet(writer,df_returns,sheet_name='Monthly Historical Returns', sample_ret=False):
    """
    Create excel sheet for historical returns
    
    Parameters:
    writer -- excel writer
    df_returns -- dataframe
    sheet_name -- string
    """

    workbook = writer.book
    cell_format = formats.set_worksheet_format(workbook)
    df_empty = pd.DataFrame()
    df_empty.to_excel(writer, sheet_name=sheet_name, startrow=0, startcol=0)
    worksheet = writer.sheets[sheet_name]
    worksheet.set_column(0, 1000, 21, cell_format)
    row = 0
    col = 0
    
    #date format
    date_fmt = formats.set_number_format(workbook, num_format='mm/dd/yyyy')
    #num format
    num_fmt = formats.set_number_format(workbook,num_format='0')
    #percent format
    pct_fmt = formats.set_number_format(workbook,num_format='0.00%')
    #neg value format
    neg_value_fmt = formats.set_neg_value_format(workbook)
        
    row_dim = row + df_returns.shape[0]
    col_dim = col + df_returns.shape[1]
    
    df_returns.to_excel(writer, sheet_name=sheet_name, startrow=row , startcol=col)   
    if sample_ret:
        worksheet.conditional_format(row,col, row_dim, col,{'type':'no_blanks',
                                  'format':num_fmt})
    else:
        worksheet.conditional_format(row,col, row_dim, col,{'type':'no_blanks',
                                  'format':date_fmt})
    worksheet.conditional_format(row+1,col+1, row_dim, col_dim,{'type':'no_blanks',
                                  'format':pct_fmt})
    worksheet.conditional_format(row+1,col+1, row_dim, col_dim,{'type': 'cell',
                                                               'criteria': 'less than',
                                                               'value': 0,
                                                               'format': neg_value_fmt})
    if sample_ret:
        worksheet.insert_image(2, col_dim+2, 'simulated_returns.png',
                               {'x_scale': 0.5, 'y_scale': 0.5})
    
    return 0

def set_corr_sheet(writer,corr_df,sheet_name='Correlations', color=True):
    """
    Create excel sheet for correlations
    
    Parameters:
    writer -- excel writer
    corr_df -- dataframe
    sheet_name -- string
    """

    workbook = writer.book
    cell_format = formats.set_worksheet_format(workbook)
    df_empty = pd.DataFrame()
    df_empty.to_excel(writer, sheet_name=sheet_name, startrow=0, startcol=0)
    worksheet = writer.sheets[sheet_name]
    worksheet.set_column(0, 1000, 21, cell_format)
    row = 0
    col = 0
    
    #digits format
    digits_fmt = formats.set_number_format(workbook,num_format='0.0000')
        
    row_dim = row + corr_df.shape[0]
    col_dim = col + corr_df.shape[1]
    
    corr_df.to_excel(writer, sheet_name=sheet_name, startrow=row , startcol=col)   
    worksheet.conditional_format(row+1,col+1, row_dim, col_dim,{'type':'duplicate',
                                  'format':digits_fmt})
    if color:
        worksheet.conditional_format(row+1,col+1, row_dim, col_dim,{'type':'3_color_scale'})
    return 0

def set_ret_vol_sheet(writer,ret_vol_df,sheet_name='Return Statistics'):
    """
    Create excel sheet for ret and volatilty analytics
    
    Parameters:
    writer -- excel writer
    ret_vol_df -- dataframe
    sheet_name -- string
    """

    workbook = writer.book
    cell_format = formats.set_worksheet_format(workbook)
    df_empty = pd.DataFrame()
    df_empty.to_excel(writer, sheet_name=sheet_name, startrow=0, startcol=0)
    worksheet = writer.sheets[sheet_name]
    worksheet.set_column(0, 1000, 21, cell_format)
    row = 0
    col = 0
    
    #percent format
    pct_fmt = formats.set_number_format(workbook,num_format='0.00%')
    #digits format
    digits_fmt = formats.set_number_format(workbook,num_format='0.0000')
    
    row_dim = row + ret_vol_df.shape[0]
    col_dim = col + ret_vol_df.shape[1]
    
    ret_vol_df.to_excel(writer, sheet_name=sheet_name, startrow=row , startcol=col)   
    worksheet.conditional_format(row+1,col+1, row_dim, col+2,{'type':'no_blanks',
                                  'format':pct_fmt})
    worksheet.conditional_format(row+1,col+3, row_dim, col_dim,{'type':'no_blanks',
                                  'format':digits_fmt})
    
    return 0

def set_wgts_sheet(writer,wgts_df,sheet_name='Weights'):
    """
    Create excel sheet for plan weights
    
    Parameters:
    writer -- excel writer
    wgts_df -- dataframe
    sheet_name -- string
    """

    workbook = writer.book
    cell_format = formats.set_worksheet_format(workbook)
    df_empty = pd.DataFrame()
    df_empty.to_excel(writer, sheet_name=sheet_name, startrow=0, startcol=0)
    worksheet = writer.sheets[sheet_name]
    worksheet.set_column(0, 1000, 21, cell_format)
    row = 0
    col = 0
    
    #percent format
    pct_fmt = formats.set_number_format(workbook,num_format='0.00%')
    
    #digits format
    digits_fmt = formats.set_number_format(workbook,num_format='0.00')
        
    row_dim = row + wgts_df.shape[0]
    col_dim = col + wgts_df.shape[1]
    
    wgts_df.to_excel(writer, sheet_name=sheet_name, startrow=row , startcol=col)   
    worksheet.conditional_format(row+1,col+1, row_dim, col+1,{'type':'no_blanks',
                                  'format':pct_fmt})
    worksheet.conditional_format(row+1,col+2, row_dim, col+2,{'type':'no_blanks',
                                  'format':digits_fmt})
    worksheet.conditional_format(row+1,col+3, row_dim, col_dim,{'type':'no_blanks',
                                  'format':pct_fmt})
    return 0

def set_ef_port_sheet(writer,ports_df,sheet_name='Efficient Frontier Data'):
    """
    Create excel sheet for Efficient Frontier Portfolio data
    
    Parameters:
    writer -- excel writer
    ports_df -- dataframe
    sheet_name -- string
    """

    workbook = writer.book
    cell_format = formats.set_worksheet_format(workbook)
    df_empty = pd.DataFrame()
    df_empty.to_excel(writer, sheet_name=sheet_name, startrow=0, startcol=0)
    worksheet = writer.sheets[sheet_name]
    worksheet.set_column(0, 1000, 21, cell_format)
    row = 0
    col = 0
    
    #percent format
    pct_fmt = formats.set_number_format(workbook,num_format='0.00%')
    #digits format
    digits_fmt = formats.set_number_format(workbook,num_format='0.0000')
    #neg value format
    neg_value_fmt = formats.set_neg_value_format(workbook)
        
    row_dim = row + ports_df.shape[0]
    col_dim = col + ports_df.shape[1]
    
    #add df to sheet
    ports_df.to_excel(writer, sheet_name=sheet_name, startrow=row , startcol=col)   
    
    #ret_vol format
    worksheet.conditional_format(row+1,col+1, row_dim, col+3,{'type':'no_blanks',
                                  'format':pct_fmt})
    #sharpe format
    worksheet.conditional_format(row+1,col+4, row_dim, col+4,{'type':'no_blanks',
                                  'format':digits_fmt})
    #weights format
    worksheet.conditional_format(row+1,col+5, row_dim, col_dim,{'type':'no_blanks',
                                  'format':pct_fmt})
    #neg value format
    worksheet.conditional_format(row+1,col+1, row_dim, col_dim,{'type': 'cell',
                                                               'criteria': 'less than',
                                                               'value': 0,
                                                               'format': neg_value_fmt})
    
    #get asset alloc and eff front images
    aa_image_data = plots.get_image_data(plots.get_aa_fig(ports_df))
    ef_image_data = plots.get_image_data(plots.get_ef_fig(ports_df))
    
    worksheet.insert_image(2, col_dim+2, 'plotly.png', {'image_data': aa_image_data})
    worksheet.insert_image(30, col_dim+2, 'plotly.png', {'image_data': ef_image_data})

    return 0

def set_resamp_corr_sheet(writer, resamp_corr_df, sheet_name = 'Resamp Corr'):
    workbook = writer.book
    cell_format = formats.set_worksheet_format(workbook)
    df_empty = pd.DataFrame()
    df_empty.to_excel(writer, sheet_name=sheet_name, startrow=0, startcol=0)
    worksheet = writer.sheets[sheet_name]
    worksheet.set_column(0, 1000, 21, cell_format)
    row = 0
    col = 0
    #digits format
    digits_fmt = formats.set_number_format(workbook,num_format='0.0000')
        
    row_dim = row + resamp_corr_df.shape[0]
    col_dim = col + resamp_corr_df.shape[1]
    
    resamp_corr_df.to_excel(writer, sheet_name=sheet_name, startrow=row , startcol=col)   
    worksheet.conditional_format(row+1,col+1, row_dim, col_dim,{'type':'duplicate',
                                  'format':digits_fmt})
    
    return 0

def set_ff_ratio_matrix_sheet(writer,plan, fulfill_ret_dict):
    workbook = writer.book
    cell_format = formats.set_worksheet_format(workbook)
    df_empty = pd.DataFrame()
    df_empty.to_excel(writer, sheet_name=plan, startrow=0, startcol=0)
    worksheet = writer.sheets[plan]
    worksheet.set_column(0, 100, 21, cell_format)
    row = 2
    col = 0
    #title format
    title_format = formats.set_title_format(workbook)
    pct_fmt = formats.set_number_format(workbook,num_format='0.00%')
    
    for key in fulfill_ret_dict[plan]:
        row_dim = row + fulfill_ret_dict[plan][key].shape[0]
        col_dim = col + fulfill_ret_dict[plan][key].shape[1]
        worksheet.write(row-1,col,'{} Fully Funded Ratio Matrix'.format(key),title_format)
        fulfill_ret_dict[plan][key].to_excel(writer, sheet_name=plan, startrow=row, startcol=col)
        worksheet.conditional_format(row+1, col+1, row_dim, col_dim,{'type':'no_blanks','format':pct_fmt})
        row = row_dim + 2 + 1
    return 0