# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 21:41:41 2021

@author: Powis Forjoe, Maddie Choi
"""

# TODO: rewrite code to make OOP-too many repeated code
import pandas as pd
from AssetAllocation.reporting import formats
from AssetAllocation.analytics import util
import AssetAllocation.reporting.formats as formats
import AssetAllocation.reporting.plots as plots

# TODO: add comments in it


def set_return_sheet(writer, df_returns, sheet_name='Monthly Historical Returns', sample_ret=False, set_neg_value_format=True):
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

    # date format
    date_fmt = formats.set_number_format(workbook, num_format='mm/dd/yyyy')
    # num format
    num_fmt = formats.set_number_format(workbook, num_format='0')
    # percent format
    pct_fmt = formats.set_number_format(workbook, num_format='0.00%')
    # neg value format
    neg_value_fmt = formats.set_neg_value_format(workbook)

    row_dim = row + df_returns.shape[0]
    col_dim = col + df_returns.shape[1]

    df_returns.to_excel(writer, sheet_name=sheet_name,
                        startrow=row, startcol=col)
    if sample_ret:
        worksheet.conditional_format(row, col, row_dim, col, {'type': 'no_blanks',
                                                              'format': num_fmt})
    else:
        worksheet.conditional_format(row, col, row_dim, col, {'type': 'no_blanks',
                                                              'format': date_fmt})
        worksheet.conditional_format(row+1, col+1, row_dim, col_dim, {'type': 'no_blanks',
                                                                      'format': pct_fmt})
        if set_neg_value_format:
            worksheet.conditional_format(row+1, col+1, row_dim, col_dim, {'type': 'cell',
                                                                          'criteria': 'less than',
                                                                          'value': 0,
                                                                          'format': neg_value_fmt})
    if sample_ret:
        worksheet.insert_image(2, col_dim+2, 'simulated_returns.png',
                               {'x_scale': 0.5, 'y_scale': 0.5})

    return 0


def set_corr_sheet(writer, corr_df, sheet_name='Correlations', color=True):
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

    # digits format
    digits_fmt = formats.set_number_format(workbook, num_format='0.0000')

    row_dim = row + corr_df.shape[0]
    col_dim = col + corr_df.shape[1]

    corr_df.to_excel(writer, sheet_name=sheet_name, startrow=row, startcol=col)
    worksheet.conditional_format(row+1, col+1, row_dim, col_dim, {'type': 'duplicate',
                                                                  'format': digits_fmt})
    if color:
        worksheet.conditional_format(
            row+1, col+1, row_dim, col_dim, {'type': '3_color_scale'})
    return 0


def set_ret_vol_sheet(writer, ret_vol_df, sheet_name='Return Statistics'):
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

    # percent format
    pct_fmt = formats.set_number_format(workbook, num_format='0.00%')
    # digits format
    digits_fmt = formats.set_number_format(workbook, num_format='0.0000')

    row_dim = row + ret_vol_df.shape[0]
    col_dim = col + ret_vol_df.shape[1]

    ret_vol_df.to_excel(writer, sheet_name=sheet_name,
                        startrow=row, startcol=col)
    worksheet.conditional_format(row+1, col+1, row_dim, col+2, {'type': 'no_blanks',
                                                                'format': pct_fmt})
    worksheet.conditional_format(row+1, col+3, row_dim, col_dim, {'type': 'no_blanks',
                                                                  'format': digits_fmt})

    return 0


def set_wgts_sheet(writer, wgts_df, sheet_name='Weights'):
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

    # percent format
    pct_fmt = formats.set_number_format(workbook, num_format='0.00%')

    # digits format
    digits_fmt = formats.set_number_format(workbook, num_format='0.00')

    row_dim = row + wgts_df.shape[0]
    col_dim = col + wgts_df.shape[1]

    wgts_df.to_excel(writer, sheet_name=sheet_name, startrow=row, startcol=col)
    worksheet.conditional_format(row+1, col+1, row_dim, col+1, {'type': 'no_blanks',
                                                                'format': pct_fmt})
    worksheet.conditional_format(row+1, col+2, row_dim, col+2, {'type': 'no_blanks',
                                                                'format': digits_fmt})
    worksheet.conditional_format(row+1, col+3, row_dim, col_dim, {'type': 'no_blanks',
                                                                  'format': pct_fmt})
    return 0


def set_ef_port_sheet(writer, ports_df, sheet_name='Efficient Frontier Data'):
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

    # percent format
    pct_fmt = formats.set_number_format(workbook, num_format='0.00%')
    # digits format
    digits_fmt = formats.set_number_format(workbook, num_format='0.0000')
    # neg value format
    neg_value_fmt = formats.set_neg_value_format(workbook)

    row_dim = row + ports_df.shape[0]
    col_dim = col + ports_df.shape[1]

    # add df to sheet
    ports_df.to_excel(writer, sheet_name=sheet_name,
                      startrow=row, startcol=col)

    # ret_vol format
    worksheet.conditional_format(row+1, col+1, row_dim, col+3, {'type': 'no_blanks',
                                                                'format': pct_fmt})
    # sharpe format
    worksheet.conditional_format(row+1, col+4, row_dim, col+4, {'type': 'no_blanks',
                                                                'format': digits_fmt})
    # weights format
    worksheet.conditional_format(row+1, col+5, row_dim, col_dim, {'type': 'no_blanks',
                                                                  'format': pct_fmt})
    # neg value format
    worksheet.conditional_format(row+1, col+1, row_dim, col_dim, {'type': 'cell',
                                                                  'criteria': 'less than',
                                                                  'value': 0,
                                                                  'format': neg_value_fmt})

    # get asset alloc and eff front images
    aa_image_data = plots.get_image_data(plots.get_aa_fig(ports_df))
    ef_image_data = plots.get_image_data(plots.get_ef_fig(ports_df))

    worksheet.insert_image(2, col_dim+2, 'plotly.png',
                           {'image_data': aa_image_data})
    worksheet.insert_image(30, col_dim+2, 'plotly.png',
                           {'image_data': ef_image_data})

    return 0


def set_resamp_corr_sheet(writer, resamp_corr_df, sheet_name='Resamp Corr'):
    workbook = writer.book
    cell_format = formats.set_worksheet_format(workbook)
    df_empty = pd.DataFrame()
    df_empty.to_excel(writer, sheet_name=sheet_name, startrow=0, startcol=0)
    worksheet = writer.sheets[sheet_name]
    worksheet.set_column(0, 1000, 21, cell_format)
    row = 0
    col = 0
    # digits format
    digits_fmt = formats.set_number_format(workbook, num_format='0.0000')

    row_dim = row + resamp_corr_df.shape[0]
    col_dim = col + resamp_corr_df.shape[1]

    resamp_corr_df.to_excel(writer, sheet_name=sheet_name,
                            startrow=row, startcol=col)
    worksheet.conditional_format(row+1, col+1, row_dim, col_dim, {'type': 'duplicate',
                                                                  'format': digits_fmt})

    return 0


def set_ff_ratio_matrix_sheet(writer, plan, fulfill_ret_dict):
    workbook = writer.book
    cell_format = formats.set_worksheet_format(workbook)
    df_empty = pd.DataFrame()
    df_empty.to_excel(writer, sheet_name=plan, startrow=0, startcol=0)
    worksheet = writer.sheets[plan]
    worksheet.set_column(0, 100, 21, cell_format)
    row = 2
    col = 0
    # title format
    title_format = formats.set_title_format(workbook)
    pct_fmt = formats.set_number_format(workbook, num_format='0.00%')

    for key in fulfill_ret_dict[plan]:
        row_dim = row + fulfill_ret_dict[plan][key].shape[0]
        col_dim = col + fulfill_ret_dict[plan][key].shape[1]
        worksheet.write(
            row-1, col, '{} Fully Funded Ratio Matrix'.format(key), title_format)
        fulfill_ret_dict[plan][key].to_excel(
            writer, sheet_name=plan, startrow=row, startcol=col)
        worksheet.conditional_format(
            row+1, col+1, row_dim, col_dim, {'type': 'no_blanks', 'format': pct_fmt})
        row = row_dim + 2 + 1
    return 0


def set_asset_liability_sheet(writer, tables_dict, sheet_name="Asset-Liability Returns", num_values=False):
    '''
    Create excel sheet for asset liability returns or market value tables

    Parameters
    ----------
    writer : Excel Writer
    tables_dict : Dictionary
    sheet_name : string
    mkt_values : Boolean
        True if values should be formatted in dollar values 
        False if values should be formatted in percentages

    '''
    workbook = writer.book
    cell_format = formats.set_worksheet_format(workbook)
    df_empty = pd.DataFrame()
    df_empty.to_excel(writer, sheet_name=sheet_name, startrow=0, startcol=0)
    worksheet = writer.sheets[sheet_name]
    worksheet.set_column(0, 1000, 21, cell_format)
    row = 1
    col = 0

    title_format = formats.set_title_format(workbook, center=True)
    # date format
    date_fmt = formats.set_number_format(
        workbook, num_format='mm/dd/yyyy', bold=True)
    # negative value format
    neg_value_fmt = formats.set_neg_value_format(workbook)

    if num_values:
        # number format
        value_fmt = formats.set_number_format(
            workbook, num_format='"$" #,##0.00')
    else:
        # percent format
        value_fmt = formats.set_number_format(workbook, num_format='0.00%')

    for key in tables_dict:
        row_dim = row + tables_dict[key].shape[0]
        col_dim = col + tables_dict[key].shape[1]
        worksheet.write(row-1, col+1, key, title_format)
        #worksheet.merge_range(row-1,col+1, row-1, col+2, key, title_format)
        tables_dict[key].to_excel(
            writer, sheet_name=sheet_name, startrow=row, startcol=col)
        worksheet.conditional_format(row, col, row_dim, col, {'type': 'no_blanks',
                                                              'format': date_fmt})
        worksheet.conditional_format(
            row+1, col+1, row_dim, col_dim, {'type': 'no_blanks', 'format': value_fmt})
        worksheet.conditional_format(row+1, col+1, row_dim, col_dim, {'type': 'cell', 'criteria': 'less than', 'value': 0,
                                                                      'format': neg_value_fmt})
        col = col_dim + 2

    return 0


def set_dollar_values_sheet(writer, df, sheet_name):
    """
    Create excel sheet for market values and present values to format values into $.00

    Parameters:
    writer -- excel writer
    df_pvs -- dataframe
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

    # date format
    date_fmt = formats.set_number_format(workbook, num_format='mm/dd/yyyy')
    # num format
    num_fmt = formats.set_number_format(workbook, num_format='"$" #,##0.00')

    row_dim = row + df.shape[0]
    col_dim = col + df.shape[1]

    df.to_excel(writer, sheet_name=sheet_name, startrow=row, startcol=col)

    worksheet.conditional_format(row, col, row_dim, col, {'type': 'no_blanks',
                                                          'format': date_fmt})
    worksheet.conditional_format(row+1, col+1, row_dim, col_dim, {'type': 'no_blanks',
                                                                  'format': num_fmt})

    return 0


def set_fs_data_sheet(writer, fs_data_dict, sheet_name="Funded Status Volatility"):
    '''
    Create excel sheet for asset liability returns or market value tables

    Parameters
    ----------
    writer : Excel Writer
    tables_dict : Dictionary
    sheet_name : string
    mkt_values : Boolean
        True if values should be formatted in dollar values 
        False if values should be formatted in percentages

    '''
    workbook = writer.book
    cell_format = formats.set_worksheet_format(workbook)
    df_empty = pd.DataFrame()
    df_empty.to_excel(writer, sheet_name=sheet_name, startrow=0, startcol=0)
    worksheet = writer.sheets[sheet_name]
    worksheet.set_column(0, 1000, 21, cell_format)
    row = 1
    col = 0

    title_format = formats.set_title_format(workbook, center=True)
    # date format
    date_fmt = formats.set_number_format(
        workbook, num_format='mm/dd/yyyy', bold=True)
    # percent format
    pct_fmt = formats.set_number_format(workbook, num_format='0.00%')
    # dollar format
    dollar_fmt = formats.set_number_format(workbook, num_format='"$" #,##0.00')
    # num format
    # num_fmt = formats.set_number_format(workbook,num_format='#,##0.00')

    # negative value format
    neg_value_fmt = formats.set_neg_value_format(workbook)

    for key in fs_data_dict:
        row_dim = row + fs_data_dict[key].shape[0]
        col_dim = col + fs_data_dict[key].shape[1]
        worksheet.write(row-1, col+1, key, title_format)
        fs_data_dict[key].to_excel(
            writer, sheet_name=sheet_name, startrow=row, startcol=col)

        worksheet.conditional_format(row, col, row_dim, col, {'type': 'no_blanks',
                                                              'format': date_fmt})

        # format asset and liability market values
        worksheet.conditional_format(
            row+1, col+1, row_dim, col_dim - 4, {'type': 'no_blanks', 'format': dollar_fmt})

        # funded status format
        worksheet.conditional_format(
            row+1, col+2, row_dim, col_dim - 3, {'type': 'no_blanks', 'format': pct_fmt})

        # fs gap format
        worksheet.conditional_format(
            row+1, col+3, row_dim, col_dim - 2, {'type': 'no_blanks', 'format': dollar_fmt})

        # 1yr and 6mo format
        worksheet.conditional_format(
            row+1, col+1, row_dim, col_dim, {'type': 'no_blanks', 'format': pct_fmt})

        worksheet.conditional_format(row+1, col+1, row_dim, col_dim, {'type': 'cell', 'criteria': 'less than', 'value': 0,
                                                                      'format': neg_value_fmt})
        col = col_dim + 2

    return 0


def set_ftse_data_sheet(writer, df, sheet_name):
    """
    Create excel sheet for market values and present values to format values into $.00

    Parameters:
    writer -- excel writer
    df_pvs -- dataframe
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

    # date format
    date_fmt = formats.set_number_format(workbook, num_format='mm/dd/yyyy')
    # num format
    num_fmt = formats.set_number_format(workbook, num_format='0.000000')

    #
    period_fmt = formats.set_number_format(workbook, num_format='0.00')

    row_dim = row + df.shape[0]
    col_dim = col + df.shape[1]

    df.to_excel(writer, sheet_name=sheet_name, startrow=row, startcol=col)

    worksheet.conditional_format(row, col, row, col_dim+1, {'type': 'no_blanks',
                                                            'format': date_fmt})
    worksheet.conditional_format(row, col, row, col, {'type': 'no_blanks',
                                                      'format': period_fmt})
    worksheet.conditional_format(row+1, col+1, row_dim, col_dim, {'type': 'no_blanks',
                                                                  'format': num_fmt})

    return 0

# TODO: Add market value table and edit fs_data table


def set_plan_ldi_sheet(writer, returns,  qtd_returns, ytd_returns, fs_data, sheet_name, dashboard_graphs=True):
    '''


    Parameters
    ----------
    writer : Excel Writer
    sheet_name : string
    returns_dict : Dictionary
        Dictionary with asset/liability returns tables
    liab_data_dict : Dictionary
        Dictionary with PV and IRR tables
    fs_data_dict : Dictionary
        Dictionary with Funded Status data tables
    plan : string

    '''
    workbook = writer.book
    cell_format = formats.set_worksheet_format(workbook)
    df_empty = pd.DataFrame()
    df_empty.to_excel(writer, sheet_name=sheet_name, startrow=0, startcol=0)
    worksheet = writer.sheets[sheet_name]
    worksheet.set_column(0, 1000, 21, cell_format)
    row = 1
    col = 0

    # title format
    title_format = formats.set_title_format(workbook, center=True)
    # date format
    date_fmt = formats.set_number_format(
        workbook, num_format='mm/dd/yyyy', bold=True)
    # percent format
    pct_fmt = formats.set_number_format(workbook, num_format='0.00%')
    # dollar format
    dollar_fmt = formats.set_number_format(workbook, num_format='"$" #,##0.00')
    # negative value format
    neg_value_fmt = formats.set_neg_value_format(workbook)

   # get dimensions of each dataframe
    ret_row_dim = row + returns.shape[0]
    ret_col_dim = col + returns.shape[1]

    qtd_row_dim = row + qtd_returns.shape[0]
    qtd_col_dim = col + qtd_returns.shape[1]

    ytd_row_dim = row + ytd_returns.shape[0]
    ytd_col_dim = col + ytd_returns.shape[1]

    fs_row_dim = row + fs_data.shape[0]

    # write titles of each table
    #monthly returns in column B
    worksheet.write(row-1, col+1, "Monthly Returns", title_format)

    # qtd returns in column f
    worksheet.write(row-1, col + ret_col_dim + 3,
                    "QTD Returns", title_format)

    # ytd returns in column J
    worksheet.write(row - 1, col + ret_col_dim  + qtd_col_dim + 5,
                    "YTD Returns", title_format)

    # ytd returns in column N
    worksheet.write(row-1, col + ret_col_dim  + qtd_col_dim + ytd_col_dim  + 7,
                     "Funded Status", title_format)

    # write  dataframes to excel
    returns.to_excel(writer, sheet_name=sheet_name, startrow=row, startcol=col)

    qtd_returns.to_excel(writer, sheet_name=sheet_name,
                    startrow=row, startcol=col + ret_col_dim + 2)

    ytd_returns.to_excel(writer, sheet_name=sheet_name,
                         startrow=row, startcol=col + ret_col_dim + qtd_col_dim + 4)

    fs_data.to_excel(writer, sheet_name=sheet_name, startrow=row,
                     startcol=col + ret_col_dim + qtd_col_dim + ytd_col_dim +  6)

    if dashboard_graphs:
        plots.get_asset_liab_ret_bar_plot(
            workbook, worksheet, sheet_name, ret_row_dim, position='V4')

        plots.get_fs_chart(workbook, worksheet, sheet_name, fs_row_dim,
                           fs_col_dim= col + ret_col_dim + qtd_col_dim + ytd_col_dim + 6, position='V22')

        plots.get_fs_vol_chart(workbook, worksheet, sheet_name, fs_row_dim,
                               fs_col_dim= col + ret_col_dim + qtd_col_dim + ytd_col_dim + 6, position='V40')
        
        plots.get_ytd_chart(workbook, worksheet, sheet_name, qtd_row_dim, col_dim =  col + ret_col_dim + 4, position = 'V62', plot_title = 'QTD Returns')
        
        plots.get_ytd_chart(workbook, worksheet, sheet_name,ytd_row_dim, col_dim = col + ret_col_dim + qtd_col_dim + 6, position = 'V85')


    # format asset and liability returns values
    worksheet.conditional_format(row, col, ret_row_dim, col, {
                                 'type': 'no_blanks', 'format': date_fmt})
    worksheet.conditional_format(
        row+1, col+1, ret_row_dim, ret_col_dim, {'type': 'no_blanks', 'format': pct_fmt})
    worksheet.conditional_format(row+1, col+1, ret_row_dim, ret_col_dim, {'type': 'cell', 'criteria': 'less than', 'value': 0,
                                                                          'format': neg_value_fmt})

    # qtd returns format
    worksheet.conditional_format(row+1, col + ret_col_dim + 2, qtd_row_dim,
                                 col + ret_col_dim + 2, {'type': 'no_blanks', 'format': date_fmt})
    worksheet.conditional_format(row+1, col + ret_col_dim + 3, qtd_row_dim,
                                 col + ret_col_dim + 3, {'type': 'no_blanks', 'format': pct_fmt})
    worksheet.conditional_format(row+1, col + ret_col_dim + 4, qtd_row_dim,
                                 col + ret_col_dim + 4, {'type': 'no_blanks', 'format': pct_fmt})

    # ytd returns format
    worksheet.conditional_format(row+1, col + ret_col_dim + qtd_col_dim + 4, ytd_row_dim,
                                 col + ret_col_dim + qtd_col_dim + 4, {'type': 'no_blanks', 'format': date_fmt})
    worksheet.conditional_format(row+1, col + ret_col_dim + qtd_col_dim + 5,ytd_row_dim,
                                 col + ret_col_dim + qtd_col_dim + 5, {'type': 'no_blanks', 'format': pct_fmt})
    worksheet.conditional_format(row+1, col + ret_col_dim + qtd_col_dim + 6,ytd_row_dim,
                                 col + ret_col_dim + qtd_col_dim + 6, {'type': 'no_blanks', 'format': pct_fmt})

    # Funded status format

    worksheet.conditional_format(row+1, col + ret_col_dim + qtd_col_dim + ytd_col_dim + 6, fs_row_dim,
                                 col + ret_col_dim + qtd_col_dim + ytd_col_dim + 6, {'type': 'no_blanks', 'format': date_fmt})
    # asset mv
    worksheet.conditional_format(row+1, col + ret_col_dim + qtd_col_dim + ytd_col_dim + 7, fs_row_dim,
                                 col + ret_col_dim + qtd_col_dim + ytd_col_dim + 7, {'type': 'no_blanks', 'format': dollar_fmt})
    #PV
    worksheet.conditional_format(row+1, col + ret_col_dim + qtd_col_dim + ytd_col_dim + 8, fs_row_dim,
                                 col + ret_col_dim + qtd_col_dim + ytd_col_dim + 8, {'type': 'no_blanks', 'format': dollar_fmt})
    #IRR
    worksheet.conditional_format(row+1, col + ret_col_dim + qtd_col_dim + ytd_col_dim + 9, fs_row_dim,
                                 col + ret_col_dim + qtd_col_dim + ytd_col_dim + 9, {'type': 'no_blanks', 'format': pct_fmt})
    #Funded  Status
    worksheet.conditional_format(row+1, col + ret_col_dim + qtd_col_dim + ytd_col_dim + 10, fs_row_dim,
                                 col + ret_col_dim + qtd_col_dim + ytd_col_dim + 10, {'type': 'no_blanks', 'format': pct_fmt})
    #FS Gap
    worksheet.conditional_format(row+1, col + ret_col_dim + qtd_col_dim + ytd_col_dim + 11, fs_row_dim,
                                 col + ret_col_dim + qtd_col_dim + ytd_col_dim + 11, {'type': 'no_blanks', 'format': dollar_fmt})
    #1y FSV
    worksheet.conditional_format(row+1, col + ret_col_dim + qtd_col_dim + ytd_col_dim + 12, fs_row_dim,
                                 col + ret_col_dim + qtd_col_dim + ytd_col_dim + 12, {'type': 'no_blanks', 'format': pct_fmt})
    #6mo FSV
    worksheet.conditional_format(row+1, col + ret_col_dim + qtd_col_dim + ytd_col_dim + 13, fs_row_dim,
                                 col + ret_col_dim + qtd_col_dim + ytd_col_dim + 13, {'type': 'no_blanks', 'format': pct_fmt})

    return 0


def set_liab_mv_cf_sheet(writer, df, sheet_name):
    """
    Create excel sheet for market values and present values to format values into $.00

    Parameters:
    writer -- excel writer
    df_pvs -- dataframe
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

    # date format
    date_fmt = formats.set_number_format(workbook, num_format='mm/dd/yyyy')
    # num format
    num_fmt = formats.set_number_format(workbook, num_format='"$" #,##0.00')

    row_dim = row + df.shape[0]
    col_dim = col + df.shape[1]

    df.to_excel(writer, sheet_name=sheet_name, startrow=row, startcol=col)

    worksheet.conditional_format(row, col, row_dim, col, {'type': 'no_blanks',
                                                          'format': date_fmt})
    worksheet.conditional_format(row, col, row, col_dim, {'type': 'no_blanks',
                                                          'format': date_fmt})
    worksheet.conditional_format(row+1, col+1, row_dim, col_dim, {'type': 'no_blanks',
                                                                  'format': num_fmt})

    return 0



def set_fs_data_sheet_SAA(writer, fs_data_df, sheet_name="Funded Status"):
    '''
    Create excel sheet for asset liability returns or market value tables

    Parameters
    ----------
    writer : Excel Writer
    tables_dict : Dictionary
    sheet_name : string
    mkt_values : Boolean
        True if values should be formatted in dollar values
        False if values should be formatted in percentages

    '''
    workbook = writer.book
    cell_format = formats.set_worksheet_format(workbook)
    df_empty = pd.DataFrame()
    df_empty.to_excel(writer, sheet_name=sheet_name, startrow=0, startcol=0)
    worksheet = writer.sheets[sheet_name]
    worksheet.set_column(0, 1000, 21, cell_format)
    row = 1
    col = 0

    title_format = formats.set_title_format(workbook, center=True)
    # date format
    date_fmt = formats.set_number_format(
        workbook, num_format='mm/dd/yyyy', bold=True)
    # percent format
    pct_fmt = formats.set_number_format(workbook, num_format='0.00%')
    # dollar format
    dollar_fmt = formats.set_number_format(workbook, num_format='"$" #,##0.00')
    # num format
    # num_fmt = formats.set_number_format(workbook,num_format='#,##0.00')

    # negative value format
    neg_value_fmt = formats.set_neg_value_format(workbook)


    row_dim = row + fs_data_df.shape[0]
    col_dim = col + fs_data_df.shape[1]
    worksheet.write(row-1, col+1, key, title_format)
    fs_data_dict[key].to_excel(
        writer, sheet_name=sheet_name, startrow=row, startcol=col)

    worksheet.conditional_format(row, col, row_dim, col, {'type': 'no_blanks',
                                                          'format': date_fmt})

    # format asset and liability market values
    worksheet.conditional_format(
        row+1, col+1, row_dim, col_dim - 4, {'type': 'no_blanks', 'format': dollar_fmt})

    # funded status format
    worksheet.conditional_format(
        row+1, col+2, row_dim, col_dim - 3, {'type': 'no_blanks', 'format': pct_fmt})

    # fs gap format
    worksheet.conditional_format(
        row+1, col+3, row_dim, col_dim - 2, {'type': 'no_blanks', 'format': dollar_fmt})

    # 1yr and 6mo format
    worksheet.conditional_format(
        row+1, col+1, row_dim, col_dim, {'type': 'no_blanks', 'format': pct_fmt})

    worksheet.conditional_format(row+1, col+1, row_dim, col_dim, {'type': 'cell', 'criteria': 'less than', 'value': 0,
                                                                  'format': neg_value_fmt})
    col = col_dim + 2

    return 0

