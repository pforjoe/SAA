# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 21:40:37 2021

@author: Powis Forjoe
"""

def set_worksheet_format(workbook):
    """
    Format sheet
    
    Parameters:
    workbook
    """

    cell_format = workbook.add_format()
    cell_format.set_font_name('Calibri')
    cell_format.set_font_size(11)
    cell_format.set_bg_color('#FFFFFF')
    cell_format.set_align('center')
    cell_format.set_align('vcenter')
    return cell_format

def set_number_format(workbook,num_format, bold=False):
    """
    Format numbers
    
    Parameters:
    workbook
    num_format
    bold -- boolean
    """

    num_format = workbook.add_format({'num_format': num_format, 'bold':bold})
    return num_format

def set_neg_value_format(workbook):
    return workbook.add_format({'font_color':'#9C0006'})

def set_title_format(workbook, center = False):
    """
    Format title
    
    Parameters:
    workbook
    """
    title_format = workbook.add_format()
    title_format.set_bold()
    title_format.set_font_color('#000000')
    title_format.set_font_name('Calibri')
    title_format.set_font_size(14)
    if center == True:
        title_format.set_align('center')
        
    return title_format


def set_date_format(workbook):
    date_format = workbook.add_format({"num_format": 'm-yy'})
    date_format.set_bold()
    date_format.set_font_color('#000000')
    date_format.set_font_name('Calibri')
    date_format.set_font_size(14)
    return date_format

def set_merge_format(workbook):
    merge_format = workbook.add_format()
    merge_format.set_bold()
    merge_format.set_font_color('#000000')
    merge_format.set_font_name('Calibri')
    merge_format.set_font_size(14)
    merge_format.set_align('center')
    merge_format.set_valign('vcenter')
    merge_format.set_border(1)