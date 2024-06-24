# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 17:33:37 2021

@authors: Roxton McNeal, Matt Johnston, Powis Forjoe
"""
import numpy as np
from datetime import datetime
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.figsize'] = 16, 8
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
import plotly.graph_objects as go
import plotly.express as px

from io import BytesIO


sns.set(color_codes=True, font_scale=1.2)

CMAP_DEFAULT = sns.diverging_palette(20, 220, as_cmap=True)

COLOR_DICT = {'15+ STRIPS':'forestgreen','Long Corporate':'lightgreen','Ultra 30Y Futures':'greenyellow',
              'Equity':'deeppink','Liquid Alternatives':'blue',
              'Private Equity':'darkred','Credit':'yellow','Real Estate':'orange',
              'Hedges':'blueviolet','Cash':'khaki'}

def get_image_data(fig, width=950):
    return BytesIO(fig.to_image(format="png", width=width))

def plot_mc_ports(sim_ports_df, max_sharpe_port):
    # Visualize the simulated portfolio for risk and return
    fig = plt.figure()
    ax = plt.axes()
    # matplotlib.rcParams['figure.figsize'] = 16, 8
    
    ax.set_title('Monte Carlo Simulated Allocation')
    
    # Simulated portfolios
    fig.colorbar(ax.scatter(sim_ports_df['volatility'],sim_ports_df['returns'], c=sim_ports_df['sharpe_ratio'], 
                        marker='o', cmap='RdYlGn', edgecolors='black'), label='Sharpe Ratio') 
    
    # Maximum sharpe ratio portfolio
    ax.scatter(max_sharpe_port['volatility'], max_sharpe_port['returns'], c='red', marker='*', s = 300, label='Max Sharpe Ratio')
    
    ax.set_xlabel('Expected Volatility')
    ax.set_ylabel('Expected Return')
    ax.grid(True)
    
def plot_risk_ret(targetrets,targetvols,plan,opt_sharpe, opt_var):
    
    fig = plt.figure()
    ax = plt.axes()
    # matplotlib.rcParams['figure.figsize'] = 16, 8
    
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0, symbol='%', is_latex=False))
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0, symbol='%', is_latex=False))
    
    ax.set_title('Efficient Frontier Portfolio')
    
    # Efficient Frontier
    fig.colorbar(ax.scatter(targetvols, targetrets, c=targetrets / targetvols, 
                            marker='x', cmap='RdYlGn', edgecolors='black'), label='Sharpe Ratio') 
    
    # Maximum Sharpe Portfolio
    ax.plot(plan.portfolio_stats(opt_sharpe['x'])[1], plan.portfolio_stats(opt_sharpe['x'])[0], 'r*', markersize =15.0)
    
    # Minimum Variance Portfolio
    ax.plot(plan.portfolio_stats(opt_var['x'])[1], plan.portfolio_stats(opt_var['x'])[0], 'b*', markersize =15.0)
    
    # Policy Portfolio
    ax.plot(plan.fsv,plan.policy_rets, 'k*', markersize =15.0)
    
    ax.set_xlabel('Surplus Volatility')
    ax.set_ylabel('Excess Return')
    ax.grid(True)

def get_aa_fig(ports_df, color_dict = COLOR_DICT):
    
    df = format_df(ports_df)
    
    #Asset Allocation Plot
    aa_fig = go.Figure()
    
    for key in color_dict:
        aa_fig.add_trace(go.Scatter(
         x= df['Excess Return'], y = df[key],
         name = key,
         mode = 'lines',
         line=dict(width=0.5, color=color_dict[key]),
         stackgroup = 'one'))
    aa_fig.update_layout(
            title = {'text':"<b>Mean Variance Asset Allocation</b>",'y':0.95,'x':0.5,'xanchor': 'center','yanchor': 'top'},
            title_font_family = "Calibri",
            titlefont = {"size":20},
            xaxis = dict(ticksuffix="%"),
            yaxis = dict(ticksuffix="%"),
            legend_title="<b>Assets</b>",
            plot_bgcolor='White'
        )
    aa_fig.update_xaxes(title_font_family = "Calibri",title_text = "<b>Excess Return</b>",
                        # range=(0,get_max_range(df)['x_axis']),
        title_font = {"size": 20},showline=True,linewidth=2,linecolor='black',mirror=False)
    aa_fig.update_yaxes(title_font_family = "Calibri",title_text = "<b>Allocation</b>",
                        # range=(0,get_max_range(df)['y_axis']),
                        title_font = {"size": 20},showline=True,linewidth=2,linecolor='black',mirror=False)
    
    return aa_fig

def format_df(ports_df):
    df = ports_df.copy()
    df = 100*np.round(df,6)
    df['Sharpe'] = np.round(df['Excess Return']/df['Surplus Volatility'],4)
    return df[:get_max_return_index(df)]

def get_max_return_index(ports_df):
    excess_ret_series = ports_df['Excess Return']
    max_ret_index = excess_ret_series.index[excess_ret_series == ports_df['Excess Return'].max()][0]
    return max_ret_index + 1
    
def get_ef_fig(ports_df):
    df = format_df(ports_df)
    ef_fig = px.scatter(df, x="Surplus Volatility", y="Excess Return",color='Sharpe')
    ef_fig.update_layout(
        title={
                'text': "<b>Mean Variance Efficient Frontier</b>",
                'y':0.9,
                'x':0.5,'xanchor': 'center',
                'yanchor': 'top'
                },
        title_font_family="Calibri",
        titlefont = {"size":20},
            xaxis = dict(tickfont = dict(size=14),ticksuffix="%"),
           yaxis = dict(ticksuffix="%"),
        showlegend=True,
        plot_bgcolor='White'
                     )
    ef_fig.update_xaxes(title_font_family = "Calibri",title_text = "<b>Surplus Volatility</b>",title_font = {"size": 20},
                        showline=True,linewidth=2,linecolor='black',mirror=False)
    
    ef_fig.update_yaxes(title_font_family = "Calibri",title_text = "<b>Excess Return</b>",title_font = {"size": 20},
                     showline=True,linewidth=2,linecolor='black',mirror=False)
    return ef_fig

def get_resamp_corr_fig(corr_df, asset_liab):
    corr_fig = go.Figure()
    for col in corr_df.columns:
        corr_fig.add_trace(go.Scatter(x=corr_df.index, y=corr_df[col],
                            mode='lines+markers',
                            name=asset_liab+'/'+col))
    corr_fig.update_layout(
            title = {'text':"<b>Resampled Correlations</b>",'y':0.95,'x':0.5,'xanchor': 'center','yanchor': 'top'},
            title_font_family = "Calibri",
            titlefont = {"size":20},
            legend_title="<b>Correlations</b>",
            plot_bgcolor='White'
        )
    corr_fig.update_xaxes(title_font_family = "Calibri",title_text = "<b>Sample</b>",
                        # range=(0,get_max_range(df)['x_axis']),
        title_font = {"size": 20},showline=True,linewidth=2,linecolor='black',mirror=False)
    corr_fig.update_yaxes(title_font_family = "Calibri",title_text = "<b>Correlation</b>",
                        # range=(0,get_max_range(df)['y_axis']),
                        title_font = {"size": 20},showline=True,linewidth=2,linecolor='black',mirror=False)
    return corr_fig

def draw_heatmap(corr_df, half=True):
    """
    """
    sns.set(style="white")
    
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr_df, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = half
    
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=corr_df.shape)
    
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr_df
                ,mask=mask 
                ,cmap=CMAP_DEFAULT
                ,center=0
                ,square=True
                ,linewidths=1
                ,cbar_kws={"shrink": 1} 
                ,annot=True
                ,fmt=".2f"
                ,cbar=False
               )

def get_sim_return_fig(stochmv):
    fig = sns.pairplot(stochmv.returns_df, corner=True)
    fig.fig.suptitle("Simulated Returns")
    plt.savefig('simulated_returns.png')
    
    
    

def display_fs_vol(report_dict):
    #get last 12 months of 1yr and 6mo vol data
    #fs = report_dict['fs_data'][Plan]['Funded Status'].tail(12)
    fs_vol_1y = report_dict['fs_data']['1Y FSV'].tail(12)
    fs_vol_6mo = report_dict['fs_data']['6mo FSV'].tail(12)
    
    #set theme
    sns.set_style("whitegrid", {'axes.grid' : False})
    plt.gca().yaxis.grid(True)
    
    #set x and y axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1, bymonthday = -1))
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=None, symbol='%', is_latex=False))
    
    #plot 1yr vol and 6mo vol
    #plt.plot(fs, label = "Funded Status", color = "navy")
    
    plt.plot(fs_vol_1y, label = "1yr FSV", color = "navy")
    plt.plot(fs_vol_6mo ,label = "6mth FSV", color = "#ff8c00")
    
    # Rotate X-Axis Ticks by 45-degrees
    plt.xticks(rotation = 45) 
    #set title
    plt.title("Funded Status")
#     plt.title("Realized Funded Status Volatility")
    plt.legend()
    plt.show()
    
    
def get_asset_liab_ret_bar_plot(workbook, worksheet, sheet_name, ret_row_dim, position):
    #specify what type of chart
    returns_chart = workbook.add_chart({'type':'column'})

    #add asset returns data to bar chart
    returns_chart.add_series({
        'categories': [sheet_name, ret_row_dim-12, 0, ret_row_dim, 0], 
        'values': [sheet_name, ret_row_dim-12, 1, ret_row_dim, 1],
        'name':"Asset"})
    
    #add liability returns data to bar chart
    returns_chart.add_series({
        'categories': [sheet_name, ret_row_dim-12, 0, ret_row_dim, 0], 
        'values': [sheet_name, ret_row_dim-12, 2, ret_row_dim, 2],
        'name': 'Liabilty'})
    
    #set x axis
    returns_chart.set_x_axis({'label_position' : 'low',
                        'date_axis': True,
                       'num_format' : 'mmm-yy',
                       'num_font':{'name': 'Arial','color':'#616161 ','rotation': -45},
                      # 'minor_unit':1,
                       #'minor_unit_type': 'days',
                       #'major_unit':      1,
                       #'major_unit_type': 'months',
                       'line':{'color':'#D3D3D3'}
                       })
    
    #set y axis format
    returns_chart.set_y_axis({'num_format':'0%',
                              'num_font':  {'name': 'Arial', 'color':'#616161 '},
                              'line':{'none':True},
                             'major_gridlines': {
                                 'visible' : 1,
                                 'line' : { 'color' : '#D3D3D3'}
                                 },
                             'major_unit':0.02
                             })
    
    #set chart title
    returns_chart.set_title({'name':sheet_name + " Plan - FTSE Curve",
                             'name_font':  {'name': 'Arial','color':'#616161 ','bold':False,'size':14}})
    
    #set legend position
    returns_chart.set_legend({'position':'bottom',
                              'font': {'name': 'Arial','color':'#616161 '}
                              })
    returns_chart.set_chartarea({'border':{'none':True}})
    #add chart to sheet and scale
    returns_chart.set_size({'x_scale': 1.5, 'y_scale': 1})
    worksheet.insert_chart(position, returns_chart)   
    
    

def get_fs_chart(workbook, worksheet, sheet_name, fs_row_dim, fs_col_dim, position):
    #specify what type of chart
    fs_chart = workbook.add_chart({'type':'line'})

    #add asset returns data to bar chart
    fs_chart.add_series({
        'categories': [sheet_name, fs_row_dim-12, fs_col_dim, fs_row_dim, fs_col_dim], 
        'values': [sheet_name, fs_row_dim-12, fs_col_dim+4, fs_row_dim, fs_col_dim+4],
        'name':"Funded Status"})
    
    #set x axis
    fs_chart.set_x_axis({
                       'date_axis': True,

                     'num_format' :  'mmm-yy',
                     'num_font':{'rotation':-45,'name': 'Arial','color':'#616161 '},
                     'minor_unit':1,
                     'minor_unit_type': 'days',
                     'major_unit':      1,
                     'major_unit_type': 'months',
                     'line':{'color':'#D3D3D3'}

                       })
    
    #set y axis format
    fs_chart.set_y_axis({'num_format':'0%',

                         'num_font':  {'name': 'Arial', 'color':'#616161 '},
                         'line':{'none':True},
                        'major_gridlines': {
                            'visible' : 1,
                            'line' : { 'color' : '#D3D3D3'}
                            },
                        'major_unit':0.02
                        })
    
    #set chart title
    fs_chart.set_title({'name':"Funded Status - " + sheet_name,
                        'name_font':  {'name': 'Arial','color':'#616161 ','bold':False,'size':14}})
    
    fs_chart.set_legend({'position': 'none',
                         'font': {'name': 'Arial','color':'#616161 '}
                         })

    fs_chart.set_chartarea({'border':{'none':True}})
    
    #add chart to sheet and scale
    fs_chart.set_size({'x_scale': 1.5, 'y_scale': 1})
    worksheet.insert_chart(position, fs_chart)  
    
def get_fs_vol_chart(workbook, worksheet, sheet_name, fs_row_dim, fs_col_dim, position):
    #specify what type of chart
    fs_vol_chart = workbook.add_chart({'type':'line'})

    #add asset returns data to bar chart
    fs_vol_chart.add_series({
        'categories': [sheet_name, fs_row_dim-24, fs_col_dim, fs_row_dim, fs_col_dim], 
        'values': [sheet_name, fs_row_dim-24, fs_col_dim+6, fs_row_dim, fs_col_dim+6],
        'name':"1yr FSV"})
    
    #set x axis
    fs_vol_chart.set_x_axis({
                       'date_axis': True,
                     'num_format' : 'mmm-yy',
                     'num_font':{'rotation':-45,'name': 'Arial','color':'#616161 '},
                     'minor_unit':1,
                     'minor_unit_type': 'days',
                     'major_unit':      1,
                     'major_unit_type': 'months',
                     'line':{'color':'#D3D3D3'}

                       })
    
    #set y axis format
    fs_vol_chart.set_y_axis({'num_format':'0%',
                             'num_font':  {'name': 'Arial', 'color':'#616161 '},
                             'line':{'none':True},
                            'major_gridlines': {
                                'visible' : 1,
                                'line' : { 'color' : '#D3D3D3'}
                                },
                            'major_unit':0.02
                            })
    
    #set chart title
    fs_vol_chart.set_title({'name':"Realized Funded Status Volatility",
                            'name_font':  {'name': 'Arial','color':'#616161 ','bold':False,'size':12}
                            })
    fs_vol_chart.set_chartarea({'border':{'none':True}})
    fs_vol_chart.set_legend({'position': 'bottom',
                             'font': {'name': 'Arial','color':'#616161 '}
                                      })
    fs_vol_chart.set_chartarea({'border':{'none':True}})
    
    #add chart to sheet and scale
    fs_vol_chart.set_size({'x_scale': 1.5, 'y_scale': 1})
    worksheet.insert_chart(position, fs_vol_chart)  
    
def get_ytd_chart(workbook, worksheet, sheet_name, row_dim, col_dim, position, plot_title = 'YTD Returns'):
    #specify what type of chart
    ytd_chart = workbook.add_chart({'type':'column'})

    #add asset returns data to bar chart
    ytd_chart.add_series({
        'categories': [sheet_name, 1, col_dim-1, 1, col_dim],
        'values': [sheet_name, row_dim, col_dim-1, row_dim, col_dim],
        'name':'YTD',
        'data_labels':{'value':True,'num_format': '0.00%','font':  {'name': 'Arial','color':'#616161 '},}
        })

    #set x axis
    ytd_chart.set_x_axis({'text_axis': True,
                          'label_position' : 'low',
                          'num_font':  {'name': 'Arial','color':'#616161 ', 'size': 9},
                          'line':{'color':'#D3D3D3'}
                          })
    
    #set y axis format
    ytd_chart.set_y_axis({'num_format':'0%',
                          'num_font':  {'name': 'Arial', 'color':'#616161 ','size': 9},
                          'line':{'none':True},
                         'major_gridlines': {
                             'visible' : 1,
                             'line' : { 'color' : '#D3D3D3'}},
                         'major_unit':0.02
    })
   
    
    #set chart title
    ytd_chart.set_title({'name': plot_title,
                         'name_font':  {'name': 'Arial','color':'#616161 ','bold':False, 'size':12}})

    ytd_chart.set_chartarea({'border':{'none':True}})
    
    ytd_chart.set_legend({'none':True})
    #add chart to sheet and scale
    ytd_chart.set_size({'x_scale': 1.5, 'y_scale': 1})
    worksheet.insert_chart(position, ytd_chart)   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    