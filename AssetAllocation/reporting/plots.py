# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 17:33:37 2021

@authors: Roxton McNeal, Matt Johnston, Powis Forjoe
"""
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.figsize'] = 16, 8
import matplotlib.ticker as mtick
import plotly.graph_objects as go
import plotly.express as px

from io import BytesIO


sns.set(color_codes=True, font_scale=1.2)

CMAP_DEFAULT = sns.diverging_palette(20, 220, as_cmap=True)

COLOR_DICT = {'15+ STRIPS':'forestgreen','Long Corporate':'lightgreen','Ultra 30-Year UST Futures':'greenyellow',
              'Equity':'deeppink','Liquid Alternatives':'blue',
              'Private Equity':'darkred','Credit':'yellow','Real Estate':'orange',
              'Equity Hedges':'blueviolet','Cash':'khaki'}
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
    
    ax.set_xlabel('Expected Volatility')
    ax.set_ylabel('Expected Return')
    ax.grid(True)

def get_aa_fig(ports_df, color_dict = COLOR_DICT):
    
    df = format_df(ports_df)
    
    #Asset Allocation Plot
    aa_fig = go.Figure()
    
    for key in color_dict:
        aa_fig.add_trace(go.Scatter(
         x= df['Return'], y = df[key],
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
    aa_fig.update_xaxes(title_font_family = "Calibri",title_text = "<b>Returns</b>",
        title_font = {"size": 20},showline=True,linewidth=2,linecolor='black',mirror=False)
    aa_fig.update_yaxes(title_font_family = "Calibri",title_text = "<b>Weights</b>",range=(0,162),title_font = {"size": 20},
        showline=True,linewidth=2,linecolor='black',mirror=False)
    
    return aa_fig

def format_df(ports_df):
    df = ports_df.copy()
    df = 100*np.around(df,6)
    df['Sharpe'] = df['Return']/df['Volatility']
    return df

def get_ef_fig(ports_df):
    df = format_df(ports_df)
    ef_fig = px.scatter(df, x="Volatility", y="Return",color='Sharpe')
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
    ef_fig.update_xaxes(title_font_family = "Calibri",title_text = "<b>Expected Volatility</b>",title_font = {"size": 20},
                        showline=True,linewidth=2,linecolor='black',mirror=False)
    
    ef_fig.update_yaxes(title_font_family = "Calibri",title_text = "<b>Expected Return</b>",title_font = {"size": 20},
                     showline=True,linewidth=2,linecolor='black',mirror=False)
    return ef_fig

   
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
