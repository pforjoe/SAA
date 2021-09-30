# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 11:41:04 2021

@author: NVG9HXP
"""
import pandas as pd
import scipy.interpolate
import numpy as np
#INPUTS
#broaad mkt ret assumptions
TSY_MKT_RET = .02
MKT_RISK_PREM = .04
FI_TERM_PREM = 0
BOND_OAS_CR = .75

tsy_mkt_ret = .02
mkt_risk_prem = .04
fi_term_prem = 0
bond_oas_cr = .75


#fi
duration_list = [20.29202932,25.463069,15.451229,4.572578,19.400993]
spread_list = [0.00887228863550573,0,0.01223386,0.006225489,0]
oas_list = [0.0125,0,0.01223386,0.006225489,0]
hist_spread = [.01221987588,0,.01750158059,.01096302079,0]
fi_hist_vol_list = [.00417988718907079,0,.00551459669897851,.0061080629189123,0]
fi_index_list = ['Liability','15+ STRIPS','Long Corporate','Int Corporate','Ultra 30-Year UST Futures']
fi_col_list = ['duration', 'oas','spread', 'hist_spread', 'hist_vol']

fi_df = pd.DataFrame(list(zip(duration_list,oas_list,spread_list,hist_spread,fi_hist_vol_list)),
                            columns = fi_col_list,
                            index = fi_index_list)

fi_df['vol_ratio'] = fi_df['hist_vol'] / fi_df['hist_spread']
fi_df['vol_ratio'] = fi_df['vol_ratio'].fillna(0)
fi_df['curr_vol'] = fi_df['vol_ratio'] * fi_df['spread']
fi_df['curr_vol'] = fi_df['curr_vol'].fillna(0)

#rsa
rsa_hist_vol_list = [0.150371405,0.202616909,0.154880908,0.176093174,0.148960156,0.202616909,0.161926398,0.075167594,0.047647051,0.227901988]
rsa_pros_vol_list = [0.167458735,0.201407402,0.188566351,0.206620778,0.181357879,0.201407402,0.180326771,0.083709202,0.053061384,0.253799442]

rsa_index_list = ['S&P 500','Russell 2000','MSCI EAFE','MSCI Emerging Markets','MSCI ACWI',
                  'Private Equity','Dow Jones REIT','Barclays HY','Global HF','GS Commodity']
rsa_col_list = ['hist_vol', 'pros_vol']

rsa_df = pd.DataFrame(list(zip(rsa_hist_vol_list, rsa_pros_vol_list)),
                            columns = rsa_col_list,
                            index = rsa_index_list)

rsa_df['vol_ratio'] = rsa_df['pros_vol']/rsa_df['hist_vol']



#implied_rate vol
tsy_fut_opt_vol_3m = [.217912817112998,.613918118396888,.70058702463222,.801137099972603]

#swaption_vol
swaption_vol_3m = [.2983,.6219,.7486,.7813]
swaption_vol_12m = [.4770,.6556,.7064,.7154]

maturity =  [2,5,10,30]
col_list = ['tsy_fut_opt_vol_3m', 'swaption_vol_3m', 'swaption_vol_12m', 'maturity']
index_list = ['2y', '5y', '10y', '30y']

rates_vol_df = pd.DataFrame(list(zip(tsy_fut_opt_vol_3m, swaption_vol_3m,swaption_vol_12m,maturity)),
                            columns=col_list,
                            index=index_list)

rates_vol_df['swap_vol_ratio'] = rates_vol_df['swaption_vol_12m'] / rates_vol_df['swaption_vol_3m']
rates_vol_df['pros_rate_vol'] = rates_vol_df['tsy_fut_opt_vol_3m'] * rates_vol_df['swap_vol_ratio']


x = list(rates_vol_df['maturity'])
y = list(rates_vol_df['pros_rate_vol'])
y_interp = scipy.interpolate.interp1d(x, y)

def compute_vol_of_rate(asset, fi_df, rates_vol_df):
    x = list(rates_vol_df['maturity'])
    y = list(rates_vol_df['pros_rate_vol'])
    y_interp = scipy.interpolate.interp1d(x, y)
    return y_interp(fi_df['duration'][asset])/1000

def compute_vol_of_spread(asset, fi_df):
    return fi_df['curr_vol'][asset]
    
def get_rsa_vol(asset,rsa_df):
    return rsa_df['pros_vol'][asset]


vol_df =pd.read_excel('inputs.xlsx', sheet_name='data')

vol_assump = []
for ind in vol_df.index:
    asset = vol_df['Factor Description'][ind]
    risk_unit = vol_df['Risk Unit'][ind]
    if vol_df['Fundamental Factor Group'][ind] == 'Liability':
        if risk_unit == 'Vol of Rate':
            vol_assump.append(compute_vol_of_rate('Liability', fi_df, rates_vol_df))
        elif risk_unit == 'Vol of Spread':
            vol_assump.append(compute_vol_of_spread('Liability', fi_df))
    elif vol_df['Fundamental Factor Group'][ind] == 'Cash':
        vol_assump.append(0.000001)
    else:
        if risk_unit == 'Vol of Rate':
            vol_assump.append(compute_vol_of_rate(asset, fi_df, rates_vol_df))
        elif risk_unit == 'Vol of Spread':
            vol_assump.append(compute_vol_of_spread(asset, fi_df))
        else:
            vol_assump.append(get_rsa_vol(asset, rsa_df))


vol_df['vol_assump'] = vol_assump

vol_df = vol_df[['Fundamental Factor Group', 'Factor Description', 'Risk Unit','vol_assump',
                'Liability', '15+ STRIPS', 'Long Corporate', 'Int Corporate',
                'Ultra 30-Year UST Futures', 'S&P 500', 'Russell 2000', 'MSCI EAFE',
                'MSCI Emerging Markets', 'MSCI ACWI', 'Private Equity',
                'Dow Jones REIT', 'Barclays HY', 'Global HF', 'GS Commodity', 'Cash'
                ]]

vol = np.array(vol_assump)[:,np.newaxis]
vol_1 =   vol_df['vol_assump'].to_numpy()[:,np.newaxis]  

corr_df = pd.read_excel('inputs.xlsx', sheet_name='corr', index_col=0).to_numpy()

cov = (vol @ vol.T)*corr_df

asset_list = ['Liability','15+ STRIPS','Long Corporate','Int Corporate','Ultra 30-Year UST Futures',
              'S&P 500','Russell 2000','MSCI EAFE','MSCI Emerging Markets','MSCI ACWI',
              'Private Equity','Dow Jones REIT','Barclays HY','Global HF','GS Commodity','Cash']

    

for asset in fi_index_list:
    if asset == '15+ STRIPS':
        vol_df.loc[(vol_df['Factor Description'] == asset) & (vol_df['Fundamental Factor Group'] == 'FI Rates'), asset] = fi_df['duration'][asset]  
        vol_df.loc[(vol_df['Factor Description'] != asset) | (vol_df['Fundamental Factor Group'] != 'FI Rates'), asset] = 0
    else:
        vol_df[asset]=vol_df['Factor Description'].apply(lambda x: fi_df['duration'][asset] 
                                                                 if x.startswith(asset) else 0)        

new_list = rsa_index_list + ['Cash']
for asset in new_list:
    vol_df[asset] = vol_df['Factor Description'].apply(lambda x: 1 if x == asset else 0)

from numpy.linalg import multi_dot

x = vol_df['Liability'].to_numpy()[:,np.newaxis]
np.sqrt(multi_dot([x.T,cov,x]))
vol_list=[]
for asset in vol_df.columns[4:]:
    x = vol_df[asset].to_numpy()[:,np.newaxis]
    asset_vol = np.sqrt(multi_dot([x.T,cov,x]))
    vol_list.append(asset_vol[0][0])

vol_arr = np.array(vol_list)[:,np.newaxis]

a = vol_arr*vol_arr.T

b = vol_df.iloc[:,4:].to_numpy()

c  = multi_dot([b.T,cov,b])

d = c/a    
corr_output = pd.DataFrame(d, columns=vol_df.columns[4:], index=vol_df.columns[4:])

def compute_fi_return(tsy_ret, oas, oas_ratio, prem, duration):
    return tsy_ret+oas*oas_ratio+prem*duration

def compute_rsa_return(beta, tsy_ret=tsy_mkt_ret, prem=mkt_risk_prem):
    return tsy_ret+prem*beta

def compute_beta(asset, vol_ret_df, corr_output, mkt = 'S&P 500'):
    return (vol_ret_df['Vol'][asset]/vol_ret_df['Vol'][mkt]) * corr_output[mkt][asset]

vol_ret_df = pd.DataFrame(vol_list, columns=['Vol'], index=corr_output.columns)
ret_assump=[]
for asset in fi_index_list:
    oas = fi_df['oas'][asset]
    duration = fi_df['duration'][asset]
    oas_ratio = 1 if asset == 'Liability' else bond_oas_cr
    ret_assump.append(compute_fi_return(tsy_mkt_ret, oas, oas_ratio, fi_term_prem, duration))

for asset in new_list:
    
    beta = compute_beta(asset, vol_ret_df, corr_output)
    ret_assump.append(compute_rsa_return(beta))

vol_ret_df['Return'] = ret_assump

mkt_factor_premium = {'Ultra 30-Year UST Futures':-vol_ret_df['Return']['Cash'],
                      'MSCI Emerging Markets':.015,
                      'Global HF': .02}

acwi_weights = {'S&P 500':0.615206,
                'MSCI EAFE':0.265242,
                'MSCI Emerging Markets':0.119552}

for key in mkt_factor_premium:
        vol_ret_df['Return'][key] += mkt_factor_premium[key]
        
new_acwi_ret = 0
for key in acwi_weights:
    new_acwi_ret += vol_ret_df['Return'][key]*acwi_weights[key]

vol_ret_df['Return']['MSCI ACWI'] = new_acwi_ret

duration_dict = {}
for asset in fi_df.index:
    duration_dict[asset] = fi_df['duration'][asset]
def get_vol_type_dict(vol_df, vol_type):
    vol_dict = {}
    for ind in vol_df.index:
        asset = vol_df['Factor Description'][ind]
        risk_unit = vol_df['Risk Unit'][ind]
        if vol_df['Fundamental Factor Group'][ind] == 'Liability':
            if risk_unit == vol_type:
                vol_dict['Liability'] = vol_df['vol_assump'][ind]
        elif risk_unit == vol_type:
            vol_dict[asset] = vol_df['vol_assump'][ind]
    return vol_dict
vol_of_rate_dict = get_vol_type_dict(vol_df, 'Vol of Rate')
vol_of_spread_dict = get_vol_type_dict(vol_df, 'Vol of Spread')
vol_of_ret_dict = get_vol_type_dict(vol_df, 'Vol of Return')
      
fsv_df = vol_ret_df.copy()

title_dict = {'OAS':duration_dict, 'Spread Vol':vol_of_spread_dict,
              'Rate Vol':vol_of_rate_dict, 'Equity Vol':vol_of_ret_dict}

def create_new_col(col_name, data_dict, df):
    temp_list
    for asset in df.index:
        if asset in list(data_dict.keys()):
            temp_list.append(data_dict[asset])
        else:
            temp_list.append(0)
    df[col_name] = temp_list
    return df            

for key in title_dict:
    temp_list=[]
    for asset in fsv_df.index:
        if asset in list(title_dict[key].keys()):
            temp_list.append(title_dict[key][asset])
        else:
            temp_list.append(0)
    fsv_df[key] = temp_list
    
weights = [-1,.14,.14,0,0,0,0,0,0,.43,.08,.05,.05,.09,0,.02]
fs_loadings = [1]+ [1.02] *15
fsv_df['Weights'] = [-1]+weights        
fsv_df['FS Loading'] = fs_loadings
def compute_oad