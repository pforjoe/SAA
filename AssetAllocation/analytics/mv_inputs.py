# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 14:18:54 2021

@author: Powis Forjoe
"""

import numpy as np
import pandas as pd
from numpy.linalg import multi_dot
from .util import add_dimension

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Import optimization module from scipy
import scipy.interpolate


# ACWI_WEIGHTS_DICT = {'S&P 500':0.615206,
#                     'MSCI EAFE':0.265242,
#                     'MSCI Emerging Markets':0.119552}

class mv_inputs():
    
    def __init__(self, ret_assump, mkt_factor_prem,fi_data,rsa_data,rv_data,vol_defs,corr,weights,illiquidity):
        """
        

        Parameters
        ----------
        ret_assump : TYPE
            DESCRIPTION.
        mkt_factor_prem : TYPE
            DESCRIPTION.
        fi_data : TYPE
            DESCRIPTION.
        rsa_data : TYPE
            DESCRIPTION.
        rv_data : TYPE
            DESCRIPTION.
        vol_defs : TYPE
            DESCRIPTION.
        corr : TYPE
            DESCRIPTION.
        weights : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.ret_assump = ret_assump
        self.illiquidity = illiquidity
        self.mkt_factor_prem = mkt_factor_prem
        self.fi_data = fi_data
        self.rsa_data = rsa_data
        self.rv_data = rv_data
        self.vol_defs = vol_defs
        self.corr = corr
        self.weights = weights
        self.symbols = list(self.weights.index)
        self.factor_wgts = self.create_factor_wgts()
        self.vol_assump = self.compute_vol_assump()
        self.cov = self.compute_cov()
        self.plan_vols = self.compute_plan_vols()
        self.plan_corr = self.compute_plan_corr()
        
    
    def create_factor_wgts(self):
        """
        

        Returns
        -------
        factor_wgts : TYPE
            DESCRIPTION.

        """
        #Get vol definitions
        factor_wgts = self.vol_defs.copy()
        
        #Add Factor Weightings
        for asset in self.fi_data.index:
            if asset == '15+ STRIPS':
                factor_wgts.loc[(factor_wgts['Factor Description'] == asset) & (factor_wgts['Fundamental Factor Group'] == 'FI Rates'), asset] = self.fi_data['Duration'][asset]  
                factor_wgts.loc[(factor_wgts['Factor Description'] != asset) | (factor_wgts['Fundamental Factor Group'] != 'FI Rates'), asset] = 0
            else:
                factor_wgts[asset] = factor_wgts['Factor Description'].apply(lambda x: self.fi_data['Duration'][asset] 
                                                                         if x.startswith(asset) else 0)        
        
        for asset in self.symbols[len(self.fi_data.index):]:
            factor_wgts[asset] = factor_wgts['Factor Description'].apply(lambda x: 1 if x == asset else 0)
        
        return factor_wgts

    def compute_vol_of_rate(self,asset):
        """
        

        Parameters
        ----------
        asset : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        x = list(self.rv_data['Maturity'])
        y = list(self.rv_data['Pros Rate Vol'])
        y_interp = scipy.interpolate.interp1d(x, y)
        return y_interp(self.fi_data['Duration'][asset])/10000
    
    def compute_vol_of_spread(self, asset):
        """
        

        Parameters
        ----------
        asset : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return self.fi_data['Current Vol'][asset]/10000
        
    def get_rsa_vol(self, asset):
        """
        

        Parameters
        ----------
        asset : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return self.rsa_data['Prospective Vol'][asset]
    
    def compute_vol_assump(self):
        """
        

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        vol_assump = []
        for ind in self.factor_wgts.index:
            asset = self.factor_wgts['Factor Description'][ind]
            risk_unit = self.factor_wgts['Risk Unit'][ind]
            if self.factor_wgts['Fundamental Factor Group'][ind] == 'Liability':
                if risk_unit == 'Vol of Rate':
                    vol_assump.append(self.compute_vol_of_rate('Liability'))
                elif risk_unit == 'Vol of Spread':
                    vol_assump.append(self.compute_vol_of_spread('Liability'))
            elif self.factor_wgts['Fundamental Factor Group'][ind] == 'Cash':
                vol_assump.append(0.000001)
            else:
                if risk_unit == 'Vol of Rate':
                    vol_assump.append(self.compute_vol_of_rate(asset))
                elif risk_unit == 'Vol of Spread':
                    vol_assump.append(self.compute_vol_of_spread(asset))
                else:
                    vol_assump.append(self.get_rsa_vol(asset))
        return np.array(vol_assump)[:,np.newaxis]
    
    def compute_cov(self):
        """
        

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return (self.vol_assump @ self.vol_assump.T)*(self.corr.to_numpy())
    
    def compute_plan_vols(self):
        """
        

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        vol_list=[]
        for asset in self.symbols:
            x = add_dimension(self.factor_wgts[asset])
            asset_vol = np.sqrt(multi_dot([x.T,self.cov,x]))
            vol_list.append(asset_vol[0][0])
        return pd.DataFrame(vol_list, index=self.symbols, columns=['Volatility'])
    
    def compute_plan_corr(self):
        """
        

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        vol_array = add_dimension(self.plan_vols['Volatility'])
        comp_weights = self.factor_wgts.iloc[:,3:].to_numpy()
        corr_array = multi_dot([comp_weights.T,self.cov,comp_weights]) / (vol_array*vol_array.T)
        return pd.DataFrame(corr_array, columns=self.symbols, index=self.symbols)
    
    def compute_plan_wgts(self):
        """
        

        Returns
        -------
        policy_wgts : TYPE
            DESCRIPTION.

        """
        policy_wgts = self.weights.copy()
        policy_wgts['FS AdjWeights'] = self.weights['Weights'] * self.weights['Factor Loadings']
    
        return policy_wgts
        
    def compute_fi_return(self,oas, oas_ratio, duration):
        """
        

        Parameters
        ----------
        oas : TYPE
            DESCRIPTION.
        oas_ratio : TYPE
            DESCRIPTION.
        duration : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return self.ret_assump['TSY_MKT_RET'] + oas*oas_ratio + self.ret_assump['FI_TERM_PREM']*duration
    
    def compute_rsa_return(self,beta):
        """
        

        Parameters
        ----------
        beta : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return self.ret_assump['TSY_MKT_RET'] + self.ret_assump['MKT_RISK_PREM']*beta

    def compute_beta(self, asset, mkt='S&P 500'):
        """
        

        Parameters
        ----------
        asset : TYPE
            DESCRIPTION.
        mkt : TYPE, optional
            DESCRIPTION. The default is MKT.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return (self.plan_vols['Volatility'][asset]/self.plan_vols['Volatility'][mkt]) * self.plan_corr[mkt][asset]
    
    def compute_plan_return(self, mkt='S&P 500'):
        """
        

        Returns
        -------
        ret_df : TYPE
            DESCRIPTION.

        """
        ret_list=[]
        for asset in self.fi_data.index:
            oas = self.fi_data['Spread'][asset]
            duration = self.fi_data['Duration'][asset]
            oas_ratio = 1 if asset == 'Liability' else self.ret_assump['BOND_OAS_CR']
            ret_list.append(self.compute_fi_return(oas, oas_ratio, duration))
        
        for asset in self.symbols[len(self.fi_data.index):]:
            beta = self.compute_beta(asset,mkt)
            ret_list.append(self.compute_rsa_return(beta))
        
        ret_df = pd.DataFrame(ret_list, columns=['Return'], index=self.symbols)
        
        for key in self.mkt_factor_prem:
            ret_df['Return'][key] += (self.mkt_factor_prem[key] + self.illiquidity[key])
            
        # new_acwi_ret = 0
        # for key in ACWI_WEIGHTS_DICT:
        #     new_acwi_ret += ret_df['Return'][key]*ACWI_WEIGHTS_DICT[key]
        
        # ret_df['Return']['MSCI ACWI'] = new_acwi_ret
        return ret_df
    
    def get_output(self, mkt='Equity'):
        """
        

        Returns
        -------
        dict
            DESCRIPTION.

        """
        return {'Return':self.compute_plan_return(mkt),
                'Volatility':self.plan_vols,
                'weights': self.compute_plan_wgts(),
                'corr':self.plan_corr}