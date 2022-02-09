# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 21:32:09 2021

@author: Powis Forjoe
"""

from scipy.optimize import fsolve
import pandas as pd
import numpy as np
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')


class liabilityModel():
    
    def __init__(self, pbo_cashflows, disc_factors, sc_cashflows, contrb_pct, asset_mv, asset_returns,liab_curve=pd.DataFrame,disc_rates=pd.DataFrame):
        """
        

        Parameters
        ----------
        pbo_cashflows : array
            DESCRIPTION.
        disc_factors : array
            DESCRIPTION.
        sc_cashflows : array
            DESCRIPTION.
        liab_curve : dataframe
            DESCRIPTION.
        disc_rates : dataframe
            DESCRIPTION.
        contrb_pct : float
            DESCRIPTION.
        asset_mv : float
            DESCRIPTION.

        Returns
        -------
        liability model.

        """
        self.pbo_cashflows = np.array(pbo_cashflows)
        self.disc_factors = np.array(disc_factors)
        self.contrb_pct = contrb_pct
        self.sc_cashflows = np.array(sc_cashflows)
        self.total_cashflows = self.contrb_pct*self.sc_cashflows + self.pbo_cashflows
        self.liab_curve = liab_curve
        self.disc_rates = disc_rates
        self.asset_mv = asset_mv
        self.asset_returns = asset_returns
        self.present_values = self.compute_pvs()
        self.disc_rates_pvs = self.compute_disc_rates_pvs()
        self.irr_array = self.compute_irr()
        self.returns_ts = self.compute_liab_ret()
        self.funded_status = self.compute_funded_status()
        self.fulfill_irr = None
        self.excess_return = None
        self.ret = self.disc_rates['IRR'][-1]
        self.data_dict = self.get_liab_model_dict(pbo_cashflows, sc_cashflows)
                
        
    def get_liab_model_dict(self,pbo_cashflows, sc_cashflows):
        cf_frame = {'Total Cashflows': (self.contrb_pct*sc_cashflows + pbo_cashflows),
                    'PBO Cashflows':pbo_cashflows, 'SC Cashflows': sc_cashflows}
        cf_df = pd.DataFrame(cf_frame, index=pbo_cashflows.index)
        ret_df = self.returns_ts.copy()
        ret_df.columns = ['Return']
        return {'Cashflows': cf_df, 'Present Values': self.present_values, 'Liability Returns': ret_df,
                'IRR': self.irr_df, 'Asset Returns': self.asset_returns, 'Market Values': self.asset_mv}

    def compute_pvs(self):
        pv_dict={}
        for col in self.liab_curve.columns:
            temp_pv = 0
            for j in range (0,len(self.total_cashflows)):
                temp_pv += (self.total_cashflows[j]/((1+self.liab_curve[col][j]/100)**self.disc_factors[j]))
            pv_dict[col] = temp_pv
        return pd.DataFrame(pv_dict, index = ['Present Value']).transpose()
    
    def compute_liab_ret(self):
        liab_ret = np.zeros(len(self.present_values))
        irr_ret = self.transform_irr_array()
        for i in range (0,len(self.present_values)-1):
            liab_ret[i+1] += irr_ret[i+1] + ((self.present_values['Present Value'][i+1])/self.present_values['Present Value'][i])-1
        
        return pd.DataFrame(liab_ret, columns=['Liability'], index=self.present_values.index)
    
    def compute_disc_rates_pvs(self):
        disc_rates_pv_array = np.zeros(len(self.disc_rates))
        for i in range(len(self.disc_rates)):
            for j in range (0,len(self.total_cashflows)):
                disc_rates_pv_array[i] += (self.total_cashflows[j]/((1+self.disc_rates['IRR'][i])**self.disc_factors[j]))
            
        return pd.DataFrame(disc_rates_pv_array, columns=['Present Value'], index=self.disc_rates.index)
    
    def npv(self,irr, cfs, yrs):  
        return np.sum(cfs / (1. + irr) ** yrs)
    
    def irr(self,cfs, yrs, x0, **kwargs):
        return np.asscalar(fsolve(self.npv, x0=x0,args=(cfs,yrs), **kwargs))
    
    def compute_irr(self):
        irr_array = np.zeros(len(self.disc_rates_pvs))
        for j in range (len(self.disc_rates_pvs)):
            cashflows = np.append(np.negative(self.disc_rates_pvs['Present Value'][j]),self.total_cashflows)
            yrs = np.append(0, self.disc_factors)
            irr_array[j] += self.irr(cashflows, yrs, .03)
        return irr_array
    
    def compute_fulfill_ret(self, yrs_to_ff, ff_ratio,x0=.01):
            self.fulfill_irr = np.asscalar(fsolve(self.fulfill_solve, x0=x0,
                                      args=(yrs_to_ff, ff_ratio)))
            self.excess_return = self.fulfill_irr - self.ret

    def fulfill_solve(self,fulfill_return, yrs_to_ff, ff_ratio):
        erf_pvs_array = np.zeros(len(self.disc_factors))
        asset_mv_array = np.zeros(len(self.disc_factors))
        x = yrs_to_ff/self.disc_factors[0]
        x = x.astype(int)
        
        for j in range(len(self.disc_factors)):
            if (j == 0):
                for i in range(j,len(self.total_cashflows)):
                    erf_pvs_list[j] += (self.total_cashflows[i]/((1+self.irr_df['IRR'][-1])**self.disc_factors[i-j]))
                    asset_mv_list[j] = self.asset_mv.iloc[-1:]['Market Value'][0]
        
            else:
                for i in range(j,len(self.total_cashflows)):
                    erf_pvs_array[j] += (self.total_cashflows[i]/((1+self.disc_rates['IRR'][-1])**self.disc_factors[i-j]))
                    asset_mv_array[j] = (asset_mv_array[j-1]*(1+fulfill_return)**self.disc_factors[0].tolist())-self.total_cashflows[j-1]
         
        return asset_mv_array[x] - erf_pvs_array[x]*ff_ratio
    
    def compute_funded_status(self):
        return self.asset_mv.iloc[-1:]/self.present_values.iloc[-1:]['Present Value'][0]
    #self.asset_mv.iloc[-1:]['Market Value'][0]/self.present_values.iloc[-1:]['Present Value'][0]
    def get_return(self):
        return self.irr_df['IRR'][-1]

            
