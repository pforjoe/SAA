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


class liability_model():
    
    def __init__(self, pbo_cashflows, disc_factors, sc_cashflows, liab_curve,disc_rates,contrb_pct):
        """
        

        Parameters
        ----------
        pbo_cashflows : TYPE
            DESCRIPTION.
        disc_factors : TYPE
            DESCRIPTION.
        sc_cashflows : TYPE
            DESCRIPTION.
        liab_curve : TYPE
            DESCRIPTION.
        disc_rates : TYPE
            DESCRIPTION.
        contrb_pct : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.pbo_cashflows = pbo_cashflows
        self.disc_factors = disc_factors
        self.contrb_pct = contrb_pct
        self.sc_cashflows = sc_cashflows
        self.total_cashflows = self.contrb_pct*self.sc_cashflows + self.pbo_cashflows
        self.liab_curve = liab_curve
        self.disc_rates = disc_rates
        self.present_values = self.compute_pvs()
        self.returns = self.compute_liab_ret()
        self.disc_rates_pvs = self.compute_disc_rates_pvs()
    
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
        for i in range (0,len(self.present_values)-1):
            liab_ret[i+1] = ((self.present_values['Present Value'][i+1])/self.present_values['Present Value'][i])-1
        
        return pd.DataFrame(liab_ret, columns=['Liability'], index=self.present_values.index)
    
    def compute_disc_rates_pvs(self):
        disc_rates_pv_list = np.zeros(len(self.disc_rates))
        for i in range(len(self.disc_rates)):
            for j in range (0,len(self.total_cashflows)):
                disc_rates_pv_list[i] += (self.total_cashflows[j]/((1+self.disc_rates['IRR'][i])**self.disc_factors[j]))
            
        return pd.DataFrame(disc_rates_pv_list, columns=['Present Value'], index=self.disc_rates.index)
    
    def npv(self,irr, cfs, yrs):  
        return np.sum(cfs / (1. + irr) ** yrs)
    
    def irr(self,cfs, yrs, x0, **kwargs):
        return np.asscalar(fsolve(self.npv, x0=x0,args=(cfs,yrs), **kwargs))
    
    def compute_irr(self):
        irr_list = np.zeros(len(self.disc_rates_pvs))
        for j in range (len(self.disc_rates_pvs)):
            cashflows = np.append(np.negative(self.disc_rates_pvs['Present Value'][j]),self.total_cashflows)
            yrs = np.append(0, self.disc_factors)
            irr_list[j] += self.irr(cashflows, yrs, .03)
        return irr_list
    
    def full_solve(self, initial_mv,yrs_to_ff, ff_ratio,x0):
            return np.asscalar(fsolve(self.fullfill_solve, x0=x0,
                                      args=(initial_mv, yrs_to_ff, ff_ratio)))

    def fullfill_solve(self,fullfill_return, initial_mv, yrs_to_ff, ff_ratio):
        erf_pvs_list = np.zeros(len(self.disc_factors))
        asset_mv_list = np.zeros(len(self.disc_factors))
        x = yrs_to_ff/self.disc_factors[0]
        x = x.astype(int)
        
        for j in range(len(self.disc_factors)):
            if (j == 0):
                for i in range(j,len(self.total_cashflows)):
                    erf_pvs_list[j] += (self.total_cashflows[i]/((1+self.disc_rates['IRR'][-1])**self.disc_factors[i-j]))
                    asset_mv_list[j] = initial_mv
        
            else:
                for i in range(j,len(self.total_cashflows)):
                    erf_pvs_list[j] += (self.total_cashflows[i]/((1+self.disc_rates['IRR'][-1])**self.disc_factors[i-j]))
                    asset_mv_list[j] = (asset_mv_list[j-1]*(1+fullfill_return)**self.disc_factors[0].tolist())-self.total_cashflows[j-1]
         
        return asset_mv_list[x] - erf_pvs_list[x]*ff_ratio