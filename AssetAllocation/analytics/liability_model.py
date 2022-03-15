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
    
    #TODO: take out disc_rates option
    def __init__(self, pbo_cashflows, disc_factors, sc_cashflows, contrb_pct, asset_mv, liab_mv_cfs, asset_returns,liab_curve=pd.DataFrame,disc_rates=pd.DataFrame):
        """
        

        Parameters
        ----------
        pbo_cashflows : array
            DESCRIPTION.
        disc_factors : array
            DESCRIPTION.
        sc_cashflows : array
            DESCRIPTION.
        contrb_pct : double
            DESCRIPTION.
        asset_mv : Dataframe
            DESCRIPTION.
        liab_mv_cfs : Dataframe
            DESCRIPTION.
        asset_returns : Dataframe
            DESCRIPTION.
        liab_curve : Dataframe, optional
            DESCRIPTION. The default is pd.DataFrame.
        disc_rates : Dataframe, optional
            DESCRIPTION. The default is pd.DataFrame.

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
        self.liab_mv_cfs = liab_mv_cfs
        self.asset_returns = asset_returns
        self.present_values = self.compute_pvs()
        self.irr_df = self.compute_irr()
        self.liab_mv = self.get_plan_liab_mv()
        self.returns_ts = self.compute_liab_ret()
        self.funded_status = self.compute_funded_status()
        self.fulfill_irr = None
        self.excess_return = None
        self.ret = self.get_return()
        self.data_dict = self.get_liab_data_dict(pbo_cashflows, sc_cashflows)
                
        
    def get_liab_data_dict(self,pbo_cashflows, sc_cashflows):
        """
        return liability data in a dict

        Parameters
        ----------
        pbo_cashflows : array
            DESCRIPTION.
        sc_cashflows : array
            DESCRIPTION.

        Returns
        -------
        dict containing cashflows, liability present values, liability returns,
            irr, asset returns, asset market values
            DESCRIPTION.

        """
        #cash flow dictionary
        cf_frame = {'Total Cashflows': (self.contrb_pct*sc_cashflows + pbo_cashflows),
                    'PBO Cashflows':pbo_cashflows, 'SC Cashflows': sc_cashflows}
        #cash flow dataframe
        cf_df = pd.DataFrame(cf_frame, index=pbo_cashflows.index)
        
        #liability returns
        ret_df = self.returns_ts.copy()
        ret_df.columns = ['Return']
        

        return {'Cashflows': cf_df, 'Present Values': self.present_values, 'Liability Returns': ret_df,
                'Liability Market Values':self.liab_mv, 'IRR': self.irr_df, 'Asset Returns': self.asset_returns, 
                'Asset Market Values': self.asset_mv}

    #TODO: Take out disc rates option
    def compute_pvs(self):
        """
        Return dataframe containing monthly present values of cashflows

        """
        if self.disc_rates.empty:
            pv_dict={}
            #compute present values of monthly cash flows using liab curve data
            for col in self.liab_curve.columns:
                temp_pv = 0
                for j in range (0,len(self.total_cashflows)):
                    temp_pv += (self.total_cashflows[j]/((1+self.liab_curve[col][j]/100)**self.disc_factors[j]))
                pv_dict[col] = temp_pv
            return pd.DataFrame(pv_dict, index = ['Present Value']).transpose()
        else:
            #compute present values of monthly cash flows using irr (disc_rates)
            disc_rates_pv_array = np.zeros(len(self.disc_rates))
            for i in range(len(self.disc_rates)):
                for j in range (0,len(self.total_cashflows)):
                    disc_rates_pv_array[i] += (self.total_cashflows[j]/((1+self.disc_rates['IRR'][i])**self.disc_factors[j]))
                
            return pd.DataFrame(disc_rates_pv_array, columns=['Present Value'], index=self.disc_rates.index)
    
    #TODO: Take out disc rates option
    def compute_liab_ret(self):
        """
        Return dataframe containing liability returns

        """
        liab_ret = np.zeros(len(self.present_values))

        # if self.disc_rates.empty:
        for i in range (0,len(self.present_values)-1):
            #compute liab pv return and add iff
            liab_ret[i+1] += self.irr_df['IRR'][i]/12 + ((self.present_values['Present Value'][i+1])/self.present_values['Present Value'][i])-1
        # else:
        #     for i in range (0,len(self.present_values)-1):
        #         liab_ret[i+1] += self.disc_rates['IRR'][i]/12 + ((self.present_values['Present Value'][i+1])/self.present_values['Present Value'][i])-1
            
        return pd.DataFrame(liab_ret, columns=['Liability'], index=self.present_values.index)
    
    #TODO:remove method
    def compute_disc_rates_pvs(self):
        disc_rates_pv_list = np.zeros(len(self.disc_rates))
        for i in range(len(self.disc_rates)):
            for j in range (0,len(self.total_cashflows)):
                disc_rates_pv_list[i] += (self.total_cashflows[j]/((1+self.disc_rates['IRR'][i])**self.disc_factors[j]))
            
        return pd.DataFrame(disc_rates_pv_list, columns=['Present Value'], index=self.disc_rates.index)
    
    def npv(self,irr, cfs, yrs):
        """
        Returns net present value of cashflows given an irr

        Parameters
        ----------
        irr : double
            IRR.
        cfs : array
            cashflows.
        yrs : array
            periods.

        Returns
        -------
        double

        """
        return np.sum(cfs / (1. + irr) ** yrs)
    
   
    def irr(self,cfs, yrs, x0, **kwargs):
        """
        Compute internal rate of return(IRR)

        Parameters
        ----------
        cfs : array
            cashflows.
        yrs : array
            periods.
        x0 : double
            guess.
        
        Returns
        -------
        double
            IRR.

        """
        return np.asscalar(fsolve(self.npv, x0=x0,args=(cfs,yrs), **kwargs))
    
    def compute_irr(self):
        """
        Returns a dataframe containing IRR data for a give time period

        """
        irr_array = np.zeros(len(self.present_values))
        for j in range (len(self.present_values)):
            cashflows = np.append(np.negative(self.present_values['Present Value'][j]),self.total_cashflows)
            yrs = np.append(0, self.disc_factors)
            irr_array[j] += self.irr(cashflows, yrs, .03)
        return pd.DataFrame(irr_array, columns=['IRR'], index=self.present_values.index)
    
    def compute_fulfill_ret(self, yrs_to_ff, ff_ratio,x0=.01):
        self.fulfill_irr = np.asscalar(fsolve(self.fulfill_solve, x0=x0,
                                  args=(yrs_to_ff, ff_ratio)))
        self.excess_return = self.fulfill_irr - self.ret

    def fulfill_solve(self,fulfill_return, yrs_to_ff, ff_ratio):
        erf_pvs_list = np.zeros(len(self.disc_factors))
        asset_mv_list = np.zeros(len(self.disc_factors))
        x = yrs_to_ff/self.disc_factors[0]
        x = x.astype(int)
        
        for j in range(len(self.disc_factors)):
            if (j == 0):
                for i in range(j,len(self.total_cashflows)):
                    erf_pvs_list[j] += (self.total_cashflows[i]/((1+self.irr_df['IRR'][-1])**self.disc_factors[i-j]))
                    asset_mv_list[j] = self.asset_mv.iloc[-1:]['Market Value'][0]
        
            else:
                for i in range(j,len(self.total_cashflows)):
                    erf_pvs_list[j] += (self.total_cashflows[i]/((1+self.irr_df['IRR'][-1])**self.disc_factors[i-j]))
                    asset_mv_list[j] = (asset_mv_list[j-1]*(1+fulfill_return)**self.disc_factors[0].tolist())-self.total_cashflows[j-1]
         
        return asset_mv_list[x] - erf_pvs_list[x]*ff_ratio
    
    #TODO: revisit this method
    def compute_funded_status(self):
        return self.asset_mv.iloc[-1:]/self.present_values.iloc[-1:]['Present Value'][0]
    
    #TODOD: revisit this method
    def get_return(self):
        return self.irr_df['IRR'][-1]

            
    def get_plan_liab_mv(self):
        #get yrs
        yrs = list(range(1,len(self.liab_mv_cfs)+1))
        pbo = []
        #puts pbo for each time period into a list
        for i in list(range(0,len(self.liab_mv_cfs.columns))):
            cfs = list(self.liab_mv_cfs.iloc[:,i])
            pbo.append(self.npv( self.irr_df['IRR'][self.liab_mv_cfs.columns[i]]/12, cfs, yrs))
        return pd.DataFrame(pbo, index = self.liab_mv_cfs.columns, columns = self.asset_mv.columns)

