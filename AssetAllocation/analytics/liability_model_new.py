# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 20:47:00 2023

@author: RRQ1FYQ
"""

from AssetAllocation.datamanager import datamanager as dm
from AssetAllocation.analytics import ts_analytics as ts

from scipy.optimize import fsolve
import pandas as pd
import numpy as np
from .ts_analytics import get_ann_vol
from .util import offset_df
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')


class liabilityModelNew():
    
    #TODO: take out disc_rates option
    def __init__(self, sc_accrual, pbo_cashflows, disc_factors, sc_cashflows, pbo_cfs_dict, sc_cfs_dict, contrb_pct, asset_mv, liab_mv_cfs, asset_returns,liab_curve=pd.DataFrame,disc_rates=pd.DataFrame):
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
        self.sc_accrual = sc_accrual
        self.pbo_cashflows = np.array(pbo_cashflows)
        self.disc_factors = np.array(disc_factors)
        self.contrb_pct = contrb_pct
        self.sc_cashflows = np.array(sc_cashflows)
        self.total_cashflows = self.contrb_pct*self.sc_cashflows + self.pbo_cashflows
        
        self.pbo_cfs_dict = pbo_cfs_dict
        self.sc_cfs_dict = sc_cfs_dict
        
        
        self.liab_curve = liab_curve
        self.disc_rates = disc_rates
        self.accrual_factors = [0] + self.disc_factors.tolist()
        #todo: dont hard code
        self.asset_mv = pd.DataFrame({'Market Value':asset_mv})

        self.asset_returns = pd.DataFrame(asset_returns)
        self.qtd_asset_returns = self.compute_asset_ret(freq = '1Q').transpose()
        self.ytd_asset_returns = self.compute_asset_ret(freq = '1Y').transpose()

        self.new_pv_irr = self.get_pv_irr()

        self.pv_new = self.new_pv_irr['Present Value']
        self.present_values = self.concat_data(self.compute_pvs(),self.pv_new)
        
        
        self.irr_df = self.concat_data(self.compute_irr(), self.new_pv_irr['IRR'])
        
        self.returns_ts = self.concat_data(self.compute_liab_ret(),self.compute_liab_ret_new(freq = '1M'))
        self.qtd_liab_returns = self.compute_liab_ret_new(freq = '1Q').transpose()
        self.ytd_liab_returns = self.compute_liab_ret_new(freq = '1Y').transpose()

        self.liab_mv_cfs = pd.DataFrame(liab_mv_cfs)
        self.liab_mv = self.get_plan_liab_mv()
        self.funded_status = self.concat_fs_data(self.compute_funded_status(), self.compute_funded_status_new())
        
        self.funded_status_new = self.compute_funded_status_new()
        self.fulfill_irr = None
        self.excess_return = None
        self.ret = self.get_return()
        self.data_dict = self.get_liab_data_dict(pbo_cashflows, sc_cashflows)
             
        
    def concat_data(self, old_df, ldi_df):
        temp_df = old_df.iloc[:-len(ldi_df)]
        pd.concat([temp_df, ldi_df])
        return pd.concat([temp_df, ldi_df])
    
    def concat_fs_data(self, old_df, ldi_df):
        new_fs_df = pd.DataFrame()
        for col in old_df.columns:
            temp_ldi_df = ldi_df[col].dropna()
            temp_old_df = old_df[col].iloc[:-len(temp_ldi_df)]
            new_fs_df[col] = pd.concat([temp_old_df, temp_ldi_df])
            
        return new_fs_df 
    
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

        return {'Present Values': self.present_values, 'Liability Returns': self.returns_ts,
                'Liability YTD Returns': pd.DataFrame(self.ytd_liab_returns).transpose(),'Liability QTD Returns': pd.DataFrame(self.qtd_liab_returns).transpose(),
                
                'Liability Market Values':self.liab_mv, 'IRR': self.irr_df, 'Asset Returns': self.asset_returns, 
                'Asset YTD Returns': pd.DataFrame(self.ytd_asset_returns).transpose(),'Asset QTD Returns': pd.DataFrame(self.qtd_asset_returns).transpose(),
                'Asset Market Values': self.asset_mv, 'Funded Status': self.funded_status, 
                               
                }
   

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
    
    def compute_funded_status(self):
        #compute funded status: asset/liability
        fs_df = self.asset_mv / self.liab_mv
        fs_df.columns = ['Funded Status']
        
        #compute funded status gap: liability(i.e PBO) - asset
        fs_df['Funded Status Gap'] = self.liab_mv - self.asset_mv   
        fs_df.dropna(inplace=True)
        
        #compute funded status difference between each date
        gap_diff = fs_df['Funded Status Gap'].diff()
           
        #compute funded status gap difference percent: funded status gap/liability
        gap_diff_percent = gap_diff/self.liab_mv['Market Value']
           
        #compute fs vol 
        gap_diff_percent.dropna(inplace=True)
        fs_df['1Y FSV'] = gap_diff_percent.rolling(window = 12).apply(get_ann_vol)
        fs_df['6mo FSV'] = gap_diff_percent.rolling(window = 6).apply(get_ann_vol)
          
        return fs_df
    
    def compute_funded_status_new(self):
        #compute funded status: asset/liability
        df_temp = dm.merge_dfs(self.pv_new, self.asset_mv)
        
        fs_df = df_temp['Market Value']/ df_temp['Present Value']
        fs_df = pd.DataFrame(fs_df)
        fs_df.columns = ['Funded Status']
        
        #compute funded status gap: liability(i.e PBO) - asset
        fs_df['Funded Status Gap'] = df_temp['Present Value'] - df_temp['Market Value'] 
        fs_df.dropna(inplace=True)
        
        #compute funded status difference between each date
        gap_diff = fs_df['Funded Status Gap'].diff()
           
        #compute funded status gap difference percent: funded status gap/liability
        gap_diff_percent = gap_diff/df_temp['Present Value']
           
        #compute fs vol 
        gap_diff_percent.dropna(inplace=True)
        fs_df['1Y FSV'] = gap_diff_percent.rolling(window = 12).apply(get_ann_vol)
        fs_df['6mo FSV'] = gap_diff_percent.rolling(window = 6).apply(get_ann_vol)
          
        return fs_df
    
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

    def get_cf_table(self, cf_df, no_of_cols, accrual = False):
        '''
        

        Parameters
        ----------
        cf_df : Data Frame
            DESCRIPTION.
        no_of_cols : Float
            DESCRIPTION.
        accrual : TYPE, optional
             The default is False.

        Returns
        -------
        cf_offset : TYPE
            DESCRIPTION.

        '''
        #initiate data frames
        cf_table = pd.DataFrame(columns=list(range(0,no_of_cols+1)), index = cf_df.index)

        #loop through year and set pbo cashflows in dataframe
        for col in cf_table.columns:
            if accrual:
                cf_table[col] = cf_df * self.accrual_factors[col]
            else:
                cf_table[col] = cf_df

            #fill zeros for unfilled rows
            cf_table.loc[:col,col] = 0

        cf_offset = offset_df(cf_table)
        cf_offset.columns = list( cf_df.index[0:no_of_cols+1])

        return cf_offset
    
        
        
    def get_pv_irr(self):
        pv_df = pd.DataFrame()
        irr_df = pd.DataFrame()
        pbo_series = pd.Series()
       
        #loop through ech year
        for year in dm.SHEET_LIST_LDI:
            #get number of cols
            no_of_cols = dm.get_no_cols(year)
           
            #get first 12 pbo cf for each year
            pbo_series = pbo_series.append(self.pbo_cfs_dict[year].iloc[:no_of_cols+1])

            #TODO: make cashflow tables outside of liab_model_new and make input into the model
            #old pvs just take first col of monthize cf
            #get total cashflow table
            pbo_table = self.get_cf_table(self.pbo_cfs_dict[year],no_of_cols)
            pbo_table = self.get_cf_table(self.pbo_cfs_dict[year],no_of_cols)
            sc_table = self.get_cf_table(self.sc_cfs_dict[year],no_of_cols, accrual= self.sc_accrual)
            total_cf_table = pbo_table + sc_table
        
            pv = pd.DataFrame()
            temp_irr =  pd.DataFrame()
            cf = pd.DataFrame()
            
            #loop through each column and discount cashdlows
            #TODO: break into two seperate methods
            for col in total_cf_table.columns:
                #TODO: create own method (get_pv_series)

                temp_pv = total_cf_table[col].values/((1+self.liab_curve[col].values/100)**self.disc_factors)
                pv[col] = [temp_pv.sum()]

                #get irr
                #TODO: create IRR into its own method (get_irr_series)
                cf[col] = np.append(-pv[col], total_cf_table[col])
                temp_irr[col] = [self.irr( cf[col], self.accrual_factors, 0.03)]
            
            pv_df = pv_df.append(pv.transpose())
            irr_df = irr_df.append(temp_irr.transpose())
    
        pv_df.columns = ['Present Value']
        irr_df.columns = ['IRR']
        
        return{'Present Value': pv_df, 'IRR': irr_df,'PBO Series': pbo_series}
        

    def compute_liab_ret_new(self,freq = '1M'):
        
        lookback_window = dm.get_lookback_windows(self.pv_new, freq)
        liability_returns = []
        pv_series = self.pv_new['Present Value']
        
        for row in list(range(0,len(self.pv_new))):
            #get roll window for month
            roll = lookback_window[row]
            #get sum of pbos that need to come off
            pbo = sum(self.new_pv_irr['PBO Series'][row-roll:row])
            
            #dont compute return for pv if no value before it
            if row - roll>=0:    
                liability_returns += [(pv_series[row]+pbo)/ pv_series[row-roll]-1]
            else:
                continue
                
        return_df = pd.DataFrame( liability_returns, index = self.pv_new.index[-len(liability_returns):], columns=['Liability'])
            
        return return_df
    
    #TODO: move asset ret to ts analytics
    def compute_asset_ret(self,freq = '1Q'):
        
        lookback_window = dm.get_lookback_windows(self.asset_returns, freq)

       #TODO: take product instead of re computing
        price_df = dm.get_prices_df(self.asset_returns)
        price_series = price_df.stack()
        asset_returns = []
        
        for row in list(range(0,len(price_series))):
            #get roll window for month
            roll = lookback_window[row]
            asset_returns += [(price_series[row])/price_series[row-roll]-1]
                
        return_df = pd.DataFrame( asset_returns, index = price_df.index, columns=['Asset'])
                        
        return return_df
    
