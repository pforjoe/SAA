  # -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 12:27:19 2021

@author: Antoine Tan, Powis Forjoe
"""
from .plan_params import planParams
from ..datamanager import datamanager as dm
from .corr_matrix_sampler import CorrelationMatrixSampler
import numpy as np
import pandas as pd

class stochMV():

    def __init__(self, init_plan, iter:int):

        self.init_plan = init_plan
        self.iter = iter
        self.simulated_plans = []
        self.returns_df = None
        self.avg_weights = None
        self.opt_ports_df = None
        self.resamp_corr_dict = {}
        self.ef_portfolio_dict = {}
        
    def generate_plans(self, nb_period=5):

        #Sample the covariance matrices
        correl = CorrelationMatrixSampler(self.init_plan.ret_df, 0)
        sample_corr = correl.randomly_sample_correlation_matrices(self.iter)

        #Convert the correlation matrices to covariance matrices
        sample_cov=[]
        for corr in sample_corr:
            sample_cov.append((self.init_plan.vol @ self.init_plan.vol.T)*corr.to_numpy())

        #Sample the returns based on the mean return, volatilities, and sampled correlation matrix
        np.random.seed(0)
        df = pd.DataFrame(columns=self.init_plan.symbols)
        for i in range(0, self.iter):
            #draw nb_period multivariate gaussian sample
            rd = np.random.multivariate_normal(self.init_plan.ret.to_numpy(), sample_cov[i], nb_period)

            #compound returns over the number of period
            nav = np.ones(len(self.init_plan))
            for j in range(0, nb_period):
                nav = nav * np.exp(rd[j, :])
            returns = pd.Series(np.power(nav, 1/nb_period) - 1, index=self.init_plan.symbols)

            plan = planParams(self.init_plan.policy_wgts, returns, self.init_plan.vol,
                                  sample_corr[i], self.init_plan.symbols, self.init_plan.funded_status)

            #add the simulated plan to the list of plans and add the return vector to the return dataframe
            self.simulated_plans.append(plan)
            df = pd.concat([df, returns.to_frame().T], ignore_index=True)
            df.index.name = 'Sample'
        self.returns_df = df

    def generate_efficient_frontiers(self, bnds, cons, num_ports=20):

        #Compute the efficient frontier on the initial plan
        self.init_plan.compute_eff_frontier(bnds, cons, num_ports)
        
        avg_weights = np.zeros((len(self.init_plan.eff_frontier_tweights), len(self.init_plan)))
        #For each simulated plan compute the efficient frontier
        
        sample = 0
        for plan in self.simulated_plans:
            plan.compute_eff_frontier(bnds, cons,num_ports)
            avg_weights = avg_weights + plan.eff_frontier_tweights
            self.ef_portfolio_dict[sample] = dm.format_ports_df(dm.get_ports_df(plan.eff_frontier_trets, plan.eff_frontier_tvols, 
                                                                                plan.eff_frontier_assetvols, plan.eff_frontier_tweights,plan.symbols), plan.ret)
            sample += 1
        #Average of the weights across the simulated plans
        self.avg_weights = avg_weights/self.iter

        #create the dataframe that contains the averaged efficient frontier data
        i = 0
        ret = np.array([])
        vol = np.array([])
        asset_vols = np.array([])
        for wgts in self.avg_weights:
            ret = np.append(ret, self.init_plan.portfolio_stats(self.avg_weights[i, :])[0])
            vol = np.append(vol, self.init_plan.portfolio_stats(self.avg_weights[i, :])[1])
            asset_vols = np.append(asset_vols, self.init_plan.portfolio_stats(self.init_plan.get_asset_vol(self.avg_weights[i, :]))[1])
            i = i+1

        self.opt_ports_df = dm.get_ports_df(ret, vol, asset_vols,self.avg_weights,self.init_plan.symbols)
        self.opt_ports_df = dm.format_ports_df(self.opt_ports_df,self.init_plan.ret)
        
    def generate_resamp_corr_dict(self):
        
        asset_liab_list = list(self.init_plan.symbols)
        asset_liab_list.remove('Cash')
        
        for asset_liab in asset_liab_list:
            col_list = list(asset_liab_list)
            col_list.remove(asset_liab)
            
            resamp_corr_df = pd.DataFrame(columns=col_list, index=list(range(0,self.iter)))
            resamp_corr_df.index.name = 'Sample'
            
            
            for col in col_list:
                for ind in resamp_corr_df.index:
                    resamp_corr_df[col][ind] = self.simulated_plans[ind].corr[asset_liab][col]
            
            self.resamp_corr_dict[asset_liab] = resamp_corr_df