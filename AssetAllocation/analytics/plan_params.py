# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 22:05:54 2021

@authors: Roxton McNeal, Matt Johnston, Powis Forjoe
"""

import numpy as np
#import pandas as pd
from numpy.linalg import multi_dot
from util import add_dimension
from AssetAllocation.datamanger import datamanger as dm
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Import optimization module from scipy
import scipy.optimize as sco

class plan_params():
    
    def __init__(self, policy_wgts,ret,vol,corr,symbols):
        """
        

        Parameters
        ----------
        policy_wgts : TYPE
            DESCRIPTION.
        ret : TYPE
            DESCRIPTION.
        vol : TYPE
            DESCRIPTION.
        corr : TYPE
            DESCRIPTION.
        symbols : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.policy_wgts = policy_wgts
        self.ret = ret
        self.vol = vol
        self.corr = corr
        self.symbols = symbols
        self.policy_rets = self.compute_policy_return()
        self.cov = self.compute_cov()
        self.var = self.compute_var()
        self.fsv = self.compute_fsv()
        self.eff_frontier_tvols = None
        self.eff_frontier_trets = None
        self.eff_frontier_tweights = None
        self.ports_df = None
    
    def compute_policy_return(self):
        """
        

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return self.policy_wgts.T @ add_dimension(self.ret)
    
    def compute_cov(self):
        """
        

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return (self.vol @ self.vol.T)*self.corr
    
    def compute_var(self):
        """
        

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return multi_dot([self.policy_wgts.T, 
                          self.cov, 
                          self.policy_wgts])
    
    def compute_fsv(self):
        """
        

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return np.sqrt(self.var)
    
    def __len__(self):
        """
        

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return len(self.policy_wgts)
   
    def portfolio_stats(self,weights):
        """
        

        Parameters
        ----------
        weights : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        port_wgts = np.array(weights)[:,np.newaxis]
        port_rets = port_wgts.T @ self.ret[:,np.newaxis]
        port_vols = np.sqrt(multi_dot([port_wgts.T, self.cov, port_wgts])) 
        port_vars = port_vols**2 
        return np.array([port_rets, port_vols, port_rets/port_vols, port_vars]).flatten()
    
    # Maximizing sharpe ratio
    def min_sharpe_ratio(self,weights):
        """
        

        Parameters
        ----------
        weights : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return -self.portfolio_stats(weights)[2]
    
    # Maximizing sharpe ratio
    def min_ret(self,weights):
        """
        

        Parameters
        ----------
        weights : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return -self.portfolio_stats(weights)[0]
    
    # Minimizing variance
    def min_variance(self,weights):
        """
        

        Parameters
        ----------
        weights : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return self.portfolio_stats(weights)[3]
    
    # Minimize the volatility
    def min_volatility(self,weights):
        """
        

        Parameters
        ----------
        weights : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return self.portfolio_stats(weights)[1]
    
    def optimize(self, fun, bnds, cons, method='SLSQP'):
        """
        

        Parameters
        ----------
        fun : function
            DESCRIPTION.
        bnds : TYPE
            DESCRIPTION.
        cons : TYPE
            DESCRIPTION.
        method : TYPE, optional
            DESCRIPTION. The default is 'SLSQP'.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return sco.minimize(fun, self.policy_wgts, method=method, bounds=bnds, constraints=cons)


    def compute_eff_frontier(self, bnds, cons,num_ports=100):
        #initial minimimum and maximum returns to define the bounds of the efficient frontier
        opt_var = self.optimize(self.min_variance, bnds, cons)
        min_ret = self.portfolio_stats(opt_var['x'])[0]
        opt_ret = self.optimize(self.min_ret, bnds, cons)
        max_ret = np.around(self.portfolio_stats(opt_ret['x']), 4)[0]

        #Getting data for efficient frontier portfolios
        self.eff_frontier_trets = np.linspace(min_ret, max_ret, num_ports)
        t_vols = []
        t_weights = []

        for tr in self.eff_frontier_trets:

            #adding return constraints to optimizer constraints
            ef_cons = ()
            ef_cons = ef_cons + cons
            ef_cons = ef_cons + ({'type': 'eq', 'fun': lambda x: self.portfolio_stats(x)[0] - tr},)
            #run optimization
            opt_ef = self.optimize(self.min_volatility, bnds, ef_cons)
            #store result
            t_vols.append(opt_ef['fun'])
            t_weights.append(opt_ef['x'])

        self.eff_frontier_tvols = np.array(t_vols)
        self.eff_frontier_tweights = np.array(t_weights)
        self.ports_df = dm.get_ports_df(self.eff_frontier_trets, self.eff_frontier_tvols, self.eff_frontier_tweights,
                                        self.symbols, raw=False)
        return None



    def run_mc_simulation(self, num_ports=5000):
        # Initialize the lists
        rets = []; vols = []; wts = []
        num_of_assets = len(self)
        sum_of_weights = np.sum(self.policy_wgts[0:num_of_assets])
        
        # Simulate 5,000 portfolios
        for i in range (num_ports):
            
            # Generate random weights
            weights = np.random.random(num_of_assets-1)
            
            # Set weights such that sum of weights equals 1.02
            weights /= sum(weights)*(1/sum_of_weights)
            
            # Add the constant Liability
            weights = np.insert(weights,0,-1)[:,np.newaxis]
            
            # Portfolio statistics
            rets.append(weights.T @ add_dimension(self.ret))        
            vols.append(np.sqrt(multi_dot([weights.T, self.cov, weights])))
            wts.append(weights.flatten())
        
        # Record values     
        port_rets = np.array(rets).flatten()
        port_vols = np.array(vols).flatten()
        port_wts = np.array(wts)
        
        # Create a dataframe for analysis
        return {'returns': port_rets,
                'volatility': port_vols,
                'sharpe_ratio': port_rets/port_vols,
                'weights': list(port_wts)}



