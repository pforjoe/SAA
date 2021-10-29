import math
from AssetAllocation.analytics import plan_params as pp
from AssetAllocation.datamanger import datamanger as dm
from pandas import Series as se
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

    def generate_plans(self,nb_period = 5):

        
        np.random.seed(0)
        df = pd.DataFrame(columns=self.init_plan.symbols)

        for i in range(0, self.iter):
            #draw nb_period multivariate gaussian sample
            rd = np.random.multivariate_normal(self.init_plan.ret.to_numpy(), self.init_plan.cov, nb_period)

            #compound returns over the number of period
            nav = np.ones(len(self.init_plan))
            for j in range(0, nb_period):
                nav = nav * np.exp(rd[j, :])
            returns = pd.Series(np.power(nav, 1/nb_period) - 1, index=self.init_plan.symbols)

            plan = pp.plan_params(self.init_plan.policy_wgts, returns, self.init_plan.vol,
                                  self.init_plan.corr, self.init_plan.symbols)

            #add the simulated plan to the list of plans and add the return vector to the return dataframe
            self.simulated_plans.append(plan)
            df = pd.concat([df, returns.to_frame().T], ignore_index=True)
        self.returns_df = df

    def generate_efficient_frontiers(self, bnds, cons, num_ports=20):

        self.init_plan.compute_eff_frontier(bnds, cons, num_ports)
        #average the weights across all the simulated plans
        avg_weights = np.zeros((len(self.init_plan.eff_frontier_tweights), len(self.init_plan)))
        for plan in self.simulated_plans:
            plan.compute_eff_frontier(bnds, cons,num_ports)
            avg_weights = avg_weights + plan.eff_frontier_tweights
        self.avg_weights = avg_weights/self.iter

        #create the dataframe that contains the averaged efficient frontier data
        i = 0
        ret = np.array([])
        vol = np.array([])
        for wgts in self.avg_weights:
            ret = np.append(ret, self.init_plan.portfolio_stats(self.avg_weights[i, :])[0])
            vol = np.append(vol, self.init_plan.portfolio_stats(self.avg_weights[i, :])[1])
            i = i+1

        self.opt_ports_df = dm.get_ports_df(ret, vol, self.avg_weights,
                                            self.init_plan.symbols, raw=True)
