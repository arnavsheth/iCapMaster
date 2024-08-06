import numpy as np
# from ConnectionObject import ConnectionObject
import pandas as pd
from typing import List
class HyperParameter(object):
    
    def __init__(self, K: int, P: int, beta_ids_to_modify: List[int], 
                 is_quarterly:bool = False, is_ares:bool = False,
                 is_fscredit:bool = False) -> None:
        self.v = 0.01
        self.kappa0 = 1.250728863 if is_quarterly else 1.5
        self.delta0 = 1.66763848396501 if is_quarterly else 1.5
        self.alphay0 = 1.234568
        self.zetay0 = 9.38272E-05 if is_quarterly else 0.0000059 if is_fscredit else 2.34568e-05
        self.alphax0 = 1.234568
        self.zetax0 = 3.0400000 if is_fscredit else 5.864197531
        self.alphaphi0 = 0.5
        self.zetaphi0 = 0.5
        self.alphabeta0 = 0.5
        self.zetabeta0 = 0.5
        self.alphanu0 = (4+np.sqrt(816))/400*4 + 1.0
        self.zetanu0 = (4+np.sqrt(816))/400
        self.m0default = 4.0 if is_quarterly else 9.0
        self.beta0 = np.zeros((K,))
        if is_quarterly or is_ares:
            print("Setting market and smallcap beta priors to 1")
            for idbeta in beta_ids_to_modify:
                self.beta0[idbeta] = 1
        if is_fscredit:
            self.beta0[11] = 0.4
            # self.beta0[11] = self.beta0[13] = 0.1
            # self.beta0[6] = 1 # equity market exposure
            # self.beta0[9] = 1 # equity small cap exposure
        self.betadelta0 = np.zeros((K,))
        self.a0 = np.ones((K,)) * ( self.zetay0 * self.zetax0 * self.zetabeta0 / (self.alphay0 * self.alphax0 * self.alphabeta0) )
        self.a0[0] *= 10000
        self.m0 = np.ones((P,)) * ( self.zetay0*self.zetaphi0 / (self.alphay0*self.alphaphi0) ) * self.m0default
        self.phi0 = np.zeros((P,))

    # def update_from_priors_table(self, asset_id: int, db_connection: ConnectionObject)->None:

    #     prior_table_key = pd.read_sql(f""" SELECT "prior_key" FROM asset WHERE "id" = {asset_id} """,db_connection.ipidb_conn).values[0]
    #     if prior_table_key:
    #         return
    #     else:
    #         prior_values = pd.read_sql(f""" SELECT * FROM prior_table WHERE "key" = {prior_table_key} """,db_connection.ipidb_conn).to_dict()
    #         self.v = prior_values['v']
    #         self.kappa0 = prior_values['kappa0']
    #         self.delta0 = prior_values['delta0']
    #         self.alphay0 = prior_values['alphay0']
    #         self.zetay0 = prior_values['zetay0']
    #         self.alphax0 = prior_values['alphax0']
    #         self.zetax0 = prior_values['zetax0']
    #         self.alphaphi0 = prior_values['alphaphi0']
    #         self.zetaphi0 = prior_values['zetaphi0']
    #         self.alphabeta0 = prior_values['alphabeta0']
    #         self.zetabeta0 = prior_values['zetabeta0']
    #         self.alphanu0 = prior_values['alphanu0']
    #         self.zetanu0 = prior_values['zetanu0']
    #         self.m0default = prior_values['m0default']
    #         self.beta0 = prior_values['beta0']
    #         self.betadelta0 = prior_values['betadelta0']
    #         self.a0 = prior_values['a0']
    #         self.m0 = prior_values['m0']
    #         self.phi0 = prior_values['phi0']