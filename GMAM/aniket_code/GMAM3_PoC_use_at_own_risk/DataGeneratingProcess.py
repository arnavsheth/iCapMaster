import numpy as np
from datetime import datetime
from typing import Dict, Union, Tuple, List
from numpy.random import default_rng
from minimax_tilting_sampler import TruncatedMVN
from scipy.stats import gamma as GA
from numpy.random import gamma as GA_NP
from numpy.random import binomial as BN_NP
from numpy.random import normal as norm
from scipy.special import loggamma as loggamma_function
from random import sample, choices
from statsmodels.tsa.stattools import acovf
from itertools import chain
from HyperParameter import HyperParameter
from scipy.stats import rv_continuous
from scipy.linalg import lapack as lp
from enum import Enum
from logging import warning
from PoolUtils import NestablePool

class Statistic(Enum):

    MEAN = 'mean'
    STANDARD_DEVIATION = 'standard_deviation'
    SUM = 'sum'
    SQUARED_SUM = 'sum_of_squared'
    Q0P1 = 0.1
    Q0P5 = 0.5
    Q1P0 = 1.0
    Q5P0 = 5.0
    Q10P0 = 10.0
    Q25P0 = 25.0
    Q50P0 = 50.0
    Q75P0 = 75.0
    Q90P0 = 90.0
    Q95P0 = 95.0
    Q99P0 = 99.0
    Q99P5 = 99.5
    Q99P9 = 99.9


class Parameter(Enum):
    PSI = 'psi'
    BETA = 'beta'
    GAMMA = 'gamma'
    NU = 'nu'
    OMEGA = 'omega'
    PHI = 'phi'
    TAU_BETA = 'tau_beta'
    TAU_PHI = 'tau_phi'
    TAU_X = 'tau_x'
    TAU_Y = 'tau_y'
    X = 'x'

BURNIN_LENGTH = 20000
CHAIN_LENGTH = 20000 + BURNIN_LENGTH
TIMECHECK_ITERS = 1000
TIME_TO_QUIT = 5*60
MIN_ESS = 20
BOOTSTRAP_DRAWS = 1000
MIN_LAGS = 10
MAX_LAG_FRACTION = 0.25
AUTOCOVARIANCE_THRESHOLD = 0.02
MAX_CHAINLENGTH = 400000
NUMERICAL_ZERO = np.sqrt(np.finfo(np.float64).eps)
GAMMA_THRESHOLD = 2.1
PHI_LOWER_BOUND = -np.inf # 0.0 # 
PHI_UPPER_BOUND = np.inf # 1.0 # 
PARAMETERS_FOR_ESS = [Parameter.TAU_X, Parameter.TAU_Y, Parameter.TAU_PHI, Parameter.TAU_BETA, Parameter.NU, Parameter.BETA, Parameter.PHI]
DR = default_rng()

class truncated_gamma(rv_continuous):
    
    def _pdf(self, x: Union[float, np.ndarray], alpha: float, zeta: float)->float:
        
        base_dist = GA(a=alpha,scale=1/zeta)
        cdf_threshold = base_dist.cdf(GAMMA_THRESHOLD)
        if isinstance(x,float):
            return 0 if x<GAMMA_THRESHOLD else base_dist.pdf(x)/(1-cdf_threshold)
        else:
            ret_val = base_dist.pdf(x)/(1-cdf_threshold)
            ret_val[x<GAMMA_THRESHOLD] = 0
            return ret_val
    
    def _ppf(self, x: Union[float, np.ndarray], alpha: float, zeta: float)->float:
        
        base_dist = GA(a=alpha,scale=1/zeta)
        cdf_threshold = base_dist.cdf(GAMMA_THRESHOLD)
        
        return base_dist.ppf(x*(1-cdf_threshold) + cdf_threshold)

def get_gamma_mean(alpha: Union[float, np.ndarray], zeta: Union[float, np.ndarray])->Union[float, np.ndarray]:
    return alpha/zeta

def fix_spd_matrix(spd_mat: np.ndarray)->np.ndarray:
    S, V, eig_info = lp.dsyev(spd_mat,compute_v=1)
    if eig_info!=0:
        raise Exception('Eigen-decomposition failed for given SPD matrix')
    S_max = max(S)
    scale_factor = 1e-6
    S_threshold = S_max*scale_factor
    S[S<S_threshold] = S_threshold
    return V.dot(np.diag(S)).dot(V.transpose())
    
def get_fast_inverse_of_spd(spd_mat: np.ndarray) -> np.ndarray:

    cholesky, info_chol = lp.dpotrf(spd_mat,lower=1)
    if info_chol!=0:
        cholesky, info_chol = lp.dpotrf(fix_spd_matrix(spd_mat),lower=1)
        if info_chol!=0:
            raise Exception('Failed to fix spd matrix for cholesky')
    inverse_cholesky, info_inverse = lp.strtri(cholesky,lower=1)
    if info_inverse!=0:
        raise Exception('Cholesky has zero(s) on diagonal')
    return inverse_cholesky.transpose().dot(inverse_cholesky)

class DataGeneratingProcess(object):

    def __init__(self, hyper_parameters: HyperParameter, y: np.ndarray, F: np.ndarray, cash_returns: np.ndarray, pool_jobs: NestablePool = None):

        start_time = datetime.now()
        # Inputs and Constants
        self.hyper_parameters = hyper_parameters
        self.y = y
        self.F = F
        self.cash_returns = cash_returns
        self.Fr = np.concatenate([F,cash_returns.reshape(-1,1)],axis=1)
        self.covFr: np.ndarray = np.cov(self.Fr,rowvar=False)
        self.P = hyper_parameters.phi0.shape[0]
        self.lower_bound_for_phi = np.ones((self.P,))*PHI_LOWER_BOUND
        self.upper_bound_for_phi = np.ones((self.P,))*PHI_UPPER_BOUND
        self.S = y.shape[0]
        self.K = F.shape[1]
        self.T = F.shape[0]
        self.dT = int((self.T - self.P)/self.S) # T = P + S*dT
        self.iota_matrix_transpose = self._get_iota_matrix().transpose()
        self.R_matrix = self._get_R_matrix()
        self.N_parameters = len(Parameter)
        self.current_iteration = 1
        self.trunc_gamma: truncated_gamma = truncated_gamma(a=0)
        self.trunc_gamma_draws: np.ndarray = self.trunc_gamma.rvs(size=CHAIN_LENGTH+1,alpha=hyper_parameters.alphanu0, zeta=hyper_parameters.zetanu0)
        self.trunc_gamma_logpdfs: np.ndarray = self.trunc_gamma.logpdf(self.trunc_gamma_draws,alpha=hyper_parameters.alphanu0, zeta=hyper_parameters.zetanu0)
        self.nu_draw_idx = 0
        self.active_index: Dict[Parameter,int] = dict(zip(Parameter,[0 for _ in range(self.N_parameters)]))
        self.timers: Dict[Parameter, float] = dict(zip(Parameter,[0.0 for _ in range(self.N_parameters)]))
        self.convergence_timer: float = 0
        self.aggregation_timer: float = 0
        self.extension_timer: float = 0
        self.pool_jobs: NestablePool = pool_jobs

        # Initialize Parameters
        initial_standard_normals =  DR.standard_normal(size=(CHAIN_LENGTH+1,self.K+self.P+self.T))
        self.parameters: Dict[Parameter,np.ndarray] = {Parameter.PSI: np.zeros((CHAIN_LENGTH+1,self.T)),
                                                        Parameter.BETA : initial_standard_normals[:,:self.K],
                                                        Parameter.GAMMA : np.zeros((CHAIN_LENGTH+1,self.K)),
                                                        Parameter.NU : np.zeros((CHAIN_LENGTH+1,)),
                                                        Parameter.OMEGA : np.zeros((CHAIN_LENGTH+1,)),
                                                        Parameter.PHI : np.zeros((CHAIN_LENGTH+1,self.P)),#initial_standard_normals[:,self.K:self.K+self.P],
                                                        Parameter.TAU_BETA : np.zeros((CHAIN_LENGTH+1,)),
                                                        Parameter.TAU_PHI : np.zeros((CHAIN_LENGTH+1,)),
                                                        Parameter.TAU_X : np.zeros((CHAIN_LENGTH+1,)),
                                                        Parameter.TAU_Y : np.zeros((CHAIN_LENGTH+1,)),
                                                        Parameter.X : initial_standard_normals[:,self.K+self.P:]}
        self.parameters[Parameter.TAU_BETA][0] = get_gamma_mean(hyper_parameters.alphabeta0,hyper_parameters.zetabeta0)
        self.parameters[Parameter.TAU_PHI][0] = get_gamma_mean(hyper_parameters.alphaphi0,hyper_parameters.zetaphi0)
        self.parameters[Parameter.PHI][0,:] = hyper_parameters.phi0
        self.parameters[Parameter.OMEGA][0] = hyper_parameters.kappa0/(hyper_parameters.kappa0 + hyper_parameters.delta0)
        self.parameters[Parameter.GAMMA][0, 0] = 1
        self.parameters[Parameter.BETA][0,:] = hyper_parameters.betadelta0/self._get_sparsity_vector() + hyper_parameters.beta0
        self.parameters[Parameter.X][0,self.P:] = np.concatenate([np.ones(self.dT,)*dy for dy in self.y/self.dT])
        self.parameters[Parameter.X][0,:self.P] = 0
        x_Fb = self._get_conditional_explained_x(self.parameters[Parameter.BETA][0,:])
        self.parameters[Parameter.TAU_Y][0] = 1/np.var(self._get_y_tilda(x_Fb) - self._get_Xl_tilda_dot_phi(x_Fb, 
                                                                                            self.parameters[Parameter.PHI][0,:]), ddof=1)
        x_minus_x_Fb = self.parameters[Parameter.X][0,:]-x_Fb
        self.parameters[Parameter.TAU_X][0] = 1/(np.var(x_minus_x_Fb,ddof=1)*self.parameters[Parameter.TAU_Y][0])
        self.parameters[Parameter.NU][0] = get_gamma_mean(hyper_parameters.alphanu0,hyper_parameters.zetanu0)
        self.trunc_gamma_logpdfs[0] = self.trunc_gamma.logpdf(self.parameters[Parameter.NU][0],alpha=hyper_parameters.alphanu0, zeta=hyper_parameters.zetanu0)
        self.parameters[Parameter.PSI][0,:] = get_gamma_mean(np.ones((self.T,))*(self.parameters[Parameter.NU][0]*0.5+0.5),
                                                    self.parameters[Parameter.TAU_X][0]*self.parameters[Parameter.TAU_Y][0]*0.5*(x_minus_x_Fb)**2+
                                                    self.parameters[Parameter.NU][0]*0.5)
        self.initialization_timer: float = (datetime.now() - start_time).total_seconds()
        
    def get_backcasted_returns(self, full_factor_returns: np.ndarray, full_cash_returns: np.ndarray, is_qtrly: bool)->Dict[str, Dict[str,Union[float,np.ndarray]]]:
        
        backcasted_xF = full_factor_returns.dot(self.parameters[Parameter.BETA][BURNIN_LENGTH:self.current_iteration,:].T) + full_cash_returns.reshape(-1,1)
        all_phis = [np.concatenate([phi,1 - self.iota_matrix_transpose.dot(phi).reshape(-1)]) for phi in self.parameters[Parameter.PHI][BURNIN_LENGTH:self.current_iteration,:]]
        backcasted_yFs = np.concatenate([np.concatenate([np.roll(cFb,i).reshape(-1,1) 
                                for i in range(phi_tilda.shape[0])],axis=1).dot(phi_tilda[::-1])[phi_tilda.shape[0]:].reshape(-1,1) 
                                                for cFb, phi_tilda in zip(backcasted_xF.T,all_phis)],axis=1)
        if is_qtrly:
            spliced_yFs = backcasted_yFs[::-3,:].copy()[::-1,:]
        else:
            spliced_yFs = backcasted_yFs.copy()
        yF_stats = self._get_parameter_statistics(spliced_yFs.T)
        spliced_yFs[:,-self.y.shape[0]:] = self.y
        annualized_rets = spliced_yFs.mean(axis=0)*(4 if is_qtrly else 12)
        
        return {'backcasted_yF':yF_stats,
                'backcasted_xF':self._get_parameter_statistics(backcasted_xF.T),
                'annualized_returns':self._get_parameter_statistics(annualized_rets)}

    def get_aggregate_metrics(self)->Dict[str, Dict[str,Union[float,np.ndarray]]]:
        if self.current_iteration<=BURNIN_LENGTH:
            return {}
        start_time = datetime.now()
        #change to get entire 10,000 x 14 distributions rather than summary stats
        aggregate_metrics = {param.value: self._get_parameter_statistics(param) for param in Parameter if param!=Parameter.PSI}

        mu_x: np.ndarray = self._get_conditional_explained_x(self.parameters[Parameter.BETA][BURNIN_LENGTH:self.current_iteration,:])
        half_nu_vec: np.ndarray = self.parameters[Parameter.NU][BURNIN_LENGTH:self.current_iteration]*0.5
        psi_mat: np.ndarray = GA_NP(half_nu_vec,1/half_nu_vec,size=(self.T,half_nu_vec.shape[0])).transpose()
        std_dev_x: np.ndarray = np.concatenate([1/np.sqrt(row.reshape(-1,1)*tau_x*tau_y) 
                                                for row,tau_x,tau_y in zip(psi_mat,
                                                    self.parameters[Parameter.TAU_X][BURNIN_LENGTH:self.current_iteration],
                                                    self.parameters[Parameter.TAU_Y][BURNIN_LENGTH:self.current_iteration])],axis=1)
        xF_values = norm(mu_x,std_dev_x,size=mu_x.shape).transpose()
        mu_y: np.ndarray = np.concatenate([(self._get_Xl_tilda_dot_phi(row_x,row_phi)+self._get_xs_vector(row_x)).reshape(1,-1) 
                                            for row_x, row_phi in zip(xF_values,self.parameters[Parameter.PHI][BURNIN_LENGTH:self.current_iteration])])
        yF_values = norm(mu_y,np.outer(1/np.sqrt(self.parameters[Parameter.TAU_Y][BURNIN_LENGTH:self.current_iteration]),np.ones((1,self.S))),size=mu_y.shape)
        yF_deterministic = np.concatenate([(self._get_Xl_tilda_dot_phi(row_x,row_phi)+self._get_xs_vector(row_x)).reshape(1,-1) 
                                            for row_x, row_phi in zip(mu_x.T,self.parameters[Parameter.PHI][BURNIN_LENGTH:self.current_iteration])])
        reliability_metric = 1 - (np.linalg.norm(self.y.reshape(-1,1) - yF_deterministic.T,axis=0)/np.linalg.norm(self.y-np.mean(self.y)))**2
        reliability_indicator = np.array([float(metric >= 0) for metric in reliability_metric])
        aggregate_metrics.update({'reliability_metric':self._get_parameter_statistics(reliability_metric)})
        aggregate_metrics.update({"reliability_indicator": self._get_parameter_statistics(reliability_indicator)})
        aggregate_metrics.update({'insample_R2': self._get_parameter_statistics(np.array([np.corrcoef(self.y,row_yF)[0,1]**2 for row_yF in yF_deterministic]))})
        aggregate_metrics.update(dict(zip(['xF','yF'],[self._get_parameter_statistics(xF_values), self._get_parameter_statistics(yF_values)])))
        aggregate_metrics.update(**self._get_stats_sigma2())
        self.aggregation_timer = (datetime.now() - start_time).total_seconds()
        return aggregate_metrics
    
    # def get_unaggregated_data(self) -> Dict[str, Dict[str,Union[float,np.ndarray]]]:
    #     if self.current_iteration <= BURNIN_LENGTH:
    #         return {}
    
    #     # Collect raw data for each parameter
    #     unaggregated_data = {param.value: self.parameters[param][BURNIN_LENGTH:self.current_iteration, :]
    #                           for param in Parameter if param != Parameter.PSI}
    
    #     # Collect specific calculations as raw data
    #     mu_x = self._get_conditional_explained_x(self.parameters[Parameter.BETA][BURNIN_LENGTH:self.current_iteration, :])
    #     unaggregated_data['mu_x'] = mu_x
    
    #     half_nu_vec = self.parameters[Parameter.NU][BURNIN_LENGTH:self.current_iteration] * 0.5
    #     psi_mat = GA_NP(half_nu_vec, 1/half_nu_vec, size=(self.T, half_nu_vec.shape[0])).transpose()
    #     std_dev_x = np.concatenate([1/np.sqrt(row.reshape(-1, 1) * tau_x * tau_y)
    #                                 for row, tau_x, tau_y in zip(psi_mat,
    #                                                               self.parameters[Parameter.TAU_X][BURNIN_LENGTH:self.current_iteration],
    #                                                               self.parameters[Parameter.TAU_Y][BURNIN_LENGTH:self.current_iteration])], axis=1)
    #     unaggregated_data['std_dev_x'] = std_dev_x
    
    #     xF_values = norm(mu_x, std_dev_x, size=mu_x.shape).transpose()
    #     unaggregated_data['xF_values'] = xF_values
    
    #     mu_y = np.concatenate([(self._get_Xl_tilda_dot_phi(row_x, row_phi) + self._get_xs_vector(row_x)).reshape(1, -1)
    #                             for row_x, row_phi in zip(xF_values, self.parameters[Parameter.PHI][BURNIN_LENGTH:self.current_iteration])])
    #     yF_values = norm(mu_y, np.outer(1/np.sqrt(self.parameters[Parameter.TAU_Y][BURNIN_LENGTH:self.current_iteration]), np.ones((1, self.S))), size=mu_y.shape)
    #     unaggregated_data['yF_values'] = yF_values
    
    #     yF_deterministic = np.concatenate([(self._get_Xl_tilda_dot_phi(row_x, row_phi) + self._get_xs_vector(row_x)).reshape(1, -1)
    #                                         for row_x, row_phi in zip(mu_x.T, self.parameters[Parameter.PHI][BURNIN_LENGTH:self.current_iteration])])
    #     unaggregated_data['yF_deterministic'] = yF_deterministic
    
    #     reliability_metric = 1 - (np.linalg.norm(self.y.reshape(-1, 1) - yF_deterministic.T, axis=0)/np.linalg.norm(self.y - np.mean(self.y)))**2
    #     unaggregated_data['reliability_metric'] = reliability_metric
    
    #     reliability_indicator = np.array([float(metric >= 0) for metric in reliability_metric])
    #     unaggregated_data['reliability_indicator'] = reliability_indicator
    
    #     return unaggregated_data


    def update_parameters_randomly(self):

        def update_all_parameters_and_increase_counter():
            [func() for func in sample(functions_to_call,self.N_parameters)]
            self.current_iteration += 1

        functions_to_call = [self._get_updated_psi, self._get_updated_beta, self._get_updated_gamma, self._get_updated_nu, self._get_updated_omega, self._get_updated_phi, 
                             self._get_updated_tau_beta, self._get_updated_tau_phi, self._get_updated_tau_x, self._get_updated_tau_y, self._get_updated_x]
        convergence_flag = False
        while not convergence_flag:
            output_tuple = self._check_if_converged()
            convergence_flag = output_tuple[0]
            additional_draws = output_tuple[1]
            self._extend_parameter_arrays(additional_draws)
            count_i = 0
            while count_i<additional_draws:
                iters_to_run = min(TIMECHECK_ITERS,additional_draws-count_i)
                [update_all_parameters_and_increase_counter() for _ in range(iters_to_run)]
                count_i += iters_to_run
                if sum(self.timers.values())+self.aggregation_timer+self.convergence_timer+self.initialization_timer >= TIME_TO_QUIT:
                    print(" Timer check triggered ")
                    output_tuple = self._check_if_converged() if self.current_iteration>BURNIN_LENGTH else (False, 0, np.array([]))
                    convergence_flag = True
                    break
        self.convergence_flag = output_tuple[0]
        # self.final_ess_parameters = output_tuple[2]

    def _get_active_param_value(self, param:Parameter)->Union[float,np.ndarray]:
        return self.parameters[param][self.active_index[param]]

    def _check_if_converged(self)->Tuple[bool,int, np.ndarray]:
        
        # def get_param_convergence_flag(param_values: np.ndarray)->bool:
            
        #     lag_value = get_lag_where_decorrelated(param_values)
        #     if lag_values==-1:
        #         return False
        #     bootstrap_means: np.ndarray = get_block_bootstrap_means(lag_value, param_values)
        #     ess_of_parameter = param_values.var() / bootstrap_means.var()
        #     return (ess_of_parameter>=MIN_ESS) or np.isnan(ess_of_parameter)

        def get_lag_where_decorrelated(values: np.ndarray)->int:
            auto_covar = acovf(values,nlag=num_lags)
            if max(abs(auto_covar))<NUMERICAL_ZERO:
                return 1
            try:
                return max(1,np.where(np.abs(auto_covar)<=(AUTOCOVARIANCE_THRESHOLD*auto_covar[0]))[0][0])
            except:
                return -1
        
        def get_block_bootstrap_means(param_idx: int)->np.ndarray:
            lag = lag_values[param_idx]
            new_range = int(np.ceil(num_draws/lag))
            return parameters_to_test[param_idx, np.array([val%num_draws for val in chain.from_iterable(
                                [range(i*lag,(i+1)*lag) for i in choices(range(new_range),k=new_range*BOOTSTRAP_DRAWS)])
                            ]).reshape(BOOTSTRAP_DRAWS,new_range*lag)[:,:num_draws]].mean(axis=1)
        
        # def get_block_bootstrap_means(lag: int, param_values: np.ndarray)->np.ndarray:
        #     new_range = int(np.ceil(num_draws/lag))
        #     return param_values[np.array([val%num_draws for val in chain.from_iterable(
        #                         [range(i*lag,(i+1)*lag) for i in choices(range(new_range),k=new_range*BOOTSTRAP_DRAWS)])
        #                     ]).reshape(BOOTSTRAP_DRAWS,new_range*lag)[:,:num_draws]].mean(axis=1)

        if self.current_iteration==1:
            return False, CHAIN_LENGTH, np.array([])
        if self.current_iteration>=MAX_CHAINLENGTH:
            return True, 0, np.array([])
        start_time = datetime.now()
        additional_draws = 0
        num_draws = self.current_iteration - BURNIN_LENGTH
        num_lags = int(np.ceil(max(MIN_LAGS,MAX_LAG_FRACTION*num_draws)))
        parameters_to_test = [self.parameters[param][BURNIN_LENGTH:self.current_iteration] for param in PARAMETERS_FOR_ESS]
        parameters_to_test: np.ndarray = np.concatenate([param.reshape(1,-1) if param.ndim==1 else param.transpose() for param in parameters_to_test])
        # if not self.pool_jobs:
        lag_values = np.array([get_lag_where_decorrelated(row) for row in parameters_to_test])
        converged = False if any(lag_values==-1) else True
        bootstrap_means: np.ndarray = np.concatenate([get_block_bootstrap_means(param_idx).reshape(1,-1) 
                                                      for param_idx in range(parameters_to_test.shape[0]) if lag_values[param_idx]!=-1])
        ess_of_parameters = parameters_to_test[lag_values!=-1,:].var(axis=1) / bootstrap_means.var(axis=1)
        converged = all((ess_of_parameters>=MIN_ESS)|(np.isnan(ess_of_parameters))) and converged
        # else:
        #     converged = all(pool_jobs.map)
        if not converged:
            additional_draws = min(MAX_CHAINLENGTH-self.current_iteration,int(min(10,(MIN_ESS/min(ess_of_parameters))*(1+0.2))*num_draws))
        self.convergence_timer += (datetime.now() - start_time).total_seconds()
        return converged, additional_draws#, ess_of_parameters
    
    def _extend_parameter_arrays(self,N: int):

        if N==0 or self.current_iteration==1:
            return
        start_time = datetime.now()
        new_standard_normals = DR.standard_normal(size=(N,self.K+self.P+self.T))
        for param in Parameter:
            self.parameters[param] = np.append(self.parameters[param],np.zeros((N,)) if self.parameters[param].ndim==1 
                                                                                     else np.zeros((N,self.parameters[param].shape[1]))
                                                                                         if param not in [Parameter.X, Parameter.BETA]#, Parameter.PHI]
                                                                                         else new_standard_normals[:,:self.K]
                                                                                             if param==Parameter.BETA
                                                                                             # else new_standard_normals[:,self.K:self.K+self.P]
                                                                                             #     if param==Parameter.PHI
                                                                                                 else new_standard_normals[:,self.K+self.P:],axis=0)
        new_trunc_draws = self.trunc_gamma.rvs(size=N,alpha=self.hyper_parameters.alphanu0, zeta=self.hyper_parameters.zetanu0)
        self.trunc_gamma_draws = np.append(self.trunc_gamma_draws, new_trunc_draws)
        self.trunc_gamma_logpdfs = np.append(self.trunc_gamma_logpdfs, 
                                             self.trunc_gamma.logpdf(new_trunc_draws,
                                                                     alpha=self.hyper_parameters.alphanu0, 
                                                                     zeta=self.hyper_parameters.zetanu0))
        self.extension_timer += (datetime.now() - start_time).total_seconds()
    
    def _get_updated_phi(self):

        start_time = datetime.now()
        param = Parameter.PHI
        tau_y = self._get_active_param_value(Parameter.TAU_Y)
        # sqrt_tau_y = np.sqrt(self._get_active_param_value(Parameter.TAU_Y))
        tau_phi = self._get_active_param_value(Parameter.TAU_PHI)
        m0: np.ndarray = self.hyper_parameters.m0
        phi_0 = self.hyper_parameters.phi0
        Xl_tilda = self._get_Xl_tilda()
        y_tilda = self._get_y_tilda()
        Lambda_phi_by_tau_y = Xl_tilda.transpose().dot(Xl_tilda) + np.diag(tau_phi*m0) 
        # cholesky, mu_phi, _ = lp.dposv(Lambda_phi_by_tau_y,Xl_tilda.transpose().dot(y_tilda) + tau_phi*m0*phi_0,lower=1)
        # cholesky = np.tril(cholesky)
        # self.parameters[param][self.current_iteration, :] = lp.dtrtrs(cholesky, self.parameters[param][self.current_iteration, :]/sqrt_tau_y,lower=1,trans=1)[0] + mu_phi
        Sigma_phi_times_tau_y: np.ndarray = get_fast_inverse_of_spd(Lambda_phi_by_tau_y) 
        Sigma_phi = Sigma_phi_times_tau_y/tau_y
        mu_phi: np.ndarray = Sigma_phi_times_tau_y.dot(Xl_tilda.transpose().dot(y_tilda) + tau_phi*m0*phi_0)
        
        try:
            self.parameters[param][self.current_iteration,:] = TruncatedMVN(mu_phi,Sigma_phi, self.lower_bound_for_phi, 
                                                                            self.upper_bound_for_phi).sample(1).reshape(-1)
        except:
            self.parameters[param][self.current_iteration,:] = TruncatedMVN(mu_phi,fix_spd_matrix(Sigma_phi), 
                                                                            self.lower_bound_for_phi, self.upper_bound_for_phi).sample(1).reshape(-1)
            warning(f'Phi update failed for iteration {self.current_iteration}. \
                  Condition number of Sigma_phi is {np.linalg.cond(Sigma_phi)}. Using modified covariance.')
        self.active_index[param] += 1
        self.timers[param] += (datetime.now() - start_time).total_seconds()

    def _get_updated_x(self):

        start_time = datetime.now()
        param = Parameter.X
        Phi_mat: np.ndarray = self._get_phi_tilda_matrix()
        tau_x = self._get_active_param_value(Parameter.TAU_X)
        sqrt_tau_y = np.sqrt(self._get_active_param_value(Parameter.TAU_Y))
        psi = self._get_active_param_value(Parameter.PSI)
        Lambda_x_by_tau_y = Phi_mat.transpose().dot(Phi_mat) + np.diag(tau_x*psi) 
        cholesky, mu_x, _ = lp.dposv(Lambda_x_by_tau_y,Phi_mat.transpose().dot(self.y) + tau_x*self._get_conditional_explained_x()*psi,lower=1)
        cholesky = np.tril(cholesky)
        self.parameters[param][self.current_iteration, :] = lp.dtrtrs(cholesky, self.parameters[param][self.current_iteration, :]/sqrt_tau_y,lower=1,trans=1)[0] + mu_x
        self.active_index[param] += 1
        self.timers[param] += (datetime.now() - start_time).total_seconds()

    def _get_updated_tau_y(self):
        
        start_time = datetime.now()
        param = Parameter.TAU_Y
        phi = self._get_active_param_value(Parameter.PHI)
        tau_phi = self._get_active_param_value(Parameter.TAU_PHI)
        a0 = self.hyper_parameters.a0
        m0 = self.hyper_parameters.m0
        phi0 = self.hyper_parameters.phi0
        tau_x = self._get_active_param_value(Parameter.TAU_X)
        psi = self._get_active_param_value(Parameter.PSI)
        beta_tilda = self._get_beta_tilda()
        tau_beta = self._get_active_param_value(Parameter.TAU_BETA)
        d = self._get_sparsity_vector()
        vec1: np.ndarray = (self._get_y_tilda() - self._get_Xl_tilda_dot_phi())
        vec2: np.ndarray = (phi - phi0)
        vec3: np.ndarray = (self._get_active_param_value(Parameter.X)-self._get_conditional_explained_x())
        vec4: np.ndarray = (beta_tilda - self.hyper_parameters.betadelta0/d)
        alphay = 0.5*(self.S + self.T + self.P + self.K) + self.hyper_parameters.alphay0
        zetay = self.hyper_parameters.zetay0 + 0.5*(vec1.dot(vec1) + tau_phi*vec2.dot(m0*vec2) + tau_x*vec3.dot(psi*vec3) + 
                                                    tau_x*tau_beta*(vec4.dot(vec4*(d**2)*a0)))
        self.parameters[param][self.current_iteration] = GA_NP(alphay,1/zetay,1)[0]
        self.active_index[param] += 1
        self.timers[param] += (datetime.now() - start_time).total_seconds()

    def _get_updated_tau_x(self):

        start_time = datetime.now()
        param = Parameter.TAU_X
        tau_y = self._get_active_param_value(Parameter.TAU_Y)
        tau_beta = self._get_active_param_value(Parameter.TAU_BETA)
        a0 = self.hyper_parameters.a0
        psi = self._get_active_param_value(Parameter.PSI)
        beta_tilda = self._get_beta_tilda()
        d = self._get_sparsity_vector()
        vec3: np.ndarray = (self._get_active_param_value(Parameter.X)-self._get_conditional_explained_x())
        vec4: np.ndarray = (beta_tilda - self.hyper_parameters.betadelta0/d)
        alphax = 0.5*(self.T+self.K) + self.hyper_parameters.alphax0
        zetax = self.hyper_parameters.zetax0 + 0.5*tau_y*(vec3.dot(psi*vec3) + tau_beta*vec4.dot(vec4*(d**2)*a0))
        self.parameters[param][self.current_iteration] = GA_NP(alphax,1/zetax,1)[0] 
        self.active_index[param] += 1
        self.timers[param] += (datetime.now() - start_time).total_seconds()
    
    def _get_updated_tau_phi(self):
        
        start_time = datetime.now()
        param = Parameter.TAU_PHI
        tau_y = self._get_active_param_value(Parameter.TAU_Y)
        phi = self._get_active_param_value(Parameter.PHI)
        phi0 = self.hyper_parameters.phi0
        m0 = self.hyper_parameters.m0
        vec2: np.ndarray = (phi - phi0)
        alphaphi = self.hyper_parameters.alphaphi0 + self.P*0.5
        zetaphi = self.hyper_parameters.zetaphi0 + 0.5*tau_y*vec2.dot(m0*vec2)
        self.parameters[param][self.current_iteration] = GA_NP(alphaphi,1/zetaphi,1)[0] 
        self.active_index[param] += 1
        self.timers[param] += (datetime.now() - start_time).total_seconds()

    def _get_updated_tau_beta(self):

        start_time = datetime.now()
        param = Parameter.TAU_BETA
        tau_x = self._get_active_param_value(Parameter.TAU_X)
        tau_y = self._get_active_param_value(Parameter.TAU_Y)
        beta_tilda = self._get_beta_tilda()
        betadelta0 = self.hyper_parameters.betadelta0
        d = self._get_sparsity_vector()
        a0 = self.hyper_parameters.a0
        alphabeta = 0.5*self.K + self.hyper_parameters.alphabeta0
        vec4: np.ndarray = (beta_tilda - betadelta0/d)
        zetabeta = tau_x*tau_y*0.5*vec4.dot(vec4*a0*(d**2)) + self.hyper_parameters.zetabeta0
        self.parameters[param][self.current_iteration] = GA_NP(alphabeta,1/zetabeta,1)[0] 
        self.active_index[param] += 1
        self.timers[param] += (datetime.now() - start_time).total_seconds()

    def _get_updated_beta(self):

        start_time = datetime.now()
        param = Parameter.BETA
        d = self._get_sparsity_vector()
        tau_x = self._get_active_param_value(Parameter.TAU_X)
        tau_beta = self._get_active_param_value(Parameter.TAU_BETA)
        tau_y = self._get_active_param_value(Parameter.TAU_Y)
        sqrt_tau_x_tau_y = np.sqrt(tau_x*tau_y)
        a0 = self.hyper_parameters.a0
        psi = self._get_active_param_value(Parameter.PSI)
        psi_F = np.concatenate([psi[j]*self.F[j,:].reshape(1,-1) for j in range(psi.shape[0])])
        beta0 = self.hyper_parameters.beta0
        betadelta0 = self.hyper_parameters.betadelta0
        Lambda_beta_by_tau_x_tau_y = self.F.transpose().dot(psi_F) + np.diag(tau_beta*a0*(d**2)) 
        cholesky, mu_beta, _ = lp.dposv(Lambda_beta_by_tau_x_tau_y,self.F.transpose().dot(psi*(self._get_active_param_value(Parameter.X) - 
                                                                                self.cash_returns)) + tau_beta*d*a0*(d*beta0+betadelta0),lower=1)
        cholesky = np.tril(cholesky)
        self.parameters[param][self.current_iteration, :] = lp.dtrtrs(cholesky, self.parameters[param][self.current_iteration, :]/sqrt_tau_x_tau_y,lower=1,trans=1)[0] + mu_beta
        self.active_index[param] += 1
        self.timers[param] += (datetime.now() - start_time).total_seconds()

    def _get_updated_gamma(self):
        
        start_time = datetime.now()
        param = Parameter.GAMMA
        tau_x = self._get_active_param_value(Parameter.TAU_X)
        tau_y = self._get_active_param_value(Parameter.TAU_Y)
        tau_beta = self._get_active_param_value(Parameter.TAU_BETA)
        a0 = self.hyper_parameters.a0
        beta_tilda = self._get_beta_tilda()
        betadelta0 = self.hyper_parameters.betadelta0
        omega = self._get_active_param_value(Parameter.OMEGA)
        v = self.hyper_parameters.v
        ptilda_1 = omega*np.exp(-0.5*tau_x*tau_y*tau_beta*a0*(beta_tilda**2 - 2*betadelta0*beta_tilda))
        ptilda_0 = (1-omega)/v * np.exp(-0.5*tau_x*tau_y*tau_beta*a0*((beta_tilda/v)**2 - 2*betadelta0*(beta_tilda/v)))
        pgamma = ptilda_1 / (ptilda_1 + ptilda_0)
        self.parameters[param][self.current_iteration,:] = BN_NP(1,pgamma,size=(pgamma.shape[0],)).reshape(-1)
        self.active_index[param] += 1
        self.timers[param] += (datetime.now() - start_time).total_seconds()

    def _get_updated_omega(self):

        start_time = datetime.now()
        param = Parameter.OMEGA
        gamma_sum = self._get_active_param_value(Parameter.GAMMA).sum()
        kappa = self.hyper_parameters.kappa0 + gamma_sum
        delta = self.hyper_parameters.delta0 + self.K - gamma_sum
        self.parameters[param][self.current_iteration] = DR.beta(kappa,delta,size=1)[0]
        self.active_index[param] += 1
        self.timers[param] += (datetime.now() - start_time).total_seconds()
    
    def _get_updated_psi(self):

        start_time = datetime.now()
        param = Parameter.PSI
        tau_x = self._get_active_param_value(Parameter.TAU_X)
        tau_y = self._get_active_param_value(Parameter.TAU_Y)
        nu = self._get_active_param_value(Parameter.NU)
        alphapsi = np.ones((self.T,))*0.5*(1+nu)
        zetapsi = 0.5*(nu + tau_x*tau_y*(self._get_active_param_value(Parameter.X)-self._get_conditional_explained_x())**2 )
        self.parameters[param][self.current_iteration,:] = GA_NP(alphapsi,1/zetapsi,size=(self.T,))
        self.active_index[param] += 1
        self.timers[param] += (datetime.now() - start_time).total_seconds()

    def _get_updated_nu(self):

        start_time = datetime.now()
        param = Parameter.NU
        def log_prob_nu(nu_val)->float:
            return np.log(0.5*nu_val)*(0.5*self.T*nu_val + alphanu0-1) - self.T*loggamma_function(0.5*nu_val) + 0.5*nu_val*eta1

        psi = self._get_active_param_value(Parameter.PSI)
        nu = self._get_active_param_value(param)
        zetanu0 = self.hyper_parameters.zetanu0
        eta1 = sum(np.log(psi)-psi) - 2*zetanu0
        alphanu0 = self.hyper_parameters.alphanu0
        log_prob_nu0 = log_prob_nu(nu)
        log_r_nu0 = self.trunc_gamma_logpdfs[self.nu_draw_idx]
        nu_prime = self.trunc_gamma_draws[self.current_iteration] 
        log_prob_nu_prime = log_prob_nu(nu_prime)
        log_r_nu_prime = self.trunc_gamma_logpdfs[self.current_iteration] 
        prob_ratio = np.exp((log_prob_nu_prime-log_r_nu_prime)-(log_prob_nu0-log_r_nu0))
        if prob_ratio>=1:
            self.parameters[param][self.current_iteration] = nu_prime
            self.nu_draw_idx = self.current_iteration
        elif np.random.uniform(0,1,1)<=prob_ratio:
            self.parameters[param][self.current_iteration] = nu_prime
            self.nu_draw_idx = self.current_iteration
        else:
            self.parameters[param][self.current_iteration] = nu 
        self.active_index[param] += 1
        self.timers[param] += (datetime.now() - start_time).total_seconds()

    def _get_iota_matrix(self)->np.ndarray:
        eye_dT = np.eye(self.dT)
        n = int(self.P/self.dT)
        r = self.P%self.dT
        iota_matrix: np.ndarray = np.concatenate([eye_dT for _ in range(n)])
        if r!=0:
            iota_matrix: np.ndarray = np.concatenate([iota_matrix, eye_dT[:r,:]])
        return iota_matrix
    
    def _get_R_matrix(self)->np.ndarray:
        return np.concatenate([np.eye(self.P),-self.iota_matrix_transpose])
    
    def _get_sparsity_vector(self)->np.ndarray:
        values = self._get_active_param_value(Parameter.GAMMA)
        return np.sqrt(values + (1-values)/(self.hyper_parameters.v**2))
    
    def _get_phi_tilda(self, phi: np.ndarray)->np.ndarray:
        if phi is None:
            phi = self._get_active_param_value(Parameter.PHI)
        phi_tilda = np.zeros((self.T,))
        phi_tilda[:self.P+self.dT] = np.concatenate([phi,1 - self.iota_matrix_transpose.dot(phi).reshape(-1)])
        return phi_tilda
    
    def _get_phi_tilda_matrix(self, phi: np.ndarray = None)->np.ndarray:
        if phi is None:
            phi = self._get_active_param_value(Parameter.PHI)
        phi_tilda = self._get_phi_tilda(phi)
        return np.concatenate([np.roll(phi_tilda,i).reshape(1,-1) for i in range(0,self.T-self.P,self.dT)])
    
    def _get_xs_vector(self, x: np.ndarray)->np.ndarray:
        return np.array([sum(x[self.P+i:self.P+i+self.dT]) for i in range(0,self.T-self.P,self.dT)])
    
    def _get_y_tilda(self, x: np.ndarray = None)->np.ndarray:
        if x is None:
            x = self._get_active_param_value(Parameter.X)
        return self.y - self._get_xs_vector(x)
    
    def _get_Xl_tilda_dot_phi(self, x: np.ndarray = None, phi: np.ndarray = None) -> np.ndarray:
        if x is None or phi is None:
            x = self._get_active_param_value(Parameter.X)
            phi = self._get_active_param_value(Parameter.PHI)
        return self._get_Xl_tilda(x).dot(phi.transpose())

    def _get_Xl_tilda(self, x: np.ndarray = None)->np.ndarray:
        if x is None:
            x = self._get_active_param_value(Parameter.X)
        return np.concatenate([x[i:i+self.P+self.dT].reshape(1,-1) for i in range(0,self.T-self.P,self.dT)]).dot( self.R_matrix )
    
    def _get_beta_tilda(self, beta: np.ndarray = None)->np.ndarray:
        if beta is None:
            beta = self._get_active_param_value(Parameter.BETA)
        return beta - self.hyper_parameters.beta0
    
    def _get_conditional_explained_x(self, beta: np.ndarray = None)->np.ndarray:
        if beta is None:
            beta = self._get_active_param_value(Parameter.BETA)
        if beta.ndim == 1:
            return self.F.dot(beta.transpose()) + self.cash_returns
        else:
            return self.F.dot(beta.transpose()) + self.cash_returns.reshape(-1,1)
    
    def _get_parameter_statistics(self, param: Union[Parameter,np.ndarray])->Dict[str, Union[float, np.ndarray]]:
        if isinstance(param,Parameter):
            values = self.parameters[param][BURNIN_LENGTH:self.current_iteration]
        else:
            values = param
        stats = [values.mean(axis=0), values.std(axis=0,ddof=1), values.sum(axis=0), (values*values).sum(axis=0)]
        stats.extend([row for row in np.percentile(values,[key.value 
                                                           for key in Statistic if key not in [Statistic.MEAN, Statistic.STANDARD_DEVIATION, 
                                                                                               Statistic.SUM, Statistic.SQUARED_SUM]],axis=0)])
        return dict(zip([str(key.value) for key in Statistic],stats))
    
    def _get_stats_sigma2(self)->Dict[str, Dict[str,Union[float,np.ndarray]]]:
        beta_ones: np.ndarray = np.concatenate([self.parameters[Parameter.BETA], np.ones((self.parameters[Parameter.BETA].shape[0],1))],axis=1)[BURNIN_LENGTH:self.current_iteration]
        nu = self.parameters[Parameter.NU][BURNIN_LENGTH:self.current_iteration]
        tau_x = self.parameters[Parameter.TAU_X][BURNIN_LENGTH:self.current_iteration]
        tau_y = self.parameters[Parameter.TAU_Y][BURNIN_LENGTH:self.current_iteration]
        phi = self.parameters[Parameter.PHI][BURNIN_LENGTH:self.current_iteration]
        phi_tilda_norm = np.array([np.linalg.norm(self._get_phi_tilda(row)) for row in phi])
        sigma_x_eps: np.ndarray = (nu/((nu-2)*(tau_x*tau_y)))
        sigma_x_eps_adj: np.ndarray = sigma_x_eps + 1/(tau_y*self.dT)
        sigma_x_vec: np.ndarray = np.array([row.dot(self.covFr.dot(row)) for row in beta_ones]) + sigma_x_eps
        sigma_x_adj: np.ndarray = sigma_x_vec + 1/(tau_y*self.dT)
        sigma_y_eps: np.ndarray = sigma_x_eps*phi_tilda_norm + 1/tau_y
        sigma_y_vec: np.ndarray = sigma_x_vec*phi_tilda_norm + 1/tau_y
        return {variable:self._get_parameter_statistics(values) 
                for variable, values in zip(['sigma_x_eps','sigma_x_eps_adj','sigma_x_vec','sigma_x_adj','sigma_y_eps','sigma_y_vec'],
                                                        [sigma_x_eps,sigma_x_eps_adj,sigma_x_vec,sigma_x_adj,sigma_y_eps,sigma_y_vec])}