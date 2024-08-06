"""
class that represents an analysis object
"""

from typing import Dict, Union, Tuple, List
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
import statsmodels.api as sm

from services.rainbow_portfolio_analytics.models.return_frequency import Frequency
from services.rainbow_portfolio_analytics.utils.factor_model_utils import get_Sidak_correction

CV_R2_THRESHOLD = 0.1

class AnalysisObject(object):
    """
    object that performs analysis given inputs
    """
    def __init__(self,
                 returns_series: pd.Series,
                 scaled_factors: pd.DataFrame,
                 frequency: Frequency,
                 scaling_model: StandardScaler,
                 window_size: str,
                 lambda_value: float
    ):
        
        if (lambda_value<0) and window_size=='lifetime':
            logging.error(" Negative lambda value received for lifetime analysis ")
            raise ValueError("Negative lambda value received for lifetime analysis")
        self.window_size:int = returns_series.shape[0] if window_size=='lifetime' else \
                                int(window_size[:-1])*12 if frequency==Frequency.MONTHLY else \
                                int(window_size[:-1])*4 if frequency==Frequency.QUARTERLY else int(window_size[:-1])*252
        if returns_series.shape[0]<self.window_size:
            logging.error(" Incorrect window size and frequency was provided ")
            raise ValueError("Incorrect window size and frequency was provided")
        self.returns_series:np.ndarray = returns_series.values[-self.window_size:]
        self.factor_returns:np.ndarray = scaled_factors.values[-self.window_size:,:]
        self.singlestep_flag: bool = False if window_size=='lifetime' else False if lambda_value>0 else True
        self.lambda_value: float = lambda_value if not self.singlestep_flag else self._get_singlestep_lambda(self.window_size)
        self.scaling_model: StandardScaler = scaling_model
        self.intercept, self.exposures = \
            self._fit_two_step_model(self.returns_series,self.factor_returns,self.lambda_value) if not self.singlestep_flag else \
            self._fit_single_step_model(self.returns_series,self.factor_returns,self.lambda_value)
        if window_size=='lifetime':
            cv_r2: float = self._get_cross_validated_R2(
                self.returns_series,
                self._get_predicted_returns(self.returns_series,self.lambda_value,self.factor_returns)
            )
            if cv_r2<CV_R2_THRESHOLD:
                self.singlestep_flag = True
                self.intercept, self.exposures = self._fit_single_step_model(
                    self.returns_series,
                    self.factor_returns,
                    self._get_singlestep_lambda(self.window_size)
                )
        self.exposures: Dict[Union[int,str],float] = {
            scaled_factors.columns[i]:self.exposures[i]/scaling_model.scale_[i] for i in range(self.exposures.shape[0])
        }
        
    def _get_singlestep_lambda(self, size:int)->float:
        # The numerical values here are inherited from the old AlphaInterpolator class
        progress = min(max((size - 52) / (520 - 52),0),1)
        return 0.055 * (1 - progress) + progress * 0.01

    def _get_cross_validated_R2(self,full_y:np.ndarray, oos_y:np.ndarray)->float:
    
        running_mean:np.ndarray = np.roll(np.cumsum(full_y)/np.arange(1,len(full_y)+1),1)
        Numerator: float = np.linalg.norm(full_y[-oos_y.shape[0]:]-oos_y)
        Denominator: float = np.linalg.norm(full_y[-oos_y.shape[0]:] - running_mean[-oos_y.shape[0]:])
        return 1 - (Numerator/Denominator)**2
    
    def _get_out_of_sample_value(self,insample_y:np.ndarray, insample_X:np.ndarray, oos_X:np.ndarray, lambda_param:float)->float:
    
        insample_intercept, insample_betas = self._fit_two_step_model(insample_y,insample_X,lambda_param)
        return insample_betas.dot(oos_X.reshape(-1))+insample_intercept

    def _get_predicted_returns(self, return_series:np.ndarray, given_lambda: float, factor_returns: np.ndarray)-> np.ndarray:
        
        return np.array([self._get_out_of_sample_value(return_series[:i], factor_returns[:i,:], factor_returns[i,:],given_lambda) 
                                                                        for i in range(int(return_series.shape[0]/2),return_series.shape[0])])
    
    def _fit_two_step_model(self, returns_series:np.ndarray, factor_returns:np.ndarray, lambda_value:float)->Tuple[float,np.ndarray]:
        
        eligible_factors: List[int] = self._get_lasso_selected_factor_ids(lambda_value, returns_series, factor_returns)
        insample_intercept, insample_betas, F_test_pval = self._get_OLS_fit_and_F_test(eligible_factors, returns_series, factor_returns)
        is_reliable = F_test_pval<get_Sidak_correction(factor_returns.shape[1],returns_series.shape[0])
        if not is_reliable:
            insample_intercept = np.mean(returns_series)
            insample_betas = np.zeros(insample_betas.shape)
        return insample_intercept, insample_betas
            
    def _fit_single_step_model(self, returns_series:np.ndarray, factor_returns:np.ndarray, lambda_value:float)->Tuple[float,np.ndarray]:
        
        lasso = Lasso(alpha=lambda_value).fit(factor_returns,returns_series)
        return lasso.intercept_, lasso.coef_

    def _get_lasso_selected_factor_ids(self,lambda_param: float, insample_y: np.ndarray, insample_X: np.ndarray)->List[int]:

        if lambda_param==0:            
            return [idx for idx in range(insample_X.shape[1])]
        lasso_coefficients = Lasso(alpha=lambda_param,max_iter=10000).fit(insample_X,insample_y).coef_
        return [idx for idx in range(len(lasso_coefficients)) if lasso_coefficients[idx]!=0]
    
    def _get_OLS_fit_and_F_test(self, eligible_columns: List[int], insample_y: np.ndarray, insample_X: np.ndarray)->Tuple[float, np.ndarray, float]:

        beta_values = np.zeros((insample_X.shape[1],))
        if len(eligible_columns)==0:
            return (np.mean(insample_y), beta_values,0.0)
        ols_fit = sm.OLS(endog=insample_y,exog=sm.add_constant(insample_X[:,eligible_columns]),hasconst=True).fit()
        const_model = sm.OLS(endog=insample_y,exog=sm.add_constant(insample_X[:,[]]),hasconst=True).fit()
        beta_values[eligible_columns] = ols_fit.params[1:]
        F_test_pval = ols_fit.compare_f_test(const_model)[1]
        return ols_fit.params[0], beta_values, F_test_pval
