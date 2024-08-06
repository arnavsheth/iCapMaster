"""
calculator that computes optimal lambdas for gmam 2
"""

import logging
from typing import List, Dict, Tuple, Union
from itertools import chain
import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd
from sklearn.linear_model import LinearRegression, lars_path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from services.rainbow_portfolio_analytics.calculators.analysis_object import AnalysisObject
from services.rainbow_portfolio_analytics.config import GMAM_3_RISK_MODEL_NAME
from services.rainbow_portfolio_analytics.helpers.date_helpers import DEFAULT_DATE_FORMAT
from services.rainbow_portfolio_analytics.helpers.date_helpers import get_intersection_dates
from services.rainbow_portfolio_analytics.models.analytics import ReturnStreamInformation
from services.rainbow_portfolio_analytics.models.data_generating_process import DataGeneratingProcess
from services.rainbow_portfolio_analytics.models.hyperparameter import HyperParameter

from services.rainbow_portfolio_analytics.models.return_frequency import Frequency
from services.rainbow_portfolio_analytics.utils.constants import MIN_DATAPOINTS_FORGMAM3, MONTHS_IN_QUARTER, TOTAL_TERMS_IN_MA_PROCESS
from services.rainbow_portfolio_analytics.utils.factor_model_utils import add_constant
from simon.common.simon_logging import LOGGER

def get_clean_df_from_return_stream(list_of_return_streams) -> pd.DataFrame:
    """
    get clean df when passed return streams

    Args:
        list_of_return_streams: self-explanatory
    Returns:
        dataframe of return streams with ids as column names
    """
    return_series = [rs.return_series for rs in list_of_return_streams]
    ids = [rs.id for rs in list_of_return_streams]
    df = pd.concat(return_series,axis=1).dropna()
    df.columns = ids
    return df

def get_clean_df_from_series(list_of_series:List[pd.Series])->pd.DataFrame:
    """
    helper function that concatenates series into a df
    
    Args:
        list_of_series: list of pandas series
    
    Returns:
        dataframe of the concatenated series
    """
    return pd.concat(list_of_series,axis=1).dropna()

def get_intersected_indices(
    object1: Union[pd.Series,pd.DataFrame],
    object2: Union[pd.Series,pd.DataFrame]
) -> Tuple[List[pd.Timestamp],Union[pd.Series,pd.DataFrame],Union[pd.Series,pd.DataFrame]]:
    """
    function that gets the intersection of dates for two series

    Args:
        object1: first object
        object2: second object
    Returns:
        common intersection, truncated objects 
    """
    common = sorted(list(set(object1.index.unique()).intersection(set(object2.index.unique()))))
    truncated_object1 = object1.loc[common].sort_index()
    truncated_object2 = object2.loc[common].sort_index()
    return common, truncated_object1, truncated_object2

class FactorAnalysisCalculator(object):
    """
    Factor Analysis Calculator for the overnight processes

    Args:
        asset_returnstreams: asset return streams
        factor_returnstreams: factor return streams
        cash_asset: cash asset
        model_name: risk model name
    Returns:
        None
    """

    def __init__(self,
                 asset_returnstreams,
                 factor_returnstreams,
                 cash_asset,
                 model_name: str,
                 frequency: Frequency = Frequency.MONTHLY
    ):
        self.asset_returnstreams = asset_returnstreams
        self.frequency = frequency
        self.factors_returnstreams = factor_returnstreams
        self.cash_asset = cash_asset
        self.model_name = model_name

    def get_analysis_results_for_storage(self):
        """
        gets the analysis results when passed the input

        Args:
            None
        Returns:
            analysis results including lambda, factor exposures, and betas
        """
        results: List[Dict[str,Union[int,str,float,Dict[str,Union[Frequency,float]]]]] = list(
            chain.from_iterable(
                [self._get_factor_analysis_results(
                    asset,
                    self.factors_returnstreams,
                    self.cash_asset,
                    self.model_name,
                    self.frequency,
                    str(pd.Timestamp.now().date())
                ) for asset in self.asset_returnstreams]
            )
        )
        return results
    
    def get_gmam3_analysis_results_for_storage(self):
        """
        gets the gmam 3 analysis results when passed the input

        Args:
            None
        Returns:
            analysis results including lambda, factor exposures, and betas
        """
        results: List[Dict[str,Union[int,str,float,Dict[str,Union[Frequency,float]]]]] = list(
            chain.from_iterable(
                [self._get_gmam_3_factor_analysis_results(
                    asset,
                    self.factors_returnstreams,
                    self.cash_asset,
                    self.model_name,
                    self.frequency,
                    str(pd.Timestamp.now().date())
                ) for asset in self.asset_returnstreams]
            )
        )
        return results

    def _get_list_of_window_sizes(self, max_window_size_in_yrs:int)-> List[str]:
        """
        gets the analysis results when passed the input

        Args:
            None
        Returns:
            analysis results including lambda, factor exposures, and betas
        """
        windows_in_yrs = []
        window_sizes = [3, 5, 8, 10, 15] # replace according to FE
        window = 1
        for window in window_sizes:
            if window < max_window_size_in_yrs:
                windows_in_yrs.append(window)

        return [str(idx)+'y' for idx in windows_in_yrs]

    def _get_train_test_indices(
        self,
        n_portfolio:int,
        window_len:int = -1
    )->List[Tuple[List[int],List[int]]]:
        """
        get train test indices

        Args:
            n_portfolio: size of portfolio
            window_len: window length
        Returns:
            train test indices
        """
        n_oos = 1
        n_folds = 20
        if window_len!=-1:
            oos_size = max(n_oos,int((n_portfolio - window_len/2)/n_folds))
            min_insample = int(window_len/2)
            return [
                (list(range(max(0,i-window_len),i)),list(range(i,min(i+oos_size,n_portfolio))))
                for i in range(min_insample,n_portfolio,oos_size)
            ]
        else:
            oos_size = max(n_oos,int(n_portfolio/(2*n_folds)))
            min_insample = int(n_portfolio/2)
            return [
                (list(range(i)),list(range(i,min(i+oos_size,n_portfolio))))
                for i in range(min_insample,n_portfolio,oos_size)
            ]
        
    def _get_excess_returns(self, asset_returns, cash_returns)->pd.Series:
        """
        get excess returns

        Args:
            asset_returns: asset return streams
            cash_returns: cash return stream
        Returns:
            excess return stream
        """
        return (asset_returns.return_series - cash_returns.return_series).dropna()
    
    def _get_scaled_factor_returns(self, factors)->Tuple[pd.DataFrame,StandardScaler]:
        """
        get scaled factor returns

        Args:
            factors: the factor return streams
        Returns:
            scaled factor return streams
        """
        unscaled_factor_returns: pd.DataFrame = get_clean_df_from_return_stream(factors)
        scaling_model: StandardScaler = StandardScaler(with_mean=False).fit(unscaled_factor_returns)
        return pd.DataFrame(
            scaling_model.transform(unscaled_factor_returns),index=unscaled_factor_returns.index,columns=unscaled_factor_returns.columns
        ), scaling_model

    def _get_factor_analysis_results(self,
        asset_returnstream,
        factors,
        cash_asset,
        model_name: str,
        frequency: Frequency,
        asset_version_date: str
    )->List[Dict[str,Union[int,str,float,Dict[str,Union[Frequency,float]]]]]:
        """
        get the factor analysis results

        Args:
            asset_returnstream: list of asset returns
            factors: factor return streams
            cash_asset: cash asset return stream,
            model_name: model name
            frequency: the frequency of returns
            asset_version_date: the date that we should version the results

        Returns:
            factor analysis results to store in db
        """
        return_list = []
        if self.frequency == Frequency.MONTHLY:
            pd_freq = "M"
        elif self.frequency == Frequency.QUARTERLY:
            pd_freq = "Q"
        model_start, model_end, portfolio_start, portfolio_end = get_intersection_dates([asset_returnstream], factors, cash_asset)
        asset_returnstream = asset_returnstream.trim_to_range(portfolio_end, portfolio_start)
        cash_asset_portfolio = cash_asset.trim_to_range(portfolio_end, portfolio_start)
        cash_asset_model = cash_asset.trim_to_range(model_end, model_start)
        factors = [factor.trim_to_range(model_end, model_start) for factor in factors]
        time_labels = [d.strftime(DEFAULT_DATE_FORMAT) for d in pd.date_range(start=model_start, end=model_end, freq=pd_freq)]

        excess_returns: pd.Series = self._get_excess_returns(asset_returnstream,cash_asset_portfolio)
        factor_returns, scaling_model = self._get_scaled_factor_returns(factors)
        common, loc_excess_returns, loc_factor_returns = get_intersected_indices(excess_returns,factor_returns)
        if loc_excess_returns.index.has_duplicates or loc_factor_returns.index.has_duplicates:
            logging.warning(" Encountered duplicate dates in returnstreams for asset or factors; attempting to resolve... ")
            if loc_excess_returns.shape[0]!=len(common):
                loc_excess_returns.drop_duplicates(inplace=True)
            else:
                loc_factor_returns.drop_duplicates(inplace=True)
            if all([date in loc_excess_returns.index for date in common]) \
                and all([date in loc_factor_returns.index for date in common]):
                logging.warning(" Duplicate dates issue resolved; will continue calculations... ")
            return return_list
        
        max_window_size_in_yrs = (max(common) - min(common))/np.timedelta64(1,'Y')
        window_sizes = ['lifetime']
        window_sizes.extend(self._get_list_of_window_sizes(max_window_size_in_yrs))
        estimationSingleStepFlag = False
        for window_size in window_sizes:
            # because window size is string ending in y char
            is_lifetime = window_size == 'lifetime'
            window_val = -1 if is_lifetime else int(window_size[:-1])*12 
            calculated_lambda = self._get_optimal_lambda(
                loc_factor_returns.values,
                loc_excess_returns.values,
                window_val
            )
            if is_lifetime:
                analysis_results: AnalysisObject = AnalysisObject(
                    loc_excess_returns,
                    loc_factor_returns,
                    frequency,
                    scaling_model,
                    window_size,
                    calculated_lambda
                )
                estimationSingleStepFlag = analysis_results.singlestep_flag
            else:
                analysis_results: AnalysisObject = \
                    AnalysisObject(
                        loc_excess_returns,
                        loc_factor_returns,
                        frequency,
                        scaling_model,
                        window_size,
                        calculated_lambda
                    ) if not estimationSingleStepFlag \
                    else AnalysisObject(loc_excess_returns, loc_factor_returns, frequency, scaling_model, window_size, -1)
            
            returnmat = np.array([[retn.value for retn in factor_rs] for factor_rs in factors])
            cashmat = np.array(
                [(retn.value / 100.0) + 1.0 for retn in cash_asset_model.returns])
            
            exposures = [analysis_results.exposures[factor.id] for factor in factors]
            coefs = np.array(exposures)
            constants = np.add(np.array(
                analysis_results.intercept * np.ones(len(returnmat[0]))), (cashmat-1)*100)
            fb_rets = (((np.dot(coefs, returnmat) + constants) / 100.0) + 1.0).tolist()

            return_list.append(
                {
                    'assetId': asset_returnstream.id,
                    'betas': [{
                        "factorId": factor.id,
                        "beta": analysis_results.exposures[factor.id]
                    } for factor in factors],
                    'factorBasedReturns': [{
                        "date": date,
                        "value": value
                    } for (date, value) in zip(time_labels, fb_rets)],
                    'riskModel': model_name,
                    'lambda': calculated_lambda,
                    'lifetime': is_lifetime,
                    'versionDate': asset_version_date,
                    'windowSize': window_val
                }
            )
        return return_list

    def _get_optimal_lambda(self, factor_returns:np.ndarray, portfolio_returns:np.ndarray, window_size:int=-1)->float:
        """
        get optimal lambda

        Args:
            factor_returns: factor returns
            portfolio_returns: portfolio returns
            window_size: the window size

        Returns:
            the optimal lambda
        """
        truncated_portfolios:np.ndarray = np.round(portfolio_returns,5)
        truncated_factors: np.ndarray = np.round(factor_returns,5)
        train_test_iterator = self._get_train_test_indices(truncated_portfolios.shape[0],window_size)
        SSE_at_breakpoints_per_fold = [
            self._get_SSE_at_breakpoints_given_fold(
                truncated_factors[id_tr_ts[0],:],truncated_portfolios[id_tr_ts[0]],
                truncated_factors[id_tr_ts[1],:],truncated_portfolios[id_tr_ts[1]]
            ) 
            for id_tr_ts in train_test_iterator
        ]
        total_length_of_training_folds = sum([len(fold[0]) for fold in train_test_iterator])
        SSE_at_breakpoints_per_fold = [
            {
                idx:data/total_length_of_training_folds for idx, data in fold.items()
            } for fold in SSE_at_breakpoints_per_fold
        ]
        avg_MSE_series = pd.Series(
            self._get_total_sse_across_folds(SSE_at_breakpoints_per_fold))/sum([len(fold[1]) for fold in train_test_iterator]
        )
        min_lambda = avg_MSE_series.index[np.argmin(avg_MSE_series)]
        if window_size==-1:
            max_allowed_lambda = max(self._get_adjusted_lars_path(truncated_factors, truncated_portfolios).keys())
            min_lambda = max(0,max_allowed_lambda-1e-3) if min_lambda>=max_allowed_lambda else min_lambda
        return min_lambda

    def _get_adjusted_lars_path(self,X:np.ndarray,y:np.ndarray)->Dict[float,List[int]]:
        """
        get adjusted lars path

        Args:
            x: the x input
            y: the y input

        Returns:
            expanding factor ids
        """
        y_demeaned = StandardScaler(with_std=False).fit_transform(y.reshape(-1,1)).reshape(-1)
        x_demeaned = StandardScaler(with_std=False).fit_transform(X)
        breakpoints, _, lasso_coefs = lars_path(x_demeaned,y_demeaned,method='lasso')
        expanding_factor_ids = {breakpoints[j]:list(np.where(lasso_coefs[:,j]!=0)[0]) for j in range(len(breakpoints))}
        expanding_factor_ids = {idx: data for idx, data in expanding_factor_ids.items() if len(data)<y.shape[0]-1}
        return expanding_factor_ids

    def _get_SSE_at_breakpoints_given_fold(self,
        x_insample:np.ndarray,
        y_insample:np.ndarray,
        x_oos:np.ndarray,
        y_oos:np.ndarray
    )->Dict[float,float]:
        """
        get total sse across folds 

        Args:
            x_insample: in sample x
            y_insample: in sample y
            x_oos: x out of sample
            y_oos: y out of sample

        Returns:
            sse at breakpoints given fold
        """
        expanding_factor_ids = self._get_adjusted_lars_path(x_insample, y_insample)
        breakpoints = list(expanding_factor_ids.keys())
        if len(set(breakpoints))!=len(breakpoints):
            logging.error('non-unique breakpoints were returned by lars_path; aborting analysis')
            raise ValueError("non-unique breakpoints were returned by lars_path; aborting analysis")
        if (not all(np.isfinite(breakpoints))) or any([val<0 for val in breakpoints]):
            logging.error('breakpoints contain nan/inf or negative values; aborting analysis')
            raise ValueError("breakpoints contain nan/inf or negative values; aborting analysis")
        ols_models = {j: LinearRegression(n_jobs=-1).fit(x_insample[:,expanding_factor_ids[j]],y_insample) for j in breakpoints[1:]}
        sse_by_breakpoint = {breakpoints[0]:np.linalg.norm(y_oos-np.mean(y_insample))**2*len(y_insample)}
        sse_by_breakpoint.update(
            {j:mean_squared_error(y_oos,ols_models[j].predict(x_oos[:,expanding_factor_ids[j]]))*len(y_oos)*len(y_insample) for j in breakpoints[1:]}
        )
        return sse_by_breakpoint

    def _get_total_sse_across_folds(self, sse_at_breakpoints_per_fold: List[Dict[float, float]])->Dict[float, float]:
        """
        get total sse across folds 

        Args:
            sse_at_breakpoints_per_fold: self-explanatory

        Returns:
            the total sse across folds
        """
        if len(sse_at_breakpoints_per_fold)==0:
            logging.warning("No folds were passed to 'get_total_SSE_across_folds'; will return an empty dictionary")
            return {}
        all_breakpoints = sorted([idx for fold in sse_at_breakpoints_per_fold for idx in fold.keys()])
        all_breakpoints = [0.5*(all_breakpoints[i]+all_breakpoints[i+1]) for i in range(len(all_breakpoints)-1)]
        all_mse = np.zeros((len(all_breakpoints),))
        for fold in sse_at_breakpoints_per_fold:    
            all_mse += self._get_stepwise_sse_function(all_breakpoints,np.array(list(fold.keys())),np.array(list(fold.values())))
        return dict(zip(all_breakpoints,all_mse))
    
    def _get_stepwise_sse_function(self, x_vector: np.ndarray, breakpoints: np.ndarray, mses: np.ndarray)->np.ndarray:
        """
        stepwise sse function

        Args:
            x_vector: the ndarray
            breakpoints: the breakpoints
            mses: the mean squared errors
        Returns:
            None
        """
        list_of_conditions = [x_vector>=breakpoints[0]]
        list_of_conditions.extend([((x_vector>=breakpoints[i]) & (x_vector<breakpoints[i-1])) for i in range(1,len(breakpoints))])
        list_of_conditions.extend([x_vector<breakpoints[-1]])
        return np.piecewise(x_vector,list_of_conditions,np.append(mses,np.inf))

    def _get_gmam_3_factor_analysis_results(
        self,
        asset_returnstream,
        factors,
        cash_asset,
        model_name: str,
        frequency: Frequency,
        asset_version_date: str
    ) -> List[Dict[str,Union[int,str,float,Dict[str,Union[Frequency,float]]]]]:
        """
        GMAM 3 factor analysis results

        Args:
            asset_returnstream,
            factors,
            cash_asset,
            model_name: str,
            frequency: Frequency,
            asset_version_date: str
        Returns:
            None
        """
        asset_id = asset_returnstream.id

        def get_log_returns(x: np.ndarray)->np.ndarray:
            return np.log(1+ x/100.0)
        
        def get_pct_returns(x: np.ndarray)->np.ndarray:
            return 100.0*(np.exp(x)-1)
        
        # this full_yF calculation is incorrect and will be depricated
        def get_full_yF()->pd.Series:

            eye_dT = np.eye(dT)
            n = int(P/dT)
            r = P%dT
            iota_matrix: np.ndarray = np.concatenate([eye_dT for _ in range(n)])
            if r!=0:
                iota_matrix: np.ndarray = np.concatenate([iota_matrix, eye_dT[:r,:]])
            betas = dict_of_aggregate_metrics['beta']['mean']
            phi = dict_of_aggregate_metrics['phi']['mean']
            phi_tilda = np.concatenate([phi,1 - iota_matrix.transpose().dot(phi).reshape(-1)])
            cFb = add_constant(get_log_returns(factor_returns.values)).dot(betas) + \
                get_log_returns(cash_asset.return_series.loc[factor_returns.index].values)
            return np.concatenate([
                np.roll(cFb,i).reshape(-1,1) for i in range(phi_tilda.shape[0])
            ],axis=1).dot(phi_tilda[::-1])[phi_tilda.shape[0]:]

        return_list = []
        dT = MONTHS_IN_QUARTER if frequency==Frequency.QUARTERLY else 1
        P = TOTAL_TERMS_IN_MA_PROCESS - dT
        asset_returns: pd.Series = asset_returnstream.return_series
        factor_returns: pd.DataFrame = get_clean_df_from_series([factor.return_series for factor in factors])
        factor_min_date = factor_returns.index.min()
        offset_months = MonthEnd(P+dT-1)
        pseudo_factor_min_date = factor_min_date + offset_months
        min_date: pd.Timestamp = max(asset_returns.index.min(),pseudo_factor_min_date)
        max_date: pd.Timestamp = min(asset_returns.index.max(),factor_returns.index.max())
        loc_asset_returns: pd.Series = asset_returns[(asset_returns.index>=min_date) & (asset_returns.index<=max_date)]
        loc_factor_returns: pd.DataFrame = factor_returns[
            (factor_returns.index>=loc_asset_returns.index.min()-offset_months) & \
            (factor_returns.index<=loc_asset_returns.index.max())
        ]
        if loc_asset_returns.index.has_duplicates:
            logging.warning(" Encountered duplicate dates in returnstreams for asset; resolving... ")
            loc_asset_returns.drop_duplicates(inplace=True)
        if loc_factor_returns.index.has_duplicates:
            logging.warning(" Encountered duplicate dates in returnstreams for factors; resolving... ")
            loc_factor_returns.drop_duplicates(inplace=True)
        if loc_asset_returns.shape[0]<MIN_DATAPOINTS_FORGMAM3:
            logging.info(
                'for asset: '+str(asset_id)+' no analysis will be performed for riskmodel '+str(id)+\
                    ' because '+str(MIN_DATAPOINTS_FORGMAM3)+' months/quarters of common history is unavailable at this time'
            )
            return return_list
        
        loc_cash_asset = cash_asset.return_series.loc[loc_factor_returns.index]
        # fix some of the hyper-parameters
        K = loc_factor_returns.shape[1]+1 # add one for constant term
        # here we need to specify which betas to set to 1, so currently we're identifying the factor by it's id below
        # TODO: change the way we set betas in the HyperParameter class
        # 47 is Equity Market and 61 is Equity Small Cap
        idx_of_betas_to_modify = [idx + 1 for idx, factor in enumerate(factors) if factor.id in ["47","61"]] 
        hyper_parameters : HyperParameter = HyperParameter(K,P, idx_of_betas_to_modify, True if frequency==Frequency.QUARTERLY else False)
        dict_of_parameters = DataGeneratingProcess(hyper_parameters,
                                               get_log_returns(loc_asset_returns.values).reshape(-1),
                                               add_constant(get_log_returns(loc_factor_returns.values)),
                                               get_log_returns(loc_cash_asset.values).reshape(-1))
        dict_of_parameters.update_parameters_randomly()
        dict_of_aggregate_metrics = dict_of_parameters.get_aggregate_metrics()
        if dict_of_aggregate_metrics and dict_of_parameters.convergence_flag:
            full_yF: np.ndarray = get_pct_returns(get_full_yF()) # get_pct_returns(
                                # dict_of_parameters.get_backcasted_returns(
                                #                 add_constant(get_log_returns(factor_returns.values)),
                                #                 get_log_returns(cash_asset.return_series.loc[factor_returns.index].values).reshape(-1)
                                #                                         )
                                #                 )# 
            time_index = factor_returns.index[-full_yF.shape[0]:].to_list()

            if frequency==Frequency.QUARTERLY:
                incorrect_yF = pd.Series(full_yF,index=time_index)
                correct_yF = pd.Series(0,index=pd.date_range(min(time_index),max(time_index),freq='Q-DEC')).resample('M').bfill()
                yf_qtrly = incorrect_yF.resample('Q-DEC').last()
                common_dates = [idx for idx in yf_qtrly.index if idx in correct_yF.index]
                correct_yF.loc[common_dates] = yf_qtrly.loc[common_dates]
                full_yF = correct_yF.values
                time_index = correct_yF.index.to_list()

            intercept = dict_of_aggregate_metrics['beta']['mean'][0]
            phi_list = list(dict_of_aggregate_metrics['phi']['mean'])
            gamma_list = list(dict_of_aggregate_metrics['gamma']['mean'])
            LOGGER.info(f"GMAM 3 parameters for asset: {asset_id}")
            LOGGER.info(f"Intercept for {asset_id}: {intercept}")
            LOGGER.info(f"Phi for {asset_id}: {phi_list}")
            LOGGER.info(f"Gamma for {asset_id}: {gamma_list}")
            return_list.append(
                {
                    'assetId': asset_id,
                    'betas': [{
                        "factorId": factor.id,
                        "beta": dict_of_aggregate_metrics['beta']['mean'][i+1]
                    } for i, factor in enumerate(factors)],
                    'factorBasedReturns': [{
                        "date": date.strftime(DEFAULT_DATE_FORMAT),
                        "value": value / 100.0 + 1.0 # scaling returns
                    } for (date, value) in zip(time_index, full_yF)],
                    'riskModel': model_name,
                    'lambda': -1,
                    'lifetime': True,
                    'versionDate': asset_version_date,
                    'windowSize': -1
                }
            )
        LOGGER.info(f"Asset Convergence Flag for {asset_id}: {dict_of_parameters.convergence_flag}")
        return [return_list, dict_of_parameters.convergence_flag]


class SIFactorAnalysisCalculator(object):
    """
    SI Factor Analysis Calculator for the overnight processes

    Args:
        si_returnstreams: si return streams
        underlier_returnstreams: underlier return streams
        underlier_factor_analysis_result: si underlier's factor analysis result
        si_id: si id
        underlier_id: underlier id
        model_name: risk model name
    Returns:
        None
    """

    def __init__(self,
                 si_returnstreams: ReturnStreamInformation,
                 underlier_returnstreams: ReturnStreamInformation,
                 underlier_factor_analysis_result,
                 si_id: str,
                 underlier_id: str,
                 model_name: str,
                 frequency: Frequency = Frequency.MONTHLY
                 ):
        self.si_returnstreams = si_returnstreams
        self.underlier_returnstreams = underlier_returnstreams
        self.underlier_factor_analysis_result = underlier_factor_analysis_result
        self.si_id = si_id
        self.underlier_id = underlier_id
        self.model_name = model_name
        self.frequency = frequency

    def get_si_gmam3_analysis_result_for_storage(self):
        """
        get factor analysis results for SI products based on GMAM 3.0 model

        Args:
            None
        Returns:
            factor analysis results including lambda, factor exposures, and betas
        """
        underlier_factor_exposures = self.underlier_factor_analysis_result["betas"]
        delta = self._get_scaling_factor_for_si()
        si_factor_analysis_result = []
        si_factor_exposures = []
        for beta_info in underlier_factor_exposures:
            si_factor_id = beta_info["factorId"]
            si_beta = beta_info["beta"] * delta
            si_factor_exposures.append({
                "factorId": si_factor_id,
                "beta": si_beta
            })
        si_factor_analysis_result.append(
            {
                'assetId': self.si_id,
                'betas': si_factor_exposures,
                'factorBasedReturns': None,
                'riskModel': GMAM_3_RISK_MODEL_NAME,
                'lambda': -1,
                'lifetime': True,
                'versionDate': str(pd.Timestamp.now().date()),
                'windowSize': -1
            }
        )
        return si_factor_analysis_result

    def _get_scaling_factor_for_si(self) -> float:
        """
        Perform regression for SI and its underlier to get Beta, which will be used for scaling

        Args:
            si_returnstreams: SI avarage rolling monthly returns
            underlier_returnstreams: underlier monthly returns
        Returns:
            delta: scaling factor
        """
        x_values = np.array([returns["value"] for returns in self.underlier_returnstreams["data"]]).reshape(-1, 1)
        y_values = np.array([returns["value"] for returns in self.si_returnstreams["data"]]).reshape(-1, 1)
        regr = LinearRegression()
        regr.fit(x_values, y_values)
        r_squared = round(regr.score(x_values, y_values), 4)
        logging.info('For SI: ' + str(self.si_id) + ', the R-Squared of its regression between SI returns and underlier returns is' + str(r_squared))
        delta = regr.coef_[0][0]
        logging.info('For SI: ' + str(self.si_id) + ', the delta between this SI and its underlier is' + str(delta))

        return delta
