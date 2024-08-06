import numpy as np
import pandas as pd
from statsmodels.tools import add_constant
import multiprocessing as mp

import DataUtils as du
import MathUtils as mu
from PoolUtils import NestablePool
from Return import Frequency
from HyperParameter import HyperParameter
import DataGeneratingProcess as dgp

#LOCATION_OF_ASSET_RETURNS = '/Users/asheth/Library/CloudStorage/OneDrive-iCapitalNetwork/FactorPrediction/Kunpeng/historical_stacked.csv'
LOCATION_OF_ASSET_RETURNS = '/Users/asheth/Library/CloudStorage/OneDrive-iCapitalNetwork/GMAM/aniket_code/historical_stacked.csv'

PRINT_LOGS = True
FROM_IPI = True # False for test_assets
RUN_IN_PARALLEL = False # not implemented, dont change
WRITE_SANITY_CHECK_CSV = False # true for test assets
BETA_SIGNIFICANCE_LEVEL = 0.2
IS_ARES = False
IS_FSCREDIT = False

def do_gmam3_in_parallel(loc_factor_returns, loc_asset_returns):
    
    start_time = pd.Timestamp.now()
    backcasted_factors = full_factor_returns.loc[full_factor_returns.index<=
                                    loc_asset_returns.index.max()]
    backcasted_cash = full_cash_returns.loc[backcasted_factors.index]
    lowest_common_frequency, P = du.get_frequency_MAterms(loc_asset_returns)
    loc_cash_asset = full_cash_returns.loc[loc_factor_returns.index]
    K = loc_factor_returns.shape[1]+1
    is_qtrly = True if lowest_common_frequency==Frequency.QUARTERLY else False
    if is_qtrly:
        backcasted_index = backcasted_factors.index[::-3].copy()[::-1]
    else:
        backcasted_index = backcasted_factors.index.copy()
    hyper_parameters : HyperParameter = HyperParameter(K,P, betas_to_set_to_one, is_qtrly, 
                                                       'Aligned' in loc_asset_returns.name,
                                                       # IS_ARES, 
                                                       IS_FSCREDIT)
    dgp_object = dgp.DataGeneratingProcess(hyper_parameters,
                                           loc_asset_returns.values.reshape(-1),
                                           add_constant(loc_factor_returns.values),
                                           loc_cash_asset.values.reshape(-1))
    dgp_object.update_parameters_randomly()

    # dict_of_aggregate_metrics = dgp_object.get_unaggregated_data ()
    # dict_of_aggregate_metrics['beta'] = pd.DataFrame(dict_of_aggregate_metrics['beta'],
    #                                                                  index=factor_names)
    # dict_of_aggregate_metrics['gamma'] = pd.DataFrame(dict_of_aggregate_metrics['gamma'],
    #                                                                  index=factor_names)
    # dict_of_aggregate_metrics['x'] = pd.DataFrame(dict_of_aggregate_metrics['x'],
    #                                                   index=loc_factor_returns.index)
    # dict_of_aggregate_metrics['convergence_flag'] = dgp_object.convergence_flag
    # dict_of_aggregate_metrics['iterations'] = dgp_object.current_iteration
    # dict_of_aggregate_metrics['reliability_flag'] = \
    #                     dict_of_aggregate_metrics['reliability_indicator']['mean']>0.8 or \
    #                     any(mu.check_statistical_significance_from_percentiles(
    #                         dict_of_aggregate_metrics['beta'],BETA_SIGNIFICANCE_LEVEL)) # remove intercept from test, currently wrong
    # backcasted_returns = dgp_object.get_backcasted_returns(
    #             add_constant(backcasted_factors.values), backcasted_cash.values, is_qtrly)
    # dict_of_aggregate_metrics['backcasted_yF'] = pd.DataFrame(backcasted_returns['backcasted_yF'],
    #             index=backcasted_index[-backcasted_returns['backcasted_yF']['mean'].shape[0]:])
    # dict_of_aggregate_metrics['backcasted_xF'] = pd.DataFrame(backcasted_returns['backcasted_xF'],
    #                                     index = backcasted_factors.index)
    # dict_of_aggregate_metrics['annualized_rets'] = backcasted_returns['annualized_returns']
    # dict_of_aggregate_metrics['total_time'] = (pd.Timestamp.now() - start_time).total_seconds()
    # return dict_of_aggregate_metrics

    dict_of_aggregate_metrics = dgp_object.get_aggregate_metrics()
    dict_of_aggregate_metrics['beta'] = pd.DataFrame(dict_of_aggregate_metrics['beta'],
                                                                      index=factor_names)
    dict_of_aggregate_metrics['gamma'] = pd.DataFrame(dict_of_aggregate_metrics['gamma'],
                                                                      index=factor_names)
    dict_of_aggregate_metrics['x'] = pd.DataFrame(dict_of_aggregate_metrics['x'],
                                                      index=loc_factor_returns.index)
    dict_of_aggregate_metrics['convergence_flag'] = dgp_object.convergence_flag
    dict_of_aggregate_metrics['iterations'] = dgp_object.current_iteration
    dict_of_aggregate_metrics['reliability_flag'] = \
                        dict_of_aggregate_metrics['reliability_indicator']['mean']>0.8 or \
                        any(mu.check_statistical_significance_from_percentiles(
                            dict_of_aggregate_metrics['beta'],BETA_SIGNIFICANCE_LEVEL)) # remove intercept from test, currently wrong
    backcasted_returns = dgp_object.get_backcasted_returns(
                add_constant(backcasted_factors.values), backcasted_cash.values, is_qtrly)
    dict_of_aggregate_metrics['backcasted_yF'] = pd.DataFrame(backcasted_returns['backcasted_yF'],
                index=backcasted_index[-backcasted_returns['backcasted_yF']['mean'].shape[0]:])
    dict_of_aggregate_metrics['backcasted_xF'] = pd.DataFrame(backcasted_returns['backcasted_xF'],
                                        index = backcasted_factors.index)
    dict_of_aggregate_metrics['annualized_rets'] = backcasted_returns['annualized_returns']
    dict_of_aggregate_metrics['total_time'] = (pd.Timestamp.now() - start_time).total_seconds()
    return dict_of_aggregate_metrics

full_factor_returns, full_cash_returns = du.get_factors_and_cash()
full_asset_returns = du.get_asset_returns(LOCATION_OF_ASSET_RETURNS, FROM_IPI)
factor_name_id_map = du.get_factor_name_id_map()
factor_name_id_map = factor_name_id_map[factor_name_id_map.name!='Max Latent Factor']
factor_names = ['intercept']
factor_names.extend([colname for colname in full_factor_returns.columns])
betas_to_set_to_one = [idx+1 for idx in factor_name_id_map[
                        factor_name_id_map.name.isin(['Equity Market','Equity SmallCap'])].index]
intersected_input_tuples = {idx: du.get_intersected_returns(full_factor_returns, 
                            asset_returns.dropna()) for idx, asset_returns in 
                            full_asset_returns.to_dict(orient='series').items()}
analyzable_input_tuples = {idx: data for idx, data in intersected_input_tuples.items()
                           if du.check_asset_return_quality(data[1])}
small_assets = {idx: data for idx, data in analyzable_input_tuples.items() if data[0].shape[0]<=32}

if RUN_IN_PARALLEL:
    pool_jobs = NestablePool(mp.cpu_count())
    results = dict(zip(small_assets.keys(),pool_jobs.starmap(do_gmam3_in_parallel,[(data[0],data[1])
                                                for data in small_assets.values()])))
    pool_jobs.close()
    pool_jobs.join()
else:
    # pool_jobs = NestablePool(mp.cpu_count())
    results = {}
    unreliable_result_ids = []
    unconverged_ids = []
    errored_ids = []
    for idx, data in analyzable_input_tuples.items():
        if idx == 'Class D': #in results:
            continue
        try:
            if PRINT_LOGS:
                print(f"starting asset {idx}")
            results[idx] = do_gmam3_in_parallel(data[0], data[1])
            # results[idx] = []
            # for i in range(int(np.ceil(450/mp.cpu_count()))):
            #     results[idx].extend(pool_jobs.starmap(do_gmam3_in_parallel,[(data[0],data[1]) for _ in range(mp.cpu_count())]))
            #     if PRINT_LOGS:
            #         print(f"Completed {(i+1)*mp.cpu_count()} chains")
            if PRINT_LOGS:
                print(f"finished asset {idx} with convergence flag {results[idx]['convergence_flag']} and reliability flag {results[idx]['reliability_flag']} in {results[idx]['total_time']}s")
            if not results[idx]['reliability_flag']:
                unreliable_result_ids.append(idx)
            if not results[idx]['convergence_flag']:
                unconverged_ids.append(idx)
        except Exception as e:
            print(f"asset {idx} failed with exception: {e}")
            errored_ids.append(idx)
    # pool_jobs.close()
    # pool_jobs.join()
            
if WRITE_SANITY_CHECK_CSV:
    
    stacked_factors = pd.DataFrame.from_dict({idx: data['beta']['mean'] 
                                              for idx, data in results.items()}).stack().to_frame()
    stacked_factors.columns = ['mean_value']
    stacked_factors.index.set_names( ['variable_id','assetId'] , inplace = True)
    stacked_factors['std_dev_value'] = pd.DataFrame.from_dict({
        idx: data['beta']['standard_deviation'] for idx, data in results.items()}).stack()
    stacked_factors['variable'] = 'factor'
    stacked_factors.reset_index(inplace=True)
    
    stacked_backcasted_yF = pd.DataFrame.from_dict({idx: data['backcasted_yF']['mean']
                                              for idx, data in results.items()}).stack().to_frame()
    stacked_backcasted_yF.columns = ['mean_value']
    stacked_backcasted_yF.index.set_names( ['variable_id','assetId'] , inplace = True)
    stacked_backcasted_yF['std_dev_value'] = pd.DataFrame.from_dict({
        idx: data['backcasted_yF']['standard_deviation'] for idx, data in results.items()}).stack()
    stacked_backcasted_yF['variable'] = 'factor-based returns'
    stacked_backcasted_yF.reset_index(inplace=True)
    
    stacked_annualized_returns = pd.Series({idx: data['annualized_rets']['mean']
                                              for idx, data in results.items()},name = 'mean_value').to_frame()
    stacked_annualized_returns['std_dev_value'] = pd.Series({idx: data['annualized_rets']['standard_deviation'] 
                                                             for idx, data in results.items()})
    stacked_annualized_returns['variable_id'] = 'spliced values'
    stacked_annualized_returns['variable'] = 'annualized_returns'
    stacked_annualized_returns.index.name = 'assetId'
    stacked_annualized_returns.reset_index(inplace=True)
    
    stacked_R2 = pd.Series({idx: data['insample_R2']['mean']
                                              for idx, data in results.items()},name = 'mean_value').to_frame()
    stacked_R2['std_dev_value'] = pd.Series({idx: data['insample_R2']['standard_deviation'] 
                                                             for idx, data in results.items()})
    stacked_R2['variable_id'] = 'insample'
    stacked_R2['variable'] = 'R2'
    stacked_R2.index.name = 'assetId'
    stacked_R2.reset_index(inplace=True)
    
    stacked_results = pd.concat([stacked_R2,stacked_annualized_returns,stacked_backcasted_yF,stacked_factors],axis=0,ignore_index=True)
    stacked_results = stacked_results[~stacked_results.assetId.isin(unreliable_result_ids)]