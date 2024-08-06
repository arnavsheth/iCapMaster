import pandas as pd
import numpy as np
from typing import Tuple, Union
from pandas.tseries.offsets import MonthEnd

from MathUtils import get_log_returns_from_pct_returns
from Return import Frequency

MONTHS_IN_QUARTER = 3
TOTAL_TERMS_IN_MA_PROCESS = 6
LOCATION_OF_FACTOR_RETURNS = '/Users/asheth/Library/CloudStorage/OneDrive-iCapitalNetwork/GMAM/aniket_code/factor_returns.csv'
LOCATION_OF_CASH_RETURNS = '/Users/asheth/Library/CloudStorage/OneDrive-iCapitalNetwork/GMAM/aniket_code/cash_returns.csv'
LOCATION_OF_FACTOR_NAME_ID_MAP = '/Users/asheth/Library/CloudStorage/OneDrive-iCapitalNetwork/GMAM/aniket_code/factor_name_id_map.csv'

def get_factors_and_cash()->Tuple[pd.DataFrame,pd.Series]:
    
    return read_from_csv(LOCATION_OF_FACTOR_RETURNS,'name').dropna(), \
           read_from_csv(LOCATION_OF_CASH_RETURNS).dropna()
    
def get_factor_name_id_map()->pd.DataFrame:
    
    return pd.read_csv(LOCATION_OF_FACTOR_NAME_ID_MAP)

def get_asset_returns(location_of_asset_returns: str, from_ipi: bool = True)->pd.DataFrame:
    
    if from_ipi:
        return read_from_csv(location_of_asset_returns,'asset')
    else:
        return get_asset_returns_from_rainbow(location_of_asset_returns)

def check_asset_return_quality(asset_returns: pd.Series)->Tuple[bool,str]:
        
    if not asset_returns.index.inferred_freq:
        return False,"unknown frequency"
    if any(asset_returns.apply(np.isinf)) or any(asset_returns.apply(np.isnan)):
        return False,"inf or nan in data"
    if asset_returns.shape[0]<8:
        return False, "insufficient data"
    return True,"good"

def get_frequency_MAterms(asset_returns: pd.Series)->Tuple[Frequency,int]:
    lowest_common_frequency = Frequency.MONTHLY if asset_returns.index.inferred_freq in ['M',
                                                                'BM'] else Frequency.QUARTERLY
    dT = MONTHS_IN_QUARTER if lowest_common_frequency==Frequency.QUARTERLY else 1
    P = TOTAL_TERMS_IN_MA_PROCESS - dT
    return lowest_common_frequency, P

def get_intersected_returns(factor_returns: pd.DataFrame, 
                            asset_returns: pd.Series)->Tuple[pd.DataFrame, pd.Series]:
    
    lowest_common_frequency, P = get_frequency_MAterms(asset_returns)
    factor_min_date = factor_returns.index.min()
    offset_months = MonthEnd(TOTAL_TERMS_IN_MA_PROCESS-1)
    pseudo_factor_min_date = factor_min_date + offset_months
    min_date: pd.Timestamp = max(asset_returns.index.min(),pseudo_factor_min_date)
    max_date: pd.Timestamp = min(asset_returns.index.max(),factor_returns.index.max())
    loc_asset_returns: pd.Series = asset_returns[(asset_returns.index>=min_date) & 
                                                 (asset_returns.index<=max_date)]
    loc_factor_returns: pd.DataFrame = factor_returns[
                    (factor_returns.index>=loc_asset_returns.index.min()-offset_months) & 
                                    (factor_returns.index<=loc_asset_returns.index.max())
                                                     ]
    
    return loc_factor_returns, loc_asset_returns

def get_asset_returns_from_rainbow(location_of_asset_returns: str)->pd.DataFrame:
    raw_returns = pd.read_csv(location_of_asset_returns).set_index('id')
    returns_dict = {idx: data.dropna() for idx, data in raw_returns.iterrows()}
    return pd.concat([pd.Series(index=pd.to_datetime(data[1:int((data.shape[0]-1)/2)+1].values, format='mixed'),
                           data=data[int((data.shape[0]-1)/2)+1:].astype(float).values,
                           name=idx) for idx, data in returns_dict.items()],
                     axis=1).applymap(get_log_returns_from_pct_returns)
    
def read_from_csv(location_of_csv: str, 
                  column_name: str = None)->Union[pd.DataFrame, pd.Series]:
    
    if column_name:
        df = pd.read_csv(location_of_csv).pivot(index='date',columns=column_name,
                                    values='value').applymap(get_log_returns_from_pct_returns)
        retval = df
    else:
        ds = pd.read_csv(LOCATION_OF_CASH_RETURNS).set_index('date'
                                                ).value.apply(get_log_returns_from_pct_returns)
        retval = ds
    retval.index = pd.to_datetime(retval.index, format='mixed')
    retval.sort_index(inplace=True)
    
    return retval