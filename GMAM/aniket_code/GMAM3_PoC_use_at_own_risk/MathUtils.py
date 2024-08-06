import pandas as pd
import numpy as np
from typing import List, Union, Dict

def get_log_returns_from_pct_returns(x: np.ndarray)->np.ndarray:
    return np.log(1+ x/100.0)

def get_pct_returns_from_log_returns(x: np.ndarray)->np.ndarray:
    return 100.0*(np.exp(x)-1)

def check_statistical_significance_from_values(parameter_values: np.ndarray, 
                            alpha: float = 0.05)->Union[List[bool],bool]:
    
    percentile_values = [100*(alpha/2), 100*(1-alpha/2)]
    return np.prod(np.percentile(parameter_values, percentile_values, axis=parameter_values.ndim-1),axis=1)>0

def check_statistical_significance_from_percentiles(
        parameter_percentile_dict: Dict[str,Union[float, np.ndarray]], 
        alpha: float = 0.05)->Union[List[bool],bool]:
    
    percentile_values = [str(100*(alpha/2)), str(100*(1-alpha/2))]
    return np.prod([parameter_percentile_dict[val] for val in percentile_values],axis=0)>0