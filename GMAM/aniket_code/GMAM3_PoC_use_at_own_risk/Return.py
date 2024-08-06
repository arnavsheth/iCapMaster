import pandas as pd
from enum import IntEnum

class Frequency(IntEnum):
    DAILY = 1
    WEEKLY = 2
    MONTHLY = 3
    QUARTERLY = 4


class ReturnStream(object):

    def __init__(self, returns_series: pd.Series, freq:Frequency, series_name:str = 'value'):
        if not isinstance(freq,Frequency):
            raise ValueError("Unrecognized or unsupported frequency provided")
        if returns_series.empty:
            self.returns_series: pd.Series = returns_series
            self.returns_series.name = series_name
            self.min_date: pd.Timestamp = None
            self.max_date: pd.Timestamp = None
            self.frequency: Frequency = freq
        else:
            if returns_series.isna().sum()>0:
                raise ValueError("Encountered NA values in ReturnStream")
            returns_series.sort_index(inplace=True)
            self.returns_series: pd.Series = returns_series
            self.returns_series.name = series_name
            self.min_date: pd.Timestamp = returns_series.index[0]
            self.max_date: pd.Timestamp = returns_series.index[-1]
            self.frequency: Frequency = freq