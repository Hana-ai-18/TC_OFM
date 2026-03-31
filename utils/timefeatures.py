"""utils/timefeatures.py - Time feature extraction"""
import numpy as np
import pandas as pd
from typing import List
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset


class TimeFeature:
    def __init__(self):
        pass
    
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass
    
    def __repr__(self):
        return self.__class__.__name__ + "()"


class SecondOfMinute(TimeFeature):
    """Second of minute normalized to [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5


class MinuteOfHour(TimeFeature):
    """Minute of hour normalized to [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5


class HourOfDay(TimeFeature):
    """Hour of day normalized to [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5


class DayOfWeek(TimeFeature):
    """Day of week normalized to [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    """Day of month normalized to [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    """Day of year normalized to [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5


class MonthOfYear(TimeFeature):
    """Month of year normalized to [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
    """Week of year normalized to [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.isocalendar().week.values.astype(np.float32) - 1) / 52.0 - 0.5


def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    Return list of time features based on frequency
    For TC: typically '6H' (6 hours)
    """
    features_by_offsets = {
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Minute: [MinuteOfHour, HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Second: [SecondOfMinute, MinuteOfHour, HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
    }
    
    offset = to_offset(freq_str)
    
    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]
    
    raise RuntimeError(f"Frequency {freq_str} not supported.")


def time_features(dates, timeenc=1, freq='h'):
    """
    Main function to extract time features
    - timeenc=0: Return integers (for Embedding layer)
    - timeenc=1: Return normalized floats [-0.5, 0.5]
    """
    if timeenc == 0:
        df = pd.DataFrame({'date': dates})
        df['date'] = pd.to_datetime(df['date'])
        
        df['month'] = df.date.dt.month
        df['day'] = df.date.dt.day
        df['weekday'] = df.date.dt.weekday
        df['hour'] = df.date.dt.hour
        df['minute'] = df.date.dt.minute // 15
        
        freq_map = {
            'y': [],
            'm': ['month'],
            'w': ['month'],
            'd': ['month', 'day', 'weekday'],
            'b': ['month', 'day', 'weekday'],
            'h': ['month', 'day', 'weekday', 'hour'],
            't': ['month', 'day', 'weekday', 'hour', 'minute'],
        }
        return df[freq_map[freq.lower()]].values
    
    if timeenc == 1:
        if isinstance(dates, pd.DataFrame):
            index = pd.to_datetime(dates.iloc[:, 0].values)
        else:
            index = pd.to_datetime(dates)
        
        index = pd.DatetimeIndex(index)
        return np.vstack([feat(index) for feat in time_features_from_frequency_str(freq)]).transpose(1, 0)