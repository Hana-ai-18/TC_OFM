"""
utils/timefeatures.py  ── v10-fixed
======================================
Time feature extraction for TC sequence data.

FIXES vs original:
  1. WeekOfYear: .isocalendar().week returns a pandas Series on newer
     pandas (>= 1.1).  Calling .values then .astype(np.float32) raises
     AttributeError on pandas >= 2.0 because the Series index is an
     IntegerIndex, not a plain ndarray index.  Fixed to use explicit
     .to_numpy(dtype=np.float32) which works on pandas >= 0.25.
  2. DayOfYear: normalization was (dayofyear - 1) / 365.0.  For
     years with 366 days (leap years) the maximum value is 366/365 > 1,
     breaking the [-0.5, 0.5] contract.  Fixed to divide by 366.0.
  3. time_features_from_frequency_str: '6H'/'6h' (typical TC interval)
     was not matched by the offsets.Hour branch because pandas >= 2.2
     changed the canonical offset for '6H' from Hour to a multiplied
     form.  Added explicit string-prefix fallback for unmatched offsets.
  4. time_features (timeenc=0): 'h'/'6h'/'6H' all collapsed to 'h'
     correctly but the function returned empty for unknown freq strings.
     Added an explicit 'h' fallback so TC use-case never raises.
  5. time_features (timeenc=1): bare list input (not DataFrame) was
     passed directly to pd.to_datetime which works but emitted a
     FutureWarning on pandas >= 2.0.  Wrapped in explicit list coercion.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset


class TimeFeature:
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"


class SecondOfMinute(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5


class MinuteOfHour(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5


class HourOfDay(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5


class DayOfWeek(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    """FIX: divide by 366 (not 365) to keep values in [-0.5, 0.5] for leap years."""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 366.0 - 0.5


class MonthOfYear(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
    """FIX: use .to_numpy() instead of .values.astype() for pandas >= 2.0 compat."""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        iso = index.isocalendar().week
        # pandas >= 2.0: isocalendar() returns an IntegerIndex Series
        return iso.to_numpy(dtype=np.float32) / 52.0 - 0.5


def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    Return list of time features based on a pandas frequency string.

    Typical TC interval is '6H' or '6h'.

    FIX: pandas >= 2.2 changed how multiplied offsets (like '6H') are
    represented internally.  Added a string-prefix fallback so '6H',
    '12H', etc. all resolve correctly to the Hour feature list.
    """
    features_by_offsets = {
        offsets.YearEnd:      [],
        offsets.QuarterEnd:   [MonthOfYear],
        offsets.MonthEnd:     [MonthOfYear],
        offsets.Week:         [DayOfMonth, WeekOfYear],
        offsets.Day:          [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay:  [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour:         [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Minute:       [MinuteOfHour, HourOfDay,
                               DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Second:       [SecondOfMinute, MinuteOfHour, HourOfDay,
                               DayOfWeek, DayOfMonth, DayOfYear],
    }

    offset = to_offset(freq_str)

    # Primary match
    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

    # FIX: fallback via string prefix for multiplied offsets (e.g. '6H', '12T')
    freq_upper = freq_str.upper().lstrip("0123456789")
    prefix_map = {
        "H":  [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        "T":  [MinuteOfHour, HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        "MIN":[MinuteOfHour, HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        "S":  [SecondOfMinute, MinuteOfHour, HourOfDay,
               DayOfWeek, DayOfMonth, DayOfYear],
        "D":  [DayOfWeek, DayOfMonth, DayOfYear],
        "W":  [DayOfMonth, WeekOfYear],
        "M":  [MonthOfYear],
        "Q":  [MonthOfYear],
        "A":  [],
        "Y":  [],
    }
    if freq_upper in prefix_map:
        return [cls() for cls in prefix_map[freq_upper]]

    raise RuntimeError(
        f"Frequency '{freq_str}' not supported. "
        "Supported: Y, Q, M, W, D, B, H, T/min, S (with optional integer prefix)."
    )


def time_features(dates, timeenc: int = 1, freq: str = "h") -> np.ndarray:
    """
    Extract time features from a list of datetime-like values.

    timeenc=0 : integer columns for an Embedding lookup
    timeenc=1 : float columns normalised to [-0.5, 0.5]

    FIX (timeenc=1): coerce input to list before pd.to_datetime to
    silence FutureWarning on pandas >= 2.0 for non-list iterables.
    FIX (timeenc=0): unknown freq strings fall back to 'h' columns
    instead of raising KeyError.
    """
    if timeenc == 0:
        df = pd.DataFrame({"date": pd.to_datetime(list(dates))})

        freq_map = {
            "y":  [],
            "m":  ["month"],
            "w":  ["month"],
            "d":  ["month", "day", "weekday"],
            "b":  ["month", "day", "weekday"],
            "h":  ["month", "day", "weekday", "hour"],
            "t":  ["month", "day", "weekday", "hour", "minute"],
            "s":  ["month", "day", "weekday", "hour", "minute"],
        }
        df["month"]   = df.date.dt.month
        df["day"]     = df.date.dt.day
        df["weekday"] = df.date.dt.weekday
        df["hour"]    = df.date.dt.hour
        df["minute"]  = df.date.dt.minute // 15

        # FIX: normalise freq key; fall back to 'h' for unknown strings
        freq_key = freq.lower().lstrip("0123456789")
        cols = freq_map.get(freq_key, freq_map["h"])
        return df[cols].values if cols else np.zeros((len(df), 0))

    # timeenc == 1
    index = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    feats = time_features_from_frequency_str(freq)
    return np.vstack([f(index) for f in feats]).transpose(1, 0)