
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

def isPivot(candle: int, df: pd.DataFrame) -> int:

    window = 3  # Fixed window size from original implementation
    
    if candle - window < 0 or candle + window >= len(df):
        return 0
    
    pivotHigh = 1
    pivotLow = 2
    
    for i in range(candle - window, candle + window + 1):
        if df.iloc[candle].close > df.iloc[i].close:
            pivotLow = 0
        if df.iloc[candle].close < df.iloc[i].close:
            pivotHigh = 0
            
    if pivotHigh and pivotLow:
        return 3
    elif pivotHigh:
        return pivotHigh
    elif pivotLow:
        return pivotLow
    else:
        return 0

def find_pivot_points_minmax(df: pd.DataFrame) -> tuple: # for not no real difference in these two method. will be changed in the future

    order = 3  # Fixed order from original implementation
    
    high_idx = argrelextrema(df['close'].values, np.greater, order=order)[0]
    low_idx = argrelextrema(df['close'].values, np.less, order=order)[0]
    
    return high_idx, low_idx

def get_pivot_points(df: pd.DataFrame, method: int = 1) -> tuple:
    """Get pivot points using selected method"""
    if method == 1:
        df['isPivot'] = df.apply(lambda x: isPivot(x.name, df), axis=1)
        high_pivots = df[df['isPivot'] == 1].index
        low_pivots = df[df['isPivot'] == 2].index
    else:
        high_pivots, low_pivots = find_pivot_points_minmax(df)
    
    return high_pivots, low_pivots