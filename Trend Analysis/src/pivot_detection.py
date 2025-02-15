
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

def get_pivot_points(df: pd.DataFrame, window: int = 5) -> tuple:
    """
    Get pivot points using argrelextrema method.
    window: size of the window to look for pivot points (equivalent to order parameter)
    """
    high_idx = argrelextrema(df['close'].values, np.greater, order=window)[0]
    low_idx = argrelextrema(df['close'].values, np.less, order=window)[0]
    
    return high_idx, low_idx

