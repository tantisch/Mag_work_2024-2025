import pandas as pd
import numpy as np

def load_data(filename: str) -> pd.DataFrame:
    """Load data from CSV file"""
    df = pd.read_csv(filename)
    return df

def pointpos(x: pd.Series) -> float:
    """Helper function to determine pivot point position for plotting
    From original implementation in indicator.py"""
    if x['isPivot'] == 2:  # Low pivot
        return x['close']
    elif x['isPivot'] == 1:  # High pivot
        return x['close']
    else:
        return np.nan

def calculate_true_range(df: pd.DataFrame) -> pd.Series:
    """
    Calculate True Range for ATR computation
    
    True Range is the greatest of:
    1. Current High - Current Low
    2. |Current High - Previous Close|
    3. |Current Low - Previous Close|
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR)
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing 'high', 'low', 'close' columns
    period : int, optional
        ATR calculation period, default is 14
        
    Returns:
    --------
    pd.Series
        ATR values
    """
    tr = calculate_true_range(df)
    atr = tr.rolling(window=period).mean()
    
    # Fill NaN values with first valid ATR value
    return atr.fillna(method='bfill')

def prepare_data_with_atr(df: pd.DataFrame, atr_period: int = 14) -> pd.DataFrame:
    """
    Prepare DataFrame with ATR calculations
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame with price data
    atr_period : int, optional
        Period for ATR calculation, default is 14
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with added ATR column
    """
    df = df.copy()
    df['atr'] = calculate_atr(df, atr_period)
    return df