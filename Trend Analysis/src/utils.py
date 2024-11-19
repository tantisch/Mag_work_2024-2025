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