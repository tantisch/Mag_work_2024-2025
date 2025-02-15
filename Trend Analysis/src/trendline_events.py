from dataclasses import dataclass
from typing import List, Set, Tuple
import pandas as pd
import numpy as np
from .utils import calculate_atr

@dataclass
class TrendlineEvents:
    touches: Set[int]
    breakouts: Set[int]
    throwbacks: Set[int]
    false_breakouts: Set[int]

def get_dynamic_margin(df: pd.DataFrame, index: int, atr_multiplier: float = 0.5) -> float:
    """Calculate dynamic margin based on ATR at a given index"""
    if not hasattr(df, 'atr'):
        df['atr'] = calculate_atr(df)
    return df['atr'].iloc[index] * atr_multiplier

def detect_events(line: Tuple[float, float, int],
                 df: pd.DataFrame,
                 high_pivots: List[int],
                 low_pivots: List[int],
                 is_support: bool,
                 atr_multiplier: float = 0.5) -> TrendlineEvents:
    """
    Detect trendline events according to the following rules:
    1. Touch: Pivot point formed within margin of trendline
       - Support line: Low pivot within margin
       - Resistance line: High pivot within margin
       
    2. Breakout: 
       - Resistance line: Break above line + low pivot above/within margin
       - Support line: Break below line + high pivot below/within margin
       
    3. False Breakout:
       - Resistance line: Break above line + low pivot below line
       - Support line: Break below line + high pivot above line
       
    4. Throwback: Price returns to line after valid breakout
       - Support line: High pivot within margin
       - Resistance line: Low pivot within margin
    """
    slope, intercept, start_point = line
    
    # Ensure ATR is calculated
    if not hasattr(df, 'atr'):
        df['atr'] = calculate_atr(df)
    
    events = TrendlineEvents(
        touches=set(),
        breakouts=set(),
        throwbacks=set(),
        false_breakouts=set()
    )
    
    # Find first touch to establish the line
    first_touch = None
    for idx in range(start_point, len(df)):
        price = df['close'].iloc[idx]
        line_value = slope * idx + intercept
        margin = get_dynamic_margin(df, idx, atr_multiplier)
        distance = price - line_value
        
        if abs(distance) <= margin:
            if (is_support and idx in low_pivots) or (not is_support and idx in high_pivots):
                events.touches.add(idx)
                first_touch = idx
                break
    
    if first_touch is None:
        return events

    # Track state
    in_breakout = False
    potential_breakout = None
    had_valid_breakout = False
    waiting_for_pivot = False
    
    # Process all candles after first touch
    for idx in range(first_touch + 1, len(df)):
        price = df['close'].iloc[idx]
        line_value = slope * idx + intercept
        margin = get_dynamic_margin(df, idx, atr_multiplier)
        distance = price - line_value
        
        is_high_pivot = idx in high_pivots
        is_low_pivot = idx in low_pivots
        
        # Within margin of line
        if abs(distance) <= margin:
            # if not waiting_for_pivot:
            if is_support:
                if is_low_pivot:# and not had_valid_breakout:
                    # Touch: Low pivot within margin (support)
                    events.touches.add(idx)
                elif is_high_pivot:# and had_valid_breakout:
                    # Throwback: High pivot within margin after breakout (support)
                    events.throwbacks.add(idx)
            else:  # Resistance
                if is_high_pivot:# and not had_valid_breakout:
                    # Touch: High pivot within margin (resistance)
                    events.touches.add(idx)
                elif is_low_pivot:# and had_valid_breakout:
                    # Throwback: Low pivot within margin after breakout (resistance)
                    events.throwbacks.add(idx)
                        
        # Beyond margin
        elif (is_support and distance < -margin) or (not is_support and distance > margin):
            if not in_breakout and not waiting_for_pivot:
                potential_breakout = idx
                waiting_for_pivot = True
                in_breakout = True
                
        # Check for pivot confirmation after potential breakout
        if waiting_for_pivot:
            if is_support:
                if is_high_pivot:
                    # Check if high pivot is below or within margin
                    pivot_distance = price - (slope * idx + intercept)
                    if pivot_distance <= margin:
                        # Valid breakout: high pivot below/within margin
                        events.breakouts.add(potential_breakout)
                        had_valid_breakout = True
                    else:
                        # False breakout: high pivot above line
                        events.false_breakouts.add(potential_breakout)
                    waiting_for_pivot = False
                    potential_breakout = None
            else:  # Resistance
                if is_low_pivot:
                    # Check if low pivot is above or within margin
                    pivot_distance = price - (slope * idx + intercept)
                    if pivot_distance >= -margin:
                        # Valid breakout: low pivot above/within margin
                        events.breakouts.add(potential_breakout)
                        had_valid_breakout = True
                    else:
                        # False breakout: low pivot below line
                        events.false_breakouts.add(potential_breakout)
                    waiting_for_pivot = False
                    potential_breakout = None
    
    # Handle any remaining potential breakout at end of data
    if potential_breakout is not None:
        events.false_breakouts.add(potential_breakout)
    
    return events

def calculate_trendline_score(events: TrendlineEvents) -> float:
    """
    Calculate score for a trendline based on its events.
    
    Scoring system:
    - Each touch: +5.0 points (very good)
    - Each throwback: +3.0 points (good)
    - Each false breakout: -2.0 points (not good)
    """
    score = 0.0
    score += len(events.touches) * 5.0
    score += len(events.throwbacks) * 3.0
    score -= len(events.false_breakouts) * 2.0
    return score