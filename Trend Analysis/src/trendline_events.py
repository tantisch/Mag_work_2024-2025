from dataclasses import dataclass
from typing import List, Set, Tuple
import pandas as pd

@dataclass
class TrendlineEvents:
    touches: Set[int]
    breakouts: Set[int]
    throwbacks: Set[int]
    false_breakouts: Set[int]

def detect_events(line: Tuple[float, float, int],
                 df: pd.DataFrame,
                 high_pivots: List[int],
                 low_pivots: List[int],
                 is_support: bool,
                 margin: float = 10.0) -> TrendlineEvents:
    """
    Detect all events for a trendline 
    """
    slope, intercept, start_point = line
    
    events = TrendlineEvents(
        touches=set(),
        breakouts=set(),
        throwbacks=set(),
        false_breakouts=set()
    )
    
    # Find first touch after main pivot to establish the line
    pivot_indices = sorted(list(high_pivots | low_pivots))
    trendline_start = None
    
    for idx in pivot_indices:
        if idx <= start_point:
            continue
            
        price = df['close'].iloc[idx]
        line_value = slope * idx + intercept
        distance = price - line_value
        
        if abs(distance) <= margin:
            if (is_support and idx in low_pivots) or (not is_support and idx in high_pivots):
                events.touches.add(idx)
                trendline_start = idx
                break
    
    if trendline_start is None:
        return events
    
    # Track breakout state
    last_distance = 0
    in_breakout = False
    last_breakout = None
    breakout_candles = 0
    
    # Process all candles after first touch
    for idx in range(trendline_start + 1, len(df)):
        price = df['close'].iloc[idx]
        line_value = slope * idx + intercept
        distance = price - line_value
        
        is_high_pivot = idx in high_pivots
        is_low_pivot = idx in low_pivots
        
        if is_support:
            # Check for line crossing (breakout)
            if last_distance >= -margin and distance < -margin and not in_breakout:
                events.breakouts.add(idx)
                in_breakout = True
                last_breakout = idx
                breakout_candles = 0
            
            # During breakout
            elif distance < -margin and in_breakout:
                breakout_candles += 1
                
            # Check for return to line
            elif abs(distance) <= margin:
                if in_breakout:
                    if breakout_candles <= 3:# and distance <= 0:  # False breakout
                        events.false_breakouts.add(last_breakout)
                        events.breakouts.remove(last_breakout)
                    if is_high_pivot:  # Throwback
                        events.throwbacks.add(idx)
                    in_breakout = False
                    last_breakout = None
                elif is_low_pivot:  # Normal touch
                    events.touches.add(idx)
                    
        else:  # Resistance
            # Check for line crossing (breakout)
            if last_distance <= margin and distance > margin and not in_breakout:
                events.breakouts.add(idx)
                in_breakout = True
                last_breakout = idx
                breakout_candles = 0
            
            # During breakout
            elif distance > margin and in_breakout:
                breakout_candles += 1
                
            # Check for return to line
            elif abs(distance) <= margin:
                if in_breakout:
                    if breakout_candles <= 3:# and distance >= 0:  # False breakout
                        events.false_breakouts.add(last_breakout)
                        events.breakouts.remove(last_breakout)
                    if is_low_pivot:  # Throwback
                        events.throwbacks.add(idx)
                    in_breakout = False
                    last_breakout = None
                elif is_high_pivot:  # Normal touch
                    events.touches.add(idx)
        
        last_distance = distance
    
    return events

def calculate_trendline_score(events: TrendlineEvents) -> float:
    """
    Calculate score for a trendline based on its events.
    
    Scoring system:
    - Each touch: +2.0 points (very good)
    - Each throwback: +1.5 points (good)
    - Each breakout: +0.5 points (ok)
    - Each false breakout: -0.5 points (not good)
    """
    score = 0.0
    
    # Very good: touches
    score += len(events.touches) * 5.0
    
    # Good: throwbacks
    score += len(events.throwbacks) * 3
    
    # Ok: breakouts
    # score += len(events.breakouts) * 0.5
    
    # Not good: false breakouts
    score -= len(events.false_breakouts) * 2
    
    return score