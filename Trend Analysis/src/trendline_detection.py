import numpy as np
from typing import List, Tuple, Set, Optional
import pandas as pd
from scipy import stats
from enum import Enum
from dataclasses import dataclass
from .hough_transform import hough_transform_from_point
from .trendline_events import detect_events, TrendlineEvents, calculate_trendline_score, get_dynamic_margin
from .utils import calculate_atr

def simple_trendlines(pivot_points: List[int], df: pd.DataFrame, is_support: bool = True) -> List[Tuple[float, float, int, int]]:
    """simple trendline finder with dynamic margin"""
    valid_lines = []
    
    # Ensure ATR is calculated
    if not hasattr(df, 'atr'):
        df['atr'] = calculate_atr(df)
    
    for i in range(len(pivot_points) - 1):
        x1, x2 = pivot_points[i], pivot_points[i + 1]
        y1, y2 = df['close'].iloc[x1], df['close'].iloc[x2]
        
        # Calculate slope and intercept for the trendline
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        
        valid = True
        breakout_point = len(df)
        
        # Check future breakout points beyond the last pivot
        for k in range(x2, len(df)):
            y_line = slope * k + intercept
            margin = get_dynamic_margin(df, k)
            if is_support and df['close'].iloc[k] < y_line - margin:
                breakout_point = k
                break
            elif not is_support and df['close'].iloc[k] > y_line + margin:
                breakout_point = k
                break
        
        # Verify line validity between x1 and x2
        y_line = slope * np.array(range(x1, x2 + 1)) + intercept
        for k in range(x1, x2 + 1):
            margin = get_dynamic_margin(df, k)
            if is_support and df['close'].iloc[k] < y_line[k - x1] - margin:
                valid = False
                break
            elif not is_support and df['close'].iloc[k] > y_line[k - x1] + margin:
                valid = False
                break
        
        if valid:
            valid_lines.append((slope, intercept, x1, breakout_point))
    
    return valid_lines

def get_points_on_line(line: Tuple[float, float, int], 
                      pivot_points: List[int],
                      df: pd.DataFrame,
                      atr_multiplier: float = 0.5) -> List[int]:
    """
    Get consecutive valid touches of the line with dynamic margin
    """
    slope, intercept, start_point = line
    points_on_line = []
    
    sorted_pivots = sorted([p for p in pivot_points if p >= start_point])
    
    for pivot in sorted_pivots:
        y_line = slope * pivot + intercept
        margin = get_dynamic_margin(df, pivot, atr_multiplier)
        if abs(df['close'].iloc[pivot] - y_line) <= margin:
            points_on_line.append(pivot)
    
    return points_on_line

def is_line_valid_between_pivots(line: Tuple[float, float, int],
                                first_pivot: int,
                                second_pivot: int,
                                df: pd.DataFrame,
                                is_support: bool,
                                atr_multiplier: float = 0.5) -> bool:
    """
    Check if a line is valid between two pivot points by verifying that all price
    points respect the line's support/resistance nature within the ATR margin.
    """
    slope, intercept, _ = line
    
    # Check every point between the pivots
    for idx in range(first_pivot, second_pivot + 1):
        price = df['close'].iloc[idx]
        line_value = slope * idx + intercept
        margin = get_dynamic_margin(df, idx, atr_multiplier)
        distance = price - line_value  # Signed distance
        
        if is_support:
            # For support line, no price should be below the line by more than ATR margin
            if distance < -margin:
                return False
        else:
            # For resistance line, no price should be above the line by more than ATR margin
            if distance > margin:
                return False
    
    return True

def find_valid_line(main_point: Tuple[int, float], 
                    points_array: np.ndarray,
                    pivot_points: List[int],
                    high_pivots: Set[int],
                    low_pivots: Set[int],
                    df: pd.DataFrame,
                    is_support: bool,
                    atr_multiplier: float = 0.5) -> Optional[Tuple]:
    """
    Find a valid line from main point with dynamic margin
    """
    theta, rho, votes = hough_transform_from_point(main_point, points_array)
    
    if theta is not None:
        if abs(np.sin(theta)) > 1e-10:
            slope = -np.cos(theta) / np.sin(theta)
            intercept = main_point[1] - slope * main_point[0]
            start_point = int(main_point[0])
            line = (slope, intercept, start_point)
            
            supporting_points = get_points_on_line(line, pivot_points, df, atr_multiplier)
            
            if len(supporting_points) >= 2:
                first_two_valid = is_line_valid_between_pivots(
                    line, 
                    supporting_points[0],
                    supporting_points[1],
                    df,
                    is_support,
                    atr_multiplier
                )
                
                if first_two_valid:
                    return line, supporting_points
    
    return None

def hough_transform_trendlines(pivot_points: List[int], 
                             df: pd.DataFrame, 
                             is_support: bool = True,
                             high_pivots: List[int] = None, 
                             low_pivots: List[int] = None,
                             future_pivot_ranges: List[int] = [8, 20],
                             min_score: float = 5.0,
                             max_false_breakouts: int = 2,
                             atr_multiplier: float = 0.5) -> List[Tuple[float, float, int, TrendlineEvents]]:
    """
    Find valid lines for different ranges of future pivots with dynamic ATR-based margin
    """
    if not hasattr(df, 'atr'):
        df['atr'] = calculate_atr(df)
        
    high_pivots = set(high_pivots if high_pivots is not None else [])
    low_pivots = set(low_pivots if low_pivots is not None else [])
    
    all_pivot_indices = sorted(list(high_pivots | low_pivots))
    pivot_sequence = {idx: seq for seq, idx in enumerate(all_pivot_indices)}
    
    points = [(idx, df['close'].iloc[idx]) for idx in pivot_points]
    valid_lines = []
    
    # Process each main point for each future pivot range
    for i, main_point in enumerate(points):
        main_idx = main_point[0]
        main_seq = pivot_sequence[main_idx]
        
        for max_future_pivots in future_pivot_ranges:
            max_seq = main_seq + max_future_pivots
            future_indices = [idx for idx in all_pivot_indices 
                            if main_idx < idx and pivot_sequence[idx] <= max_seq]
            
            future_points = [(idx, df['close'].iloc[idx]) for idx in future_indices]
            if not future_points:
                continue
                
            points_array = np.array([(main_point[0], main_point[1])] + future_points)
            
            result = find_valid_line(
                main_point,
                points_array,
                pivot_points,
                high_pivots,
                low_pivots,
                df,
                is_support,
                atr_multiplier
            )
            
            if result is not None:
                line, supporting_points = result
                valid_lines.append((line, supporting_points, max_future_pivots))
    
    # Second phase: calculate events and scores
    scored_lines = []
    for line, supporting_points, range_used in valid_lines:
        events = detect_events(line, df, high_pivots, low_pivots, is_support, atr_multiplier)
        
        # Skip lines with too many false breakouts
        if len(events.false_breakouts) > max_false_breakouts:
            continue
            
        score = calculate_trendline_score(events)
        if score >= min_score:
            scored_lines.append((line, supporting_points, events, score, range_used))
    
    # Sort by start time and range
    scored_lines.sort(key=lambda x: x[0][2])
    
    # Filter out redundant lines
    final_lines = []
    lines_with_points = []  # Store tuples of (line, points_set)
    
    for line, points_on_line, events, score, _ in scored_lines:
        points_set = set(points_on_line)
        is_redundant = False
        
        # Check overlap with each existing line individually
        for existing_line, existing_points in lines_with_points:
            overlap_count = len(points_set & existing_points)
            if overlap_count >= 2:
                is_redundant = True
                break
        
        if not is_redundant:
            final_lines.append((line[0], line[1], line[2], events))
            lines_with_points.append((line, points_set))
    
    return final_lines