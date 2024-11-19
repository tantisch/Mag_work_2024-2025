import numpy as np
from typing import List, Tuple, Set, Optional
import pandas as pd
from scipy import stats
from enum import Enum
from dataclasses import dataclass
from .hough_transform import hough_transform_from_point
from .trendline_events import detect_events, TrendlineEvents, calculate_trendline_score

def simple_trendlines(pivot_points: List[int], df: pd.DataFrame, is_support: bool = True) -> List[Tuple[float, float, int, int]]:
    """simple trendline finder"""
    valid_lines = []
    
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
            if is_support and df['close'].iloc[k] < y_line:
                breakout_point = k
                break
            elif not is_support and df['close'].iloc[k] > y_line:
                breakout_point = k
                break
        
        # Verify line validity between x1 and x2
        y_line = slope * np.array(range(x1, x2 + 1)) + intercept
        for k in range(x1, x2 + 1):
            if is_support and df['close'].iloc[k] < y_line[k - x1]:
                valid = False
                break
            elif not is_support and df['close'].iloc[k] > y_line[k - x1]:
                valid = False
                break
        
        if valid:
            valid_lines.append((slope, intercept, x1, breakout_point))
    
    return valid_lines

def get_points_on_line(line: Tuple[float, float, int], 
                      pivot_points: List[int],
                      df: pd.DataFrame,
                      margin: float = 5.0) -> List[int]:
    """
    Get consecutive valid touches of the line.
    Returns only points that form a valid sequence.
    """
    slope, intercept, start_point = line
    points_on_line = []
    
    sorted_pivots = sorted([p for p in pivot_points if p >= start_point])
    
    for pivot in sorted_pivots:
        y_line = slope * pivot + intercept
        if abs(df['close'].iloc[pivot] - y_line) <= margin:
            points_on_line.append(pivot)
    
    return points_on_line

def is_line_valid_between_pivots(line: Tuple[float, float, int],
                                first_pivot: int,
                                second_pivot: int,
                                high_pivots: Set[int],
                                low_pivots: Set[int],
                                df: pd.DataFrame,
                                is_support: bool) -> bool:
    """
    Check if there are any violating pivots between the two pivots that establish the line.
    """
    slope, intercept, _ = line
    
    all_pivots = high_pivots | low_pivots
    between_pivots = [p for p in all_pivots 
                     if first_pivot < p < second_pivot]
    
    for pivot_idx in between_pivots:
        price = df['close'].iloc[pivot_idx]
        line_value = slope * pivot_idx + intercept
        
        if is_support:
            # For support, check if any low pivot is below the line
            if pivot_idx in low_pivots and price < line_value:
                return False
        else:
            # For resistance, check if any high pivot is above the line
            if pivot_idx in high_pivots and price > line_value:
                return False
    
    return True

def find_valid_line(main_point: Tuple[int, float], 
                    points_array: np.ndarray,
                    pivot_points: List[int],
                    high_pivots: Set[int],
                    low_pivots: Set[int],
                    df: pd.DataFrame,
                    is_support: bool) -> Optional[Tuple]:
    """
    Find a valid line from main point (only checking for intersections)
    """
    theta, rho, votes = hough_transform_from_point(main_point, points_array)
    
    if theta is not None:
        if abs(np.sin(theta)) > 1e-10:
            slope = -np.cos(theta) / np.sin(theta)
            intercept = main_point[1] - slope * main_point[0]
            start_point = int(main_point[0])
            line = (slope, intercept, start_point)
            
            # Get points on line
            supporting_points = get_points_on_line(line, pivot_points, df)
            
            if len(supporting_points) >= 2:
                # Only check for intersections, not score
                first_two_valid = is_line_valid_between_pivots(
                    line, 
                    supporting_points[0],
                    supporting_points[1],
                    high_pivots,
                    low_pivots,
                    df,
                    is_support
                )
                
                if first_two_valid:
                    return line, supporting_points
    
    return None

def hough_transform_trendlines(pivot_points: List[int], 
                             df: pd.DataFrame, 
                             is_support: bool = True,
                             high_pivots: List[int] = None, 
                             low_pivots: List[int] = None,
                             max_future_pivots: int = 10,
                             min_score: float = 3.0) -> List[Tuple[float, float, int, TrendlineEvents]]:
    """
    Hough tranform method for trendlines
    """
    high_pivots = set(high_pivots if high_pivots is not None else [])
    low_pivots = set(low_pivots if low_pivots is not None else [])
    
    all_pivot_indices = sorted(list(high_pivots | low_pivots))
    pivot_sequence = {idx: seq for seq, idx in enumerate(all_pivot_indices)}
    
    points = [(idx, df['close'].iloc[idx]) for idx in pivot_points]
    valid_lines = []
    
    # First phase: find all valid lines (no intersections)
    for i, main_point in enumerate(points):
        main_idx = main_point[0]
        main_seq = pivot_sequence[main_idx]
        
        max_seq = main_seq + max_future_pivots
        future_indices = [idx for idx in all_pivot_indices 
                         if main_idx < idx and pivot_sequence[idx] <= max_seq]
        
        future_points = [(idx, df['close'].iloc[idx]) for idx in future_indices]
        if not future_points:
            continue
            
        points_array = np.array([(main_point[0], main_point[1])] + future_points)
        
        # Try to find a valid line
        result = find_valid_line(
            main_point,
            points_array,
            pivot_points,
            high_pivots,
            low_pivots,
            df,
            is_support
        )
        
        if result is not None:
            line, supporting_points = result
            valid_lines.append((line, supporting_points))
    
    # Second phase: calculate events and scores for valid lines
    scored_lines = []
    for line, supporting_points in valid_lines:
        events = detect_events(line, df, high_pivots, low_pivots, is_support)
        score = calculate_trendline_score(events)
        if score >= min_score:
            scored_lines.append((line, supporting_points, events, score))
    
    # Sort by start time
    scored_lines.sort(key=lambda x: x[0][2])
    
    # Filter out redundant lines
    final_lines = []
    used_points = set()
    
    for line, points_on_line, events, score in scored_lines:
        unique_points = set(points_on_line) - used_points
        if len(unique_points) >= 2:
            final_lines.append((line[0], line[1], line[2], events))
            used_points.update(points_on_line)
    
    return final_lines