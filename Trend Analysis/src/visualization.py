import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List, Tuple, Union
from .trendline_events import TrendlineEvents

def plot_analysis(df: pd.DataFrame, 
                 high_pivots: np.ndarray, 
                 low_pivots: np.ndarray,
                 support_lines: List[Tuple], 
                 resistance_lines: List[Tuple]):
    """Plot price data with pivot points, trendlines and events"""
    plt.figure(figsize=(15, 7))
    
    # Plot price data
    plt.plot(df.index, df['close'], color='blue', alpha=0.5)
    
    # Plot pivot points
    plt.scatter(high_pivots, df['close'].iloc[high_pivots], 
               color='red', marker='^', label='High Pivots')
    plt.scatter(low_pivots, df['close'].iloc[low_pivots], 
               color='green', marker='v', label='Low Pivots')
    
    # Set y-axis limits based on price action with small padding
    price_range = df['close'].max() - df['close'].min()
    padding = price_range * 0.05
    plt.ylim(df['close'].min() - padding, df['close'].max() + padding)
    
    def plot_line_with_events(line, is_support=True):
        # print(type(line[3]))
        # if len(line) == 4:  # New method with events
        if isinstance(line[3], TrendlineEvents):
            slope, intercept, start_point, events = line
            color = 'green' if is_support else 'red'
            
            # Plot main line from start to end
            x_line = np.array(range(start_point, df.index[-1]))
            y_line = slope * x_line + intercept
            plt.plot(x_line, y_line, '--', color=color, alpha=0.8)
            
            # Plot touches
            if events.touches:
                y_touches = [slope * x + intercept for x in events.touches]
                plt.scatter(list(events.touches), y_touches, 
                          color=color, marker='o', s=100, alpha=0.5,
                          label='Touches' if is_support else None)
            
            # Plot breakouts
            if events.breakouts:
                y_breakouts = [slope * x + intercept for x in events.breakouts]
                plt.scatter(list(events.breakouts), y_breakouts, 
                          color=color, marker='x', s=100,
                          label='Breakouts' if is_support else None)
            
            # Plot throwbacks
            if events.throwbacks:
                y_throwbacks = [slope * x + intercept for x in events.throwbacks]
                plt.scatter(list(events.throwbacks), y_throwbacks, 
                          color=color, marker='s', s=100,
                          label='Throwbacks' if is_support else None)
            
            # Plot false breakouts
            if events.false_breakouts:
                y_false = [slope * x + intercept for x in events.false_breakouts]
                plt.scatter(list(events.false_breakouts), y_false, 
                          color=color, marker='d', s=100,
                          label='False Breakouts' if is_support else None)
                
        else:  # Original method
            slope, intercept, start_point, breakout = line
            color = 'green' if is_support else 'red'
            x_line = np.array(range(start_point, breakout))
            y_line = slope * x_line + intercept
            plt.plot(x_line, y_line, '--', color=color, alpha=0.8)
            
            # Mark breakout point if it exists and is not at the end
            if breakout < len(df):
                breakout_y = slope * breakout + intercept
                plt.scatter(breakout, breakout_y, color=color, s=100, 
                          facecolors='none', edgecolors=color, linewidth=2)
    
    # Plot support lines
    for line in support_lines:
        plot_line_with_events(line, is_support=True)
    
    # Plot resistance lines
    for line in resistance_lines:
        plot_line_with_events(line, is_support=False)
    
    plt.xlabel('Index')
    plt.ylabel('Price')
    plt.title('Price Analysis with Pivot Points, Trendlines and Events')
    plt.grid(True, alpha=0.3)
    
    # Add legend with unique entries
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    plt.tight_layout()
    plt.show()