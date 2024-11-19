import pandas as pd
import numpy as np
from src.pivot_detection import get_pivot_points
from src.trendline_detection import simple_trendlines, hough_transform_trendlines
from src.visualization import plot_analysis
from src.hough_transform import hough_transform_from_point

def main():
    # Load data
    print("Loading data...")
    df = pd.read_csv('data.csv')
    
     # Get user choice for methods
    print("\nPivot Point Detection Method:")
    print("1: Fractal method (from indicator.py)")
    print("2: MinMax method (from plots3.py)")
    pivot_method = int(input("Enter choice (1 or 2): "))
    
    print("\nTrendline Detection Method:")
    print("1: Linear regression method (from plots3.py)")
    print("2: Hough transform method")
    trendline_method = int(input("Enter choice (1 or 2): "))
    
    # Detect pivot points
    high_pivots, low_pivots = get_pivot_points(df, method=pivot_method)
    
    # Calculate trendlines
    if trendline_method == 1:
        
        support_lines = simple_trendlines(low_pivots, df, is_support=True)
        resistance_lines = simple_trendlines(high_pivots, df, is_support=False)
    else:
        # Use the Hough transform method
        support_lines = hough_transform_trendlines(low_pivots, df, is_support=True, 
                                                 high_pivots=high_pivots, low_pivots=low_pivots)
        resistance_lines = hough_transform_trendlines(high_pivots, df, is_support=False,
                                                    high_pivots=high_pivots, low_pivots=low_pivots)
    
    # Plot results
    plot_analysis(df, high_pivots, low_pivots, support_lines, resistance_lines)

if __name__ == "__main__":
    main()