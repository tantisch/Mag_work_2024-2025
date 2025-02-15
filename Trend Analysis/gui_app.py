import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
from src.pivot_detection import get_pivot_points
from src.trendline_detection import simple_trendlines, hough_transform_trendlines
from src.visualization import plot_analysis
from src.utils import prepare_data_with_atr
import os

class TrendAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Trend Analysis Tool")
        self.root.geometry("500x550")  # Increased height for new controls
        
        # Configure style
        self.style = ttk.Style()
        self.style.configure('Modern.TFrame', background='#f0f0f0')
        self.style.configure('Modern.TLabel', background='#f0f0f0', font=('Helvetica', 10))
        self.style.configure('Modern.TRadiobutton', background='#f0f0f0', font=('Helvetica', 10))
        self.style.configure('Modern.TButton', font=('Helvetica', 10))
        
        self.create_widgets()
    
    def create_tooltip(self, widget, text):
        """Create a tooltip for a given widget"""
        def show_tooltip(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            
            label = ttk.Label(tooltip, text=text, justify=tk.LEFT,
                            relief='solid', borderwidth=1,
                            padding=5, background="#ffffe0")
            label.pack()
            
            def hide_tooltip():
                tooltip.destroy()
                
            label.bind('<Leave>', lambda e: hide_tooltip())
            widget.bind('<Leave>', lambda e: hide_tooltip())
            
        widget.bind('<Enter>', show_tooltip)

    def browse_file(self):
        """Open file dialog to select CSV file"""
        filename = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.file_path.set(filename)
        
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20", style='Modern.TFrame')
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # File selection
        file_frame = ttk.LabelFrame(main_frame, text="Data File", padding="10")
        file_frame.grid(row=0, column=0, columnspan=2, pady=(0, 20), sticky=(tk.W, tk.E))
        
        self.file_path = tk.StringVar()
        file_entry = ttk.Entry(file_frame, textvariable=self.file_path, width=50)
        file_entry.grid(row=0, column=0, padx=(0, 10))
        
        browse_button = ttk.Button(file_frame, text="Browse", command=self.browse_file)
        browse_button.grid(row=0, column=1)
        
        # Methods frame
        methods_frame = ttk.Frame(main_frame)
        methods_frame.grid(row=1, column=0, columnspan=2, pady=(0, 20), sticky=(tk.W, tk.E))
        
        # Pivot detection parameters
        pivot_frame = ttk.LabelFrame(methods_frame, text="Pivot Point Parameters", padding="10")
        pivot_frame.grid(row=0, column=0, padx=(0, 10), sticky=(tk.W, tk.E, tk.N, tk.S))
        
        ttk.Label(pivot_frame, text="Window Size:").grid(row=0, column=0, padx=(0, 10))
        self.window_size = tk.StringVar(value="5")
        ttk.Entry(pivot_frame, textvariable=self.window_size, width=5).grid(row=0, column=1)
        
        # Trendline detection method
        trendline_frame = ttk.LabelFrame(methods_frame, text="Trendline Detection Method", padding="10")
        trendline_frame.grid(row=0, column=1, padx=(10, 0), sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.trendline_method = tk.IntVar(value=1)
        ttk.Radiobutton(trendline_frame, text="Linear Regression", variable=self.trendline_method, 
                       value=1, style='Modern.TRadiobutton').grid(row=0, column=0, sticky=tk.W)
        ttk.Radiobutton(trendline_frame, text="Hough Transform", variable=self.trendline_method, 
                       value=2, style='Modern.TRadiobutton').grid(row=1, column=0, sticky=tk.W)
        
        # ATR Settings frame
        atr_frame = ttk.LabelFrame(main_frame, text="ATR Settings", padding="10")
        atr_frame.grid(row=2, column=0, columnspan=2, pady=(0, 20), sticky=(tk.W, tk.E))
        
        # ATR Period
        ttk.Label(atr_frame, text="ATR Period:").grid(row=0, column=0, padx=(0, 10))
        self.atr_period = tk.StringVar(value="14")
        ttk.Entry(atr_frame, textvariable=self.atr_period, width=5).grid(row=0, column=1)
        
        # ATR Multiplier
        ttk.Label(atr_frame, text="ATR Multiplier:").grid(row=0, column=2, padx=(20, 10))
        self.atr_multiplier = tk.StringVar(value="0.5")
        ttk.Entry(atr_frame, textvariable=self.atr_multiplier, width=5).grid(row=0, column=3)
        
        # Add tooltip for ATR settings
        atr_tooltip = ("ATR Period: Number of periods for ATR calculation\n"
                      "ATR Multiplier: Sensitivity of trendline detection\n"
                      "- Lower values (0.3-0.5): Tighter trendlines\n"
                      "- Higher values (0.6-0.8): More forgiving trendlines")
        self.create_tooltip(atr_frame, atr_tooltip)
        
        # Hough parameters frame
        params_frame = ttk.LabelFrame(main_frame, text="Hough Transform Configuration", padding="10")
        params_frame.grid(row=3, column=0, columnspan=2, pady=(0, 20), sticky=(tk.W, tk.E))
        
        # Add explanation text
        explanation = ("Analysis is performed simultaneously with two different ranges,\n"
                      "allowing detection of both short-term and long-term trend lines.\n"
                      "Short range focuses on immediate trends, long range on broader patterns.")
        explanation_label = ttk.Label(params_frame, text=explanation, wraplength=500, justify=tk.LEFT)
        explanation_label.grid(row=0, column=0, columnspan=3, pady=(0, 10), sticky=tk.W)
        
        # Minimum score frame
        score_frame = ttk.Frame(params_frame)
        score_frame.grid(row=1, column=0, padx=5, pady=(0, 10))
        
        ttk.Label(score_frame, text="Min Score:").grid(row=0, column=0, padx=(0, 5))
        self.min_score = tk.StringVar(value="5.0")
        ttk.Entry(score_frame, textvariable=self.min_score, width=5).grid(row=0, column=1)
        
        # Add tooltip for minimum score
        score_tooltip = ("Minimum score required for a trendline to be considered valid.\n"
                        "Higher values mean more strict detection.\n"
                        "Score calculation:\n"
                        "- Each touch: +5.0 points\n"
                        "- Each throwback: +3.0 points\n"
                        "- Each false breakout: -2.0 points")
        self.create_tooltip(score_frame, score_tooltip)
        
        # Short range frame
        short_range_frame = ttk.Frame(params_frame)
        short_range_frame.grid(row=2, column=0, padx=5)
        
        ttk.Label(short_range_frame, text="Short Range:").grid(row=0, column=0, padx=(0, 5))
        self.range1 = tk.StringVar(value="10")
        ttk.Entry(short_range_frame, textvariable=self.range1, width=5).grid(row=0, column=1)
        ttk.Label(short_range_frame, text="pivots", style='Modern.TLabel').grid(row=0, column=2, padx=(5, 0))
        
        # Long range frame
        long_range_frame = ttk.Frame(params_frame)
        long_range_frame.grid(row=1, column=1, padx=5)
        
        ttk.Label(long_range_frame, text="Long Range:").grid(row=0, column=0, padx=(0, 5))
        self.range2 = tk.StringVar(value="25")
        ttk.Entry(long_range_frame, textvariable=self.range2, width=5).grid(row=0, column=1)
        ttk.Label(long_range_frame, text="pivots", style='Modern.TLabel').grid(row=0, column=2, padx=(5, 0))
        
        # Analysis button
        analyze_button = ttk.Button(main_frame, text="Analyze", command=self.run_analysis)
        analyze_button.grid(row=4, column=0, columnspan=2, pady=(0, 10))
        
    def run_analysis(self):
        """Run the analysis with selected parameters"""
        if not self.file_path.get():
            messagebox.showerror("Error", "Please select a data file first.")
            return
            
        try:
            # Load and prepare data with ATR
            df = pd.read_csv(self.file_path.get())
            atr_period = int(self.atr_period.get())
            atr_multiplier = float(self.atr_multiplier.get())
            df = prepare_data_with_atr(df, atr_period)
            
            # Get pivot points with specified window
            window = int(self.window_size.get())
            high_pivots, low_pivots = get_pivot_points(df, window=window)
            
            # Calculate trendlines
            if self.trendline_method.get() == 1:
                support_lines = simple_trendlines(low_pivots, df, is_support=True)
                resistance_lines = simple_trendlines(high_pivots, df, is_support=False)
            else:
                # Use Hough transform with specified ranges
                try:
                    range1 = int(self.range1.get())
                    range2 = int(self.range2.get())
                    future_pivot_ranges = [range1, range2]
                except ValueError:
                    messagebox.showerror("Error", "Future pivot ranges must be valid numbers.")
                    return
                
                min_score = float(self.min_score.get())
                
                support_lines = hough_transform_trendlines(
                    low_pivots, df, is_support=True,
                    high_pivots=high_pivots, low_pivots=low_pivots,
                    future_pivot_ranges=future_pivot_ranges,
                    atr_multiplier=atr_multiplier,
                    min_score=min_score
                )
                
                resistance_lines = hough_transform_trendlines(
                    high_pivots, df, is_support=False,
                    high_pivots=high_pivots, low_pivots=low_pivots,
                    future_pivot_ranges=future_pivot_ranges,
                    atr_multiplier=atr_multiplier,
                    min_score=min_score
                )
            
            # Plot results
            plot_analysis(df, high_pivots, low_pivots, support_lines, resistance_lines)
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

def main():
    root = tk.Tk()
    app = TrendAnalyzerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()