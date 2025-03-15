import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
from src.pivot_detection import get_pivot_points
from src.trendline_detection import simple_trendlines, hough_transform_trendlines
from src.visualization import plot_analysis
from src.utils import prepare_data_with_atr
import os

class ModernUI(ttk.Style):
    """Custom style for a modern UI appearance"""
    def __init__(self):
        super().__init__()
        self.theme_use('clam')  # Use clam as base theme
        
        # Configure colors
        bg_color = "#f5f5f7"
        accent_color = "#0066cc"
        text_color = "#333333"
        
        # Configure frame styles
        self.configure('Modern.TFrame', background=bg_color)
        self.configure('Card.TFrame', background='white', relief='flat')
        
        # Configure label styles
        self.configure('Modern.TLabel', background=bg_color, foreground=text_color, font=('Segoe UI', 10))
        self.configure('Header.TLabel', background=bg_color, foreground=text_color, font=('Segoe UI', 12, 'bold'))
        self.configure('Card.TLabel', background='white', foreground=text_color, font=('Segoe UI', 10))
        self.configure('CardHeader.TLabel', background='white', foreground=text_color, font=('Segoe UI', 11, 'bold'))
        
        # Configure button styles
        self.configure('Modern.TButton', background=accent_color, foreground='white', font=('Segoe UI', 10))
        self.map('Modern.TButton', 
                background=[('active', '#0055aa'), ('pressed', '#004488')],
                foreground=[('active', 'white'), ('pressed', 'white')])
        
        # Configure entry styles
        self.configure('Modern.TEntry', font=('Segoe UI', 10))
        
        # Configure checkbox and radio button styles
        self.configure('Modern.TCheckbutton', background=bg_color, font=('Segoe UI', 10))
        self.configure('Modern.TRadiobutton', background=bg_color, font=('Segoe UI', 10))
        
        # Configure separator
        self.configure('Modern.TSeparator', background='#dddddd')

class TrendAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Trend Analysis Tool")
        self.root.geometry("800x800")
        self.root.configure(bg="#f5f5f7")
        
        # Apply modern style
        self.style = ModernUI()
        
        # Trading state variables
        self.current_trade = None
        
        self.create_widgets()
    
    def create_tooltip(self, widget, text):
        """Create a tooltip for a given widget"""
        def show_tooltip(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            tooltip.configure(bg="#ffffcc")
            
            label = ttk.Label(tooltip, text=text, justify=tk.LEFT,
                            relief='solid', borderwidth=1,
                            padding=8, background="#ffffcc", 
                            font=('Segoe UI', 9), wraplength=300)
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
            # Update status
            self.status_var.set(f"File selected: {os.path.basename(filename)}")
        
    def create_widgets(self):
        # Main container
        main_container = ttk.Frame(self.root, style='Modern.TFrame', padding=20)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header_frame = ttk.Frame(main_container, style='Modern.TFrame')
        header_frame.pack(fill=tk.X, pady=(0, 15))
        
        header_label = ttk.Label(header_frame, text="Trend Analysis Tool", 
                               style='Header.TLabel', font=('Segoe UI', 16, 'bold'))
        header_label.pack(side=tk.LEFT)
        
        # File selection card
        file_card = ttk.Frame(main_container, style='Card.TFrame', padding=15)
        file_card.pack(fill=tk.X, pady=10)
        
        file_header = ttk.Label(file_card, text="Data Source", style='CardHeader.TLabel')
        file_header.pack(anchor=tk.W, pady=(0, 10))
        
        file_frame = ttk.Frame(file_card, style='Card.TFrame')
        file_frame.pack(fill=tk.X)
        
        self.file_path = tk.StringVar()
        file_entry = ttk.Entry(file_frame, textvariable=self.file_path, width=60, style='Modern.TEntry')
        file_entry.pack(side=tk.LEFT, padx=(0, 10), fill=tk.X, expand=True)
        
        browse_button = ttk.Button(file_frame, text="Browse", command=self.browse_file, style='Modern.TButton')
        browse_button.pack(side=tk.RIGHT)
        
        # Settings container with two columns
        settings_container = ttk.Frame(main_container, style='Modern.TFrame')
        settings_container.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Left column for detection settings
        left_column = ttk.Frame(settings_container, style='Modern.TFrame')
        left_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Right column for trading settings
        right_column = ttk.Frame(settings_container, style='Modern.TFrame')
        right_column.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # Detection Method Card
        method_card = ttk.Frame(left_column, style='Card.TFrame', padding=15)
        method_card.pack(fill=tk.X, pady=10)
        
        method_header = ttk.Label(method_card, text="Detection Method", style='CardHeader.TLabel')
        method_header.pack(anchor=tk.W, pady=(0, 10))
        
        self.trendline_method = tk.IntVar(value=1)
        method_frame = ttk.Frame(method_card, style='Card.TFrame')
        method_frame.pack(fill=tk.X)
        
        ttk.Radiobutton(method_frame, text="Linear Regression", variable=self.trendline_method, 
                       value=1, style='Modern.TRadiobutton').pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(method_frame, text="Hough Transform", variable=self.trendline_method, 
                       value=2, style='Modern.TRadiobutton').pack(anchor=tk.W, pady=2)
        
        # Pivot Parameters Card
        pivot_card = ttk.Frame(left_column, style='Card.TFrame', padding=15)
        pivot_card.pack(fill=tk.X, pady=10)
        
        pivot_header = ttk.Label(pivot_card, text="Pivot Point Parameters", style='CardHeader.TLabel')
        pivot_header.pack(anchor=tk.W, pady=(0, 10))
        
        # Grid for pivot parameters
        pivot_grid = ttk.Frame(pivot_card, style='Card.TFrame')
        pivot_grid.pack(fill=tk.X)
        
        # Window Size
        ttk.Label(pivot_grid, text="Window Size:", style='Card.TLabel').grid(row=0, column=0, sticky=tk.W, pady=5)
        self.window_size = tk.StringVar(value="5")
        ttk.Entry(pivot_grid, textvariable=self.window_size, width=8, style='Modern.TEntry').grid(row=0, column=1, padx=10, pady=5)
        
        # Event Window
        ttk.Label(pivot_grid, text="Event Window:", style='Card.TLabel').grid(row=1, column=0, sticky=tk.W, pady=5)
        self.event_window = tk.StringVar(value="3")
        ttk.Entry(pivot_grid, textvariable=self.event_window, width=8, style='Modern.TEntry').grid(row=1, column=1, padx=10, pady=5)
        
        # Add tooltip for pivot parameters
        pivot_tooltip = ("Window Size: Number of bars to look back/forward for pivot detection\n"
                        "Event Window: Number of bars to wait after throwback before entering a trade\n"
                        "- Smaller values: Faster entry after throwback confirmation\n"
                        "- Larger values: More confirmation but later entry")
        self.create_tooltip(pivot_card, pivot_tooltip)
        
        # ATR Settings Card
        atr_card = ttk.Frame(left_column, style='Card.TFrame', padding=15)
        atr_card.pack(fill=tk.X, pady=10)
        
        atr_header = ttk.Label(atr_card, text="ATR Settings", style='CardHeader.TLabel')
        atr_header.pack(anchor=tk.W, pady=(0, 10))
        
        # Grid for ATR settings
        atr_grid = ttk.Frame(atr_card, style='Card.TFrame')
        atr_grid.pack(fill=tk.X)
        
        # ATR Period
        ttk.Label(atr_grid, text="ATR Period:", style='Card.TLabel').grid(row=0, column=0, sticky=tk.W, pady=5)
        self.atr_period = tk.StringVar(value="14")
        ttk.Entry(atr_grid, textvariable=self.atr_period, width=8, style='Modern.TEntry').grid(row=0, column=1, padx=10, pady=5)
        
        # ATR Multiplier
        ttk.Label(atr_grid, text="ATR Multiplier:", style='Card.TLabel').grid(row=1, column=0, sticky=tk.W, pady=5)
        self.atr_multiplier = tk.StringVar(value="0.4")
        ttk.Entry(atr_grid, textvariable=self.atr_multiplier, width=8, style='Modern.TEntry').grid(row=1, column=1, padx=10, pady=5)
        
        # Add tooltip for ATR settings
        atr_tooltip = ("ATR Period: Number of periods for ATR calculation\n"
                      "ATR Multiplier: Sensitivity of trendline detection\n"
                      "- Lower values (0.3-0.5): Tighter trendlines\n"
                      "- Higher values (0.6-0.8): More forgiving trendlines")
        self.create_tooltip(atr_card, atr_tooltip)
        
        # Hough Transform Card
        hough_card = ttk.Frame(right_column, style='Card.TFrame', padding=15)
        hough_card.pack(fill=tk.X, pady=10)
        
        hough_header = ttk.Label(hough_card, text="Hough Transform Settings", style='CardHeader.TLabel')
        hough_header.pack(anchor=tk.W, pady=(0, 10))
        
        # Add explanation text
        explanation = ("Analysis is performed with two different ranges,\n"
                      "allowing detection of both short-term and long-term trend lines.")
        explanation_label = ttk.Label(hough_card, text=explanation, style='Card.TLabel', wraplength=300)
        explanation_label.pack(anchor=tk.W, pady=(0, 10))
        
        # Grid for Hough parameters
        hough_grid = ttk.Frame(hough_card, style='Card.TFrame')
        hough_grid.pack(fill=tk.X)
        
        # Min Score
        ttk.Label(hough_grid, text="Min Score:", style='Card.TLabel').grid(row=0, column=0, sticky=tk.W, pady=5)
        self.min_score = tk.StringVar(value="15")
        ttk.Entry(hough_grid, textvariable=self.min_score, width=8, style='Modern.TEntry').grid(row=0, column=1, padx=10, pady=5)
        
        # Short Range
        ttk.Label(hough_grid, text="Short Range:", style='Card.TLabel').grid(row=1, column=0, sticky=tk.W, pady=5)
        self.range1 = tk.StringVar(value="10")
        ttk.Entry(hough_grid, textvariable=self.range1, width=8, style='Modern.TEntry').grid(row=1, column=1, padx=10, pady=5)
        ttk.Label(hough_grid, text="pivots", style='Card.TLabel').grid(row=1, column=2, sticky=tk.W, pady=5)
        
        # Long Range
        ttk.Label(hough_grid, text="Long Range:", style='Card.TLabel').grid(row=2, column=0, sticky=tk.W, pady=5)
        self.range2 = tk.StringVar(value="0")
        ttk.Entry(hough_grid, textvariable=self.range2, width=8, style='Modern.TEntry').grid(row=2, column=1, padx=10, pady=5)
        ttk.Label(hough_grid, text="pivots", style='Card.TLabel').grid(row=2, column=2, sticky=tk.W, pady=5)
        
        # Add tooltip for minimum score
        score_tooltip = ("Minimum score required for a trendline to be considered valid.\n"
                        "Higher values mean more strict detection.\n"
                        "Score calculation:\n"
                        "- Each touch: +5.0 points\n"
                        "- Each throwback: +3.0 points\n"
                        "- Each false breakout: -2.0 points")
        self.create_tooltip(hough_card, score_tooltip)
        
        # Trading Settings Card
        trading_card = ttk.Frame(right_column, style='Card.TFrame', padding=15)
        trading_card.pack(fill=tk.X, pady=10)
        
        trading_header = ttk.Label(trading_card, text="Trading Settings", style='CardHeader.TLabel')
        trading_header.pack(anchor=tk.W, pady=(0, 10))
        
        # Show trades checkbox
        self.show_trades = tk.BooleanVar(value=True)
        show_trades_check = ttk.Checkbutton(trading_card, text="Show Potential Trades at Throwbacks", 
                                          variable=self.show_trades, style='Modern.TCheckbutton')
        show_trades_check.pack(anchor=tk.W, pady=5)
        
        # Grid for trading parameters
        trading_grid = ttk.Frame(trading_card, style='Card.TFrame')
        trading_grid.pack(fill=tk.X, pady=5)
        
        # Risk-Reward Ratio
        ttk.Label(trading_grid, text="Risk-Reward Ratio:", style='Card.TLabel').grid(row=0, column=0, sticky=tk.W, pady=5)
        ratio_frame = ttk.Frame(trading_grid, style='Card.TFrame')
        ratio_frame.grid(row=0, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(ratio_frame, text="1:", style='Card.TLabel').pack(side=tk.LEFT)
        self.reward_ratio = tk.StringVar(value="2")
        ttk.Entry(ratio_frame, textvariable=self.reward_ratio, width=5, style='Modern.TEntry').pack(side=tk.LEFT)
        
        # SL/TP ATR Multiplier
        ttk.Label(trading_grid, text="SL/TP ATR Multiplier:", style='Card.TLabel').grid(row=1, column=0, sticky=tk.W, pady=5)
        self.atr_multiplier_trade = tk.StringVar(value="2")
        ttk.Entry(trading_grid, textvariable=self.atr_multiplier_trade, width=8, style='Modern.TEntry').grid(row=1, column=1, sticky=tk.W, pady=5, padx=10)
        
        # Risk Per Trade
        ttk.Label(trading_grid, text="Risk Per Trade ($):", style='Card.TLabel').grid(row=2, column=0, sticky=tk.W, pady=5)
        self.risk_per_trade = tk.StringVar(value="100.0")
        ttk.Entry(trading_grid, textvariable=self.risk_per_trade, width=8, style='Modern.TEntry').grid(row=2, column=1, sticky=tk.W, pady=5, padx=10)
        
        # Add tooltip for trading settings
        trading_tooltip = ("Show Potential Trades: Display entry, stop loss, and take profit levels\n"
                          "at throwback points based on ATR.\n"
                          "Risk-Reward Ratio: Ratio of potential profit to potential loss (e.g., 1:2)\n"
                          "SL/TP ATR Multiplier: ATR multiplier for stop loss and take profit\n"
                          "- SL distance = ATR × Multiplier\n"
                          "- TP distance = ATR × Multiplier × Risk-Reward Ratio\n"
                          "Risk Per Trade: Fixed dollar amount to risk per trade")
        self.create_tooltip(trading_card, trading_tooltip)
        
        # Action buttons
        button_frame = ttk.Frame(main_container, style='Modern.TFrame')
        button_frame.pack(fill=tk.X, pady=15)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready. Please select a data file to begin.")
        status_bar = ttk.Label(main_container, textvariable=self.status_var, 
                             relief=tk.SUNKEN, padding=8, background="#f0f0f0",
                             font=('Segoe UI', 9))
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Analyze button
        analyze_button = ttk.Button(button_frame, text="Analyze", command=self.run_analysis, 
                                  style='Modern.TButton', padding=(20, 10))
        analyze_button.pack(side=tk.LEFT)
        
        # Exit button
        exit_button = ttk.Button(button_frame, text="Exit", command=self.root.destroy, 
                               style='Modern.TButton', padding=(20, 10))
        exit_button.pack(side=tk.RIGHT)
        
    def run_analysis(self):
        """Run the analysis with selected parameters"""
        if not self.file_path.get():
            messagebox.showerror("Error", "Please select a data file first.")
            return
            
        try:
            self.status_var.set("Loading data and calculating trendlines...")
            self.root.update_idletasks()  # Update the UI to show status
            
            # Load and prepare data with ATR
            df = pd.read_csv(self.file_path.get())
            atr_period = int(self.atr_period.get())
            atr_multiplier = float(self.atr_multiplier.get())
            df = prepare_data_with_atr(df, atr_period)
            
            # Get pivot points with specified window
            window = int(self.window_size.get())
            high_pivots, low_pivots = get_pivot_points(df, window=window)
            
            # Get event window size
            try:
                event_window = int(self.event_window.get())
            except ValueError:
                event_window = 3  # Default if invalid input
            
            # Get trading parameters
            show_trades = self.show_trades.get()
            try:
                reward_ratio = float(self.reward_ratio.get())
            except ValueError:
                reward_ratio = 2.0  # Default if invalid input
                
            # Get SL/TP ATR multiplier
            try:
                atr_multiplier_trade = float(self.atr_multiplier_trade.get())
            except ValueError:
                atr_multiplier_trade = 2.0  # Default if invalid input
                
            # Get risk per trade
            try:
                risk_per_trade = float(self.risk_per_trade.get())
            except ValueError:
                risk_per_trade = 100.0  # Default if invalid input
            
            self.status_var.set("Detecting trendlines...")
            self.root.update_idletasks()  # Update the UI to show status
            
            # Calculate trendlines
            if self.trendline_method.get() == 1:
                # Use simple trendlines
                support_lines = simple_trendlines(
                    low_pivots, df, is_support=True,
                    high_pivots=high_pivots, low_pivots=low_pivots,
                    atr_multiplier=atr_multiplier
                )
                resistance_lines = simple_trendlines(
                    high_pivots, df, is_support=False,
                    high_pivots=high_pivots, low_pivots=low_pivots,
                    atr_multiplier=atr_multiplier
                )
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
            
            self.status_var.set("Generating visualization...")
            self.root.update_idletasks()  # Update the UI to show status
            
            # Plot results with trading parameters
            result = plot_analysis(
                df, high_pivots, low_pivots, 
                support_lines, resistance_lines,
                show_trades=show_trades,
                reward_ratio=reward_ratio,
                pivot_window=window,
                event_window=event_window,
                atr_multiplier=atr_multiplier_trade,
                risk_per_trade=risk_per_trade
            )
            
            # Display trade statistics if available
            if result:
                stats_message = (
                    f"Trade Statistics:\n"
                    f"Total Trades: {len(result['trades'])}\n"
                    f"Win Rate: {result['win_rate']:.1f}%\n"
                    f"Total P/L: ${result['total_pnl']:.2f}"
                )
                messagebox.showinfo("Trade Statistics", stats_message)
                
            self.status_var.set("Analysis complete.")
            
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

def main():
    root = tk.Tk()
    app = TrendAnalyzerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()