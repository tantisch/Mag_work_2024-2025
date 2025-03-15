import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List, Tuple, Union, Dict, Optional
from .trendline_events import TrendlineEvents

def plot_analysis(df: pd.DataFrame, 
                 high_pivots: np.ndarray, 
                 low_pivots: np.ndarray,
                 support_lines: List[Tuple], 
                 resistance_lines: List[Tuple],
                 show_trades: bool = False,
                 reward_ratio: float = 2.0,
                 pivot_window: int = 5,
                 event_window: int = 3,
                 atr_multiplier: float = 1.0,
                 risk_per_trade: float = 100.0):  # Default risk of $100 per trade
    """Plot price data with pivot points, trendlines and events"""
    # Calculate TP ATR multiplier from reward_ratio and atr_multiplier
    tp_atr_multiplier = atr_multiplier * reward_ratio
    
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
    
    # List to store all throwbacks for chronological processing
    all_throwbacks = []
    
    # List to store completed trades for visualization
    completed_trades = []
    
    # Track total profit/loss
    total_pnl = 0.0
    
    # Plot trendlines and collect throwbacks
    def plot_trendline(line, is_support=True):
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
            
            # Plot throwbacks and collect them for trade processing
            if events.throwbacks:
                y_throwbacks = [slope * x + intercept for x in events.throwbacks]
                plt.scatter(list(events.throwbacks), y_throwbacks, 
                          color=color, marker='s', s=100,
                          label='Throwbacks' if is_support else None)
                
                # Collect throwbacks for later processing
                for throwback_idx in events.throwbacks:
                    all_throwbacks.append({
                        'idx': throwback_idx,
                        'is_support': is_support,
                        'entry_idx': throwback_idx + event_window
                    })
            
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
    
    # Plot all trendlines and collect throwbacks
    for line in support_lines:
        plot_trendline(line, is_support=True)
    
    for line in resistance_lines:
        plot_trendline(line, is_support=False)
    
    # Process trades if showing trades is enabled
    if show_trades and 'atr' in df.columns and all_throwbacks:
        # Sort throwbacks by entry index to process them in chronological order
        all_throwbacks.sort(key=lambda x: x['entry_idx'])
        
        # Initialize trade state
        in_trade = False
        trade_entry_idx = None
        trade_entry_price = None
        trade_sl_price = None
        trade_tp_price = None
        trade_type = None
        trade_position_size = None
        
        # Process each bar in the dataframe
        for current_idx in range(len(df)):
            # First check if we're in a trade and need to check for TP/SL
            if in_trade and current_idx > trade_entry_idx:
                high = df['high'].iloc[current_idx]
                low = df['low'].iloc[current_idx]
                
                if trade_type == 'LONG':
                    # Check if TP hit (price went up to target)
                    if high >= trade_tp_price:
                        # Calculate profit
                        price_diff = trade_tp_price - trade_entry_price
                        profit = price_diff * trade_position_size
                        total_pnl += profit
                        
                        # Add to completed trades
                        completed_trades.append({
                            'entry_idx': trade_entry_idx,
                            'entry_price': trade_entry_price,
                            'sl_price': trade_sl_price,
                            'tp_price': trade_tp_price,
                            'exit_idx': current_idx,
                            'exit_price': trade_tp_price,
                            'type': trade_type,
                            'result': 'TP',
                            'position_size': trade_position_size,
                            'profit': profit
                        })
                        
                        # Reset trade state
                        in_trade = False
                        trade_entry_idx = None
                        trade_entry_price = None
                        trade_sl_price = None
                        trade_tp_price = None
                        trade_type = None
                        trade_position_size = None
                        
                    # Check if SL hit (price went down to stop)
                    elif low <= trade_sl_price:
                        # Calculate loss (should be close to risk_per_trade)
                        price_diff = trade_sl_price - trade_entry_price
                        loss = price_diff * trade_position_size
                        total_pnl += loss  # This will be negative
                        
                        # Add to completed trades
                        completed_trades.append({
                            'entry_idx': trade_entry_idx,
                            'entry_price': trade_entry_price,
                            'sl_price': trade_sl_price,
                            'tp_price': trade_tp_price,
                            'exit_idx': current_idx,
                            'exit_price': trade_sl_price,
                            'type': trade_type,
                            'result': 'SL',
                            'position_size': trade_position_size,
                            'profit': loss  # This will be negative
                        })
                        
                        # Reset trade state
                        in_trade = False
                        trade_entry_idx = None
                        trade_entry_price = None
                        trade_sl_price = None
                        trade_tp_price = None
                        trade_type = None
                        trade_position_size = None
                        
                elif trade_type == 'SHORT':
                    # Check if TP hit (price went down to target)
                    if low <= trade_tp_price:
                        # Calculate profit
                        price_diff = trade_entry_price - trade_tp_price
                        profit = price_diff * trade_position_size
                        total_pnl += profit
                        
                        # Add to completed trades
                        completed_trades.append({
                            'entry_idx': trade_entry_idx,
                            'entry_price': trade_entry_price,
                            'sl_price': trade_sl_price,
                            'tp_price': trade_tp_price,
                            'exit_idx': current_idx,
                            'exit_price': trade_tp_price,
                            'type': trade_type,
                            'result': 'TP',
                            'position_size': trade_position_size,
                            'profit': profit
                        })
                        
                        # Reset trade state
                        in_trade = False
                        trade_entry_idx = None
                        trade_entry_price = None
                        trade_sl_price = None
                        trade_tp_price = None
                        trade_type = None
                        trade_position_size = None
                        
                    # Check if SL hit (price went up to stop)
                    elif high >= trade_sl_price:
                        # Calculate loss (should be close to risk_per_trade)
                        price_diff = trade_sl_price - trade_entry_price
                        loss = price_diff * trade_position_size  # This will be negative for shorts
                        total_pnl += loss
                        
                        # Add to completed trades
                        completed_trades.append({
                            'entry_idx': trade_entry_idx,
                            'entry_price': trade_entry_price,
                            'sl_price': trade_sl_price,
                            'tp_price': trade_tp_price,
                            'exit_idx': current_idx,
                            'exit_price': trade_sl_price,
                            'type': trade_type,
                            'result': 'SL',
                            'position_size': trade_position_size,
                            'profit': loss
                        })
                        
                        # Reset trade state
                        in_trade = False
                        trade_entry_idx = None
                        trade_entry_price = None
                        trade_sl_price = None
                        trade_tp_price = None
                        trade_type = None
                        trade_position_size = None
            
            # If we're not in a trade, check if we should enter one
            if not in_trade:
                # Find throwbacks that match the current index
                potential_entries = [tb for tb in all_throwbacks if tb['entry_idx'] == current_idx]
                
                if potential_entries:
                    # Take the first one (if multiple throwbacks occur at the same bar)
                    throwback = potential_entries[0]
                    
                    # Calculate entry details
                    entry_price = df['close'].iloc[current_idx]
                    atr_value = df['atr'].iloc[current_idx]
                    
                    if throwback['is_support']:  # SHORT trade
                        # Stop loss is atr_multiplier * ATR above entry
                        sl_price = entry_price + (atr_value * atr_multiplier)
                        # Take profit is tp_atr_multiplier * ATR below entry
                        tp_price = entry_price - (atr_value * tp_atr_multiplier)
                        trade_type = 'SHORT'
                        
                        # Calculate position size based on risk
                        price_diff = abs(sl_price - entry_price)
                        if price_diff > 0:
                            position_size = risk_per_trade / price_diff
                        else:
                            position_size = 0  # Skip trade if no risk (shouldn't happen)
                            continue
                            
                    else:  # LONG trade
                        # Stop loss is atr_multiplier * ATR below entry
                        sl_price = entry_price - (atr_value * atr_multiplier)
                        # Take profit is tp_atr_multiplier * ATR above entry
                        tp_price = entry_price + (atr_value * tp_atr_multiplier)
                        trade_type = 'LONG'
                        
                        # Calculate position size based on risk
                        price_diff = abs(entry_price - sl_price)
                        if price_diff > 0:
                            position_size = risk_per_trade / price_diff
                        else:
                            position_size = 0  # Skip trade if no risk (shouldn't happen)
                            continue
                    
                    # Enter the trade
                    in_trade = True
                    trade_entry_idx = current_idx
                    trade_entry_price = entry_price
                    trade_sl_price = sl_price
                    trade_tp_price = tp_price
                    trade_position_size = position_size
        
        # If we reach the end of data and still have an open trade
        if in_trade:
            # Calculate current profit/loss
            current_price = df['close'].iloc[-1]
            
            if trade_type == 'LONG':
                price_diff = current_price - trade_entry_price
            else:  # SHORT
                price_diff = trade_entry_price - current_price
                
            current_pnl = price_diff * trade_position_size
            total_pnl += current_pnl
            
            # Add to completed trades with OPEN status
            completed_trades.append({
                'entry_idx': trade_entry_idx,
                'entry_price': trade_entry_price,
                'sl_price': trade_sl_price,
                'tp_price': trade_tp_price,
                'exit_idx': len(df) - 1,
                'exit_price': current_price,
                'type': trade_type,
                'result': 'OPEN',
                'position_size': trade_position_size,
                'profit': current_pnl
            })
    
    # Visualize all completed trades
    if show_trades and completed_trades:
        # Add a text box with trade statistics
        win_count = sum(1 for trade in completed_trades if trade['result'] == 'TP')
        loss_count = sum(1 for trade in completed_trades if trade['result'] == 'SL')
        open_count = sum(1 for trade in completed_trades if trade['result'] == 'OPEN')
        total_trades = len(completed_trades)
        
        if total_trades > 0:
            win_rate = win_count / (win_count + loss_count) * 100 if (win_count + loss_count) > 0 else 0
            
            stats_text = (
                f"Total Trades: {total_trades}\n"
                f"Wins: {win_count} | Losses: {loss_count} | Open: {open_count}\n"
                f"Win Rate: {win_rate:.1f}%\n"
                f"Total P/L: ${total_pnl:.2f}"
            )
            
            # Add text box with statistics in the top right corner
            plt.annotate(stats_text, xy=(0.98, 0.98), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
                        va='top', ha='right', fontsize=10)
        
        for trade in completed_trades:
            entry_idx = trade['entry_idx']
            entry_price = trade['entry_price']
            sl_price = trade['sl_price']
            tp_price = trade['tp_price']
            exit_idx = trade['exit_idx']
            exit_price = trade['exit_price']
            result = trade['result']
            
            # Only draw lines from entry to exit
            x_range = np.array(range(entry_idx, exit_idx + 1))
            
            # Entry line
            plt.plot(x_range, [entry_price] * len(x_range), 
                   color='blue', linestyle='-', linewidth=1.5,
                   label='Entry' if trade['type'] == 'LONG' else None)
            
            # Stop Loss line
            plt.plot(x_range, [sl_price] * len(x_range), 
                   color='red', linestyle='-', linewidth=1.5,
                   label='Stop Loss' if trade['type'] == 'LONG' else None)
            
            # Take Profit line
            plt.plot(x_range, [tp_price] * len(x_range), 
                   color='green', linestyle='-', linewidth=1.5,
                   label='Take Profit' if trade['type'] == 'LONG' else None)
            
            # Mark entry point
            plt.scatter(entry_idx, entry_price, color='blue', s=100, marker='o')
            
            # Mark exit point with appropriate color
            if result == 'TP':
                exit_color = 'green'
            elif result == 'SL':
                exit_color = 'red'
            else:  # OPEN
                exit_color = 'orange'
                
            plt.scatter(exit_idx, exit_price, color=exit_color, s=100, marker='*')
    
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
    
    # Return trade statistics for further analysis if needed
    if show_trades and completed_trades:
        return {
            'trades': completed_trades,
            'total_pnl': total_pnl,
            'win_count': win_count,
            'loss_count': loss_count,
            'win_rate': win_rate if (win_count + loss_count) > 0 else 0
        }
    return None