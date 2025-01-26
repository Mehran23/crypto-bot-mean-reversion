import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime
import numpy as np
import json
import os
from matplotlib.dates import DateFormatter, AutoDateLocator
from matplotlib.ticker import FuncFormatter

# Set the style to dark theme
plt.style.use('dark_background')

# Custom aqua theme colors
ORANGE_THEME = {
    'primary': '#00CED1',     # Dark turquoise
    'secondary': '#40E0D0',   # Turquoise
    'text': '#FFFFFF',        # White text
    'grid': '#333333',        # Dark gray for grid
    'background': '#000000',  # Pure black background
    'accent': '#00FFFF',      # Cyan accent
    'positive': '#00FF9F',    # Bright green
    'negative': '#FF4444',    # Bright red
    'ma_line': '#87CEEB',     # Sky blue for moving average
    'fill': '#00CED1',        # Fill color matching primary
    'muted_text': '#AAAAAA'   # Muted text for less important elements
}

# Apply custom theme
mpl.rcParams['text.color'] = ORANGE_THEME['text']
mpl.rcParams['axes.labelcolor'] = ORANGE_THEME['text']
mpl.rcParams['xtick.color'] = ORANGE_THEME['text']
mpl.rcParams['ytick.color'] = ORANGE_THEME['text']
mpl.rcParams['axes.facecolor'] = ORANGE_THEME['background']
mpl.rcParams['figure.facecolor'] = ORANGE_THEME['background']

def get_file_path(filename):    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, filename)

def load_json_data(filename):
    file_path = get_file_path(filename)
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found in the script's directory.")
        print(f"Expected location: {file_path}")
        raise

def save_json_data(filename, data):
    with open(get_file_path(filename), 'w') as f:
        json.dump(data, f, indent=2)

def fetch_account_balance_history():
    balance_data = load_json_data('balance_history.json')
    return balance_data['dates'], balance_data['balances']

def fetch_transactions():
    return load_json_data('transactions.json')

def adjust_balance_for_transactions(dates, balances, transactions):
    adjusted_balances = balances.copy()
    # First, create a list of all transactions for each date
    date_transactions = {date: 0 for date in dates}
    for date, amount in transactions.items():
        if date in dates:
            date_transactions[date] += amount

    # Then apply the transactions to all subsequent balances
    cumulative_adjustment = 0
    for i, date in enumerate(dates):
        if date_transactions[date] != 0:
            # For withdrawals (negative amounts), we ADD the absolute value
            # For deposits (positive amounts), we SUBTRACT the amount
            cumulative_adjustment -= date_transactions[date]  # Negative becomes positive, positive becomes negative
        adjusted_balances[i] += cumulative_adjustment  # Add the adjustment
    
    return adjusted_balances

def fetch_btc_closing_prices(dates):
    np.random.seed(42)
    base_price = 30000
    price_changes = np.random.normal(loc=0, scale=500, size=len(dates))
    btc_prices = base_price + np.cumsum(price_changes)
    return btc_prices

def calculate_max_drawdown(balances):
    running_max = float('-inf')
    max_drawdown = 0
    
    for balance in balances:
        if balance > running_max:
            running_max = balance
        if running_max > 0:  # Avoid division by zero
            drawdown = ((running_max - balance) / running_max) * 100
            max_drawdown = max(max_drawdown, drawdown)
    
    return max_drawdown

def calculate_gross_profit_and_loss(balances):
    gross_profit = 0
    gross_loss = 0
    
    for i in range(1, len(balances)):
        change = balances[i] - balances[i - 1]
        if change > 0:
            gross_profit += change
        else:
            gross_loss += abs(change)
    
    return gross_profit, gross_loss

def calculate_average_daily_return(balances):
    daily_returns = [(balances[i] - balances[i - 1]) / balances[i - 1] for i in range(1, len(balances))]
    average_daily_return = np.mean(daily_returns) * 100
    return average_daily_return

def calculate_risk_of_ruin(balances, risk_free_rate=0.01):
    daily_returns = [(balances[i] - balances[i - 1]) / balances[i - 1] for i in range(1, len(balances))]
    mean_return = np.mean(daily_returns)
    std_dev_return = np.std(daily_returns)
    
    if std_dev_return == 0:
        return float('inf')  # Return infinity to indicate no variability, high risk of ruin
    
    risk_of_ruin = (mean_return - risk_free_rate) / std_dev_return
    return risk_of_ruin

def money_formatter(x, p):
    return f'${x:,.2f}'

def calculate_moving_average(data, window=7):
    return np.convolve(data, np.ones(window)/window, mode='valid')

def calculate_daily_changes(balances):
    return np.diff(balances)

def calculate_sharpe_ratio(balances, risk_free_rate=0.01):
    """
    Calculate the Sharpe Ratio
    :param balances: List of daily balances
    :param risk_free_rate: Annual risk-free rate (default 1%)
    :return: Annualized Sharpe Ratio
    """
    # Calculate daily returns
    daily_returns = [(balances[i] - balances[i - 1]) / balances[i - 1] for i in range(1, len(balances))]
    
    # Convert annual risk-free rate to daily
    daily_rf_rate = (1 + risk_free_rate) ** (1/252) - 1
    
    # Calculate excess returns (return - risk free rate)
    excess_returns = [r - daily_rf_rate for r in daily_returns]
    
    if len(excess_returns) == 0 or np.std(excess_returns) == 0:
        return 0
    
    # Calculate annualized Sharpe Ratio
    # First annualize the returns
    avg_excess_return = np.mean(excess_returns) * 252
    # Then annualize the standard deviation
    std_dev = np.std(excess_returns) * np.sqrt(252)
    
    sharpe = avg_excess_return / std_dev if std_dev > 0 else 0
    return sharpe

def calculate_average_win(balances):
    profits = []
    for i in range(1, len(balances)):
        change = balances[i] - balances[i - 1]
        if change > 0:
            profits.append(change)
    return np.mean(profits) if profits else 0

def calculate_volatility(balances):
    """Calculate annualized volatility"""
    daily_returns = [(balances[i] - balances[i - 1]) / balances[i - 1] for i in range(1, len(balances))]
    return np.std(daily_returns) * np.sqrt(252) * 100

def calculate_sortino_ratio(balances, risk_free_rate=0.01):
    """Calculate Sortino ratio (like Sharpe but only downside volatility)"""
    daily_returns = [(balances[i] - balances[i - 1]) / balances[i - 1] for i in range(1, len(balances))]
    daily_rf_rate = (1 + risk_free_rate) ** (1/252) - 1
    excess_returns = [r - daily_rf_rate for r in daily_returns]
    
    # Calculate downside returns
    downside_returns = [r for r in excess_returns if r < 0]
    if not downside_returns:
        return float('inf')
    
    downside_std = np.std(downside_returns) if len(downside_returns) > 1 else 0
    if downside_std == 0:
        return float('inf')
    
    return (np.mean(excess_returns) * 252) / (downside_std * np.sqrt(252))

def calculate_calmar_ratio(balances, days=None):
    """
    Calculate Calmar ratio adjusted for any timeframe
    Max DD is already a percentage, so we just need to annualize returns
    """
    if days is None:
        days = len(balances)
    
    max_dd = calculate_max_drawdown(balances) / 100  # Convert to decimal
    if max_dd == 0:
        return float('inf')
    
    # Calculate return and annualize it based on actual number of days
    total_return = (balances[-1] / balances[0]) - 1
    annual_return = ((1 + total_return) ** (252/days)) - 1
    
    return annual_return / max_dd

def calculate_recovery_factor(balances):
    """Calculate Recovery Factor (Total Return / Max DD)"""
    max_dd = calculate_max_drawdown(balances) / 100  # Convert to decimal
    if max_dd == 0:
        return float('inf')
    
    total_return = (balances[-1] - balances[0]) / balances[0]
    return total_return / max_dd

if __name__ == "__main__":
    try:
        # Fetch and process data
        dates, balances = fetch_account_balance_history()
        transactions = fetch_transactions()
        
        # First adjust balances for transactions
        adjusted_balances = adjust_balance_for_transactions(dates, balances, transactions)
        
        # Convert dates to datetime
        dates = [datetime.strptime(date, "%Y-%m-%d") for date in dates]
        
        # Print debug information
        print("Debug Information:")
        print("Original Balances:", balances)
        print("Transactions:", transactions)
        print("Adjusted Balances:", adjusted_balances)
        
        # Calculate basic metrics
        start_balance = adjusted_balances[0]
        end_balance = adjusted_balances[-1]
        total_profit = end_balance - start_balance
        overall_return = ((end_balance - start_balance) / start_balance) * 100
        
        # Calculate all metrics using adjusted_balances
        max_drawdown = calculate_max_drawdown(adjusted_balances)
        gross_profit, gross_loss = calculate_gross_profit_and_loss(adjusted_balances)
        average_daily_return = calculate_average_daily_return(adjusted_balances)
        monthly_return = average_daily_return * 30  # Approximate monthly return
        risk_of_ruin = calculate_risk_of_ruin(adjusted_balances)
        sharpe_ratio = calculate_sharpe_ratio(adjusted_balances)
        average_win = calculate_average_win(adjusted_balances)
        
        # Calculate additional metrics
        daily_changes = calculate_daily_changes(adjusted_balances)
        ma_20 = calculate_moving_average(adjusted_balances, 20)
        ma_7 = calculate_moving_average(adjusted_balances, 7)
        
        # Calculate additional institutional metrics
        days_elapsed = (dates[-1] - dates[0]).days or 1  # Use actual days elapsed, minimum 1
        volatility = calculate_volatility(adjusted_balances)
        sortino_ratio = calculate_sortino_ratio(adjusted_balances)
        calmar_ratio = calculate_calmar_ratio(adjusted_balances, days_elapsed)
        recovery_factor = calculate_recovery_factor(adjusted_balances)
        
        # Create figure with custom background
        fig = plt.figure(figsize=(12, 8.5))
        
        # Create GridSpec with correct height ratios
        gs = plt.GridSpec(2, 1, figure=fig, height_ratios=[5.5, 1])
        
        # Main plot
        ax1 = fig.add_subplot(gs[0])
        ax_metrics = fig.add_subplot(gs[1])
        
        ax1.set_facecolor(ORANGE_THEME['background'])
        fig.patch.set_facecolor(ORANGE_THEME['background'])
        
        # Create watermark text
        watermark_text = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        # Add institutional-style header info
        header_text = (
            f"PERFORMANCE ANALYSIS REPORT\n"
            f"Period: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}"
        )
        
        # Add header on left
        plt.figtext(0.02, 0.98, header_text,
                   fontsize=10, color=ORANGE_THEME['muted_text'],
                   ha='left', va='top',
                   fontfamily='monospace')
        
        # Add generation time on right
        plt.figtext(0.98, 0.98, f"Generated: {watermark_text}",
                   fontsize=10, color=ORANGE_THEME['muted_text'],
                   ha='right', va='top',
                   fontfamily='monospace')
        
        # Plot main equity curve with enhanced styling
        line = ax1.plot(dates, adjusted_balances, color=ORANGE_THEME['primary'], 
                       linewidth=2.5, zorder=3)[0]
        
        # Add end dot
        ax1.scatter(dates[-1], adjusted_balances[-1], 
                   color=ORANGE_THEME['primary'],
                   s=100,  # Size of the dot
                   zorder=4,  # Make sure it's on top
                   edgecolor='white',  # White edge
                   linewidth=1)  # Edge width
        
        # Plot moving averages with enhanced styling
        if len(dates) > 20:
            ax1.plot(dates[19:], ma_20, color=ORANGE_THEME['ma_line'], 
                    linewidth=1.2, alpha=0.7, zorder=2,
                    linestyle='--')
        if len(dates) > 7:
            ax1.plot(dates[6:], ma_7, color=ORANGE_THEME['secondary'], 
                    linewidth=1.2, alpha=0.7, zorder=2,
                    linestyle='-.')

        # Enhanced gradient fill
        ax1.fill_between(dates, adjusted_balances, min(adjusted_balances), 
                        color=ORANGE_THEME['fill'], alpha=0.15, zorder=1)

        # Style the main plot
        # Grid
        ax1.grid(True, linestyle='--', linewidth=0.5, color=ORANGE_THEME['grid'], alpha=0.15)
        ax1.grid(True, linestyle=':', linewidth=0.3, color=ORANGE_THEME['grid'], alpha=0.1, which='minor')
        ax1.minorticks_on()
        
        # Spines
        for spine in ax1.spines.values():
            spine.set_color(ORANGE_THEME['grid'])
            spine.set_linewidth(0.5)
        
        # Date formatting
        ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        ax1.tick_params(axis='x', rotation=45, labelsize=9, colors=ORANGE_THEME['muted_text'])
        ax1.tick_params(axis='y', labelsize=9, colors=ORANGE_THEME['muted_text'])
        
        # Y-axis formatting
        ax1.yaxis.set_major_formatter(FuncFormatter(money_formatter))
        
        # Labels with enhanced styling
        ax1.set_ylabel('Profits', color=ORANGE_THEME['muted_text'], fontsize=10)

        # Enhanced metrics display with more data
        metrics_data = [
            ['$ Total Profit', '⚫ Balance', '▲ Return', '» Monthly Avg', '◆ Daily Return', '⚡ Sharpe'],
            [f'${total_profit:,.2f}', f'${end_balance:,.2f}', f'{overall_return:.2f}%', 
             f'{monthly_return:.2f}%', f'{average_daily_return:.2f}%', f'{sharpe_ratio:.2f}'],
            ['★ Gross Profit', '▼ Gross Loss', '⚡ Profit Factor', '⬇ Max Drawdown', '◈ Volatility', '⚡ Sortino'],
            [f'${gross_profit:,.2f}', f'${gross_loss:,.2f}', 
             f'{gross_profit/gross_loss if gross_loss > 0 else "∞"}', f'{max_drawdown:.2f}%',
             f'{volatility:.2f}%', f'{sortino_ratio:.2f}'],
            ['◆ Avg Win', '⚠ Risk/Ruin', '↻ Recovery', '⚡ Calmar Ratio', '# Days', '▲ Ann. Return'],
            [f'${average_win:.2f}', f'{risk_of_ruin:.2f}', f'{recovery_factor:.2f}', 
             f'{calmar_ratio:.2f}', f'{len(dates)}', f'{((1 + overall_return/100)**(252/len(dates)) - 1)*100:.1f}%']
        ]

        # Create and style metrics table
        ax_metrics.axis('off')
        table = ax_metrics.table(
            cellText=metrics_data,
            cellLoc='center',
            loc='center',
            cellColours=[[ORANGE_THEME['background']]*6]*6,  # Use pure black for all cells
            colWidths=[0.17]*6
        )
        
        # Enhanced table styling
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        for cell in table._cells:
            cell_obj = table._cells[cell]
            cell_obj.set_text_props(color=ORANGE_THEME['text'])
            cell_obj.set_edgecolor(ORANGE_THEME['grid'])
            cell_obj.set_linewidth(0.3)  # Thinner lines
            if cell[0] in [0, 2, 4]:  # Headers
                cell_obj.set_text_props(weight='bold', color=ORANGE_THEME['primary'])
            cell_obj.set_facecolor('#000000')  # Pure black background
            cell_obj.PAD = 0.3

        # Add version number in corner
        plt.figtext(0.02, 0.02, "v1.0.0",
                   fontsize=8, color=ORANGE_THEME['muted_text'],
                   ha='left', va='bottom',
                   alpha=0.7, fontfamily='monospace')

        # Enhanced title
        plt.suptitle('Profit Tracker 2025', 
                    color=ORANGE_THEME['primary'],
                    fontsize=14,
                    fontweight='bold',
                    y=0.95)

        # Layout
        plt.tight_layout()
        
        # Show plot
        plt.show()

    except FileNotFoundError:
        print("Please ensure that 'balance_history.json' and 'transactions.json' are in the same directory as this script.")
    except json.JSONDecodeError:
        print("Error decoding JSON. Please check that your JSON files are correctly formatted.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")