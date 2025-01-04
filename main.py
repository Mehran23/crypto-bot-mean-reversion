import sys
import requests
from concurrent.futures import ThreadPoolExecutor
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QTableWidget, QTableWidgetItem, QHBoxLayout, QComboBox
from PyQt5.QtCore import QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd
import numpy as np

def calculate_zigzag(prices, highs, lows, deviation_threshold=0.04):
    """Calculate ZigZag pivots based on price movements."""
    if not prices or len(prices) < 2:
        return [], []
    
    pivots = [(0, prices[0])]
    last_pivot_idx = 0
    last_pivot_price = prices[0]
    direction = None
    
    for i in range(1, len(prices)):
        current_high = highs[i]
        current_low = lows[i]
        
        if direction is None:
            up_move = (current_high - prices[0]) / prices[0]
            down_move = (current_low - prices[0]) / prices[0]
            
            if abs(up_move) > abs(down_move):
                direction = 1 if up_move > deviation_threshold else None
            else:
                direction = -1 if abs(down_move) > deviation_threshold else None
            continue
            
        if direction == 1:
            if current_high > last_pivot_price:
                last_pivot_idx = i
                last_pivot_price = current_high
            else:
                move = (current_low - last_pivot_price) / last_pivot_price
                if abs(move) >= deviation_threshold:
                    pivots.append((last_pivot_idx, last_pivot_price))
                    last_pivot_idx = i
                    last_pivot_price = current_low
                    direction = -1
        else:
            if current_low < last_pivot_price:
                last_pivot_idx = i
                last_pivot_price = current_low
            else:
                move = (current_high - last_pivot_price) / last_pivot_price
                if abs(move) >= deviation_threshold:
                    pivots.append((last_pivot_idx, last_pivot_price))
                    last_pivot_idx = i
                    last_pivot_price = current_high
                    direction = 1
    
    if last_pivot_idx != pivots[-1][0]:
        pivots.append((last_pivot_idx, last_pivot_price))
    
    indices, values = zip(*pivots)
    return list(indices), list(values)

class TokenScanner:
    """A class to scan and filter tokens using the MEXC API."""
    
    API_BASE = "https://api.mexc.com/api/v3/ticker/24hr"

    def __init__(self):
        self.symbols_data = []

    def fetch_symbols(self):
        """Fetch all available symbols from MEXC."""
        try:
            response = requests.get(self.API_BASE)
            if response.status_code == 200:
                self.symbols_data = response.json()
                print(f"Fetched {len(self.symbols_data)} symbols.")
            else:
                print(f"Failed to fetch symbols: {response.status_code}")
        except Exception as e:
            print(f"Error fetching symbols: {e}")

    def filter_symbols(self, min_volume=70000, keywords_to_exclude=None):
        """Filter symbols based on volume and exclusion criteria."""
        if keywords_to_exclude is None:
            keywords_to_exclude = ['3S', '3L', '2L', '2S', '5L', '5S', 'UP', 'DOWN', 
                                'AUSD', 'USDJ', 'USDP', 'BUSD', 'OUSD', 'USDD', 'FDUSD', 'TUSD']
        
        filtered = [
            symbol for symbol in self.symbols_data
            if float(symbol.get("quoteVolume", 0)) >= min_volume
            and 'USDT' in symbol.get("symbol", "")
            and not any(keyword in symbol.get("symbol", "") for keyword in keywords_to_exclude)
        ]
        return filtered

    def fetch_ohlcv(self, symbol, timeframe, limit=100):
        """Fetch OHLCV data for a specific symbol."""
        ohlcv_api = f"https://api.mexc.com/api/v3/klines?symbol={symbol}&interval={timeframe}&limit={limit}"
        try:
            response = requests.get(ohlcv_api)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Failed to fetch OHLCV for {symbol}: {response.status_code}")
                return []
        except Exception as e:
            print(f"Error fetching OHLCV for {symbol}: {e}")
            return []


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("USDT Token Scanner")

        # Initialize TokenScanner
        self.scanner = TokenScanner()

        # Main widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Layout
        self.layout = QHBoxLayout(self.central_widget)

        # Left Layout: Table and Button
        self.left_layout = QVBoxLayout()
        self.layout.addLayout(self.left_layout)

        # Button to start token scan
        self.scan_button = QPushButton("Start Token Scan")
        self.left_layout.addWidget(self.scan_button)

        # Dropdown for timeframe selection
        self.timeframe_selector = QComboBox()
        self.timeframe_selector.addItems(["1m", "5m", "15m", "60m", "4h", "1d"])  # Timeframe options
        self.left_layout.addWidget(self.timeframe_selector)

        # Table to display token data
        self.token_table = QTableWidget()
        self.left_layout.addWidget(self.token_table)

        # Right Layout: Placeholder for Graph
        self.right_layout = QVBoxLayout()
        self.layout.addLayout(self.right_layout)

        # Matplotlib Canvas
        self.canvas = FigureCanvas(Figure(figsize=(10, 6)))
        self.right_layout.addWidget(self.canvas)
        self.ax = self.canvas.figure.add_subplot(111)  # Add a subplot to the canvas

        # Connect buttons
        self.scan_button.clicked.connect(self.start_token_scan)
        self.token_table.cellClicked.connect(self.show_chart)  # Connect cell click event

    def start_token_scan(self):
        """Scan for tokens ending in USDT and display them."""
        self.scanner.fetch_symbols()
        filtered_symbols = self.scanner.filter_symbols()

        # Update table columns for additional data
        self.token_table.setRowCount(len(filtered_symbols))
        self.token_table.setColumnCount(5)  # Symbol, Base, Quote, Price, Volume
        self.token_table.setHorizontalHeaderLabels(["Symbol", "Base", "Quote", "Price", "Volume"])

        for i, token in enumerate(filtered_symbols):
            # Extract Base and Quote from the symbol
            symbol = token["symbol"]
            base, quote = symbol[:-4], symbol[-4:]  # Split symbol into base and quote
            price = token.get("lastPrice", "N/A")  # Get the last price, default to "N/A"
            volume = token.get("quoteVolume", "N/A")  # Get the volume, default to "N/A"

            # Populate the table
            self.token_table.setItem(i, 0, QTableWidgetItem(symbol))
            self.token_table.setItem(i, 1, QTableWidgetItem(base))
            self.token_table.setItem(i, 2, QTableWidgetItem(quote))
            self.token_table.setItem(i, 3, QTableWidgetItem(str(price)))
            self.token_table.setItem(i, 4, QTableWidgetItem(str(volume)))

    def show_chart(self, row, column):
        """Show the chart for the selected token."""
        if column == 0:  # Ensure we're clicking the "Symbol" column
            symbol = self.token_table.item(row, column).text()
            timeframe = self.timeframe_selector.currentText()  # Get selected timeframe
            self.plot_candlestick_chart(symbol, timeframe)

    def plot_candlestick_chart(self, symbol, timeframe):
        """Plot a candlestick chart with ZigZag pivots connected by a black line and display pivot count."""
        ohlcv = self.scanner.fetch_ohlcv(symbol, timeframe)

        if not ohlcv:
            print(f"No data available for {symbol} at {timeframe}")
            self.ax.clear()
            self.ax.set_title(f"No data available for {symbol} ({timeframe})", fontsize=14)
            self.canvas.draw()
            return

        # Process OHLCV data
        df = pd.DataFrame(
            ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume']
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

        # Extract prices, highs, and lows
        prices = df['close'].astype(float).tolist()
        highs = df['high'].astype(float).tolist()
        lows = df['low'].astype(float).tolist()

        # Calculate ZigZag pivots
        pivot_indices, pivot_values = calculate_zigzag(prices, highs, lows)

        # Count the number of pivots (red points)
        pivot_count = len(pivot_indices)

        # Clear the previous plot
        self.ax.clear()

        # Plot the candlestick-like chart
        self.ax.plot(df['timestamp'], prices, label='Close Price', color='blue')
        self.ax.fill_between(
            df['timestamp'],
            df['low'].astype(float),
            df['high'].astype(float),
            color='lightgray',
            alpha=0.3,
            label='High-Low Range'
        )

        # Plot ZigZag pivots
        pivot_timestamps = [df['timestamp'][i] for i in pivot_indices]
        self.ax.plot(pivot_timestamps, pivot_values, color='black', label=f'ZigZag Line')
        self.ax.scatter(pivot_timestamps, pivot_values, color='red', label=f'ZigZag Pivots ({pivot_count})')

        # Set titles and labels
        self.ax.set_title(f'{symbol} Candlestick Chart ({timeframe})', fontsize=14)
        self.ax.set_xlabel('Timestamp', fontsize=12)
        self.ax.set_ylabel('Price', fontsize=12)
        self.ax.legend(fontsize=10)

        # Improve readability of x-axis labels
        self.ax.tick_params(axis='x', labelrotation=45, labelsize=10)
        self.ax.tick_params(axis='y', labelsize=10)
        self.canvas.figure.tight_layout()  # Adjust layout to prevent overlap

        # Render the chart
        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())