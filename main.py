import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QTableWidget, QTableWidgetItem, QHBoxLayout, QComboBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import ccxt
import pandas as pd


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("USDT Token Scanner")

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
        self.timeframe_selector.addItems(["1m", "5m", "15m", "1h", "4h", "1d"])  # Timeframe options
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
        exchange = ccxt.mexc()  # Use MEXC as the exchange
        markets = exchange.load_markets()  # Load all available markets

        # Filter tokens ending with USDT
        usdt_markets = [
            {"symbol": symbol, "base": market['base'], "quote": market['quote']}
            for symbol, market in markets.items()
            if market['quote'] == 'USDT'
        ]

        # Populate table with token data
        self.token_table.setRowCount(len(usdt_markets))
        self.token_table.setColumnCount(3)
        self.token_table.setHorizontalHeaderLabels(["Symbol", "Base", "Quote"])

        for i, token in enumerate(usdt_markets):
            self.token_table.setItem(i, 0, QTableWidgetItem(token["symbol"]))
            self.token_table.setItem(i, 1, QTableWidgetItem(token["base"]))
            self.token_table.setItem(i, 2, QTableWidgetItem(token["quote"]))

    def show_chart(self, row, column):
        """Show the chart for the selected token."""
        if column == 0:  # Ensure we're clicking the "Symbol" column
            symbol = self.token_table.item(row, column).text()
            timeframe = self.timeframe_selector.currentText()  # Get selected timeframe
            self.plot_candlestick_chart(symbol, timeframe)

    def plot_candlestick_chart(self, symbol, timeframe):
        """Plot a candlestick chart for the given symbol and timeframe."""
        exchange = ccxt.mexc()
        limit = 100  # Fetch the last 100 candles

        # Fetch OHLCV data
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

        # Convert to DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # Clear previous plot
        self.ax.clear()

        # Plot candlestick-like chart (high-low range with close price)
        self.ax.plot(df['timestamp'], df['close'], label='Close Price', color='blue')
        self.ax.fill_between(df['timestamp'], df['low'], df['high'], color='lightgray', alpha=0.3, label='High-Low Range')

        # Set titles and labels
        self.ax.set_title(f'{symbol} Candlestick Chart ({timeframe})', fontsize=14)
        self.ax.set_xlabel('Timestamp', fontsize=12)
        self.ax.set_ylabel('Price', fontsize=12)
        self.ax.legend(fontsize=10)

        # Improve readability of x-axis labels
        self.ax.tick_params(axis='x', labelrotation=45, labelsize=10)  # Rotate timestamps
        self.ax.tick_params(axis='y', labelsize=10)
        self.canvas.figure.tight_layout()  # Adjust layout to prevent overlap

        # Render the canvas
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())
