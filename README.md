# Algorithmic Trading Bot

An advanced algorithmic trading system built with Python that integrates with Zerodha Kite API for automated trading in the Indian stock market.

## 🚀 Features

- **Automated Trading Bot**: Executes trades based on technical analysis and rating systems
- **Live Dashboard**: Real-time monitoring of positions, P&L, and market data
- **Rating System**: Comprehensive stock rating based on multiple technical indicators
- **Risk Management**: Built-in position sizing and risk controls
- **Multi-timeframe Analysis**: Supports 1min, 3min, 5min, 15min, 30min, and 60min timeframes (Daily removed)
- **Real-time Data**: Live market data streaming and historical data management
- **Web Interface**: User-friendly dashboard for monitoring and control

## 📁 Project Structure

```
├── automated_trading_bot.py    # Main automated trading bot (includes dashboard)
├── kiteConnect.py            # Zerodha Kite API integration
├── watchlist_rating_system.py  # Stock rating algorithms for watchlist selection
├── live_rating_system.py     # Real-time rating updates
├── watchlist.py              # Stock watchlist management
├── indicator_calculator.py    # Technical indicators calculation
├── historical_data/          # Stores downloaded historical price data
├── instruments/              # Market instrument master data
├── logs/                     # Trading logs and system events
├── ratings/                  # Stock rating calculations and history
├── run_automated_bot.py      # Bot launcher script
└── requirements.txt          # Python dependencies
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd algo-trading-bot
```

2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API credentials
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file with the following variables:

```bash
# Zerodha Kite API Credentials
KITE_API_KEY=your_api_key
KITE_API_SECRET=your_api_secret
KITE_REQUEST_TOKEN=your_request_token

# Trading Configuration
CAPITAL=100000
RISK_PERCENTAGE=2
MAX_POSITIONS=10

# Dashboard Configuration
FLASK_ENV=production
FLASK_DEBUG=False
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=5000
```

### Kite API Setup

1. Register at [Kite Connect](https://kite.trade/)
2. Create a new app to get API key and secret
3. Generate access token using the authentication flow
4. Update the credentials in your `.env` file

## 🚀 Usage

### Running the Trading Bot

1. Start the automated trading bot:
```bash
python run_automated_bot.py
```

2. Access the dashboard at `http://localhost:8000`

### Using Shell Scripts

The project includes convenient shell scripts:

```bash
# Set environment variables
source set_env.sh

# Run the automated bot (which includes the dashboard)
./run_automated_bot.py
```

## 📊 Dashboard Features

- **Live Positions**: Real-time view of open positions
- **P&L Tracking**: Profit and loss monitoring
- **Market Data**: Live quotes and charts
- **Trading Controls**: Manual trade execution
- **Rating Display**: Live stock ratings and signals
- **Risk Metrics**: Position sizing and risk analysis

## 🔧 Key Components

### Rating System
- Multi-factor analysis including RSI, MACD, Bollinger Bands
- Momentum and trend analysis
- Volume-based confirmation
- Support and resistance levels

### Risk Management
- Position sizing based on volatility
- Stop-loss and take-profit automation
- Maximum position limits
- Correlation-based diversification

### Technical Indicators
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Moving Averages (SMA, EMA)
- Volume indicators
- Trend following indicators

## ⚠️ Risk Disclaimer

This software is for educational and research purposes only. Trading in financial markets involves substantial risk of loss and is not suitable for all investors. Past performance is not indicative of future results. Please consult with a financial advisor before making investment decisions.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📞 Support

For support and questions, please open an issue on GitHub or contact the maintainers.

## 🔄 Updates

Check the [AUTOMATED_BOT_README.md](AUTOMATED_BOT_README.md) for specific bot configuration and operational details.
