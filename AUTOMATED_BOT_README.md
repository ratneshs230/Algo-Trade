# Automated Trading Bot

## Overview

This automated trading bot implements a fully autonomous workflow that:

1. **Validates access tokens** on startup
2. **Fetches historical data** for watchlist stocks
3. **Rates and filters** stocks to get top 20
4. **Continuously monitors** top 20 stocks using WebSocket streaming data
5. **Refreshes data** and re-rates stocks every 10 minutes
6. **Dynamically updates** live rating systems based on new top 20 lists

## Architecture

### Core Components

- **AutomatedTradingBot**: Main controller class
- **WebSocketManager**: Manages KiteTicker WebSocket connections with connection pooling
- **DataCache**: Caching mechanism for frequently accessed data
- **LiveRatingSystem**: Real-time stock rating using streaming data

### Threading Model

- **Main Thread**: Startup, coordination, status monitoring
- **Scheduler Thread**: 10-minute periodic data refresh tasks
- **WebSocket Thread**: Real-time data streaming from KiteTicker
- **Worker Thread Pool**: Parallel processing of data fetching and rating tasks
- **Save Thread**: Periodic saving of live ratings to files

## Features

### 🔄 Automated Workflow
- Fully automated operation with minimal user intervention
- Intelligent error handling with retry mechanisms
- Graceful shutdown handling

### 📡 Real-time Data Streaming
- KiteTicker WebSocket integration for live market data
- Connection pooling for efficient resource management
- Automatic reconnection on connection failures

### 📊 Dynamic Stock Selection
- Continuous rating of all watchlist stocks
- Automatic selection of top 20 performing stocks
- Dynamic switching of live monitoring based on performance

### 💾 Intelligent Caching
- Caches frequently accessed data to reduce API calls
- Configurable cache expiry times
- Thread-safe cache operations

### 🔧 Robust Error Handling
- 30-second retry delays for failed operations
- Maximum retry attempts configuration
- Detailed logging for troubleshooting

## Installation

### Prerequisites

1. Python 3.8 or higher
2. Valid KiteConnect API credentials
3. All required Python packages

### Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment variables:**
   Create a `.env` file with your KiteConnect credentials:
   ```env
   KITE_API_KEY=your_api_key_here
   KITE_API_SECRET=your_api_secret_here
   KITE_USER_ID=your_user_id_here
   KITE_PASSWORD=your_password_here
   ```

3. **Ensure watchlist is configured:**
   Make sure your `watchlist.py` file contains the stocks you want to monitor.

## Usage

### Quick Start

Run the automated bot using the launcher:

```bash
python run_automated_bot.py
```

### Manual Configuration

You can also run the bot with custom configuration:

```python
from automated_trading_bot import AutomatedTradingBot, BotConfig

# Create custom configuration
config = BotConfig(
    refresh_interval_minutes=10,    # Refresh every 10 minutes
    websocket_retry_delay=30,       # Wait 30 seconds before retry
    top_stocks_count=20,            # Track top 20 stocks
    max_worker_threads=10,          # Use 10 worker threads
    enable_caching=True,            # Enable caching
    cache_expiry_minutes=5          # Cache expires in 5 minutes
)

# Create and run bot
bot = AutomatedTradingBot(config)
bot.run()
```

## Configuration Options

### BotConfig Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `refresh_interval_minutes` | 10 | How often to refresh data and re-rate stocks |
| `websocket_retry_delay` | 30 | Seconds to wait before retrying WebSocket |
| `max_retry_attempts` | 5 | Maximum retry attempts for operations |
| `top_stocks_count` | 20 | Number of top stocks to track live |
| `max_worker_threads` | 10 | Maximum worker threads for parallel operations |
| `websocket_timeout` | 10 | WebSocket connection timeout |
| `enable_caching` | True | Enable caching mechanisms |
| `cache_expiry_minutes` | 5 | Cache expiry time in minutes |

## Output Files

The bot generates several output files:

### Log Files
- `logs/automated_bot_YYYYMMDD.log` - Daily log files
- `automated_bot.log` - Current session log

### Rating Files
- `ratings/automated_live_ratings.json` - Live ratings updated every minute
- `ratings/filtered_20.json` - Top 20 stocks from periodic rating
- `historical_data/` - Historical data for all watchlist stocks

### Data Structure

```json
{
  "timestamp": "2025-07-31T02:45:00",
  "rating_type": "Automated Live Rating System",
  "total_stocks": 20,
  "top_20_live_ratings": [
    {
      "trading_symbol": "RELIANCE",
      "final_rating": 7.85,
      "rating_text": "Strong Buy",
      "emoji": "🟢",
      "composite_score": 0.785,
      "weighted_scores": {
        "1minute": 0.30,
        "3minute": 0.25,
        "5minute": 0.20,
        "15minute": 0.15,
        "30minute": 0.06,
        "60minute": 0.03,
        "daily": 0.01
      }
    }
  ]
}
```

## Monitoring

### Real-time Status

The bot logs status updates every 5 minutes:

```
📊 STATUS: 18 active live ratings, 2 errors, WebSocket: ✅
🏆 TOP 5 LIVE RATINGS:
  1. RELIANCE     Score:   7.85 (Strong Buy)
  2. TCS          Score:   6.42 (Buy)
  3. INFY         Score:   5.78 (Buy)
  4. HDFC         Score:   4.23 (Buy)
  5. ICICI        Score:   3.91 (Buy)
```

### Key Metrics

- **Active Live Ratings**: Number of stocks with successful live ratings
- **Error Count**: Number of stocks with rating errors
- **WebSocket Status**: Connection status (✅/❌)
- **Top Performers**: Real-time ranking of best performing stocks

## Troubleshooting

### Common Issues

1. **Access Token Errors**
   - Ensure your API credentials are correct
   - Check if you need to generate a new access token
   - Verify 2FA setup for your Kite account

2. **WebSocket Connection Issues**
   - Check internet connectivity
   - Verify API rate limits haven't been exceeded
   - Look for firewall blocking WebSocket connections

3. **Historical Data Fetch Failures**
   - Check if market is open for data availability
   - Verify instrument tokens are valid
   - Ensure sufficient API quota

4. **Memory Issues**
   - Reduce `top_stocks_count` if running out of memory
   - Lower `max_worker_threads` for systems with limited resources
   - Enable caching to reduce redundant API calls

### Debug Mode

Enable debug logging by modifying the logging level:

```python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

## Performance Optimization

### Recommended Settings

For optimal performance:

```python
config = BotConfig(
    refresh_interval_minutes=10,     # Balance between freshness and API usage
    max_worker_threads=8,           # Adjust based on CPU cores
    enable_caching=True,            # Always enable for better performance
    cache_expiry_minutes=3,         # Shorter expiry for more recent data
    websocket_timeout=15            # Longer timeout for stable connections
)
```

### System Requirements

- **CPU**: Multi-core processor (4+ cores recommended)
- **RAM**: 4GB minimum, 8GB recommended
- **Network**: Stable internet connection with low latency
- **Storage**: 1GB free space for logs and data files

## Safety Features

- **Graceful Shutdown**: Handles Ctrl+C and system signals properly
- **Error Recovery**: Automatic retry mechanisms for failed operations
- **Data Validation**: Validates all incoming data before processing
- **Thread Safety**: All shared data structures are thread-safe
- **Connection Monitoring**: Continuous monitoring of WebSocket connections

## Stopping the Bot

To stop the bot gracefully:

1. Press `Ctrl+C` in the terminal
2. The bot will:
   - Stop all background threads
   - Disconnect WebSocket connections
   - Save current state
   - Clean up resources

## Support

For issues or questions:

1. Check the log files for detailed error messages
2. Verify your configuration parameters
3. Ensure all dependencies are properly installed
4. Check KiteConnect API documentation for any recent changes

## License

This automated trading bot is part of the larger trading system project. Use at your own risk and ensure compliance with all applicable trading regulations.
