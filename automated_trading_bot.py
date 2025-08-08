"""
Automated Trading Bot - Main Controller

This module implements a fully automated trading workflow that:
1. Validates access tokens on startup
2. Fetches historical data for watchlist stocks
3. Rates stocks and filters top 20
4. Continuously rates top 20 stocks using WebSocket streaming data
5. Refreshes data and re-rates stocks every 10 minutes
"""

import json
import os
import time
import threading
import queue
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor
import signal
import sys

# KiteConnect imports
from kiteconnect import KiteTicker
from kiteConnect import KiteTrader, Config, KiteLoginAutomation
from watchlist import get_watchlist
from rating_system import StockRatingSystem
from live_rating_system import LiveRatingSystem
from precision_single_trade_strategy import PrecisionSingleTradeStrategy

logger = logging.getLogger(__name__)


@dataclass
class BotConfig:
    """Configuration for the automated trading bot"""
    # Timing settings
    refresh_interval_minutes: int = 10  # How often to refresh data and re-rate stocks
    websocket_retry_delay: int = 30     # Seconds to wait before retrying WebSocket
    max_retry_attempts: int = 5         # Maximum retry attempts for operations
    
    # Stock selection
    top_stocks_count: int = 20          # Number of top stocks to track live
    
    # Threading settings
    max_worker_threads: int = 3         # Reduced for API rate limit compliance
    websocket_timeout: int = 10         # WebSocket connection timeout
    
    # API rate limiting
    api_request_delay: float = 0.6      # Delay between API requests (600ms)
    rate_limit_wait: int = 20           # Wait time when rate limit is hit
    
    # Caching settings
    enable_caching: bool = True         # Enable caching mechanisms
    cache_expiry_minutes: int = 10      # Increased cache time to reduce API calls


class WebSocketManager:
    """Manages KiteTicker WebSocket connections with connection pooling"""
    
    def __init__(self, api_key: str, access_token: str, config: BotConfig):
        self.api_key = api_key
        self.access_token = access_token
        self.config = config
        self.kws = None
        self.connected = False
        self.subscribed_tokens: Set[str] = set()
        self.tick_queue = queue.Queue()
        self.connection_lock = threading.Lock()
        self.retry_count = 0
        
        # Callback handlers
        self.tick_handlers: Dict[str, callable] = {}
        
        logger.info("WebSocket Manager initialized")
    
    def initialize_websocket(self):
        """Initialize KiteTicker WebSocket connection"""
        try:
            with self.connection_lock:
                if self.kws:
                    try:
                        self.kws.close()
                    except:
                        pass
                
                self.kws = KiteTicker(self.api_key, self.access_token)
                
                # Set up event handlers
                self.kws.on_ticks = self._on_ticks
                self.kws.on_connect = self._on_connect
                self.kws.on_close = self._on_close
                self.kws.on_error = self._on_error
                self.kws.on_reconnect = self._on_reconnect
                self.kws.on_noreconnect = self._on_noreconnect
                
                logger.info("KiteTicker initialized successfully")
                return True
                
        except Exception as e:
            logger.error(f"Error initializing WebSocket: {str(e)}")
            return False
    
    def connect(self):
        """Connect to WebSocket with retry logic"""
        for attempt in range(self.config.max_retry_attempts):
            try:
                if not self.kws:
                    if not self.initialize_websocket():
                        continue
                
                logger.info(f"Attempting WebSocket connection (attempt {attempt + 1})")
                
                # Start connection in a separate thread
                connection_thread = threading.Thread(
                    target=self.kws.connect,
                    kwargs={'threaded': True}
                )
                connection_thread.daemon = True
                connection_thread.start()
                
                # Wait for connection
                time.sleep(2)
                
                if self.connected:
                    logger.info("WebSocket connected successfully")
                    self.retry_count = 0
                    return True
                
            except Exception as e:
                logger.error(f"WebSocket connection attempt {attempt + 1} failed: {str(e)}")
            
            if attempt < self.config.max_retry_attempts - 1:
                logger.info(f"Waiting {self.config.websocket_retry_delay} seconds before retry")
                time.sleep(self.config.websocket_retry_delay)
        
        logger.error("Failed to connect WebSocket after all attempts")
        return False
    
    def subscribe_stocks(self, instrument_tokens: List[str]):
        """Subscribe to stock instrument tokens"""
        try:
            if not self.connected:
                logger.warning("WebSocket not connected, cannot subscribe")
                return False
            
            # Convert to integers for KiteTicker
            int_tokens = [int(token) for token in instrument_tokens if token.isdigit()]
            
            if int_tokens:
                self.kws.subscribe(int_tokens)
                self.kws.set_mode(self.kws.MODE_FULL, int_tokens)
                
                self.subscribed_tokens.update(instrument_tokens)
                logger.info(f"Subscribed to {len(int_tokens)} instruments")
                return True
            else:
                logger.warning("No valid instrument tokens to subscribe")
                return False
                
        except Exception as e:
            logger.error(f"Error subscribing to stocks: {str(e)}")
            return False
    
    def unsubscribe_stocks(self, instrument_tokens: List[str]):
        """Unsubscribe from stock instrument tokens"""
        try:
            if not self.connected:
                return
            
            int_tokens = [int(token) for token in instrument_tokens if token.isdigit()]
            
            if int_tokens:
                self.kws.unsubscribe(int_tokens)
                self.subscribed_tokens -= set(instrument_tokens)
                logger.info(f"Unsubscribed from {len(int_tokens)} instruments")
                
        except Exception as e:
            logger.error(f"Error unsubscribing from stocks: {str(e)}")
    
    def register_tick_handler(self, instrument_token: str, handler: callable):
        """Register a callback handler for specific instrument tick data"""
        self.tick_handlers[instrument_token] = handler
    
    def unregister_tick_handler(self, instrument_token: str):
        """Unregister tick handler for an instrument"""
        if instrument_token in self.tick_handlers:
            del self.tick_handlers[instrument_token]
    
    def _on_ticks(self, ws, ticks):
        """Handle incoming tick data"""
        try:
            for tick in ticks:
                instrument_token = str(tick.get('instrument_token', ''))
                
                # Add to general queue
                self.tick_queue.put({
                    'instrument_token': instrument_token,
                    'timestamp': datetime.now(),
                    'tick_data': tick
                })
                
                # NEW: Handle precision strategy active trade updates
                if (self.enable_precision_strategy and 
                    self.precision_strategy and 
                    hasattr(self.precision_strategy, 'active_trade') and
                    self.precision_strategy.active_trade):
                    
                    active_symbol = self.precision_strategy.active_trade.setup.candidate.symbol
                    active_token = str(self.precision_strategy.active_trade.setup.candidate.instrument_token)
                    
                    if instrument_token == active_token:
                        try:
                            # Update active trade with real-time tick data
                            trade_still_active = self.precision_strategy.update_active_trade(tick)
                            if not trade_still_active:
                                logger.info(f"Trade closed for {active_symbol}")
                        except Exception as e:
                            logger.error(f"Error updating precision strategy trade: {str(e)}")
                
                # Call specific handler if registered
                if instrument_token in self.tick_handlers:
                    try:
                        self.tick_handlers[instrument_token](tick)
                    except Exception as e:
                        logger.error(f"Error in tick handler for {instrument_token}: {str(e)}")
                        
        except Exception as e:
            logger.error(f"Error processing ticks: {str(e)}")
    
    def _on_connect(self, ws, response):
        """Handle WebSocket connection"""
        self.connected = True
        logger.info("WebSocket connected")
    
    def _on_close(self, ws, code, reason):
        """Handle WebSocket disconnection"""
        self.connected = False
        logger.warning(f"WebSocket closed: {code} - {reason}")
    
    def _on_error(self, ws, code, reason):
        """Handle WebSocket errors"""
        logger.error(f"WebSocket error: {code} - {reason}")
    
    def _on_reconnect(self, ws, attempts_count):
        """Handle WebSocket reconnection"""
        logger.info(f"WebSocket reconnecting (attempt {attempts_count})")
    
    def _on_noreconnect(self, ws):
        """Handle WebSocket no reconnection"""
        self.connected = False
        logger.warning("WebSocket will not reconnect")
        
        # Attempt manual reconnection
        threading.Thread(target=self._attempt_reconnection, daemon=True).start()
    
    def _attempt_reconnection(self):
        """Attempt manual reconnection"""
        time.sleep(self.config.websocket_retry_delay)
        logger.info("Attempting manual WebSocket reconnection")
        self.connect()
    
    def disconnect(self):
        """Disconnect WebSocket"""
        try:
            if self.kws:
                self.kws.close()
            self.connected = False
            logger.info("WebSocket disconnected")
        except Exception as e:
            logger.error(f"Error disconnecting WebSocket: {str(e)}")


class DataCache:
    """Simple caching mechanism for frequently accessed data"""
    
    def __init__(self, config: BotConfig):
        self.config = config
        self.cache: Dict[str, Dict] = {}
        self.cache_lock = threading.Lock()
    
    def get(self, key: str) -> Optional[any]:
        """Get cached data if not expired"""
        if not self.config.enable_caching:
            return None
        
        with self.cache_lock:
            if key in self.cache:
                entry = self.cache[key]
                if datetime.now() < entry['expires']:
                    return entry['data']
                else:
                    del self.cache[key]
        return None
    
    def set(self, key: str, data: any):
        """Cache data with expiration"""
        if not self.config.enable_caching:
            return
        
        with self.cache_lock:
            self.cache[key] = {
                'data': data,
                'expires': datetime.now() + timedelta(minutes=self.config.cache_expiry_minutes)
            }
    
    def clear(self):
        """Clear all cached data"""
        with self.cache_lock:
            self.cache.clear()


@dataclass
class StockUniverseVersion:
    """Represents a versioned stock universe"""
    version: int
    timestamp: datetime
    stocks: List[Dict]
    status: str  # 'active', 'transitioning', 'deprecated'
    
class AutomatedTradingBot:
    """Main automated trading bot controller"""
    
    def __init__(self, config: BotConfig = None, enable_precision_strategy: bool = False):
        self.config = config or BotConfig()
        self.running = False
        self.enable_precision_strategy = enable_precision_strategy
        
        # Core components
        self.access_token = None
        self.kite_trader = None
        self.websocket_manager = None
        self.stock_rating_system = None
        
        # NEW: Precision Single Trade Strategy
        self.precision_strategy = None
        
        # Data management
        self.instruments = []
        self.watchlist_stocks = []
        self.top_20_stocks = []
        
        # Versioned stock universe management
        self.current_version = 0
        self.stock_universe_versions: Dict[int, StockUniverseVersion] = {}
        self.live_rating_systems: Dict[str, Dict] = {}  # symbol -> {'system': LiveRatingSystem, 'version': int}
        self.transition_state = 'stable'  # 'stable', 'batch_processing', 'transitioning'
        
        # Threading
        self.scheduler_thread = None
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.max_worker_threads)
        
        # Caching
        self.data_cache = DataCache(self.config)
        
        # Synchronization
        self.top_stocks_lock = threading.Lock()
        self.universe_lock = threading.Lock()
        self.shutdown_event = threading.Event()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("Automated Trading Bot initialized with versioned stock universe")
        if enable_precision_strategy:
            logger.info("Precision Single Trade Strategy: ENABLED")
        else:
            logger.info("Precision Single Trade Strategy: DISABLED (Analysis mode only)")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown")
        self.stop()
    
    def validate_and_get_access_token(self) -> bool:
        """Validate existing access token or generate a new one"""
        try:
            logger.info("=== ACCESS TOKEN VALIDATION ===")
            
            # Check for existing token file
            token_file = 'access_token.json'
            today = datetime.now().date()
            
            if os.path.exists(token_file):
                try:
                    with open(token_file, 'r') as f:
                        token_data = json.load(f)
                    
                    token_date = datetime.strptime(token_data.get('date', ''), '%Y-%m-%d').date()
                    access_token = token_data.get('access_token', '')
                    
                    if token_date == today and access_token:
                        logger.info(f"Valid access token found for today ({today})")
                        
                        # Test the token
                        try:
                            test_trader = KiteTrader(api_key=Config.API_KEY, access_token=access_token)
                            margins = test_trader.get_margins()
                            if margins:
                                logger.info("Access token validated successfully via API test")
                                self.access_token = access_token
                                return True
                            else:
                                logger.warning("Access token validation failed - API test returned empty response")
                        except Exception as e:
                            logger.warning(f"Access token validation failed - API error: {str(e)}")
                    else:
                        if token_date != today:
                            logger.info(f"Existing token is from {token_date}, need new token for {today}")
                        else:
                            logger.warning("Existing token file is corrupted or empty")
                            
                except Exception as e:
                    logger.error(f"Error reading existing token file: {str(e)}")
            else:
                logger.info("No access token file found")
            
            # Generate new access token
            logger.info("Generating new access token...")
            automation = KiteLoginAutomation()
            access_token = automation.login()
            
            logger.info("New access token generated successfully")
            self.access_token = access_token
            return True
            
        except Exception as e:
            logger.error(f"Error in access token validation/generation: {str(e)}")
            return False
    
    def initialize_components(self) -> bool:
        """Initialize all bot components"""
        try:
            logger.info("=== INITIALIZING COMPONENTS ===")
            
            # Initialize KiteTrader
            self.kite_trader = KiteTrader(api_key=Config.API_KEY, access_token=self.access_token)
            logger.info("KiteTrader initialized")
            
            # Initialize WebSocket Manager
            self.websocket_manager = WebSocketManager(
                api_key=Config.API_KEY,
                access_token=self.access_token,
                config=self.config
            )
            
            # Initialize Stock Rating System
            self.stock_rating_system = StockRatingSystem()
            logger.info("Stock Rating System initialized")
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            return False
    
    def initialize_precision_strategy(self) -> bool:
        """Initialize Precision Single Trade Strategy"""
        try:
            if not self.enable_precision_strategy:
                logger.info("Precision strategy disabled - skipping initialization")
                return True
            
            if not self.kite_trader or not self.instruments:
                logger.error("Cannot initialize precision strategy - missing dependencies")
                return False
            
            # Load instrument data with tick sizes
            instruments_file = 'instruments/nse_instruments_watchlist.json'
            if not os.path.exists(instruments_file):
                logger.error(f"Instruments file not found: {instruments_file}")
                return False
            
            with open(instruments_file, 'r') as f:
                instruments_data = json.load(f)
            
            # Initialize precision strategy
            self.precision_strategy = PrecisionSingleTradeStrategy(
                kite_trader=self.kite_trader,
                instruments=self.instruments,
                instruments_data=instruments_data,
                margin_percentage=0.95  # Use 95% of available margin
            )
            
            logger.info("✅ Precision Single Trade Strategy initialized")
            logger.info(f"   - Loaded {len(instruments_data)} instrument definitions")
            logger.info(f"   - Ready to trade {len(self.instruments)} watchlist stocks")
            logger.info(f"   - Single position control: ACTIVE")
            logger.info(f"   - Margin allocation: 95%")
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing precision strategy: {str(e)}")
            return False
    
    def fetch_instruments_and_watchlist(self) -> bool:
        """Fetch instruments and watchlist data"""
        try:
            logger.info("=== FETCHING INSTRUMENTS AND WATCHLIST ===")
            
            # Get watchlist stocks
            self.watchlist_stocks = get_watchlist()
            logger.info(f"Loaded {len(self.watchlist_stocks)} watchlist stocks")
            
            # Get NSE instruments
            all_instruments = self.kite_trader.get_instruments(exchange='NSE')
            logger.info(f"Retrieved {len(all_instruments)} total NSE instruments")
            
            # Filter instruments for watchlist stocks
            filtered_instruments = []
            for instrument in all_instruments:
                trading_symbol = instrument.get('tradingsymbol', '')
                if trading_symbol in self.watchlist_stocks and instrument.get('instrument_type') == 'EQ':
                    filtered_instruments.append(instrument)
            
            self.instruments = filtered_instruments
            logger.info(f"Filtered to {len(self.instruments)} watchlist instruments")
            
            return len(self.instruments) > 0
            
        except Exception as e:
            logger.error(f"Error fetching instruments and watchlist: {str(e)}")
            return False
    
    def fetch_historical_data(self) -> bool:
        """Fetch historical data for watchlist stocks with rate limiting"""
        try:
            logger.info("=== FETCHING HISTORICAL DATA (RATE LIMITED) ===")
            
            # Check cache first
            cache_key = f"historical_data_{datetime.now().strftime('%Y%m%d_%H')}"  # Hourly cache
            cached_result = self.data_cache.get(cache_key)
            if cached_result:
                logger.info("Using cached historical data")
                return True
            
            # Define timeframes
            timeframes = {
                '1minute': {'interval': 'minute', 'days_back': 1, 'min_rows': 120},
                '3minute': {'interval': '3minute', 'days_back': 3, 'min_rows': 120},
                '5minute': {'interval': '5minute', 'days_back': 5, 'min_rows': 120},
                '15minute': {'interval': '15minute', 'days_back': 15, 'min_rows': 120},
                '30minute': {'interval': '30minute', 'days_back': 30, 'min_rows': 120},
                '60minute': {'interval': '60minute', 'days_back': 60, 'min_rows': 120},
                'daily': {'interval': 'day', 'days_back': 180, 'min_rows': 120}
            }
            
            # Create directory
            os.makedirs('historical_data', exist_ok=True)
            
            successful_count = 0
            total_count = len(self.instruments)
            
            # Process instruments completely sequentially - one stock at a time
            logger.info("Processing instruments one by one to respect API rate limits...")
            
            for i, instrument in enumerate(self.instruments, 1):
                try:
                    trading_symbol = instrument.get('tradingsymbol', '')
                    instrument_token = instrument.get('instrument_token', '')
                    
                    logger.info(f"[{i}/{len(self.instruments)}] Processing {trading_symbol}...")
                    
                    # Create stock directory
                    stock_dir = os.path.join('historical_data', trading_symbol)
                    os.makedirs(stock_dir, exist_ok=True)
                    
                    stock_success = True
                    
                    # Fetch each timeframe one by one for this stock
                    for timeframe, params in timeframes.items():
                        try:
                            logger.info(f"  Fetching {timeframe} data for {trading_symbol}...")
                            
                            to_date = datetime.now()
                            from_date = to_date - timedelta(days=params['days_back'])
                            
                            # Make API request
                            historical_data = self.kite_trader.get_historical_data(
                                instrument_token=int(instrument_token),
                                from_date=from_date,
                                to_date=to_date,
                                interval=params['interval']
                            )
                            
                            if historical_data and len(historical_data) >= params['min_rows']:
                                filename = f"{trading_symbol}_{timeframe}.json"
                                filepath = os.path.join(stock_dir, filename)
                                
                                with open(filepath, 'w') as f:
                                    json.dump(historical_data, f, indent=2, default=str)
                                
                                logger.info(f"    ✓ {timeframe}: {len(historical_data)} candles saved")
                            else:
                                logger.warning(f"    ✗ {timeframe}: Insufficient data ({len(historical_data) if historical_data else 0} candles)")
                                
                        except Exception as e:
                            if "Too many requests" in str(e) or "429" in str(e):
                                wait_time = getattr(self.config, 'rate_limit_wait', 20)
                                logger.warning(f"    Rate limit hit for {trading_symbol} {timeframe}")
                                logger.warning(f"    Waiting {wait_time} seconds before retry...")
                                time.sleep(wait_time)
                                
                                # Single retry
                                try:
                                    logger.info(f"    Retrying {timeframe} for {trading_symbol}...")
                                    historical_data = self.kite_trader.get_historical_data(
                                        instrument_token=int(instrument_token),
                                        from_date=from_date,
                                        to_date=to_date,
                                        interval=params['interval']
                                    )
                                    
                                    if historical_data and len(historical_data) >= params['min_rows']:
                                        filename = f"{trading_symbol}_{timeframe}.json"
                                        filepath = os.path.join(stock_dir, filename)
                                        
                                        with open(filepath, 'w') as f:
                                            json.dump(historical_data, f, indent=2, default=str)
                                        
                                        logger.info(f"    ✓ {timeframe}: {len(historical_data)} candles saved (retry)")
                                    else:
                                        logger.warning(f"    ✗ {timeframe}: Retry failed - insufficient data")
                                        stock_success = False
                                        
                                except Exception as retry_e:
                                    logger.error(f"    ✗ {timeframe}: Retry failed - {str(retry_e)}")
                                    stock_success = False
                            else:
                                logger.error(f"    ✗ {timeframe}: Error - {str(e)}")
                                stock_success = False
                    
                    if stock_success:
                        successful_count += 1
                        logger.info(f"  ✅ {trading_symbol} completed successfully")
                    else:
                        logger.warning(f"  ⚠️ {trading_symbol} completed with some errors")
                    
                    # Continue to next stock immediately
                    pass
                        
                except Exception as e:
                    logger.error(f"  ❌ {trading_symbol}: Critical error - {str(e)}")
            
            logger.info(f"Historical data fetch completed: {successful_count}/{total_count} successful")
            
            # Cache the result
            self.data_cache.set(cache_key, True)
            
            return successful_count > 0
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {str(e)}")
            return False
    
    def create_new_stock_universe_version(self, new_selected_stocks: List[Dict]) -> int:
        """Create a new versioned stock universe"""
        with self.universe_lock:
            new_version = self.current_version + 1
            
            # Create new universe version
            universe_version = StockUniverseVersion(
                version=new_version,
                timestamp=datetime.now(),
                stocks=new_selected_stocks.copy(),
                status='transitioning'
            )
            
            # Store the new version
            self.stock_universe_versions[new_version] = universe_version
            
            logger.info(f"Created stock universe version {new_version} with {len(new_selected_stocks)} stocks")
            return new_version

    def rate_and_filter_stocks(self) -> bool:
        """Rate all stocks and filter top 10 and bottom 10 with versioned universe"""
        try:
            logger.info("=== RATING AND FILTERING STOCKS (VERSIONED UNIVERSE) ===")
            
            # Set transition state
            with self.universe_lock:
                self.transition_state = 'batch_processing'
            
            # Check cache first
            cache_key = f"stock_ratings_{datetime.now().strftime('%Y%m%d_%H%M')}"
            cached_ratings = self.data_cache.get(cache_key)
            
            if cached_ratings:
                logger.info("Using cached stock ratings")
                # Get top 10 and bottom 10 from cached ratings
                if len(cached_ratings) >= 20:
                    top_10 = cached_ratings[:10]
                    bottom_10 = cached_ratings[-10:]
                    new_selected_stocks = top_10 + bottom_10
                else:
                    # If less than 20 stocks, take all available
                    new_selected_stocks = cached_ratings[:self.config.top_stocks_count]
            else:
                # Get symbols for rating
                symbols = [inst.get('tradingsymbol', '') for inst in self.instruments]
                symbols = [s for s in symbols if s]
                
                if not symbols:
                    logger.error("No symbols found for rating")
                    return False
                
                logger.info(f"Rating {len(symbols)} stocks...")
                
                # Rate all stocks
                reports = self.stock_rating_system.rate_multiple_stocks(symbols, self.instruments)
                
                # Filter out error reports and sort by score
                valid_reports = [r for r in reports if 'error' not in r]
                valid_reports.sort(key=lambda x: x.get('final_score', 0), reverse=True)
                
                # Cache the ratings
                self.data_cache.set(cache_key, valid_reports)
                
                # Get top 10 and bottom 10
                if len(valid_reports) >= 20:
                    top_10 = valid_reports[:10]  # Top 10 highest rated
                    bottom_10 = valid_reports[-10:]  # Bottom 10 lowest rated
                    new_selected_stocks = top_10 + bottom_10
                    logger.info("Selected top 10 and bottom 10 stocks for monitoring")
                else:
                    # If less than 20 stocks available, take all
                    new_selected_stocks = valid_reports[:self.config.top_stocks_count]
                    logger.info(f"Less than 20 stocks available, selected {len(new_selected_stocks)} stocks")
            
            # Create new universe version
            new_version = self.create_new_stock_universe_version(new_selected_stocks)
            
            # Update selected stocks with thread safety
            with self.top_stocks_lock:
                old_symbols = set(stock.get('symbol', '') for stock in self.top_20_stocks)
                new_symbols = set(stock.get('symbol', '') for stock in new_selected_stocks)
                
                self.top_20_stocks = new_selected_stocks
                
                # Log changes
                added = new_symbols - old_symbols
                removed = old_symbols - new_symbols
                
                if added:
                    logger.info(f"Added to monitored stocks: {', '.join(added)}")
                if removed:
                    logger.info(f"Removed from monitored stocks: {', '.join(removed)}")
            
            # Log the selected stocks with clear separation
            logger.info(f"Selected {len(self.top_20_stocks)} stocks for live monitoring (Version {new_version}):")
            
            # Separate and display top 10 and bottom 10
            if len(self.top_20_stocks) == 20:
                logger.info("📈 TOP 10 HIGHEST RATED STOCKS:")
                for i, stock in enumerate(self.top_20_stocks[:10], 1):
                    symbol = stock.get('symbol', 'N/A')
                    score = stock.get('final_score', 0)
                    rating = stock.get('rating', 'N/A')
                    logger.info(f"  {i:2d}. {symbol:<12} Score: {score:6.2f} Rating: {rating}")
                
                logger.info("📉 BOTTOM 10 LOWEST RATED STOCKS:")
                for i, stock in enumerate(self.top_20_stocks[10:], 1):
                    symbol = stock.get('symbol', 'N/A')
                    score = stock.get('final_score', 0)
                    rating = stock.get('rating', 'N/A')
                    logger.info(f"  {i:2d}. {symbol:<12} Score: {score:6.2f} Rating: {rating}")
            else:
                logger.info("📊 ALL SELECTED STOCKS:")
                for i, stock in enumerate(self.top_20_stocks, 1):
                    symbol = stock.get('symbol', 'N/A')
                    score = stock.get('final_score', 0)
                    rating = stock.get('rating', 'N/A')
                    logger.info(f"  {i:2d}. {symbol:<12} Score: {score:6.2f} Rating: {rating}")
            
            # Mark previous version as deprecated
            with self.universe_lock:
                if self.current_version > 0 and self.current_version in self.stock_universe_versions:
                    self.stock_universe_versions[self.current_version].status = 'deprecated'
                
                # Update current version
                self.current_version = new_version
                self.stock_universe_versions[new_version].status = 'active'
                self.transition_state = 'transitioning'
            
            return True
            
        except Exception as e:
            logger.error(f"Error rating and filtering stocks: {str(e)}")
            # Reset transition state on error
            with self.universe_lock:
                self.transition_state = 'stable'
            return False
    
    def update_live_rating_systems_versioned(self) -> bool:
        """Update live rating systems with versioned graceful transitions"""
        try:
            logger.info("=== UPDATING LIVE RATING SYSTEMS (VERSIONED) ===")
            
            with self.universe_lock:
                current_version = self.current_version
                universe = self.stock_universe_versions.get(current_version)
                
                if not universe:
                    logger.error(f"No universe found for version {current_version}")
                    return False
            
            # Get current stocks from active universe version
            current_stocks = {}
            for stock in universe.stocks:
                symbol = stock.get('symbol', '')
                # Find instrument token
                for instrument in self.instruments:
                    if instrument.get('tradingsymbol') == symbol:
                        instrument_token = str(instrument.get('instrument_token', ''))
                        current_stocks[symbol] = {
                            'instrument_token': instrument_token,
                            'instrument': instrument,
                            'version': current_version
                        }
                        break
            
            # Phase 1: Create new live rating systems for new stocks
            new_systems = {}
            for symbol, stock_data in current_stocks.items():
                if symbol not in self.live_rating_systems:
                    try:
                        logger.info(f"Creating new live rating system for {symbol} (v{current_version})")
                        
                        # Create new live rating system
                        live_system = LiveRatingSystem(
                            instrument_token=stock_data['instrument_token'],
                            kite_trader_instance=self.kite_trader
                        )
                        
                        # Pre-calculate strategic score immediately
                        live_system._calculate_strategic_score()
                        
                        new_systems[symbol] = {
                            'system': live_system,
                            'version': current_version,
                            'status': 'active'
                        }
                        
                        logger.info(f"Initialized live rating system for {symbol}")
                        
                    except Exception as e:
                        logger.error(f"Error creating live rating system for {symbol}: {str(e)}")
            
            # Phase 2: Mark old systems for graceful removal (keep for transition period)
            transition_systems = {}
            for symbol, system_data in self.live_rating_systems.items():
                if symbol not in current_stocks:
                    # Mark for graceful removal
                    if isinstance(system_data, dict):
                        system_data['status'] = 'deprecating'
                        transition_systems[symbol] = system_data
                        logger.info(f"Marking {symbol} for graceful removal")
                    else:
                        # Legacy format - wrap in dict
                        transition_systems[symbol] = {
                            'system': system_data,
                            'version': current_version - 1 if current_version > 0 else 0,
                            'status': 'deprecating'
                        }
                else:
                    # Stock still in current universe - keep active
                    if isinstance(system_data, dict):
                        system_data['status'] = 'active'
                        system_data['version'] = current_version
                        transition_systems[symbol] = system_data
                    else:
                        # Legacy format - wrap in dict and update
                        transition_systems[symbol] = {
                            'system': system_data,
                            'version': current_version,
                            'status': 'active'
                        }
            
            # Phase 3: Atomic update of live rating systems
            with self.top_stocks_lock:
                # Merge new systems with existing/transitioning systems
                self.live_rating_systems.update(new_systems)
                self.live_rating_systems.update(transition_systems)
                
                # Register WebSocket handlers for new systems
                for symbol, system_data in new_systems.items():
                    stock_info = current_stocks[symbol]
                    
                    def create_tick_handler(live_sys):
                        def handle_tick(tick_data):
                            try:
                                live_sys.update_with_tick(tick_data)
                            except Exception as e:
                                logger.error(f"Error updating live system with tick: {str(e)}")
                        return handle_tick
                    
                    self.websocket_manager.register_tick_handler(
                        stock_info['instrument_token'],
                        create_tick_handler(system_data['system'])
                    )
            
            # Phase 4: Update WebSocket subscriptions
            instrument_tokens = [data['instrument_token'] for data in current_stocks.values()]
            if instrument_tokens:
                # Subscribe to new tokens (existing subscriptions remain)
                self.websocket_manager.subscribe_stocks(instrument_tokens)
                self._last_subscribed_tokens = instrument_tokens
                
                logger.info(f"WebSocket subscriptions updated for {len(instrument_tokens)} instruments")
            
            # Phase 5: Set transition state to allow gradual cleanup
            with self.universe_lock:
                self.transition_state = 'stable'
                universe.status = 'active'
            
            logger.info(f"Live rating systems updated for universe version {current_version}")
            logger.info(f"Active systems: {len([s for s in self.live_rating_systems.values() if s.get('status') == 'active'])}")
            logger.info(f"Deprecating systems: {len([s for s in self.live_rating_systems.values() if s.get('status') == 'deprecating'])}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating live rating systems: {str(e)}")
            return False

    def update_live_rating_systems(self) -> bool:
        """Update live rating systems - wrapper for versioned implementation"""
        return self.update_live_rating_systems_versioned()

    def cleanup_deprecated_systems(self):
        """Cleanup deprecated live rating systems after transition period"""
        try:
            to_remove = []
            current_time = datetime.now()
            
            for symbol, system_data in self.live_rating_systems.items():
                if isinstance(system_data, dict) and system_data.get('status') == 'deprecating':
                    # Remove systems that have been deprecating for more than 2 strategic update cycles (20+ minutes)
                    system_version = system_data.get('version', 0)
                    if self.current_version - system_version >= 2:
                        to_remove.append(symbol)
            
            # Remove deprecated systems
            for symbol in to_remove:
                try:
                    # Unregister WebSocket handler
                    for instrument in self.instruments:
                        if instrument.get('tradingsymbol') == symbol:
                            instrument_token = str(instrument.get('instrument_token', ''))
                            self.websocket_manager.unregister_tick_handler(instrument_token)
                            break
                    
                    # Remove from systems dict
                    del self.live_rating_systems[symbol]
                    logger.info(f"Cleaned up deprecated live rating system for {symbol}")
                    
                except Exception as e:
                    logger.error(f"Error cleaning up system for {symbol}: {str(e)}")
            
            # Cleanup old universe versions
            with self.universe_lock:
                versions_to_remove = []
                for version, universe in self.stock_universe_versions.items():
                    if universe.status == 'deprecated' and self.current_version - version >= 3:
                        versions_to_remove.append(version)
                
                for version in versions_to_remove:
                    del self.stock_universe_versions[version]
                    logger.info(f"Cleaned up universe version {version}")
            
            if to_remove or versions_to_remove:
                logger.info(f"Cleanup completed: {len(to_remove)} systems, {len(versions_to_remove)} universe versions")
            
        except Exception as e:
            logger.error(f"Error in cleanup_deprecated_systems: {str(e)}")
    
    def get_live_ratings(self) -> Dict[str, Dict]:
        """Get current live ratings for all active stocks with version-aware access"""
        live_ratings = {}
        
        for symbol, system_data in self.live_rating_systems.items():
            try:
                # Handle both old and new system data formats
                if isinstance(system_data, dict):
                    # New versioned format
                    if system_data.get('status') == 'active':
                        live_system = system_data['system']
                        rating = live_system.get_live_rating()
                        rating['system_version'] = system_data.get('version', 0)
                        live_ratings[symbol] = rating
                    # Skip deprecating systems for live ratings display
                else:
                    # Legacy format - treat as active
                    live_system = system_data
                    rating = live_system.get_live_rating()
                    rating['system_version'] = 'legacy'
                    live_ratings[symbol] = rating
                    
            except Exception as e:
                logger.error(f"Error getting live rating for {symbol}: {str(e)}")
                live_ratings[symbol] = {
                    'symbol': symbol,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
        
        return live_ratings
    
    def periodic_refresh_task(self):
        """Periodic task that runs every 10 minutes"""
        while self.running and not self.shutdown_event.is_set():
            try:
                start_time = time.time()
                logger.info("=== PERIODIC REFRESH STARTED ===")
                
                # 1. Fetch updated historical data
                self.fetch_historical_data()
                
                # 2. Re-rate all stocks and update top 20
                self.rate_and_filter_stocks()
                
                # 3. Update live rating systems
                self.update_live_rating_systems()
                
                # 4. Clear old cache entries
                self.data_cache.clear()
                
                elapsed_time = time.time() - start_time
                logger.info(f"=== PERIODIC REFRESH COMPLETED in {elapsed_time:.2f}s ===")
                
                # Wait for next refresh (accounting for processing time)
                wait_time = max(0, (self.config.refresh_interval_minutes * 60) - elapsed_time)
                if self.shutdown_event.wait(wait_time):
                    break
                    
            except Exception as e:
                logger.error(f"Error in periodic refresh task: {str(e)}")
                if self.shutdown_event.wait(self.config.websocket_retry_delay):
                    break
    
    def save_live_ratings_periodically(self):
        """Save live ratings to file periodically"""
        while self.running and not self.shutdown_event.is_set():
            try:
                # Get current live ratings
                live_ratings = self.get_live_ratings()
                
                if live_ratings:
                    # Prepare data for saving
                    ratings_list = []
                    for symbol, rating in live_ratings.items():
                        rating_data = rating.copy()
                        rating_data['trading_symbol'] = symbol
                        ratings_list.append(rating_data)
                    
                    # Sort by final rating
                    ratings_list.sort(key=lambda x: x.get('final_rating', 0), reverse=True)
                    
                    # Save to file
                    os.makedirs('ratings', exist_ok=True)
                    save_data = {
                        'timestamp': datetime.now().isoformat(),
                        'rating_type': 'Automated Live Rating System (Top 10 + Bottom 10)',
                        'total_stocks': len(ratings_list),
                        'selected_stocks_live_ratings': ratings_list,
                        'selection_strategy': 'top_10_and_bottom_10'
                    }
                    
                    with open('ratings/live_ratings.json', 'w') as f:
                        json.dump(save_data, f, indent=2, default=str)
                
                # Wait 60 seconds before next save
                if self.shutdown_event.wait(60):
                    break
                    
            except Exception as e:
                logger.error(f"Error saving live ratings: {str(e)}")
                if self.shutdown_event.wait(60):
                    break
    
    def start(self) -> bool:
        """Start the automated trading bot"""
        try:
            logger.info("🚀 STARTING AUTOMATED TRADING BOT")
            
            # 1. Validate access token
            if not self.validate_and_get_access_token():
                logger.error("Failed to validate access token")
                return False
            
            # 2. Initialize components
            if not self.initialize_components():
                logger.error("Failed to initialize components")
                return False
            
            # 3. Fetch instruments and watchlist
            if not self.fetch_instruments_and_watchlist():
                logger.error("Failed to fetch instruments and watchlist")
                return False
            
            # 4. Fetch historical data
            if not self.fetch_historical_data():
                logger.error("Failed to fetch historical data")
                return False
            
            # 5. Rate stocks and get top 20
            if not self.rate_and_filter_stocks():
                logger.error("Failed to rate and filter stocks")
                return False
            
            # 6. Connect WebSocket
            if not self.websocket_manager.connect():
                logger.error("Failed to connect WebSocket")
                return False
            
            # 7. Initialize live rating systems
            if not self.update_live_rating_systems():
                logger.error("Failed to initialize live rating systems")
                return False
            
            # 8. Initialize Precision Single Trade Strategy
            if not self.initialize_precision_strategy():
                logger.error("Failed to initialize precision strategy")
                return False
            
            # 9. Start background threads
            self.running = True
            
            # Start periodic refresh thread
            self.scheduler_thread = threading.Thread(
                target=self.periodic_refresh_task,
                daemon=True
            )
            self.scheduler_thread.start()
            
            # Start live ratings saving thread
            save_thread = threading.Thread(
                target=self.save_live_ratings_periodically,
                daemon=True
            )
            save_thread.start()
            
            # Start precision strategy thread if enabled
            if self.enable_precision_strategy and self.precision_strategy:
                precision_thread = threading.Thread(
                    target=self.run_precision_strategy_loop,
                    daemon=True
                )
                precision_thread.start()
                logger.info("🎯 Precision Single Trade Strategy thread started")
            
            logger.info("✅ AUTOMATED TRADING BOT STARTED SUCCESSFULLY")
            logger.info(f"📊 Monitoring {len(self.top_20_stocks)} stocks (top 10 + bottom 10) with live ratings")
            logger.info(f"🔄 Refreshing data every {self.config.refresh_interval_minutes} minutes")
            logger.info("📡 WebSocket streaming active for real-time data")
            
            if self.enable_precision_strategy:
                logger.info("🚀 PRECISION SINGLE TRADE STRATEGY: ACTIVE")
                logger.info("   - Only ONE position at a time")
                logger.info("   - Full margin allocation")
                logger.info("   - 3-phase trade management")
                logger.info("   - Fibonacci trailing stops")
            else:
                logger.info("📈 ANALYSIS MODE ONLY - No live trading")
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting automated trading bot: {str(e)}")
            return False
    
    def run(self):
        """Run the bot and keep it running"""
        try:
            if not self.start():
                logger.error("Failed to start bot")
                return
            
            logger.info("Bot is running. Press Ctrl+C to stop.")
            
            # Keep the main thread alive
            while self.running:
                try:
                    # Display current status every 5 minutes
                    time.sleep(300)  # 5 minutes
                    
                    if self.running:
                        live_ratings = self.get_live_ratings()
                        active_count = len([r for r in live_ratings.values() if 'error' not in r])
                        error_count = len([r for r in live_ratings.values() if 'error' in r])
                        
                        logger.info(f"📊 STATUS: {active_count} active live ratings, {error_count} errors, WebSocket: {'✅' if self.websocket_manager.connected else '❌'}")
                        
                        # Log top 5 live ratings
                        sorted_ratings = sorted(
                            [(symbol, rating) for symbol, rating in live_ratings.items() if 'error' not in rating],
                            key=lambda x: x[1].get('final_rating', 0),
                            reverse=True
                        )
                        
                        logger.info("🏆 TOP 5 LIVE RATINGS:")
                        for i, (symbol, rating) in enumerate(sorted_ratings[:5], 1):
                            score = rating.get('final_rating', 0)
                            rating_text = rating.get('rating_text', 'N/A')
                            logger.info(f"  {i}. {symbol:<12} Score: {score:6.2f} ({rating_text})")
                
                except KeyboardInterrupt:
                    logger.info("Received interrupt signal")
                    break
                except Exception as e:
                    logger.error(f"Error in main loop: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error in bot run loop: {str(e)}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the automated trading bot"""
        logger.info("🛑 STOPPING AUTOMATED TRADING BOT")
        
        self.running = False
        self.shutdown_event.set()
        
        # Disconnect WebSocket
        if self.websocket_manager:
            self.websocket_manager.disconnect()
        
        # Wait for threads to complete
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=10)
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        
        logger.info("✅ AUTOMATED TRADING BOT STOPPED")


def main():
    """Main function"""
    try:
        # Create bot configuration
        config = BotConfig(
            refresh_interval_minutes=10,
            websocket_retry_delay=30,
            top_stocks_count=20,
            enable_caching=True
        )
        
        # Create and run bot
        bot = AutomatedTradingBot(config)
        bot.run()
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")


if __name__ == "__main__":
    main()
