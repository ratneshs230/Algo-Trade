"""
Automated Single Trade Bot - Precision Trading Strategy Implementation

This module implements the sophisticated single-trade-at-a-time strategy with:
1. Market Scanner with holistic rating system
2. Prime candidate selection (highest absolute rating)
3. Dual confirmation (Supertrend + PSAR on 1-minute)
4. Full margin allocation to single high-probability trade
5. Three-phase risk management with Fibonacci trailing stops
"""

import json
import os
import time
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np

# Core imports
from kiteConnect import KiteTrader, Config
from watchlist import get_watchlist
from rating_system import StockRatingSystem
from indicator_calculator import IndicatorCalculator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TradingState(Enum):
    """Trading state machine states"""
    SCANNING = "scanning"
    CONFIRMING = "confirming"
    PHASE_A = "phase_a_initial_risk"
    PHASE_B = "phase_b_breakeven"
    PHASE_C = "phase_c_profit_ladder"
    PAUSED = "paused"
    ERROR = "error"


class TradeDirection(Enum):
    """Trade direction"""
    LONG = "BUY"
    SHORT = "SELL"


@dataclass
class TradingConfig:
    """Configuration for the automated trading bot"""
    # Scanner settings
    scan_interval_seconds: int = 30
    rating_threshold: float = 2.0  # Minimum absolute rating to consider
    
    # Confirmation settings
    confirmation_timeout_seconds: int = 300  # 5 minutes max for confirmation
    supertrend_period: int = 5
    supertrend_factor: float = 3.0
    psar_acceleration: float = 0.02
    psar_maximum: float = 0.2
    
    # Risk management
    max_daily_trades: int = 3
    max_daily_loss_percent: float = 2.0  # 2% of capital
    
    # Position sizing
    use_full_margin: bool = True
    fixed_quantity: Optional[int] = None
    
    # Fibonacci settings
    fibonacci_lookback_candles: int = 50
    fibonacci_levels: List[float] = field(default_factory=lambda: [0.236, 0.382, 0.5, 0.618, 0.786])
    fibonacci_extensions: List[float] = field(default_factory=lambda: [1.272, 1.618, 2.0, 2.618])


@dataclass
class TradingCandidate:
    """Represents a potential trading candidate"""
    symbol: str
    rating: float
    direction: TradeDirection
    instrument_token: str
    price: float
    timestamp: datetime


@dataclass
class ActiveTrade:
    """Represents an active trade with all its parameters"""
    symbol: str
    direction: TradeDirection
    entry_price: float
    quantity: int
    order_id: str
    stop_loss_price: float
    breakeven_price: float
    current_phase: TradingState
    entry_time: datetime
    supertrend_level: float
    fibonacci_levels: Dict[float, float] = field(default_factory=dict)
    current_target_level: Optional[float] = None
    stop_loss_order_id: Optional[str] = None


class SupertrendCalculator:
    """Calculate Supertrend indicator"""
    
    @staticmethod
    def calculate(data: pd.DataFrame, period: int = 5, factor: float = 3.0) -> pd.DataFrame:
        """
        Calculate Supertrend indicator
        
        Args:
            data: DataFrame with OHLC data
            period: ATR period
            factor: Supertrend factor
            
        Returns:
            DataFrame with Supertrend values
        """
        if len(data) < period:
            return data
        
        # Calculate ATR
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(period).mean()
        
        # Calculate basic upper and lower bands
        hl2 = (data['high'] + data['low']) / 2
        basic_upper_band = hl2 + (factor * atr)
        basic_lower_band = hl2 - (factor * atr)
        
        # Calculate final upper and lower bands
        final_upper_band = basic_upper_band.copy()
        final_lower_band = basic_lower_band.copy()
        
        for i in range(1, len(data)):
            # Final upper band
            if basic_upper_band.iloc[i] < final_upper_band.iloc[i-1] or data['close'].iloc[i-1] > final_upper_band.iloc[i-1]:
                final_upper_band.iloc[i] = basic_upper_band.iloc[i]
            else:
                final_upper_band.iloc[i] = final_upper_band.iloc[i-1]
                
            # Final lower band
            if basic_lower_band.iloc[i] > final_lower_band.iloc[i-1] or data['close'].iloc[i-1] < final_lower_band.iloc[i-1]:
                final_lower_band.iloc[i] = basic_lower_band.iloc[i]
            else:
                final_lower_band.iloc[i] = final_lower_band.iloc[i-1]
        
        # Calculate Supertrend
        supertrend = pd.Series(index=data.index, dtype=float)
        direction = pd.Series(index=data.index, dtype=int)
        
        for i in range(len(data)):
            if i == 0:
                supertrend.iloc[i] = final_lower_band.iloc[i]
                direction.iloc[i] = 1
            else:
                if direction.iloc[i-1] == 1:
                    if data['close'].iloc[i] <= final_lower_band.iloc[i]:
                        supertrend.iloc[i] = final_upper_band.iloc[i]
                        direction.iloc[i] = -1
                    else:
                        supertrend.iloc[i] = final_lower_band.iloc[i]
                        direction.iloc[i] = 1
                else:
                    if data['close'].iloc[i] >= final_upper_band.iloc[i]:
                        supertrend.iloc[i] = final_lower_band.iloc[i]
                        direction.iloc[i] = 1
                    else:
                        supertrend.iloc[i] = final_upper_band.iloc[i]
                        direction.iloc[i] = -1
        
        data = data.copy()
        data['supertrend'] = supertrend
        data['supertrend_direction'] = direction
        return data


class PSARCalculator:
    """Calculate Parabolic SAR indicator"""
    
    @staticmethod
    def calculate(data: pd.DataFrame, acceleration: float = 0.02, maximum: float = 0.2) -> pd.DataFrame:
        """
        Calculate Parabolic SAR indicator
        
        Args:
            data: DataFrame with OHLC data
            acceleration: Acceleration factor
            maximum: Maximum acceleration
            
        Returns:
            DataFrame with PSAR values
        """
        if len(data) < 2:
            return data
        
        psar = pd.Series(index=data.index, dtype=float)
        bull = pd.Series(index=data.index, dtype=bool)
        af = pd.Series(index=data.index, dtype=float)
        ep = pd.Series(index=data.index, dtype=float)
        
        # Initialize
        psar.iloc[0] = data['low'].iloc[0]
        bull.iloc[0] = True
        af.iloc[0] = acceleration
        ep.iloc[0] = data['high'].iloc[0]
        
        for i in range(1, len(data)):
            if bull.iloc[i-1]:  # Bullish trend
                psar.iloc[i] = psar.iloc[i-1] + af.iloc[i-1] * (ep.iloc[i-1] - psar.iloc[i-1])
                
                # Check if trend reverses
                if data['low'].iloc[i] < psar.iloc[i]:
                    bull.iloc[i] = False
                    psar.iloc[i] = ep.iloc[i-1]
                    af.iloc[i] = acceleration
                    ep.iloc[i] = data['low'].iloc[i]
                else:
                    bull.iloc[i] = True
                    if data['high'].iloc[i] > ep.iloc[i-1]:
                        ep.iloc[i] = data['high'].iloc[i]
                        af.iloc[i] = min(af.iloc[i-1] + acceleration, maximum)
                    else:
                        ep.iloc[i] = ep.iloc[i-1]
                        af.iloc[i] = af.iloc[i-1]
                        
                    # Ensure PSAR doesn't exceed previous two lows
                    psar.iloc[i] = min(psar.iloc[i], data['low'].iloc[i-1], data['low'].iloc[i-2] if i > 1 else data['low'].iloc[i-1])
            
            else:  # Bearish trend
                psar.iloc[i] = psar.iloc[i-1] + af.iloc[i-1] * (ep.iloc[i-1] - psar.iloc[i-1])
                
                # Check if trend reverses
                if data['high'].iloc[i] > psar.iloc[i]:
                    bull.iloc[i] = True
                    psar.iloc[i] = ep.iloc[i-1]
                    af.iloc[i] = acceleration
                    ep.iloc[i] = data['high'].iloc[i]
                else:
                    bull.iloc[i] = False
                    if data['low'].iloc[i] < ep.iloc[i-1]:
                        ep.iloc[i] = data['low'].iloc[i]
                        af.iloc[i] = min(af.iloc[i-1] + acceleration, maximum)
                    else:
                        ep.iloc[i] = ep.iloc[i-1]
                        af.iloc[i] = af.iloc[i-1]
                        
                    # Ensure PSAR doesn't exceed previous two highs
                    psar.iloc[i] = max(psar.iloc[i], data['high'].iloc[i-1], data['high'].iloc[i-2] if i > 1 else data['high'].iloc[i-1])
        
        data = data.copy()
        data['psar'] = psar
        data['psar_bull'] = bull
        return data


class FibonacciCalculator:
    """Calculate Fibonacci retracement and extension levels"""
    
    @staticmethod
    def calculate_levels(data: pd.DataFrame, lookback: int = 50, 
                        retracement_levels: List[float] = None,
                        extension_levels: List[float] = None) -> Dict[str, Dict[float, float]]:
        """
        Calculate Fibonacci levels based on swing high/low
        
        Args:
            data: DataFrame with OHLC data
            lookback: Number of candles to look back for swing points
            retracement_levels: Fibonacci retracement levels
            extension_levels: Fibonacci extension levels
            
        Returns:
            Dictionary with retracement and extension levels
        """
        if retracement_levels is None:
            retracement_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        if extension_levels is None:
            extension_levels = [1.272, 1.618, 2.0, 2.618]
        
        if len(data) < lookback:
            return {"retracements": {}, "extensions": {}}
        
        # Get recent data
        recent_data = data.tail(lookback)
        
        # Find swing high and low
        swing_high = recent_data['high'].max()
        swing_low = recent_data['low'].min()
        
        # Calculate range
        swing_range = swing_high - swing_low
        
        # Calculate retracement levels (from high to low)
        retracements = {}
        for level in retracement_levels:
            retracements[level] = swing_high - (swing_range * level)
        
        # Calculate extension levels (beyond the range)
        extensions = {}
        for level in extension_levels:
            extensions[level] = swing_low - (swing_range * (level - 1))  # Below swing low for shorts
            # For longs, it would be: swing_high + (swing_range * (level - 1))
        
        return {
            "retracements": retracements,
            "extensions": extensions,
            "swing_high": swing_high,
            "swing_low": swing_low,
            "range": swing_range
        }


class AutomatedSingleTradeBot:
    """Main automated trading bot implementing the precision single-trade strategy"""
    
    def __init__(self, access_token: str, config: TradingConfig = None):
        self.access_token = access_token
        self.config = config or TradingConfig()
        
        # Initialize components
        self.kite_trader = KiteTrader(api_key=Config.API_KEY, access_token=access_token)
        self.stock_rating_system = StockRatingSystem()
        
        # State management
        self.current_state = TradingState.SCANNING
        self.active_trade: Optional[ActiveTrade] = None
        self.current_candidate: Optional[TradingCandidate] = None
        
        # Instruments and data
        self.instruments = []
        self.watchlist_stocks = []
        self.stock_data_cache = {}
        
        # Tracking
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.start_time = datetime.now()
        
        # Threading
        self.running = False
        self.main_thread = None
        self.state_lock = threading.Lock()
        
        logger.info("Automated Single Trade Bot initialized")
    
    def initialize(self) -> bool:
        """Initialize the bot with required data"""
        try:
            logger.info("Initializing bot...")
            
            # Load watchlist
            self.watchlist_stocks = get_watchlist()
            logger.info(f"Loaded {len(self.watchlist_stocks)} watchlist stocks")
            
            # Get instruments
            all_instruments = self.kite_trader.get_instruments(exchange='NSE')
            self.instruments = [
                inst for inst in all_instruments
                if inst.get('tradingsymbol') in self.watchlist_stocks
                and inst.get('instrument_type') == 'EQ'
            ]
            logger.info(f"Found {len(self.instruments)} tradeable instruments")
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing bot: {str(e)}")
            return False
    
    def start(self) -> bool:
        """Start the automated trading bot"""
        try:
            if not self.initialize():
                return False
            
            self.running = True
            self.main_thread = threading.Thread(target=self._main_loop, daemon=True)
            self.main_thread.start()
            
            logger.info("✅ Automated Single Trade Bot started successfully")
            logger.info(f"📊 Configuration: Max daily trades: {self.config.max_daily_trades}, "
                       f"Max daily loss: {self.config.max_daily_loss_percent}%")
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting bot: {str(e)}")
            return False
    
    def stop(self):
        """Stop the automated trading bot"""
        logger.info("🛑 Stopping Automated Single Trade Bot...")
        self.running = False
        
        if self.active_trade:
            logger.warning("Bot stopped with active trade - manual intervention may be required")
        
        if self.main_thread and self.main_thread.is_alive():
            self.main_thread.join(timeout=10)
        
        logger.info("✅ Bot stopped successfully")
    
    def _main_loop(self):
        """Main trading loop"""
        logger.info("🚀 Main trading loop started")
        
        while self.running:
            try:
                with self.state_lock:
                    current_state = self.current_state
                
                # State machine execution
                if current_state == TradingState.SCANNING:
                    self._handle_scanning_state()
                elif current_state == TradingState.CONFIRMING:
                    self._handle_confirming_state()
                elif current_state == TradingState.PHASE_A:
                    self._handle_phase_a_state()
                elif current_state == TradingState.PHASE_B:
                    self._handle_phase_b_state()
                elif current_state == TradingState.PHASE_C:
                    self._handle_phase_c_state()
                elif current_state == TradingState.PAUSED:
                    self._handle_paused_state()
                elif current_state == TradingState.ERROR:
                    self._handle_error_state()
                
                # Sleep between iterations
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                with self.state_lock:
                    self.current_state = TradingState.ERROR
                time.sleep(5)
        
        logger.info("🏁 Main trading loop ended")
    
    def _handle_scanning_state(self):
        """Handle the scanning state - find the best trading opportunity"""
        try:
            # Check daily limits
            if self._check_daily_limits():
                logger.info("Daily limits reached, pausing trading")
                with self.state_lock:
                    self.current_state = TradingState.PAUSED
                return
            
            logger.info("🔍 Scanning for trading opportunities...")
            
            # Get ratings for all watchlist stocks
            symbols = [inst.get('tradingsymbol') for inst in self.instruments]
            ratings = self.stock_rating_system.rate_multiple_stocks(symbols, self.instruments)
            
            # Filter valid ratings and find highest absolute rating
            valid_ratings = [r for r in ratings if 'error' not in r and abs(r.get('final_score', 0)) >= self.config.rating_threshold]
            
            if not valid_ratings:
                logger.info("No valid trading opportunities found")
                time.sleep(self.config.scan_interval_seconds)
                return
            
            # Sort by absolute rating (highest magnitude first)
            valid_ratings.sort(key=lambda x: abs(x.get('final_score', 0)), reverse=True)
            
            best_candidate = valid_ratings[0]
            rating = best_candidate.get('final_score', 0)
            symbol = best_candidate.get('symbol', '')
            
            # Determine trade direction
            direction = TradeDirection.LONG if rating > 0 else TradeDirection.SHORT
            
            # Find instrument token
            instrument_token = None
            for inst in self.instruments:
                if inst.get('tradingsymbol') == symbol:
                    instrument_token = str(inst.get('instrument_token'))
                    break
            
            if not instrument_token:
                logger.error(f"Could not find instrument token for {symbol}")
                return
            
            # Get current price
            try:
                quote = self.kite_trader.get_ltp(f"NSE:{symbol}")
                current_price = quote[f"NSE:{symbol}"]['last_price']
            except Exception as e:
                logger.error(f"Error getting price for {symbol}: {str(e)}")
                return
            
            # Create candidate
            self.current_candidate = TradingCandidate(
                symbol=symbol,
                rating=rating,
                direction=direction,
                instrument_token=instrument_token,
                price=current_price,
                timestamp=datetime.now()
            )
            
            logger.info(f"🎯 Selected candidate: {symbol} (Rating: {rating:.2f}, "
                       f"Direction: {direction.value}, Price: ₹{current_price})")
            
            # Move to confirmation state
            with self.state_lock:
                self.current_state = TradingState.CONFIRMING
            
        except Exception as e:
            logger.error(f"Error in scanning state: {str(e)}")
            time.sleep(self.config.scan_interval_seconds)
    
    def _handle_confirming_state(self):
        """Handle the confirming state - validate entry with Supertrend + PSAR"""
        try:
            if not self.current_candidate:
                logger.error("No candidate to confirm")
                with self.state_lock:
                    self.current_state = TradingState.SCANNING
                return
            
            symbol = self.current_candidate.symbol
            direction = self.current_candidate.direction
            
            logger.info(f"🔍 Confirming entry for {symbol} ({direction.value})")
            
            # Get 1-minute data for confirmation
            try:
                to_date = datetime.now()
                from_date = to_date - timedelta(hours=2)  # Get 2 hours of 1-minute data
                
                historical_data = self.kite_trader.get_historical_data(
                    instrument_token=int(self.current_candidate.instrument_token),
                    from_date=from_date,
                    to_date=to_date,
                    interval='minute'
                )
                
                if len(historical_data) < 50:
                    logger.warning(f"Insufficient data for {symbol}, skipping")
                    self._reset_to_scanning()
                    return
                
                # Convert to DataFrame
                df = pd.DataFrame(historical_data)
                df['timestamp'] = pd.to_datetime(df['date'])
                
                # Calculate indicators
                df = SupertrendCalculator.calculate(df, self.config.supertrend_period, self.config.supertrend_factor)
                df = PSARCalculator.calculate(df, self.config.psar_acceleration, self.config.psar_maximum)
                
                # Get latest values
                latest = df.iloc[-1]
                current_price = latest['close']
                supertrend = latest['supertrend']
                supertrend_direction = latest['supertrend_direction']
                psar = latest['psar']
                psar_bull = latest['psar_bull']
                
                logger.info(f"📊 {symbol} - Price: ₹{current_price:.2f}, Supertrend: ₹{supertrend:.2f}, "
                           f"PSAR: ₹{psar:.2f}, ST_Dir: {supertrend_direction}, PSAR_Bull: {psar_bull}")
                
                # Check dual confirmation
                confirmation_met = False
                
                if direction == TradeDirection.LONG:
                    # For long: Price above Supertrend AND PSAR below price (bullish)
                    if current_price > supertrend and psar_bull:
                        confirmation_met = True
                        logger.info("✅ LONG confirmation met: Price > Supertrend AND PSAR bullish")
                    else:
                        logger.info("❌ LONG confirmation failed")
                        
                elif direction == TradeDirection.SHORT:
                    # For short: Price below Supertrend AND PSAR above price (bearish)
                    if current_price < supertrend and not psar_bull:
                        confirmation_met = True
                        logger.info("✅ SHORT confirmation met: Price < Supertrend AND PSAR bearish")
                    else:
                        logger.info("❌ SHORT confirmation failed")
                
                if confirmation_met:
                    # Execute the trade
                    if self._execute_trade(current_price, supertrend):
                        logger.info(f"🚀 Trade executed successfully for {symbol}")
                    else:
                        logger.error(f"Failed to execute trade for {symbol}")
                        self._reset_to_scanning()
                else:
                    logger.info(f"Confirmation failed for {symbol}, returning to scanning")
                    self._reset_to_scanning()
                
            except Exception as e:
                logger.error(f"Error getting data for confirmation: {str(e)}")
                self._reset_to_scanning()
                
        except Exception as e:
            logger.error(f"Error in confirming state: {str(e)}")
            self._reset_to_scanning()
    
    def _execute_trade(self, current_price: float, supertrend_level: float) -> bool:
        """Execute the trade with full margin allocation"""
        try:
            if not self.current_candidate:
                return False
            
            symbol = self.current_candidate.symbol
            direction = self.current_candidate.direction
            
            logger.info(f"💰 Executing trade: {symbol} {direction.value} at ₹{current_price}")
            
            # Calculate position size
            if self.config.use_full_margin:
                # Get available margin
                margins = self.kite_trader.get_margins()
                available_margin = margins['equity']['net']
                
                # Calculate quantity based on available margin
                quantity = int(available_margin / current_price)
                
                # Minimum quantity check
                if quantity < 1:
                    logger.error("Insufficient margin for even 1 quantity")
                    return False
            else:
                quantity = self.config.fixed_quantity or 1
            
            # Calculate transaction costs and breakeven
            brokerage = min(20, quantity * current_price * 0.01 / 100)  # Max ₹20 or 0.01%
            stt = quantity * current_price * 0.025 / 100  # 0.025% on delivery
            exchange_charges = quantity * current_price * 0.00345 / 100  # ~0.00345%
            gst = (brokerage + exchange_charges) * 0.18  # 18% GST
            sebi_charges = quantity * current_price * 0.0001 / 100  # 0.0001%
            
            total_charges = brokerage + stt + exchange_charges + gst + sebi_charges
            
            # Calculate breakeven price
            if direction == TradeDirection.LONG:
                breakeven_price = current_price + (total_charges * 2 / quantity)  # Double charges for buy+sell
            else:
                breakeven_price = current_price - (total_charges * 2 / quantity)
            
            # Get tick size for precision
            tick_size = 0.05  # Default NSE tick size
            for inst in self.instruments:
                if inst.get('tradingsymbol') == symbol:
                    tick_size = inst.get('tick_size', 0.05)
                    break
            
            # Adjust breakeven to next tick
            if direction == TradeDirection.LONG:
                breakeven_price = breakeven_price + tick_size
            else:
                breakeven_price = breakeven_price - tick_size
            
            # Round to tick size
            breakeven_price = round(breakeven_price / tick_size) * tick_size
            
            logger.info(f"📊 Position details: Qty={quantity}, Entry=₹{current_price:.2f}, "
                       f"Breakeven=₹{breakeven_price:.2f}, Charges=₹{total_charges:.2f}")
            
            # Place the order
            try:
                order_response = self.kite_trader.place_order(
                    variety=self.kite_trader.VARIETY_REGULAR,
                    exchange=self.kite_trader.EXCHANGE_NSE,
                    tradingsymbol=symbol,
                    transaction_type=direction.value,
                    quantity=quantity,
                    product=self.kite_trader.PRODUCT_MIS,  # Intraday
                    order_type=self.kite_trader.ORDER_TYPE_MARKET
                )
                
                order_id = order_response['order_id']
                logger.info(f"✅ Order placed successfully: {order_id}")
                
                # Wait for order execution
                time.sleep(2)
                
                # Check order status
                order_history = self.kite_trader.get_order_history(order_id)
                latest_order = order_history[-1]
                
                if latest_order['status'] == 'COMPLETE':
                    executed_price = latest_order['average_price'] or current_price
                    executed_quantity = latest_order['filled_quantity']
                    
                    # Recalculate breakeven with actual execution price
                    if direction == TradeDirection.LONG:
                        actual_breakeven = executed_price + (total_charges * 2 / executed_quantity) + tick_size
                    else:
                        actual_breakeven = executed_price - (total_charges * 2 / executed_quantity) - tick_size
                    
                    actual_breakeven = round(actual_breakeven / tick_size) * tick_size
                    
                    # Create active trade
                    self.active_trade = ActiveTrade(
                        symbol=symbol,
                        direction=direction,
                        entry_price=executed_price,
                        quantity=executed_quantity,
                        order_id=order_id,
                        stop_loss_price=supertrend_level,
                        breakeven_price=actual_breakeven,
                        current_phase=TradingState.PHASE_A,
                        entry_time=datetime.now(),
                        supertrend_level=supertrend_level
                    )
                    
                    # Place initial stop-loss order
                    self._place_stop_loss_order(supertrend_level)
                    
                    # Update daily trades counter
                    self.daily_trades += 1
                    
                    # Move to Phase A
                    with self.state_lock:
                        self.current_state = TradingState.PHASE_A
                    
                    logger.info(f"🎯 Trade active: {symbol} {direction.value} {executed_quantity}@₹{executed_price:.2f}")
                    logger.info(f"📊 Stop Loss: ₹{supertrend_level:.2f}, Breakeven: ₹{actual_breakeven:.2f}")
                    
                    return True
                else:
                    logger.error(f"Order not executed: {latest_order['status']}")
                    return False
                
            except Exception as e:
                logger.error(f"Error placing order: {str(e)}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
            return False
    
    def _place_stop_loss_order(self, stop_price: float) -> bool:
        """Place or modify stop-loss order"""
        try:
            if not self.active_trade:
                return False
            
            # Cancel existing stop-loss order if any
            if self.active_trade.stop_loss_order_id:
                try:
                    self.kite_trader.cancel_order(
                        variety=self.kite_trader.VARIETY_REGULAR,
                        order_id=self.active_trade.stop_loss_order_id
                    )
                    logger.info(f"Cancelled existing stop-loss order: {self.active_trade.stop_loss_order_id}")
                except Exception as e:
                    logger.warning(f"Could not cancel existing stop-loss: {str(e)}")
            
            # Determine stop-loss transaction type
            stop_transaction_type = TradeDirection.SHORT.value if self.active_trade.direction == TradeDirection.LONG else TradeDirection.LONG.value
            
            # Place new stop-loss order
            stop_order_response = self.kite_trader.place_order(
                variety=self.kite_trader.VARIETY_REGULAR,
                exchange=self.kite_trader.EXCHANGE_NSE,
                tradingsymbol=self.active_trade.symbol,
                transaction_type=stop_transaction_type,
                quantity=self.active_trade.quantity,
                product=self.kite_trader.PRODUCT_MIS,
                order_type=self.kite_trader.ORDER_TYPE_SLM,
                trigger_price=stop_price
            )
            
            self.active_trade.stop_loss_order_id = stop_order_response['order_id']
            self.active_trade.stop_loss_price = stop_price
            
            logger.info(f"✅ Stop-loss placed at ₹{stop_price:.2f} (Order: {self.active_trade.stop_loss_order_id})")
            return True
            
        except Exception as e:
            logger.error(f"Error placing stop-loss order: {str(e)}")
            return False
    
    def _handle_phase_a_state(self):
        """Handle Phase A - Initial risk phase, waiting for breakeven"""
        try:
            if not self.active_trade:
                logger.error("No active trade in Phase A")
                with self.state_lock:
                    self.current_state = TradingState.SCANNING
                return
            
            symbol = self.active_trade.symbol
            
            # Check if trade is still active
            if not self._is_trade_active():
                logger.info(f"Trade {symbol} has been closed, returning to scanning")
                self._handle_trade_closure()
                return
            
            # Get current price
            try:
                quote = self.kite_trader.get_ltp(f"NSE:{symbol}")
                current_price = quote[f"NSE:{symbol}"]['last_price']
            except Exception as e:
                logger.error(f"Error getting current price: {str(e)}")
                time.sleep(5)
                return
            
            # Check if breakeven target is hit
            breakeven_hit = False
            
            if self.active_trade.direction == TradeDirection.LONG:
                if current_price >= self.active_trade.breakeven_price:
                    breakeven_hit = True
            else:  # SHORT
                if current_price <= self.active_trade.breakeven_price:
                    breakeven_hit = True
            
            if breakeven_hit:
                logger.info(f"🎯 Breakeven hit for {symbol} at ₹{current_price:.2f}")
                
                # Move stop-loss to breakeven
                self._place_stop_loss_order(self.active_trade.entry_price)
                
                # Calculate Fibonacci levels for Phase C
                self._calculate_fibonacci_levels()
                
                # Move to Phase B
                self.active_trade.current_phase = TradingState.PHASE_B
                with self.state_lock:
                    self.current_state = TradingState.PHASE_B
                
                logger.info(f"✅ Moved to Phase B - Risk-free zone for {symbol}")
            
            else:
                # Log current status periodically
                if datetime.now().minute % 5 == 0:  # Every 5 minutes
                    unrealized_pnl = self._calculate_unrealized_pnl(current_price)
                    logger.info(f"📊 Phase A - {symbol}: Price=₹{current_price:.2f}, "
                               f"Target=₹{self.active_trade.breakeven_price:.2f}, "
                               f"P&L=₹{unrealized_pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Error in Phase A: {str(e)}")
            time.sleep(5)
    
    def _handle_phase_b_state(self):
        """Handle Phase B - Risk-free phase, setting up Fibonacci targets"""
        try:
            if not self.active_trade:
                logger.error("No active trade in Phase B")
                with self.state_lock:
                    self.current_state = TradingState.SCANNING
                return
            
            # Automatically move to Phase C after setting up Fibonacci levels
            logger.info(f"Moving {self.active_trade.symbol} to Phase C - Profit ladder")
            
            # Set first Fibonacci target
            if self.active_trade.fibonacci_levels:
                first_level = min(self.active_trade.fibonacci_levels.keys())
                self.active_trade.current_target_level = first_level
                
                logger.info(f"🎯 First Fibonacci target set at {first_level} level "
                           f"(₹{self.active_trade.fibonacci_levels[first_level]:.2f})")
            
            # Move to Phase C
            self.active_trade.current_phase = TradingState.PHASE_C
            with self.state_lock:
                self.current_state = TradingState.PHASE_C
            
        except Exception as e:
            logger.error(f"Error in Phase B: {str(e)}")
            time.sleep(5)
    
    def _handle_phase_c_state(self):
        """Handle Phase C - Fibonacci ladder trailing stops"""
        try:
            if not self.active_trade:
                logger.error("No active trade in Phase C")
                with self.state_lock:
                    self.current_state = TradingState.SCANNING
                return
            
            symbol = self.active_trade.symbol
            
            # Check if trade is still active
            if not self._is_trade_active():
                logger.info(f"Trade {symbol} has been closed, returning to scanning")
                self._handle_trade_closure()
                return
            
            # Get current price
            try:
                quote = self.kite_trader.get_ltp(f"NSE:{symbol}")
                current_price = quote[f"NSE:{symbol}"]['last_price']
            except Exception as e:
                logger.error(f"Error getting current price: {str(e)}")
                time.sleep(5)
                return
            
            # Check if current Fibonacci target is hit
            if (self.active_trade.current_target_level and 
                self.active_trade.current_target_level in self.active_trade.fibonacci_levels):
                
                target_price = self.active_trade.fibonacci_levels[self.active_trade.current_target_level]
                target_hit = False
                
                if self.active_trade.direction == TradeDirection.LONG:
                    if current_price >= target_price:
                        target_hit = True
                else:  # SHORT
                    if current_price <= target_price:
                        target_hit = True
                
                if target_hit:
                    logger.info(f"🎯 Fibonacci level {self.active_trade.current_target_level} hit at ₹{current_price:.2f}")
                    
                    # Move stop-loss to this level
                    self._place_stop_loss_order(target_price)
                    
                    # Find next Fibonacci level
                    current_levels = sorted(self.active_trade.fibonacci_levels.keys())
                    current_index = current_levels.index(self.active_trade.current_target_level)
                    
                    if current_index < len(current_levels) - 1:
                        next_level = current_levels[current_index + 1]
                        self.active_trade.current_target_level = next_level
                        logger.info(f"🎯 Next target: {next_level} level "
                                   f"(₹{self.active_trade.fibonacci_levels[next_level]:.2f})")
                    else:
                        logger.info("🏆 All Fibonacci levels achieved, keeping trailing stop")
                        self.active_trade.current_target_level = None
            
            # Log current status periodically
            if datetime.now().minute % 2 == 0:  # Every 2 minutes
                unrealized_pnl = self._calculate_unrealized_pnl(current_price)
                target_info = ""
                if self.active_trade.current_target_level:
                    target_price = self.active_trade.fibonacci_levels[self.active_trade.current_target_level]
                    target_info = f", Target=₹{target_price:.2f}"
                
                logger.info(f"📊 Phase C - {symbol}: Price=₹{current_price:.2f}"
                           f"{target_info}, P&L=₹{unrealized_pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Error in Phase C: {str(e)}")
            time.sleep(5)
    
    def _handle_paused_state(self):
        """Handle paused state - wait for market conditions or daily reset"""
        current_time = datetime.now()
        
        # Reset daily counters at start of new trading day (9:15 AM)
        if current_time.hour == 9 and current_time.minute >= 15:
            if self.start_time.date() < current_time.date():
                logger.info("🆕 New trading day - resetting daily counters")
                self.daily_trades = 0
                self.daily_pnl = 0.0
                self.start_time = current_time
                
                with self.state_lock:
                    self.current_state = TradingState.SCANNING
                return
        
        # Check if we should resume (e.g., if daily limits are no longer breached)
        if not self._check_daily_limits():
            logger.info("Daily limits no longer breached, resuming trading")
            with self.state_lock:
                self.current_state = TradingState.SCANNING
        
        time.sleep(60)  # Wait 1 minute before checking again
    
    def _handle_error_state(self):
        """Handle error state - try to recover"""
        logger.warning("⚠️ Bot in error state, attempting recovery...")
        
        # Basic recovery - return to scanning
        time.sleep(10)
        with self.state_lock:
            self.current_state = TradingState.SCANNING
        
        logger.info("Recovery attempted, resuming operations")
    
    def _is_trade_active(self) -> bool:
        """Check if the current trade is still active"""
        try:
            if not self.active_trade:
                return False
            
            # Get current positions
            positions = self.kite_trader.get_positions()['net']
            
            for position in positions:
                if (position['tradingsymbol'] == self.active_trade.symbol and
                    position['quantity'] != 0):
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking trade status: {str(e)}")
            return True  # Assume active on error to be safe
    
    def _calculate_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L for the active trade"""
        if not self.active_trade:
            return 0.0
        
        entry_price = self.active_trade.entry_price
        quantity = self.active_trade.quantity
        
        if self.active_trade.direction == TradeDirection.LONG:
            pnl = (current_price - entry_price) * quantity
        else:  # SHORT
            pnl = (entry_price - current_price) * quantity
        
        return pnl
    
    def _calculate_fibonacci_levels(self):
        """Calculate Fibonacci levels for the active trade"""
        try:
            if not self.active_trade:
                return
            
            symbol = self.active_trade.symbol
            instrument_token = None
            
            # Find instrument token
            for inst in self.instruments:
                if inst.get('tradingsymbol') == symbol:
                    instrument_token = str(inst.get('instrument_token'))
                    break
            
            if not instrument_token:
                logger.error(f"Could not find instrument token for {symbol}")
                return
            
            # Get 1-minute data for Fibonacci calculation
            to_date = datetime.now()
            from_date = to_date - timedelta(hours=2)
            
            historical_data = self.kite_trader.get_historical_data(
                instrument_token=int(instrument_token),
                from_date=from_date,
                to_date=to_date,
                interval='minute'
            )
            
            df = pd.DataFrame(historical_data)
            
            # Calculate Fibonacci levels
            fib_data = FibonacciCalculator.calculate_levels(
                df, 
                lookback=self.config.fibonacci_lookback_candles,
                retracement_levels=self.config.fibonacci_levels,
                extension_levels=self.config.fibonacci_extensions
            )
            
            # Store appropriate levels based on trade direction
            if self.active_trade.direction == TradeDirection.LONG:
                # For longs, use retracement levels above current price
                levels = {}
                for level, price in fib_data['retracements'].items():
                    if price > self.active_trade.entry_price:
                        levels[level] = price
            else:  # SHORT
                # For shorts, use extension levels below current price  
                levels = {}
                for level, price in fib_data['extensions'].items():
                    if price < self.active_trade.entry_price:
                        levels[level] = price
            
            self.active_trade.fibonacci_levels = levels
            
            logger.info(f"📊 Calculated Fibonacci levels for {symbol}:")
            for level, price in sorted(levels.items()):
                logger.info(f"   {level}: ₹{price:.2f}")
            
        except Exception as e:
            logger.error(f"Error calculating Fibonacci levels: {str(e)}")
    
    def _handle_trade_closure(self):
        """Handle trade closure and cleanup"""
        try:
            if not self.active_trade:
                return
            
            symbol = self.active_trade.symbol
            
            # Cancel any pending stop-loss orders
            if self.active_trade.stop_loss_order_id:
                try:
                    self.kite_trader.cancel_order(
                        variety=self.kite_trader.VARIETY_REGULAR,
                        order_id=self.active_trade.stop_loss_order_id
                    )
                except:
                    pass  # Order might already be executed or cancelled
            
            # Calculate final P&L
            try:
                positions = self.kite_trader.get_positions()['net']
                realized_pnl = 0.0
                
                for position in positions:
                    if position['tradingsymbol'] == symbol:
                        realized_pnl = position['pnl']
                        break
                
                self.daily_pnl += realized_pnl
                
                logger.info(f"✅ Trade closed: {symbol} | P&L: ₹{realized_pnl:.2f} | Daily P&L: ₹{self.daily_pnl:.2f}")
                
            except Exception as e:
                logger.error(f"Error calculating final P&L: {str(e)}")
            
            # Clean up
            self.active_trade = None
            self.current_candidate = None
            
            # Return to scanning
            with self.state_lock:
                self.current_state = TradingState.SCANNING
            
        except Exception as e:
            logger.error(f"Error handling trade closure: {str(e)}")
    
    def _check_daily_limits(self) -> bool:
        """Check if daily trading limits have been reached"""
        # Check max daily trades
        if self.daily_trades >= self.config.max_daily_trades:
            return True
        
        # Check max daily loss
        if self.daily_pnl < 0:
            loss_percent = abs(self.daily_pnl) / 100000 * 100  # Assuming 1L capital
            if loss_percent >= self.config.max_daily_loss_percent:
                return True
        
        return False
    
    def _reset_to_scanning(self):
        """Reset state to scanning"""
        self.current_candidate = None
        with self.state_lock:
            self.current_state = TradingState.SCANNING
        time.sleep(self.config.scan_interval_seconds)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current bot status"""
        with self.state_lock:
            return {
                "current_state": self.current_state.value,
                "running": self.running,
                "daily_trades": self.daily_trades,
                "daily_pnl": self.daily_pnl,
                "max_daily_trades": self.config.max_daily_trades,
                "active_trade": {
                    "symbol": self.active_trade.symbol,
                    "direction": self.active_trade.direction.value,
                    "entry_price": self.active_trade.entry_price,
                    "quantity": self.active_trade.quantity,
                    "current_phase": self.active_trade.current_phase.value,
                    "entry_time": self.active_trade.entry_time.isoformat()
                } if self.active_trade else None,
                "current_candidate": {
                    "symbol": self.current_candidate.symbol,
                    "rating": self.current_candidate.rating,
                    "direction": self.current_candidate.direction.value
                } if self.current_candidate else None
            }


def main():
    """Main function to run the automated single trade bot"""
    try:
        # Load access token
        with open('access_token.json', 'r') as f:
            token_data = json.load(f)
        access_token = token_data['access_token']
        
        # Create configuration
        config = TradingConfig(
            scan_interval_seconds=30,
            max_daily_trades=3,
            max_daily_loss_percent=2.0,
            use_full_margin=True
        )
        
        # Create and start bot
        bot = AutomatedSingleTradeBot(access_token, config)
        
        if bot.start():
            logger.info("Bot started successfully. Press Ctrl+C to stop.")
            
            try:
                while bot.running:
                    time.sleep(60)  # Print status every minute
                    status = bot.get_status()
                    logger.info(f"Status: {status['current_state']} | "
                               f"Daily Trades: {status['daily_trades']} | "
                               f"Daily P&L: ₹{status['daily_pnl']:.2f}")
            except KeyboardInterrupt:
                logger.info("Received stop signal")
                bot.stop()
        else:
            logger.error("Failed to start bot")
    
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")


if __name__ == "__main__":
    main()
