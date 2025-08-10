"""
Live Rating System for Real-Time Stock Analysis

This module implements a high-frequency, real-time stock rating system using a 
Hybrid Calculation Model that separates slow-moving strategic indicators from 
fast-moving tactical indicators for maximum performance.

Architecture:
- Tier 1: Strategic Base Score (60min + 30min + 15min) - Updated every 10 minutes
- Tier 2: Tactical Live Score (1min + 3min + 5min) - Updated every few seconds
"""

import pandas as pd
import numpy as np
import json
import pandas_ta as ta # Import pandas_ta
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
from indicator_calculator import IndicatorCalculator, extract_ohlcv_from_historical_data


class LiveRatingSystem:
    """
    High-frequency real-time stock rating system with hybrid calculation model
    """
    
    def __init__(self, instrument_token: str, tradingsymbol: str, kite_trader_instance, parent_bot=None):
        """
        Initialize the live rating system for a single stock
        
        Args:
            instrument_token (str): Unique token for the stock
            tradingsymbol (str): Trading symbol of the stock (e.g., "RELIANCE")
            kite_trader_instance: Initialized KiteTrader instance (will be removed later)
            parent_bot: Reference to parent AutomatedTradingBot for state coordination
        """
        self.instrument_token = instrument_token
        self.symbol = tradingsymbol  # Store the trading symbol
        self.kite_trader = kite_trader_instance # This will be removed later
        self.parent_bot = parent_bot
        self.calculator = IndicatorCalculator()
        
        # Market hours and caching
        self._cached_tactical_score = 0.0
        self.last_trading_update = None
        self.last_market_close_rating = None
        
        # Hierarchical timeframe weights (higher weight to shorter timeframes)
        self.timeframe_weights = {
            '1minute': 0.30,    # 30% - Highest weight for immediate signals
            '3minute': 0.25,    # 25% - Very high weight for short-term momentum
            '5minute': 0.20,    # 20% - High weight for quick trends
            '15minute': 0.15,   # 15% - Medium weight for intermediate trends
            '30minute': 0.06,   # 6% - Lower weight for longer trends
            '60minute': 0.03,   # 3% - Low weight for hourly trends
            'daily': 0.01       # 1% - Minimal weight for daily trends
        }
        
        # Strategic vs Tactical weight distribution
        self.strategic_weight = self.timeframe_weights['60minute'] + self.timeframe_weights['30minute'] + self.timeframe_weights['15minute']  # 0.24 (removed daily)
        self.tactical_weight = self.timeframe_weights['5minute'] + self.timeframe_weights['3minute'] + self.timeframe_weights['1minute']  # 0.75
        
        # In-memory data store
        self.data_frames = {
            'daily': pd.DataFrame(),
            '60minute': pd.DataFrame(),
            '30minute': pd.DataFrame(),
            '15minute': pd.DataFrame(),
            '5minute': pd.DataFrame(),
            '3minute': pd.DataFrame(),
            '1minute': pd.DataFrame()
        }
        
        # Strategic score storage
        self.strategic_score = 0.0
        self.last_strategic_update = None
        self.last_60min_candle_time = None
        
        # Current incomplete candles
        self.current_candles = {
            '1minute': {'open': 0, 'high': 0, 'low': float('inf'), 'close': 0, 'volume': 0, 'start_time': None},
            '3minute': {'open': 0, 'high': 0, 'low': float('inf'), 'close': 0, 'volume': 0, 'start_time': None},
            '5minute': {'open': 0, 'high': 0, 'low': float('inf'), 'close': 0, 'volume': 0, 'start_time': None}
        }
        
        # Initialize indicator state variables
        self.last_ema_12 = None
        self.last_ema_20 = None
        self.last_ema_50 = None
        self.last_macd_line = None
        self.last_signal_line = None
        self.last_histogram = None
        self.last_rsi = None
        self.last_avg_gain = None # For incremental RSI
        self.last_avg_loss = None # For incremental RSI
        self.last_bb_upper = None
        self.last_bb_middle = None
        self.last_bb_lower = None
        self.last_psar = None
        self.last_supertrend = None
        self.last_adx = None
        self.last_di_plus = None
        self.last_di_minus = None
        
        # Initialize with historical data
        self._load_and_prepare_data()
    
    def _load_and_prepare_data(self):
        """
        Load initial historical data from local files and perform one-time full indicator calculation.
        This method no longer makes API calls.
        """
        print(f"Loading historical data for {self.symbol} from local files...")
        
        timeframe_intervals = {
            'daily': 'daily',
            '60minute': '60minute',
            '30minute': '30minute',
            '15minute': '15minute',
            '5minute': '5minute',
            '3minute': '3minute',
            '1minute': '1minute'
        }
        
        for timeframe_key, timeframe_name in timeframe_intervals.items():
            file_path = f"historical_data/{self.symbol}/{self.symbol}_{timeframe_name}.json"
            try:
                with open(file_path, 'r') as f:
                    historical_data = json.load(f)
                
                if historical_data and len(historical_data) > 0:
                    df = pd.DataFrame(historical_data)
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                    
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    self.data_frames[timeframe_key] = df
                    print(f"  {self.symbol} - {timeframe_key}: {len(df)} candles loaded from {file_path}")
                else:
                    print(f"  {self.symbol} - {timeframe_key}: No data available in {file_path}")
                    
            except FileNotFoundError:
                print(f"  Warning: Local data file not found for {self.symbol} - {timeframe_key}: {file_path}")
            except Exception as e:
                print(f"  Error loading local data for {self.symbol} - {timeframe_key} from {file_path}: {str(e)}")
        
        # Perform initial full indicator calculation for 1-minute data
        # These will be used as starting points for incremental updates
        if not self.data_frames['1minute'].empty:
            df_1min = self.data_frames['1minute']
            
            # Ensure columns are numeric
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df_1min[col] = pd.to_numeric(df_1min[col], errors='coerce')
            
            # Calculate EMAs
            df_1min.ta.ema(length=12, append=True)
            df_1min.ta.ema(length=20, append=True)
            df_1min.ta.ema(length=50, append=True)
            
            # Calculate MACD
            df_1min.ta.macd(fast=12, slow=26, signal=9, append=True)
            
            # Calculate RSI
            df_1min.ta.rsi(length=14, append=True)
            
            # Calculate Bollinger Bands
            df_1min.ta.bbands(length=20, std=2, append=True)
            
            # Calculate PSAR
            df_1min.ta.psar(append=True)
            
            # Calculate Supertrend (using default 7,3)
            df_1min.ta.supertrend(append=True)
            
            # Calculate ADX
            df_1min.ta.adx(length=14, append=True)
            
            # Store the last values of each indicator
            if not df_1min.empty:
                self.last_ema_12 = df_1min['EMA_12'].iloc[-1] if 'EMA_12' in df_1min.columns else None
                self.last_ema_20 = df_1min['EMA_20'].iloc[-1] if 'EMA_20' in df_1min.columns else None
                self.last_ema_50 = df_1min['EMA_50'].iloc[-1] if 'EMA_50' in df_1min.columns else None
                
                self.last_macd_line = df_1min['MACD_12_26_9'].iloc[-1] if 'MACD_12_26_9' in df_1min.columns else None
                self.last_signal_line = df_1min['MACDs_12_26_9'].iloc[-1] if 'MACDs_12_26_9' in df_1min.columns else None
                self.last_histogram = df_1min['MACDh_12_26_9'].iloc[-1] if 'MACDh_12_26_9' in df_1min.columns else None
                
                self.last_rsi = df_1min['RSI_14'].iloc[-1] if 'RSI_14' in df_1min.columns else None
                
                # For incremental RSI, we need previous average gain/loss
                # This is more complex and might require custom calculation or a different library
                # For now, we'll use the full calculation for RSI and update it in _update_live_indicators
                # For a true incremental RSI, you'd need to track previous avg_gain and avg_loss
                # For simplicity, we'll re-calculate RSI on a small window in _update_live_indicators if needed
                
                self.last_bb_upper = df_1min['BBU_20_2.0'].iloc[-1] if 'BBU_20_2.0' in df_1min.columns else None
                self.last_bb_middle = df_1min['BBM_20_2.0'].iloc[-1] if 'BBM_20_2.0' in df_1min.columns else None
                self.last_bb_lower = df_1min['BBL_20_2.0'].iloc[-1] if 'BBL_20_2.0' in df_1min.columns else None
                
                self.last_psar = df_1min['PSARl_0.02_0.2'].iloc[-1] if 'PSARl_0.02_0.2' in df_1min.columns else \
                                 (df_1min['PSARs_0.02_0.2'].iloc[-1] if 'PSARs_0.02_0.2' in df_1min.columns else None)
                
                self.last_supertrend = df_1min['SUPERT_7_3.0'].iloc[-1] if 'SUPERT_7_3.0' in df_1min.columns else None
                
                self.last_adx = df_1min['ADX_14'].iloc[-1] if 'ADX_14' in df_1min.columns else None
                self.last_di_plus = df_1min['DMP_14'].iloc[-1] if 'DMP_14' in df_1min.columns else None
                self.last_di_minus = df_1min['DMN_14'].iloc[-1] if 'DMN_14' in df_1min.columns else None
                
                print(f"  {self.symbol} - Initial indicator calculation complete for 1-minute data.")
            else:
                print(f"  {self.symbol} - No 1-minute data for initial indicator calculation.")
        
        # Calculate initial Strategic Base Score
        self._calculate_strategic_score()
    
    def _update_live_indicators(self, current_price: float):
        """
        Perform incremental updates for live indicators using the latest price.
        This method updates the state variables (self.last_ema_X, etc.).
        """
        if self.last_ema_12 is not None:
            multiplier_12 = 2 / (12 + 1)
            self.last_ema_12 = (current_price * multiplier_12) + (self.last_ema_12 * (1 - multiplier_12))
        
        if self.last_ema_20 is not None:
            multiplier_20 = 2 / (20 + 1)
            self.last_ema_20 = (current_price * multiplier_20) + (self.last_ema_20 * (1 - multiplier_20))
            
        if self.last_ema_50 is not None:
            multiplier_50 = 2 / (50 + 1)
            self.last_ema_50 = (current_price * multiplier_50) + (self.last_ema_50 * (1 - multiplier_50))
            
        # MACD (requires incremental EMA for fast and slow, then signal)
        # This is complex for true incremental. For simplicity, we'll re-calculate MACD on a small window
        # or assume MACD is updated less frequently. For now, we'll use the full calculation in _calculate_tactical_score
        # or re-evaluate if a truly incremental MACD is needed here.
        # For a proper incremental MACD, you'd need to track last_ema_fast, last_ema_slow, and last_signal_ema
        # For this task, we'll simplify and use the last values from the full calculation.
        
        # RSI (requires incremental average gain/loss)
        # This is also complex for true incremental. For simplicity, we'll re-calculate RSI on a small window
        # or assume RSI is updated less frequently.
        
        # Bollinger Bands (requires standard deviation, which is not easily incremental)
        # For simplicity, we'll use the last values from the full calculation.
        
        # PSAR, Supertrend, ADX are also not easily incremental with just one price.
        # These will be re-calculated on a small window or used from the last full calculation.
        
        # For the purpose of this refactoring, we will assume that for MACD, RSI, BB, PSAR, Supertrend, ADX
        # the 'last_X' values are updated by a periodic full calculation on a small window of recent data
        # or that their contribution to the tactical score is less sensitive to every single tick.
        # The primary focus for incremental update is on EMAs.
        
        # If a new candle is finalized, we would re-run full calculations for these.
        # For now, we'll just update the EMAs incrementally.
        pass # Placeholder for other incremental updates if implemented
    
    def is_market_open(self) -> bool:
        """
        Check if the NSE market is currently open
        
        Returns:
            bool: True if market is open, False otherwise
        """
        try:
            current_time = datetime.now()
            current_day = current_time.weekday()  # 0=Monday, 6=Sunday
            current_hour_min = current_time.strftime('%H:%M')
            
            # NSE trading hours: 09:15 to 15:30, Monday to Friday
            if current_day >= 5:  # Weekend (Saturday=5, Sunday=6)
                return False
            
            if '09:15' <= current_hour_min <= '15:30':
                return True
            
            return False
            
        except Exception as e:
            print(f"Error checking market hours: {str(e)}")
            return False  # Assume market closed on error

    def _calculate_strategic_score(self):
        """
        Calculate the Strategic Base Score using all strategic timeframes
        """
        try:
            strategic_scores = {}
            
            # Calculate scores for strategic timeframes only (removed daily)
            for timeframe in ['60minute', '30minute', '15minute']:
                df = self.data_frames[timeframe]
                
                if len(df) < 30:  # Reduced minimum data requirement for faster timeframes
                    strategic_scores[timeframe] = 0.0
                    continue
                
                # Convert DataFrame to OHLCV format for indicator calculator
                ohlcv_data = {
                    'open': df['open'].tolist(),
                    'high': df['high'].tolist(),
                    'low': df['low'].tolist(),
                    'close': df['close'].tolist(),
                    'volume': df['volume'].tolist()
                }
                
                # Calculate all indicator scores
                indicator_scores = self.calculator.calculate_all_indicators(ohlcv_data)
                
                # Calculate weighted timeframe score (equal weight for all indicators)
                timeframe_score = sum(indicator_scores.values()) / len(indicator_scores)
                strategic_scores[timeframe] = timeframe_score
            
            # Calculate weighted strategic score
            total_strategic_weight = 0.0
            weighted_strategic_score = 0.0
            
            for timeframe in ['60minute', '30minute', '15minute']:
                weight = self.timeframe_weights[timeframe]
                score = strategic_scores.get(timeframe, 0.0)
                
                if score != 0.0:  # Only include timeframes with valid data
                    weighted_strategic_score += weight * score
                    total_strategic_weight += weight
            
            # Normalize by actual weights used
            if total_strategic_weight > 0:
                self.strategic_score = weighted_strategic_score / total_strategic_weight
            else:
                self.strategic_score = 0.0
            
            self.last_strategic_update = datetime.now()
            
            # Store the timestamp of the last 60-minute candle
            if len(self.data_frames['60minute']) > 0:
                self.last_60min_candle_time = self.data_frames['60minute'].index[-1]
            
            print(f"Strategic Base Score calculated: {self.strategic_score:.4f} (60min+30min+15min)")
            
        except Exception as e:
            print(f"Error calculating strategic score: {str(e)}")
            self.strategic_score = 0.0
    
    def _get_candle_start_time(self, current_time: datetime, interval_minutes: int) -> datetime:
        """
        Get the start time of the current candle for a given interval
        
        Args:
            current_time (datetime): Current timestamp
            interval_minutes (int): Candle interval in minutes
            
        Returns:
            datetime: Start time of the current candle
        """
        # Round down to the nearest interval
        minutes_since_midnight = current_time.hour * 60 + current_time.minute
        candle_number = minutes_since_midnight // interval_minutes
        candle_start_minute = candle_number * interval_minutes
        
        start_hour = candle_start_minute // 60
        start_minute = candle_start_minute % 60
        
        return current_time.replace(hour=start_hour, minute=start_minute, second=0, microsecond=0)
    
    def _update_candle(self, timeframe: str, tick_data: Dict[str, Any]):
        """
        Update the current incomplete candle with new tick data
        
        Args:
            timeframe (str): Timeframe to update ('1minute', '3minute', '5minute')
            tick_data (Dict): Tick data containing price and volume information
        """
        try:
            current_time = datetime.now()
            interval_minutes = int(timeframe.replace('minute', ''))
            candle_start_time = self._get_candle_start_time(current_time, interval_minutes)
            
            candle = self.current_candles[timeframe]
            
            # Check if this is a new candle
            if candle['start_time'] is None or candle_start_time > candle['start_time']:
                # Finalize previous candle if it exists
                if candle['start_time'] is not None and candle['open'] > 0:
                    self._finalize_candle(timeframe, candle)
                
                # Start new candle
                candle['start_time'] = candle_start_time
                candle['open'] = tick_data.get('last_price', 0)
                candle['high'] = tick_data.get('last_price', 0)
                candle['low'] = tick_data.get('last_price', 0)
                candle['close'] = tick_data.get('last_price', 0)
                candle['volume'] = tick_data.get('volume_traded', 0)
            else:
                # Update existing candle
                price = tick_data.get('last_price', 0)
                if price > 0:
                    candle['high'] = max(candle['high'], price)
                    candle['low'] = min(candle['low'], price)
                    candle['close'] = price
                    candle['volume'] = tick_data.get('volume_traded', 0)
                    
        except Exception as e:
            print(f"Error updating {timeframe} candle: {str(e)}")
    
    def _finalize_candle(self, timeframe: str, candle_data: Dict[str, Any]):
        """
        Finalize a completed candle and add it to the DataFrame
        
        Args:
            timeframe (str): Timeframe of the candle
            candle_data (Dict): Completed candle data
        """
        try:
            if candle_data['open'] <= 0 or candle_data['start_time'] is None:
                return
            
            # Create new row
            new_row = pd.DataFrame({
                'open': [candle_data['open']],
                'high': [candle_data['high']],
                'low': [candle_data['low']],
                'close': [candle_data['close']],
                'volume': [candle_data['volume']]
            }, index=[candle_data['start_time']])
            
            # Add to DataFrame
            if len(self.data_frames[timeframe]) == 0:
                self.data_frames[timeframe] = new_row
            else:
                self.data_frames[timeframe] = pd.concat([self.data_frames[timeframe], new_row])
            
            # Keep only recent data (memory management)
            max_rows = {'1minute': 500, '3minute': 300, '5minute': 200}
            if len(self.data_frames[timeframe]) > max_rows.get(timeframe, 200):
                self.data_frames[timeframe] = self.data_frames[timeframe].tail(max_rows.get(timeframe, 200))
            
            print(f"Finalized {timeframe} candle at {candle_data['start_time']}")
            
        except Exception as e:
            print(f"Error finalizing {timeframe} candle: {str(e)}")
    
    def update_with_tick(self, tick_data: Dict[str, Any]):
        """
        Process a new tick from the WebSocket
        
        Args:
            tick_data (Dict): Tick object from KiteTicker WebSocket
        """
        try:
            # Update current candles for tactical timeframes
            for timeframe in ['1minute', '3minute', '5minute']:
                self._update_candle(timeframe, tick_data)
            
            # Update live indicators with the latest price
            current_price = tick_data.get('last_price', 0)
            if current_price > 0:
                self._update_live_indicators(current_price)
                
        except Exception as e:
            print(f"Error processing tick: {str(e)}")
    
    def _calculate_tactical_score(self) -> float:
        """
        Calculate the Tactical Live Score using the live, stateful indicator values.
        This method no longer performs heavy calculations itself.
        
        Returns:
            float: Weighted tactical score
        """
        try:
            tactical_scores = {}
            
            # Ensure all required indicators have been initialized
            if any(x is None for x in [self.last_ema_12, self.last_ema_20, self.last_ema_50,
                                       self.last_macd_line, self.last_signal_line, self.last_histogram,
                                       self.last_rsi, self.last_bb_upper, self.last_bb_middle, self.last_bb_lower,
                                       self.last_psar, self.last_supertrend, self.last_adx, self.last_di_plus, self.last_di_minus]):
                print(f"  Warning: Not all tactical indicators initialized for {self.symbol}. Skipping tactical score calculation.")
                return 0.0

            # --- Scoring based on live indicator values ---
            # Each indicator contributes a score (e.g., -1 to +1)
            # These scores are then weighted and summed.

            # EMA Crossover Score (e.g., 12 > 20 > 50 for bullish)
            ema_score = 0.0
            if self.last_ema_12 > self.last_ema_20 and self.last_ema_20 > self.last_ema_50:
                ema_score = 1.0 # Strong bullish
            elif self.last_ema_12 < self.last_ema_20 and self.last_ema_20 < self.last_ema_50:
                ema_score = -1.0 # Strong bearish
            elif self.last_ema_12 > self.last_ema_20:
                ema_score = 0.5 # Weak bullish
            elif self.last_ema_12 < self.last_ema_20:
                ema_score = -0.5 # Weak bearish
            
            # MACD Score
            macd_score = 0.0
            if self.last_macd_line > self.last_signal_line and self.last_histogram > 0:
                macd_score = 1.0 # Bullish crossover above zero
            elif self.last_macd_line < self.last_signal_line and self.last_histogram < 0:
                macd_score = -1.0 # Bearish crossover below zero
            elif self.last_macd_line > self.last_signal_line:
                macd_score = 0.5 # Bullish crossover
            elif self.last_macd_line < self.last_signal_line:
                macd_score = -0.5 # Bearish crossover

            # RSI Score
            rsi_score = 0.0
            if self.last_rsi > 70:
                rsi_score = -1.0 # Overbought
            elif self.last_rsi < 30:
                rsi_score = 1.0 # Oversold
            elif self.last_rsi > 50:
                rsi_score = 0.5 # Bullish momentum
            elif self.last_rsi < 50:
                rsi_score = -0.5 # Bearish momentum

            # Bollinger Bands Score (using last_bb_middle as current price reference)
            bb_score = 0.0
            if self.current_candles['1minute']['close'] > self.last_bb_upper:
                bb_score = -1.0 # Overbought (price above upper band)
            elif self.current_candles['1minute']['close'] < self.last_bb_lower:
                bb_score = 1.0 # Oversold (price below lower band)
            elif self.current_candles['1minute']['close'] > self.last_bb_middle:
                bb_score = 0.5 # Price above middle band
            elif self.current_candles['1minute']['close'] < self.last_bb_middle:
                bb_score = -0.5 # Price below middle band

            # PSAR Score
            psar_score = 0.0
            if self.current_candles['1minute']['close'] > self.last_psar:
                psar_score = 1.0 # Bullish (PSAR below price)
            elif self.current_candles['1minute']['close'] < self.last_psar:
                psar_score = -1.0 # Bearish (PSAR above price)

            # Supertrend Score
            supertrend_score = 0.0
            if self.current_candles['1minute']['close'] > self.last_supertrend:
                supertrend_score = 1.0 # Bullish (price above Supertrend)
            elif self.current_candles['1minute']['close'] < self.last_supertrend:
                supertrend_score = -1.0 # Bearish (price below Supertrend)

            # ADX Score
            adx_score = 0.0
            if self.last_adx > 25:
                if self.last_di_plus > self.last_di_minus:
                    adx_score = 1.0 # Strong uptrend
                elif self.last_di_minus > self.last_di_plus:
                    adx_score = -1.0 # Strong downtrend
            elif self.last_adx < 20:
                adx_score = 0.0 # Weak/no trend

            # Combine tactical scores with equal weighting for now
            # You can adjust these weights based on strategy importance
            total_tactical_score = (
                ema_score + macd_score + rsi_score + bb_score + 
                psar_score + supertrend_score + adx_score
            )
            
            # Normalize to a range (e.g., -1 to +1)
            # Max possible score is 7.0, min is -7.0
            normalized_tactical_score = total_tactical_score / 7.0
            
            return normalized_tactical_score
                
        except Exception as e:
            print(f"Error calculating tactical score for {self.symbol}: {str(e)}")
            return 0.0
    
    def _should_update_strategic_score(self) -> bool:
        """
        Check if the strategic score needs to be updated (every 10 minutes)
        with awareness of batch processing state

        Returns:
            bool: True if strategic score should be updated
        """
        if self.last_strategic_update is None:
            return True

        # Check if parent bot is in batch processing state
        if hasattr(self, 'parent_bot') and self.parent_bot:
            try:
                import threading
                universe_lock = getattr(self.parent_bot, 'universe_lock', threading.Lock())
                with universe_lock:
                    transition_state = getattr(self.parent_bot, 'transition_state', 'stable')
                    
                    # Don't update strategic score during batch processing or transitioning
                    if transition_state in ['batch_processing', 'transitioning']:
                        return False
            except Exception as e:
                print(f"Warning: Could not check parent bot state: {str(e)}")

        # Update every 10 minutes
        time_since_update = datetime.now() - self.last_strategic_update
        return time_since_update.total_seconds() >= 600  # 10 minutes
    
    def get_live_rating(self) -> Dict[str, Any]:
        """
        Get the final, up-to-the-second rating for the stock with market hours awareness

        Returns:
            Dict: Contains final rating, component scores, and metadata
        """
        try:
            market_open = self.is_market_open()
            
            # Cache rating during market closure
            if not market_open and self.last_market_close_rating:
                # Return cached rating with updated timestamp
                cached_rating = self.last_market_close_rating.copy()
                cached_rating['timestamp'] = datetime.now().isoformat()
                cached_rating['market_open'] = False
                cached_rating['data_source'] = 'cached'
                return cached_rating
            
            # Check for 10-minute strategic update (only during market hours)
            if self._should_update_strategic_score():
                print("Updating strategic score (10-minute cycle)...")
                self._calculate_strategic_score()
            
            # Calculate tactical score (cached during market closure)
            tactical_score = self._calculate_tactical_score()
            
            # Combine scores with proper weighting
            # Strategic score is already weighted internally
            # Tactical score needs to be weighted relative to strategic
            strategic_contribution = self.strategic_score * self.strategic_weight
            tactical_contribution = tactical_score * self.tactical_weight
            
            # Final composite score
            composite_score = strategic_contribution + tactical_contribution
            
            # Scale to final rating (-10 to +10)
            final_rating = composite_score * 10
            
            # Determine rating category
            if final_rating >= 7.5:
                rating_text = "Strong Buy"
                emoji = "🟢"
            elif final_rating >= 2.5:
                rating_text = "Buy"
                emoji = "🟢"
            elif final_rating >= -2.4:
                rating_text = "Neutral"
                emoji = "🟡"
            elif final_rating >= -7.4:
                rating_text = "Sell"
                emoji = "🔴"
            else:
                rating_text = "Strong Sell"
                emoji = "🔴"
            
            # Prepare rating data
            rating_data = {
                'instrument_token': self.instrument_token,
                'timestamp': datetime.now().isoformat(),
                'final_rating': round(final_rating, 2),
                'rating_text': rating_text,
                'emoji': emoji,
                'composite_score': round(composite_score, 4),
                'strategic_score': round(self.strategic_score, 4),
                'tactical_score': round(tactical_score, 4),
                'strategic_contribution': round(strategic_contribution, 4),
                'tactical_contribution': round(tactical_contribution, 4),
                'last_strategic_update': self.last_strategic_update.isoformat() if self.last_strategic_update else None,
                'last_trading_update': self.last_trading_update.isoformat() if self.last_trading_update else None,
                'market_open': market_open,
                'data_source': 'live' if market_open else 'cached',
                'data_quality': {
                    'daily_candles': len(self.data_frames['daily']),
                    '60min_candles': len(self.data_frames['60minute']),
                    '5min_candles': len(self.data_frames['5minute']),
                    '3min_candles': len(self.data_frames['3minute']),
                    '1min_candles': len(self.data_frames['1minute'])
                }
            }
            
            # Cache rating for use during market closure
            if market_open:
                self.last_market_close_rating = rating_data.copy()
            
            return rating_data
            
        except Exception as e:
            print(f"Error getting live rating: {str(e)}")
            return {
                'instrument_token': self.instrument_token,
                'timestamp': datetime.now().isoformat(),
                'final_rating': 0.0,
                'rating_text': 'Error',
                'emoji': '❌',
                'market_open': self.is_market_open(),
                'data_source': 'error',
                'error': str(e)
            }
    
    def get_current_candles_status(self) -> Dict[str, Any]:
        """
        Get status of current incomplete candles for debugging
        
        Returns:
            Dict: Current candle information
        """
        status = {}
        for timeframe, candle in self.current_candles.items():
            status[timeframe] = {
                'start_time': candle['start_time'].isoformat() if candle['start_time'] else None,
                'open': candle['open'],
                'high': candle['high'],
                'low': candle['low'] if candle['low'] != float('inf') else 0,
                'close': candle['close'],
                'volume': candle['volume']
            }
        return status


# Example usage and testing
if __name__ == "__main__":
    print("LiveRatingSystem module loaded successfully")
    print("This module requires a KiteTrader instance to function properly")
    print("Example usage:")
    print("""
    from kiteConnect import KiteTrader, Config
    from live_rating_system import LiveRatingSystem
    
    # Initialize trader
    trader = KiteTrader(api_key=Config.API_KEY, access_token=access_token)
    
    # Initialize live rating system for a stock
    live_system = LiveRatingSystem(instrument_token="738561", kite_trader_instance=trader)
    
    # Get live rating
    rating = live_system.get_live_rating()
    print(f"Live Rating: {rating['final_rating']} - {rating['rating_text']}")
    
    # Process tick data (from WebSocket)
    tick_data = {'last_price': 2450.50, 'volume_traded': 1000}
    live_system.update_with_tick(tick_data)
    """)
