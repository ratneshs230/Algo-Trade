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
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
from indicator_calculator import IndicatorCalculator, extract_ohlcv_from_historical_data


class LiveRatingSystem:
    """
    High-frequency real-time stock rating system with hybrid calculation model
    """
    
    def __init__(self, instrument_token: str, kite_trader_instance, parent_bot=None):
        """
        Initialize the live rating system for a single stock
        
        Args:
            instrument_token (str): Unique token for the stock
            kite_trader_instance: Initialized KiteTrader instance for data fetching
            parent_bot: Reference to parent AutomatedTradingBot for state coordination
        """
        self.instrument_token = instrument_token
        self.kite_trader = kite_trader_instance
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
        
        # Initialize with historical data
        self._load_and_prepare_data()
    
    def _load_and_prepare_data(self):
        """
        Fetch initial historical data and calculate the Strategic Base Score
        """
        try:
            print(f"Loading historical data for instrument {self.instrument_token}...")
            
            # Fetch historical data for all timeframes
            timeframe_intervals = {
                'daily': {'interval': 'day', 'days_back': 200},
                '60minute': {'interval': '60minute', 'days_back': 60},
                '30minute': {'interval': '30minute', 'days_back': 30},
                '15minute': {'interval': '15minute', 'days_back': 15},
                '5minute': {'interval': '5minute', 'days_back': 10},
                '3minute': {'interval': '3minute', 'days_back': 5},
                '1minute': {'interval': 'minute', 'days_back': 2}
            }
            
            for timeframe, params in timeframe_intervals.items():
                try:
                    # Calculate date range
                    to_date = datetime.now()
                    from_date = to_date - timedelta(days=params['days_back'])
                    
                    # Fetch historical data
                    historical_data = self.kite_trader.get_historical_data(
                        instrument_token=int(self.instrument_token),
                        from_date=from_date,
                        to_date=to_date,
                        interval=params['interval']
                    )
                    
                    if historical_data and len(historical_data) > 0:
                        # Convert to DataFrame
                        df = pd.DataFrame(historical_data)
                        df['date'] = pd.to_datetime(df['date'])
                        df = df.set_index('date')
                        
                        # Ensure numeric columns
                        for col in ['open', 'high', 'low', 'close', 'volume']:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        
                        self.data_frames[timeframe] = df
                        print(f"  {timeframe}: {len(df)} candles loaded")
                    else:
                        print(f"  {timeframe}: No data available")
                        
                except Exception as e:
                    print(f"  {timeframe}: Error loading data - {str(e)}")
            
            # Calculate initial Strategic Base Score
            self._calculate_strategic_score()
            
        except Exception as e:
            print(f"Error in _load_and_prepare_data: {str(e)}")
            raise
    
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
                
        except Exception as e:
            print(f"Error processing tick: {str(e)}")
    
    def _calculate_tactical_score(self) -> float:
        """
        Calculate the Tactical Live Score using 1-minute, 3-minute, and 5-minute data
        
        Returns:
            float: Weighted tactical score
        """
        try:
            tactical_scores = {}
            
            # Calculate scores for tactical timeframes
            for timeframe in ['5minute', '3minute', '1minute']:
                df = self.data_frames[timeframe]
                
                if len(df) < 20:  # Minimum data requirement for tactical
                    tactical_scores[timeframe] = 0.0
                    continue
                
                # Include current incomplete candle for real-time analysis
                current_candle = self.current_candles[timeframe]
                if current_candle['start_time'] is not None and current_candle['open'] > 0:
                    # Create temporary DataFrame with current candle
                    current_row = pd.DataFrame({
                        'open': [current_candle['open']],
                        'high': [current_candle['high']],
                        'low': [current_candle['low']],
                        'close': [current_candle['close']],
                        'volume': [current_candle['volume']]
                    }, index=[current_candle['start_time']])
                    
                    temp_df = pd.concat([df, current_row])
                else:
                    temp_df = df
                
                # Convert to OHLCV format
                ohlcv_data = {
                    'open': temp_df['open'].tolist(),
                    'high': temp_df['high'].tolist(),
                    'low': temp_df['low'].tolist(),
                    'close': temp_df['close'].tolist(),
                    'volume': temp_df['volume'].tolist()
                }
                
                # Calculate indicator scores
                indicator_scores = self.calculator.calculate_all_indicators(ohlcv_data)
                
                # Calculate weighted timeframe score
                timeframe_score = sum(indicator_scores.values()) / len(indicator_scores)
                tactical_scores[timeframe] = timeframe_score
            
            # Calculate weighted tactical score
            total_tactical_weight = 0.0
            weighted_tactical_score = 0.0
            
            for timeframe in ['5minute', '3minute', '1minute']:
                weight = self.timeframe_weights[timeframe]
                score = tactical_scores.get(timeframe, 0.0)
                
                if score != 0.0:  # Only include timeframes with valid data
                    weighted_tactical_score += weight * score
                    total_tactical_weight += weight
            
            # Normalize by actual weights used
            if total_tactical_weight > 0:
                return weighted_tactical_score / total_tactical_weight
            else:
                return 0.0
                
        except Exception as e:
            print(f"Error calculating tactical score: {str(e)}")
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
