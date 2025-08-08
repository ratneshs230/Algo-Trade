"""
Precision Single Trade Strategy Engine

This module implements a sophisticated single-position trading strategy with:
1. Market Scanner: Selects highest absolute-rated stock from watchlist
2. Precision Entry: Dual confirmation using Supertrend + PSAR on 1-minute chart
3. Risk Management: Full margin allocation with Supertrend stop-loss
4. Dynamic Trade Management: 3-phase progression (Risk → Risk-Free → Profit-Maximizing)
5. Fibonacci Trailing: Dynamic stops using Fibonacci levels from swing data

Strategy ensures only ONE position is open at a time with maximum capital utilization.
"""

import json
import os
import time
import threading
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

from indicator_calculator import IndicatorCalculator, extract_ohlcv_from_historical_data
from kiteConnect import KiteTrader
from TradeChargesCalculator import TradeChargesCalculator

logger = logging.getLogger(__name__)


class TradePhase(Enum):
    """Trade management phases"""
    RISK = "RISK"                           # Phase A: Initial risk with Supertrend stop-loss
    RISK_FREE = "RISK_FREE"                 # Phase B: Risk eliminated, stop at breakeven
    PROFIT_MAXIMIZING = "PROFIT_MAXIMIZING" # Phase C: Fibonacci ladder trailing


class TradeDirection(Enum):
    """Trade direction"""
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class TradingCandidate:
    """Represents a potential trading candidate from market scan"""
    symbol: str
    instrument_token: str
    final_rating: float
    absolute_rating: float
    rating_data: Dict
    instrument_info: Dict


@dataclass
class TradeSetup:
    """Complete trade setup information"""
    candidate: TradingCandidate
    direction: TradeDirection
    entry_price: float
    entry_time: datetime
    quantity: int
    margin_used: float
    
    # Risk management
    initial_stop_loss: float
    breakeven_target: float
    supertrend_value: float
    psar_value: float
    
    # Order management
    entry_order_id: Optional[str] = None
    stop_loss_order_id: Optional[str] = None
    is_active: bool = False


@dataclass
class TradeState:
    """Current state of active trade"""
    setup: TradeSetup
    phase: TradePhase
    current_price: float
    current_stop_loss: float
    fibonacci_levels: Dict[str, float]
    last_fibonacci_level: Optional[str]
    
    # Phase tracking
    breakeven_hit: bool = False
    profit_secured: float = 0.0
    max_profit_reached: float = 0.0
    
    # Performance tracking
    unrealized_pnl: float = 0.0
    phase_start_time: datetime = datetime.now()


class PrecisionSingleTradeStrategy:
    """
    Main strategy class implementing precision single-position trading
    """
    
    def __init__(self, kite_trader: KiteTrader, instruments: List[Dict], 
                 instruments_data: List[Dict], margin_percentage: float = 0.95):
        """
        Initialize the strategy
        
        Args:
            kite_trader: KiteTrader instance for order execution
            instruments: List of filtered watchlist instruments
            instruments_data: Instrument data with tick sizes
            margin_percentage: Percentage of available margin to use (default 95%)
        """
        self.kite_trader = kite_trader
        self.instruments = instruments
        self.instruments_data = instruments_data
        self.margin_percentage = margin_percentage
        
        # Strategy components
        self.indicator_calculator = IndicatorCalculator()
        
        # Position management
        self.active_trade: Optional[TradeState] = None
        self.position_lock = threading.Lock()
        
        # Instrument lookup for quick access
        self.instrument_lookup = {
            inst['tradingsymbol']: inst for inst in instruments_data
        }
        
        # Strategy configuration
        self.min_rating_threshold = 2.5  # Minimum absolute rating to consider
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_profit = 0.0
        
        logger.info("Precision Single Trade Strategy initialized")
        logger.info(f"Monitoring {len(self.instruments)} instruments")
        logger.info(f"Margin allocation: {margin_percentage*100:.1f}%")
    
    def can_trade(self) -> bool:
        """Check if strategy can execute a new trade"""
        with self.position_lock:
            return self.active_trade is None
    
    def get_tick_size(self, symbol: str) -> float:
        """Get tick size for a symbol from instruments data"""
        if symbol in self.instrument_lookup:
            return float(self.instrument_lookup[symbol].get('tick_size', 0.05))
        return 0.05  # Default tick size
    
    def round_to_tick_size(self, price: float, symbol: str, round_up: bool = True) -> float:
        """Round price to valid tick size"""
        tick_size = self.get_tick_size(symbol)
        if round_up:
            return math.ceil(price / tick_size) * tick_size
        else:
            return math.floor(price / tick_size) * tick_size
    
    def select_trading_candidate(self, live_ratings: Dict[str, Dict]) -> Optional[TradingCandidate]:
        """
        Phase 1: Market Scanner - Select highest absolute-rated stock
        
        Args:
            live_ratings: Current live ratings from rating system
            
        Returns:
            TradingCandidate with highest absolute rating or None
        """
        try:
            if not self.can_trade():
                logger.debug("Cannot trade - position already active")
                return None
            
            candidates = []
            
            for symbol, rating_data in live_ratings.items():
                if 'error' in rating_data:
                    continue
                
                final_rating = rating_data.get('final_rating', 0)
                absolute_rating = abs(final_rating)
                
                # Filter by minimum threshold
                if absolute_rating < self.min_rating_threshold:
                    continue
                
                # Find instrument info
                instrument_info = None
                for inst in self.instruments:
                    if inst.get('tradingsymbol') == symbol:
                        instrument_info = inst
                        break
                
                if not instrument_info:
                    continue
                
                candidate = TradingCandidate(
                    symbol=symbol,
                    instrument_token=str(instrument_info.get('instrument_token', '')),
                    final_rating=final_rating,
                    absolute_rating=absolute_rating,
                    rating_data=rating_data,
                    instrument_info=instrument_info
                )
                
                candidates.append(candidate)
            
            if not candidates:
                logger.debug("No suitable trading candidates found")
                return None
            
            # Select candidate with highest absolute rating
            best_candidate = max(candidates, key=lambda x: x.absolute_rating)
            
            logger.info(f"Selected trading candidate: {best_candidate.symbol}")
            logger.info(f"  Final Rating: {best_candidate.final_rating:.2f}")
            logger.info(f"  Absolute Rating: {best_candidate.absolute_rating:.2f}")
            logger.info(f"  Direction Bias: {'BULLISH' if best_candidate.final_rating > 0 else 'BEARISH'}")
            
            return best_candidate
            
        except Exception as e:
            logger.error(f"Error selecting trading candidate: {str(e)}")
            return None
    
    def get_1minute_ohlcv_data(self, symbol: str, periods: int = 100) -> Optional[Dict[str, List[float]]]:
        """Get recent 1-minute OHLCV data for a symbol"""
        try:
            # Try to read from historical data files first
            data_file = f"historical_data/{symbol}/{symbol}_1minute.json"
            if os.path.exists(data_file):
                with open(data_file, 'r') as f:
                    historical_data = json.load(f)
                
                if len(historical_data) >= periods:
                    # Take the most recent data
                    recent_data = historical_data[-periods:]
                    return extract_ohlcv_from_historical_data(recent_data)
            
            # Fallback to API call if file not available
            instrument_token = None
            for inst in self.instruments:
                if inst.get('tradingsymbol') == symbol:
                    instrument_token = int(inst.get('instrument_token', 0))
                    break
            
            if not instrument_token:
                return None
            
            to_date = datetime.now()
            from_date = to_date - timedelta(days=2)  # Get last 2 days of 1-minute data
            
            historical_data = self.kite_trader.get_historical_data(
                instrument_token=instrument_token,
                from_date=from_date,
                to_date=to_date,
                interval='minute'
            )
            
            if historical_data and len(historical_data) >= periods:
                recent_data = historical_data[-periods:]
                return extract_ohlcv_from_historical_data(recent_data)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting 1-minute data for {symbol}: {str(e)}")
            return None
    
    def check_entry_confirmation(self, candidate: TradingCandidate) -> Optional[Tuple[TradeDirection, float, float]]:
        """
        Phase 2: Precision Entry Trigger - Check Supertrend + PSAR confirmation
        
        Args:
            candidate: Selected trading candidate
            
        Returns:
            Tuple of (direction, supertrend_value, psar_value) or None
        """
        try:
            logger.info(f"Checking entry confirmation for {candidate.symbol}")
            
            # Get 1-minute OHLCV data
            ohlcv_data = self.get_1minute_ohlcv_data(candidate.symbol)
            if not ohlcv_data:
                logger.warning(f"No 1-minute data available for {candidate.symbol}")
                return None
            
            high = ohlcv_data['high']
            low = ohlcv_data['low']
            close = ohlcv_data['close']
            
            if len(close) < 20:  # Need minimum data for calculations
                logger.warning(f"Insufficient data for {candidate.symbol}: {len(close)} candles")
                return None
            
            # Calculate Supertrend (ATR Period 5, Factor 3)
            supertrend_signal = self.indicator_calculator.get_supertrend_signal(
                high, low, close, period=5, factor=3.0
            )
            supertrend_values = self.indicator_calculator.calculate_supertrend(
                high, low, close, period=5, factor=3.0
            )
            
            # Calculate PSAR
            psar_signal = self.indicator_calculator.get_psar_signal(high, low, close)
            psar_values = self.indicator_calculator.calculate_parabolic_sar(high, low, close)
            
            if len(supertrend_values) == 0 or len(psar_values) == 0:
                logger.warning(f"Could not calculate indicators for {candidate.symbol}")
                return None
            
            current_price = close[-1]
            current_supertrend = supertrend_values[-1] if not math.isnan(supertrend_values[-1]) else None
            current_psar = psar_values[-1] if not math.isnan(psar_values[-1]) else None
            
            if current_supertrend is None or current_psar is None:
                logger.warning(f"Invalid indicator values for {candidate.symbol}")
                return None
            
            logger.info(f"  Current Price: {current_price:.2f}")
            logger.info(f"  Supertrend: {current_supertrend:.2f} (Signal: {supertrend_signal})")
            logger.info(f"  PSAR: {current_psar:.2f} (Signal: {psar_signal})")
            
            # Check for dual confirmation
            # Long Entry: Price > Supertrend AND PSAR < Price
            if (supertrend_signal == "BULLISH" and psar_signal == "BULLISH" and
                current_price > current_supertrend and current_psar < current_price):
                
                logger.info(f"✅ LONG entry confirmed for {candidate.symbol}")
                return (TradeDirection.LONG, current_supertrend, current_psar)
            
            # Short Entry: Price < Supertrend AND PSAR > Price  
            elif (supertrend_signal == "BEARISH" and psar_signal == "BEARISH" and
                  current_price < current_supertrend and current_psar > current_price):
                
                logger.info(f"✅ SHORT entry confirmed for {candidate.symbol}")
                return (TradeDirection.SHORT, current_supertrend, current_psar)
            
            else:
                logger.info(f"❌ No entry confirmation for {candidate.symbol}")
                logger.info(f"   Supertrend: {supertrend_signal}, PSAR: {psar_signal}")
                return None
                
        except Exception as e:
            logger.error(f"Error checking entry confirmation: {str(e)}")
            return None
    
    def calculate_position_size(self, entry_price: float, stop_loss: float, 
                              direction: TradeDirection) -> Tuple[int, float]:
        """
        Calculate position size based on available margin
        
        Args:
            entry_price: Entry price for the trade
            stop_loss: Stop loss price
            direction: Trade direction
            
        Returns:
            Tuple of (quantity, margin_required)
        """
        try:
            # Get available margins
            margins = self.kite_trader.get_margins()
            if not margins or 'equity' not in margins:
                logger.error("Could not fetch margin information")
                return 0, 0.0
            
            available_cash = float(margins['equity'].get('available', {}).get('cash', 0))
            available_margin = available_cash * self.margin_percentage
            
            logger.info(f"Available cash: ₹{available_cash:,.2f}")
            logger.info(f"Using {self.margin_percentage*100:.1f}% = ₹{available_margin:,.2f}")
            
            # Calculate quantity based on available margin
            # For MIS (intraday), we can get leverage
            # Conservative approach: assume 4x leverage for most stocks
            leverage = 4.0
            buying_power = available_margin * leverage
            
            quantity = int(buying_power / entry_price)
            
            # Ensure minimum quantity of 1
            quantity = max(1, quantity)
            
            # Calculate actual margin required
            margin_required = (quantity * entry_price) / leverage
            
            logger.info(f"Calculated position size:")
            logger.info(f"  Quantity: {quantity}")
            logger.info(f"  Entry Price: ₹{entry_price:.2f}")
            logger.info(f"  Total Value: ₹{quantity * entry_price:,.2f}")
            logger.info(f"  Margin Required: ₹{margin_required:,.2f}")
            logger.info(f"  Leverage Used: {leverage}x")
            
            return quantity, margin_required
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0, 0.0
    
    def calculate_breakeven_price(self, entry_price: float, symbol: str, 
                                quantity: int, direction: TradeDirection) -> float:
        """
        Calculate precise breakeven price by iteratively accounting for exit costs.
        
        Args:
            entry_price: Entry price
            symbol: Symbol for tick size lookup
            quantity: Quantity traded
            direction: Trade direction
            
        Returns:
            Breakeven price rounded to the nearest valid tick size.
        """
        try:
            logger.info(f"Calculating precise breakeven for {symbol}...")
            logger.info(f"  Entry Price: ₹{entry_price:.2f}, Quantity: {quantity}")

            # Initial guess for breakeven is just the entry price
            breakeven_guess = entry_price
            
            # Iteratively refine the breakeven price
            for i in range(10): # 10 iterations is more than enough for convergence
                # Calculate total charges assuming we exit at our current guessed breakeven price
                charges_calculator = TradeChargesCalculator(
                    quantity=quantity, 
                    buy_price=entry_price, 
                    sell_price=breakeven_guess
                )
                total_costs = charges_calculator.total_charges()
                cost_per_share = total_costs / quantity

                # Calculate the new breakeven price based on these costs
                if direction == TradeDirection.LONG:
                    new_breakeven = entry_price + cost_per_share
                else: # SHORT
                    new_breakeven = entry_price - cost_per_share

                # If the new breakeven price is very close to the last guess, we've converged
                if abs(new_breakeven - breakeven_guess) < 0.0001:
                    break
                
                breakeven_guess = new_breakeven

            # Final rounding to the valid tick size
            if direction == TradeDirection.LONG:
                final_breakeven = self.round_to_tick_size(breakeven_guess, symbol, round_up=True)
            else: # SHORT
                final_breakeven = self.round_to_tick_size(breakeven_guess, symbol, round_up=False)

            final_charges = TradeChargesCalculator(quantity, entry_price, final_breakeven).total_charges()

            logger.info(f"Breakeven calculation converged:")
            logger.info(f"  Total Charges (estimated): ₹{final_charges:.2f}")
            logger.info(f"  Cost per Share: ₹{final_charges / quantity:.4f}")
            logger.info(f"  Final Breakeven Target: ₹{final_breakeven:.2f}")
            
            return final_breakeven
            
        except Exception as e:
            logger.error(f"Error calculating breakeven price: {str(e)}")
            # Fallback to a simple calculation on error
            return entry_price + (entry_price * 0.001) if direction == TradeDirection.LONG else entry_price - (entry_price * 0.001)
    
    def execute_trade(self, candidate: TradingCandidate, direction: TradeDirection, 
                     supertrend_value: float, psar_value: float) -> Optional[TradeState]:
        """
        Phase 3: Execute trade with full risk management setup
        
        Args:
            candidate: Trading candidate
            direction: Trade direction
            supertrend_value: Current Supertrend value
            psar_value: Current PSAR value
            
        Returns:
            TradeState if successful, None otherwise
        """
        try:
            with self.position_lock:
                if self.active_trade is not None:
                    logger.warning("Cannot execute trade - position already active")
                    return None
                
                logger.info(f"🚀 Executing {direction.value} trade for {candidate.symbol}")
                
                # Get current market price
                ltp_data = self.kite_trader.get_ltp([f"NSE:{candidate.symbol}"])
                if not ltp_data or f"NSE:{candidate.symbol}" not in ltp_data:
                    logger.error(f"Could not get LTP for {candidate.symbol}")
                    return None
                
                entry_price = float(ltp_data[f"NSE:{candidate.symbol}"]["last_price"])
                
                # Calculate position size
                quantity, margin_required = self.calculate_position_size(
                    entry_price, supertrend_value, direction
                )
                
                if quantity == 0:
                    logger.error("Cannot calculate position size")
                    return None
                
                # Calculate breakeven target
                breakeven_target = self.calculate_breakeven_price(
                    entry_price, candidate.symbol, quantity, direction
                )
                
                # Place market order for entry
                transaction_type = "BUY" if direction == TradeDirection.LONG else "SELL"
                
                entry_order = self.kite_trader.place_order(
                    variety="regular",
                    exchange="NSE",
                    tradingsymbol=candidate.symbol,
                    transaction_type=transaction_type,
                    quantity=quantity,
                    product="MIS",  # Intraday
                    order_type="MARKET",
                    validity="DAY"
                )
                
                if entry_order["status"] != "success":
                    logger.error(f"Failed to place entry order: {entry_order}")
                    return None
                
                entry_order_id = entry_order["order_id"]
                logger.info(f"✅ Entry order placed: {entry_order_id}")
                
                # Wait a moment for order execution
                time.sleep(2)
                
                # Place initial stop-loss order at Supertrend level
                stop_loss_price = self.round_to_tick_size(supertrend_value, candidate.symbol)
                stop_transaction_type = "SELL" if direction == TradeDirection.LONG else "BUY"
                
                stop_order = self.kite_trader.place_order(
                    variety="regular",
                    exchange="NSE",
                    tradingsymbol=candidate.symbol,
                    transaction_type=stop_transaction_type,
                    quantity=quantity,
                    product="MIS",
                    order_type="SL-M",  # Stop-Loss Market
                    trigger_price=stop_loss_price,
                    validity="DAY"
                )
                
                stop_loss_order_id = None
                if stop_order["status"] == "success":
                    stop_loss_order_id = stop_order["order_id"]
                    logger.info(f"✅ Stop-loss order placed: {stop_loss_order_id}")
                else:
                    logger.warning(f"Failed to place stop-loss order: {stop_order}")
                
                # Create trade setup
                trade_setup = TradeSetup(
                    candidate=candidate,
                    direction=direction,
                    entry_price=entry_price,
                    entry_time=datetime.now(),
                    quantity=quantity,
                    margin_used=margin_required,
                    initial_stop_loss=stop_loss_price,
                    breakeven_target=breakeven_target,
                    supertrend_value=supertrend_value,
                    psar_value=psar_value,
                    entry_order_id=entry_order_id,
                    stop_loss_order_id=stop_loss_order_id,
                    is_active=True
                )
                
                # Create trade state
                trade_state = TradeState(
                    setup=trade_setup,
                    phase=TradePhase.RISK,
                    current_price=entry_price,
                    current_stop_loss=stop_loss_price,
                    fibonacci_levels={},
                    last_fibonacci_level=None
                )
                
                self.active_trade = trade_state
                
                logger.info("🎯 Trade executed successfully!")
                logger.info(f"  Symbol: {candidate.symbol}")
                logger.info(f"  Direction: {direction.value}")
                logger.info(f"  Entry Price: ₹{entry_price:.2f}")
                logger.info(f"  Quantity: {quantity}")
                logger.info(f"  Initial Stop: ₹{stop_loss_price:.2f}")
                logger.info(f"  Breakeven Target: ₹{breakeven_target:.2f}")
                logger.info(f"  Phase: {TradePhase.RISK.value}")
                
                return trade_state
                
        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
            return None
    
    def update_stop_loss_order(self, new_stop_price: float) -> bool:
        """Update the stop-loss order with new trigger price"""
        try:
            if not self.active_trade or not self.active_trade.setup.stop_loss_order_id:
                return False
            
            stop_loss_order_id = self.active_trade.setup.stop_loss_order_id
            symbol = self.active_trade.setup.candidate.symbol
            quantity = self.active_trade.setup.quantity
            direction = self.active_trade.setup.direction
            
            # Round to tick size
            new_stop_price = self.round_to_tick_size(new_stop_price, symbol)
            
            # Modify existing stop-loss order
            modify_result = self.kite_trader.modify_order(
                variety="regular",
                order_id=stop_loss_order_id,
                trigger_price=new_stop_price
            )
            
            if modify_result["status"] == "success":
                self.active_trade.current_stop_loss = new_stop_price
                logger.info(f"✅ Stop-loss updated to ₹{new_stop_price:.2f}")
                return True
            else:
                logger.error(f"Failed to update stop-loss: {modify_result}")
                return False
                
        except Exception as e:
            logger.error(f"Error updating stop-loss order: {str(e)}")
            return False
    
    def calculate_fibonacci_levels(self) -> Dict[str, float]:
        """Calculate Fibonacci levels from recent swing high/low"""
        try:
            if not self.active_trade:
                return {}
            
            symbol = self.active_trade.setup.candidate.symbol
            direction = self.active_trade.setup.direction
            
            # Get 1-minute data for last 50 candles
            ohlcv_data = self.get_1minute_ohlcv_data(symbol, 50)
            if not ohlcv_data:
                return {}
            
            # Calculate Fibonacci levels
            fib_levels = self.indicator_calculator.calculate_fibonacci_levels(
                ohlcv_data['high'], 
                ohlcv_data['low'], 
                lookback_periods=50,
                direction="uptrend" if direction == TradeDirection.LONG else "downtrend"
            )
            
            logger.info(f"Calculated Fibonacci levels for {symbol}:")
            for level, price in fib_levels.items():
                if level.startswith('level_'):
                    logger.info(f"  {level}: ₹{price:.2f}")
            
            return fib_levels
            
        except Exception as e:
            logger.error(f"Error calculating Fibonacci levels: {str(e)}")
            return {}
    
    def manage_trade_phases(self, current_price: float) -> bool:
        """
        Phase 4: Dynamic Trade Management - Handle phase transitions
        
        Args:
            current_price: Current market price
            
        Returns:
            True if trade is still active, False if closed
        """
        try:
            if not self.active_trade:
                return False
            
            trade = self.active_trade
            setup = trade.setup
            direction = setup.direction
            
            # Update current price and PnL
            trade.current_price = current_price
            
            if direction == TradeDirection.LONG:
                trade.unrealized_pnl = (current_price - setup.entry_price) * setup.quantity
            else:
                trade.unrealized_pnl = (setup.entry_price - current_price) * setup.quantity
            
            # Track maximum profit
            if trade.unrealized_pnl > trade.max_profit_reached:
                trade.max_profit_reached = trade.unrealized_pnl
            
            # Phase A: RISK - Wait for breakeven target
            if trade.phase == TradePhase.RISK:
                return self.manage_risk_phase(current_price)
            
            # Phase B: RISK_FREE - Calculate Fibonacci levels and wait for first target
            elif trade.phase == TradePhase.RISK_FREE:
                return self.manage_risk_free_phase(current_price)
            
            # Phase C: PROFIT_MAXIMIZING - Fibonacci ladder trailing
            elif trade.phase == TradePhase.PROFIT_MAXIMIZING:
                return self.manage_profit_maximizing_phase(current_price)
            
            return True
            
        except Exception as e:
            logger.error(f"Error managing trade phases: {str(e)}")
            return False
    
    def manage_risk_phase(self, current_price: float) -> bool:
        """Manage Phase A: Risk Phase - waiting for breakeven target"""
        try:
            trade = self.active_trade
            setup = trade.setup
            direction = setup.direction
            
            # Check if breakeven target is hit
            target_hit = False
            if direction == TradeDirection.LONG:
                target_hit = current_price >= setup.breakeven_target
            else:
                target_hit = current_price <= setup.breakeven_target
            
            if target_hit:
                logger.info("🎉 BREAKEVEN TARGET HIT - Moving to Risk-Free Phase")
                
                # Update stop-loss to breakeven target to cover costs
                if self.update_stop_loss_order(setup.breakeven_target):
                    trade.phase = TradePhase.RISK_FREE
                    trade.breakeven_hit = True
                    trade.current_stop_loss = setup.breakeven_target
                    trade.phase_start_time = datetime.now()
                    
                    # Calculate Fibonacci levels for next phase
                    trade.fibonacci_levels = self.calculate_fibonacci_levels()
                    
                    logger.info("✅ Phase transition: RISK → RISK_FREE")
                    logger.info(f"  Stop-loss moved to breakeven target: ₹{setup.breakeven_target:.2f}")
                    logger.info("  Risk eliminated - worst case is a zero-loss exit")
                else:
                    logger.error("Failed to update stop-loss to breakeven target")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in risk phase management: {str(e)}")
            return False
    
    def manage_risk_free_phase(self, current_price: float) -> bool:
        """Manage Phase B: Risk-Free Phase - waiting for first Fibonacci target"""
        try:
            trade = self.active_trade
            setup = trade.setup
            direction = setup.direction
            
            if not trade.fibonacci_levels:
                # Try to calculate again if not available
                trade.fibonacci_levels = self.calculate_fibonacci_levels()
                if not trade.fibonacci_levels:
                    return True  # Continue in this phase
            
            # Find the first Fibonacci target in the trade direction
            fib_targets = []
            for level_name, price in trade.fibonacci_levels.items():
                if level_name.startswith('level_') and level_name != 'level_0':
                    fib_targets.append((level_name, price))
            
            if not fib_targets:
                return True  # Continue waiting
            
            # Sort targets by price
            if direction == TradeDirection.LONG:
                fib_targets.sort(key=lambda x: x[1])  # Ascending for longs
                # Find first target above current price
                for level_name, price in fib_targets:
                    if price > setup.entry_price:
                        first_target = (level_name, price)
                        break
                else:
                    return True  # No suitable target found
            else:
                fib_targets.sort(key=lambda x: x[1], reverse=True)  # Descending for shorts
                # Find first target below current price
                for level_name, price in fib_targets:
                    if price < setup.entry_price:
                        first_target = (level_name, price)
                        break
                else:
                    return True  # No suitable target found
            
            target_name, target_price = first_target
            
            # Check if first Fibonacci target is hit
            target_hit = False
            if direction == TradeDirection.LONG:
                target_hit = current_price >= target_price
            else:
                target_hit = current_price <= target_price
            
            if target_hit:
                logger.info(f"🎯 FIRST FIBONACCI TARGET HIT: {target_name} at ₹{target_price:.2f}")
                
                # Move stop-loss to this Fibonacci level
                if self.update_stop_loss_order(target_price):
                    trade.phase = TradePhase.PROFIT_MAXIMIZING
                    trade.last_fibonacci_level = target_name
                    trade.profit_secured = trade.unrealized_pnl
                    trade.phase_start_time = datetime.now()
                    
                    logger.info("✅ Phase transition: RISK_FREE → PROFIT_MAXIMIZING")
                    logger.info(f"  Stop-loss moved to: ₹{target_price:.2f}")
                    logger.info(f"  Profit secured: ₹{trade.profit_secured:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in risk-free phase management: {str(e)}")
            return False
    
    def manage_profit_maximizing_phase(self, current_price: float) -> bool:
        """Manage Phase C: Profit-Maximizing Phase - Fibonacci ladder trailing"""
        try:
            trade = self.active_trade
            setup = trade.setup
            direction = setup.direction
            
            if not trade.fibonacci_levels:
                return True  # Continue with existing stop
            
            # Get sorted Fibonacci levels
            fib_targets = []
            for level_name, price in trade.fibonacci_levels.items():
                if level_name.startswith('level_'):
                    fib_targets.append((level_name, price))
            
            if direction == TradeDirection.LONG:
                fib_targets.sort(key=lambda x: x[1])  # Ascending for longs
                # Look for next higher level to trail to
                for level_name, price in fib_targets:
                    if (price > trade.current_stop_loss and 
                        current_price >= price):
                        # Price has reached this level - trail stop to it
                        logger.info(f"🪜 Fibonacci ladder step: {level_name} at ₹{price:.2f}")
                        
                        if self.update_stop_loss_order(price):
                            trade.last_fibonacci_level = level_name
                            additional_profit = (price - trade.current_stop_loss) * setup.quantity
                            trade.profit_secured += additional_profit
                            
                            logger.info(f"✅ Stop trailed to: ₹{price:.2f}")
                            logger.info(f"   Additional profit secured: ₹{additional_profit:.2f}")
                            logger.info(f"   Total profit secured: ₹{trade.profit_secured:.2f}")
                        break
            
            else:  # SHORT
                fib_targets.sort(key=lambda x: x[1], reverse=True)  # Descending for shorts
                # Look for next lower level to trail to
                for level_name, price in fib_targets:
                    if (price < trade.current_stop_loss and 
                        current_price <= price):
                        # Price has reached this level - trail stop to it
                        logger.info(f"🪜 Fibonacci ladder step: {level_name} at ₹{price:.2f}")
                        
                        if self.update_stop_loss_order(price):
                            trade.last_fibonacci_level = level_name
                            additional_profit = (trade.current_stop_loss - price) * setup.quantity
                            trade.profit_secured += additional_profit
                            
                            logger.info(f"✅ Stop trailed to: ₹{price:.2f}")
                            logger.info(f"   Additional profit secured: ₹{additional_profit:.2f}")
                            logger.info(f"   Total profit secured: ₹{trade.profit_secured:.2f}")
                        break
            
            return True
            
        except Exception as e:
            logger.error(f"Error in profit maximizing phase: {str(e)}")
            return False
    
    def close_trade(self, reason: str = "Manual close") -> bool:
        """Close active trade and reset strategy state"""
        try:
            with self.position_lock:
                if not self.active_trade:
                    logger.warning("No active trade to close")
                    return False
                
                trade = self.active_trade
                setup = trade.setup
                
                # Calculate final PnL
                if setup.direction == TradeDirection.LONG:
                    final_pnl = (trade.current_price - setup.entry_price) * setup.quantity
                else:
                    final_pnl = (setup.entry_price - trade.current_price) * setup.quantity
                
                # Update statistics
                self.total_trades += 1
                if final_pnl > 0:
                    self.winning_trades += 1
                self.total_profit += final_pnl
                
                # Log trade summary
                logger.info("📊 TRADE CLOSED")
                logger.info(f"  Symbol: {setup.candidate.symbol}")
                logger.info(f"  Direction: {setup.direction.value}")
                logger.info(f"  Entry Price: ₹{setup.entry_price:.2f}")
                logger.info(f"  Exit Price: ₹{trade.current_price:.2f}")
                logger.info(f"  Quantity: {setup.quantity}")
                logger.info(f"  Final PnL: ₹{final_pnl:.2f}")
                logger.info(f"  Phase Reached: {trade.phase.value}")
                logger.info(f"  Reason: {reason}")
                logger.info(f"  Duration: {datetime.now() - setup.entry_time}")
                
                # Clear active trade
                self.active_trade = None
                
                return True
                
        except Exception as e:
            logger.error(f"Error closing trade: {str(e)}")
            return False
    
    def scan_and_execute(self, live_ratings: Dict[str, Dict]) -> bool:
        """
        Main strategy loop: Phase 1-3 execution
        Scan market, confirm entry, execute trade
        
        Args:
            live_ratings: Current live ratings from rating system
            
        Returns:
            True if trade was executed, False otherwise
        """
        try:
            # Phase 1: Market Scanner
            candidate = self.select_trading_candidate(live_ratings)
            if not candidate:
                return False
            
            # Phase 2: Precision Entry Confirmation
            confirmation = self.check_entry_confirmation(candidate)
            if not confirmation:
                logger.info(f"No entry confirmation for {candidate.symbol} - continuing scan")
                return False
            
            direction, supertrend_value, psar_value = confirmation
            
            # Phase 3: Trade Execution
            trade_state = self.execute_trade(candidate, direction, supertrend_value, psar_value)
            if trade_state:
                logger.info(f"🚀 Trade executed successfully for {candidate.symbol}")
                return True
            else:
                logger.error(f"Failed to execute trade for {candidate.symbol}")
                return False
                
        except Exception as e:
            logger.error(f"Error in scan_and_execute: {str(e)}")
            return False
    
    def update_active_trade(self, tick_data: Dict[str, Any]) -> bool:
        """
        Update active trade with new tick data
        Called from WebSocket tick handler
        
        Args:
            tick_data: Tick data from WebSocket
            
        Returns:
            True if trade is still active, False if closed
        """
        try:
            if not self.active_trade:
                return False
            
            # Extract current price from tick data
            current_price = float(tick_data.get('last_price', 0))
            if current_price == 0:
                return True  # Invalid tick data, continue
            
            # Update trade management
            return self.manage_trade_phases(current_price)
            
        except Exception as e:
            logger.error(f"Error updating active trade: {str(e)}")
            return False
    
    def get_strategy_status(self) -> Dict[str, Any]:
        """Get current strategy status for monitoring"""
        try:
            status = {
                'strategy': 'Precision Single Trade Strategy',
                'active_trade': None,
                'can_trade': self.can_trade(),
                'performance': {
                    'total_trades': self.total_trades,
                    'winning_trades': self.winning_trades,
                    'win_rate': (self.winning_trades / max(self.total_trades, 1)) * 100,
                    'total_profit': self.total_profit
                },
                'timestamp': datetime.now().isoformat()
            }
            
            if self.active_trade:
                trade = self.active_trade
                setup = trade.setup
                
                status['active_trade'] = {
                    'symbol': setup.candidate.symbol,
                    'direction': setup.direction.value,
                    'entry_price': setup.entry_price,
                    'current_price': trade.current_price,
                    'quantity': setup.quantity,
                    'unrealized_pnl': trade.unrealized_pnl,
                    'phase': trade.phase.value,
                    'current_stop_loss': trade.current_stop_loss,
                    'breakeven_target': setup.breakeven_target,
                    'entry_time': setup.entry_time.isoformat(),
                    'duration_minutes': (datetime.now() - setup.entry_time).total_seconds() / 60,
                    'fibonacci_levels': trade.fibonacci_levels,
                    'last_fibonacci_level': trade.last_fibonacci_level,
                    'profit_secured': trade.profit_secured,
                    'max_profit_reached': trade.max_profit_reached
                }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting strategy status: {str(e)}")
            return {'error': str(e)}
    
    def should_close_market_position(self) -> bool:
        """Check if position should be closed due to market close"""
        try:
            current_time = datetime.now()
            current_hour = current_time.hour
            current_minute = current_time.minute
            
            # NSE closes at 15:30 - start closing positions at 15:25
            if current_hour == 15 and current_minute >= 25:
                if self.active_trade:
                    logger.info("Market closing soon - closing active position")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking market close: {str(e)}")
            return False


if __name__ == "__main__":
    print("Precision Single Trade Strategy module loaded successfully")
    print("This module requires integration with the main trading bot")
    print('''
    from precision_single_trade_strategy import PrecisionSingleTradeStrategy
    from kiteConnect import KiteTrader, Config
    
    # Initialize trader and strategy
    trader = KiteTrader(api_key=Config.API_KEY, access_token=access_token)
    strategy = PrecisionSingleTradeStrategy(trader, instruments, instruments_data)
    
    # Main trading loop
    while market_open:
        live_ratings = get_live_ratings()
        
        if strategy.can_trade():
            # Scan and execute new trade
            strategy.scan_and_execute(live_ratings)
        else:
            # Update active trade with current data
            current_tick = get_current_tick_data()
            strategy.update_active_trade(current_tick)
        
        time.sleep(1)  # 1-second update cycle
    ''')
