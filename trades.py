"""
Live Trading Management System

This module handles:
- Automatic trade placement based on rating scores
- Stop loss and target management
- Order monitoring and execution
- Position tracking and risk management
- Trade logging and performance tracking
"""

import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
import threading
from dataclasses import dataclass, asdict
from kiteConnect import KiteTrader, Config


class TradeStatus(Enum):
    """Trade status enumeration"""
    PENDING = "pending"
    EXECUTED = "executed"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    TARGET_HIT = "target_hit"
    STOPLOSS_HIT = "stoploss_hit"
    ERROR = "error"


class OrderType(Enum):
    """Order type enumeration"""
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class TradeConfig:
    """Configuration for trading parameters"""
    # Risk management
    max_position_size: float = 10000.0  # Maximum position size in rupees
    max_positions: int = 5  # Maximum number of open positions
    min_rating_threshold: float = 2.0  # Minimum absolute rating score to trade
    
    # Stop loss and target settings
    default_stoploss_pct: float = 2.0  # Default stop loss percentage
    default_target_pct: float = 4.0   # Default target percentage
    trailing_stoploss: bool = True     # Enable trailing stop loss
    
    # Order settings
    order_timeout: int = 300  # Order timeout in seconds
    retry_attempts: int = 3   # Number of retry attempts for orders
    
    # Position monitoring
    monitor_interval: int = 30  # Position monitoring interval in seconds
    market_hours_start: str = "09:15"  # Market start time
    market_hours_end: str = "15:30"    # Market end time


@dataclass
class Trade:
    """Trade data structure"""
    symbol: str
    trading_symbol: str
    instrument_token: str
    trade_type: OrderType
    rating_score: float
    entry_price: float
    quantity: int
    target_price: float
    stoploss_price: float
    
    # Order tracking
    entry_order_id: Optional[str] = None
    target_order_id: Optional[str] = None
    stoploss_order_id: Optional[str] = None
    
    # Status tracking
    status: TradeStatus = TradeStatus.PENDING
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    
    # P&L tracking
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    
    # Risk management
    current_price: float = 0.0
    trailing_stoploss_price: Optional[float] = None
    
    # Metadata
    trade_id: str = ""
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.trade_id == "":
            self.trade_id = f"{self.symbol}_{self.timestamp.strftime('%Y%m%d_%H%M%S')}"


class TradeManager:
    """Main trading management class"""
    
    def __init__(self, kite_trader: KiteTrader, config: TradeConfig = None):
        self.kite_trader = kite_trader
        self.config = config or TradeConfig()
        
        # Trade tracking
        self.active_trades: Dict[str, Trade] = {}
        self.completed_trades: List[Trade] = []
        
        # Threading for monitoring
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Data persistence
        self.trades_file = "trades_data.json"
        self.load_trades_data()
        
        print("TradeManager initialized successfully")
    
    def analyze_ratings_for_trading(self, ratings_data: List[Dict]) -> Optional[Dict]:
        """
        Analyze rating data to find the best trading opportunity
        
        Args:
            ratings_data: List of rating dictionaries
            
        Returns:
            Dict with trade recommendation or None
        """
        try:
            # Filter valid ratings (exclude errors)
            valid_ratings = [r for r in ratings_data if 'error' not in r]
            
            if not valid_ratings:
                print("No valid ratings found for trading analysis")
                return None
            
            # Find highest absolute rating score
            best_trade = None
            max_abs_score = 0
            
            for rating in valid_ratings:
                abs_score = abs(rating.get('final_rating', 0))
                
                # Check if score meets minimum threshold
                if abs_score >= self.config.min_rating_threshold and abs_score > max_abs_score:
                    max_abs_score = abs_score
                    best_trade = rating
            
            if best_trade:
                trade_type = OrderType.BUY if best_trade['final_rating'] > 0 else OrderType.SELL
                
                print(f"Best trading opportunity found:")
                print(f"  Symbol: {best_trade.get('trading_symbol', 'N/A')}")
                print(f"  Score: {best_trade['final_rating']:.2f}")
                print(f"  Action: {trade_type.value}")
                
                return {
                    'rating_data': best_trade,
                    'trade_type': trade_type,
                    'score': best_trade['final_rating']
                }
            else:
                print(f"No ratings above threshold ({self.config.min_rating_threshold}) found")
                return None
                
        except Exception as e:
            print(f"Error analyzing ratings: {str(e)}")
            return None
    
    def calculate_position_size(self, price: float, rating_score: float) -> int:
        """
        Calculate position size based on price and rating confidence
        
        Args:
            price: Current stock price
            rating_score: Rating score for confidence
            
        Returns:
            Number of shares to trade
        """
        try:
            # Base position size
            base_amount = self.config.max_position_size
            
            # Adjust based on rating confidence (higher absolute score = larger position)
            confidence_multiplier = min(abs(rating_score) / 10.0, 1.0)  # Cap at 1.0
            
            # Calculate position amount
            position_amount = base_amount * confidence_multiplier
            
            # Calculate quantity (minimum 1 share)
            quantity = max(1, int(position_amount / price))
            
            print(f"Position calculation: Price={price:.2f}, Confidence={confidence_multiplier:.2f}, Qty={quantity}")
            
            return quantity
            
        except Exception as e:
            print(f"Error calculating position size: {str(e)}")
            return 1
    
    def calculate_target_stoploss(self, entry_price: float, trade_type: OrderType, rating_score: float) -> Tuple[float, float]:
        """
        Calculate target and stop loss prices
        
        Args:
            entry_price: Entry price
            trade_type: BUY or SELL
            rating_score: Rating score for adjustment
            
        Returns:
            Tuple of (target_price, stoploss_price)
        """
        try:
            # Adjust percentages based on rating strength
            score_multiplier = min(abs(rating_score) / 5.0, 2.0)  # Cap at 2x
            
            target_pct = self.config.default_target_pct * score_multiplier
            stoploss_pct = self.config.default_stoploss_pct
            
            if trade_type == OrderType.BUY:
                target_price = entry_price * (1 + target_pct / 100)
                stoploss_price = entry_price * (1 - stoploss_pct / 100)
            else:  # SELL
                target_price = entry_price * (1 - target_pct / 100)
                stoploss_price = entry_price * (1 + stoploss_pct / 100)
            
            print(f"Target/SL calculation: Entry={entry_price:.2f}, Target={target_price:.2f}, SL={stoploss_price:.2f}")
            
            return round(target_price, 2), round(stoploss_price, 2)
            
        except Exception as e:
            print(f"Error calculating target/stoploss: {str(e)}")
            return entry_price, entry_price
    
    def get_current_price(self, instrument_token: str, trading_symbol: str) -> Optional[float]:
        """
        Get current market price for an instrument
        
        Args:
            instrument_token: Instrument token
            trading_symbol: Trading symbol
            
        Returns:
            Current price or None if error
        """
        try:
            # Get LTP data
            instruments = [f"NSE:{trading_symbol}"]
            ltp_data = self.kite_trader.get_ltp(instruments)
            
            if instruments[0] in ltp_data:
                price = ltp_data[instruments[0]].get('last_price', 0)
                return float(price) if price else None
            else:
                print(f"No LTP data found for {trading_symbol}")
                return None
                
        except Exception as e:
            print(f"Error getting current price for {trading_symbol}: {str(e)}")
            return None
    
    def place_trade(self, trade_recommendation: Dict) -> Optional[Trade]:
        """
        Place a new trade based on recommendation
        
        Args:
            trade_recommendation: Trade recommendation from analysis
            
        Returns:
            Trade object or None if failed
        """
        try:
            rating_data = trade_recommendation['rating_data']
            trade_type = trade_recommendation['trade_type']
            
            # Check position limits
            if len(self.active_trades) >= self.config.max_positions:
                print(f"Maximum positions ({self.config.max_positions}) reached. Cannot place new trade.")
                return None
            
            # Extract instrument details - handle both field name formats
            trading_symbol = rating_data.get('trading_symbol') or rating_data.get('tradingsymbol')
            instrument_token = str(rating_data.get('instrument_token', ''))
            
            if not trading_symbol or not instrument_token:
                print("Missing instrument details for trade")
                return None
            
            # Get current price
            current_price = self.get_current_price(instrument_token, trading_symbol)
            if not current_price:
                print(f"Could not get current price for {trading_symbol}")
                return None
            
            # Calculate position size
            quantity = self.calculate_position_size(current_price, trade_recommendation['score'])
            
            # Calculate target and stop loss
            target_price, stoploss_price = self.calculate_target_stoploss(
                current_price, trade_type, trade_recommendation['score']
            )
            
            # Create trade object
            trade = Trade(
                symbol=rating_data.get('company_name') or rating_data.get('name', trading_symbol),
                trading_symbol=trading_symbol,
                instrument_token=instrument_token,
                trade_type=trade_type,
                rating_score=trade_recommendation['score'],
                entry_price=current_price,
                quantity=quantity,
                target_price=target_price,
                stoploss_price=stoploss_price,
                current_price=current_price
            )
            
            # Place entry order
            entry_response = self._place_entry_order(trade)
            
            if entry_response and entry_response.get('status') == 'success':
                trade.entry_order_id = entry_response.get('order_id')
                trade.status = TradeStatus.PENDING
                
                # Add to active trades
                self.active_trades[trade.trade_id] = trade
                
                # Save trades data
                self.save_trades_data()
                
                print(f"Trade placed successfully: {trade.trade_id}")
                return trade
            else:
                print(f"Failed to place entry order: {entry_response}")
                return None
                
        except Exception as e:
            print(f"Error placing trade: {str(e)}")
            return None
    
    def _place_entry_order(self, trade: Trade) -> Dict:
        """
        Place entry order for a trade (regular limit order only)

        Args:
            trade: Trade object
            
        Returns:
            Order response
        """
        try:
            # Place regular limit order
            response = self.kite_trader.place_limit_order(
                exchange="NSE",
                tradingsymbol=trade.trading_symbol,
                transaction_type=trade.trade_type.value,
                quantity=trade.quantity,
                price=trade.entry_price
            )
            
            if response.get('status') == 'success':
                print(f"Manual stop-loss and target management required:")
                print(f"  Entry: ₹{trade.entry_price:.2f}")
                print(f"  Stop Loss: ₹{trade.stoploss_price:.2f}")
                print(f"  Target: ₹{trade.target_price:.2f}")
            
            return response
            
        except Exception as e:
            print(f"Error placing entry order: {str(e)}")
            return {"status": "failed", "error": str(e)}
    
    def update_trailing_stoploss(self, trade: Trade) -> bool:
        """
        Update trailing stop loss for a trade
        
        Args:
            trade: Trade object
            
        Returns:
            True if updated successfully
        """
        try:
            if not self.config.trailing_stoploss or trade.status != TradeStatus.EXECUTED:
                return False
            
            current_price = self.get_current_price(trade.instrument_token, trade.trading_symbol)
            if not current_price:
                return False
            
            trade.current_price = current_price
            
            # Calculate new trailing stop loss
            if trade.trade_type == OrderType.BUY:
                # For BUY trades, move stop loss up
                new_stoploss = current_price * (1 - self.config.default_stoploss_pct / 100)
                if trade.trailing_stoploss_price is None or new_stoploss > trade.trailing_stoploss_price:
                    trade.trailing_stoploss_price = new_stoploss
                    print(f"Updated trailing stop loss for {trade.trading_symbol}: {new_stoploss:.2f}")
                    return True
            else:  # SELL
                # For SELL trades, move stop loss down
                new_stoploss = current_price * (1 + self.config.default_stoploss_pct / 100)
                if trade.trailing_stoploss_price is None or new_stoploss < trade.trailing_stoploss_price:
                    trade.trailing_stoploss_price = new_stoploss
                    print(f"Updated trailing stop loss for {trade.trading_symbol}: {new_stoploss:.2f}")
                    return True
            
            return False
            
        except Exception as e:
            print(f"Error updating trailing stop loss: {str(e)}")
            return False
    
    def monitor_positions(self):
        """Monitor active positions and update status"""
        while self.monitoring_active:
            try:
                if not self.active_trades:
                    time.sleep(self.config.monitor_interval)
                    continue
                
                print(f"Monitoring {len(self.active_trades)} active trades...")
                
                # Get current positions
                positions = self.kite_trader.get_positions()
                
                for trade_id, trade in list(self.active_trades.items()):
                    try:
                        # Update current price
                        current_price = self.get_current_price(trade.instrument_token, trade.trading_symbol)
                        if current_price:
                            trade.current_price = current_price
                            
                            # Calculate unrealized P&L
                            if trade.trade_type == OrderType.BUY:
                                trade.unrealized_pnl = (current_price - trade.entry_price) * trade.quantity
                            else:
                                trade.unrealized_pnl = (trade.entry_price - current_price) * trade.quantity
                        
                        # Update trailing stop loss
                        self.update_trailing_stoploss(trade)
                        
                        # Check if trade is completed (this would need order status checking)
                        # For now, we'll implement basic logic
                        
                    except Exception as e:
                        print(f"Error monitoring trade {trade_id}: {str(e)}")
                
                # Save updated data
                self.save_trades_data()
                
                time.sleep(self.config.monitor_interval)
                
            except Exception as e:
                print(f"Error in position monitoring: {str(e)}")
                time.sleep(self.config.monitor_interval)
    
    def start_monitoring(self):
        """Start position monitoring in background thread"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(target=self.monitor_positions, daemon=True)
            self.monitor_thread.start()
            print("Position monitoring started")
    
    def stop_monitoring(self):
        """Stop position monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        print("Position monitoring stopped")
    
    def close_trade(self, trade_id: str, reason: str = "manual") -> bool:
        """
        Close a specific trade
        
        Args:
            trade_id: Trade ID to close
            reason: Reason for closing
            
        Returns:
            True if closed successfully
        """
        try:
            if trade_id not in self.active_trades:
                print(f"Trade {trade_id} not found in active trades")
                return False
            
            trade = self.active_trades[trade_id]
            
            # Get current price for exit
            current_price = self.get_current_price(trade.instrument_token, trade.trading_symbol)
            if not current_price:
                print(f"Could not get current price for closing {trade.trading_symbol}")
                return False
            
            # Place exit order
            exit_response = self._place_exit_order(trade, current_price)
            
            if exit_response and exit_response.get('status') == 'success':
                # Update trade details
                trade.exit_price = current_price
                trade.exit_time = datetime.now()
                trade.status = TradeStatus.TARGET_HIT if reason == "target" else TradeStatus.CANCELLED
                
                # Calculate realized P&L
                if trade.trade_type == OrderType.BUY:
                    trade.realized_pnl = (current_price - trade.entry_price) * trade.quantity
                else:
                    trade.realized_pnl = (trade.entry_price - current_price) * trade.quantity
                
                # Move to completed trades
                self.completed_trades.append(trade)
                del self.active_trades[trade_id]
                
                # Save data
                self.save_trades_data()
                
                print(f"Trade {trade_id} closed successfully. P&L: ₹{trade.realized_pnl:.2f}")
                return True
            else:
                print(f"Failed to close trade {trade_id}: {exit_response}")
                return False
                
        except Exception as e:
            print(f"Error closing trade {trade_id}: {str(e)}")
            return False
    
    def _place_exit_order(self, trade: Trade, exit_price: float) -> Dict:
        """
        Place exit order for a trade (regular limit order only)
        
        Args:
            trade: Trade object
            exit_price: Exit price
            
        Returns:
            Order response
        """
        try:
            # Place opposite limit order to close position
            transaction_type = "SELL" if trade.trade_type == OrderType.BUY else "BUY"
            
            response = self.kite_trader.place_limit_order(
                exchange="NSE",
                tradingsymbol=trade.trading_symbol,
                transaction_type=transaction_type,
                quantity=trade.quantity,
                price=exit_price
            )
            
            return response
            
        except Exception as e:
            print(f"Error placing exit order: {str(e)}")
            return {"status": "failed", "error": str(e)}
    
    def close_all_trades(self, reason: str = "eod") -> int:
        """
        Close all active trades
        
        Args:
            reason: Reason for closing all trades
            
        Returns:
            Number of trades closed
        """
        closed_count = 0
        trade_ids = list(self.active_trades.keys())
        
        for trade_id in trade_ids:
            if self.close_trade(trade_id, reason):
                closed_count += 1
        
        print(f"Closed {closed_count} out of {len(trade_ids)} trades")
        return closed_count
    
    def get_trade_summary(self) -> Dict:
        """
        Get summary of all trades
        
        Returns:
            Dictionary with trade statistics
        """
        active_count = len(self.active_trades)
        completed_count = len(self.completed_trades)
        
        # Calculate P&L
        total_realized_pnl = sum(trade.realized_pnl for trade in self.completed_trades)
        total_unrealized_pnl = sum(trade.unrealized_pnl for trade in self.active_trades.values())
        
        # Win rate calculation
        profitable_trades = sum(1 for trade in self.completed_trades if trade.realized_pnl > 0)
        win_rate = (profitable_trades / completed_count * 100) if completed_count > 0 else 0
        
        return {
            'active_trades': active_count,
            'completed_trades': completed_count,
            'total_trades': active_count + completed_count,
            'realized_pnl': round(total_realized_pnl, 2),
            'unrealized_pnl': round(total_unrealized_pnl, 2),
            'total_pnl': round(total_realized_pnl + total_unrealized_pnl, 2),
            'win_rate': round(win_rate, 2),
            'profitable_trades': profitable_trades
        }
    
    def save_trades_data(self):
        """Save trades data to file"""
        try:
            data = {
                'active_trades': {tid: asdict(trade) for tid, trade in self.active_trades.items()},
                'completed_trades': [asdict(trade) for trade in self.completed_trades],
                'config': asdict(self.config),
                'last_updated': datetime.now().isoformat()
            }
            
            # Convert datetime objects to strings for JSON serialization
            def convert_datetime(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                return obj
            
            with open(self.trades_file, 'w') as f:
                json.dump(data, f, indent=2, default=convert_datetime)
                
        except Exception as e:
            print(f"Error saving trades data: {str(e)}")
    
    def load_trades_data(self):
        """Load trades data from file"""
        try:
            if os.path.exists(self.trades_file):
                with open(self.trades_file, 'r') as f:
                    data = json.load(f)
                
                # Load active trades
                for tid, trade_data in data.get('active_trades', {}).items():
                    # Convert datetime strings back to datetime objects
                    if trade_data.get('timestamp'):
                        trade_data['timestamp'] = datetime.fromisoformat(trade_data['timestamp'])
                    if trade_data.get('entry_time'):
                        trade_data['entry_time'] = datetime.fromisoformat(trade_data['entry_time'])
                    
                    # Convert status and trade_type back to enums
                    trade_data['status'] = TradeStatus(trade_data['status'])
                    trade_data['trade_type'] = OrderType(trade_data['trade_type'])
                    
                    self.active_trades[tid] = Trade(**trade_data)
                
                # Load completed trades
                for trade_data in data.get('completed_trades', []):
                    # Convert datetime strings back to datetime objects
                    if trade_data.get('timestamp'):
                        trade_data['timestamp'] = datetime.fromisoformat(trade_data['timestamp'])
                    if trade_data.get('entry_time'):
                        trade_data['entry_time'] = datetime.fromisoformat(trade_data['entry_time'])
                    if trade_data.get('exit_time'):
                        trade_data['exit_time'] = datetime.fromisoformat(trade_data['exit_time'])
                    
                    # Convert status and trade_type back to enums
                    trade_data['status'] = TradeStatus(trade_data['status'])
                    trade_data['trade_type'] = OrderType(trade_data['trade_type'])
                    
                    self.completed_trades.append(Trade(**trade_data))
                
                print(f"Loaded {len(self.active_trades)} active trades and {len(self.completed_trades)} completed trades")
            else:
                print("No existing trades data found. Starting fresh.")
                
        except Exception as e:
            print(f"Error loading trades data: {str(e)}")
    
    def print_active_trades(self):
        """Print summary of active trades"""
        if not self.active_trades:
            print("No active trades")
            return
        
        print("\n" + "="*80)
        print("ACTIVE TRADES")
        print("="*80)
        print(f"{'Symbol':<12} {'Type':<4} {'Entry':<8} {'Current':<8} {'Target':<8} {'SL':<8} {'P&L':<10} {'Status'}")
        print("-"*80)
        
        for trade in self.active_trades.values():
            pnl_str = f"₹{trade.unrealized_pnl:.0f}"
            print(f"{trade.trading_symbol:<12} {trade.trade_type.value:<4} {trade.entry_price:<8.1f} "
                  f"{trade.current_price:<8.1f} {trade.target_price:<8.1f} {trade.stoploss_price:<8.1f} "
                  f"{pnl_str:<10} {trade.status.value}")
        
        # Print summary
        summary = self.get_trade_summary()
        print("-"*80)
        print(f"Total P&L: ₹{summary['total_pnl']} | Unrealized: ₹{summary['unrealized_pnl']}")
        print("="*80)


def main():
    """Main function for testing trade manager"""
    print("Trade Manager Module")
    print("This module is designed to be imported and used by bot.py")
    print("Example usage:")
    print("""
    from trades import TradeManager, TradeConfig
    from kiteConnect import KiteTrader, Config
    
    # Initialize trader
    trader = KiteTrader(api_key=Config.API_KEY, access_token=access_token)
    
    # Initialize trade manager
    trade_manager = TradeManager(trader)
    
    # Start monitoring
    trade_manager.start_monitoring()
    
    # Analyze ratings and place trades
    ratings_data = [...] # From rating system
    recommendation = trade_manager.analyze_ratings_for_trading(ratings_data)
    if recommendation:
        trade = trade_manager.place_trade(recommendation)
    """)


if __name__ == "__main__":
    main()
