#!/usr/bin/env python3
"""
Runner script for the Automated Single Trade Bot

This script starts the sophisticated precision trading strategy with:
- Market Scanner with holistic rating system
- Prime candidate selection (highest absolute rating)
- Dual confirmation (Supertrend + PSAR on 1-minute)
- Full margin allocation to single high-probability trade
- Three-phase risk management with Fibonacci trailing stops
"""

import os
import sys
import json
import logging
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from automated_single_trade_bot import AutomatedSingleTradeBot, TradingConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'single_trade_bot_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def load_access_token():
    """Load access token from file"""
    try:
        with open('access_token.json', 'r') as f:
            token_data = json.load(f)
        
        access_token = token_data.get('access_token')
        if not access_token:
            raise ValueError("No access token found in file")
        
        # Check if token is for today
        token_date = token_data.get('date', '')
        today = datetime.now().strftime('%Y-%m-%d')
        
        if token_date != today:
            logger.warning(f"Access token is from {token_date}, but today is {today}")
            logger.warning("Token might be expired - consider regenerating")
        
        return access_token
        
    except FileNotFoundError:
        logger.error("access_token.json file not found")
        logger.error("Please run the login process first to generate access token")
        return None
    except Exception as e:
        logger.error(f"Error loading access token: {str(e)}")
        return None


def create_trading_config():
    """Create trading configuration"""
    config = TradingConfig(
        # Scanner settings
        scan_interval_seconds=30,           # Scan every 30 seconds
        rating_threshold=2.0,               # Minimum absolute rating to consider
        
        # Confirmation settings  
        confirmation_timeout_seconds=300,   # 5 minutes max for confirmation
        supertrend_period=5,                # Supertrend ATR period
        supertrend_factor=3.0,              # Supertrend factor
        psar_acceleration=0.02,             # PSAR acceleration
        psar_maximum=0.2,                   # PSAR maximum acceleration
        
        # Risk management
        max_daily_trades=3,                 # Maximum 3 trades per day
        max_daily_loss_percent=2.0,         # Stop trading if daily loss > 2%
        
        # Position sizing
        use_full_margin=True,               # Use full available margin
        fixed_quantity=None,                # Not using fixed quantity
        
        # Fibonacci settings
        fibonacci_lookback_candles=50,      # Look back 50 candles for swing points
        fibonacci_levels=[0.236, 0.382, 0.5, 0.618, 0.786],  # Retracement levels
        fibonacci_extensions=[1.272, 1.618, 2.0, 2.618]       # Extension levels
    )
    
    return config


def main():
    """Main function"""
    print("=" * 80)
    print("🚀 AUTOMATED SINGLE TRADE BOT - PRECISION TRADING STRATEGY")
    print("=" * 80)
    print()
    
    # Display strategy overview
    print("📋 STRATEGY OVERVIEW:")
    print("1. 🔍 Market Scanner - Holistic rating of all watchlist stocks")
    print("2. 🎯 Prime Candidate - Select highest absolute rating (direction-agnostic)")
    print("3. ✅ Dual Confirmation - Supertrend + PSAR alignment on 1-minute chart")
    print("4. 💰 Full Margin - Allocate entire margin to single high-probability trade")
    print("5. 📊 Three-Phase Management:")
    print("   • Phase A: Initial risk with Supertrend stop-loss")
    print("   • Phase B: Risk-free zone after hitting breakeven")
    print("   • Phase C: Fibonacci ladder trailing stops for profit maximization")
    print()
    
    # Load access token
    print("🔑 Loading access token...")
    access_token = load_access_token()
    if not access_token:
        print("❌ Failed to load access token. Exiting.")
        return
    
    print("✅ Access token loaded successfully")
    print()
    
    # Create configuration
    print("⚙️ Creating trading configuration...")
    config = create_trading_config()
    
    print("✅ Configuration created:")
    print(f"   • Scan interval: {config.scan_interval_seconds} seconds")
    print(f"   • Rating threshold: {config.rating_threshold}")
    print(f"   • Max daily trades: {config.max_daily_trades}")
    print(f"   • Max daily loss: {config.max_daily_loss_percent}%")
    print(f"   • Supertrend: Period={config.supertrend_period}, Factor={config.supertrend_factor}")
    print(f"   • PSAR: Acceleration={config.psar_acceleration}, Max={config.psar_maximum}")
    print(f"   • Fibonacci lookback: {config.fibonacci_lookback_candles} candles")
    print()
    
    # Risk disclaimer
    print("⚠️ RISK DISCLAIMER:")
    print("This is a fully automated trading system that will:")
    print("• Place real trades with real money")
    print("• Use your entire available margin for each trade")
    print("• Make independent trading decisions without manual intervention")
    print("• Trading involves substantial risk of loss")
    print()
    
    # Get user confirmation
    while True:
        response = input("Do you want to start the automated trading bot? (yes/no): ").strip().lower()
        if response in ['yes', 'y']:
            break
        elif response in ['no', 'n']:
            print("Exiting without starting the bot.")
            return
        else:
            print("Please enter 'yes' or 'no'")
    
    print()
    print("🚀 Starting Automated Single Trade Bot...")
    print("=" * 80)
    
    try:
        # Create and start bot
        bot = AutomatedSingleTradeBot(access_token, config)
        
        if bot.start():
            print("✅ Bot started successfully!")
            print()
            print("📊 LIVE STATUS MONITORING:")
            print("Press Ctrl+C to stop the bot")
            print("-" * 60)
            
            try:
                # Monitor bot status
                while bot.running:
                    import time
                    time.sleep(60)  # Update every minute
                    
                    status = bot.get_status()
                    current_time = datetime.now().strftime("%H:%M:%S")
                    
                    print(f"[{current_time}] State: {status['current_state']} | "
                          f"Daily Trades: {status['daily_trades']}/{status['max_daily_trades']} | "
                          f"Daily P&L: ₹{status['daily_pnl']:.2f}")
                    
                    # Show active trade details if any
                    if status['active_trade']:
                        trade = status['active_trade']
                        print(f"   🎯 Active Trade: {trade['symbol']} {trade['direction']} "
                              f"{trade['quantity']}@₹{trade['entry_price']:.2f} | "
                              f"Phase: {trade['current_phase']}")
                    
                    # Show current candidate if any
                    if status['current_candidate']:
                        candidate = status['current_candidate']
                        print(f"   🔍 Candidate: {candidate['symbol']} "
                              f"(Rating: {candidate['rating']:.2f}, Direction: {candidate['direction']})")
                
            except KeyboardInterrupt:
                print("\n🛑 Received stop signal...")
                print("Stopping bot gracefully...")
                bot.stop()
                
        else:
            print("❌ Failed to start bot")
            
    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
        print(f"❌ Critical error occurred: {str(e)}")
    
    print()
    print("=" * 80)
    print("🏁 Automated Single Trade Bot session ended")
    print("=" * 80)


if __name__ == "__main__":
    main()
