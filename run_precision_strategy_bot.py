#!/usr/bin/env python3
"""
Run Automated Trading Bot with Precision Single Trade Strategy

This script runs the automated trading bot with the precision single trade strategy enabled.
The strategy will:
1. Scan all watchlist stocks and select the highest absolute-rated candidate
2. Confirm entry using Supertrend + PSAR dual confirmation on 1-minute chart
3. Execute trade with full margin allocation and Supertrend stop-loss
4. Manage trade through 3 phases: Risk → Risk-Free → Profit-Maximizing
5. Trail stops using Fibonacci levels for maximum profit extraction

IMPORTANT: This will execute REAL trades with real money!
Only run this if you understand the risks and have tested thoroughly.
"""

import sys
import os
import logging
from datetime import datetime

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from automated_trading_bot import AutomatedTradingBot, BotConfig

def setup_logging():
    """Setup logging configuration"""
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Configure logging
    log_filename = f"logs/precision_strategy_bot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def main():
    """Main function to run precision strategy bot"""
    logger = setup_logging()
    
    try:
        print("="*80)
        print("🎯 PRECISION SINGLE TRADE STRATEGY BOT")
        print("="*80)
        print()
        print("⚠️  WARNING: This bot will execute REAL trades with REAL money!")
        print("   - Only ONE position will be active at a time")
        print("   - Full margin allocation (95% of available cash)")
        print("   - Automated entry, stop-loss, and profit-taking")
        print("   - 3-phase dynamic trade management")
        print("   - Fibonacci trailing stops")
        print()
        
        # Confirmation prompt
        response = input("Are you sure you want to start LIVE TRADING? (type 'YES' to continue): ").strip()
        if response != 'YES':
            print("❌ Live trading aborted by user")
            return
        
        print("\n✅ Starting Precision Single Trade Strategy Bot...")
        print("   Press Ctrl+C to stop safely")
        print()
        
        # Create bot configuration optimized for precision strategy
        config = BotConfig(
            refresh_interval_minutes=10,   # Refresh market scan every 10 minutes
            websocket_retry_delay=30,      # WebSocket retry delay
            top_stocks_count=20,           # Monitor top 20 stocks for candidates
            enable_caching=True,           # Enable caching for performance
            max_worker_threads=3,          # Conservative thread count
            api_request_delay=0.6,         # API rate limiting
        )
        
        # Create and run bot with precision strategy ENABLED
        bot = AutomatedTradingBot(
            config=config, 
            enable_precision_strategy=True  # 🎯 ENABLE LIVE TRADING
        )
        
        logger.info("Starting Automated Trading Bot with Precision Strategy")
        logger.info("LIVE TRADING MODE: ENABLED")
        
        # Run the bot
        bot.run()
        
    except KeyboardInterrupt:
        logger.info("Bot stopped by user (Ctrl+C)")
        print("\n🛑 Bot stopped safely")
    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
        print(f"\n❌ Critical error: {str(e)}")
    finally:
        print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()
