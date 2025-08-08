#!/usr/bin/env python3
"""
Run Automated Trading Bot in Analysis Mode (No Live Trading)

This script runs the automated trading bot in analysis mode only.
It will:
1. Perform market scanning and rating
2. Show which trades the precision strategy WOULD execute
3. Display entry confirmations and setup details
4. Track hypothetical performance
5. NOT execute any real trades

This is safe to run for analysis, testing, and strategy validation.
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
    log_filename = f"logs/analysis_bot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
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
    """Main function to run analysis bot"""
    logger = setup_logging()
    
    try:
        print("="*80)
        print("📊 AUTOMATED TRADING BOT - ANALYSIS MODE")
        print("="*80)
        print()
        print("✅ SAFE MODE: This bot will NOT execute any real trades")
        print("   - Market scanning and stock rating")
        print("   - Live rating system updates")
        print("   - Strategy analysis and signals")
        print("   - Performance tracking and logging")
        print("   - WebSocket real-time data streaming")
        print()
        print("🔍 Perfect for:")
        print("   - Strategy testing and validation")
        print("   - Market analysis and research")
        print("   - System monitoring and debugging")
        print("   - Learning and experimentation")
        print()
        
        print("🚀 Starting Analysis Bot...")
        print("   Press Ctrl+C to stop")
        print()
        
        # Create bot configuration
        config = BotConfig(
            refresh_interval_minutes=10,   # Refresh every 10 minutes
            websocket_retry_delay=30,      # WebSocket retry delay
            top_stocks_count=20,           # Monitor top 20 stocks
            enable_caching=True,           # Enable caching
            max_worker_threads=3,          # Conservative thread count
        )
        
        # Create and run bot with precision strategy DISABLED (analysis only)
        bot = AutomatedTradingBot(
            config=config, 
            enable_precision_strategy=False  # 📊 ANALYSIS MODE ONLY
        )
        
        logger.info("Starting Automated Trading Bot in Analysis Mode")
        logger.info("LIVE TRADING MODE: DISABLED (Safe)")
        
        # Run the bot
        bot.run()
        
    except KeyboardInterrupt:
        logger.info("Analysis bot stopped by user (Ctrl+C)")
        print("\n🛑 Analysis bot stopped")
    except Exception as e:
        logger.error(f"Error in analysis bot: {str(e)}")
        print(f"\n❌ Error: {str(e)}")
    finally:
        print("\n👋 Analysis complete!")


if __name__ == "__main__":
    main()
