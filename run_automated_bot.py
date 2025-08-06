#!/usr/bin/env python3
"""
Automated Trading Bot Launcher

This script launches the fully automated trading bot with proper error handling
and logging configuration.
"""

import sys
import os
import logging
from datetime import datetime

def setup_logging():
    """Setup logging configuration"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(f'logs/automated_bot_{datetime.now().strftime("%Y%m%d")}.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    """Main launcher function"""
    try:
        # Setup logging
        setup_logging()
        logger = logging.getLogger(__name__)
        
        logger.info("="*80)
        logger.info("🚀 AUTOMATED TRADING BOT LAUNCHER")
        logger.info("="*80)
        
        # Import and run the bot
        from automated_trading_bot import AutomatedTradingBot, BotConfig
        
        # Create configuration with API rate limit friendly settings
        config = BotConfig(
            refresh_interval_minutes=15,      # Refresh every 15 minutes (reduced frequency)
            websocket_retry_delay=30,         # Wait 30 seconds before retry
            top_stocks_count=20,              # Track top 20 stocks
            max_worker_threads=3,             # Reduced threads for rate limit compliance
            enable_caching=True,              # Enable caching
            cache_expiry_minutes=10,          # Longer cache time to reduce API calls
            api_request_delay=0.8,            # 800ms delay between requests
            rate_limit_wait=20                # Wait 20 seconds on rate limit
        )
        
        logger.info("Configuration:")
        logger.info(f"  📊 Refresh interval: {config.refresh_interval_minutes} minutes")
        logger.info(f"  🔄 Retry delay: {config.websocket_retry_delay} seconds")
        logger.info(f"  🎯 Top stocks count: {config.top_stocks_count}")
        logger.info(f"  🧵 Worker threads: {config.max_worker_threads}")
        logger.info(f"  💾 Caching enabled: {config.enable_caching}")
        
        # Create and run the bot
        bot = AutomatedTradingBot(config)
        
        logger.info("Starting automated trading bot...")
        logger.info("Press Ctrl+C to stop the bot gracefully")
        
        # Run the bot
        bot.run()
        
    except KeyboardInterrupt:
        logger.info("Bot stopped by user (Ctrl+C)")
    except ImportError as e:
        logger.error(f"Import error: {str(e)}")
        logger.error("Please ensure all required dependencies are installed:")
        logger.error("  pip install -r requirements.txt")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error("Check the log files for more details")
    finally:
        logger.info("Automated Trading Bot Launcher finished")

if __name__ == "__main__":
    main()
