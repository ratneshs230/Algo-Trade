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
        
        # Import and run the bot's main function
        from automated_trading_bot import main as automated_bot_main
        
        logger.info("Starting automated trading bot (including dashboard)...")
        logger.info("Access dashboard at http://localhost:8000")
        logger.info("Press Ctrl+C to stop the bot gracefully")
        
        # Run the main function of the automated trading bot
        automated_bot_main()
        
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
