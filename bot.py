"""
Trading Bot - Daily Setup Functions
"""

import json
import os
import hashlib
import time
from datetime import datetime, date, timedelta
from kiteConnect import KiteLoginAutomation, KiteTrader, Config
from watchlist import get_watchlist, get_watchlist_count
from rating_system import StockRatingSystem
from live_rating_system import LiveRatingSystem
from trades import TradeManager, TradeConfig


def get_file_hash(data):
    """
    Calculate MD5 hash of data for comparison
    
    Args:
        data: Data to hash (will be converted to JSON string)
        
    Returns:
        str: MD5 hash of the data
    """
    json_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.md5(json_str.encode()).hexdigest()


def file_needs_update(filepath, new_data):
    """
    Check if file needs to be updated by comparing with new data
    
    Args:
        filepath (str): Path to the existing file
        new_data: New data to compare
        
    Returns:
        bool: True if file needs update, False otherwise
    """
    if not os.path.exists(filepath):
        return True
    
    try:
        with open(filepath, 'r') as f:
            existing_data = json.load(f)
        
        existing_hash = get_file_hash(existing_data)
        new_hash = get_file_hash(new_data)
        
        return existing_hash != new_hash
    except Exception:
        return True


def save_instruments_file(instruments, filename, file_type="instruments"):
    """
    Save instruments to file only if there are changes
    
    Args:
        instruments (list): List of instruments to save
        filename (str): Filename to save
        file_type (str): Type of file for logging
        
    Returns:
        bool: True if file was updated, False if no changes
    """
    # Ensure instruments directory exists
    os.makedirs('instruments', exist_ok=True)
    
    filepath = os.path.join('instruments', filename)
    
    if file_needs_update(filepath, instruments):
        with open(filepath, 'w') as f:
            json.dump(instruments, f, indent=2, default=str)
        print(f"{file_type} updated and saved to {filepath}")
        return True
    else:
        print(f"No changes detected in {file_type}. File {filepath} not updated.")
        return False


def validate_and_get_access_token():
    """
    Validate existing access token or generate a new one if needed
    
    Returns:
        str: Valid access token for the day
    """
    try:
        print("="*50)
        print("ACCESS TOKEN VALIDATION")
        print("="*50)
        
        # Check for existing token file
        token_file = 'access_token.json'
        today = date.today()
        
        if os.path.exists(token_file):
            try:
                with open(token_file, 'r') as f:
                    token_data = json.load(f)
                
                # Check if token is for today
                token_date = datetime.strptime(token_data.get('date', ''), '%Y-%m-%d').date()
                access_token = token_data.get('access_token', '')
                
                if token_date == today and access_token:
                    print(f"✓ Valid access token found for today ({today})")
                    print(f"  Token: {access_token[:20]}...")
                    
                    # Test the token by making a simple API call
                    try:
                        test_trader = KiteTrader(api_key=Config.API_KEY, access_token=access_token)
                        # Try to get margins as a simple API test
                        margins = test_trader.get_margins()
                        if margins:
                            print("✓ Access token validated successfully via API test")
                            print("="*50)
                            return access_token
                        else:
                            print("✗ Access token validation failed - API test returned empty response")
                    except Exception as e:
                        print(f"✗ Access token validation failed - API error: {str(e)}")
                else:
                    if token_date != today:
                        print(f"✗ Existing token is from {token_date}, need new token for {today}")
                    else:
                        print("✗ Existing token file is corrupted or empty")
                        
            except Exception as e:
                print(f"✗ Error reading existing token file: {str(e)}")
        else:
            print("✗ No access token file found")
        
        # Generate new access token
        print("\n🔄 Generating new access token...")
        automation = KiteLoginAutomation()
        access_token = automation.login()
        
        print(f"✓ New access token generated successfully: {access_token[:20]}...")
        print("="*50)
        return access_token
        
    except Exception as e:
        print(f"✗ Error in access token validation/generation: {str(e)}")
        raise


def get_access_token():
    """
    Retrieve Kite access token - only needs to be done once per day
    (Legacy function - now calls validate_and_get_access_token)
    
    Returns:
        str: Access token for the day
    """
    return validate_and_get_access_token()


def get_nse_instruments(access_token, filter_watchlist=True):
    """
    Retrieve NSE instruments - either all or filtered by watchlist
    
    Args:
        access_token (str): Valid access token
        filter_watchlist (bool): If True, filter by watchlist stocks. If False, get all instruments.
        
    Returns:
        list: List of NSE instruments (filtered or all)
    """
    try:
        print("Getting NSE instruments list...")
        
        # Initialize trader with access token
        trader = KiteTrader(api_key=Config.API_KEY, access_token=access_token)
        
        # Get all NSE instruments
        all_instruments = trader.get_instruments(exchange='NSE')
        
        print(f"Retrieved {len(all_instruments)} total NSE instruments")
        
        if not filter_watchlist:
            # Filter instruments with lot size = 1 (equity stocks)
            filtered_all_instruments = []
            for instrument in all_instruments:
                lot_size = instrument.get('lot_size', 0)
                if lot_size == 1:
                    filtered_all_instruments.append(instrument)
            
            print(f"Filtered to {len(filtered_all_instruments)} instruments with lot size = 1")
            
            # Save filtered instruments with change detection
            instruments_file = "nse_instruments_all.json"
            save_instruments_file(filtered_all_instruments, instruments_file, "All NSE instruments (lot size = 1)")
            return filtered_all_instruments
        
        # Get watchlist stocks from external file
        watchlist_stocks = get_watchlist()
        
        # Filter instruments for watchlist stocks
        filtered_instruments = []
        found_stocks = []
        
        for instrument in all_instruments:
            trading_symbol = instrument.get('tradingsymbol', '')
            
            # Check if this instrument matches any of our watchlist stocks
            for watchlist_stock in watchlist_stocks:
                if trading_symbol == watchlist_stock:
                    filtered_instruments.append(instrument)
                    if watchlist_stock not in found_stocks:
                        found_stocks.append(watchlist_stock)
                    break
        
        print(f"Found {len(filtered_instruments)} instruments for {len(found_stocks)} watchlist stocks")
        
        # Display found stocks
        print("Found stocks:")
        for stock in sorted(found_stocks):
            print(f"  - {stock}")
        
        # Display missing stocks
        missing_stocks = [stock for stock in watchlist_stocks if stock not in found_stocks]
        if missing_stocks:
            print(f"\nMissing stocks ({len(missing_stocks)}):")
            for stock in missing_stocks:
                print(f"  - {stock}")
        
        # Save filtered instruments to file with change detection
        instruments_file = "nse_instruments_watchlist.json"
        save_instruments_file(filtered_instruments, instruments_file, "Watchlist instruments")
        return filtered_instruments
        
    except Exception as e:
        print(f"Error getting NSE instruments: {str(e)}")
        raise


def get_historical_data_for_watchlist(access_token, instruments):
    """
    Fetch historical data for all watchlist stocks across multiple timeframes
    
    Args:
        access_token (str): Valid access token
        instruments (list): List of instruments from watchlist
        
    Returns:
        dict: Summary of data fetched
    """
    try:
        print("="*60)
        print("FETCHING HISTORICAL DATA FOR WATCHLIST STOCKS")
        print("="*60)
        
        # Initialize trader
        trader = KiteTrader(api_key=Config.API_KEY, access_token=access_token)
        
        # Define timeframes and their parameters
        timeframes = {
            '1minute': {'interval': 'minute', 'days_back': 1, 'min_rows': 120},
            '3minute': {'interval': '3minute', 'days_back': 3, 'min_rows': 120},
            '5minute': {'interval': '5minute', 'days_back': 5, 'min_rows': 120},
            '15minute': {'interval': '15minute', 'days_back': 15, 'min_rows': 120},
            '30minute': {'interval': '30minute', 'days_back': 30, 'min_rows': 120},
            '60minute': {'interval': '60minute', 'days_back': 60, 'min_rows': 120},
            'daily': {'interval': 'day', 'days_back': 180, 'min_rows': 120}
        }
        
        # Create historical_data directory
        os.makedirs('historical_data', exist_ok=True)
        
        # Summary tracking
        summary = {
            'total_stocks': 0,
            'successful_stocks': 0,
            'failed_stocks': [],
            'timeframes_processed': list(timeframes.keys()),
            'total_files_created': 0,
            'total_files_updated': 0
        }
        
        # Filter instruments to only equity stocks (exclude derivatives)
        equity_instruments = [
            inst for inst in instruments 
            if inst.get('segment') == 'NSE' and inst.get('instrument_type') == 'EQ'
        ]
        
        summary['total_stocks'] = len(equity_instruments)
        print(f"Processing {len(equity_instruments)} equity instruments...")
        
        for i, instrument in enumerate(equity_instruments, 1):
            trading_symbol = instrument.get('tradingsymbol', '')
            instrument_token = instrument.get('instrument_token', '')
            company_name = instrument.get('name', '')
            
            print(f"\n[{i}/{len(equity_instruments)}] Processing: {trading_symbol} ({company_name})")
            
            stock_success = True
            stock_dir = os.path.join('historical_data', trading_symbol)
            os.makedirs(stock_dir, exist_ok=True)
            
            # Fetch data for each timeframe
            for timeframe, params in timeframes.items():
                try:
                    print(f"  Fetching {timeframe} data...", end=' ')
                    
                    # Calculate date range
                    to_date = datetime.now()
                    from_date = to_date - timedelta(days=params['days_back'])
                    
                    # Fetch historical data
                    historical_data = trader.get_historical_data(
                        instrument_token=int(instrument_token),
                        from_date=from_date,
                        to_date=to_date,
                        interval=params['interval']
                    )
                    
                    if len(historical_data) >= params['min_rows']:
                        # Save data to file
                        filename = f"{trading_symbol}_{timeframe}.json"
                        filepath = os.path.join(stock_dir, filename)
                        
                        # Check if file needs update
                        if file_needs_update(filepath, historical_data):
                            with open(filepath, 'w') as f:
                                json.dump(historical_data, f, indent=2, default=str)
                            print(f"✓ Updated ({len(historical_data)} rows)")
                            summary['total_files_updated'] += 1
                        else:
                            print(f"✓ No changes ({len(historical_data)} rows)")
                        
                        summary['total_files_created'] += 1
                    else:
                        print(f"✗ Insufficient data ({len(historical_data)} rows, need {params['min_rows']})")
                        stock_success = False
                        
                except Exception as e:
                    print(f"✗ Error: {str(e)}")
                    stock_success = False
            
            if stock_success:
                summary['successful_stocks'] += 1
            else:
                summary['failed_stocks'].append(trading_symbol)
        
        # Print summary
        print("\n" + "="*60)
        print("HISTORICAL DATA FETCH SUMMARY")
        print("="*60)
        print(f"Total Stocks Processed: {summary['total_stocks']}")
        print(f"Successful: {summary['successful_stocks']}")
        print(f"Failed: {len(summary['failed_stocks'])}")
        print(f"Total Files: {summary['total_files_created']}")
        print(f"Files Updated: {summary['total_files_updated']}")
        print(f"Timeframes: {', '.join(summary['timeframes_processed'])}")
        
        if summary['failed_stocks']:
            print(f"\nFailed Stocks: {', '.join(summary['failed_stocks'])}")
        
        print("="*60)
        
        return summary
        
    except Exception as e:
        print(f"Error in historical data fetch: {str(e)}")
        raise


def get_live_ratings_for_top_stocks(access_token, instruments, max_stocks=20):
    """
    Use LiveRatingSystem to rate top stocks with real-time analysis
    
    Args:
        access_token (str): Valid access token
        instruments (list): List of instruments
        max_stocks (int): Maximum number of stocks to rate
        
    Returns:
        list: List of live rating results
    """
    try:
        print("="*60)
        print("LIVE RATING SYSTEM - REAL-TIME ANALYSIS")
        print("="*60)
        
        # Initialize trader
        trader = KiteTrader(api_key=Config.API_KEY, access_token=access_token)
        
        # Filter to equity instruments only
        equity_instruments = [
            inst for inst in instruments 
            if inst.get('segment') == 'NSE' and inst.get('instrument_type') == 'EQ'
        ]
        
        # Limit to max_stocks for performance
        if len(equity_instruments) > max_stocks:
            equity_instruments = equity_instruments[:max_stocks]
            print(f"Limited to top {max_stocks} stocks for live rating")
        
        print(f"Initializing live rating systems for {len(equity_instruments)} stocks...")
        
        live_ratings = []
        successful_ratings = 0
        failed_ratings = []
        
        for i, instrument in enumerate(equity_instruments, 1):
            trading_symbol = instrument.get('tradingsymbol', '')
            instrument_token = instrument.get('instrument_token', '')
            company_name = instrument.get('name', '')
            
            print(f"\n[{i}/{len(equity_instruments)}] Initializing: {trading_symbol} ({company_name})")
            
            try:
                # Initialize LiveRatingSystem for this stock
                live_system = LiveRatingSystem(
                    instrument_token=str(instrument_token),
                    kite_trader_instance=trader
                )
                
                # Get live rating
                rating_result = live_system.get_live_rating()
                
                # Add additional metadata
                rating_result['trading_symbol'] = trading_symbol
                rating_result['company_name'] = company_name
                
                live_ratings.append(rating_result)
                successful_ratings += 1
                
                print(f"  ✓ {rating_result['rating_text']} ({rating_result['final_rating']:.2f})")
                
            except Exception as e:
                print(f"  ✗ Error: {str(e)}")
                failed_ratings.append(trading_symbol)
                
                # Add error entry
                live_ratings.append({
                    'trading_symbol': trading_symbol,
                    'company_name': company_name,
                    'instrument_token': str(instrument_token),
                    'timestamp': datetime.now().isoformat(),
                    'final_rating': 0.0,
                    'rating_text': 'Error',
                    'emoji': '❌',
                    'error': str(e)
                })
        
        # Sort by final rating (descending)
        live_ratings.sort(key=lambda x: x.get('final_rating', 0), reverse=True)
        
        # Print summary
        print("\n" + "="*60)
        print("LIVE RATING SYSTEM SUMMARY")
        print("="*60)
        print(f"Total Stocks Processed: {len(equity_instruments)}")
        print(f"Successful Ratings: {successful_ratings}")
        print(f"Failed Ratings: {len(failed_ratings)}")
        
        if failed_ratings:
            print(f"Failed Stocks: {', '.join(failed_ratings)}")
        
        # Print top ratings
        print(f"\nTop 10 Live Ratings:")
        print("-" * 80)
        print(f"{'Symbol':<12} {'Score':<8} {'Rating':<15} {'Strategic':<10} {'Tactical':<10}")
        print("-" * 80)
        
        for rating in live_ratings[:10]:
            if 'error' not in rating:
                symbol = rating.get('trading_symbol', 'N/A')
                score = f"{rating.get('final_rating', 0):.2f}"
                rating_text = f"{rating.get('emoji', '')} {rating.get('rating_text', 'N/A')}"
                strategic = f"{rating.get('strategic_score', 0):.3f}"
                tactical = f"{rating.get('tactical_score', 0):.3f}"
                
                print(f"{symbol:<12} {score:<8} {rating_text:<15} {strategic:<10} {tactical:<10}")
            else:
                symbol = rating.get('trading_symbol', 'N/A')
                print(f"{symbol:<12} {'N/A':<8} {'❌ Error':<15} {'N/A':<10} {'N/A':<10}")
        
        print("="*60)
        
        # Save live ratings to file
        save_live_ratings_to_file(live_ratings)
        
        return live_ratings
        
    except Exception as e:
        print(f"Error in live rating system: {str(e)}")
        raise


def save_live_ratings_to_file(live_ratings, filename="live_ratings.json"):
    """
    Save live ratings to file with top 10 and bottom 10 selection
    
    Args:
        live_ratings (list): List of live rating results
        filename (str): Filename to save
    """
    try:
        # Ensure ratings directory exists
        os.makedirs('ratings', exist_ok=True)
        filepath = os.path.join('ratings', filename)
        
        # Filter out error reports for ranking
        valid_ratings = [r for r in live_ratings if 'error' not in r]
        error_ratings = [r for r in live_ratings if 'error' in r]
        
        # Get top 10 and bottom 10
        top_10 = valid_ratings[:10] if len(valid_ratings) >= 10 else valid_ratings
        bottom_10 = valid_ratings[-10:] if len(valid_ratings) >= 10 else []
        
        # Combine for filtered results
        if len(valid_ratings) <= 20:
            filtered_ratings = valid_ratings
        else:
            filtered_ratings = top_10 + [r for r in bottom_10 if r not in top_10]
        
        # Prepare data for saving
        save_data = {
            'timestamp': datetime.now().isoformat(),
            'rating_type': 'Live Rating System',
            'total_stocks_analyzed': len(live_ratings),
            'successful_ratings': len(valid_ratings),
            'failed_ratings': len(error_ratings),
            'filtered_count': len(filtered_ratings),
            'selection_criteria': 'Top 10 and Bottom 10 by live rating score',
            'top_10_scores': [r.get('final_rating', 0) for r in top_10],
            'bottom_10_scores': [r.get('final_rating', 0) for r in bottom_10],
            'ratings': filtered_ratings
        }
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        print(f"\nLive ratings (top 10 + bottom 10) saved to {filepath}")
        print(f"Saved {len(filtered_ratings)} out of {len(valid_ratings)} successful ratings")
        
    except Exception as e:
        print(f"Error saving live ratings: {str(e)}")


def auto_trading_loop(access_token, instruments, trade_manager):
    """
    Main auto-trading loop
    
    Args:
        access_token (str): Valid access token
        instruments (list): List of instruments
        trade_manager (TradeManager): Trade manager instance
    """
    try:
        print("\n" + "="*60)
        print("AUTO-TRADING LOOP STARTED")
        print("="*60)
        print("Press Ctrl+C to stop auto-trading")
        
        trader = KiteTrader(api_key=Config.API_KEY, access_token=access_token)
        
        # Filter to equity instruments
        equity_instruments = [
            inst for inst in instruments 
            if inst.get('segment') == 'NSE' and inst.get('instrument_type') == 'EQ'
        ]
        
        trading_active = True
        iteration = 0
        
        while trading_active:
            try:
                iteration += 1
                print(f"\n--- Trading Iteration {iteration} ---")
                print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Check if we should continue trading (market hours, max positions, etc.)
                if len(trade_manager.active_trades) >= trade_manager.config.max_positions:
                    print(f"Maximum positions ({trade_manager.config.max_positions}) reached. Monitoring existing trades...")
                    trade_manager.print_active_trades()
                    time.sleep(60)  # Wait 1 minute before checking again
                    continue
                
                # Get live ratings for top 20 stocks
                live_ratings = get_live_ratings_for_top_stocks(
                    access_token=access_token,
                    instruments=equity_instruments[:20],  # Limit for performance
                    max_stocks=20
                )
                
                if live_ratings:
                    # Analyze ratings for trading opportunity
                    trade_recommendation = trade_manager.analyze_ratings_for_trading(live_ratings)
                    
                    if trade_recommendation:
                        print(f"\nTrading opportunity detected!")
                        print(f"Symbol: {trade_recommendation['rating_data'].get('trading_symbol', 'N/A')}")
                        print(f"Score: {trade_recommendation['score']:.2f}")
                        print(f"Action: {trade_recommendation['trade_type'].value}")
                        
                        # Ask user for confirmation (in auto mode, you might want to auto-approve)
                        while True:
                            confirm = input("\nPlace this trade? (y/n/s=stop): ").strip().lower()
                            if confirm == 'y':
                                # Place the trade
                                trade = trade_manager.place_trade(trade_recommendation)
                                if trade:
                                    print(f"Trade placed successfully: {trade.trade_id}")
                                    trade_manager.print_active_trades()
                                else:
                                    print("Failed to place trade")
                                break
                            elif confirm == 'n':
                                print("Trade skipped")
                                break
                            elif confirm == 's':
                                print("Stopping auto-trading...")
                                trading_active = False
                                break
                            else:
                                print("Please enter 'y' for yes, 'n' for no, or 's' to stop")
                    else:
                        print("No suitable trading opportunities found")
                        print("Current active trades:")
                        trade_manager.print_active_trades()
                else:
                    print("No rating data available")
                
                if trading_active:
                    # Wait before next iteration (adjust as needed)
                    print(f"\nWaiting 5 minutes before next iteration...")
                    time.sleep(300)  # 5 minutes
                
            except KeyboardInterrupt:
                print("\nReceived stop signal. Ending auto-trading...")
                break
            except Exception as e:
                print(f"Error in trading iteration: {str(e)}")
                print("Continuing to next iteration...")
                time.sleep(60)  # Wait 1 minute before retrying
        
        # Cleanup
        print("\nAuto-trading stopped. Final summary:")
        summary = trade_manager.get_trade_summary()
        print(f"Active trades: {summary['active_trades']}")
        print(f"Completed trades: {summary['completed_trades']}")
        print(f"Total P&L: ₹{summary['total_pnl']}")
        
        # Ask if user wants to close all trades
        if summary['active_trades'] > 0:
            while True:
                close_all = input("\nClose all active trades? (y/n): ").strip().lower()
                if close_all == 'y':
                    closed = trade_manager.close_all_trades("manual_stop")
                    print(f"Closed {closed} trades")
                    break
                elif close_all == 'n':
                    print("Keeping trades open. Use trade_manager methods to manage them manually.")
                    break
                else:
                    print("Please enter 'y' or 'n'")
        
        # Stop monitoring
        trade_manager.stop_monitoring()
        
    except Exception as e:
        print(f"Error in auto-trading loop: {str(e)}")
        if 'trade_manager' in locals():
            trade_manager.stop_monitoring()


def daily_setup_with_token(access_token, filter_watchlist=None):
    """
    Perform daily setup with pre-validated access token
    
    Args:
        access_token (str): Pre-validated access token
        filter_watchlist (bool, optional): If True, filter by watchlist. If False, get all instruments.
                                         If None, prompt user for choice.
    
    Returns:
        tuple: (access_token, instruments_list)
    """
    print("="*50)
    print("DAILY SETUP - INSTRUMENTS & DATA")
    print("="*50)
    
    try:
        # Step 1: Ask user for instrument preference if not specified
        if filter_watchlist is None:
            print("\nInstrument Retrieval Options:")
            print(f"1. Watchlist only ({get_watchlist_count()} specific stocks)")
            print("2. All NSE instruments (complete list)")
            
            while True:
                choice = input("\nEnter your choice (1 or 2): ").strip()
                if choice == "1":
                    filter_watchlist = True
                    break
                elif choice == "2":
                    filter_watchlist = False
                    break
                else:
                    print("Invalid choice. Please enter 1 or 2.")
        
        # Step 2: Get NSE instruments based on choice
        if filter_watchlist:
            print("\nFetching watchlist instruments...")
            instruments = get_nse_instruments(access_token, filter_watchlist=True)
        else:
            print("\nFetching all NSE instruments...")
            instruments = get_nse_instruments(access_token, filter_watchlist=False)
        
        print("="*50)
        print("DAILY SETUP COMPLETED SUCCESSFULLY")
        print(f"Access Token: {access_token[:20]}... (validated)")
        print(f"NSE Instruments: {len(instruments)} symbols")
        if filter_watchlist:
            print("Mode: Watchlist only")
        else:
            print("Mode: All NSE instruments")
        print("="*50)
        
        return access_token, instruments
        
    except Exception as e:
        print(f"Daily setup failed: {str(e)}")
        raise


def daily_setup(filter_watchlist=None):
    """
    Perform daily setup - get access token and instruments list
    (Legacy function - now validates token first)
    
    Args:
        filter_watchlist (bool, optional): If True, filter by watchlist. If False, get all instruments.
                                         If None, prompt user for choice.
    
    Returns:
        tuple: (access_token, instruments_list)
    """
    # Validate access token first
    access_token = validate_and_get_access_token()
    
    # Run setup with validated token
    return daily_setup_with_token(access_token, filter_watchlist)


if __name__ == "__main__":
    # Step 1: Validate and get access token first
    print("🚀 STARTING TRADING BOT")
    print("="*50)
    
    try:
        # Validate access token before doing anything else
        access_token = validate_and_get_access_token()
        
        # Run daily setup with validated token
        print("\n📊 PROCEEDING WITH DAILY SETUP")
        _, instruments = daily_setup_with_token(access_token)
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {str(e)}")
        print("Bot cannot continue without a valid access token.")
        exit(1)
    
    # Ask if user wants to fetch historical data
    if instruments:
        print("\nHistorical Data Options:")
        print("1. Fetch historical data for stocks")
        print("2. Skip historical data fetch")
        
        historical_data_fetched = False
        while True:
            choice = input("\nEnter your choice (1 or 2): ").strip()
            if choice == "1":
                print("\nFetching historical data...")
                summary = get_historical_data_for_watchlist(access_token, instruments)
                historical_data_fetched = True
                break
            elif choice == "2":
                print("Skipping historical data fetch.")
                break
            else:
                print("Invalid choice. Please enter 1 or 2.")
        
        # Check if historical data exists (either just fetched or already available)
        has_historical_data = historical_data_fetched or os.path.exists('historical_data')
        
        if has_historical_data:
            if not historical_data_fetched:
                print("\nFound existing historical data directory.")
            
            print("\nStock Rating Options:")
            print("1. Generate stock ratings using technical analysis (batch mode)")
            print("2. Generate live ratings using real-time analysis (top 20 stocks)")
            print("3. Auto-trading mode (live ratings + automatic trade placement)")
            print("4. Skip stock rating")
            
            while True:
                choice = input("\nEnter your choice (1, 2, 3, or 4): ").strip()
                if choice == "1":
                    print("\nInitializing Stock Rating System (Batch Mode)...")
                    try:
                        # Initialize rating system
                        rating_system = StockRatingSystem()
                        
                        # Get symbols from instruments
                        symbols = []
                        for instrument in instruments:
                            if instrument.get('instrument_type') == 'EQ':
                                symbols.append(instrument.get('tradingsymbol', ''))
                        
                        symbols = [s for s in symbols if s]  # Remove empty strings
                        
                        if symbols:
                            print(f"Rating {len(symbols)} stocks...")
                            
                            # Rate all stocks with instrument details
                            reports = rating_system.rate_multiple_stocks(symbols, instruments)
                            
                            # Print summary table
                            rating_system.print_summary_table(reports)
                            
                            # Save ratings to file
                            rating_system.save_ratings_to_file(reports)
                            
                            print("\nStock rating completed successfully!")
                        else:
                            print("No equity symbols found for rating.")
                            
                    except Exception as e:
                        print(f"Error in stock rating: {str(e)}")
                    
                    break
                elif choice == "2":
                    print("\nInitializing Live Rating System...")
                    try:
                        # Use live rating system for top 20 stocks
                        live_ratings = get_live_ratings_for_top_stocks(
                            access_token=access_token,
                            instruments=instruments,
                            max_stocks=20
                        )
                        
                        print("\nLive rating system completed successfully!")
                        
                    except Exception as e:
                        print(f"Error in live rating system: {str(e)}")
                    
                    break
                elif choice == "3":
                    print("\nInitializing Auto-Trading Mode...")
                    try:
                        # Initialize trade manager
                        trade_config = TradeConfig()
                        trade_manager = TradeManager(
                            KiteTrader(api_key=Config.API_KEY, access_token=access_token),
                            trade_config
                        )
                        
                        # Start monitoring
                        trade_manager.start_monitoring()
                        
                        print("Auto-trading mode activated with the following settings:")
                        print(f"  - Max positions: {trade_config.max_positions}")
                        print(f"  - Max position size: ₹{trade_config.max_position_size:,.0f}")
                        print(f"  - Min rating threshold: {trade_config.min_rating_threshold}")
                        print(f"  - Default stop loss: {trade_config.default_stoploss_pct}%")
                        print(f"  - Default target: {trade_config.default_target_pct}%")
                        
                        # Run auto-trading loop
                        auto_trading_loop(access_token, instruments, trade_manager)
                        
                    except Exception as e:
                        print(f"Error in auto-trading mode: {str(e)}")
                    
                    break
                elif choice == "4":
                    print("Skipping stock rating.")
                    break
                else:
                    print("Invalid choice. Please enter 1, 2, 3, or 4.")
        else:
            print("\nNo historical data found.")
            print("Stock rating requires historical data. Please run historical data fetch first.")
            print("You can restart the program and choose option 1 to fetch historical data.")
