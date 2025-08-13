"""
Watchlist Rating System

This module implements a comprehensive stock rating system that:
1. Calculates technical indicator scores across 6 timeframes (excluding daily)
2. Applies timeframe-specific weights with short-term focus
3. Generates final composite scores and ratings
4. Provides actionable buy/sell/neutral recommendations
5. Used for selecting top 20 stocks from watchlist for live monitoring
"""

import json
import os
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from indicator_calculator import IndicatorCalculator, extract_ohlcv_from_historical_data


class StockRatingSystem:
    """Comprehensive watchlist rating system with 6-timeframe analysis (excluding daily)"""
    
    # Configuration constants
    MIN_CANDLES_FOR_RATING = 30  # Minimum candles required for rating calculation
    
    def __init__(self):
        self.calculator = IndicatorCalculator()
        
        # Timeframe weights (short-term focused approach) - Daily removed
        self.timeframe_weights = {
            '1minute': 0.30,    # 30% - Highest weight for immediate signals
            '3minute': 0.25,    # 25% - Very high weight for short-term momentum
            '5minute': 0.20,    # 20% - High weight for quick trends
            '15minute': 0.15,   # 15% - Medium weight for intermediate trends
            '30minute': 0.06,   # 6% - Lower weight for longer trends
            '60minute': 0.04    # 4% - Adjusted weight for hourly trends (was 3% + 1% from daily)
        }
        
        # Indicator weights (equal weighting for simplicity)
        self.indicator_weights = {
            'ema_crossover': 0.125,      # 12.5%
            'price_vs_ma200': 0.125,     # 12.5%
            'macd': 0.125,               # 12.5%
            'adx': 0.125,                # 12.5%
            'rsi': 0.125,                # 12.5%
            'stochastic': 0.125,         # 12.5%
            'bollinger_bands': 0.125,    # 12.5%
            'obv': 0.125                 # 12.5%
        }
        
        # Rating thresholds
        self.rating_thresholds = {
            'strong_buy': (7.5, 10.0),
            'buy': (2.5, 7.4),
            'neutral': (-2.4, 2.4),
            'sell': (-7.4, -2.5),
            'strong_sell': (-10.0, -7.5)
        }
    
    def load_historical_data(self, symbol: str, timeframe: str) -> Optional[List[Dict]]:
        """Load historical data for a specific symbol and timeframe (excluding daily)"""
        try:
            # Skip daily timeframe entirely
            if timeframe == 'daily':
                print(f"Daily timeframe excluded from rating system for {symbol}")
                return None
                
            filepath = os.path.join('historical_data', symbol, f"{symbol}_{timeframe}.json")
            
            if not os.path.exists(filepath):
                print(f"Warning: No data found for {symbol} {timeframe}")
                return None
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            return data
        
        except Exception as e:
            print(f"Error loading data for {symbol} {timeframe}: {str(e)}")
            return None
    
    def calculate_timeframe_score(self, symbol: str, timeframe: str) -> Tuple[float, Dict[str, float]]:
        """Calculate composite score for a specific timeframe"""
        # Load historical data
        historical_data = self.load_historical_data(symbol, timeframe)
        
        # Check minimum candle requirement
        if not historical_data or len(historical_data) < self.MIN_CANDLES_FOR_RATING:
            return 0.0, {}
        
        # Extract OHLCV data
        ohlcv_data = extract_ohlcv_from_historical_data(historical_data)
        
        # Calculate all indicator scores
        indicator_scores = self.calculator.calculate_all_indicators(ohlcv_data)
        
        # Calculate weighted timeframe score
        timeframe_score = 0.0
        for indicator, score in indicator_scores.items():
            weight = self.indicator_weights.get(indicator, 0.0)
            timeframe_score += weight * score
        
        return timeframe_score, indicator_scores
    
    def calculate_composite_score(self, symbol: str) -> Dict:
        """Calculate final composite score across all timeframes"""
        timeframe_scores = {}
        timeframe_indicators = {}
        
        # Calculate scores for each timeframe
        for timeframe in self.timeframe_weights.keys():
            score, indicators = self.calculate_timeframe_score(symbol, timeframe)
            timeframe_scores[timeframe] = score
            timeframe_indicators[timeframe] = indicators
        
        # Calculate weighted composite score
        composite_score = 0.0
        total_weight = 0.0
        
        for timeframe, score in timeframe_scores.items():
            weight = self.timeframe_weights[timeframe]
            if score != 0.0:  # Only include timeframes with valid data
                composite_score += weight * score
                total_weight += weight
        
        # Normalize by actual weights used
        if total_weight > 0:
            composite_score = composite_score / total_weight
        
        # Scale to final display score (-10 to +10)
        final_score = composite_score * 10
        
        return {
            'symbol': symbol,
            'final_score': final_score,
            'composite_score': composite_score,
            'timeframe_scores': timeframe_scores,
            'timeframe_indicators': timeframe_indicators,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_rating_from_score(self, score: float) -> Tuple[str, str, str]:
        """Convert numerical score to rating, emoji, and interpretation"""
        if self.rating_thresholds['strong_buy'][0] <= score <= self.rating_thresholds['strong_buy'][1]:
            return "Strong Buy", "🟢", "High-conviction bullish signal. Strong trend and momentum alignment across most timeframes."
        
        elif self.rating_thresholds['buy'][0] <= score <= self.rating_thresholds['buy'][1]:
            return "Buy", "🟢", "Clear bullish bias. The dominant trend is up, but some conflicting signals may exist."
        
        elif self.rating_thresholds['neutral'][0] <= score <= self.rating_thresholds['neutral'][1]:
            return "Neutral", "🟡", "No clear directional edge. Signals are conflicting or the market is in a sideways chop."
        
        elif self.rating_thresholds['sell'][0] <= score <= self.rating_thresholds['sell'][1]:
            return "Sell", "🔴", "Clear bearish bias. The dominant trend is down, though counter-trend bounces may occur."
        
        elif self.rating_thresholds['strong_sell'][0] <= score <= self.rating_thresholds['strong_sell'][1]:
            return "Strong Sell", "🔴", "High-conviction bearish signal. Strong downtrend and negative momentum across most timeframes."
        
        else:
            return "Neutral", "🟡", "Score out of expected range. No clear directional signal."
    
    def generate_detailed_report(self, symbol: str) -> Dict:
        """Generate comprehensive rating report for a symbol"""
        # Calculate composite score
        score_data = self.calculate_composite_score(symbol)
        
        # Get rating information
        rating, emoji, interpretation = self.get_rating_from_score(score_data['final_score'])
        
        # Create detailed report
        report = {
            'symbol': symbol,
            'timestamp': score_data['timestamp'],
            'final_score': round(score_data['final_score'], 2),
            'rating': rating,
            'emoji': emoji,
            'interpretation': interpretation,
            'timeframe_analysis': {},
            'indicator_summary': {},
            'data_quality': {}
        }
        
        # Add timeframe analysis
        for timeframe, score in score_data['timeframe_scores'].items():
            weight = self.timeframe_weights[timeframe]
            indicators = score_data['timeframe_indicators'].get(timeframe, {})
            
            report['timeframe_analysis'][timeframe] = {
                'score': round(score, 3),
                'weight': weight,
                'weighted_contribution': round(score * weight, 3),
                'indicators': {k: round(v, 3) for k, v in indicators.items()}
            }
        
        # Add indicator summary (average across timeframes)
        indicator_averages = {}
        for indicator in self.indicator_weights.keys():
            scores = []
            for tf_indicators in score_data['timeframe_indicators'].values():
                if indicator in tf_indicators and tf_indicators[indicator] != 0.0:
                    scores.append(tf_indicators[indicator])
            
            if scores:
                indicator_averages[indicator] = sum(scores) / len(scores)
            else:
                indicator_averages[indicator] = 0.0
        
        report['indicator_summary'] = {k: round(v, 3) for k, v in indicator_averages.items()}
        
        # Add data quality information
        valid_timeframes = sum(1 for score in score_data['timeframe_scores'].values() if score != 0.0)
        total_timeframes = len(self.timeframe_weights)
        
        report['data_quality'] = {
            'valid_timeframes': valid_timeframes,
            'total_timeframes': total_timeframes,
            'data_coverage': round(valid_timeframes / total_timeframes * 100, 1)
        }
        
        return report
    
    def rate_multiple_stocks(self, symbols: List[str], instruments_data: List[Dict] = None) -> List[Dict]:
        """Rate multiple stocks and return sorted results"""
        reports = []
        
        # Create a lookup dictionary for instrument details
        instruments_lookup = {}
        if instruments_data:
            for instrument in instruments_data:
                trading_symbol = instrument.get('tradingsymbol', '')
                if trading_symbol:
                    instruments_lookup[trading_symbol] = instrument
        
        print(f"Rating {len(symbols)} stocks...")
        
        for i, symbol in enumerate(symbols, 1):
            print(f"[{i}/{len(symbols)}] Rating {symbol}...", end=' ')
            
            try:
                report = self.generate_detailed_report(symbol)
                
                # Add instrument details if available
                if symbol in instruments_lookup:
                    instrument = instruments_lookup[symbol]
                    report['instrument_details'] = {
                        'instrument_token': instrument.get('instrument_token'),
                        'exchange_token': instrument.get('exchange_token'),
                        'tradingsymbol': instrument.get('tradingsymbol'),
                        'name': instrument.get('name'),
                        'last_price': instrument.get('last_price'),
                        'expiry': instrument.get('expiry'),
                        'strike': instrument.get('strike'),
                        'tick_size': instrument.get('tick_size'),
                        'lot_size': instrument.get('lot_size'),
                        'instrument_type': instrument.get('instrument_type'),
                        'segment': instrument.get('segment'),
                        'exchange': instrument.get('exchange')
                    }
                
                reports.append(report)
                print(f"✓ {report['rating']} ({report['final_score']:.1f})")
                
            except Exception as e:
                print(f"✗ Error: {str(e)}")
                # Add error report
                error_report = {
                    'symbol': symbol,
                    'error': str(e),
                    'final_score': 0.0,
                    'rating': 'Error',
                    'emoji': '❌'
                }
                
                # Add instrument details for error reports too
                if symbol in instruments_lookup:
                    instrument = instruments_lookup[symbol]
                    error_report['instrument_details'] = {
                        'instrument_token': instrument.get('instrument_token'),
                        'exchange_token': instrument.get('exchange_token'),
                        'tradingsymbol': instrument.get('tradingsymbol'),
                        'name': None,
                        'last_price': None,
                        'expiry': None,
                        'strike': None,
                        'tick_size': None,
                        'lot_size': None,
                        'instrument_type': None,
                        'segment': None,
                        'exchange': None
                    }
                
                reports.append(error_report)
        
        # Sort by final score (descending)
        reports.sort(key=lambda x: x.get('final_score', 0), reverse=True)
        
        return reports
    
    def save_ratings_to_file(self, reports: List[Dict], filename: str = None):
        """Save only top 10 and bottom 10 rating reports to JSON file with simplified format"""
        if filename is None:
            filename = "filtered_20.json"
        
        # Ensure ratings directory exists
        os.makedirs('ratings', exist_ok=True)
        filepath = os.path.join('ratings', filename)
        
        # Filter out error reports for ranking
        valid_reports = [r for r in reports if 'error' not in r]
        error_reports = [r for r in reports if 'error' in r]
        
        # Sort by final score (already sorted in rate_multiple_stocks, but ensuring)
        valid_reports.sort(key=lambda x: x.get('final_score', 0), reverse=True)
        
        # Get top 10 and bottom 10
        top_10 = valid_reports[:10] if len(valid_reports) >= 10 else valid_reports
        bottom_10 = valid_reports[-10:] if len(valid_reports) >= 10 else []
        
        # If we have less than 20 stocks total, take what we have
        if len(valid_reports) <= 20:
            filtered_reports = valid_reports
        else:
            # Combine top 10 and bottom 10, avoiding duplicates
            filtered_reports = top_10 + [r for r in bottom_10 if r not in top_10]
        
        # Simplify the reports to include only essential information with instrument details
        simplified_ratings = []
        for report in filtered_reports:
            simplified_report = {
                'final_score': report['final_score'],
                'rating': report['rating'],
                'emoji': report['emoji']
            }
            
            # Add instrument details if available
            if 'instrument_details' in report:
                simplified_report.update(report['instrument_details'])
            else:
                # Fallback if instrument details not available
                simplified_report.update({
                    'instrument_token': None,
                    'exchange_token': None,
                    'tradingsymbol': report['symbol'],
                    'name': None,
                    'last_price': None,
                    'expiry': None,
                    'strike': None,
                    'tick_size': None,
                    'lot_size': None,
                    'instrument_type': None,
                    'segment': None,
                    'exchange': None
                })
            
            simplified_ratings.append(simplified_report)
        
        # Prepare data for saving
        save_data = {
            'timestamp': datetime.now().isoformat(),
            'total_stocks_analyzed': len(reports),
            'successful_ratings': len(valid_reports),
            'failed_ratings': len(error_reports),
            'filtered_count': len(filtered_reports),
            'selection_criteria': 'Top 10 and Bottom 10 by final score',
            'top_10_scores': [r.get('final_score', 0) for r in top_10],
            'bottom_10_scores': [r.get('final_score', 0) for r in bottom_10],
            'ratings': simplified_ratings
        }
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        print(f"\nFiltered ratings (top 10 + bottom 10) saved to {filepath}")
        print(f"Saved {len(filtered_reports)} out of {len(valid_reports)} successful ratings")
        return filepath
    
    def print_summary_table(self, reports: List[Dict]):
        """Print a formatted summary table of ratings"""
        print("\n" + "="*80)
        print("STOCK RATING SUMMARY")
        print("="*80)
        print(f"{'Symbol':<12} {'Score':<8} {'Rating':<12} {'Coverage':<10} {'Interpretation'}")
        print("-"*80)
        
        for report in reports:
            if 'error' in report:
                print(f"{report['symbol']:<12} {'N/A':<8} {'Error':<12} {'N/A':<10} {report.get('error', 'Unknown error')[:40]}")
            else:
                symbol = report['symbol']
                score = f"{report['final_score']:.1f}"
                rating = f"{report['emoji']} {report['rating']}"
                coverage = f"{report['data_quality']['data_coverage']:.0f}%"
                interpretation = report['interpretation'][:40] + "..." if len(report['interpretation']) > 40 else report['interpretation']
                
                print(f"{symbol:<12} {score:<8} {rating:<12} {coverage:<10} {interpretation}")
        
        print("="*80)
        
        # Print distribution summary
        rating_counts = {}
        for report in reports:
            if 'error' not in report:
                rating = report['rating']
                rating_counts[rating] = rating_counts.get(rating, 0) + 1
        
        print("\nRating Distribution:")
        for rating, count in rating_counts.items():
            print(f"  {rating}: {count}")
        
        print(f"\nTotal Stocks Rated: {len([r for r in reports if 'error' not in r])}")
        print(f"Failed Ratings: {len([r for r in reports if 'error' in r])}")


def get_symbols_from_instruments():
    """Get list of symbols from watchlist instruments"""
    try:
        instruments_file = os.path.join('instruments', 'nse_instruments_watchlist.json')
        
        if not os.path.exists(instruments_file):
            print("Error: Watchlist instruments file not found. Run bot.py first.")
            return []
        
        with open(instruments_file, 'r') as f:
            instruments = json.load(f)
        
        # Extract trading symbols from equity instruments
        symbols = []
        for instrument in instruments:
            if instrument.get('instrument_type') == 'EQ':
                symbols.append(instrument.get('tradingsymbol', ''))
        
        return [s for s in symbols if s]  # Remove empty strings
        
    except Exception as e:
        print(f"Error loading symbols: {str(e)}")
        return []


if __name__ == "__main__":
    # Initialize rating system
    rating_system = StockRatingSystem()
    
    # Get symbols from watchlist
    symbols = get_symbols_from_instruments()
    
    if not symbols:
        print("No symbols found. Please run automated_trading_bot.py first to download instruments and historical data.")
        exit(1)
    
    print(f"Found {len(symbols)} symbols in watchlist")
    print("Using 6 timeframes (1min, 3min, 5min, 15min, 30min, 60min) - Daily excluded")
    
    # Rate all stocks
    reports = rating_system.rate_multiple_stocks(symbols)
    
    # Print summary
    rating_system.print_summary_table(reports)
    
    # Save to file
    rating_system.save_ratings_to_file(reports)
    
    print("\nWatchlist rating process completed!")
