"""
Technical Indicator Calculator for Stock Rating System

This module calculates various technical indicators used in the rating system:
- EMA Crossover (20, 50)
- Price vs MA (200)
- MACD (12, 26, 9)
- ADX (14)
- RSI (14)
- Stochastic (14, 3, 3)
- Bollinger Bands (20, 2)
- On-Balance Volume (OBV)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import math


class IndicatorCalculator:
    """Calculate technical indicators for stock rating system"""
    
    def __init__(self):
        self.indicators = {}
    
    def calculate_ema(self, data: List[float], period: int) -> List[float]:
        """Calculate Exponential Moving Average"""
        if len(data) < period:
            return [np.nan] * len(data)
        
        ema = [np.nan] * len(data)
        multiplier = 2 / (period + 1)
        
        # Start with SMA for first value
        ema[period - 1] = sum(data[:period]) / period
        
        for i in range(period, len(data)):
            ema[i] = (data[i] * multiplier) + (ema[i - 1] * (1 - multiplier))
        
        return ema
    
    def calculate_sma(self, data: List[float], period: int) -> List[float]:
        """Calculate Simple Moving Average"""
        if len(data) < period:
            return [np.nan] * len(data)
        
        sma = [np.nan] * len(data)
        for i in range(period - 1, len(data)):
            sma[i] = sum(data[i - period + 1:i + 1]) / period
        
        return sma
    
    def calculate_atr(self, high: List[float], low: List[float], close: List[float], period: int = 14) -> List[float]:
        """Calculate Average True Range"""
        if len(high) < 2:
            return [np.nan] * len(high)
        
        true_ranges = [np.nan]
        for i in range(1, len(high)):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i - 1])
            tr3 = abs(low[i] - close[i - 1])
            true_ranges.append(max(tr1, tr2, tr3))
        
        return self.calculate_sma(true_ranges, period)
    
    def calculate_ema_crossover_score(self, close: List[float]) -> float:
        """Calculate EMA Crossover (20, 50) score"""
        if len(close) < 50:
            return 0.0
        
        ema20 = self.calculate_ema(close, 20)
        ema50 = self.calculate_ema(close, 50)
        
        if np.isnan(ema20[-1]) or np.isnan(ema50[-1]):
            return 0.0
        
        # Check for crossover in last few periods
        current_diff = ema20[-1] - ema50[-1]
        prev_diff = ema20[-2] - ema50[-2] if len(ema20) > 1 else 0
        
        # Bullish crossover
        if prev_diff <= 0 and current_diff > 0:
            return 1.0
        # Bearish crossover
        elif prev_diff >= 0 and current_diff < 0:
            return -1.0
        # Maintain direction with decay
        elif current_diff > 0:
            return min(0.5, current_diff / abs(current_diff) * 0.5)
        elif current_diff < 0:
            return max(-0.5, current_diff / abs(current_diff) * -0.5)
        
        return 0.0
    
    def calculate_price_vs_ma200_score(self, close: List[float], high: List[float], low: List[float]) -> float:
        """Calculate Price vs MA(200) score"""
        if len(close) < 200:
            return 0.0
        
        ma200 = self.calculate_sma(close, 200)
        atr = self.calculate_atr(high, low, close, 14)
        
        if np.isnan(ma200[-1]) or np.isnan(atr[-1]) or atr[-1] == 0:
            return 0.0
        
        # Normalize by ATR and use tanh
        normalized_distance = (close[-1] - ma200[-1]) / atr[-1]
        return math.tanh(normalized_distance)
    
    def calculate_macd_score(self, close: List[float]) -> float:
        """Calculate MACD (12, 26, 9) composite score"""
        if len(close) < 35:
            return 0.0
        
        ema12 = self.calculate_ema(close, 12)
        ema26 = self.calculate_ema(close, 26)
        
        if np.isnan(ema12[-1]) or np.isnan(ema26[-1]):
            return 0.0
        
        # MACD line
        macd_line = [ema12[i] - ema26[i] for i in range(len(ema12))]
        
        # Signal line (EMA of MACD)
        signal_line = self.calculate_ema(macd_line, 9)
        
        if np.isnan(signal_line[-1]):
            return 0.0
        
        # Histogram
        histogram = macd_line[-1] - signal_line[-1]
        
        # Composite score
        # 1. Histogram momentum
        hist_score = math.tanh(histogram / max(abs(histogram), 0.001))
        
        # 2. Signal line crossover
        crossover_score = 0.0
        if len(signal_line) > 1:
            if macd_line[-2] <= signal_line[-2] and macd_line[-1] > signal_line[-1]:
                crossover_score = 0.5  # Bullish crossover
            elif macd_line[-2] >= signal_line[-2] and macd_line[-1] < signal_line[-1]:
                crossover_score = -0.5  # Bearish crossover
        
        # 3. Zero line position
        zero_score = 0.3 if macd_line[-1] > 0 else -0.3
        
        return max(-1.0, min(1.0, hist_score * 0.5 + crossover_score + zero_score * 0.2))
    
    def calculate_adx_score(self, high: List[float], low: List[float], close: List[float], period: int = 14) -> float:
        """Calculate ADX (14) score with directional movement"""
        if len(high) < period + 1:
            return 0.0
        
        # Calculate True Range and Directional Movement
        tr_list = []
        plus_dm = []
        minus_dm = []
        
        for i in range(1, len(high)):
            # True Range
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i - 1])
            tr3 = abs(low[i] - close[i - 1])
            tr_list.append(max(tr1, tr2, tr3))
            
            # Directional Movement
            up_move = high[i] - high[i - 1]
            down_move = low[i - 1] - low[i]
            
            plus_dm.append(up_move if up_move > down_move and up_move > 0 else 0)
            minus_dm.append(down_move if down_move > up_move and down_move > 0 else 0)
        
        if len(tr_list) < period:
            return 0.0
        
        # Smooth the values
        tr_smooth = sum(tr_list[-period:]) / period
        plus_dm_smooth = sum(plus_dm[-period:]) / period
        minus_dm_smooth = sum(minus_dm[-period:]) / period
        
        if tr_smooth == 0:
            return 0.0
        
        # Calculate DI
        plus_di = (plus_dm_smooth / tr_smooth) * 100
        minus_di = (minus_dm_smooth / tr_smooth) * 100
        
        # Calculate ADX
        dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100 if (plus_di + minus_di) > 0 else 0
        
        # Simplified ADX (normally would need more periods)
        adx = dx
        
        # Direction score
        direction_score = (plus_di - minus_di) / 100
        
        # Strength multiplier (ADX/100)
        strength = min(adx / 100, 1.0)
        
        return max(-1.0, min(1.0, direction_score * strength))
    
    def calculate_rsi_score(self, close: List[float], period: int = 14) -> float:
        """Calculate RSI (14) composite score"""
        if len(close) < period + 1:
            return 0.0
        
        gains = []
        losses = []
        
        for i in range(1, len(close)):
            change = close[i] - close[i - 1]
            gains.append(max(change, 0))
            losses.append(max(-change, 0))
        
        if len(gains) < period:
            return 0.0
        
        # Average gains and losses
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        # Momentum score (distance from 50)
        momentum_score = (rsi - 50) / 50
        
        # Reversal score (from extreme zones)
        reversal_score = 0.0
        if len(close) > 1:
            prev_rsi_approx = 50  # Simplified
            if prev_rsi_approx > 70 and rsi < 70:
                reversal_score = -0.5  # Reversal from overbought
            elif prev_rsi_approx < 30 and rsi > 30:
                reversal_score = 0.5   # Reversal from oversold
        
        return max(-1.0, min(1.0, momentum_score * 0.7 + reversal_score * 0.3))
    
    def calculate_stochastic_score(self, high: List[float], low: List[float], close: List[float], 
                                 k_period: int = 14, d_period: int = 3) -> float:
        """Calculate Stochastic (14, 3, 3) score"""
        if len(high) < k_period:
            return 0.0
        
        # Calculate %K
        k_values = []
        for i in range(k_period - 1, len(high)):
            highest_high = max(high[i - k_period + 1:i + 1])
            lowest_low = min(low[i - k_period + 1:i + 1])
            
            if highest_high == lowest_low:
                k_values.append(50)
            else:
                k = ((close[i] - lowest_low) / (highest_high - lowest_low)) * 100
                k_values.append(k)
        
        if len(k_values) < d_period:
            return 0.0
        
        # Calculate %D (SMA of %K)
        d_values = []
        for i in range(d_period - 1, len(k_values)):
            d = sum(k_values[i - d_period + 1:i + 1]) / d_period
            d_values.append(d)
        
        if len(d_values) < 2:
            return 0.0
        
        k_current = k_values[-1]
        d_current = d_values[-1]
        k_prev = k_values[-2] if len(k_values) > 1 else k_current
        d_prev = d_values[-2] if len(d_values) > 1 else d_current
        
        # Strong signals in extreme zones
        if k_current < 20 and d_current < 20:
            if k_prev <= d_prev and k_current > d_current:
                return 1.0  # Strong buy signal
        elif k_current > 80 and d_current > 80:
            if k_prev >= d_prev and k_current < d_current:
                return -1.0  # Strong sell signal
        
        return 0.0
    
    def calculate_bollinger_bands_score(self, close: List[float], period: int = 20, std_dev: float = 2.0) -> float:
        """Calculate Bollinger Bands (20, 2) score"""
        if len(close) < period:
            return 0.0
        
        sma = self.calculate_sma(close, period)
        
        if np.isnan(sma[-1]):
            return 0.0
        
        # Calculate standard deviation
        recent_closes = close[-period:]
        variance = sum([(x - sma[-1]) ** 2 for x in recent_closes]) / period
        std = math.sqrt(variance)
        
        upper_band = sma[-1] + (std_dev * std)
        lower_band = sma[-1] - (std_dev * std)
        
        # Check for "walking the bands"
        if len(close) >= 3:
            # Walking upper band (bullish)
            if close[-1] > upper_band and close[-2] > upper_band:
                return 1.0
            # Walking lower band (bearish)
            elif close[-1] < lower_band and close[-2] < lower_band:
                return -1.0
        
        return 0.0
    
    def calculate_obv_score(self, close: List[float], volume: List[float]) -> float:
        """Calculate On-Balance Volume (OBV) score"""
        if len(close) < 10 or len(volume) < 10:
            return 0.0
        
        obv = [volume[0]]
        for i in range(1, len(close)):
            if close[i] > close[i - 1]:
                obv.append(obv[-1] + volume[i])
            elif close[i] < close[i - 1]:
                obv.append(obv[-1] - volume[i])
            else:
                obv.append(obv[-1])
        
        # Check trend confirmation
        price_trend = 1 if close[-1] > close[-10] else -1
        obv_trend = 1 if obv[-1] > obv[-10] else -1
        
        if price_trend == obv_trend:
            return float(price_trend)  # Confirmation
        else:
            return 0.0  # Divergence
    
    def calculate_all_indicators(self, ohlcv_data: Dict) -> Dict[str, float]:
        """Calculate all indicators and return scores"""
        high = ohlcv_data.get('high', [])
        low = ohlcv_data.get('low', [])
        close = ohlcv_data.get('close', [])
        volume = ohlcv_data.get('volume', [])
        
        if not all([high, low, close, volume]) or len(close) < 50:
            return {indicator: 0.0 for indicator in [
                'ema_crossover', 'price_vs_ma200', 'macd', 'adx', 
                'rsi', 'stochastic', 'bollinger_bands', 'obv'
            ]}
        
        scores = {
            'ema_crossover': self.calculate_ema_crossover_score(close),
            'price_vs_ma200': self.calculate_price_vs_ma200_score(close, high, low),
            'macd': self.calculate_macd_score(close),
            'adx': self.calculate_adx_score(high, low, close),
            'rsi': self.calculate_rsi_score(close),
            'stochastic': self.calculate_stochastic_score(high, low, close),
            'bollinger_bands': self.calculate_bollinger_bands_score(close),
            'obv': self.calculate_obv_score(close, volume)
        }
        
        return scores


def extract_ohlcv_from_historical_data(historical_data: List[Dict]) -> Dict[str, List[float]]:
    """Extract OHLCV data from historical data format"""
    if not historical_data:
        return {'high': [], 'low': [], 'close': [], 'volume': [], 'open': []}
    
    ohlcv = {
        'open': [],
        'high': [],
        'low': [],
        'close': [],
        'volume': []
    }
    
    for candle in historical_data:
        ohlcv['open'].append(float(candle.get('open', 0)))
        ohlcv['high'].append(float(candle.get('high', 0)))
        ohlcv['low'].append(float(candle.get('low', 0)))
        ohlcv['close'].append(float(candle.get('close', 0)))
        ohlcv['volume'].append(float(candle.get('volume', 0)))
    
    return ohlcv


if __name__ == "__main__":
    # Test the indicator calculator
    calculator = IndicatorCalculator()
    
    # Sample data for testing
    sample_data = {
        'high': [100 + i + np.random.random() for i in range(200)],
        'low': [99 + i + np.random.random() for i in range(200)],
        'close': [99.5 + i + np.random.random() for i in range(200)],
        'volume': [1000 + np.random.randint(0, 500) for _ in range(200)]
    }
    
    scores = calculator.calculate_all_indicators(sample_data)
    print("Sample Indicator Scores:")
    for indicator, score in scores.items():
        print(f"{indicator}: {score:.3f}")
