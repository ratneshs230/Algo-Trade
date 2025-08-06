# Automated Single Trade Bot - Precision Trading Strategy

## Overview

This document describes the sophisticated single-trade-at-a-time automated trading strategy implemented in `automated_single_trade_bot.py`. The strategy focuses on finding the single best trading opportunity at any moment and managing it through multiple phases with dynamic risk management.

## Strategy Philosophy

**"One Trade, Maximum Precision, Dynamic Risk Management"**

Rather than managing multiple positions with divided attention and capital, this strategy concentrates all resources on the single highest-probability opportunity, ensuring maximum focus and capital efficiency.

## 🔄 Complete Trading Workflow

### Phase 1: Opportunity Identification (Market Scanner)

**Objective**: Find the single best trading candidate from the entire watchlist

**Process**:
1. **Holistic Rating Analysis**
   - Analyze every stock in `watchlist.py`
   - Generate composite ratings using multiple technical indicators
   - Consider multiple timeframes (1min, 3min, 5min, 15min, 30min, 60min, daily)
   - Each stock gets a final score ranging from -10 (strong sell) to +10 (strong buy)

2. **Absolute Magnitude Selection**
   - Sort all stocks by **absolute rating** (not directional rating)
   - Select the stock with the highest absolute value
   - Example: Stock A (+7.8) vs Stock B (-9.2) → Select Stock B for short trade
   - This ensures we always trade the strongest signal regardless of direction

3. **Direction Determination**
   - Positive rating → Long position (BUY)
   - Negative rating → Short position (SELL)

**Key Insight**: We trade magnitude of conviction, not direction bias

### Phase 2: Entry Confirmation (Precision Trigger)

**Objective**: Validate the opportunity with dual technical confirmation

**Requirements**: Both conditions must be met simultaneously on 1-minute chart:

**Indicator 1: Supertrend (Primary Trend Filter)**
- ATR Period: 5
- Factor: 3.0
- **Long Entry**: Current price > Supertrend line
- **Short Entry**: Current price < Supertrend line

**Indicator 2: Parabolic SAR (Momentum Confirmation)**
- Acceleration: 0.02
- Maximum: 0.2
- **Long Entry**: PSAR dots below price candles (bullish momentum)
- **Short Entry**: PSAR dots above price candles (bearish momentum)

**Confirmation Logic**:
```
Long Trade = (Price > Supertrend) AND (PSAR is bullish)
Short Trade = (Price < Supertrend) AND (PSAR is bearish)
```

**Timeout**: If confirmation is not achieved within 5 minutes, abort and return to scanning

### Phase 3: Trade Execution (Full Commitment)

**Position Sizing**:
- **Full Margin Allocation**: Use entire available margin for single trade
- **Quantity Calculation**: `Available Margin ÷ Stock Price = Quantity`
- **Product Type**: MIS (Intraday) for maximum leverage

**Transaction Cost Calculation**:
- Brokerage: Min(₹20, 0.01% of trade value)
- STT: 0.025% on delivery trades
- Exchange Charges: ~0.00345%
- GST: 18% on (Brokerage + Exchange charges)
- SEBI Charges: 0.0001%

**Breakeven Price Calculation**:
```
Total Charges = Brokerage + STT + Exchange + GST + SEBI
Breakeven (Long) = Entry Price + (Total Charges × 2 ÷ Quantity) + Tick Size
Breakeven (Short) = Entry Price - (Total Charges × 2 ÷ Quantity) - Tick Size
```

**Initial Stop Loss**:
- Set at exact Supertrend level at time of entry
- This provides a logical technical level for risk management

### Phase 4: Three-Phase Risk Management

The trade progresses through three distinct phases, each with specific objectives and management rules:

#### Phase A: Initial Risk Phase

**Objective**: Survive the initial risk and reach breakeven

**Management**:
- Stop Loss: Fixed at Supertrend level
- Target: Precise breakeven price (including all transaction costs)
- Action: Monitor price movement toward breakeven target
- Risk: Full position risk until breakeven is achieved

**Transition Trigger**: Price touches breakeven target

#### Phase B: Risk-Free Transition Phase

**Objective**: Eliminate position risk and set up profit maximization

**Management**:
- **Immediate Action**: Move stop loss to entry price (breakeven)
- **Risk Status**: Position is now risk-free (worst case = ₹0 P&L)
- **Fibonacci Calculation**: Calculate retracement/extension levels based on last 50 one-minute candles
- **Target Setup**: Identify first Fibonacci target in trade direction

**Duration**: Very brief transition phase

#### Phase C: Profit Ladder Phase

**Objective**: Maximize profit through Fibonacci-based trailing stops

**The "Climb and Lock" Mechanism**:

1. **No Take-Profit Orders**: Never place target orders (prevents premature exits)
2. **Trailing Stop Logic**: All exits happen via stop-loss triggers
3. **Fibonacci Progression**:
   ```
   Price hits Fib Level 0.236 → Move stop to 0.236 level → Target becomes 0.382
   Price hits Fib Level 0.382 → Move stop to 0.382 level → Target becomes 0.5
   Price hits Fib Level 0.5   → Move stop to 0.5 level   → Target becomes 0.618
   ... and so on
   ```

**Fibonacci Levels Used**:
- **Retracements**: 0.236, 0.382, 0.5, 0.618, 0.786
- **Extensions**: 1.272, 1.618, 2.0, 2.618

**Level Selection by Direction**:
- **Long Trades**: Use retracement levels above entry price
- **Short Trades**: Use extension levels below entry price

### Phase 5: Trade Completion and Reset

**Exit Triggers**:
- Stop loss hit at any phase
- Manual intervention (emergency stop)
- Daily limits reached

**Cleanup Process**:
1. Cancel any pending stop-loss orders
2. Calculate final realized P&L
3. Update daily statistics
4. Reset bot state to scanning mode
5. Archive trade data for analysis

**Immediate Action**: Return to Phase 1 (Market Scanner) for next opportunity

## 🛡️ Risk Management Framework

### Daily Limits
- **Maximum Trades**: 3 per day (prevents overtrading)
- **Maximum Loss**: 2% of capital (circuit breaker)
- **Reset Time**: 9:15 AM (market open) each trading day

### Position Limits
- **Single Position Only**: Never more than one active trade
- **Full Capital Commitment**: No position sizing - all or nothing approach
- **Intraday Only**: All positions must be squared off by market close

### Emergency Controls
- **Manual Stop**: Ctrl+C for graceful shutdown
- **Error Handling**: Automatic recovery and state management
- **Connection Loss**: Retry mechanisms with exponential backoff

## 🚀 Getting Started

### Prerequisites
1. Valid Zerodha Kite Connect API credentials
2. Active trading account with sufficient margin
3. Access token generated for the trading day
4. Watchlist configured in `watchlist.py`

### Starting the Bot
```bash
python run_single_trade_bot.py
```

### Monitoring
The bot provides real-time status updates including:
- Current state (Scanning/Confirming/Phase A/B/C)
- Active trade details
- Daily trade count and P&L
- Current candidate being evaluated

### Stopping the Bot
- **Graceful Stop**: Press Ctrl+C
- **Emergency Stop**: Close terminal (not recommended if trade is active)

## ⚠️ Important Considerations

### Risk Factors
1. **Full Margin Usage**: Each trade uses entire available capital
2. **Automated Execution**: No manual approval for trades once started
3. **Market Risk**: Subject to sudden market movements and gaps
4. **Technical Risk**: Dependent on API connectivity and data quality

This strategy represents a sophisticated approach to automated trading that combines technical analysis, risk management, and systematic execution for consistent performance in the Indian equity markets.
