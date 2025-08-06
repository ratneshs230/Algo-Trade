"""
Trading Bot Watchlist Configuration
"""

# Define the watchlist of stocks to monitor
WATCHLIST_STOCKS = ["HDFCBANK","ICICIBANK","SBIN","KOTAKBANK","AXISBANK","INDUSINDBK","BAJFINANCE","BAJAJFINSV","PNB","BANKBARODA","SHRIRAMFIN","HDFCAMC","PFC","RECLTD","JIOFIN","TCS","INFY","HCLTECH","WIPRO","TECHM","LTIM","PERSISTENT","COFORGE","TATAMOTORS","MARUTI","M&M","BAJAJ-AUTO","HEROMOTOCO","TVSMOTOR","EICHERMOT","ASHOKLEY","RELIANCE","ONGC","NTPC","POWERGRID","TATAPOWER","ADANIPOWER","IOC","BPCL","COALINDIA","ADANIGREEN","TATASTEEL","JSWSTEEL","HINDALCO","VEDL","SAIL","NATIONALUM","SUNPHARMA","DRREDDY","CIPLA","DIVISLAB","LUPIN","AUROPHARMA","ZYDUSLIFE","HINDUNILVR","ITC","NESTLEIND","BRITANNIA","DABUR","LT","BHARTIARTL","ADANIENT","ADANIPORTS","DLF","INDIGO","TITAN","ASIANPAINT","ULTRACEMCO","GRASIM","HAL","BEL","IRCTC","ZEEL","INDUSTOWER","SIEMENS","PIDILITIND","TRENT","DMART","IEX","BHEL","CANBK","IDFCFIRSTB","CHOLAFIN","SBILIFE","ICICIPRULI","SBICARD","GODREJPROP","ACC","AMBUJACEM","JINDALSTEL","MUTHOOTFIN","VOLTAS","HAVELLS","JUBLFOOD","MCDOWELL-N","CUMMINSIND","TORNTPHARM","BIOCON","GLAND","OFSS","LTTS","TATAELXSI","UPL","COROMANDEL","MANAPPURAM","GMRINFRA","DIXON","POLYCAB","ESCORTS","DEEPAKNTR","AARTIIND","ABB","PVRINOX","OBEROIRLTY","CDSL","BANDHANBNK","BALKRISIND","EXIDEIND","NMDC","TATACOMM","VOLTAS","CONCOR","CHAMBLFERT","GNFC","IBULHSGFIN","RAMCOCEM","METROPOLIS","MPHASIS","RBLBANK","TATACHEM","FEDERALBNK","GAIL","PEL","L&TFH","M_MFIN","MARICO","GODREJCP","IGL","PETRONET","BOSCHLTD","MRF","APOLLOTYRE","BERGEPAINT","DALBHARAT","LAURUSLABS","GRANULES","FORTIS","HINDCOPPER","HINDZINC","NFL","RCF","SUNTV","DELTACORP","INDHOTEL","MOTHERSON","PAGEIND","BATAINDIA","ATUL","NAVINFLUOR","CROMPTON","IRFC","IDEA","YESBANK","INDIACEM","BSOFT","MAZDOCK","COCHINSHIP","RAIN","SHRIRAMPS","GLENMARK","ALKEM","ABCAPITAL","ABFRL","AUBANK","DEEPAKFERT","GSFC","UBL","SUPREMEIND","ASTRAL","KAJARIACER","OFSS","GAIL","PEL","L&TFH","M_MFIN","MARICO","GODREJCP","IGL","PETRONET","BOSCHLTD","MRF","APOLLOTYRE","BERGEPAINT","DALBHARAT","LAURUSLABS","GRANULES","FORTIS","HINDCOPPER","HINDZINC","NFL","RCF","SUNTV","DELTACORP","INDHOTEL","MOTHERSON","PAGEIND","BATAINDIA","ATUL","NAVINFLUOR","CROMPTON","IRFC","IDEA","YESBANK","INDIACEM","BSOFT","MAZDOCK","COCHINSHIP","RAIN","SHRIRAMPS","GLENMARK","ALKEM","ABCAPITAL","ABFRL","AUBANK","DEEPAKFERT","GSFC","UBL","SUPREMEIND","ASTRAL","KAJARIACER","OFSS"]

def get_watchlist():
    """
    Get the current watchlist of stocks
    
    Returns:
        list: List of stock names in the watchlist
    """
    return WATCHLIST_STOCKS.copy()

def get_watchlist_count():
    """
    Get the count of stocks in the watchlist
    
    Returns:
        int: Number of stocks in the watchlist
    """
    return len(WATCHLIST_STOCKS)

def is_in_watchlist(stock_name):
    """
    Check if a stock is in the watchlist
    
    Args:
        stock_name (str): Name of the stock to check
        
    Returns:
        bool: True if stock is in watchlist, False otherwise
    """
    return stock_name in WATCHLIST_STOCKS

def add_to_watchlist(stock_name):
    """
    Add a stock to the watchlist
    
    Args:
        stock_name (str): Name of the stock to add
        
    Returns:
        bool: True if added successfully, False if already exists
    """
    if stock_name not in WATCHLIST_STOCKS:
        WATCHLIST_STOCKS.append(stock_name)
        return True
    return False

def remove_from_watchlist(stock_name):
    """
    Remove a stock from the watchlist
    
    Args:
        stock_name (str): Name of the stock to remove
        
    Returns:
        bool: True if removed successfully, False if not found
    """
    if stock_name in WATCHLIST_STOCKS:
        WATCHLIST_STOCKS.remove(stock_name)
        return True
    return False

def print_watchlist():
    """
    Print the current watchlist in a formatted way
    """
    print(f"\nCurrent Watchlist ({len(WATCHLIST_STOCKS)} stocks):")
    print("=" * 50)
    for i, stock in enumerate(WATCHLIST_STOCKS, 1):
        print(f"{i:2d}. {stock}")
    print("=" * 50)

if __name__ == "__main__":
    # Display the watchlist when run directly
    print_watchlist()
