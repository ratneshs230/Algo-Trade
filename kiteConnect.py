"""
Kite Connect API Login Automation Module
y
This module automates the login process for Kite Connect API, handling:
- Web-based login using Selenium
- Two-factor authentication (2FA/TOTP)
- Access token generation and persistence
- Trading operations through KiteTrader class
"""

import os
import time
import json
from datetime import datetime, date
from urllib.parse import urlparse, parse_qs
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from kiteconnect import KiteConnect
import pyotp
import stat # Import stat module

# Configuration Module
class Config:
    """Secure configuration management for Kite Connect credentials"""
    
    # Load from environment variables or config file
    API_KEY = os.getenv('KITE_API_KEY', '')
    API_SECRET = os.getenv('KITE_API_SECRET', '')
    USER_ID = os.getenv('KITE_USER_ID', '')
    PASSWORD = os.getenv('KITE_PASSWORD', '')
    TOTP_SECRET_KEY = os.getenv('KITE_TOTP_SECRET_KEY', '')
    CHROMEDRIVER_PATH = os.getenv('CHROMEDRIVER_PATH', '') # NEW: Chromedriver path
    
    @classmethod
    def validate(cls):
        """Validate that all required configuration is present"""
        missing = []
        for field in ['API_KEY', 'API_SECRET', 'USER_ID', 'PASSWORD', 'TOTP_SECRET_KEY']:
            if not getattr(cls, field):
                missing.append(field)
        
        if missing:
            raise ValueError(f"Missing required configuration: {', '.join(missing)}")


class KiteLoginAutomation:
    """Handles automated login process for Kite Connect"""
    
    def __init__(self):
        Config.validate()
        self.api_key = Config.API_KEY
        self.api_secret = Config.API_SECRET
        self.user_id = Config.USER_ID
        self.password = Config.PASSWORD
        self.kite = KiteConnect(api_key=self.api_key)
        self.driver = None
        
    def _init_webdriver(self):
        """Initialize Chrome WebDriver with appropriate options and ensure executable permissions"""
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        
        driver_path = None
        
        # NEW: Check for CHROMEDRIVER_PATH environment variable first
        if Config.CHROMEDRIVER_PATH and os.path.exists(Config.CHROMEDRIVER_PATH) and os.access(Config.CHROMEDRIVER_PATH, os.X_OK):
            driver_path = Config.CHROMEDRIVER_PATH
            print(f"Using CHROMEDRIVER_PATH from environment variable: {driver_path}")
        else:
            try:
                # Try to use webdriver-manager to automatically manage ChromeDriver
                driver_path = ChromeDriverManager().install()
                print(f"Initial driver path from webdriver-manager: {driver_path}")
                
                # Fix common issue where webdriver-manager returns wrong file path
                if driver_path and ('THIRD_PARTY' in driver_path or not driver_path.endswith('chromedriver')):
                    print("Detected incorrect driver path, searching for actual chromedriver...")
                    # Look for actual chromedriver executable in the same directory and parent directories
                    import glob
                    driver_dir = os.path.dirname(driver_path)
                    parent_dir = os.path.dirname(driver_dir)
                    
                    # Search in current dir, parent dir, and subdirectories
                    search_dirs = [driver_dir, parent_dir]
                    found_driver = None
                    
                    for search_dir in search_dirs:
                        if os.path.exists(search_dir):
                            # Look for chromedriver files
                            patterns = [
                                os.path.join(search_dir, 'chromedriver'),
                                os.path.join(search_dir, 'chromedriver.exe'),
                                os.path.join(search_dir, '**/chromedriver'),
                                os.path.join(search_dir, '**/chromedriver.exe'),
                            ]
                            
                            for pattern in patterns:
                                matches = glob.glob(pattern, recursive=True)
                                for match in matches:
                                    if (os.path.exists(match) and os.path.isfile(match) and 
                                        'THIRD_PARTY' not in match and 'LICENSE' not in match):
                                        print(f"Found potential chromedriver at: {match}")
                                        found_driver = match
                                        break
                                if found_driver:
                                    break
                        if found_driver:
                            break
                    
                    if found_driver:
                        driver_path = found_driver
                        print(f"Using corrected driver path: {driver_path}")
                    else:
                        print("Could not find valid chromedriver, will try system fallback")
                        driver_path = None
                
                # Ensure the downloaded chromedriver is executable
                if driver_path and os.path.exists(driver_path) and os.path.isfile(driver_path):
                    st = os.stat(driver_path)
                    os.chmod(driver_path, st.st_mode | stat.S_IEXEC)
                    print(f"Set executable permissions for: {driver_path}")
                else:
                    print(f"Warning: Chromedriver not found at {driver_path} after installation attempt.")
                    driver_path = None # Fallback if not found
                    
            except Exception as e:
                print(f"Error with webdriver_manager: {str(e)}. Attempting fallback to system path.")
                driver_path = None # Fallback to system path
                
            if driver_path is None:
                # Fallback to common system paths for chromedriver
                common_paths = [
                    '/usr/bin/chromedriver',
                    '/usr/local/bin/chromedriver',
                    '/opt/google/chrome/chromedriver'
                ]
                for path in common_paths:
                    if os.path.exists(path) and os.access(path, os.X_OK):
                        driver_path = path
                        print(f"Using system chromedriver from: {driver_path}")
                        break
            
        if driver_path is None:
            raise Exception(
                "Chromedriver not found or not executable. "
                "Please ensure Chrome browser is installed and a compatible Chromedriver "
                "is manually downloaded and placed in a system PATH directory (e.g., /usr/local/bin) "
                "or set the CHROMEDRIVER_PATH environment variable to its location."
            )

        service = Service(driver_path)
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        
    def _perform_login(self):
        """Perform initial login with username and password"""
        try:
            # Generate and navigate to login URL
            login_url = self.kite.login_url()
            self.driver.get(login_url)
            
            # Wait for login page to load
            wait = WebDriverWait(self.driver, 10)
            
            # Enter User ID
            userid_field = wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input#userid"))
            )
            userid_field.clear()
            userid_field.send_keys(self.user_id)
            
            # Enter Password
            password_field = wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input#password"))
            )
            password_field.clear()
            password_field.send_keys(self.password)
            
            # Click Login button
            login_button = wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "button[type='submit']"))
            )
            login_button.click()
            
        except Exception as e:
            raise Exception(f"Login failed: {str(e)}")
    
    def _handle_2fa(self):
        """Handle two-factor authentication"""
        try:
            # Wait for 2FA page to load
            wait = WebDriverWait(self.driver, 15)
            
            print("Waiting for 2FA page to load...")
            
            # Wait for TOTP input field
            totp_field = wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input#userid"))
            )
            
            print("2FA field found, generating TOTP code...")
            
            # Generate TOTP code programmatically
            totp = pyotp.TOTP(Config.TOTP_SECRET_KEY)
            totp_code = totp.now()
            
            print(f"Generated TOTP code: {totp_code}")
            print(f"Time remaining for this code: {30 - (int(time.time()) % 30)} seconds")
            
            # Clear field and enter TOTP code
            totp_field.clear()
            time.sleep(0.5)  # Small delay
            totp_field.send_keys(totp_code)
            
            print("TOTP code entered, waiting for submission...")
            
            # Try to find and click submit button if available
            try:
                submit_button = wait.until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "button[type='submit'], .button-orange"))
                )
                submit_button.click()
                print("Submit button clicked")
            except:
                # If no submit button, just wait for automatic submission
                print("No submit button found, waiting for automatic submission...")
                time.sleep(3)
            
            # Wait a bit longer to see if 2FA succeeds
            time.sleep(5)
            
            print(f"Current URL after 2FA attempt: {self.driver.current_url}")
            
        except Exception as e:
            raise Exception(f"2FA authentication failed: {str(e)}")
    
    def _capture_request_token(self):
        """Capture request token from redirect URL"""
        try:
            # Wait for redirect after successful login
            wait = WebDriverWait(self.driver, 45)  # Increased timeout
            
            print("Waiting for redirect after 2FA...")
            
            # Wait until URL contains success parameter or request_token
            def check_redirect(driver):
                current_url = driver.current_url
                print(f"Current URL: {current_url}")
                return "status=success" in current_url or "request_token=" in current_url
            
            wait.until(check_redirect)
            
            # Additional wait to ensure page is fully loaded
            time.sleep(3)
            
            # Get current URL
            current_url = self.driver.current_url
            print(f"Final redirect URL: {current_url}")
            
            # Parse URL to extract request_token
            parsed_url = urlparse(current_url)
            query_params = parse_qs(parsed_url.query)
            
            if 'request_token' not in query_params:
                # Try to find request_token in URL fragment as well
                if '#' in current_url and 'request_token=' in current_url:
                    fragment_params = parse_qs(current_url.split('#')[1])
                    if 'request_token' in fragment_params:
                        request_token = fragment_params['request_token'][0]
                        return request_token
                
                raise ValueError(f"Request token not found in redirect URL: {current_url}")
            
            request_token = query_params['request_token'][0]
            print(f"Successfully captured request token: {request_token[:20]}...")
            return request_token
            
        except Exception as e:
            print(f"Error details - Current URL: {self.driver.current_url if self.driver else 'No driver'}")
            raise Exception(f"Failed to capture request token: {str(e)}")
    
    def _generate_access_token(self, request_token):
        """Generate access token using request token"""
        try:
            # Generate session to get access token
            data = self.kite.generate_session(request_token, api_secret=self.api_secret)
            
            if 'access_token' not in data:
                raise ValueError("Access token not found in session data")
            
            access_token = data['access_token']
            return access_token
            
        except Exception as e:
            raise Exception(f"Failed to generate access token: {str(e)}")
    
    def _load_existing_token(self):
        """Load existing access token if it exists and is valid for today"""
        try:
            token_file = 'access_token.json'
            if os.path.exists(token_file):
                with open(token_file, 'r') as f:
                    token_data = json.load(f)
                
                # Check if token is for today
                token_date = datetime.strptime(token_data.get('date', ''), '%Y-%m-%d').date()
                today = date.today()
                
                if token_date == today:
                    print(f"Found valid access token for today ({today})")
                    return token_data.get('access_token')
                else:
                    print(f"Existing token is from {token_date}, need new token for {today}")
                    return None
            else:
                print("No existing token file found")
                return None
        except Exception as e:
            print(f"Error loading existing token: {str(e)}")
            return None
    
    def _save_access_token(self, access_token):
        """Save access token to file with current date"""
        try:
            token_data = {
                'access_token': access_token,
                'date': date.today().strftime('%Y-%m-%d'),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Save as JSON with date
            with open('access_token.json', 'w') as f:
                json.dump(token_data, f, indent=2)
            
            # Also save as plain text for backward compatibility
            with open('access_token.txt', 'w') as f:
                f.write(access_token)
            
            print(f"\nAccess token saved with date: {token_data['date']}")
            print(f"Files: access_token.json, access_token.txt")
        except Exception as e:
            print(f"Warning: Could not save access token to file: {str(e)}")
    
    def login(self):
        """Main method to perform complete login process"""
        try:
            # Check for existing valid token first
            existing_token = self._load_existing_token()
            if existing_token:
                print("Using existing access token for today")
                return existing_token
            
            print("No valid token found, proceeding with login...")
            
            # Initialize WebDriver
            self._init_webdriver()
            
            # Step 1: Perform initial login
            print("Performing initial login...")
            self._perform_login()
            
            # Step 2: Handle 2FA
            print("Handling two-factor authentication...")
            self._handle_2fa()
            
            # Step 3: Capture request token
            print("Capturing request token...")
            request_token = self._capture_request_token()
            print(f"Request token captured successfully")
            
            # Step 4: Generate access token
            print("Generating access token...")
            access_token = self._generate_access_token(request_token)
            print(f"Access token generated successfully")
            
            # Step 5: Save access token
            self._save_access_token(access_token)
            
            return access_token
            
        except Exception as e:
            print(f"\nLogin process failed: {str(e)}")
            raise
            
        finally:
            # Always close the browser
            if self.driver:
                self.driver.quit()
                print("\nBrowser closed.")


class KiteTrader:
    """Primary interface for all trading-related activities using Kite Connect API"""
    
    def __init__(self, api_key, access_token):
        """
        Initialize KiteTrader with API credentials
        
        Args:
            api_key (str): Kite Connect API key
            access_token (str): Valid access token obtained from login
        """
        self.api_key = api_key
        self.access_token = access_token
        self.kite = KiteConnect(api_key=api_key)
        self.kite.set_access_token(access_token)
        
    # Market Data Methods
    
    def get_instruments(self, exchange=None):
        """
        Fetches and returns the full list of tradable instruments
        
        Args:
            exchange (str, optional): Exchange code (e.g., 'NSE', 'NFO', 'BSE', 'MCX')
        
        Returns:
            list: List of instrument dictionaries containing tradingsymbol, exchange_token, etc.
        """
        try:
            if exchange:
                instruments = self.kite.instruments(exchange)
            else:
                instruments = self.kite.instruments()
            return instruments
        except Exception as e:
            print(f"Error fetching instruments: {str(e)}")
            return []
    
    def get_quote(self, instruments):
        """
        Retrieves full quote data including market depth for given instruments
        
        Args:
            instruments (list): List of instrument identifiers in format EXCHANGE:TRADINGSYMBOL
        
        Returns:
            dict: Dictionary with instrument identifiers as keys and quote data as values
        """
        try:
            quotes = self.kite.quote(instruments)
            return quotes
        except Exception as e:
            print(f"Error fetching quotes: {str(e)}")
            return {}
    
    def get_ltp(self, instruments):
        """
        Retrieves last traded price for given instruments
        
        Args:
            instruments (list): List of instrument identifiers in format EXCHANGE:TRADINGSYMBOL
        
        Returns:
            dict: Dictionary with instrument identifiers as keys and LTP data as values
        """
        try:
            ltp_data = self.kite.ltp(instruments)
            return ltp_data
        except Exception as e:
            print(f"Error fetching LTP: {str(e)}")
            return {}
    
    def get_historical_data(self, instrument_token, from_date, to_date, interval):
        """
        Fetches historical OHLC data for an instrument
        
        Args:
            instrument_token (int): Instrument token
            from_date (datetime): Start date for historical data
            to_date (datetime): End date for historical data
            interval (str): Candle interval (minute, day, 3minute, 5minute, 10minute, 15minute, 30minute, 60minute)
        
        Returns:
            list: List of historical data candles with date, open, high, low, close, volume
        """
        try:
            historical_data = self.kite.historical_data(
                instrument_token=instrument_token,
                from_date=from_date,
                to_date=to_date,
                interval=interval
            )
            return historical_data
        except Exception as e:
            print(f"Error fetching historical data: {str(e)}")
            return []
    
    # Order Management Methods
    
    def place_limit_order(self, exchange, tradingsymbol, transaction_type, quantity, price):
        """
        Places a regular limit order
        
        Args:
            exchange (str): Exchange (NSE, BSE, NFO, MCX, etc.)
            tradingsymbol (str): Trading symbol of the instrument
            transaction_type (str): BUY or SELL
            quantity (int): Quantity to trade
            price (float): Limit price
        
        Returns:
            dict: Order response containing order_id
        """
        try:
            order_id = self.kite.place_order(
                variety="regular",
                exchange=exchange,
                tradingsymbol=tradingsymbol,
                transaction_type=transaction_type,
                quantity=quantity,
                product="MIS",
                order_type="LIMIT",
                price=price,
                validity="DAY"
            )
            print(f"Limit order placed successfully. Order ID: {order_id}")
            print(f"  {transaction_type} {quantity} shares of {tradingsymbol} at ₹{price:.2f}")
            return {"order_id": order_id, "status": "success"}
        except Exception as e:
            print(f"Error placing limit order: {str(e)}")
            return {"order_id": None, "status": "failed", "error": str(e)}
    
    def place_order(self, variety, exchange, tradingsymbol, transaction_type, 
                   quantity, product, order_type, price=None, trigger_price=None,
                   validity=None, tag=None):
        """
        Places an order with specified parameters
        
        Args:
            variety (str): Order variety (regular, bo, co, amo)
            exchange (str): Exchange (NSE, BSE, NFO, MCX, etc.)
            tradingsymbol (str): Trading symbol of the instrument
            transaction_type (str): BUY or SELL
            quantity (int): Quantity to trade
            product (str): Product type (CNC, NRML, MIS)
            order_type (str): Order type (MARKET, LIMIT, SL, SL-M)
            price (float, optional): Order price for LIMIT orders
            trigger_price (float, optional): Trigger price for SL orders
            validity (str, optional): Order validity (DAY, IOC, TTL)
            tag (str, optional): Optional order tag for tracking
        
        Returns:
            dict: Order response containing order_id
        """
        try:
            order_id = self.kite.place_order(
                variety=variety,
                exchange=exchange,
                tradingsymbol=tradingsymbol,
                transaction_type=transaction_type,
                quantity=quantity,
                product=product,
                order_type=order_type,
                price=price,
                trigger_price=trigger_price,
                validity=validity,
                tag=tag
            )
            print(f"Order placed successfully. Order ID: {order_id}")
            return {"order_id": order_id, "status": "success"}
        except Exception as e:
            print(f"Error placing order: {str(e)}")
            return {"order_id": None, "status": "failed", "error": str(e)}
    
    def place_bracket_order(self, exchange, tradingsymbol, transaction_type,
                           quantity, price, stoploss, target, trailing_stoploss=None):
        """
        Places a Bracket Order (BO) with profit target and stoploss
        
        Args:
            exchange (str): Exchange (NSE, BSE, NFO, MCX, etc.)
            tradingsymbol (str): Trading symbol of the instrument
            transaction_type (str): BUY or SELL
            quantity (int): Quantity to trade
            price (float): Entry price
            stoploss (float): Absolute stoploss value
            target (float): Absolute target value
            trailing_stoploss (float, optional): Trailing stoploss value
        
        Returns:
            dict: Order response containing order_id
        """
        try:
            # Calculate stoploss and target prices relative to entry price
            if transaction_type == "BUY":
                stoploss_price = price - abs(price - stoploss)
                target_price = price + abs(target - price)
            else:  # SELL
                stoploss_price = price + abs(price - stoploss)
                target_price = price - abs(target - price)
            
            order_id = self.kite.place_order(
                variety="bo",  # Use string instead of constant
                exchange=exchange,
                tradingsymbol=tradingsymbol,
                transaction_type=transaction_type,
                quantity=quantity,
                product="MIS",  # Use string instead of constant
                order_type="LIMIT",  # Use string instead of constant
                price=price,
                stoploss=stoploss_price,
                squareoff=target_price,
                trailing_stoploss=trailing_stoploss
            )
            print(f"Bracket order placed successfully. Order ID: {order_id}")
            return {"order_id": order_id, "status": "success"}
        except Exception as e:
            print(f"Error placing bracket order: {str(e)}")
            print(f"Note: Bracket orders may not be available. Falling back to regular order with manual stop-loss management.")
            # Fallback to regular order
            return self._place_regular_order_with_manual_sl(
                exchange, tradingsymbol, transaction_type, quantity, price, stoploss, target
            )
    
    def place_cover_order(self, exchange, tradingsymbol, transaction_type,
                         quantity, price, trigger_price):
        """
        Places a Cover Order (CO) with a compulsory stop loss
        
        Args:
            exchange (str): Exchange (NSE, BSE, NFO, MCX, etc.)
            tradingsymbol (str): Trading symbol of the instrument
            transaction_type (str): BUY or SELL
            quantity (int): Quantity to trade
            price (float): Limit price (0 for market order)
            trigger_price (float): Stop loss trigger price
        
        Returns:
            dict: Order response containing order_id
        """
        try:
            order_type = "MARKET" if price == 0 else "LIMIT"
            order_id = self.kite.place_order(
                variety="co",  # Use string instead of constant
                exchange=exchange,
                tradingsymbol=tradingsymbol,
                transaction_type=transaction_type,
                quantity=quantity,
                product="MIS",  # Use string instead of constant
                order_type=order_type,
                price=price if price != 0 else None,
                trigger_price=trigger_price
            )
            print(f"Cover order placed successfully. Order ID: {order_id}")
            return {"order_id": order_id, "status": "success"}
        except Exception as e:
            print(f"Error placing cover order: {str(e)}")
            return {"order_id": None, "status": "failed", "error": str(e)}
    
    def modify_order(self, variety, order_id, parent_order_id=None, quantity=None,
                    price=None, order_type=None, trigger_price=None, validity=None):
        """
        Modifies a pending order
        
        Args:
            variety (str): Order variety (regular, bo, co, amo)
            order_id (str): Order ID to modify
            parent_order_id (str, optional): Parent order ID for BO/CO orders
            quantity (int, optional): New quantity
            price (float, optional): New price
            order_type (str, optional): New order type
            trigger_price (float, optional): New trigger price
            validity (str, optional): New validity
        
        Returns:
            dict: Modification response
        """
        try:
            self.kite.modify_order(
                variety=variety,
                order_id=order_id,
                parent_order_id=parent_order_id,
                quantity=quantity,
                price=price,
                order_type=order_type,
                trigger_price=trigger_price,
                validity=validity
            )
            print(f"Order {order_id} modified successfully")
            return {"status": "success", "order_id": order_id}
        except Exception as e:
            print(f"Error modifying order: {str(e)}")
            return {"status": "failed", "error": str(e)}
    
    def cancel_order(self, variety, order_id, parent_order_id=None):
        """
        Cancels a pending order
        
        Args:
            variety (str): Order variety (regular, bo, co, amo)
            order_id (str): Order ID to cancel
            parent_order_id (str, optional): Parent order ID for BO/CO orders
        
        Returns:
            dict: Cancellation response
        """
        try:
            self.kite.cancel_order(
                variety=variety,
                order_id=order_id,
                parent_order_id=parent_order_id
            )
            print(f"Order {order_id} cancelled successfully")
            return {"status": "success", "order_id": order_id}
        except Exception as e:
            print(f"Error cancelling order: {str(e)}")
            return {"status": "failed", "error": str(e)}
    
    # Position & Fund Management Methods
    
    def get_positions(self):
        """
        Retrieves net positions for the day
        
        Returns:
            dict: Dictionary containing 'day' and 'net' positions
        """
        try:
            positions = self.kite.positions()
            return positions
        except Exception as e:
            print(f"Error fetching positions: {str(e)}")
            return {"day": [], "net": []}
    
    def get_holdings(self):
        """
        Retrieves user's holdings (long-term positions)
        
        Returns:
            list: List of holding dictionaries containing quantity, average_price, etc.
        """
        try:
            holdings = self.kite.holdings()
            return holdings
        except Exception as e:
            print(f"Error fetching holdings: {str(e)}")
            return []
    
    def get_margins(self):
        """
        Retrieves available margins in equity and commodity segments
        
        Returns:
            dict: Dictionary containing equity and commodity margin details
        """
        try:
            margins = self.kite.margins()
            return margins
        except Exception as e:
            print(f"Error fetching margins: {str(e)}")
            return {}


def main():
    """Main function to execute Kite Connect login automation"""
    print("Kite Connect Login Automation")
    print("="*50)
    
    try:
        # Create automation instance and perform login
        automation = KiteLoginAutomation()
        access_token = automation.login()
        
        print("\n" + "="*50)
        print("Login successful!")
        print(f"Access Token: {access_token[:20]}..." if len(access_token) > 20 else f"Access Token: {access_token}")
        print("="*50)
        
        # Instantiate KiteTrader with the obtained access token
        trader = KiteTrader(api_key=Config.API_KEY, access_token=access_token)
        
        # Example usage of KiteTrader functions (commented out)
        # Uncomment the examples below to test the functionality
        
        # # Get available margins
        # margins = trader.get_margins()
        # print("\nAvailable Margins:")
        # print(f"Equity: {margins.get('equity', {})}")
        # print(f"Commodity: {margins.get('commodity', {})}")
        
        # # Get current positions
        # positions = trader.get_positions()
        # print("\nCurrent Positions:")
        # print(f"Day positions: {len(positions.get('day', []))}")
        # print(f"Net positions: {len(positions.get('net', []))}")
        
        # # Get holdings
        # holdings = trader.get_holdings()
        # print(f"\nTotal Holdings: {len(holdings)}")
        
        # # Get LTP for some instruments
        # instruments = ["NSE:RELIANCE", "NSE:INFY"]
        # ltp_data = trader.get_ltp(instruments)
        # print("\nLast Traded Prices:")
        # for instrument, data in ltp_data.items():
        #     print(f"{instrument}: ₹{data.get('last_price', 0)}")
        
        # # Place a sample order (DO NOT UNCOMMENT unless you want to place a real order)
        # # order_response = trader.place_order(
        # #     variety="regular",
        # #     exchange="NSE",
        # #     tradingsymbol="RELIANCE",
        # #     transaction_type="BUY",
        # #     quantity=1,
        # #     product="CNC",
        # #     order_type="LIMIT",
        # #     price=2400.00
        # # )
        # # print(f"\nOrder Response: {order_response}")
        
        return access_token
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        return None


if __name__ == "__main__":
    main()
