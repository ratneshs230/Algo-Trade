"""
Trading Dashboard Web Application

Flask-based web dashboard for the algorithmic trading system with 
Kite Connect integration and access token management.
"""

import os
import json
import threading
import time
from datetime import datetime, date
from flask import Flask, render_template, request, jsonify, session
from kiteConnect import KiteLoginAutomation, Config
import logging
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-in-production'

# Global variables for managing login process
login_automation = None
login_thread = None
login_status = {
    'status': 'idle',  # idle, starting, waiting_for_totp, processing, success, error
    'message': '',
    'access_token': None,
    'error': None,
    'request_token': None
}

class DashboardKiteLogin:
    """Modified KiteLoginAutomation for dashboard use"""
    
    def __init__(self):
        self.automation = None
        self.driver = None
        
    def start_login_process(self):
        """Start the login process and return request token"""
        try:
            global login_status
            login_status['status'] = 'starting'
            login_status['message'] = 'Initializing login process...'
            
            # Check for existing valid token first
            existing_token = self._load_existing_token()
            if existing_token:
                login_status['status'] = 'success'
                login_status['message'] = 'Using existing valid token for today'
                login_status['access_token'] = existing_token
                return {'success': True, 'access_token': existing_token}
            
            login_status['message'] = 'No valid token found, starting new login...'
            
            # Initialize the automation
            self.automation = KiteLoginAutomation()
            self.automation._init_webdriver()
            
            # Perform initial login
            login_status['message'] = 'Performing initial login with credentials...'
            self.automation._perform_login()
            
            # Wait for 2FA page and get request token
            login_status['status'] = 'waiting_for_totp'
            login_status['message'] = 'Login successful. Please enter TOTP code in the dashboard.'
            
            # Wait for 2FA completion (will be handled by complete_login_with_totp)
            return {'success': True, 'waiting_for_totp': True}
            
        except Exception as e:
            login_status['status'] = 'error'
            login_status['message'] = f'Login failed: {str(e)}'
            login_status['error'] = str(e)
            if self.automation and self.automation.driver:
                self.automation.driver.quit()
            return {'success': False, 'error': str(e)}
    
    def complete_login_with_totp(self, totp_code):
        """Complete login process with TOTP code"""
        try:
            global login_status
            login_status['status'] = 'processing'
            login_status['message'] = 'Processing TOTP code...'
            
            if not self.automation or not self.automation.driver:
                raise Exception("Login process not initialized")
            
            # Handle 2FA with provided TOTP
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            
            # Validate TOTP format
            if not totp_code.isdigit() or len(totp_code) != 6:
                raise ValueError("Invalid TOTP code. Must be 6 digits.")
            
            # Enter TOTP
            wait = WebDriverWait(self.automation.driver, 10)
            totp_field = wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input#userid"))
            )
            totp_field.clear()
            totp_field.send_keys(totp_code)
            
            # Wait for automatic submission
            time.sleep(2)
            
            login_status['message'] = 'Capturing request token...'
            
            # Capture request token
            request_token = self.automation._capture_request_token()
            
            login_status['message'] = 'Generating access token...'
            
            # Generate access token
            access_token = self.automation._generate_access_token(request_token)
            
            login_status['message'] = 'Saving access token...'
            
            # Save access token
            self.automation._save_access_token(access_token)
            
            # Update status
            login_status['status'] = 'success'
            login_status['message'] = 'Access token generated and saved successfully!'
            login_status['access_token'] = access_token
            
            # Close browser
            if self.automation.driver:
                self.automation.driver.quit()
            
            return {'success': True, 'access_token': access_token}
            
        except Exception as e:
            login_status['status'] = 'error'
            login_status['message'] = f'TOTP processing failed: {str(e)}'
            login_status['error'] = str(e)
            if self.automation and self.automation.driver:
                self.automation.driver.quit()
            return {'success': False, 'error': str(e)}
    
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
                    return token_data.get('access_token')
            return None
        except Exception as e:
            logger.error(f"Error loading existing token: {str(e)}")
            return None

# Dashboard routes
@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/kite/start_login', methods=['POST'])
def start_kite_login():
    """Start the Kite login process"""
    global login_automation, login_thread
    
    try:
        # Reset login status
        global login_status
        login_status = {
            'status': 'idle',
            'message': '',
            'access_token': None,
            'error': None,
            'request_token': None
        }
        
        # Create new login automation instance
        login_automation = DashboardKiteLogin()
        
        # Start login in a separate thread
        def run_login():
            result = login_automation.start_login_process()
            
        login_thread = threading.Thread(target=run_login)
        login_thread.daemon = True
        login_thread.start()
        
        return jsonify({'success': True, 'message': 'Login process started'})
        
    except Exception as e:
        logger.error(f"Error starting login: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/kite/submit_totp', methods=['POST'])
def submit_totp():
    """Submit TOTP code to complete login"""
    global login_automation
    
    try:
        data = request.get_json()
        totp_code = data.get('totp_code', '').strip()
        
        if not totp_code:
            return jsonify({'success': False, 'error': 'TOTP code is required'})
        
        if not login_automation:
            return jsonify({'success': False, 'error': 'Login process not started'})
        
        # Complete login with TOTP in a separate thread
        def complete_login():
            result = login_automation.complete_login_with_totp(totp_code)
            
        totp_thread = threading.Thread(target=complete_login)
        totp_thread.daemon = True
        totp_thread.start()
        
        return jsonify({'success': True, 'message': 'TOTP submitted, processing...'})
        
    except Exception as e:
        logger.error(f"Error submitting TOTP: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/kite/status')
def get_login_status():
    """Get current login status"""
    global login_status
    return jsonify(login_status)

@app.route('/api/kite/token_info')
def get_token_info():
    """Get current access token information"""
    try:
        token_file = 'access_token.json'
        if os.path.exists(token_file):
            with open(token_file, 'r') as f:
                token_data = json.load(f)
            
            # Hide the actual token for security
            token_data['access_token'] = token_data.get('access_token', '')[:20] + '...' if token_data.get('access_token') else 'Not available'
            
            return jsonify({
                'success': True,
                'token_info': token_data
            })
        else:
            return jsonify({
                'success': False,
                'message': 'No token file found'
            })
    except Exception as e:
        logger.error(f"Error getting token info: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/config/check')
def check_config():
    """Check if configuration is valid"""
    try:
        # Check environment variables
        config_status = {
            'API_KEY': bool(os.getenv('KITE_API_KEY')),
            'API_SECRET': bool(os.getenv('KITE_API_SECRET')),
            'USER_ID': bool(os.getenv('KITE_USER_ID')),
            'PASSWORD': bool(os.getenv('KITE_PASSWORD'))
        }
        
        all_configured = all(config_status.values())
        
        return jsonify({
            'success': True,
            'configured': all_configured,
            'config_status': config_status
        })
        
    except Exception as e:
        logger.error(f"Error checking config: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/ratings/live')
def get_live_ratings():
    """Get current live ratings data"""
    try:
        ratings_file = 'ratings/live_ratings.json'
        if os.path.exists(ratings_file):
            with open(ratings_file, 'r') as f:
                ratings_data = json.load(f)
            
            return jsonify({
                'success': True,
                'data': ratings_data
            })
        else:
            return jsonify({
                'success': False,
                'message': 'No live ratings data found'
            })
    except Exception as e:
        logger.error(f"Error getting live ratings: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/ratings/top_bottom')
def get_top_bottom_ratings():
    """Get separated top 10 and bottom 10 ratings"""
    try:
        ratings_file = 'ratings/live_ratings.json'
        if os.path.exists(ratings_file):
            with open(ratings_file, 'r') as f:
                ratings_data = json.load(f)
            
            ratings = ratings_data.get('selected_stocks_live_ratings', [])
            
            # Sort by final_rating descending
            sorted_ratings = sorted(ratings, key=lambda x: x.get('final_rating', 0), reverse=True)
            
            # Split into top 10 and bottom 10
            top_10 = sorted_ratings[:10] if len(sorted_ratings) >= 10 else sorted_ratings
            bottom_10 = sorted_ratings[-10:] if len(sorted_ratings) >= 10 else []
            
            return jsonify({
                'success': True,
                'data': {
                    'timestamp': ratings_data.get('timestamp'),
                    'total_stocks': len(sorted_ratings),
                    'top_10': top_10,
                    'bottom_10': bottom_10
                }
            })
        else:
            return jsonify({
                'success': False,
                'message': 'No ratings data found'
            })
    except Exception as e:
        logger.error(f"Error getting top/bottom ratings: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/ratings/summary')
def get_ratings_summary():
    """Get summary statistics of live ratings"""
    try:
        ratings_file = 'ratings/live_ratings.json'
        if os.path.exists(ratings_file):
            with open(ratings_file, 'r') as f:
                ratings_data = json.load(f)
            
            ratings = ratings_data.get('selected_stocks_live_ratings', [])
            
            if not ratings:
                return jsonify({
                    'success': False,
                    'message': 'No ratings data available'
                })
            
            # Calculate summary statistics
            scores = [r.get('final_rating', 0) for r in ratings]
            
            summary = {
                'total_stocks': len(ratings),
                'timestamp': ratings_data.get('timestamp'),
                'highest_score': max(scores) if scores else 0,
                'lowest_score': min(scores) if scores else 0,
                'average_score': sum(scores) / len(scores) if scores else 0,
                'positive_count': len([s for s in scores if s > 0]),
                'negative_count': len([s for s in scores if s < 0]),
                'neutral_count': len([s for s in scores if s == 0]),
                'rating_distribution': {
                    'strong_buy': len([s for s in scores if s >= 7.5]),
                    'buy': len([s for s in scores if 2.5 <= s < 7.5]),
                    'neutral': len([s for s in scores if -2.4 <= s < 2.5]),
                    'sell': len([s for s in scores if -7.4 <= s < -2.4]),
                    'strong_sell': len([s for s in scores if s < -7.4])
                }
            }
            
            return jsonify({
                'success': True,
                'data': summary
            })
        else:
            return jsonify({
                'success': False,
                'message': 'No ratings data found'
            })
    except Exception as e:
        logger.error(f"Error getting ratings summary: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    # Ensure templates and static directories exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    
    print("Starting Trading Dashboard...")
    print("Dashboard will be available at: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
