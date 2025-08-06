#!/usr/bin/env python3
"""
Dashboard Runner Script

This script starts the Trading Dashboard web application.
"""

import os
import sys
import subprocess

def check_requirements():
    """Check if Flask is installed"""
    try:
        import flask
        print("✓ Flask is available")
        return True
    except ImportError:
        print("✗ Flask is not installed")
        return False

def install_flask():
    """Install Flask if not available"""
    print("Installing Flask...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "flask"])
        print("✓ Flask installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install Flask: {e}")
        return False

def main():
    """Main function to run the dashboard"""
    print("=" * 50)
    print("Trading Dashboard Launcher")
    print("=" * 50)
    
    # Check if Flask is available
    if not check_requirements():
        print("\nFlask is required to run the dashboard.")
        install_choice = input("Would you like to install Flask? (y/n): ").lower().strip()
        
        if install_choice in ['y', 'yes']:
            if not install_flask():
                print("Failed to install Flask. Please install it manually:")
                print("pip install flask")
                return 1
        else:
            print("Dashboard cannot run without Flask. Exiting.")
            return 1
    
    # Check environment variables
    print("\nChecking configuration...")
    required_vars = ['KITE_API_KEY', 'KITE_API_SECRET', 'KITE_USER_ID', 'KITE_PASSWORD']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("⚠️  Missing environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease set these variables in your .env file or environment.")
        print("The dashboard will still run but Kite login may not work.")
    else:
        print("✓ All required environment variables are set")
    
    # Create directories if they don't exist
    directories = ['templates', 'static/css', 'static/js']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("\n" + "=" * 50)
    print("Starting Trading Dashboard...")
    print("=" * 50)
    print("Dashboard URL: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Import and run the dashboard
    try:
        from dashboard import app
        app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
    except KeyboardInterrupt:
        print("\n\nDashboard stopped by user.")
        return 0
    except Exception as e:
        print(f"\nError starting dashboard: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
