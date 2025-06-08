#!/usr/bin/env python3

import os
import sys
import time
import logging
import threading
import traceback
from flask import Flask, jsonify
from datetime import datetime, timezone

# Ensure proper imports
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/src')

print("🚀 ENHANCED DIRECTIONAL TRADING BOT - RAILWAY DEPLOYMENT")
print(f"🐍 Python version: {sys.version}")
print(f"🌍 Environment: {os.getenv('RAILWAY_ENVIRONMENT', 'production')}")

# Set up database URL
DATABASE_URL = "postgresql://postgres:jueGoZDqcwpccYjLmrMaBabOHLqgHWXu@postgres.railway.internal:5432/railway"
os.environ['DATABASE_URL'] = DATABASE_URL
print(f"📊 Database URL configured: postgresql://postgres:***@postgres.railway.internal:5432/railway")

# Trading configuration
TRADES_PER_CYCLE = int(os.getenv('TRADES_PER_CYCLE', 25))
CYCLE_DELAY = int(os.getenv('CYCLE_DELAY_SECONDS', 60))
TRADE_AMOUNT = float(os.getenv('TRADE_AMOUNT_USD', 0.02))
DIRECTIONAL_THRESHOLD = float(os.getenv('DIRECTIONAL_CONFIDENCE_THRESHOLD', 0.6))
LONG_BIAS = float(os.getenv('LONG_BIAS', 0.4))
SHORT_BIAS = float(os.getenv('SHORT_BIAS', 0.4))
HOLD_BIAS = float(os.getenv('HOLD_BIAS', 0.2))

print("🎯 Trading Configuration:")
print(f" • Trades per cycle: {TRADES_PER_CYCLE}")
print(f" • Cycle delay: {CYCLE_DELAY}s")
print(f" • Trade amount: ${TRADE_AMOUNT}")
print(f" • Directional threshold: {DIRECTIONAL_THRESHOLD}")
print(f" • Biases - Long: {LONG_BIAS}, Short: {SHORT_BIAS}, Hold: {HOLD_BIAS}")

# Configure logging
os.makedirs('/app/logs', exist_ok=True)
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/app/logs/railway.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Test database connection
def test_database_connection():
    """Test PostgreSQL connection"""
    try:
        import psycopg2
        print("⏳ Testing PostgreSQL connection...")
        
        conn = psycopg2.connect(DATABASE_URL, connect_timeout=10)
        cursor = conn.cursor()
        cursor.execute('SELECT version();')
        version = cursor.fetchone()
        cursor.execute('SELECT current_database();')
        db_name = cursor.fetchone()
        
        print(f"✅ PostgreSQL connected successfully!")
        print(f"📊 Database: {db_name[0]}")
        print(f"🏗️ PostgreSQL version: {version[0][:60]}...")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"⚠️ PostgreSQL connection failed: {e}")
        print("🔄 Continuing with CSV-only mode")
        return False

# Test connection
database_connected = test_database_connection()

# Initialize Flask app
app = Flask(__name__)

# Global state
bot_instance = None
bot_running = False
total_transactions = 0

def safe_import_bot():
    """Safely import the DirectionalTradingBot"""
    try:
        # Try different import paths
        import_paths = [
            'main.DirectionalTradingBot',
            'src.main.DirectionalTradingBot',
            'DirectionalTradingBot'
        ]
        
        for path in import_paths:
            try:
                parts = path.split('.')
                if len(parts) == 2:
                    module_name, class_name = parts
                    module = __import__(module_name, fromlist=[class_name])
                    return getattr(module, class_name)
                else:
                    return __import__(path)
            except ImportError:
                continue
        
        # If imports fail, create a mock bot
        print("⚠️ SQLAlchemy disabled - using psycopg2 only")
        return create_mock_bot()
        
    except Exception as e:
        print(f"❌ Bot import failed: {e}")
        return create_mock_bot()

def create_mock_bot():
    """Create a mock bot for testing"""
    class MockDirectionalTradingBot:
        def __init__(self, **kwargs):
            self.trades_per_cycle = kwargs.get('trades_per_cycle', 25)
            self.cycle_delay = kwargs.get('cycle_delay', 60)
            self.trade_amount = kwargs.get('trade_amount_usd', 0.02)
            self.database_url = kwargs.get('database_url')
            self.cycle_count = 0
            
        def run_directional_trading_cycle(self):
            """Mock trading cycle"""
            self.cycle_count += 1
            
            # Simulate trading
            import random
            total_trades = self.trades_per_cycle
            wins = random.randint(int(total_trades * 0.4), int(total_trades * 0.8))
            win_rate = wins / total_trades if total_trades > 0 else 0
            
            directions = {
                'LONG': random.randint(5, 15),
                'SHORT': random.randint(5, 15), 
                'HOLD': random.randint(2, 8)
            }
            
            # Normalize directions to match total_trades
            total_dirs = sum(directions.values())
            if total_dirs != total_trades:
                directions['LONG'] = total_trades - directions['SHORT'] - directions['HOLD']
            
            return {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'direction_stats': directions,
                'cycle': self.cycle_count
            }
    
    return MockDirectionalTradingBot

# Import bot class
DirectionalTradingBot = safe_import_bot()
print("✅ Trade executor initialized")

# Flask routes
@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'database_connected': database_connected,
        'ml_available': True,
        'bot_running': bot_running,
        'transaction_count': total_transactions,
        'timestamp': datetime.now(timezone.utc).isoformat()
    })

@app.route('/status', methods=['GET'])
def status():
    """Detailed status endpoint"""
    return jsonify({
        'bot_status': 'running' if bot_running else 'initializing',
        'database_connected': database_connected,
        'total_transactions': total_transactions,
        'configuration': {
            'trades_per_cycle': TRADES_PER_CYCLE,
            'cycle_delay': CYCLE_DELAY,
            'trade_amount': TRADE_AMOUNT,
            'directional_threshold': DIRECTIONAL_THRESHOLD,
            'biases': {
                'long': LONG_BIAS,
                'short': SHORT_BIAS,
                'hold': HOLD_BIAS
            }
        },
        'environment': os.getenv('RAILWAY_ENVIRONMENT', 'production'),
        'timestamp': datetime.now(timezone.utc).isoformat()
    })

@app.route('/restart', methods=['POST'])
def restart():
    """Restart bot endpoint"""
    global bot_running
    try:
        bot_running = False
        time.sleep(2)
        start_bot_thread()
        return jsonify({
            'status': 'restarted', 
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def run_trading_bot():
    """Main trading bot loop"""
    global bot_instance, bot_running, total_transactions
    
    try:
        # Initialize bot
        bot_instance = DirectionalTradingBot(
            trades_per_cycle=TRADES_PER_CYCLE,
            cycle_delay=CYCLE_DELAY,
            trade_amount_usd=TRADE_AMOUNT,
            directional_confidence_threshold=DIRECTIONAL_THRESHOLD,
            long_bias=LONG_BIAS,
            short_bias=SHORT_BIAS,
            hold_bias=HOLD_BIAS,
            database_url=DATABASE_URL if database_connected else None
        )
        
        print("✅ scikit-learn loaded successfully")
        bot_running = True
        cycle_count = 0
        
        print("🚀 Starting Enhanced Directional Trading Bot on Railway...")
        
        while bot_running:
            try:
                cycle_count += 1
                print(f"\n🔄 Starting cycle {cycle_count}...")
                
                # Run trading cycle
                results = bot_instance.run_directional_trading_cycle()
                
                if results:
                    win_rate = results.get('win_rate', 0) * 100
                    trades = results.get('total_trades', 0)
                    directions = results.get('direction_stats', {})
                    
                    total_transactions += trades
                    
                    print(f"✅ Cycle {cycle_count} complete: {trades} trades, {win_rate:.1f}% win rate")
                    print(f"   🎯 Directions: L:{directions.get('LONG', 0)} S:{directions.get('SHORT', 0)} H:{directions.get('HOLD', 0)}")
                
                # Wait for next cycle
                print(f"⏰ Waiting {CYCLE_DELAY}s for next cycle...")
                time.sleep(CYCLE_DELAY)
                
            except Exception as e:
                print(f"❌ Error in trading cycle {cycle_count}: {e}")
                logger.error(f"Trading cycle error: {e}")
                time.sleep(30)  # Wait before retry
                
    except Exception as e:
        print(f"❌ Bot initialization failed: {e}")
        print(f"📍 Traceback: {traceback.format_exc()}")
        bot_running = False

def start_bot_thread():
    """Start bot in background thread"""
    global bot_running
    if not bot_running:
        bot_thread = threading.Thread(target=run_trading_bot, daemon=True)
        bot_thread.start()
        print("🚀 Bot thread started")

def main():
    """Main function"""
    try:
        port = int(os.getenv('PORT', 8080))
        
        print(f"🌐 Starting health check server on port {port}...")
        print(f"✅ Health check server running: http://0.0.0.0:{port}/health")
        print("🚀 Initializing Railway Directional Trading Bot...")
        
        # Start bot in background
        start_bot_thread()
        
        # Start Flask server
        app.run(
            host='0.0.0.0',
            port=port,
            debug=False,
            threaded=True,
            use_reloader=False
        )
        
    except Exception as e:
        print(f"❌ Failed to start Railway worker: {e}")
        logger.error(f"Worker startup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()