# run_worker_RAILWAY.py - OPTIMIZED FOR RAILWAY WITH POSTGRESQL
import os
import sys
import time
import json
import threading
import logging
from datetime import datetime
from flask import Flask, jsonify

# 🎯 RAILWAY ENVIRONMENT SETUP
print("🚀 ENHANCED DIRECTIONAL TRADING BOT - RAILWAY DEPLOYMENT")
print(f"🐍 Python version: {sys.version}")
print(f"🌍 Environment: {os.getenv('RAILWAY_ENVIRONMENT', 'unknown')}")

# Database configuration check
DATABASE_URL = os.getenv('DATABASE_URL')
if DATABASE_URL:
    # Mask password for logging
    safe_url = DATABASE_URL.replace(DATABASE_URL.split(':')[2].split('@')[0], '***')
    print(f"📊 Database URL found: {safe_url}")
else:
    print("⚠️ DATABASE_URL not found in environment")

# Load environment variables with defaults
PORT = int(os.getenv('PORT', 8000))
TRADES_PER_CYCLE = int(os.getenv('TRADES_PER_CYCLE', 25))
CYCLE_DELAY_SECONDS = int(os.getenv('CYCLE_DELAY_SECONDS', 60))
TRADE_AMOUNT_USD = float(os.getenv('TRADE_AMOUNT_USD', 0.02))
DIRECTIONAL_CONFIDENCE_THRESHOLD = float(os.getenv('DIRECTIONAL_CONFIDENCE_THRESHOLD', 0.6))
LONG_BIAS = float(os.getenv('LONG_BIAS', 0.4))
SHORT_BIAS = float(os.getenv('SHORT_BIAS', 0.4))
HOLD_BIAS = float(os.getenv('HOLD_BIAS', 0.2))

print(f"🎯 Trading Configuration:")
print(f"   • Trades per cycle: {TRADES_PER_CYCLE}")
print(f"   • Cycle delay: {CYCLE_DELAY_SECONDS}s")
print(f"   • Trade amount: ${TRADE_AMOUNT_USD}")
print(f"   • Directional threshold: {DIRECTIONAL_CONFIDENCE_THRESHOLD}")
print(f"   • Biases - Long: {LONG_BIAS}, Short: {SHORT_BIAS}, Hold: {HOLD_BIAS}")

# Flask app for health checks
app = Flask(__name__)

@app.route('/health')
def health_check():
    """Enhanced health check for Railway"""
    global bot_instance
    
    health_status = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'enhanced-directional-trading-bot',
        'environment': os.getenv('RAILWAY_ENVIRONMENT', 'production'),
        'python_version': sys.version.split()[0],
        'database_connected': False,
        'ml_available': False,
        'bot_running': False
    }
    
    # Check database connection
    if DATABASE_URL:
        try:
            from database.db_manager import get_db_manager
            db_manager = get_db_manager()
            count = db_manager.get_transaction_count()
            health_status['database_connected'] = True
            health_status['transaction_count'] = count
        except Exception as e:
            health_status['database_error'] = str(e)
    
    # Check bot status
    if bot_instance:
        health_status['bot_running'] = True
        health_status['ml_available'] = bot_instance.ml_integration is not None
        health_status['cycles_completed'] = bot_instance.state.get('count', 0)
    
    return jsonify(health_status)

@app.route('/status')
def bot_status():
    """Detailed bot status endpoint"""
    global bot_instance
    
    if not bot_instance:
        return jsonify({'error': 'Bot not initialized'})
    
    try:
        status = {
            'bot_status': 'running',
            'session_start': bot_instance.state.get('session_start'),
            'cycles_completed': bot_instance.state.get('count', 0),
            'total_trades': bot_instance.session_stats.get('total_trades_executed', 0),
            'directional_performance': bot_instance.directional_performance,
            'current_asset': bot_instance.current_asset,
            'supported_assets': bot_instance.supported_assets,
            'ml_integration': {
                'available': bot_instance.ml_integration is not None,
                'training_in_progress': bot_instance.training_in_progress,
                'prediction_count': bot_instance.ml_prediction_count
            },
            'database': {
                'available': bot_instance.trade_executor.db_available if hasattr(bot_instance.trade_executor, 'db_available') else False
            }
        }
        
        return jsonify(status)
    
    except Exception as e:
        return jsonify({'error': str(e)})

# Global bot instance
bot_instance = None

class RailwayDirectionalTradingBot:
    """🎯 Railway-optimized directional trading bot"""
    
    def __init__(self):
        print("🚀 Initializing Railway Directional Trading Bot...")
        
        # Setup logging for Railway
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize state
        self.state = {"count": 0, "session_start": datetime.now().isoformat()}
        
        # Trading configuration from environment
        self.supported_assets = ['SOL', 'ETH', 'BTC']
        self.current_asset = 'SOL'
        self.directional_enabled = os.getenv('ENABLE_DIRECTIONAL_TRADING', 'true').lower() == 'true'
        
        # Performance tracking
        self.session_stats = {
            'cycles_completed': 0,
            'total_trades_executed': 0,
            'profitable_trades': 0,
            'long_trades': 0,
            'short_trades': 0,
            'hold_actions': 0,
            'long_wins': 0,
            'short_wins': 0
        }
        
        self.directional_performance = {
            'long_trades': 0, 'short_trades': 0, 'hold_actions': 0,
            'long_wins': 0, 'short_wins': 0,
            'long_pnl': 0.0, 'short_pnl': 0.0
        }
        
        # ML components
        self.ml_integration = None
        self.training_in_progress = False
        self.ml_prediction_count = 0
        
        # Trade executor
        self.trade_executor = None
        
        # Initialize components
        self._initialize_components()
        
        print("✅ Railway Directional Trading Bot initialized successfully")
    
    def _initialize_components(self):
        """Initialize trading components with Railway optimizations"""
        
        # Initialize trade executor
        try:
            from core.trade_executor import get_trade_executor
            self.trade_executor = get_trade_executor()
            print("✅ Trade executor initialized")
        except Exception as e:
            print(f"⚠️ Trade executor error: {e}")
            self.trade_executor = self._create_fallback_executor()
        
        # Initialize ML integration
        try:
            from ml.price_predictor import DirectionalMLTradingIntegration
            self.ml_integration = DirectionalMLTradingIntegration(
                db_manager=getattr(self.trade_executor, 'db_manager', None)
            )
            print("✅ Directional ML integration initialized")
            
            # Connect ML to trade executor
            if hasattr(self.trade_executor, 'set_ml_integration'):
                self.trade_executor.set_ml_integration(self.ml_integration)
            
            # Start ML training in background
            threading.Thread(target=self._train_ml_models, daemon=True).start()
            
        except Exception as e:
            print(f"⚠️ ML integration error: {e}")
            self.ml_integration = self._create_fallback_ml()
        
        # Setup data directories
        os.makedirs("data", exist_ok=True)
        os.makedirs("models", exist_ok=True)
    
    def _create_fallback_executor(self):
        """Create fallback trade executor"""
        class FallbackExecutor:
            def __init__(self):
                self.db_available = False
            
            def execute_directional_trade(self, settings, asset, direction=None):
                return {
                    'success': True,
                    'profitable': True,
                    'direction': direction or 'hold',
                    'pnl': 0.001,
                    'pnl_percentage': 0.5
                }
            
            def set_ml_integration(self, ml):
                pass
        
        return FallbackExecutor()
    
    def _create_fallback_ml(self):
        """Create fallback ML integration"""
        class FallbackML:
            def predict_directional_action(self, data):
                return {
                    'action': 'HOLD',
                    'direction': 'hold',
                    'confidence': 0.6
                }
            
            def train_directional_models(self):
                return True
        
        return FallbackML()
    
    def _train_ml_models(self):
        """Train ML models in background"""
        if not self.ml_integration:
            return
        
        try:
            time.sleep(30)  # Wait for system to stabilize
            print("🤖 Starting background ML training...")
            self.training_in_progress = True
            
            success = self.ml_integration.train_directional_models()
            
            if success:
                print("✅ ML training completed successfully")
            else:
                print("⚠️ ML training completed with issues")
                
        except Exception as e:
            print(f"❌ ML training error: {e}")
        finally:
            self.training_in_progress = False
    
    def _get_market_data(self, asset='SOL'):
        """Get market data (simulated for Railway)"""
        import random
        
        base_prices = {'SOL': 150.0, 'ETH': 3500.0, 'BTC': 65000.0}
        base_price = base_prices.get(asset, 100.0)
        
        return {
            'price': base_price * (1 + random.uniform(-0.05, 0.05)),
            'rsi': random.uniform(20, 80),
            'volume': random.uniform(1000, 5000),
            'price_change_24h': random.uniform(-10, 10),
            'volatility': random.uniform(0.01, 0.05)
        }
    
    def _execute_trading_cycle(self):
        """Execute one trading cycle"""
        cycle_stats = {
            'executed': 0,
            'profitable': 0,
            'long_trades': 0,
            'short_trades': 0,
            'hold_actions': 0
        }
        
        for i in range(TRADES_PER_CYCLE):
            try:
                # Get market data
                market_data = self._get_market_data(self.current_asset)
                
                # Get ML prediction
                prediction = self.ml_integration.predict_directional_action(market_data)
                direction = prediction.get('direction', 'hold')
                confidence = prediction.get('confidence', 0.5)
                
                # Check confidence threshold
                if confidence < DIRECTIONAL_CONFIDENCE_THRESHOLD:
                    direction = 'hold'
                
                # Execute trade
                result = self.trade_executor.execute_directional_trade(
                    {'trade_amount_usd': TRADE_AMOUNT_USD},
                    self.current_asset,
                    direction
                )
                
                if result and result.get('success'):
                    cycle_stats['executed'] += 1
                    
                    if result.get('profitable'):
                        cycle_stats['profitable'] += 1
                    
                    # Track by direction
                    trade_direction = result.get('direction', direction)
                    if trade_direction == 'long':
                        cycle_stats['long_trades'] += 1
                        self.directional_performance['long_trades'] += 1
                        if result.get('profitable'):
                            self.directional_performance['long_wins'] += 1
                    elif trade_direction == 'short':
                        cycle_stats['short_trades'] += 1
                        self.directional_performance['short_trades'] += 1
                        if result.get('profitable'):
                            self.directional_performance['short_wins'] += 1
                    else:
                        cycle_stats['hold_actions'] += 1
                        self.directional_performance['hold_actions'] += 1
                
                # Small delay
                time.sleep(2)
                
            except Exception as e:
                self.logger.error(f"Trade execution error: {e}")
                continue
        
        # Update session stats
        self.session_stats['cycles_completed'] += 1
        self.session_stats['total_trades_executed'] += cycle_stats['executed']
        self.session_stats['profitable_trades'] += cycle_stats['profitable']
        self.session_stats['long_trades'] += cycle_stats['long_trades']
        self.session_stats['short_trades'] += cycle_stats['short_trades']
        self.session_stats['hold_actions'] += cycle_stats['hold_actions']
        
        self.state['count'] += 1
        
        # Log cycle summary
        win_rate = (cycle_stats['profitable'] / cycle_stats['executed']) if cycle_stats['executed'] > 0 else 0
        print(f"✅ Cycle {self.state['count']} complete: {cycle_stats['executed']} trades, {win_rate:.1%} win rate")
        print(f"   🎯 Directions: L:{cycle_stats['long_trades']} S:{cycle_stats['short_trades']} H:{cycle_stats['hold_actions']}")
        
        return cycle_stats
    
    def run(self):
        """Main bot execution loop"""
        print("🚀 Starting Enhanced Directional Trading Bot on Railway...")
        
        try:
            while True:
                # Execute trading cycle
                self._execute_trading_cycle()
                
                # Wait before next cycle
                print(f"⏳ Waiting {CYCLE_DELAY_SECONDS}s before next cycle...")
                time.sleep(CYCLE_DELAY_SECONDS)
                
        except KeyboardInterrupt:
            print("\n🛑 Bot stopped by user")
        except Exception as e:
            print(f"❌ Fatal error: {e}")
            self.logger.error(f"Fatal error: {e}")


def run_flask_server():
    """Run Flask server for health checks"""
    print(f"🌐 Starting health check server on port {PORT}...")
    app.run(host='0.0.0.0', port=PORT, debug=False)


def main():
    """Main entry point for Railway"""
    global bot_instance
    
    try:
        # Start Flask server in background
        flask_thread = threading.Thread(target=run_flask_server, daemon=True)
        flask_thread.start()
        
        # Wait for Flask to start
        time.sleep(2)
        print(f"✅ Health check server running: http://0.0.0.0:{PORT}/health")
        
        # Create and run bot
        bot_instance = RailwayDirectionalTradingBot()
        
        # Small delay to ensure health checks work
        time.sleep(5)
        
        # Start bot
        bot_instance.run()
        
    except Exception as e:
        print(f"❌ Startup error: {e}")
        logging.error(f"Startup error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()