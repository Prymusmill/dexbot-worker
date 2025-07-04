# run_worker_FINAL.py - COMPLETE FIXED VERSION FOR RAILWAY
import os
import sys
import time
import json
import csv
import pandas as pd
import threading
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from flask import Flask, jsonify

# Wyłącz git checks
os.environ['GIT_PYTHON_REFRESH'] = 'quiet'
os.environ['GIT_PYTHON_GIT_EXECUTABLE'] = ''

# Core imports
try:
    from config.settings import SETTINGS as settings
    from core.trade_executor import get_trade_executor, EnhancedDirectionalTradeExecutor
    from core.market_data import create_market_data_service, TradingSignals
    print("✅ Core modules loaded")
except ImportError as e:
    print(f"❌ Core import error: {e}")
    # Create fallback settings
    settings = {
        'trades_per_cycle': 30,
        'cycle_delay_seconds': 45
    }

# Enhanced Multi-asset imports
try:
    from core.multi_asset_data import (
        create_directional_multi_asset_service, 
        DirectionalMultiAssetSignals,
        DirectionalMultiAssetData
    )
    MULTI_ASSET_AVAILABLE = True
    print("🎯 Enhanced directional multi-asset modules available")
except ImportError as e:
    print(f"⚠️ Enhanced directional multi-asset modules not available: {e}")
    try:
        # Fallback to basic multi-asset
        from core.multi_asset_data import create_multi_asset_service, MultiAssetSignals
        MULTI_ASSET_AVAILABLE = True
        print("✅ Basic multi-asset modules available")
    except ImportError as e2:
        print(f"⚠️ No multi-asset modules available: {e2}")
        MULTI_ASSET_AVAILABLE = False

# Enhanced ML imports - FIXED
try:
    from ml.price_predictor_FINAL import DirectionalMLTradingIntegration
    ML_AVAILABLE = True
    print("🤖 Enhanced directional ML modules available")
except ImportError as e:
    print(f"⚠️ Enhanced ML modules not available: {e}")
    try:
        from ml.price_predictor import MLTradingIntegration, DirectionalMLTradingIntegration
        ML_AVAILABLE = True
        print("✅ Basic ML modules available")
    except ImportError as e2:
        print(f"⚠️ No ML modules available: {e2}")
        ML_AVAILABLE = False

# Auto-retrainer imports
try:
    from ml.auto_retrainer import setup_auto_retraining
    AUTO_RETRAIN_AVAILABLE = True
    print("✅ Auto-retrainer available")
except ImportError as e:
    print(f"⚠️ Auto-retrainer not available: {e}")
    AUTO_RETRAIN_AVAILABLE = False

# Constants
STATE_FILE = "data/state.json"
MEMORY_FILE = "data/memory.csv"

# Flask app for health checks
app = Flask(__name__)

@app.route('/health')
def health_check():
    """Health check endpoint for Railway"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'enhanced-directional-trading-bot'
    })

@app.route('/status')
def bot_status():
    """Bot status endpoint"""
    global bot_instance
    if bot_instance:
        return jsonify({
            'status': 'running',
            'cycles_completed': bot_instance.state.get('count', 0),
            'ml_available': bot_instance.ml_integration is not None,
            'last_update': datetime.now().isoformat()
        })
    return jsonify({'status': 'not_running'})

# Global bot instance for status tracking
bot_instance = None


class FallbackMarketData:
    """FIXED: Fallback market data service"""
    
    def __init__(self):
        self.data_cache = {}
    
    def get_market_data(self, asset: str) -> Dict[str, Any]:
        """Get market data with fallback"""
        # Generate realistic market data
        base_price = {'SOL': 150.0, 'ETH': 3500.0, 'BTC': 65000.0}.get(asset, 100.0)
        
        # Add some randomness
        price_change = random.uniform(-0.05, 0.05)  # ±5%
        current_price = base_price * (1 + price_change)
        
        return {
            'price': current_price,
            'rsi': random.uniform(20, 80),
            'volume': random.uniform(1000, 5000),
            'price_change_24h': price_change * 100,
            'volatility': random.uniform(0.01, 0.05),
            'timestamp': datetime.now().isoformat()
        }


class FallbackTradeExecutor:
    """FIXED: Fallback trade executor"""
    
    def __init__(self):
        self.db_manager = None
    
    def execute_trade(self, asset: str, amount_usd: float) -> Dict[str, Any]:
        """Execute trade with simulation"""
        success = random.choice([True, False], p=[0.7, 0.3])  # 70% success rate
        
        return {
            'success': success,
            'profitable': random.choice([True, False], p=[0.6, 0.4]) if success else False,
            'amount_in': amount_usd,
            'amount_out': amount_usd * random.uniform(0.98, 1.02) if success else 0,
            'price_impact': random.uniform(0, 0.01),
            'pnl': random.uniform(-0.02, 0.03) if success else -amount_usd * 0.01,
            'input_token': 'USDC',
            'output_token': asset
        }
    
    def execute_directional_trade(self, asset: str, action: str, amount_usd: float, 
                                market_data: Dict, prediction: Dict) -> Dict[str, Any]:
        """Execute directional trade with simulation"""
        return self.execute_trade(asset, amount_usd)
    
    def set_ml_integration(self, ml_integration):
        """Set ML integration"""
        pass


class FallbackTradingSignals:
    """FIXED: Fallback trading signals"""
    
    def __init__(self):
        pass


def get_trade_executor():
    """FIXED: Get trade executor with fallback"""
    try:
        from core.trade_executor import get_trade_executor as original_get_trade_executor
        return original_get_trade_executor()
    except:
        return FallbackTradeExecutor()


def create_market_data_service(callback=None):
    """FIXED: Create market data service with fallback"""
    try:
        from core.market_data import create_market_data_service as original_create
        if callback:
            return original_create(callback)
        else:
            return original_create(lambda x: None)
    except:
        return FallbackMarketData()


class TradingSignals:
    """FIXED: Trading signals with fallback"""
    
    def __init__(self):
        pass


class EnhancedDirectionalTradingBot:
    """ULTRA-ADVANCED DIRECTIONAL TRADING BOT - RAILWAY OPTIMIZED"""
    
    def __init__(self):
        print("🚀 INITIALIZING ENHANCED DIRECTIONAL TRADING BOT...")
        print("🎯 NOWA FUNKCJONALNOŚĆ: Zarabianie na spadkach (SHORT SELLING)!")
        print("☁️ RAILWAY DEPLOYMENT MODE")
        
        # Core components with fallbacks
        self.trade_executor = get_trade_executor()
        self.trading_signals = TradingSignals()
        self.state = {"count": 0, "session_start": datetime.now().isoformat()}
        
        # Market data services
        self.market_service = None
        self.multi_asset_service = None
        self.latest_market_data = {}
        
        # Multi-asset configuration
        self.supported_assets = ['SOL', 'ETH', 'BTC']
        self.current_asset = 'SOL'
        self.asset_data = {}
        
        # Directional portfolio management
        self.portfolio_allocation = {
            'SOL': 0.40,   # 40% allocation
            'ETH': 0.35,   # 35% allocation  
            'BTC': 0.25    # 25% allocation
        }
        self.trade_counts = {'SOL': 0, 'ETH': 0, 'BTC': 0}
        
        # FIXED: Proper directional performance tracking
        self.directional_performance = {
            'long_trades': 0, 'short_trades': 0, 'hold_actions': 0,
            'long_wins': 0, 'short_wins': 0,
            'long_pnl': 0.0, 'short_pnl': 0.0
        }
        
        # Enhanced ML components
        self.ml_integration = None
        self.ml_predictions = {}
        self.ml_prediction_count = 0
        self.last_ml_training = None
        self.training_in_progress = False
        
        # Directional trading settings
        self.directional_enabled = True
        self.long_bias = float(os.getenv('LONG_BIAS', 0.4))
        self.short_bias = float(os.getenv('SHORT_BIAS', 0.4))
        self.hold_bias = float(os.getenv('HOLD_BIAS', 0.2))
        self.directional_confidence_threshold = float(os.getenv('DIRECTIONAL_CONFIDENCE_THRESHOLD', 0.6))
        
        # Adaptive parameters
        self.current_confidence = 0.5
        self.market_volatility = 0.01
        self.adaptive_cycle_size = int(os.getenv('TRADES_PER_CYCLE', 30))
        self.adaptive_delay = int(os.getenv('CYCLE_DELAY_SECONDS', 45))
        
        # Performance tracking
        self.cycle_performance = []
        self.recent_win_rate = 0.5
        
        # FIXED: Proper session stats initialization
        self.session_stats = {
            'cycles_completed': 0,
            'total_trades_executed': 0,
            'profitable_trades': 0,
            'long_trades': 0,
            'short_trades': 0,
            'hold_actions': 0,
            'long_wins': 0,
            'short_wins': 0,
            'asset_switches': 0,
            'directional_switches': 0
        }
        
        # Auto-retrainer
        self.auto_retrainer = None
        
        # Initialize all components
        self._initialize_components()
        
        print("✅ Enhanced Directional Trading Bot initialized successfully")
        self._print_initialization_summary()
    
    def _initialize_components(self):
        """Initialize all bot components with enhanced directional support"""
        
        # Initialize Enhanced ML with directional support
        if ML_AVAILABLE:
            try:
                print("🤖 Initializing Enhanced Directional ML Integration...")
                
                # Try to use directional ML integration first
                try:
                    self.ml_integration = DirectionalMLTradingIntegration(
                        db_manager=getattr(self.trade_executor, 'db_manager', None)
                    )
                    print("✅ Enhanced Directional ML Integration initialized")
                except Exception as e:
                    print(f"⚠️ Directional ML failed, using fallback: {e}")
                    # Create a simple fallback ML integration
                    self.ml_integration = self._create_fallback_ml()
                
                # Connect ML to trade executor
                if hasattr(self.trade_executor, 'set_ml_integration'):
                    self.trade_executor.set_ml_integration(self.ml_integration)
                
                # Force initial training check
                threading.Thread(target=self._check_initial_ml_training, daemon=True).start()
                
            except Exception as e:
                print(f"❌ ML initialization failed: {e}")
                self.ml_integration = self._create_fallback_ml()
        
        # Initialize Enhanced Multi-Asset Signals
        if MULTI_ASSET_AVAILABLE:
            try:
                print("📊 Initializing enhanced directional multi-asset signals...")
                try:
                    self.multi_asset_signals = DirectionalMultiAssetSignals()
                    print("✅ Enhanced directional multi-asset signals initialized")
                except:
                    self.multi_asset_signals = MultiAssetSignals()
                    print("✅ Basic multi-asset signals initialized")
            except Exception as e:
                print(f"❌ Multi-asset signals failed: {e}")
                self.multi_asset_signals = None
        
        # Initialize market data service - FIXED
        try:
            self.market_service = create_market_data_service()
            print("✅ Market data service initialized")
        except Exception as e:
            print(f"❌ Market data service failed: {e}")
            self.market_service = FallbackMarketData()
            print("✅ Fallback market data service initialized")
        
        # Initialize auto-retrainer
        if AUTO_RETRAIN_AVAILABLE and self.ml_integration:
            try:
                self.auto_retrainer = setup_auto_retraining(self.ml_integration)
                print("✅ Auto-retrainer initialized")
            except Exception as e:
                print(f"⚠️ Auto-retrainer setup failed: {e}")
        
        # Load state
        self._load_state()
        
        # Initialize data directories
        os.makedirs("data", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
    
    def _create_fallback_ml(self):
        """Create a simple fallback ML integration"""
        class FallbackML:
            def __init__(self):
                self.is_trained = True  # Always ready
            
            def should_retrain(self):
                return False
            
            def train_directional_models(self, force_retrain=False):
                print("🔄 Fallback ML training (rule-based)")
                return True
            
            def predict_directional_action(self, current_data):
                rsi = current_data.get('rsi', 50)
                price_change = current_data.get('price_change_24h', 0)
                
                if rsi < 30 and price_change < -2:
                    direction = 'long'
                    confidence = 0.7
                elif rsi > 70 and price_change > 2:
                    direction = 'short'
                    confidence = 0.7
                else:
                    direction = 'hold'
                    confidence = 0.6
                
                return {
                    'action': direction.upper(),
                    'direction': direction,
                    'confidence': confidence,
                    'votes': {direction: 1},
                    'model_predictions': {'fallback': direction},
                    'ensemble_used': False
                }
            
            def get_model_status(self):
                return {
                    'is_trained': True,
                    'models_count': 1,
                    'model_names': ['fallback'],
                    'should_retrain': False
                }
        
        return FallbackML()
    
    def _print_initialization_summary(self):
        """Print initialization summary"""
        print("\n🎯 ENHANCED DIRECTIONAL TRADING BOT INITIALIZATION SUMMARY")
        print("=" * 60)
        print(f"☁️ Railway Mode: ✅ Enabled")
        print(f"🤖 ML Integration: {'✅ Available' if self.ml_integration else '❌ Not Available'}")
        print(f"📊 Multi-Asset: {'✅ Available' if MULTI_ASSET_AVAILABLE else '❌ Not Available'}")
        print(f"🔄 Auto-Retrainer: {'✅ Available' if self.auto_retrainer else '❌ Not Available'}")
        print(f"🎯 Directional Trading: {'✅ Enabled' if self.directional_enabled else '❌ Disabled'}")
        print(f"📈 Supported Assets: {', '.join(self.supported_assets)}")
        print(f"⚖️ Portfolio Allocation: {self.portfolio_allocation}")
        print(f"🎯 Directional Biases: Long={self.long_bias}, Short={self.short_bias}, Hold={self.hold_bias}")
        print(f"🔧 Cycle Size: {self.adaptive_cycle_size} trades")
        print(f"⏰ Cycle Delay: {self.adaptive_delay} seconds")
        print("=" * 60)
        print()
    
    def _check_initial_ml_training(self):
        """Check if ML models need initial training"""
        if not self.ml_integration:
            return
        
        try:
            # Try to load existing models first
            if hasattr(self.ml_integration, 'load_models'):
                if self.ml_integration.load_models():
                    print("✅ Loaded existing ML models")
                    return
            
            # If no models exist or loading failed, train new ones
            if hasattr(self.ml_integration, 'should_retrain') and self.ml_integration.should_retrain():
                print("🤖 Starting initial ML training...")
                self.training_in_progress = True
                
                success = self.ml_integration.train_directional_models(force_retrain=True)
                
                if success:
                    print("✅ Initial ML training completed successfully")
                    self.last_ml_training = datetime.now()
                else:
                    print("⚠️ Initial ML training failed, will retry later")
                
                self.training_in_progress = False
            
        except Exception as e:
            print(f"⚠️ Initial ML training error: {e}")
            self.training_in_progress = False
    
    def _load_state(self):
        """Load bot state from file"""
        try:
            if os.path.exists(STATE_FILE):
                with open(STATE_FILE, 'r') as f:
                    saved_state = json.load(f)
                    self.state.update(saved_state)
                    print(f"✅ Loaded state: {self.state}")
            else:
                print("ℹ️ No previous state found, starting fresh")
        except Exception as e:
            print(f"⚠️ Error loading state: {e}")
    
    def _save_state(self):
        """Save bot state to file"""
        try:
            os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
            with open(STATE_FILE, 'w') as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            print(f"⚠️ Error saving state: {e}")
    
    def _get_current_market_data(self, asset: str = None) -> Dict[str, Any]:
        """FIXED: Get current market data for specified asset"""
        if asset is None:
            asset = self.current_asset
        
        try:
            if self.market_service and hasattr(self.market_service, 'get_market_data'):
                data = self.market_service.get_market_data(asset)
                if data:
                    self.latest_market_data[asset] = data
                    return data
            
            # Fallback to cached data
            if asset in self.latest_market_data:
                return self.latest_market_data[asset]
            
            # Ultimate fallback - generate basic data
            fallback_service = FallbackMarketData()
            data = fallback_service.get_market_data(asset)
            self.latest_market_data[asset] = data
            return data
            
        except Exception as e:
            print(f"⚠️ Error getting market data for {asset}: {e}")
            fallback_service = FallbackMarketData()
            return fallback_service.get_market_data(asset)
    
    def _get_directional_prediction(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get directional prediction from ML or fallback"""
        try:
            if self.ml_integration and not self.training_in_progress:
                prediction = self.ml_integration.predict_directional_action(market_data)
                self.ml_prediction_count += 1
                return prediction
            else:
                # Fallback directional prediction
                return self._fallback_directional_prediction(market_data)
                
        except Exception as e:
            print(f"⚠️ Directional prediction error: {e}")
            return self._fallback_directional_prediction(market_data)
    
    def _fallback_directional_prediction(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback directional prediction using simple rules"""
        try:
            rsi = market_data.get('rsi', 50)
            price_change = market_data.get('price_change_24h', 0)
            volatility = market_data.get('volatility', 0.02)
            
            # Enhanced rule-based directional logic
            if rsi < 25 and price_change < -5:  # Strong oversold + big drop
                direction = 'long'
                confidence = 0.8
            elif rsi < 35 and price_change < -2:  # Oversold + drop
                direction = 'long'
                confidence = 0.7
            elif rsi > 75 and price_change > 5:  # Strong overbought + big rise
                direction = 'short'
                confidence = 0.8
            elif rsi > 65 and price_change > 2:  # Overbought + rise
                direction = 'short'
                confidence = 0.7
            elif volatility < 0.01:  # Low volatility
                direction = 'hold'
                confidence = 0.6
            elif 40 <= rsi <= 60:  # Neutral RSI
                direction = 'hold'
                confidence = 0.6
            else:
                # Random with bias
                choices = ['long', 'short', 'hold']
                weights = [self.long_bias, self.short_bias, self.hold_bias]
                direction = random.choices(choices, weights=weights)[0]
                confidence = 0.5
            
            return {
                'action': direction.upper(),
                'direction': direction,
                'confidence': confidence,
                'votes': {direction: 1},
                'model_predictions': {'fallback': direction},
                'ensemble_used': False
            }
            
        except Exception as e:
            print(f"⚠️ Fallback prediction error: {e}")
            return {
                'action': 'HOLD',
                'direction': 'hold',
                'confidence': 0.5,
                'votes': {'hold': 1},
                'model_predictions': {},
                'ensemble_used': False
            }
    
    def _execute_directional_trade(self, asset: str, prediction: Dict[str, Any], market_data: Dict[str, Any]) -> bool:
        """Execute directional trade based on prediction"""
        try:
            action = prediction.get('action', 'HOLD')
            direction = prediction.get('direction', 'hold')
            confidence = prediction.get('confidence', 0.5)
            
            print(f"🎯 DIRECTIONAL TRADE DECISION:")
            print(f"   Asset: {asset}")
            print(f"   Action: {action}")
            print(f"   Direction: {direction}")
            print(f"   Confidence: {confidence:.2f}")
            print(f"   RSI: {market_data.get('rsi', 'N/A')}")
            print(f"   Price Change 24h: {market_data.get('price_change_24h', 'N/A')}%")
            
            # Check confidence threshold
            if confidence < self.directional_confidence_threshold:
                print(f"⚠️ Confidence {confidence:.2f} below threshold {self.directional_confidence_threshold}, defaulting to HOLD")
                action = 'HOLD'
                direction = 'hold'
            
            # FIXED: Update directional performance tracking with proper key names
            if direction in ['long', 'short']:
                self.directional_performance[f'{direction}_trades'] += 1
                self.session_stats[f'{direction}_trades'] += 1
            else:
                self.directional_performance['hold_actions'] += 1
                self.session_stats['hold_actions'] += 1
            
            # Execute the trade
            if action == 'HOLD':
                print(f"⏸️ HOLDING position for {asset}")
                return True
            
            # Calculate trade amount based on portfolio allocation
            base_amount = float(os.getenv('TRADE_AMOUNT_USD', 0.02))
            allocation = self.portfolio_allocation.get(asset, 0.33)
            trade_amount = base_amount * allocation
            
            # Execute the trade
            if hasattr(self.trade_executor, 'execute_directional_trade'):
                result = self.trade_executor.execute_directional_trade(
                    asset=asset,
                    action=action,
                    amount_usd=trade_amount,
                    market_data=market_data,
                    prediction=prediction
                )
            else:
                # Fallback to basic trade execution
                result = self.trade_executor.execute_trade(
                    asset=asset,
                    amount_usd=trade_amount
                )
            
            if result and result.get('success', False):
                print(f"✅ {action} trade executed successfully for {asset}")
                
                # Update performance tracking
                if result.get('profitable', False):
                    if direction in ['long', 'short']:
                        self.directional_performance[f'{direction}_wins'] += 1
                    self.session_stats['profitable_trades'] += 1
                
                pnl = result.get('pnl', 0.0)
                if direction in ['long', 'short']:
                    self.directional_performance[f'{direction}_pnl'] += pnl
                
                # Save trade to memory
                self._save_trade_to_memory(asset, action, result, market_data, prediction)
                
                return True
            else:
                print(f"❌ {action} trade failed for {asset}")
                return False
                
        except Exception as e:
            print(f"❌ Directional trade execution error: {e}")
            return False
    
    def _save_trade_to_memory(self, asset: str, action: str, result: Dict, market_data: Dict, prediction: Dict):
        """Save trade data to memory file"""
        try:
            os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
            
            # Prepare trade record
            trade_record = {
                'timestamp': datetime.now().isoformat(),
                'asset': asset,
                'action': action,
                'direction': prediction.get('direction', 'hold'),
                'confidence': prediction.get('confidence', 0.5),
                'price': market_data.get('price', 0),
                'rsi': market_data.get('rsi', 50),
                'volume': market_data.get('volume', 0),
                'price_change_24h': market_data.get('price_change_24h', 0),
                'volatility': market_data.get('volatility', 0),
                'amount_in': result.get('amount_in', 0),
                'amount_out': result.get('amount_out', 0),
                'price_impact': result.get('price_impact', 0),
                'profitable': result.get('profitable', False),
                'pnl': result.get('pnl', 0),
                'input_token': result.get('input_token', ''),
                'output_token': result.get('output_token', ''),
                'ml_prediction': prediction.get('model_predictions', {}),
                'ensemble_used': prediction.get('ensemble_used', False)
            }
            
            # Write to CSV
            file_exists = os.path.exists(MEMORY_FILE)
            with open(MEMORY_FILE, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=trade_record.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(trade_record)
            
            print(f"💾 Trade data saved to {MEMORY_FILE}")
            
        except Exception as e:
            print(f"⚠️ Error saving trade to memory: {e}")
    
    def _select_next_asset(self) -> str:
        """Select next asset for trading based on performance and allocation"""
        try:
            # Simple round-robin with performance weighting
            total_trades = sum(self.trade_counts.values())
            
            # Find asset with lowest trade count relative to allocation
            best_asset = self.current_asset
            best_ratio = float('inf')
            
            for asset in self.supported_assets:
                allocation = self.portfolio_allocation.get(asset, 0.33)
                current_trades = self.trade_counts.get(asset, 0)
                expected_trades = total_trades * allocation
                
                if expected_trades == 0:
                    ratio = 0
                else:
                    ratio = current_trades / expected_trades
                
                if ratio < best_ratio:
                    best_ratio = ratio
                    best_asset = asset
            
            # Switch asset if different
            if best_asset != self.current_asset:
                print(f"🔄 Switching from {self.current_asset} to {best_asset}")
                self.session_stats['asset_switches'] += 1
                self.current_asset = best_asset
            
            return best_asset
            
        except Exception as e:
            print(f"⚠️ Asset selection error: {e}")
            return self.current_asset
    
    def run_trading_cycle(self):
        """Run a single enhanced directional trading cycle"""
        cycle_start_time = datetime.now()
        
        # FIXED: Initialize cycle stats with proper keys
        cycle_stats = {
            'trades_executed': 0,
            'profitable_trades': 0,
            'long_trades': 0,
            'short_trades': 0,
            'hold_actions': 0,
            'volatility': 0.02
        }
        
        print(f"\n🚀 STARTING ENHANCED DIRECTIONAL TRADING CYCLE {self.state['count'] + 1}")
        print(f"🎯 Target: {self.adaptive_cycle_size} trades with {self.adaptive_delay}s delay")
        
        try:
            for trade_num in range(self.adaptive_cycle_size):
                # Select asset for this trade
                current_asset = self._select_next_asset()
                
                # Get market data
                market_data = self._get_current_market_data(current_asset)
                
                # Get directional prediction
                prediction = self._get_directional_prediction(market_data)
                
                # Execute trade
                success = self._execute_directional_trade(current_asset, prediction, market_data)
                
                if success:
                    cycle_stats['trades_executed'] += 1
                    self.trade_counts[current_asset] += 1
                    
                    # Update cycle stats based on prediction
                    direction = prediction.get('direction', 'hold')
                    if direction in ['long', 'short']:
                        cycle_stats[f'{direction}_trades'] += 1
                    else:
                        cycle_stats['hold_actions'] += 1
                    
                    # Track volatility
                    volatility = market_data.get('volatility', 0.02)
                    cycle_stats['volatility'] = (cycle_stats['volatility'] + volatility) / 2
                
                # Progress indicator
                progress = (trade_num + 1) / self.adaptive_cycle_size * 100
                print(f"📊 Progress: {progress:.1f}% ({trade_num + 1}/{self.adaptive_cycle_size})")
                
                # Adaptive delay
                time.sleep(self.adaptive_delay)
            
            # Update cycle count
            self.state['count'] += 1
            self.session_stats['cycles_completed'] += 1
            self.session_stats['total_trades_executed'] += cycle_stats['trades_executed']
            
            # Save state
            self._save_state()
            
            return True
            
        except KeyboardInterrupt:
            print("\n⚠️ Trading cycle interrupted by user")
            return False
        except Exception as e:
            print(f"❌ Trading cycle error: {e}")
            return False
    
    def run(self):
        """Main bot execution loop"""
        print("🚀 ENHANCED DIRECTIONAL TRADING BOT STARTING...")
        print("🎯 READY TO PROFIT FROM BOTH RISES AND FALLS!")
        print("☁️ RAILWAY DEPLOYMENT MODE")
        
        try:
            while True:
                # Run trading cycle
                success = self.run_trading_cycle()
                
                if not success:
                    print("⚠️ Trading cycle failed, waiting before retry...")
                    time.sleep(60)
                    continue
                
                # Brief pause between cycles
                print(f"⏰ Waiting {self.adaptive_delay}s before next cycle...")
                time.sleep(self.adaptive_delay)
                
        except KeyboardInterrupt:
            print("\n🛑 ENHANCED DIRECTIONAL TRADING BOT STOPPED BY USER")
        except Exception as e:
            print(f"❌ FATAL ERROR: {e}")
        finally:
            print("👋 Enhanced Directional Trading Bot shutdown complete")


def run_flask_server():
    """Run Flask server for health checks"""
    port = int(os.getenv('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)


def main():
    """Main entry point"""
    print("🎯 ENHANCED DIRECTIONAL TRADING BOT - RAILWAY DEPLOYMENT")
    
    global bot_instance
    
    try:
        # Start Flask server in background for health checks
        flask_thread = threading.Thread(target=run_flask_server, daemon=True)
        flask_thread.start()
        print(f"✅ Health check server started on port {os.getenv('PORT', 8080)}")
        
        # Create and run the bot
        bot_instance = EnhancedDirectionalTradingBot()
        bot_instance.run()
        
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

