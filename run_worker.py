# run_worker.py - ENHANCED DIRECTIONAL TRADING BOT (ZARABIA NA WZROSTACH I SPADKACH!)
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
    sys.exit(1)

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

# Enhanced ML imports
try:
    from ml.price_predictor import MLTradingIntegration, DirectionalMLTradingIntegration
    ML_AVAILABLE = True
    print("🤖 Enhanced directional ML modules available")
except ImportError as e:
    print(f"⚠️ Enhanced ML modules not available: {e}")
    try:
        from ml.price_predictor import MLTradingIntegration
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


class EnhancedDirectionalTradingBot:
    """🎯 ULTRA-ADVANCED DIRECTIONAL TRADING BOT - ZARABIA NA WZROSTACH I SPADKACH!"""
    
    def __init__(self):
        print("🚀 INITIALIZING ENHANCED DIRECTIONAL TRADING BOT...")
        print("🎯 NOWA FUNKCJONALNOŚĆ: Zarabianie na spadkach (SHORT SELLING)!")
        
        # Core components
        self.trade_executor = get_trade_executor()
        self.trading_signals = TradingSignals()
        self.state = {"count": 0, "session_start": datetime.now().isoformat()}
        
        # Ensure we're using the enhanced executor
        if not isinstance(self.trade_executor, EnhancedDirectionalTradeExecutor):
            print("🔄 Upgrading to Enhanced Directional Trade Executor...")
            self.trade_executor = EnhancedDirectionalTradeExecutor()
        
        # Market data services
        self.market_service = None
        self.multi_asset_service = None
        self.latest_market_data = {}  # Per-asset data
        
        # Multi-asset configuration
        self.supported_assets = ['SOL', 'ETH', 'BTC']
        self.current_asset = 'SOL'
        self.asset_data = {}
        
        # 🎯 ENHANCED: Directional portfolio management
        self.portfolio_allocation = {
            'SOL': 0.40,   # 40% allocation
            'ETH': 0.35,   # 35% allocation  
            'BTC': 0.25    # 25% allocation
        }
        self.trade_counts = {'SOL': 0, 'ETH': 0, 'BTC': 0}
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
        
        # 🎯 DIRECTIONAL TRADING SETTINGS
        self.directional_enabled = True
        self.long_bias = 0.4      # 40% bias towards long
        self.short_bias = 0.4     # 40% bias towards short  
        self.hold_bias = 0.2      # 20% bias towards hold
        self.directional_confidence_threshold = 0.6
        
        # Adaptive parameters
        self.current_confidence = 0.5
        self.market_volatility = 0.01
        self.adaptive_cycle_size = settings.get("trades_per_cycle", 30)  # Reduced for directional
        self.adaptive_delay = settings.get("cycle_delay_seconds", 45)     # Increased for analysis
        
        # Performance tracking
        self.cycle_performance = []
        self.recent_win_rate = 0.5
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
                        db_manager=self.trade_executor.db_manager if hasattr(self.trade_executor, 'db_manager') else None
                    )
                    print("✅ Enhanced Directional ML Integration initialized")
                except:
                    # Fallback to basic ML
                    self.ml_integration = MLTradingIntegration(
                        db_manager=self.trade_executor.db_manager if hasattr(self.trade_executor, 'db_manager') else None
                    )
                    print("✅ Basic ML Integration initialized (fallback)")
                
                # Connect ML to trade executor
                if hasattr(self.trade_executor, 'set_ml_integration'):
                    self.trade_executor.set_ml_integration(self.ml_integration)
                
                # Force initial training check
                threading.Thread(target=self._check_initial_ml_training, daemon=True).start()
                
            except Exception as e:
                print(f"❌ ML initialization failed: {e}")
                self.ml_integration = None
        
        # Initialize Enhanced Multi-Asset Signals
        if MULTI_ASSET_AVAILABLE:
            try:
                print("📊 Initializing enhanced directional multi-asset signals...")
                self.multi_asset_signals = DirectionalMultiAssetSignals()
                print("✅ Enhanced directional multi-asset signals initialized")
            except Exception as e:
                print(f"❌ Enhanced multi-asset signals failed: {e}")
                try:
                    # Fallback to basic signals
                    from core.multi_asset_data import MultiAssetSignals
                    self.multi_asset_signals = MultiAssetSignals()
                    print("✅ Basic multi-asset signals initialized (fallback)")
                except Exception as e2:
                    print(f"❌ All multi-asset signals failed: {e2}")
                    self.multi_asset_signals = None
        
        # Initialize auto-retrainer
        if AUTO_RETRAIN_AVAILABLE and self.ml_integration:
            try:
                print("🔄 Initializing auto-retrainer...")
                self.auto_retrainer = setup_auto_retraining(
                    ml_integration=self.ml_integration,
                    db_manager=self.trade_executor.db_manager if hasattr(self.trade_executor, 'db_manager') else None,
                    retrain_interval_hours=settings.get("ml_retrain_hours", 4.0),  # More frequent for directional
                    min_new_samples=settings.get("retrain_min_samples_trigger", 150),
                    performance_threshold=settings.get("retrain_accuracy_threshold", 0.55)
                )
                print("✅ Auto-retrainer initialized")
            except Exception as e:
                print(f"❌ Auto-retrainer setup failed: {e}")
                self.auto_retrainer = None
    
    def _check_initial_ml_training(self):
        """Check and perform initial ML training if needed"""
        time.sleep(15)  # Wait for system to stabilize
        
        try:
            if not self.ml_integration:
                return
            
            print("🔍 Checking initial directional ML training status...")
            
            if self.ml_integration.should_retrain():
                print("🚀 Starting initial directional ML training...")
                self.training_in_progress = True
                
                try:
                    result = self.ml_integration.train_models()
                    if result.get('success'):
                        print(f"✅ Initial directional ML training successful: {result.get('successful_models', [])}")
                        self.last_ml_training = datetime.now()
                    else:
                        print(f"❌ Initial directional ML training failed: {result.get('error', 'Unknown error')}")
                finally:
                    self.training_in_progress = False
            else:
                print("ℹ️ Directional ML models already trained or insufficient data")
                
        except Exception as e:
            print(f"❌ Initial directional ML training check error: {e}")
            self.training_in_progress = False
    
    def _print_initialization_summary(self):
        """Print comprehensive initialization summary"""
        print("\n" + "="*70)
        print("🎯 ENHANCED DIRECTIONAL TRADING BOT INITIALIZATION SUMMARY")
        print("="*70)
        print(f"📊 Multi-Asset Support: {'✅' if MULTI_ASSET_AVAILABLE else '❌'}")
        print(f"   • Supported Assets: {self.supported_assets}")
        print(f"   • Portfolio Allocation: {self.portfolio_allocation}")
        print(f"🤖 Enhanced Directional ML: {'✅' if self.ml_integration else '❌'}")
        print(f"🔄 Auto-Retrainer: {'✅' if self.auto_retrainer else '❌'}")
        print(f"🎯 DIRECTIONAL TRADING CAPABILITIES:")
        print(f"   • LONG Trading: ✅ (Profit on price increases)")
        print(f"   • SHORT Trading: ✅ (Profit on price decreases)")
        print(f"   • HOLD Strategy: ✅ (Wait for optimal conditions)")
        print(f"   • Position Management: ✅ (Stop loss, take profit)")
        print(f"⚙️ Current Settings:")
        print(f"   • Trades per cycle: {settings.get('trades_per_cycle', 30)}")
        print(f"   • Cycle delay: {settings.get('cycle_delay_seconds', 45)}s")
        print(f"   • Directional confidence threshold: {self.directional_confidence_threshold}")
        print(f"   • Trade amount: ${settings.get('trade_amount_usd', 0.02)}")
        print("="*70 + "\n")
    
    def start_market_data_services(self):
        """Start enhanced market data services with directional support"""
        print("🌐 Starting enhanced directional market data services...")
        
        if MULTI_ASSET_AVAILABLE:
            try:
                print(f"🎯 Connecting to enhanced directional multi-asset streams: {self.supported_assets}")
                
                # Try enhanced directional service first
                try:
                    self.multi_asset_service = create_directional_multi_asset_service(
                        self.supported_assets,
                        self.on_enhanced_multi_asset_update
                    )
                    
                    if self.multi_asset_service:
                        # Connect ML integration to multi-asset service
                        if hasattr(self.multi_asset_service, 'set_ml_integration') and self.ml_integration:
                            self.multi_asset_service.set_ml_integration(self.ml_integration)
                        
                        print("✅ Enhanced directional multi-asset service connected successfully")
                        time.sleep(7)  # Allow more time for directional analysis
                        return True
                    else:
                        print("⚠️ Enhanced directional multi-asset service failed")
                        
                except Exception as e:
                    print(f"❌ Enhanced directional service error: {e}")
                
                # Fallback to basic multi-asset
                print("🔄 Falling back to basic multi-asset service...")
                from core.multi_asset_data import create_multi_asset_service
                self.multi_asset_service = create_multi_asset_service(
                    self.supported_assets,
                    self.on_multi_asset_update
                )
                
                if self.multi_asset_service:
                    print("✅ Basic multi-asset service connected")
                    time.sleep(5)
                    return True
                else:
                    print("⚠️ Basic multi-asset service failed")
                    
            except Exception as e:
                print(f"❌ Multi-asset connection error: {e}")
        
        # Fallback to single asset
        print("🔗 Falling back to single asset mode (SOL/USDC)")
        self.market_service = create_market_data_service(self.on_market_data_update)
        
        if self.market_service:
            print("✅ Single asset market service connected")
            time.sleep(3)
            return True
        else:
            print("❌ All market data connections failed")
            return False
    
    def on_enhanced_multi_asset_update(self, asset_symbol: str, market_data: Dict):
        """🎯 Handle enhanced multi-asset market data updates with directional analysis"""
        try:
            # Store data for all assets
            self.asset_data[asset_symbol] = market_data
            self.latest_market_data[asset_symbol] = market_data
            
            # Update trade executor with asset-specific market data
            self.trade_executor.update_market_data(market_data, asset_symbol)
            
            # Update active trading asset data
            if asset_symbol == self.current_asset:
                self.market_volatility = market_data.get('volatility', 0.01)
            
            # Log enhanced updates with directional info
            self._log_enhanced_market_update(asset_symbol, market_data)
            
            # Update ML predictions for this asset if available
            if self.ml_integration and hasattr(self.multi_asset_service, 'update_ml_prediction'):
                try:
                    # Generate ML prediction for this asset
                    df = self._create_asset_prediction_dataframe(asset_symbol, market_data)
                    if df is not None and len(df) > 0:
                        if hasattr(self.ml_integration, 'get_directional_prediction'):
                            ml_prediction = self.ml_integration.get_directional_prediction(df)
                        else:
                            # Fallback to basic prediction and convert
                            basic_prediction = self.ml_integration.get_ensemble_prediction_with_reality_check(df)
                            ml_prediction = self._convert_basic_to_directional_prediction(basic_prediction)
                        
                        # Update multi-asset service with ML prediction
                        self.multi_asset_service.update_ml_prediction(asset_symbol, ml_prediction)
                        
                except Exception as e:
                    print(f"⚠️ ML prediction update error for {asset_symbol}: {e}")
            
        except Exception as e:
            print(f"❌ Error processing enhanced {asset_symbol} update: {e}")
    
    def on_multi_asset_update(self, asset_symbol: str, market_data: Dict):
        """Handle basic multi-asset market data updates (fallback)"""
        try:
            # Store data for all assets
            self.asset_data[asset_symbol] = market_data
            self.latest_market_data[asset_symbol] = market_data
            
            # Update trade executor
            self.trade_executor.update_market_data(market_data, asset_symbol)
            
            # Update active trading asset data
            if asset_symbol == self.current_asset:
                self.market_volatility = market_data.get('volatility', 0.01)
            
            # Log updates periodically
            self._log_market_update(asset_symbol, market_data)
            
        except Exception as e:
            print(f"❌ Error processing {asset_symbol} update: {e}")
    
    def on_market_data_update(self, market_data: Dict):
        """Handle single asset market data updates (fallback)"""
        self.latest_market_data['SOL'] = market_data
        self.asset_data['SOL'] = market_data
        self.trade_executor.update_market_data(market_data, 'SOL')
        self.market_volatility = market_data.get('volatility', 0.01)
        
        # Log periodically
        self._log_market_update('SOL', market_data)
    
    def _create_asset_prediction_dataframe(self, asset: str, market_data: Dict):
        """Create DataFrame for ML prediction for specific asset"""
        try:
            import pandas as pd
            
            # Get recent market data for this asset
            recent_data = []
            
            # Add current data point
            data_point = {
                'price': market_data.get('price', 100),
                'rsi': market_data.get('rsi', 50),
                'volume': market_data.get('volume_24h', 1000),
                'volatility': market_data.get('volatility', 0.02),
                'price_change_24h': market_data.get('price_change_24h', 0),
                'timestamp': datetime.now(),
                'asset': asset
            }
            recent_data.append(data_point)
            
            # If we have price history, add more data points
            price_history = market_data.get('price_history', [])
            if len(price_history) > 5:
                for i, price in enumerate(price_history[-5:]):
                    historical_point = data_point.copy()
                    historical_point['price'] = price
                    historical_point['timestamp'] = datetime.now() - timedelta(minutes=i+1)
                    recent_data.append(historical_point)
            
            return pd.DataFrame(recent_data)
            
        except Exception as e:
            print(f"⚠️ Error creating prediction DataFrame for {asset}: {e}")
            return None
    
    def _convert_basic_to_directional_prediction(self, basic_prediction: Dict) -> Dict:
        """Convert basic ML prediction to directional format"""
        try:
            if not basic_prediction:
                return {'predicted_direction': 'hold', 'confidence': 0.5, 'direction_probabilities': {'long': 0.33, 'short': 0.33, 'hold': 0.34}}
            
            predicted_profitable = basic_prediction.get('predicted_profitable', True)
            confidence = basic_prediction.get('confidence', 0.5)
            direction = basic_prediction.get('direction', 'profitable')
            
            # Convert to directional format
            if direction == 'profitable' or predicted_profitable:
                predicted_direction = 'long'
                direction_probabilities = {'long': confidence, 'short': (1-confidence)/2, 'hold': (1-confidence)/2}
            elif direction == 'unprofitable' or not predicted_profitable:
                predicted_direction = 'short'
                direction_probabilities = {'short': confidence, 'long': (1-confidence)/2, 'hold': (1-confidence)/2}
            else:
                predicted_direction = 'hold'
                direction_probabilities = {'hold': confidence, 'long': (1-confidence)/2, 'short': (1-confidence)/2}
            
            return {
                'predicted_direction': predicted_direction,
                'confidence': confidence,
                'direction_probabilities': direction_probabilities,
                'method': 'converted_from_basic'
            }
            
        except Exception as e:
            print(f"⚠️ Error converting basic to directional prediction: {e}")
            return {'predicted_direction': 'hold', 'confidence': 0.5, 'direction_probabilities': {'long': 0.33, 'short': 0.33, 'hold': 0.34}}
    
    def _log_enhanced_market_update(self, asset_symbol: str, market_data: Dict):
        """🎯 Log enhanced market data updates with directional information"""
        current_time = datetime.now()
        
        # Throttle logging (every 45 seconds for enhanced analysis)
        if hasattr(self, '_last_enhanced_market_log'):
            if (current_time - self._last_enhanced_market_log).seconds < 45:
                return
        
        self._last_enhanced_market_log = current_time
        
        try:
            price = market_data.get('price', 0)
            rsi = market_data.get('rsi', 50)
            price_change_24h = market_data.get('price_change_24h', 0)
            volatility = market_data.get('volatility', 0.01)
            
            # Asset status indicator
            status_indicator = "🎯" if asset_symbol == self.current_asset else "📊"
            
            # 🎯 ENHANCED: Directional signals
            directional_signals = market_data.get('directional_signals', {})
            recommended_direction = market_data.get('recommended_direction', 'hold')
            signal_strength = market_data.get('signal_strength', 0.0)
            position_suggestion = market_data.get('position_suggestion', 'WAIT')
            
            # Direction emoji
            direction_emoji = {'long': '🟢', 'short': '🔴', 'hold': '⚪'}.get(recommended_direction, '❓')
            
            # RSI status
            rsi_status = ""
            if rsi > 95:
                rsi_status = " ⚠️ EXTREME OB"
            elif rsi < 5:
                rsi_status = " ⚠️ EXTREME OS"
            elif rsi > 80:
                rsi_status = " 📈 Overbought"
            elif rsi < 20:
                rsi_status = " 📉 Oversold"
            
            # Open positions info
            position_info = ""
            if hasattr(self.trade_executor, 'position_tracker'):
                open_position = self.trade_executor.position_tracker.get_open_position(asset_symbol)
                if open_position:
                    pos_direction = open_position['direction']
                    entry_price = open_position['entry_price']
                    current_pnl = ((price - entry_price) / entry_price * 100) if pos_direction == 'long' else ((entry_price - price) / entry_price * 100)
                    position_info = f", POS: {pos_direction.upper()} ({current_pnl:+.1f}%)"
            
            # Portfolio allocation info
            allocation_info = ""
            if self.multi_asset_service and len(self.asset_data) > 1:
                total_trades = sum(self.trade_counts.values())
                if total_trades > 0:
                    current_pct = (self.trade_counts[asset_symbol] / total_trades) * 100
                    target_pct = self.portfolio_allocation.get(asset_symbol, 0) * 100
                    allocation_info = f", Alloc: {current_pct:.0f}%/{target_pct:.0f}%"
            
            print(f"{status_indicator} {asset_symbol}: ${price:.4f}, RSI: {rsi:.1f}{rsi_status}, "
                  f"24h: {price_change_24h:+.1f}%, Vol: {volatility:.4f}")
            
            if directional_signals:
                long_sig = directional_signals.get('long', 0)
                short_sig = directional_signals.get('short', 0) 
                hold_sig = directional_signals.get('hold', 0)
                print(f"   {direction_emoji} Direction: {recommended_direction.upper()} ({signal_strength:.2f}) → {position_suggestion}")
                print(f"   📊 Signals: L:{long_sig:.2f} S:{short_sig:.2f} H:{hold_sig:.2f}{position_info}{allocation_info}")
            
        except Exception as e:
            print(f"⚠️ Enhanced market logging error for {asset_symbol}: {e}")
    
    def _log_market_update(self, asset_symbol: str, market_data: Dict):
        """Log basic market data updates (fallback)"""
        current_time = datetime.now()
        
        # Throttle logging (every 30 seconds)
        if hasattr(self, '_last_market_log'):
            if (current_time - self._last_market_log).seconds < 30:
                return
        
        self._last_market_log = current_time
        
        try:
            price = market_data.get('price', 0)
            rsi = market_data.get('rsi', 50)
            price_change_24h = market_data.get('price_change_24h', 0)
            volatility = market_data.get('volatility', 0.01)
            
            # Asset status indicator
            status_indicator = "🎯" if asset_symbol == self.current_asset else "📊"
            
            print(f"{status_indicator} {asset_symbol}: ${price:.4f}, RSI: {rsi:.1f}, "
                  f"24h: {price_change_24h:+.1f}%, Vol: {volatility:.4f}")
            
        except Exception as e:
            print(f"⚠️ Market logging error for {asset_symbol}: {e}")
    
    def select_optimal_trading_asset(self) -> str:
        """🎯 Select optimal asset for directional trading"""
        if not self.multi_asset_service or len(self.asset_data) < 2:
            return self.current_asset
        
        try:
            # Get portfolio allocation target
            total_trades = sum(self.trade_counts.values())
            
            if total_trades == 0:
                return 'SOL'  # Start with SOL
            
            # Find most underallocated asset
            max_deficit = 0
            portfolio_target = self.current_asset
            
            for asset, target_allocation in self.portfolio_allocation.items():
                current_allocation = self.trade_counts[asset] / total_trades
                deficit = target_allocation - current_allocation
                
                if deficit > max_deficit:
                    max_deficit = deficit
                    portfolio_target = asset
            
            # 🎯 ENHANCED: Get directional signal-based recommendation
            signal_target = self.current_asset
            best_signal_strength = 0
            
            if hasattr(self, 'multi_asset_signals') and self.multi_asset_signals:
                try:
                    # Use enhanced directional analysis
                    signals = self.multi_asset_signals.analyze_directional_multi_asset_conditions(self.asset_data)
                    signal_target = self.multi_asset_signals.get_best_directional_asset_to_trade(signals)
                    
                    if signal_target and signal_target in signals:
                        best_signal_strength = signals[signal_target].get('confidence', 0)
                    
                except Exception as e:
                    print(f"⚠️ Enhanced directional signal error: {e}")
            
            # Decision logic: strong directional signals can override portfolio allocation
            final_asset = portfolio_target
            
            if signal_target and signal_target in self.asset_data and best_signal_strength > 0.75:
                final_asset = signal_target
                print(f"🎯 Strong directional signal override: {portfolio_target} → {signal_target} (strength: {best_signal_strength:.2f})")
            elif signal_target and signal_target in self.asset_data and best_signal_strength > 0.6:
                # Weigh signal vs allocation
                if max_deficit < 0.15:  # If allocation deficit is small
                    final_asset = signal_target
                    print(f"🎯 Directional signal preference: {portfolio_target} → {signal_target} (strength: {best_signal_strength:.2f})")
            
            # Execute asset switch if needed
            if final_asset != self.current_asset:
                print(f"🔄 Asset switch: {self.current_asset} → {final_asset}")
                self.current_asset = final_asset
                self.session_stats['asset_switches'] += 1
                
                # Update market data for new asset
                if final_asset in self.latest_market_data:
                    self.trade_executor.update_market_data(self.latest_market_data[final_asset], final_asset)
            
            return self.current_asset
            
        except Exception as e:
            print(f"❌ Asset selection error: {e}")
            return self.current_asset
    
    def execute_enhanced_trading_cycle(self) -> Dict[str, Any]:
        """🎯 Execute enhanced directional trading cycle"""
        cycle_start = datetime.now()
        
        # Calculate adaptive parameters
        adaptive_params = self._calculate_adaptive_parameters()
        cycle_size = adaptive_params['cycle_size']
        
        print(f"\n🎯 ENHANCED DIRECTIONAL TRADING CYCLE {self.session_stats['cycles_completed'] + 1}")
        print(f"   • Size: {cycle_size} trades (adaptive)")
        print(f"   • Asset: {self.current_asset}")
        print(f"   • Directional Trading: {'✅ ENABLED' if self.directional_enabled else '❌ DISABLED'}")
        print(f"   • Market Volatility: {self.market_volatility:.4f}")
        
        # Multi-asset: select optimal asset for directional trading
        if self.multi_asset_service:
            old_asset = self.current_asset
            self.select_optimal_trading_asset()
            if old_asset != self.current_asset:
                print(f"   🔄 Asset switched for directional opportunity: {old_asset} → {self.current_asset}")
        
        # Cycle statistics
        cycle_stats = {
            'executed': 0,
            'profitable': 0,
            'long_trades': 0,
            'short_trades': 0,
            'hold_actions': 0,
            'long_wins': 0,
            'short_wins': 0,
            'position_changes': 0,
            'skipped': 0,
            'reasons': {},
            'total_pnl': 0.0
        }
        
        # Execute directional trades
        for i in range(cycle_size):
            try:
                trade_number = self.state['count'] + 1
                
                # Log every 8th trade progress
                if i % 8 == 0:
                    print(f"🎯 Directional Trade {trade_number} (#{i + 1}/{cycle_size})")
                
                # 🎯 ENHANCED: Directional trading decision
                should_trade, direction, reason = self.should_execute_directional_trade()
                
                if should_trade:
                    # Track reason
                    cycle_stats['reasons'][reason] = cycle_stats['reasons'].get(reason, 0) + 1
                    
                    # 🎯 Execute directional trade
                    trade_result = self.trade_executor.execute_directional_trade(
                        settings, 
                        self.current_asset,
                        direction
                    )
                    
                    if trade_result:
                        cycle_stats['executed'] += 1
                        self.session_stats['total_trades_executed'] += 1
                        
                        # Track asset
                        self.trade_counts[self.current_asset] += 1
                        
                        # 🎯 ENHANCED: Track directional performance
                        trade_direction = trade_result.direction
                        is_profitable = trade_result.profitable
                        trade_pnl = trade_result.pnl
                        
                        # Update cycle stats
                        if trade_direction == 'long':
                            cycle_stats['long_trades'] += 1
                            self.session_stats['long_trades'] += 1
                            self.directional_performance['long_trades'] += 1
                            self.directional_performance['long_pnl'] += trade_pnl
                            if is_profitable:
                                cycle_stats['long_wins'] += 1
                                self.session_stats['long_wins'] += 1
                                self.directional_performance['long_wins'] += 1
                        elif trade_direction == 'short':
                            cycle_stats['short_trades'] += 1
                            self.session_stats['short_trades'] += 1
                            self.directional_performance['short_trades'] += 1
                            self.directional_performance['short_pnl'] += trade_pnl
                            if is_profitable:
                                cycle_stats['short_wins'] += 1
                                self.session_stats['short_wins'] += 1
                                self.directional_performance['short_wins'] += 1
                        elif trade_direction == 'hold':
                            cycle_stats['hold_actions'] += 1
                            self.session_stats['hold_actions'] += 1
                            self.directional_performance['hold_actions'] += 1
                        
                        # Track overall profitability
                        if is_profitable:
                            cycle_stats['profitable'] += 1
                            self.session_stats['profitable_trades'] += 1
                        
                        # Track total P&L
                        cycle_stats['total_pnl'] += trade_pnl
                        
                        # Track position changes
                        if 'open' in trade_result.action or 'close' in trade_result.action:
                            cycle_stats['position_changes'] += 1
                        
                        self.state['count'] += 1
                        
                        # 🎯 Enhanced logging for directional trades
                        pnl_pct = trade_result.pnl_percentage
                        
                        # Log significant trades
                        if (abs(pnl_pct) > 0.1 or trade_direction != 'hold' or 
                            'strong' in reason or 'signal' in reason):
                            profit_status = "✅ PROFIT" if is_profitable else "❌ LOSS"
                            direction_emoji = {'long': '🟢', 'short': '🔴', 'hold': '⚪'}.get(trade_direction, '❓')
                            print(f"   {direction_emoji} {profit_status}: {trade_direction.upper()} P&L {trade_pnl:+.6f} ({pnl_pct:+.2f}%) - {reason}")
                    
                else:
                    cycle_stats['skipped'] += 1
                    cycle_stats['reasons'][reason] = cycle_stats['reasons'].get(reason, 0) + 1
                    
                    # Log skip reasons periodically
                    if i % 15 == 0 and i > 0:  # Every 15th skip
                        print(f"   ⏸️ Recent skips: {reason}")
                
                # Progress check every 12 trades
                if (i + 1) % 12 == 0:
                    self._log_directional_cycle_progress(i + 1, cycle_size, cycle_stats)
                
                # Small delay between trades (longer for analysis)
                time.sleep(0.05)
                
            except Exception as e:
                print(f"❌ Directional trade execution error: {e}")
                continue
        
        # Calculate cycle performance
        cycle_duration = (datetime.now() - cycle_start).total_seconds()
        cycle_win_rate = (cycle_stats['profitable'] / cycle_stats['executed']) if cycle_stats['executed'] > 0 else 0.5
        
        # Update performance tracking
        self.cycle_performance.append(cycle_win_rate)
        if len(self.cycle_performance) > 10:
            self.cycle_performance = self.cycle_performance[-10:]
        
        self.recent_win_rate = sum(self.cycle_performance) / len(self.cycle_performance)
        self.session_stats['cycles_completed'] += 1
        
        # Add cycle metadata
        cycle_stats.update({
            'win_rate': cycle_win_rate,
            'duration_seconds': cycle_duration,
            'cycle_number': self.session_stats['cycles_completed'],
            'asset': self.current_asset,
            'avg_pnl_per_trade': cycle_stats['total_pnl'] / max(1, cycle_stats['executed'])
        })
        
        # Enhanced cycle summary logging
        self._log_enhanced_directional_cycle_summary(cycle_stats)
        
        return cycle_stats
    
    def should_execute_directional_trade(self) -> Tuple[bool, str, str]:
        """🎯 ENHANCED: Determine if and what type of directional trade to execute"""
        
        if self.current_asset not in self.latest_market_data:
            return True, 'hold', "no_market_data_fallback"
        
        market_data = self.latest_market_data[self.current_asset]
        
        # Get directional signals
        directional_signals = market_data.get('directional_signals', {})
        recommended_direction = market_data.get('recommended_direction', 'hold')
        signal_strength = market_data.get('signal_strength', 0.0)
        position_suggestion = market_data.get('position_suggestion', 'WAIT')
        
        # Get current market conditions
        rsi = market_data.get('rsi', 50)
        volatility = market_data.get('volatility', 0.02)
        price_change_24h = market_data.get('price_change_24h', 0)
        
        # Check for existing position
        has_position = False
        current_position_direction = None
        if hasattr(self.trade_executor, 'position_tracker'):
            has_position = self.trade_executor.position_tracker.has_open_position(self.current_asset)
            if has_position:
                position = self.trade_executor.position_tracker.get_open_position(self.current_asset)
                current_position_direction = position['direction']
        
        # 🎯 DIRECTIONAL TRADING LOGIC
        
        # 1. Strong directional signals (highest priority)
        if signal_strength > 0.8:
            if recommended_direction in ['long', 'short']:
                return True, recommended_direction, f"strong_{recommended_direction}_signal"
            elif recommended_direction == 'hold':
                if has_position:
                    return True, 'hold', "strong_hold_signal_close_position"
                else:
                    return True, 'hold', "strong_hold_signal"
        
        # 2. Good directional signals with confirmation
        elif signal_strength > 0.65:
            # RSI confirmation
            if recommended_direction == 'long' and rsi < 35:
                return True, 'long', "good_long_signal_rsi_confirm"
            elif recommended_direction == 'short' and rsi > 65:
                return True, 'short', "good_short_signal_rsi_confirm"
            elif recommended_direction == 'hold' and 40 <= rsi <= 60:
                return True, 'hold', "good_hold_signal_neutral_rsi"
        
        # 3. Extreme RSI conditions (reversal opportunities)
        if rsi <= 15:
            return True, 'long', "extreme_oversold_reversal"
        elif rsi >= 85:
            return True, 'short', "extreme_overbought_reversal"
        
        # 4. Strong momentum opportunities
        if abs(price_change_24h) > 8:
            if price_change_24h < -8 and volatility > 0.03:
                return True, 'long', "strong_dip_buying"
            elif price_change_24h > 8 and volatility > 0.03:
                return True, 'short', "strong_rally_shorting"
        
        # 5. Position management priority
        if has_position:
            # Let the trade executor handle position management
            # Don't interfere with existing positions unless strong signal
            if signal_strength < 0.4:
                return False, 'hold', "position_management_priority"
        
        # 6. Moderate signals with probability
        if signal_strength > 0.5:
            # Use probability based on signal strength
            trade_probability = signal_strength * 0.8  # Max 80% probability
            if random.random() < trade_probability:
                return True, recommended_direction, f"probability_{recommended_direction}_signal"
            else:
                return False, 'hold', f"probability_failed_{recommended_direction}"
        
        # 7. Low volatility = wait
        if volatility < 0.005:
            return False, 'hold', "low_volatility_wait"
        
        # 8. Very high volatility = caution
        if volatility > 0.1:
            if signal_strength > 0.7:  # Only trade high vol with strong signals
                return True, recommended_direction, "high_vol_strong_signal"
            else:
                return False, 'hold', "high_volatility_caution"
        
        # 9. Neutral conditions - occasional trading
        if 0.3 <= signal_strength <= 0.7:
            # Trade occasionally in neutral conditions
            if random.random() < 0.3:  # 30% chance
                return True, recommended_direction, "neutral_conditions_trade"
            else:
                return False, 'hold', "neutral_conditions_wait"
        
        # 10. Default: hold/wait
        return False, 'hold', "default_conservative_hold"
    
    def _calculate_adaptive_parameters(self) -> Dict[str, float]:
        """Calculate adaptive trading parameters for directional trading"""
        base_cycle_size = settings.get("trades_per_cycle", 30)  # Reduced for directional
        base_delay = settings.get("cycle_delay_seconds", 45)    # Increased for analysis
        
        # Directional performance factor
        directional_factor = 1.0
        if self.directional_performance['long_trades'] + self.directional_performance['short_trades'] > 10:
            total_directional = self.directional_performance['long_trades'] + self.directional_performance['short_trades']
            directional_wins = self.directional_performance['long_wins'] + self.directional_performance['short_wins']
            directional_win_rate = directional_wins / total_directional
            
            if directional_win_rate > 0.6:
                directional_factor = 1.2  # Increase when directional trading is working
            elif directional_win_rate < 0.4:
                directional_factor = 0.8  # Decrease when directional trading is not working
        
        # Volatility factor (more important for directional trading)
        volatility_factor = 1.0
        if self.market_volatility > 0.06:
            volatility_factor = 0.7  # Reduce significantly in high volatility
        elif self.market_volatility < 0.008:
            volatility_factor = 1.3  # Increase in low volatility (more opportunities)
        
        # Performance factor
        performance_factor = 1.0
        if len(self.cycle_performance) >= 3:
            recent_avg = sum(self.cycle_performance[-3:]) / 3
            if recent_avg > 0.65:
                performance_factor = 1.15
            elif recent_avg < 0.35:
                performance_factor = 0.85
        
        # Calculate final parameters
        final_factor = directional_factor * volatility_factor * performance_factor
        
        self.adaptive_cycle_size = max(20, min(50, int(base_cycle_size * final_factor)))
        self.adaptive_delay = max(30, min(90, int(base_delay / final_factor)))
        
        return {
            'cycle_size': self.adaptive_cycle_size,
            'delay': self.adaptive_delay,
            'directional_factor': directional_factor,
            'volatility_factor': volatility_factor,
            'performance_factor': performance_factor,
            'final_factor': final_factor
        }
    
    def _log_directional_cycle_progress(self, current: int, total: int, stats: Dict):
        """Log directional cycle progress"""
        executed = stats['executed']
        profitable = stats['profitable']
        long_trades = stats['long_trades']
        short_trades = stats['short_trades']
        hold_actions = stats['hold_actions']
        total_pnl = stats['total_pnl']
        win_rate = (profitable / executed) if executed > 0 else 0
        
        print(f"   🎯 Progress: {current}/{total}, Executed: {executed}, Profitable: {profitable} ({win_rate:.1%})")
        print(f"   📊 Directions: L:{long_trades} S:{short_trades} H:{hold_actions}, Total P&L: ${total_pnl:.6f}")
    
    def _log_enhanced_directional_cycle_summary(self, cycle_stats: Dict):
        """🎯 Enhanced cycle summary with comprehensive directional tracking"""
        executed = cycle_stats['executed']
        profitable = cycle_stats['profitable']
        long_trades = cycle_stats['long_trades']
        short_trades = cycle_stats['short_trades']
        hold_actions = cycle_stats['hold_actions']
        long_wins = cycle_stats['long_wins']
        short_wins = cycle_stats['short_wins']
        total_pnl = cycle_stats['total_pnl']
        position_changes = cycle_stats['position_changes']
        win_rate = cycle_stats['win_rate']
        duration = cycle_stats['duration_seconds']
        avg_pnl = cycle_stats['avg_pnl_per_trade']
        
        print(f"\n✅ DIRECTIONAL CYCLE {cycle_stats['cycle_number']} COMPLETE ({self.current_asset})")
        print(f"   📊 Overview: {executed} executed, {profitable} profitable ({win_rate:.1%})")
        print(f"   🎯 Directions: LONG {long_trades}({long_wins}W), SHORT {short_trades}({short_wins}W), HOLD {hold_actions}")
        print(f"   💰 P&L: Total ${total_pnl:.6f}, Avg ${avg_pnl:.6f}/trade")
        print(f"   🔄 Position changes: {position_changes}")
        print(f"   ⏱️ Duration: {duration:.1f}s")
        
        # Directional performance analysis
        if long_trades > 0:
            long_win_rate = long_wins / long_trades
            print(f"   🟢 LONG Performance: {long_win_rate:.1%} win rate")
        
        if short_trades > 0:
            short_win_rate = short_wins / short_trades
            print(f"   🔴 SHORT Performance: {short_win_rate:.1%} win rate")
        
        # Enhanced trade reasons breakdown
        if cycle_stats['reasons']:
            reasons = cycle_stats['reasons']
            print(f"   📋 Trading Reasons:")
            for reason, count in reasons.items():
                if count > 0:
                    if 'long' in reason:
                        print(f"     🟢 {reason}: {count}")
                    elif 'short' in reason:
                        print(f"     🔴 {reason}: {count}")
                    elif 'hold' in reason:
                        print(f"     ⚪ {reason}: {count}")
                    elif 'skip' in reason or 'wait' in reason:
                        print(f"     ⏸️ {reason}: {count}")
                    else:
                        print(f"     🎯 {reason}: {count}")
        
        # Current position status
        if hasattr(self.trade_executor, 'position_tracker'):
            open_positions = self.trade_executor.position_tracker.get_all_open_positions()
            if open_positions:
                print(f"   🎯 Open Positions:")
                for asset, position in open_positions.items():
                    direction = position['direction']
                    entry_price = position['entry_price']
                    current_price = self.latest_market_data.get(asset, {}).get('price', entry_price)
                    
                    if direction == 'long':
                        current_pnl_pct = ((current_price - entry_price) / entry_price) * 100
                    elif direction == 'short':
                        current_pnl_pct = ((entry_price - current_price) / entry_price) * 100
                    else:
                        current_pnl_pct = 0
                    
                    emoji = '🟢' if direction == 'long' else '🔴'
                    print(f"     {emoji} {asset}: {direction.upper()} @ ${entry_price:.4f} (P&L: {current_pnl_pct:+.1f}%)")
        
        print(f"   📈 Recent win rate: {self.recent_win_rate:.1%}")
        
        # Performance trend analysis
        if len(self.cycle_performance) >= 3:
            trend_direction = "📈" if self.cycle_performance[-1] > self.cycle_performance[-3] else "📉"
            recent_avg = sum(self.cycle_performance[-3:]) / 3
            print(f"   {trend_direction} 3-cycle trend: {recent_avg:.1%}")
    
    def _log_session_summary(self):
        """Log comprehensive session summary with enhanced directional metrics"""
        print(f"\n📊 ENHANCED DIRECTIONAL SESSION SUMMARY:")
        print(f"   • Total trades: {self.session_stats['total_trades_executed']}")
        print(f"   • Profitable: {self.session_stats['profitable_trades']}")
        total_win_rate = self.session_stats['profitable_trades'] / max(1, self.session_stats['total_trades_executed'])
        print(f"   • Overall win rate: {total_win_rate:.1%}")
        print(f"   • Recent win rate: {self.recent_win_rate:.1%}")
        
        # 🎯 ENHANCED: Directional performance breakdown
        print(f"   🎯 DIRECTIONAL BREAKDOWN:")
        long_trades = self.session_stats['long_trades']
        short_trades = self.session_stats['short_trades']
        hold_actions = self.session_stats['hold_actions']
        long_wins = self.session_stats['long_wins']
        short_wins = self.session_stats['short_wins']
        
        if long_trades > 0:
            long_win_rate = long_wins / long_trades
            long_pnl = self.directional_performance['long_pnl']
            print(f"     🟢 LONG: {long_trades} trades, {long_wins} wins ({long_win_rate:.1%}), P&L: ${long_pnl:.6f}")
        
        if short_trades > 0:
            short_win_rate = short_wins / short_trades
            short_pnl = self.directional_performance['short_pnl']
            print(f"     🔴 SHORT: {short_trades} trades, {short_wins} wins ({short_win_rate:.1%}), P&L: ${short_pnl:.6f}")
        
        if hold_actions > 0:
            print(f"     ⚪ HOLD: {hold_actions} actions")
        
        print(f"   • Cycles completed: {self.session_stats['cycles_completed']}")
        print(f"   • Asset switches: {self.session_stats['asset_switches']}")
        print(f"   • Directional switches: {self.session_stats.get('directional_switches', 0)}")
        
        # Portfolio allocation status
        total_trades = sum(self.trade_counts.values())
        if total_trades > 0:
            print(f"   📊 Portfolio allocation:")
            for asset, count in self.trade_counts.items():
                current_pct = (count / total_trades) * 100
                target_pct = self.portfolio_allocation[asset] * 100
                status = "✅" if abs(current_pct - target_pct) < 15 else "⚠️"
                print(f"     {status} {asset}: {count} trades ({current_pct:.1f}% vs {target_pct:.1f}% target)")
        
        # ML status
        if self.ml_integration:
            print(f"   🤖 ML predictions generated: {self.ml_prediction_count}")
            if self.last_ml_training:
                print(f"   🤖 Last ML training: {self.last_ml_training.strftime('%H:%M:%S')}")
        
        # Position status
        if hasattr(self.trade_executor, 'position_tracker'):
            position_summary = self.trade_executor.position_tracker.get_position_summary()
            print(f"   🎯 Position Summary: {position_summary['total_positions']} total, {position_summary['open_positions']} open")
        
        print(f"   🎯 Current trading asset: {self.current_asset}")
        if self.current_asset in self.latest_market_data:
            market_data = self.latest_market_data[self.current_asset]
            price = market_data.get('price', 0)
            rsi = market_data.get('rsi', 50)
            recommended_direction = market_data.get('recommended_direction', 'hold')
            print(f"   📊 Current {self.current_asset}: ${price:.4f}, RSI: {rsi:.1f}, Direction: {recommended_direction.upper()}")
    
    def load_state(self):
        """Load application state from file"""
        try:
            if os.path.exists(STATE_FILE):
                with open(STATE_FILE, "r") as f:
                    saved_state = json.load(f)
                    self.state.update(saved_state)
                    if "count" not in self.state:
                        self.state["count"] = 0
                print(f"📂 Enhanced state loaded: {self.state['count']} transactions")
            else:
                self.state = {"count": 0, "session_start": datetime.now().isoformat()}
        except Exception as e:
            print(f"⚠️ State loading error: {e}")
            self.state = {"count": 0, "session_start": datetime.now().isoformat()}
    
    def save_state(self) -> bool:
        """Save enhanced application state to file"""
        try:
            os.makedirs("data", exist_ok=True)
            self.state['last_save'] = datetime.now().isoformat()
            self.state['session_stats'] = self.session_stats
            self.state['trade_counts'] = self.trade_counts
            self.state['recent_win_rate'] = self.recent_win_rate
            self.state['directional_performance'] = self.directional_performance
            self.state['directional_enabled'] = self.directional_enabled
            
            with open(STATE_FILE, "w") as f:
                json.dump(self.state, f, indent=2)
            return True
        except Exception as e:
            print(f"❌ State saving error: {e}")
            return False
    
    def start(self):
        """Start the enhanced directional multi-asset trading bot"""
        start_time = datetime.now()
        
        print("🎯 STARTING ENHANCED DIRECTIONAL TRADING BOT")
        print("🚀 NOVA FUNKCJONALNOŚĆ: ZARABIANIE NA WZROSTACH I SPADKACH!")
        print(f"⏰ Session start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Setup directories
        os.makedirs("data", exist_ok=True)
        os.makedirs("data/results", exist_ok=True)
        if ML_AVAILABLE:
            os.makedirs("ml", exist_ok=True)
            os.makedirs("ml/models", exist_ok=True)
        
        # Load state
        self.load_state()
        start_count = self.state["count"]
        
        print(f"📊 Starting from transaction #{start_count + 1}")
        
        # Start enhanced market data services
        if not self.start_market_data_services():
            print("⚠️ Continuing in simulation mode (no live data)")
        
        # Wait for initial data and analysis
        print("⏳ Waiting for initial market data and directional analysis...")
        time.sleep(10)  # Longer wait for enhanced analysis
        
        print(f"🎯 Ready for directional trading! Current asset: {self.current_asset}")
        
        # Show initial directional status
        if self.current_asset in self.latest_market_data:
            market_data = self.latest_market_data[self.current_asset]
            recommended_direction = market_data.get('recommended_direction', 'hold')
            signal_strength = market_data.get('signal_strength', 0.0)
            print(f"🎯 Initial direction signal: {recommended_direction.upper()} (strength: {signal_strength:.2f})")
        
        # Main enhanced trading loop
        try:
            while True:
                # Execute enhanced directional trading cycle
                cycle_stats = self.execute_enhanced_trading_cycle()
                
                # Save state
                if self.save_state():
                    if cycle_stats['cycle_number'] % 5 == 0:  # Log save every 5 cycles
                        print(f"💾 Enhanced state saved: {self.state['count']} transactions")
                
                # Session summary every 8 cycles (less frequent due to more data)
                if cycle_stats['cycle_number'] % 8 == 0:
                    self._log_session_summary()
                
                # Adaptive delay (longer for directional analysis)
                delay = self.adaptive_delay
                print(f"⏳ Directional analysis break: {delay}s...")
                time.sleep(delay)
                
        except KeyboardInterrupt:
            print("\n🛑 Enhanced Directional Bot stopped by user")
        except Exception as e:
            print(f"\n💥 Unexpected error in directional bot: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._cleanup_and_exit(start_time, start_count)
    
    def _cleanup_and_exit(self, start_time: datetime, start_count: int):
        """Enhanced cleanup and exit procedures"""
        print("\n🧹 Cleaning up enhanced directional bot...")
        
        # Stop services
        if self.multi_asset_service:
            try:
                self.multi_asset_service.stop_tracking()
                print("✅ Enhanced multi-asset service stopped")
            except:
                pass
        
        if self.market_service:
            try:
                self.market_service.stop_stream()
                print("✅ Market service stopped")
            except:
                pass
        
        # Save final state
        if self.save_state():
            print("💾 Final enhanced state saved")
        
        # Final session report with directional performance
        session_duration = datetime.now() - start_time
        total_new_trades = self.state['count'] - start_count
        
        print(f"\n🏁 ENHANCED DIRECTIONAL SESSION COMPLETE")
        print(f"   • Duration: {session_duration}")
        print(f"   • New trades: {total_new_trades}")
        print(f"   • Total trades: {self.state['count']:,}")
        print(f"   • Final win rate: {self.recent_win_rate:.1%}")
        
        # Directional performance summary
        long_trades = self.directional_performance['long_trades']
        short_trades = self.directional_performance['short_trades']
        long_wins = self.directional_performance['long_wins']
        short_wins = self.directional_performance['short_wins']
        long_pnl = self.directional_performance['long_pnl']
        short_pnl = self.directional_performance['short_pnl']
        
        print(f"   🎯 DIRECTIONAL SUMMARY:")
        if long_trades > 0:
            long_win_rate = long_wins / long_trades
            print(f"     🟢 LONG: {long_trades} trades, {long_win_rate:.1%} win rate, ${long_pnl:.6f} P&L")
        
        if short_trades > 0:
            short_win_rate = short_wins / short_trades
            print(f"     🔴 SHORT: {short_trades} trades, {short_win_rate:.1%} win rate, ${short_pnl:.6f} P&L")
        
        total_directional_pnl = long_pnl + short_pnl
        print(f"     💰 Total Directional P&L: ${total_directional_pnl:.6f}")
        
        print(f"   🎯 Active asset: {self.current_asset}")
        
        if self.current_asset in self.latest_market_data:
            market_data = self.latest_market_data[self.current_asset]
            final_price = market_data.get('price', 0)
            final_direction = market_data.get('recommended_direction', 'hold')
            print(f"   📊 Final {self.current_asset}: ${final_price:.4f}, Direction: {final_direction.upper()}")
        
        print("✅ Enhanced Directional Bot shutdown complete")


if __name__ == "__main__":
    # Initialize and start the enhanced directional bot
    bot = EnhancedDirectionalTradingBot()
    bot.start()