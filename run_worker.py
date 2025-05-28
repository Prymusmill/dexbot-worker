# run_worker.py - OPTIMIZED with Adaptive ML Trading
import os
import sys
import time
import json
import csv
import pandas as pd
import threading
from datetime import datetime
from typing import Dict

# Wyłącz git checks
os.environ['GIT_PYTHON_REFRESH'] = 'quiet'
os.environ['GIT_PYTHON_GIT_EXECUTABLE'] = ''

# Import local modules
try:
    from config.settings import SETTINGS as settings
    from core.trade_executor import get_trade_executor
    from core.market_data import create_market_data_service, TradingSignals
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Enhanced ML Integration
ML_AVAILABLE = False
try:
    from ml.price_predictor import MLTradingIntegration
    ML_AVAILABLE = True
    print("✅ Enhanced ML modules available")
except ImportError as e:
    print(f"⚠️ ML modules not available: {e}")
    ML_AVAILABLE = False

STATE_FILE = "data/state.json"
MEMORY_FILE = "data/memory.csv"

class OptimizedTradingBot:
    def __init__(self):
        self.trade_executor = get_trade_executor()
        self.market_service = None
        self.latest_market_data = None
        self.trading_signals = TradingSignals()
        self.state = {"count": 0}
        
        # Enhanced ML attributes
        self.ml_predictions = {}
        self.ml_prediction_count = 0
        self.last_ml_training = None
        self.ml_performance_history = []
        
        # Adaptive trading attributes
        self.current_confidence = 0.5
        self.market_volatility = 0.01
        self.adaptive_cycle_size = settings.get("trades_per_cycle", 50)
        self.adaptive_delay = settings.get("cycle_delay_seconds", 30)
        
        # Performance tracking
        self.cycle_performance = []
        self.recent_win_rate = 0.5
        
        # Enhanced ML Integration
        if ML_AVAILABLE:
            try:
                self.ml_integration = MLTradingIntegration()
                print("🤖 Enhanced ML integration initialized")
            except Exception as e:
                print(f"⚠️ Enhanced ML integration failed: {e}")
                self.ml_integration = None
        else:
            self.ml_integration = None
            
    def on_market_data_update(self, market_data: Dict):
        """Enhanced callback with volatility tracking"""
        self.latest_market_data = market_data
        self.trade_executor.update_market_data(market_data)
        
        # Update market volatility
        self.market_volatility = market_data.get('volatility', 0.01)
        
        # Log market data with ML info
        if hasattr(self, '_last_market_log'):
            if (datetime.now() - self._last_market_log).seconds >= 30:
                self._log_enhanced_market_data(market_data)
                self._last_market_log = datetime.now()
        else:
            self._log_enhanced_market_data(market_data)
            self._last_market_log = datetime.now()
    
    def _log_enhanced_market_data(self, market_data: Dict):
        """Enhanced market data logging with ML insights"""
        price = market_data.get('price', 0)
        rsi = market_data.get('rsi', 0)
        volatility = market_data.get('volatility', 0.01)
        trend = 'up' if market_data.get('price_change_24h', 0) > 0 else 'down'
        
        # ML insights
        ml_info = ""
        if self.ml_predictions:
            direction = self.ml_predictions.get('direction', 'unknown')
            confidence = self.ml_predictions.get('confidence', 0)
            agreement = self.ml_predictions.get('model_agreement', 0)
            ml_info = f", ML: {direction.upper()} ({confidence:.2f} conf, {agreement:.2f} agree)"
        
        # Adaptive info
        adaptive_info = f", Adaptive: {self.adaptive_cycle_size} trades/{self.adaptive_delay}s"
        
        print(f"📊 Market: SOL/USDC ${price:.4f}, RSI: {rsi:.1f}, Vol: {volatility:.4f}, "
              f"24h: {trend}{ml_info}{adaptive_info}")
    
    def update_ml_predictions(self):
        """Enhanced ML predictions with ensemble"""
        if not self.ml_integration:
            return
        
        try:
            # Get data for ML
            data_result = self.trade_executor.get_recent_transactions_hybrid(limit=500)
            
            if data_result and data_result['count'] >= 100:
                df = data_result['data']
                print(f"🤖 Generating enhanced ML predictions from {len(df)} transactions...")
                
                prediction = self.ml_integration.get_ensemble_prediction(df)
                
                if 'predicted_price' in prediction:
                    self.ml_predictions = prediction
                    self.ml_prediction_count += 1
                    self.current_confidence = prediction.get('confidence', 0.5)
                    
                    # Log detailed prediction
                    if self.ml_prediction_count % 5 == 1:
                        direction = prediction['direction']
                        confidence = prediction['confidence']
                        price_change = prediction['price_change_pct']
                        model_count = prediction.get('model_count', 1)
                        agreement = prediction.get('model_agreement', 0)
                        
                        print(f"🎯 Enhanced ML #{self.ml_prediction_count}: {direction.upper()} "
                              f"({price_change:+.2f}%, conf: {confidence:.2f}, "
                              f"{model_count} models, agree: {agreement:.2f})")
                
                # Check for retraining
                if self.ml_integration.should_retrain() and len(df) >= 200:
                    print("🔄 Starting enhanced model retraining...")
                    threading.Thread(
                        target=self._retrain_enhanced_models, 
                        args=(df,), 
                        daemon=True
                    ).start()
            else:
                available = data_result['count'] if data_result else 0
                print(f"⚠️ Need more data for enhanced ML ({available}/100 transactions)")
        
        except Exception as e:
            print(f"⚠️ Enhanced ML prediction error: {e}")
    
    def _retrain_enhanced_models(self, df):
        """Enhanced model retraining with ensemble"""
        try:
            print("🔄 Retraining enhanced ensemble models...")
            results = self.ml_integration.train_models(df)
            
            if results.get('success'):
                successful_models = results.get('successful_models', [])
                print(f"✅ Enhanced retraining complete! Models: {successful_models}")
                
                # Log detailed performance
                performance = self.ml_integration.get_model_performance()
                for model_name in successful_models:
                    if model_name in performance:
                        metrics = performance[model_name]
                        accuracy = metrics.get('accuracy', 0)
                        r2 = metrics.get('r2', 0)
                        weight = metrics.get('ensemble_weight', 0)
                        print(f"   • {model_name}: Acc {accuracy:.1f}%, R² {r2:.3f}, Weight {weight:.2f}")
                
                self.last_ml_training = datetime.now()
            else:
                print(f"⚠️ Enhanced retraining failed: {results.get('error')}")
        
        except Exception as e:
            print(f"❌ Enhanced retraining error: {e}")
    
    def calculate_adaptive_parameters(self) -> Dict:
        """Calculate adaptive trading parameters based on ML confidence and market conditions"""
        base_cycle_size = settings.get("trades_per_cycle", 50)
        base_delay = settings.get("cycle_delay_seconds", 30)
        
        # ML confidence factor
        confidence_factor = 1.0
        if self.ml_predictions and settings.get("adaptive_trading", True):
            confidence = self.current_confidence
            
            if confidence > 0.7:
                confidence_factor = settings.get("high_confidence_multiplier", 1.5)
            elif confidence < 0.3:
                confidence_factor = settings.get("low_confidence_multiplier", 0.7)
        
        # Market volatility factor
        volatility_factor = 1.0
        vol_threshold = settings.get("market_volatility_threshold", 0.05)
        if self.market_volatility > vol_threshold:
            volatility_factor = 0.8  # Reduce trading in high volatility
        
        # Recent performance factor
        performance_factor = 1.0
        if len(self.cycle_performance) >= 3:
            recent_performance = sum(self.cycle_performance[-3:]) / 3
            if recent_performance < 0.4:  # Poor recent performance
                performance_factor = 0.9
            elif recent_performance > 0.6:  # Good recent performance
                performance_factor = 1.1
        
        # Calculate final parameters
        final_factor = confidence_factor * volatility_factor * performance_factor
        
        self.adaptive_cycle_size = max(20, min(80, int(base_cycle_size * final_factor)))
        self.adaptive_delay = max(15, min(60, int(base_delay / final_factor)))
        
        return {
            'cycle_size': self.adaptive_cycle_size,
            'delay': self.adaptive_delay,
            'confidence_factor': confidence_factor,
            'volatility_factor': volatility_factor,
            'performance_factor': performance_factor,
            'final_factor': final_factor
        }
    
    def should_execute_trade_enhanced(self) -> bool:
        """Enhanced trading decision with ML integration"""
        if not self.latest_market_data:
            return True  # Fallback
        
        # Base market signals
        signals = self.trading_signals.analyze_market_conditions(self.latest_market_data)
        base_confidence = signals.get('confidence', 0.5)
        
        # Enhanced ML integration
        enhanced_confidence = base_confidence
        ml_factor = 1.0
        
        if self.ml_integration and self.ml_predictions and settings.get("ml_enabled", True):
            try:
                ml_confidence = self.ml_predictions.get('confidence', 0.5)
                ml_direction = self.ml_predictions.get('direction', 'neutral')
                model_agreement = self.ml_predictions.get('model_agreement', 0.5)
                
                # Strong ML signals
                if ml_confidence > 0.7 and model_agreement > 0.8:
                    if ml_direction == 'up':
                        enhanced_confidence = min(base_confidence + 0.4, 1.0)
                        ml_factor = 1.3
                    else:
                        enhanced_confidence = max(base_confidence - 0.2, 0.1)
                        ml_factor = 0.8
                
                # Moderate ML signals
                elif ml_confidence > 0.5:
                    if ml_direction == 'up':
                        enhanced_confidence = min(base_confidence + 0.2, 0.9)
                        ml_factor = 1.1
                    else:
                        enhanced_confidence = max(base_confidence - 0.1, 0.2)
                        ml_factor = 0.9
                
                # Log ML enhancement
                if abs(enhanced_confidence - base_confidence) > 0.1:
                    print(f"🧠 ML Enhanced: {base_confidence:.2f} → {enhanced_confidence:.2f} "
                          f"(ML: {ml_direction}, {ml_confidence:.2f}, agree: {model_agreement:.2f})")
            
            except Exception as e:
                print(f"⚠️ ML decision enhancement error: {e}")
        
        # Market volatility check
        vol_threshold = settings.get("market_volatility_threshold", 0.05)
        if self.market_volatility > vol_threshold:
            enhanced_confidence *= 0.8
        
        # Final decision with confidence threshold
        ml_threshold = settings.get("ml_confidence_threshold", 0.3)
        
        if enhanced_confidence > 0.6:
            return True
        elif enhanced_confidence > ml_threshold:
            # Probabilistic execution based on confidence
            import random
            return random.random() < (enhanced_confidence * ml_factor)
        else:
            return random.random() < 0.4  # Low probability fallback
    
    def execute_enhanced_trade_cycle(self):
        """Enhanced trading cycle with adaptive parameters"""
        # Calculate adaptive parameters
        adaptive_params = self.calculate_adaptive_parameters()
        cycle_size = adaptive_params['cycle_size']
        
        print(f"\n🚀 Enhanced Cycle - {cycle_size} trades (adaptive)")
        print(f"   • ML Confidence: {self.current_confidence:.2f}")
        print(f"   • Market Volatility: {self.market_volatility:.4f}")
        print(f"   • Confidence Factor: {adaptive_params['confidence_factor']:.2f}")
        print(f"   • Final Factor: {adaptive_params['final_factor']:.2f}")
        
        # Update ML predictions
        if self.ml_integration:
            self.update_ml_predictions()
        
        executed_in_cycle = 0
        profitable_in_cycle = 0
        
        for i in range(cycle_size):
            try:
                print(f"🔹 Transaction {self.state['count'] + 1} (#{i+1}/{cycle_size})")
                
                # Enhanced trading decision
                if self.should_execute_trade_enhanced():
                    # Execute trade with current market data
                    trade_result = self.trade_executor.execute_trade(settings, self.latest_market_data)
                    
                    if trade_result and hasattr(trade_result, 'profitable'):
                        executed_in_cycle += 1
                        if trade_result.profitable:
                            profitable_in_cycle += 1
                        
                        self.state["count"] += 1
                else:
                    print("⏸️ Trade skipped - unfavorable conditions")
                
                # Status check every 10 trades
                if (i + 1) % 10 == 0:
                    self.check_enhanced_status()
                
                # Adaptive delay between trades
                time.sleep(0.2)
                
            except Exception as e:
                print(f"❌ Trade execution error: {e}")
                continue
        
        # Calculate cycle performance
        cycle_win_rate = (profitable_in_cycle / executed_in_cycle) if executed_in_cycle > 0 else 0.5
        self.cycle_performance.append(cycle_win_rate)
        
        # Keep only last 10 cycles
        if len(self.cycle_performance) > 10:
            self.cycle_performance = self.cycle_performance[-10:]
        
        self.recent_win_rate = sum(self.cycle_performance) / len(self.cycle_performance)
        
        print(f"✅ Enhanced cycle complete: {executed_in_cycle}/{cycle_size} executed, "
              f"{profitable_in_cycle} profitable ({cycle_win_rate:.1%} cycle win rate)")
    
    def check_enhanced_status(self):
        """Enhanced status checking with ML insights"""
        # Get database status
        db_status = self.trade_executor.get_database_status()
        
        print(f"📊 Enhanced Status:")
        print(f"   • PostgreSQL: {db_status['postgresql_count']} transactions")
        print(f"   • CSV Backup: {db_status['csv_count']} transactions")
        print(f"   • Recent Win Rate: {self.recent_win_rate:.1%}")
        
        if self.latest_market_data:
            price = self.latest_market_data.get('price', 0)
            rsi = self.latest_market_data.get('rsi', 0)
            print(f"   • Current SOL Price: ${price:.4f}")
            print(f"   • RSI: {rsi:.1f}")
        
        # ML Status
        if self.ml_integration and self.ml_predictions:
            try:
                performance = self.ml_integration.get_model_performance()
                active_models = len(performance)
                
                print(f"   • ML Models Active: {active_models}")
                print(f"   • ML Predictions: {self.ml_prediction_count}")
                
                if self.ml_predictions:
                    direction = self.ml_predictions.get('direction', 'unknown')
                    confidence = self.ml_predictions.get('confidence', 0)
                    predicted_price = self.ml_predictions.get('predicted_price', 0)
                    print(f"   • ML Forecast: {direction.upper()} → ${predicted_price:.4f} ({confidence:.2f})")
                    
            except Exception as e:
                print(f"   • ML Status Error: {e}")
    
    def load_state(self):
        """Load application state"""
        try:
            if os.path.exists(STATE_FILE):
                with open(STATE_FILE, "r") as f:
                    self.state = json.load(f)
                    if "count" not in self.state:
                        self.state["count"] = 0
                print(f"📂 State loaded: {self.state['count']} transactions")
            else:
                self.state = {"count": 0}
        except Exception as e:
            print(f"⚠️ State loading error: {e}")
            self.state = {"count": 0}
    
    def save_state(self):
        """Save application state"""
        try:
            os.makedirs("data", exist_ok=True)
            with open(STATE_FILE, "w") as f:
                json.dump(self.state, f)
            return True
        except Exception as e:
            print(f"❌ State saving error: {e}")
            return False
    
    def start(self):
        """Start enhanced trading bot"""
        print("🚀 Starting OPTIMIZED DexBot with Enhanced ML & Adaptive Trading...")
        print(f"⏰ Start: {datetime.now()}")
        print(f"🎯 Settings: {settings['trades_per_cycle']} trades/{settings['cycle_delay_seconds']}s, ML: {settings.get('ml_enabled', True)}")
        
        # Setup directories
        os.makedirs("data", exist_ok=True)
        os.makedirs("data/results", exist_ok=True)
        if ML_AVAILABLE:
            os.makedirs("ml", exist_ok=True)
            os.makedirs("ml/models", exist_ok=True)
        
        # Load state
        self.load_state()
        start_count = self.state["count"]
        
        # Start market data service
        print("🌐 Connecting to enhanced market data...")
        self.market_service = create_market_data_service(self.on_market_data_update)
        
        if not self.market_service:
            print("⚠️ Market data connection failed - continuing in simulation mode")
        else:
            print("✅ Connected to live market data")
            time.sleep(3)  # Initial data load
        
        # Initial ML setup
        if ML_AVAILABLE and self.ml_integration and start_count >= 100:
            print("🤖 Initializing enhanced ML predictions...")
            self.update_ml_predictions()
        
        print(f"🎯 Starting from transaction #{start_count + 1}")
        
        # Main enhanced trading loop
        cycle = 0
        try:
            while True:
                cycle += 1
                
                # Execute enhanced trading cycle
                self.execute_enhanced_trade_cycle()
                
                # Save state
                if self.save_state():
                    print(f"💾 State saved: {self.state['count']} transactions")
                
                # Enhanced session stats
                total_executed = self.state["count"] - start_count
                print(f"\n📈 Enhanced Session Stats:")
                print(f"   • New transactions: {total_executed}")
                print(f"   • Total transactions: {self.state['count']:,}")
                print(f"   • Cycles completed: {cycle}")
                print(f"   • Recent win rate: {self.recent_win_rate:.1%}")
                print(f"   • Adaptive cycle size: {self.adaptive_cycle_size}")
                
                if self.latest_market_data:
                    price = self.latest_market_data.get('price', 0)
                    print(f"   • Current SOL price: ${price:.4f}")
                
                # Enhanced ML status
                if self.ml_predictions:
                    try:
                        direction = self.ml_predictions.get('direction', 'unknown')
                        confidence = self.ml_predictions.get('confidence', 0)
                        predicted_price = self.ml_predictions.get('predicted_price', 0)
                        model_count = self.ml_predictions.get('model_count', 0)
                        print(f"   • ML Ensemble: {direction.upper()} → ${predicted_price:.4f}")
                        print(f"   • ML Confidence: {confidence:.2f} ({model_count} models)")
                    except Exception as e:
                        print(f"   • ML Display Error: {e}")
                
                # Adaptive delay between cycles
                print(f"⏳ Enhanced break: {self.adaptive_delay}s before next cycle...")
                time.sleep(self.adaptive_delay)
                
        except KeyboardInterrupt:
            print("\n🛑 Optimized bot stopped by user")
        except Exception as e:
            print(f"\n💥 Unexpected error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Cleanup
            if self.market_service:
                self.market_service.stop_stream()
            
            if self.save_state():
                print(f"💾 Final state saved: {self.state['count']} transactions")
            
            # Final enhanced status
            try:
                print(f"\n🏁 Enhanced Bot Session Complete:")
                print(f"   • Total transactions: {self.state['count']:,}")
                print(f"   • Recent win rate: {self.recent_win_rate:.1%}")
                print(f"   • ML predictions generated: {self.ml_prediction_count}")
                
                if self.latest_market_data:
                    final_price = self.latest_market_data.get('price', 0)
                    print(f"   • Final SOL price: ${final_price:.4f}")
                
            except Exception as e:
                print(f"🏁 Session complete (status error: {e})")

if __name__ == "__main__":
    bot = OptimizedTradingBot()
    bot.start()