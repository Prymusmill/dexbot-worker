# run_worker.py - FIXED COMPLETE VERSION with GPT integration
import os
import sys
import time
import json
import csv
import pandas as pd
import threading
import random  # FIXED: Added missing import
from datetime import datetime
from typing import Dict
from ml.auto_retrainer import setup_auto_retraining

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

# DODAJ DO IMPORTÓW W run_worker.py (około linia 25)
try:
    from core.multi_asset_data import create_multi_asset_service, MultiAssetSignals
    MULTI_ASSET_AVAILABLE = True
    print("✅ Multi-asset modules available")
except ImportError as e:
    print(f"⚠️ Multi-asset modules not available: {e}")
    MULTI_ASSET_AVAILABLE = False

# GPT Analyzer import
try:
    from ml.gpt_analyzer import setup_gpt_enhanced_trading, format_gpt_analysis_for_logging
    GPT_AVAILABLE = True
    print("✅ GPT analyzer available")
except ImportError as e:
    print(f"⚠️ GPT analyzer not available: {e}")
    GPT_AVAILABLE = False

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
        # Core trading attributes
        self.trade_executor = get_trade_executor()
        self.market_service = None
        self.latest_market_data = None
        self.trading_signals = TradingSignals()
        self.state = {"count": 0}

        # 🚀 MULTI-ASSET ATTRIBUTES - NEW!
        self.multi_asset_service = None
        self.multi_asset_signals = None
        self.supported_assets = ['SOL', 'ETH', 'BTC']  # ← DODAJ 'BTC'!
        self.current_asset = 'SOL'  # Active trading asset
        self.asset_data = {}  # Store data for all assets

        # 🚀 PORTFOLIO ALLOCATION - NEW!
        self.portfolio_allocation = {
            'SOL': 0.4,   # 40% allocation
            'ETH': 0.35,  # 35% allocation  
            'BTC': 0.25   # 25% allocation
        }
        self.trade_counts = {'SOL': 0, 'ETH': 0, 'BTC': 0}  # Track trades per asset
        
        # Enhanced ML attributes
        self.ml_predictions = {}
        self.ml_prediction_count = 0
        self.last_ml_training = None
        self.ml_performance_history = []

        # Contrarian trading attributes
        self.contrarian_trade_count = 0
        self.contrarian_wins = 0

        # GPT attributes
        self.gpt_analysis_count = 0
        self.gpt_successful_predictions = 0
        self.last_gpt_analysis = None

        # Adaptive trading attributes
        self.current_confidence = 0.5
        self.market_volatility = 0.01
        self.adaptive_cycle_size = settings.get("trades_per_cycle", 50)
        self.adaptive_delay = settings.get("cycle_delay_seconds", 30)

        # Performance tracking
        self.cycle_performance = []
        self.recent_win_rate = 0.5

        # Enhanced ML Integration
        self.ml_integration = None
        if ML_AVAILABLE:
            try:
                self.ml_integration = MLTradingIntegration()
                print("🤖 Enhanced ML integration initialized")
            except Exception as e:
                print(f"⚠️ Enhanced ML integration failed: {e}")
                self.ml_integration = None

        # Initialize GPT analyzer
        self.gpt_analyzer = None
        self.gpt_enabled = False
        if GPT_AVAILABLE:
            try:
                self.gpt_analyzer = setup_gpt_enhanced_trading()
                print("🤖 GPT-enhanced ML system initialized")
                self.gpt_enabled = True
            except Exception as e:
                print(f"⚠️ GPT initialization failed: {e}")
                self.gpt_analyzer = None
                self.gpt_enabled = False

        # 🚀 MULTI-ASSET SIGNALS - NEW!
        if MULTI_ASSET_AVAILABLE:
            try:
                self.multi_asset_signals = MultiAssetSignals()
                print("📊 Multi-asset signal analyzer initialized")
            except Exception as e:
                print(f"⚠️ Multi-asset signals failed: {e}")
                self.multi_asset_signals = None

        # Auto-retraining integration
        self.auto_retrainer = None
        if ML_AVAILABLE and self.ml_integration is not None:
            try:
                from ml.auto_retrainer import setup_auto_retraining
                self.auto_retrainer = setup_auto_retraining(
                    ml_integration=self.ml_integration,
                    db_manager=self.trade_executor.db_manager if hasattr(self.trade_executor, 'db_manager') else None,
                    retrain_interval_hours=6,
                    min_new_samples=100,
                    performance_threshold=0.55
                )
                print("🔄 Auto-retraining service initialized")
            except Exception as e:
                print(f"⚠️ Auto-retraining setup failed: {e}")
                self.auto_retrainer = None
        else:
            print("⚠️ Auto-retraining skipped: ML integration not available")

        print(f"✅ OptimizedTradingBot initialized:")
        print(f"   • ML Integration: {'✅' if self.ml_integration else '❌'}")
        print(f"   • GPT Analyzer: {'✅' if self.gpt_enabled else '❌'}")
        print(f"   • Auto-retrainer: {'✅' if self.auto_retrainer else '❌'}")
        print(f"   • Contrarian Trading: ✅ Enabled")
        print(f"   • Multi-Asset: {'✅' if MULTI_ASSET_AVAILABLE else '❌'} ({len(self.supported_assets)} assets)")  # NEW!

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
        """Enhanced market data logging with ML insights + CONTRARIAN INFO"""
        price = market_data.get('price', 0)
        rsi = market_data.get('rsi', 0)
        volatility = market_data.get('volatility', 0.01)
        trend = 'up' if market_data.get('price_change_24h', 0) > 0 else 'down'
        price_change_24h = market_data.get('price_change_24h', 0)

        # ML insights
        ml_info = ""
        contrarian_info = ""
        
        if self.ml_predictions:
            direction = self.ml_predictions.get('direction', 'unknown')
            confidence = self.ml_predictions.get('confidence', 0)
            agreement = self.ml_predictions.get('model_agreement', 0)
            ml_info = f", ML: {direction.upper()} ({confidence:.2f} conf, {agreement:.2f} agree)"
            
            # CONTRARIAN ANALYSIS
            if direction == 'unprofitable' and confidence > 0.85:
                contrarian_score = self.calculate_contrarian_score(market_data)
                if contrarian_score >= 0.3:
                    contrarian_info = f", CONTRARIAN: {contrarian_score:.2f} score"

        # GPT insights
        gpt_info = ""
        if self.last_gpt_analysis:
            gpt_action = self.last_gpt_analysis.get('action', 'UNKNOWN')
            gpt_confidence = self.last_gpt_analysis.get('confidence', 0)
            gpt_info = f", GPT: {gpt_action} ({gpt_confidence:.2f})"

        # EXTREME CONDITIONS WARNING
        extreme_info = ""
        if rsi > 95:
            extreme_info = " ⚠️ EXTREME OVERBOUGHT"
        elif rsi < 5:
            extreme_info = " ⚠️ EXTREME OVERSOLD"
        elif rsi > 85:
            extreme_info = " 📈 Very Overbought"
        elif rsi < 15:
            extreme_info = " 📉 Very Oversold"

        # Adaptive info
        adaptive_info = f", Adaptive: {self.adaptive_cycle_size} trades/{self.adaptive_delay}s"

        print(f"📊 Market: SOL/USDC ${price:.4f}, RSI: {rsi:.1f}{extreme_info}, Vol: {volatility:.4f}, "
              f"24h: {trend} ({price_change_24h:+.1f}%){ml_info}{contrarian_info}{gpt_info}{adaptive_info}")

    def calculate_contrarian_score(self, market_data: Dict) -> float:
        """Calculate contrarian trading score based on extreme market conditions"""
        if not self.ml_predictions:
            return 0.0
            
        ml_direction = self.ml_predictions.get('direction', 'neutral')
        ml_confidence = self.ml_predictions.get('confidence', 0)
        
        if ml_direction != 'unprofitable' or ml_confidence < 0.85:
            return 0.0
            
        rsi = market_data.get('rsi', 50)
        price_change_24h = market_data.get('price_change_24h', 0)
        
        contrarian_score = 0.0
        
        # RSI Extremes
        if rsi > 85:
            contrarian_score += 0.4
        elif rsi < 15:
            contrarian_score += 0.4
        elif rsi > 80:
            contrarian_score += 0.3
        elif rsi < 20:
            contrarian_score += 0.3
        
        # Price momentum extremes
        if abs(price_change_24h) > 10:
            contrarian_score += 0.2
        elif abs(price_change_24h) > 5:
            contrarian_score += 0.1
        
        # ML confidence bonus
        if ml_confidence > 0.95:
            contrarian_score += 0.2
        elif ml_confidence > 0.9:
            contrarian_score += 0.1
            
        return contrarian_score

    def should_execute_trade_enhanced(self) -> bool:
        """Enhanced trading decision with ML, GPT integration AND CONTRARIAN LOGIC"""
        if not self.latest_market_data:
            return True

        # CONTRARIAN TRADING LOGIC - Check first!
        if self.ml_predictions:
            ml_direction = self.ml_predictions.get('direction', 'neutral')
            ml_confidence = self.ml_predictions.get('confidence', 0)
            
            if ml_direction == 'unprofitable' and ml_confidence > 0.85:
                contrarian_score = self.calculate_contrarian_score(self.latest_market_data)
                
                if contrarian_score >= 0.5:
                    print(f"🔄 CONTRARIAN TRADE TRIGGERED!")
                    print(f"   • ML: {ml_direction.upper()} ({ml_confidence:.1%} confidence)")
                    print(f"   • Contrarian Score: {contrarian_score:.2f}")
                    print(f"   • Strategy: BETTING AGAINST ML PREDICTION")
                    return True
                elif contrarian_score >= 0.3:
                    contrarian_probability = contrarian_score * 1.5
                    if random.random() < contrarian_probability:
                        print(f"🎲 CONTRARIAN GAMBLE: Score {contrarian_score:.2f}")
                        return True
                        
            # Regular ML skip logic (if not contrarian)
            if ml_direction == 'unprofitable' and ml_confidence > 0.8:
                rsi = self.latest_market_data.get('rsi', 50)
                if not (rsi > 90 or rsi < 10):
                    print(f"🚫 ML Skip: {ml_direction} with {ml_confidence:.1%} confidence")
                    return False

        # Regular trading logic with LOWERED thresholds
        confidence = 0.5
        if self.ml_predictions:
            confidence = self.ml_predictions.get('confidence', 0.5)

        # Market volatility check
        vol_threshold = settings.get("market_volatility_threshold", 0.05)
        if self.market_volatility > vol_threshold:
            confidence *= 0.8

        # LOWERED ML threshold for more trades
        ml_threshold = settings.get("ml_confidence_threshold", 0.2)

        if confidence > 0.6:
            return True
        elif confidence > ml_threshold:
            return random.random() < confidence
        else:
            return random.random() < 0.6  # Increased fallback probability

    def execute_enhanced_trade_cycle(self):
        """Enhanced trading cycle with CONTRARIAN TRACKING"""
        # Calculate adaptive parameters
        adaptive_params = self.calculate_adaptive_parameters()
        cycle_size = adaptive_params['cycle_size']

        print(f"\n🚀 Enhanced Cycle - {cycle_size} trades (adaptive)")
        print(f"   • ML Confidence: {self.current_confidence:.2f}")
        print(f"   • Market Volatility: {self.market_volatility:.4f}")

        # Update ML predictions
        if self.ml_integration:
            self.update_ml_predictions()

        executed_in_cycle = 0
        profitable_in_cycle = 0
        contrarian_trades = 0
        contrarian_wins = 0

        for i in range(cycle_size):
            try:
                print(f"🔹 Transaction {self.state['count'] + 1} (#{i + 1}/{cycle_size})")

                # Track if this is a contrarian trade
                is_contrarian = False
                if self.ml_predictions and self.latest_market_data:
                    ml_direction = self.ml_predictions.get('direction', 'neutral')
                    ml_confidence = self.ml_predictions.get('confidence', 0)
                    contrarian_score = self.calculate_contrarian_score(self.latest_market_data)
                    
                    if (ml_direction == 'unprofitable' and ml_confidence > 0.85 and 
                        contrarian_score >= 0.3):
                        is_contrarian = True

                # Enhanced trading decision
                if self.should_execute_trade_enhanced():
                    trade_result = self.trade_executor.execute_trade(settings, self.latest_market_data)

                    if trade_result and hasattr(trade_result, 'profitable'):
                        executed_in_cycle += 1
                        if trade_result.profitable:
                            profitable_in_cycle += 1
                        
                        # 🚀 TRACK PORTFOLIO ALLOCATION - NEW!
                        self.trade_counts[self.current_asset] += 1
                        
                        # Track contrarian performance
                        if is_contrarian:
                            contrarian_trades += 1
                            if trade_result.profitable:
                                contrarian_wins += 1
                                print(f"🎯 CONTRARIAN WIN! ({contrarian_wins}/{contrarian_trades})")
                            else:
                                print(f"💥 Contrarian loss ({contrarian_wins}/{contrarian_trades})")

                        self.state["count"] += 1
                else:
                    print("⏸️ Trade skipped - unfavorable conditions")

                # Status check every 10 trades
                if (i + 1) % 10 == 0:
                    self.check_enhanced_status()

                time.sleep(0.2)

            except Exception as e:
                print(f"❌ Trade execution error: {e}")
                continue

        # Calculate cycle performance
        cycle_win_rate = (profitable_in_cycle / executed_in_cycle) if executed_in_cycle > 0 else 0.5
        contrarian_win_rate = (contrarian_wins / contrarian_trades) if contrarian_trades > 0 else 0.0
        
        self.cycle_performance.append(cycle_win_rate)
        
        # Keep only last 10 cycles
        if len(self.cycle_performance) > 10:
            self.cycle_performance = self.cycle_performance[-10:]
            
        self.recent_win_rate = sum(self.cycle_performance) / len(self.cycle_performance)
        
        print(f"✅ Enhanced cycle complete: {executed_in_cycle}/{cycle_size} executed, "
              f"{profitable_in_cycle} profitable ({cycle_win_rate:.1%} cycle win rate)")
        
        if contrarian_trades > 0:
            print(f"🔄 Contrarian trades: {contrarian_trades}, wins: {contrarian_wins} "
                  f"({contrarian_win_rate:.1%} contrarian win rate)")
            
            # Update global contrarian stats
            self.contrarian_trade_count += contrarian_trades
            self.contrarian_wins += contrarian_wins

        # Return stats for tracking
        return {
            'executed': executed_in_cycle,
            'profitable': profitable_in_cycle,
            'contrarian_trades': contrarian_trades,
            'contrarian_wins': contrarian_wins,
            'cycle_win_rate': cycle_win_rate,
            'contrarian_win_rate': contrarian_win_rate
        }

    def calculate_adaptive_parameters(self) -> Dict:
        """Calculate adaptive trading parameters"""
        base_cycle_size = settings.get("trades_per_cycle", 50)
        base_delay = settings.get("cycle_delay_seconds", 30)

        confidence_factor = 1.0
        if self.ml_predictions and settings.get("adaptive_trading", True):
            confidence = self.current_confidence
            if confidence > 0.7:
                confidence_factor = settings.get("high_confidence_multiplier", 1.5)
            elif confidence < 0.3:
                confidence_factor = settings.get("low_confidence_multiplier", 0.7)

        final_factor = confidence_factor
        self.adaptive_cycle_size = max(20, min(80, int(base_cycle_size * final_factor)))
        self.adaptive_delay = max(15, min(60, int(base_delay / final_factor)))

        return {
            'cycle_size': self.adaptive_cycle_size,
            'delay': self.adaptive_delay,
            'confidence_factor': confidence_factor,
            'final_factor': final_factor
        }

    def update_ml_predictions(self):
        """Enhanced ML predictions with ensemble"""
        if not self.ml_integration:
            print("⚠️ ML integration not available")
            return

        try:
            data_result = self.trade_executor.get_recent_transactions_hybrid(limit=500)

            if data_result and data_result['count'] >= 100:
                df = data_result['data']
                print(f"🤖 Generating enhanced ML predictions from {len(df)} transactions...")

                try:
                    prediction = self.ml_integration.get_ensemble_prediction_with_reality_check(df)

                    if isinstance(prediction, dict) and 'predicted_profitable' in prediction:
                        self.ml_predictions = prediction
                        self.ml_prediction_count += 1
                        self.current_confidence = prediction.get('confidence', 0.5)

                        profitable = prediction['predicted_profitable']
                        probability = prediction.get('probability_profitable', 0.5)
                        confidence = prediction['confidence']
                        model_count = prediction.get('model_count', 1)
                        recommendation = prediction.get('recommendation', 'HOLD')

                        print(f"✅ ML Prediction #{self.ml_prediction_count}:")
                        print(f"   • Profitable: {'YES' if profitable else 'NO'} ({probability:.1%} prob)")
                        print(f"   • Recommendation: {recommendation}")
                        print(f"   • Confidence: {confidence:.2f}")
                        print(f"   • Models: {model_count}")

                    else:
                        self.ml_predictions = {}

                except Exception as pred_error:
                    print(f"❌ ML Prediction Generation Error: {pred_error}")
                    self.ml_predictions = {}

            else:
                available = data_result['count'] if data_result else 0
                print(f"⚠️ Need more data for enhanced ML ({available}/100 transactions)")

        except Exception as e:
            print(f"❌ Enhanced ML prediction error: {e}")

    def check_enhanced_status(self):
        """Enhanced status checking with ML and GPT insights"""
        db_status = self.trade_executor.get_database_status()

        print(f"📊 Enhanced Status:")
        print(f"   • PostgreSQL: {db_status['postgresql_count']} transactions")
        print(f"   • CSV Backup: {db_status['csv_count']} transactions")
        print(f"   • Recent Win Rate: {self.recent_win_rate:.1%}")
        print(f"   • Contrarian Trades: {self.contrarian_trade_count}")

        if self.latest_market_data:
            price = self.latest_market_data.get('price', 0)
            rsi = self.latest_market_data.get('rsi', 0)
            print(f"   • Current SOL Price: ${price:.4f}")
            print(f"   • RSI: {rsi:.1f}")

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

    def log_performance_stats(self):
        """Log performance statistics"""
        try:
            print(f"📊 Performance Stats: Win Rate {self.recent_win_rate:.1%}, "
                  f"Contrarian {self.contrarian_trade_count} trades")
        except Exception as e:
            print(f"Stats logging error: {e}")

    def on_multi_asset_update(self, asset_symbol: str, market_data: Dict):
        """🚀 NEW: Multi-asset callback with contrarian analysis"""
        try:
            # Store data for all assets
            self.asset_data[asset_symbol] = market_data
            
            # Update active trading asset data
            if asset_symbol == self.current_asset:
                self.latest_market_data = market_data
                self.trade_executor.update_market_data(market_data)
                self.market_volatility = market_data.get('volatility', 0.01)
            
            # Enhanced logging for multi-asset
            price = market_data.get('price', 0)
            rsi = market_data.get('rsi', 0)
            price_change_24h = market_data.get('price_change_24h', 0)
            
            # Asset selection indicator
            active_indicator = "🎯" if asset_symbol == self.current_asset else "📊"
            
            # RSI extremes detection for ANY asset
            extreme_info = ""
            contrarian_info = ""
            
            if rsi > 95:
                extreme_info = " ⚠️ EXTREME OVERBOUGHT"
            elif rsi < 5:
                extreme_info = " ⚠️ EXTREME OVERSOLD"
            elif rsi > 85:
                extreme_info = " 📈 Very Overbought"
            elif rsi < 15:
                extreme_info = " 📉 Very Oversold"
            
            # Check contrarian conditions for this asset
            if self.ml_predictions and asset_symbol == self.current_asset:
                ml_direction = self.ml_predictions.get('direction', 'neutral')
                ml_confidence = self.ml_predictions.get('confidence', 0)
                
                if ml_direction == 'unprofitable' and ml_confidence > 0.85:
                    contrarian_score = self.calculate_contrarian_score(market_data)
                    if contrarian_score >= 0.3:
                        contrarian_info = f", CONTRARIAN: {contrarian_score:.2f}"
            
            # Log with multi-asset context (every 30 seconds to avoid spam)
            if hasattr(self, '_last_multi_log'):
                if (datetime.now() - self._last_multi_log).seconds >= 30:
                    print(f"{active_indicator} {asset_symbol}: ${price:.4f}, RSI: {rsi:.1f}{extreme_info}, "
                          f"24h: {price_change_24h:+.1f}%{contrarian_info}")
                    self._last_multi_log = datetime.now()
            else:
                print(f"{active_indicator} {asset_symbol}: ${price:.4f}, RSI: {rsi:.1f}{extreme_info}, "
                      f"24h: {price_change_24h:+.1f}%{contrarian_info}")
                self._last_multi_log = datetime.now()
                
        except Exception as e:
            print(f"❌ Error processing {asset_symbol} update: {e}")

    def get_portfolio_target_asset(self) -> str:
        """🚀 NEW: Select asset based on portfolio allocation"""
        try:
            # Calculate current allocation based on trade counts
            total_trades = sum(self.trade_counts.values())
            
            if total_trades == 0:
                return 'SOL'  # Start with SOL
            
            # Find most underallocated asset
            max_deficit = 0
            target_asset = self.current_asset
            
            for asset, target_allocation in self.portfolio_allocation.items():
                current_allocation = self.trade_counts[asset] / total_trades
                deficit = target_allocation - current_allocation
                
                if deficit > max_deficit:
                    max_deficit = deficit
                    target_asset = asset
            
            return target_asset
            
        except Exception as e:
            print(f"⚠️ Portfolio allocation error: {e}")
            return self.current_asset

    def select_best_trading_asset(self) -> str:
        """🚀 ENHANCED: Select best asset combining signals + portfolio allocation"""
        if not self.multi_asset_signals or len(self.asset_data) < 2:
            # Fallback to portfolio allocation
            return self.get_portfolio_target_asset()
            
        try:
            # Get signal-based best asset
            signals = self.multi_asset_signals.analyze_multi_asset_conditions(self.asset_data)
            signal_best_asset = self.multi_asset_signals.get_best_asset_to_trade(signals)
            
            # Get portfolio-based target asset
            portfolio_target_asset = self.get_portfolio_target_asset()
            
            # Decision logic: portfolio allocation wins unless signal is very strong
            final_asset = portfolio_target_asset
            
            if signal_best_asset and signal_best_asset in signals:
                signal_confidence = signals[signal_best_asset].get('confidence', 0)
                
                # If signal is very strong (>0.7), override portfolio allocation
                if signal_confidence > 0.7:
                    final_asset = signal_best_asset
                    print(f"🎯 Signal override: {portfolio_target_asset} → {signal_best_asset} (confidence: {signal_confidence:.2f})")
                else:
                    print(f"📊 Portfolio allocation: {final_asset} (signal: {signal_best_asset}, conf: {signal_confidence:.2f})")
            
            # Execute asset switch if needed
            if final_asset != self.current_asset:
                print(f"🔄 Asset switch: {self.current_asset} → {final_asset}")
                self.current_asset = final_asset
                
                # Update latest_market_data to new asset
                if final_asset in self.asset_data:
                    self.latest_market_data = self.asset_data[final_asset]
                    self.trade_executor.update_market_data(self.latest_market_data)
                    
            return self.current_asset
            
        except Exception as e:
            print(f"⚠️ Enhanced asset selection error: {e}")
            return self.get_portfolio_target_asset()
    def start(self):
        """Start enhanced trading bot with MULTI-ASSET + CONTRARIAN LOGIC"""
        print("🚀 Starting MULTI-ASSET DexBot with Enhanced ML & CONTRARIAN Trading...")
        print(f"⏰ Start: {datetime.now()}")
        print(f"🎯 Settings: {settings['trades_per_cycle']} trades/{settings['cycle_delay_seconds']}s")
        print(f"🔄 CONTRARIAN: Enabled (RSI extremes + ML confidence)")
        print(f"📊 MULTI-ASSET: {self.supported_assets} ({'✅' if MULTI_ASSET_AVAILABLE else '❌'})")

        # Setup directories
        os.makedirs("data", exist_ok=True)
        os.makedirs("data/results", exist_ok=True)
        if ML_AVAILABLE:
            os.makedirs("ml", exist_ok=True)
            os.makedirs("ml/models", exist_ok=True)

        # Load state
        self.load_state()
        start_count = self.state["count"]

        # 🚀 START MULTI-ASSET SERVICE
        print("🌐 Connecting to multi-asset market data...")
        
        if MULTI_ASSET_AVAILABLE:
            try:
                self.multi_asset_service = create_multi_asset_service(
                       self.supported_assets, 
                    self.on_multi_asset_update
                )
                
                if self.multi_asset_service:
                    print(f"✅ Multi-asset service connected: {self.supported_assets}")
                    time.sleep(5)  # Wait for initial data
                else:
                    print("⚠️ Multi-asset service failed - falling back to single asset")
                    # Fallback to single asset
                    self.market_service = create_market_data_service(self.on_market_data_update)
                    
            except Exception as e:
                print(f"⚠️ Multi-asset error: {e} - using single asset fallback")
                self.market_service = create_market_data_service(self.on_market_data_update)
        else:
            # Single asset fallback
            print("⚠️ Multi-asset not available - using single asset mode")
            self.market_service = create_market_data_service(self.on_market_data_update)
            
        if not self.multi_asset_service and not self.market_service:
            print("⚠️ Market data connection failed - continuing in simulation mode")
        else:
            print("✅ Connected to live market data")

        # Initial ML setup
        if ML_AVAILABLE and self.ml_integration and start_count >= 100:
            print("🤖 Initializing enhanced ML predictions...")
            self.update_ml_predictions()

        print(f"🎯 Starting from transaction #{start_count + 1}")
        print(f"🎯 Active trading asset: {self.current_asset}")

        # Main enhanced trading loop
        cycle = 0
        try:
            while True:
                cycle += 1

                # 🚀 MULTI-ASSET: Select best asset every cycle
                if self.multi_asset_service and len(self.asset_data) >= 2:
                    self.select_best_trading_asset()

                # Execute enhanced trading cycle
                cycle_stats = self.execute_enhanced_trade_cycle()

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
                print(f"   • Contrarian trades: {self.contrarian_trade_count}")
                print(f"   • 🎯 Active asset: {self.current_asset}")

                # 🚀 PORTFOLIO ALLOCATION STATUS - NEW!
                total_trades = sum(self.trade_counts.values())
                if total_trades > 0:
                    print(f"   • 📊 Portfolio allocation:")
                    for asset, count in self.trade_counts.items():
                        current_pct = (count / total_trades) * 100
                        target_pct = self.portfolio_allocation[asset] * 100
                        status = "✅" if abs(current_pct - target_pct) < 10 else "⚠️"
                        print(f"     {status} {asset}: {count} trades ({current_pct:.1f}% vs {target_pct:.1f}% target)")

                # 🚀 MULTI-ASSET STATUS
                if self.multi_asset_service:
                    print(f"   • 📊 Multi-asset status:")
                    connection_status = self.multi_asset_service.get_connection_status()
                    for asset, connected in connection_status.items():
                        indicator = "✅" if connected else "❌"
                        price = self.multi_asset_service.get_asset_price(asset)
                        rsi = self.multi_asset_service.get_asset_rsi(asset)
                        active = "🎯" if asset == self.current_asset else ""
                        print(f"     {indicator} {asset}: ${price:.2f}, RSI: {rsi:.1f} {active}")
                if self.latest_market_data:
                    price = self.latest_market_data.get('price', 0)
                    rsi = self.latest_market_data.get('rsi', 50)
                    contrarian_score = self.calculate_contrarian_score(self.latest_market_data)
                    print(f"   • Current {self.current_asset} price: ${price:.4f}")
                    print(f"   • RSI: {rsi:.1f} {'⚠️ EXTREME' if rsi > 90 or rsi < 10 else ''}")
                    print(f"   • Contrarian Score: {contrarian_score:.2f}")

                # Enhanced ML status
                if self.ml_predictions:
                    try:
                        direction = self.ml_predictions.get('direction', 'unknown')
                        confidence = self.ml_predictions.get('confidence', 0)
                        print(f"   • ML Forecast ({self.current_asset}): {direction.upper()} ({confidence:.2f})")
                    except Exception as e:
                        print(f"   • ML Display Error: {e}")

                # Log performance stats every 5 cycles
                if cycle % 5 == 0:
                    self.log_performance_stats()

                # Adaptive delay between cycles
                print(f"⏳ Enhanced break: {self.adaptive_delay}s before next cycle...")
                time.sleep(self.adaptive_delay)

        except KeyboardInterrupt:
            print("\n🛑 Multi-asset bot stopped by user")
        except Exception as e:
            print(f"\n💥 Unexpected error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Cleanup
            if self.multi_asset_service:
                self.multi_asset_service.stop_tracking()
            elif self.market_service:
                self.market_service.stop_stream()

            if self.save_state():
                print(f"💾 Final state saved: {self.state['count']} transactions")

            print(f"\n🏁 Enhanced Multi-Asset Bot Session Complete:")
            print(f"   • Total transactions: {self.state['count']:,}")
            print(f"   • Recent win rate: {self.recent_win_rate:.1%}")
            print(f"   • Contrarian trades: {self.contrarian_trade_count}")
            print(f"   • Assets tracked: {self.supported_assets}")
            print(f"   • Final active asset: {self.current_asset}")


if __name__ == "__main__":
    bot = OptimizedTradingBot()
    bot.start()