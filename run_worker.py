# run_worker.py - COMPLETE ENHANCED MULTI-ASSET TRADING BOT (BALANCED FIX)
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
    from core.trade_executor import get_trade_executor
    from core.market_data import create_market_data_service, TradingSignals
    print("✅ Core modules loaded")
except ImportError as e:
    print(f"❌ Core import error: {e}")
    sys.exit(1)

# Multi-asset imports
try:
    from core.multi_asset_data import create_multi_asset_service, MultiAssetSignals
    MULTI_ASSET_AVAILABLE = True
    print("✅ Multi-asset modules available")
except ImportError as e:
    print(f"⚠️ Multi-asset modules not available: {e}")
    MULTI_ASSET_AVAILABLE = False

# Enhanced ML imports
try:
    from ml.price_predictor import MLTradingIntegration
    ML_AVAILABLE = True
    print("✅ Enhanced ML modules available")
except ImportError as e:
    print(f"⚠️ Enhanced ML modules not available: {e}")
    ML_AVAILABLE = False

# Auto-retrainer imports
try:
    from ml.auto_retrainer import setup_auto_retraining
    AUTO_RETRAIN_AVAILABLE = True
    print("✅ Auto-retrainer available")
except ImportError as e:
    print(f"⚠️ Auto-retrainer not available: {e}")
    AUTO_RETRAIN_AVAILABLE = False

# GPT analyzer imports
try:
    from ml.gpt_analyzer import setup_gpt_enhanced_trading, format_gpt_analysis_for_logging
    GPT_AVAILABLE = True
    print("✅ GPT analyzer available")
except ImportError as e:
    print(f"⚠️ GPT analyzer not available: {e}")
    GPT_AVAILABLE = False

# Constants
STATE_FILE = "data/state.json"
MEMORY_FILE = "data/memory.csv"


class EnhancedTradingBot:
    """Ultra-advanced trading bot with multi-asset, ML, and contrarian capabilities"""
    
    def __init__(self):
        print("🚀 INITIALIZING ENHANCED MULTI-ASSET TRADING BOT...")
        
        # Core components
        self.trade_executor = get_trade_executor()
        self.trading_signals = TradingSignals()
        self.state = {"count": 0, "session_start": datetime.now().isoformat()}
        
        # Market data services
        self.market_service = None
        self.multi_asset_service = None
        self.latest_market_data = None
        
        # Multi-asset configuration
        self.supported_assets = ['SOL', 'ETH', 'BTC']
        self.current_asset = 'SOL'
        self.asset_data = {}
        
        # Portfolio management
        self.portfolio_allocation = {
            'SOL': 0.40,   # 40% allocation
            'ETH': 0.35,   # 35% allocation  
            'BTC': 0.25    # 25% allocation
        }
        self.trade_counts = {'SOL': 0, 'ETH': 0, 'BTC': 0}
        self.asset_performance = {'SOL': [], 'ETH': [], 'BTC': []}
        
        # Enhanced ML components
        self.ml_integration = None
        self.ml_predictions = {}
        self.ml_prediction_count = 0
        self.last_ml_training = None
        self.training_in_progress = False
        
        # FIXED: Balanced contrarian trading
        self.contrarian_trade_count = 0
        self.contrarian_wins = 0
        self.contrarian_threshold = settings.get("contrarian_threshold", 0.85)  # FIXED: Ultra-conservative for safety
        
        # GPT integration
        self.gpt_analyzer = None
        self.gpt_enabled = False
        self.gpt_analysis_count = 0
        self.last_gpt_analysis = None
        
        # Adaptive parameters
        self.current_confidence = 0.5
        self.market_volatility = 0.01
        self.adaptive_cycle_size = settings.get("trades_per_cycle", 50)
        self.adaptive_delay = settings.get("cycle_delay_seconds", 30)
        
        # Performance tracking
        self.cycle_performance = []
        self.recent_win_rate = 0.5
        self.session_stats = {
            'cycles_completed': 0,
            'total_trades_executed': 0,
            'profitable_trades': 0,
            'contrarian_trades': 0,
            'contrarian_wins': 0,
            'asset_switches': 0
        }
        
        # Auto-retrainer
        self.auto_retrainer = None
        
        # Initialize all components
        self._initialize_components()
        
        print("✅ Enhanced Trading Bot initialized successfully")
        self._print_initialization_summary()
    
    def _initialize_components(self):
        """Initialize all bot components with error handling"""
        
        # Initialize Enhanced ML
        if ML_AVAILABLE:
            try:
                print("🤖 Initializing Enhanced ML Integration...")
                self.ml_integration = MLTradingIntegration(
                    db_manager=self.trade_executor.db_manager if hasattr(self.trade_executor, 'db_manager') else None
                )
                print("✅ Enhanced ML Integration initialized")
                
                # Force initial training check
                threading.Thread(target=self._check_initial_ml_training, daemon=True).start()
                
            except Exception as e:
                print(f"❌ Enhanced ML initialization failed: {e}")
                self.ml_integration = None
        
        # Initialize GPT analyzer
        if GPT_AVAILABLE:
            try:
                print("🤖 Initializing GPT analyzer...")
                self.gpt_analyzer = setup_gpt_enhanced_trading()
                self.gpt_enabled = True
                print("✅ GPT analyzer initialized")
            except Exception as e:
                print(f"❌ GPT initialization failed: {e}")
                self.gpt_analyzer = None
                self.gpt_enabled = False
        
        # Initialize multi-asset signals
        if MULTI_ASSET_AVAILABLE:
            try:
                print("📊 Initializing multi-asset signals...")
                from core.multi_asset_data import MultiAssetSignals
                self.multi_asset_signals = MultiAssetSignals()
                print("✅ Multi-asset signals initialized")
            except Exception as e:
                print(f"❌ Multi-asset signals failed: {e}")
                self.multi_asset_signals = None
        
        # Initialize auto-retrainer
        if AUTO_RETRAIN_AVAILABLE and self.ml_integration:
            try:
                print("🔄 Initializing auto-retrainer...")
                self.auto_retrainer = setup_auto_retraining(
                    ml_integration=self.ml_integration,
                    db_manager=self.trade_executor.db_manager if hasattr(self.trade_executor, 'db_manager') else None,
                    retrain_interval_hours=settings.get("ml_retrain_hours", 6.0),
                    min_new_samples=settings.get("retrain_min_samples_trigger", 200),
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
            
            print("🔍 Checking initial ML training status...")
            
            if self.ml_integration.should_retrain():
                print("🚀 Starting initial ML training...")
                self.training_in_progress = True
                
                try:
                    result = self.ml_integration.train_models()
                    if result.get('success'):
                        print(f"✅ Initial ML training successful: {result.get('successful_models', [])}")
                        self.last_ml_training = datetime.now()
                    else:
                        print(f"❌ Initial ML training failed: {result.get('error', 'Unknown error')}")
                finally:
                    self.training_in_progress = False
            else:
                print("ℹ️ ML models already trained or insufficient data")
                
        except Exception as e:
            print(f"❌ Initial ML training check error: {e}")
            self.training_in_progress = False
    
    def _print_initialization_summary(self):
        """Print comprehensive initialization summary"""
        print("\n" + "="*60)
        print("🚀 ENHANCED TRADING BOT INITIALIZATION SUMMARY")
        print("="*60)
        print(f"📊 Multi-Asset Support: {'✅' if MULTI_ASSET_AVAILABLE else '❌'}")
        print(f"   • Supported Assets: {self.supported_assets}")
        print(f"   • Portfolio Allocation: {self.portfolio_allocation}")
        print(f"🤖 Enhanced ML: {'✅' if self.ml_integration else '❌'}")
        print(f"🧠 GPT Analyzer: {'✅' if self.gpt_enabled else '❌'}")
        print(f"🔄 Auto-Retrainer: {'✅' if self.auto_retrainer else '❌'}")
        print(f"🔄 Contrarian Trading: ✅ (threshold: {self.contrarian_threshold})")
        print(f"⚙️ Adaptive Trading: ✅")
        print(f"📈 Current Settings:")
        print(f"   • Trades per cycle: {settings.get('trades_per_cycle', 50)}")
        print(f"   • Cycle delay: {settings.get('cycle_delay_seconds', 30)}s")
        print(f"   • ML confidence threshold: {settings.get('ml_confidence_threshold', 0.75)}")
        print(f"   • Trade amount: ${settings.get('trade_amount_usd', 0.02)}")
        print("="*60 + "\n")
    
    def start_market_data_services(self):
        """Start market data services with multi-asset support"""
        print("🌐 Starting market data services...")
        
        if MULTI_ASSET_AVAILABLE:
            try:
                print(f"🔗 Connecting to multi-asset streams: {self.supported_assets}")
                self.multi_asset_service = create_multi_asset_service(
                    self.supported_assets,
                    self.on_multi_asset_update
                )
                
                if self.multi_asset_service:
                    print("✅ Multi-asset service connected successfully")
                    time.sleep(5)  # Allow initial data load
                    return True
                else:
                    print("⚠️ Multi-asset service failed")
                    
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
    
    def on_market_data_update(self, market_data: Dict):
        """Handle single asset market data updates"""
        self.latest_market_data = market_data
        self.asset_data['SOL'] = market_data
        self.trade_executor.update_market_data(market_data)
        self.market_volatility = market_data.get('volatility', 0.01)
        
        # Log periodically
        self._log_market_update('SOL', market_data)
    
    def on_multi_asset_update(self, asset_symbol: str, market_data: Dict):
        """Handle multi-asset market data updates"""
        try:
            # Store data for all assets
            self.asset_data[asset_symbol] = market_data
            
            # Update active trading asset data
            if asset_symbol == self.current_asset:
                self.latest_market_data = market_data
                self.trade_executor.update_market_data(market_data)
                self.market_volatility = market_data.get('volatility', 0.01)
            
            # Log updates periodically
            self._log_market_update(asset_symbol, market_data)
            
        except Exception as e:
            print(f"❌ Error processing {asset_symbol} update: {e}")
    
    def _log_market_update(self, asset_symbol: str, market_data: Dict):
        """Log market data updates with intelligent throttling"""
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
            
            # ML insights for active asset
            ml_info = ""
            contrarian_info = ""
            
            if asset_symbol == self.current_asset and self.ml_predictions:
                direction = self.ml_predictions.get('direction', 'unknown')
                confidence = self.ml_predictions.get('confidence', 0)
                ml_info = f", ML: {direction.upper()} ({confidence:.2f})"
                
                # Contrarian analysis
                if direction == 'unprofitable' and confidence > 0.9:
                    contrarian_score = self._calculate_contrarian_score(market_data)
                    if contrarian_score >= self.contrarian_threshold:
                        contrarian_info = f", CONTRARIAN: {contrarian_score:.2f}"
            
            # Portfolio allocation info
            allocation_info = ""
            if self.multi_asset_service and len(self.asset_data) > 1:
                total_trades = sum(self.trade_counts.values())
                if total_trades > 0:
                    current_pct = (self.trade_counts[asset_symbol] / total_trades) * 100
                    target_pct = self.portfolio_allocation.get(asset_symbol, 0) * 100
                    allocation_info = f", Alloc: {current_pct:.0f}%/{target_pct:.0f}%"
            
            print(f"{status_indicator} {asset_symbol}: ${price:.4f}, RSI: {rsi:.1f}{rsi_status}, "
                  f"24h: {price_change_24h:+.1f}%, Vol: {volatility:.4f}{ml_info}{contrarian_info}{allocation_info}")
            
        except Exception as e:
            print(f"⚠️ Market logging error for {asset_symbol}: {e}")
    
    def update_ml_predictions(self):
        """Update ML predictions with comprehensive error handling"""
        if not self.ml_integration or self.training_in_progress:
            return
        
        try:
            print("🤖 Updating ML predictions...")
            
            # Get recent data
            data_result = self.trade_executor.get_recent_transactions_hybrid(limit=1000)
            
            if not data_result or data_result['count'] < 50:
                available = data_result['count'] if data_result else 0
                print(f"⚠️ Insufficient data for ML predictions ({available}/50 minimum)")
                return
            
            df = data_result['data']
            print(f"📊 Generating ML predictions from {len(df)} transactions...")
            
            # Generate prediction
            prediction = self.ml_integration.get_ensemble_prediction_with_reality_check(df)
            
            if isinstance(prediction, dict) and 'predicted_profitable' in prediction:
                self.ml_predictions = prediction
                self.ml_prediction_count += 1
                self.current_confidence = prediction.get('confidence', 0.5)
                
                # Log prediction details
                profitable = prediction['predicted_profitable']
                confidence = prediction['confidence']
                recommendation = prediction.get('recommendation', 'HOLD')
                model_count = prediction.get('model_count', 0)
                
                print(f"✅ ML Prediction #{self.ml_prediction_count} ({self.current_asset}):")
                print(f"   • Profitable: {'YES' if profitable else 'NO'}")
                print(f"   • Confidence: {confidence:.2f}")
                print(f"   • Recommendation: {recommendation}")
                print(f"   • Models: {model_count}")
                
                # Check for retraining
                if self.ml_integration.should_retrain():
                    print("🔄 Scheduling model retraining...")
                    threading.Thread(target=self._retrain_models, args=(df,), daemon=True).start()
                
            else:
                print(f"⚠️ ML prediction failed: {prediction}")
                self.ml_predictions = {}
                
        except Exception as e:
            print(f"❌ ML prediction error: {e}")
            import traceback
            traceback.print_exc()
    
    def _retrain_models(self, df: pd.DataFrame):
        """Retrain ML models in background"""
        if self.training_in_progress:
            print("⚠️ Training already in progress, skipping...")
            return
        
        self.training_in_progress = True
        
        try:
            print("🔄 Starting model retraining...")
            result = self.ml_integration.train_models(df)
            
            if result.get('success'):
                successful_models = result.get('successful_models', [])
                print(f"✅ Retraining successful: {successful_models}")
                self.last_ml_training = datetime.now()
            else:
                print(f"❌ Retraining failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Retraining error: {e}")
        finally:
            self.training_in_progress = False
    
    def _calculate_contrarian_score(self, market_data: Dict) -> float:
        """FIXED: Ultra-conservative contrarian scoring - only extreme conditions"""
        if not self.ml_predictions:
            return 0.0
        
        ml_direction = self.ml_predictions.get('direction', 'neutral')
        ml_confidence = self.ml_predictions.get('confidence', 0)
        
        # FIXED: Require ULTRA HIGH ML confidence (98%+) for contrarian consideration
        if ml_direction != 'unprofitable' or ml_confidence < 0.98:
            return 0.0
        
        score = 0.0
        rsi = market_data.get('rsi', 50)
        price_change_24h = market_data.get('price_change_24h', 0)
        volatility = market_data.get('volatility', 0.01)
        
        # FIXED: Only EXTREME RSI conditions (99%+ or 1%-)
        if rsi >= 99.5:  # EXTREME overbought
            score += 0.4
        elif rsi <= 0.5:  # EXTREME oversold
            score += 0.4
        elif rsi >= 99:  # Near extreme
            score += 0.3
        elif rsi <= 1:  # Near extreme
            score += 0.3
        else:
            return 0.0  # No contrarian trading unless EXTREME RSI
        
        # FIXED: Only extreme price movements (25%+)
        if abs(price_change_24h) >= 40:  # EXTREME momentum
            score += 0.3
        elif abs(price_change_24h) >= 30:  # Very high momentum
            score += 0.2
        
        # FIXED: Very high volatility requirement
        if volatility > 0.3:  # Extreme volatility
            score += 0.2
        elif volatility > 0.25:  # Very high volatility
            score += 0.15
        
        # ML confidence bonus (only for extreme confidence)
        if ml_confidence >= 0.995:
            score += 0.1
        elif ml_confidence >= 0.99:
            score += 0.05
        
        # FIXED: Safety check - multiple extreme conditions = higher risk
        extreme_conditions = 0
        if rsi >= 99.5 or rsi <= 0.5:
            extreme_conditions += 2
        if abs(price_change_24h) >= 30:
            extreme_conditions += 1
        if volatility > 0.25:
            extreme_conditions += 1
        
        # If too many extreme conditions, it's too risky even for contrarian
        if extreme_conditions >= 4:
            score *= 0.3  # Very high risk reduction
        elif extreme_conditions >= 3:
            score *= 0.6  # High risk reduction
        
        return min(score, 0.9)  # Cap below threshold
    
    def should_execute_trade(self) -> Tuple[bool, str]:
        """FIXED: BALANCED trading decision logic - not too aggressive, not too conservative"""
        if not self.latest_market_data:
            return True, "fallback_no_data"
        
        # Get current market data for additional context
        rsi = self.latest_market_data.get('rsi', 50)
        volatility = self.market_volatility
        price_change_24h = self.latest_market_data.get('price_change_24h', 0)
        
        # FIXED: Ultra-conservative contrarian trading check (highest priority)
        if self.ml_predictions:
            ml_direction = self.ml_predictions.get('direction', 'neutral')
            ml_confidence = self.ml_predictions.get('confidence', 0)
            
            # Only consider contrarian with ultra-high ML confidence and extreme conditions
            if ml_direction == 'unprofitable' and ml_confidence >= 0.98:
                contrarian_score = self._calculate_contrarian_score(self.latest_market_data)
                
                if contrarian_score >= 0.9:  # Ultra-strong contrarian signal
                    print(f"🔄 ULTRA-STRONG CONTRARIAN: Score {contrarian_score:.2f} (ML: {ml_confidence:.3f})")
                    return True, "contrarian_ultra_strong"
                elif contrarian_score >= self.contrarian_threshold:  # Strong contrarian
                    # Additional extreme condition check
                    if rsi >= 99 or rsi <= 1:  # Extreme RSI required
                        if random.random() < 0.2:  # Only 20% chance even with good score
                            print(f"🎲 CONTRARIAN EXTREME: Score {contrarian_score:.2f}, RSI {rsi:.1f}")
                            return True, "contrarian_extreme_rare"
                        else:
                            print(f"🚫 Contrarian blocked by randomness (score: {contrarian_score:.2f})")
            
            # FIXED: Balanced ML guidance - not too aggressive skipping
            if ml_direction == 'unprofitable':
                if ml_confidence >= 0.95:  # Very high confidence to skip
                    # Only skip if NO extreme RSI conditions
                    if not (rsi >= 95 or rsi <= 5):
                        print(f"⏸️ ML SKIP: Very high confidence unprofitable ({ml_confidence:.2f})")
                        return False, "ml_skip_very_high_confidence"
                    else:
                        print(f"🎯 ML Override: Extreme RSI ({rsi:.1f}) overrides ML ({ml_confidence:.2f})")
                        return True, "extreme_rsi_override"
                        
                elif ml_confidence >= 0.9:  # High confidence - reduce probability
                    if random.random() < 0.3:  # 30% chance to trade anyway
                        print(f"🎲 ML Gamble: Trading despite ML ({ml_confidence:.2f})")
                        return True, "ml_gamble_against_prediction"
                    else:
                        print(f"⏸️ ML Skip: High confidence unprofitable ({ml_confidence:.2f})")
                        return False, "ml_skip_high_confidence"
                        
                elif ml_confidence >= 0.85:  # Medium-high confidence - 50% chance
                    if random.random() < 0.5:
                        print(f"⚖️ ML Neutral: 50/50 chance despite ML ({ml_confidence:.2f})")
                        return True, "ml_neutral_chance"
                    else:
                        return False, "ml_skip_medium_high_confidence"
                
                # ML confidence 0.85 or lower - trade normally
        
        # FIXED: Market condition checks
        if volatility > 0.1:  # Very high volatility
            if random.random() < 0.4:  # 40% chance in high volatility
                print(f"⚠️ High volatility trading: {volatility:.4f}")
                return True, "high_volatility_risk"
            else:
                return False, "high_volatility_skip"
        
        # FIXED: Standard confidence-based trading
        confidence = self.current_confidence
        
        # Adjust confidence based on market conditions
        adjusted_confidence = confidence
        
        # RSI adjustments
        if 30 <= rsi <= 70:  # Good RSI range
            adjusted_confidence += 0.1
        elif rsi >= 90 or rsi <= 10:  # Extreme RSI
            adjusted_confidence += 0.15  # Boost for extreme conditions
        elif rsi >= 80 or rsi <= 20:  # High RSI
            adjusted_confidence += 0.05
        
        # Volatility adjustments
        if 0.01 <= volatility <= 0.05:  # Good volatility range
            adjusted_confidence += 0.05
        elif volatility > 0.05:  # High volatility penalty
            adjusted_confidence -= 0.1
        
        # Momentum adjustments
        if 1 <= abs(price_change_24h) <= 5:  # Good momentum
            adjusted_confidence += 0.05
        elif abs(price_change_24h) > 10:  # High momentum
            adjusted_confidence -= 0.05
        
        # Cap adjusted confidence
        adjusted_confidence = max(0.1, min(0.95, adjusted_confidence))
        
        # FIXED: Trading decisions based on adjusted confidence
        if adjusted_confidence >= 0.8:
            print(f"🚀 Very High Confidence: {adjusted_confidence:.2f}")
            return True, "very_high_confidence"
        elif adjusted_confidence >= 0.7:
            print(f"📈 High Confidence: {adjusted_confidence:.2f}")
            return True, "high_confidence"
        elif adjusted_confidence >= 0.6:
            print(f"⚖️ Good Confidence: {adjusted_confidence:.2f}")
            return True, "good_confidence"
        elif adjusted_confidence >= 0.5:
            # Probability-based execution
            chance = adjusted_confidence * 1.2  # Boost chance slightly
            if random.random() < chance:
                print(f"🎲 Probability Trade: {adjusted_confidence:.2f} (rolled {chance:.2f})")
                return True, "probability_based"
            else:
                return False, "probability_failed"
        elif adjusted_confidence >= 0.4:
            # Lower confidence - reduced chance
            if random.random() < 0.4:
                print(f"🎯 Low Confidence Chance: {adjusted_confidence:.2f}")
                return True, "low_confidence_chance"
            else:
                return False, "low_confidence_skip"
        else:
            # Very low confidence - minimal chance
            if random.random() < 0.15:
                print(f"💫 Very Low Confidence Rare: {adjusted_confidence:.2f}")
                return True, "very_low_confidence_rare"
            else:
                return False, "very_low_confidence_skip"
    
    def select_optimal_trading_asset(self) -> str:
        """Select optimal asset for trading based on signals and allocation"""
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
            
            # Get signal-based recommendation
            signal_target = self.current_asset
            if hasattr(self, 'multi_asset_signals') and self.multi_asset_signals:
                try:
                    signals = self.multi_asset_signals.analyze_multi_asset_conditions(self.asset_data)
                    signal_target = self.multi_asset_signals.get_best_asset_to_trade(signals)
                except Exception as e:
                    print(f"⚠️ Multi-asset signal error: {e}")
            
            # Decision logic: portfolio allocation takes priority unless signal is very strong
            final_asset = portfolio_target
            
            if signal_target and signal_target in self.asset_data:
                signal_data = self.asset_data[signal_target]
                signal_confidence = signal_data.get('confidence', 0)
                
                # Strong signal overrides portfolio allocation
                if signal_confidence > 0.8:
                    final_asset = signal_target
                    print(f"🎯 Signal override: {portfolio_target} → {signal_target} (conf: {signal_confidence:.2f})")
            
            # Execute asset switch if needed
            if final_asset != self.current_asset:
                print(f"🔄 Asset switch: {self.current_asset} → {final_asset}")
                self.current_asset = final_asset
                self.session_stats['asset_switches'] += 1
                
                # Update market data
                if final_asset in self.asset_data:
                    self.latest_market_data = self.asset_data[final_asset]
                    self.trade_executor.update_market_data(self.latest_market_data)
            
            return self.current_asset
            
        except Exception as e:
            print(f"❌ Asset selection error: {e}")
            return self.current_asset
    
    def execute_trading_cycle(self) -> Dict[str, Any]:
        """FIXED: Enhanced trading cycle with improved ML integration and detailed tracking"""
        cycle_start = datetime.now()
        
        # Calculate adaptive parameters
        adaptive_params = self._calculate_adaptive_parameters()
        cycle_size = adaptive_params['cycle_size']
        
        print(f"\n🚀 TRADING CYCLE {self.session_stats['cycles_completed'] + 1}")
        print(f"   • Size: {cycle_size} trades (adaptive)")
        print(f"   • Asset: {self.current_asset}")
        print(f"   • ML Confidence: {self.current_confidence:.2f}")
        print(f"   • Market Volatility: {self.market_volatility:.4f}")
        
        # Update ML predictions
        if self.ml_integration and not self.training_in_progress:
            print("🤖 Updating ML predictions for trading decisions...")
            try:
                self.update_ml_predictions()
                
                # Enhanced ML status logging
                if self.ml_predictions:
                    direction = self.ml_predictions.get('direction', 'unknown')
                    confidence = self.ml_predictions.get('confidence', 0)
                    recommendation = self.ml_predictions.get('recommendation', 'HOLD')
                    model_count = self.ml_predictions.get('model_count', 0)
                    model_agreement = self.ml_predictions.get('model_agreement', 0)
                    
                    print(f"   🎯 ML Prediction: {direction.upper()} (conf: {confidence:.2f}) → {recommendation}")
                    print(f"   🤖 Models: {model_count}, Agreement: {model_agreement:.1%}")
                    
                    # FIXED: Balanced ML guidance interpretation
                    if direction == 'profitable' and confidence > 0.7:
                        print(f"   ✅ ML supports trading with confidence {confidence:.2f}")
                    elif direction == 'unprofitable' and confidence > 0.95:
                        contrarian_score = self._calculate_contrarian_score(self.latest_market_data)
                        if contrarian_score > 0.5:
                            print(f"   🔄 Strong contrarian opportunity (score: {contrarian_score:.2f})")
                        else:
                            print(f"   ⏸️ ML strongly suggests caution (conf: {confidence:.2f})")
                    elif direction == 'unprofitable' and confidence > 0.85:
                        print(f"   ⚠️ ML suggests reduced trading (conf: {confidence:.2f})")
                    else:
                        print(f"   ⚖️ ML guidance: moderate - trading decisions balanced")
                else:
                    print("   ⚠️ No ML predictions available - check training status")
                    
            except Exception as e:
                print(f"   ❌ ML prediction update failed: {e}")
        else:
            if self.training_in_progress:
                print("   🔄 ML training in progress - using last predictions")
            else:
                print("   ⚠️ ML integration not available")
        
        # Multi-asset: select optimal asset
        if self.multi_asset_service:
            old_asset = self.current_asset
            self.select_optimal_trading_asset()
            if old_asset != self.current_asset:
                print(f"   🔄 Asset switched: {old_asset} → {self.current_asset}")
        
        # Cycle statistics
        cycle_stats = {
            'executed': 0,
            'profitable': 0,
            'contrarian_trades': 0,
            'contrarian_wins': 0,
            'skipped': 0,
            'ml_guided': 0,
            'ml_overridden': 0,
            'reasons': {}
        }
        
        # Execute trades
        for i in range(cycle_size):
            try:
                trade_number = self.state['count'] + 1
                
                # Log every 10th trade progress
                if i % 10 == 0:
                    print(f"🔹 Trade {trade_number} (#{i + 1}/{cycle_size})")
                
                # Trading decision
                should_trade, reason = self.should_execute_trade()
                
                if should_trade:
                    # Track reason
                    cycle_stats['reasons'][reason] = cycle_stats['reasons'].get(reason, 0) + 1
                    
                    # Track ML interaction
                    if ('ml' in reason or 'confidence' in reason):
                        cycle_stats['ml_guided'] += 1
                    elif ('override' in reason or 'gamble' in reason):
                        cycle_stats['ml_overridden'] += 1
                    
                    # Execute trade
                    trade_result = self.trade_executor.execute_trade(
                        settings, 
                        self.latest_market_data
                    )
                    
                    if trade_result and hasattr(trade_result, 'profitable'):
                        cycle_stats['executed'] += 1
                        self.session_stats['total_trades_executed'] += 1
                        
                        # Track asset
                        self.trade_counts[self.current_asset] += 1
                        
                        # Track profitability
                        if trade_result.profitable:
                            cycle_stats['profitable'] += 1
                            self.session_stats['profitable_trades'] += 1
                        
                        # Track contrarian trades
                        if 'contrarian' in reason:
                            cycle_stats['contrarian_trades'] += 1
                            self.session_stats['contrarian_trades'] += 1
                            
                            if trade_result.profitable:
                                cycle_stats['contrarian_wins'] += 1
                                self.session_stats['contrarian_wins'] += 1
                                self.contrarian_wins += 1
                                print(f"   🎯 CONTRARIAN WIN! (Extreme conditions paid off)")
                            else:
                                print(f"   💥 Contrarian loss (expected for reversal trading)")
                            
                            self.contrarian_trade_count += 1
                        
                        self.state['count'] += 1
                        
                        # Enhanced logging for significant trades
                        pnl = trade_result.amount_out - trade_result.amount_in
                        pnl_pct = (pnl / trade_result.amount_in) * 100
                        
                        # Log important trades
                        if ('contrarian' in reason or 'override' in reason or 
                            'gamble' in reason or abs(pnl_pct) > 0.1):
                            profit_status = "✅ PROFIT" if trade_result.profitable else "❌ LOSS"
                            print(f"   {profit_status}: P&L {pnl:+.6f} ({pnl_pct:+.2f}%) - {reason}")
                    
                else:
                    cycle_stats['skipped'] += 1
                    cycle_stats['reasons'][reason] = cycle_stats['reasons'].get(reason, 0) + 1
                    
                    # Log skip reasons periodically
                    if i % 20 == 0 and i > 0:  # Every 20th skip
                        print(f"   ⏸️ Recent skips: {reason}")
                
                # Progress check every 15 trades
                if (i + 1) % 15 == 0:
                    self._log_cycle_progress(i + 1, cycle_size, cycle_stats)
                
                # Small delay between trades
                time.sleep(0.02)  # Very fast execution
                
            except Exception as e:
                print(f"❌ Trade execution error: {e}")
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
            'asset': self.current_asset
        })
        
        # Enhanced cycle summary logging
        self._log_enhanced_cycle_summary(cycle_stats)
        
        return cycle_stats
    
    def _log_enhanced_cycle_summary(self, cycle_stats: Dict):
        """FIXED: Enhanced cycle summary with detailed ML and contrarian tracking"""
        executed = cycle_stats['executed']
        profitable = cycle_stats['profitable']
        contrarian_trades = cycle_stats['contrarian_trades']
        contrarian_wins = cycle_stats['contrarian_wins']
        ml_guided = cycle_stats.get('ml_guided', 0)
        ml_overridden = cycle_stats.get('ml_overridden', 0)
        win_rate = cycle_stats['win_rate']
        duration = cycle_stats['duration_seconds']
        
        print(f"\n✅ CYCLE {cycle_stats['cycle_number']} COMPLETE ({self.current_asset})")
        print(f"   📊 Executed: {executed}, Profitable: {profitable} ({win_rate:.1%})")
        print(f"   🤖 ML-guided: {ml_guided}, ML-overridden: {ml_overridden}")
        print(f"   ⏱️ Duration: {duration:.1f}s")
        
        if contrarian_trades > 0:
            contrarian_win_rate = contrarian_wins / contrarian_trades
            print(f"   🔄 Contrarian: {contrarian_trades} trades, {contrarian_wins} wins ({contrarian_win_rate:.1%})")
        
        # Enhanced trade reasons breakdown with explanations
        if cycle_stats['reasons']:
            reasons = cycle_stats['reasons']
            print(f"   📋 Trading Reasons:")
            for reason, count in reasons.items():
                if count > 0:
                    if 'skip' in reason:
                        print(f"     ⏸️ {reason}: {count}")
                    elif 'contrarian' in reason:
                        print(f"     🔄 {reason}: {count}")
                    elif 'confidence' in reason:
                        print(f"     📈 {reason}: {count}")
                    else:
                        print(f"     🎯 {reason}: {count}")
        
        # Current ML status and market conditions
        if self.ml_predictions:
            ml_conf = self.ml_predictions.get('confidence', 0)
            ml_dir = self.ml_predictions.get('direction', 'unknown')
            print(f"   🎯 Current ML: {ml_dir} ({ml_conf:.2f})")
        
        if self.latest_market_data:
            rsi = self.latest_market_data.get('rsi', 50)
            vol = self.market_volatility
            print(f"   📊 Market: RSI {rsi:.1f}, Vol {vol:.4f}")
        
        print(f"   📈 Recent win rate: {self.recent_win_rate:.1%}")
        
        # Performance trend analysis
        if len(self.cycle_performance) >= 3:
            trend_direction = "📈" if self.cycle_performance[-1] > self.cycle_performance[-3] else "📉"
            recent_avg = sum(self.cycle_performance[-3:]) / 3
            print(f"   {trend_direction} 3-cycle trend: {recent_avg:.1%}")
    
    def _calculate_adaptive_parameters(self) -> Dict[str, float]:
        """Calculate adaptive trading parameters based on performance and conditions"""
        base_cycle_size = settings.get("trades_per_cycle", 50)
        base_delay = settings.get("cycle_delay_seconds", 30)
        
        # Confidence factor
        confidence_factor = 1.0
        if self.ml_predictions and settings.get("adaptive_trading", True):
            confidence = self.current_confidence
            if confidence > 0.75:
                confidence_factor = settings.get("high_confidence_multiplier", 1.2)
            elif confidence < 0.4:
                confidence_factor = settings.get("low_confidence_multiplier", 0.8)
        
        # Performance factor
        performance_factor = 1.0
        if len(self.cycle_performance) >= 3:
            recent_avg = sum(self.cycle_performance[-3:]) / 3
            if recent_avg > 0.65:
                performance_factor = 1.15  # Increase when performing well
            elif recent_avg < 0.35:
                performance_factor = 0.85  # Decrease when performing poorly
        
        # Volatility factor
        volatility_factor = 1.0
        if self.market_volatility > 0.05:
            volatility_factor = 0.9  # Reduce in high volatility
        elif self.market_volatility < 0.01:
            volatility_factor = 1.1  # Increase in low volatility
        
        # Calculate final parameters
        final_factor = confidence_factor * performance_factor * volatility_factor
        
        self.adaptive_cycle_size = max(30, min(70, int(base_cycle_size * final_factor)))
        self.adaptive_delay = max(20, min(60, int(base_delay / final_factor)))
        
        return {
            'cycle_size': self.adaptive_cycle_size,
            'delay': self.adaptive_delay,
            'confidence_factor': confidence_factor,
            'performance_factor': performance_factor,
            'volatility_factor': volatility_factor,
            'final_factor': final_factor
        }
    
    def _log_cycle_progress(self, current: int, total: int, stats: Dict):
        """Log detailed cycle progress"""
        executed = stats['executed']
        profitable = stats['profitable']
        ml_guided = stats.get('ml_guided', 0)
        ml_overridden = stats.get('ml_overridden', 0)
        win_rate = (profitable / executed) if executed > 0 else 0
        
        print(f"   📊 Progress: {current}/{total}, Executed: {executed}, "
              f"Profitable: {profitable} ({win_rate:.1%}), ML-guided: {ml_guided}, Overridden: {ml_overridden}")
    
    def _log_session_summary(self):
        """Log comprehensive session summary with enhanced metrics"""
        print(f"\n📊 SESSION SUMMARY:")
        print(f"   • Total trades: {self.session_stats['total_trades_executed']}")
        print(f"   • Profitable: {self.session_stats['profitable_trades']}")
        total_win_rate = self.session_stats['profitable_trades'] / max(1, self.session_stats['total_trades_executed'])
        print(f"   • Overall win rate: {total_win_rate:.1%}")
        print(f"   • Recent win rate: {self.recent_win_rate:.1%}")
        print(f"   • Cycles completed: {self.session_stats['cycles_completed']}")
        print(f"   • Contrarian trades: {self.session_stats['contrarian_trades']}")
        
        # Contrarian performance
        if self.session_stats['contrarian_trades'] > 0:
            contrarian_rate = self.session_stats['contrarian_wins'] / self.session_stats['contrarian_trades']
            print(f"   • Contrarian win rate: {contrarian_rate:.1%}")
        
        print(f"   • Asset switches: {self.session_stats['asset_switches']}")
        
        # Portfolio allocation status
        total_trades = sum(self.trade_counts.values())
        if total_trades > 0:
            print(f"   • Portfolio allocation:")
            for asset, count in self.trade_counts.items():
                current_pct = (count / total_trades) * 100
                target_pct = self.portfolio_allocation[asset] * 100
                status = "✅" if abs(current_pct - target_pct) < 15 else "⚠️"
                print(f"     {status} {asset}: {count} trades ({current_pct:.1f}% vs {target_pct:.1f}% target)")
        
        # ML status
        if self.ml_integration:
            print(f"   • ML predictions generated: {self.ml_prediction_count}")
            if self.last_ml_training:
                print(f"   • Last ML training: {self.last_ml_training.strftime('%H:%M:%S')}")
        
        print(f"   • Current trading asset: {self.current_asset}")
        if self.latest_market_data:
            price = self.latest_market_data.get('price', 0)
            rsi = self.latest_market_data.get('rsi', 50)
            print(f"   • Current {self.current_asset} price: ${price:.4f}, RSI: {rsi:.1f}")
    
    def load_state(self):
        """Load application state from file"""
        try:
            if os.path.exists(STATE_FILE):
                with open(STATE_FILE, "r") as f:
                    saved_state = json.load(f)
                    self.state.update(saved_state)
                    if "count" not in self.state:
                        self.state["count"] = 0
                print(f"📂 State loaded: {self.state['count']} transactions")
            else:
                self.state = {"count": 0, "session_start": datetime.now().isoformat()}
        except Exception as e:
            print(f"⚠️ State loading error: {e}")
            self.state = {"count": 0, "session_start": datetime.now().isoformat()}
    
    def save_state(self) -> bool:
        """Save application state to file"""
        try:
            os.makedirs("data", exist_ok=True)
            self.state['last_save'] = datetime.now().isoformat()
            self.state['session_stats'] = self.session_stats
            self.state['trade_counts'] = self.trade_counts
            self.state['recent_win_rate'] = self.recent_win_rate
            
            with open(STATE_FILE, "w") as f:
                json.dump(self.state, f, indent=2)
            return True
        except Exception as e:
            print(f"❌ State saving error: {e}")
            return False
    
    def start(self):
        """Start the enhanced multi-asset trading bot"""
        start_time = datetime.now()
        
        print("🚀 STARTING ENHANCED MULTI-ASSET TRADING BOT")
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
        
        # Start market data services
        if not self.start_market_data_services():
            print("⚠️ Continuing in simulation mode (no live data)")
        
        # Wait for initial data
        print("⏳ Waiting for initial market data...")
        time.sleep(8)  # Longer wait for stable connection
        
        print(f"🎯 Ready to trade! Current asset: {self.current_asset}")
        
        # Main trading loop
        try:
            while True:
                # Execute trading cycle
                cycle_stats = self.execute_trading_cycle()
                
                # Save state
                if self.save_state():
                    if cycle_stats['cycle_number'] % 5 == 0:  # Log save every 5 cycles
                        print(f"💾 State saved: {self.state['count']} transactions")
                
                # Session summary every 10 cycles
                if cycle_stats['cycle_number'] % 10 == 0:
                    self._log_session_summary()
                
                # Adaptive delay
                delay = self.adaptive_delay
                print(f"⏳ Cycle break: {delay}s...")
                time.sleep(delay)
                
        except KeyboardInterrupt:
            print("\n🛑 Bot stopped by user")
        except Exception as e:
            print(f"\n💥 Unexpected error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._cleanup_and_exit(start_time, start_count)
    
    def _cleanup_and_exit(self, start_time: datetime, start_count: int):
        """Cleanup and exit procedures"""
        print("\n🧹 Cleaning up...")
        
        # Stop services
        if self.multi_asset_service:
            try:
                self.multi_asset_service.stop_tracking()
                print("✅ Multi-asset service stopped")
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
            print("💾 Final state saved")
        
        # Final session report
        session_duration = datetime.now() - start_time
        total_new_trades = self.state['count'] - start_count
        
        print(f"\n🏁 SESSION COMPLETE")
        print(f"   • Duration: {session_duration}")
        print(f"   • New trades: {total_new_trades}")
        print(f"   • Total trades: {self.state['count']:,}")
        print(f"   • Final win rate: {self.recent_win_rate:.1%}")
        print(f"   • Contrarian trades: {self.contrarian_trade_count}")
        print(f"   • Active asset: {self.current_asset}")
        
        if self.latest_market_data:
            final_price = self.latest_market_data.get('price', 0)
            print(f"   • Final {self.current_asset} price: ${final_price:.4f}")
        
        print("✅ Shutdown complete")


if __name__ == "__main__":
    # Initialize and start the bot
    bot = EnhancedTradingBot()
    bot.start()