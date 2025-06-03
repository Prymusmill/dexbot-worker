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

# GPT Analyzer import
try:
    from ml.gpt_analyzer import setup_gpt_enhanced_trading, format_gpt_analysis_for_logging
    GPT_AVAILABLE = False
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
    # ZASTĄP KONSTRUKTOR __init__ w OptimizedTradingBot klasy (run_worker.py)
    def __init__(self):
        # Core trading attributes
        self.trade_executor = get_trade_executor()
        self.market_service = None  # ✅ DODANE!
        self.latest_market_data = None
        self.trading_signals = TradingSignals()
        self.state = {"count": 0}

        # Enhanced ML attributes
        self.ml_predictions = {}
        self.ml_prediction_count = 0
        self.last_ml_training = None
        self.ml_performance_history = []

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

        # Enhanced ML Integration - ✅ NAPRAWIONE!
        self.ml_integration = None  # Initialize as None first
        if ML_AVAILABLE:
            try:
                self.ml_integration = MLTradingIntegration()
                print("🤖 Enhanced ML integration initialized")
            except Exception as e:
                print(f"⚠️ Enhanced ML integration failed: {e}")
                self.ml_integration = None

        # Initialize GPT analyzer
        self.gpt_analyzer = None  # Initialize as None first
        self.gpt_enabled = False
        if GPT_AVAILABLE:
            try:
                self.gpt_analyzer = setup_gpt_enhanced_trading()
                print("🤖 GPT-enhanced ML system initialized")
                self.gpt_enabled = True  # ✅ Set to True only if successful
            except Exception as e:
                print(f"⚠️ GPT initialization failed: {e}")
                self.gpt_analyzer = None
                self.gpt_enabled = False

        # Auto-retraining integration - ✅ NAPRAWIONE!
        self.auto_retrainer = None
        if ML_AVAILABLE and self.ml_integration is not None:  # ✅ Check if ml_integration exists!
            try:
                from ml.auto_retrainer import setup_auto_retraining
                self.auto_retrainer = setup_auto_retraining(
                    ml_integration=self.ml_integration,
                    db_manager=self.trade_executor.db_manager if hasattr(self.trade_executor, 'db_manager') else None,
                    retrain_interval_hours=6,  # Retrain every 6 hours
                    min_new_samples=100,       # Need 100 new samples
                    performance_threshold=0.55  # Retrain if accuracy < 55%
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
        print(f"   • Market Service: {'✅' if self.market_service is not None else '❌ (will be set later)'}")

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

    # DODAJ DO _log_enhanced_market_data w run_worker.py

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
        
        # 🎯 CONTRARIAN ANALYSIS
        if direction == 'unprofitable' and confidence > 0.85:
            contrarian_score = 0
            if rsi > 95:
                contrarian_score += 0.4
            elif rsi > 90:
                contrarian_score += 0.2
            elif rsi < 5:
                contrarian_score += 0.4
            elif rsi < 10:
                contrarian_score += 0.2
                
            if abs(price_change_24h) > 10:
                contrarian_score += 0.2
            elif abs(price_change_24h) > 5:
                contrarian_score += 0.1
                
            if confidence > 0.95:
                contrarian_score += 0.2
            elif confidence > 0.9:
                contrarian_score += 0.1
                
            if contrarian_score >= 0.3:
                contrarian_info = f", CONTRARIAN: {contrarian_score:.2f} score"

    # GPT insights
    gpt_info = ""
    if self.last_gpt_analysis:
        gpt_action = self.last_gpt_analysis.get('action', 'UNKNOWN')
        gpt_confidence = self.last_gpt_analysis.get('confidence', 0)
        gpt_info = f", GPT: {gpt_action} ({gpt_confidence:.2f})"

    # 🎯 EXTREME CONDITIONS WARNING
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

    def execute_enhanced_trade_cycle(self):
        """Enhanced trading cycle with adaptive parameters + CONTRARIAN TRACKING"""
        # ... existing code ...
    
        executed_in_cycle = 0
        profitable_in_cycle = 0
        contrarian_trades = 0  # NEW!
        contrarian_wins = 0    # NEW!
    
        for i in range(cycle_size):
            try:
                print(f"🔹 Transaction {self.state['count'] + 1} (#{i + 1}/{cycle_size})")
            
                # Track if this is a contrarian trade
                is_contrarian = False
                if self.ml_predictions:
                    ml_direction = self.ml_predictions.get('direction', 'neutral')
                    ml_confidence = self.ml_predictions.get('confidence', 0)
                    rsi = self.latest_market_data.get('rsi', 50) if self.latest_market_data else 50
                
                    if (ml_direction == 'unprofitable' and ml_confidence > 0.85 and 
                        (rsi > 90 or rsi < 10)):
                        is_contrarian = True

                # Enhanced trading decision
                if self.should_execute_trade_enhanced():
                    # Execute trade with current market data
                    trade_result = self.trade_executor.execute_trade(settings, self.latest_market_data)

                    if trade_result and hasattr(trade_result, 'profitable'):
                        executed_in_cycle += 1
                        if trade_result.profitable:
                            profitable_in_cycle += 1
                        
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

                # ... rest of existing code ...

            except Exception as e:
                print(f"❌ Trade execution error: {e}")
                continue

        # Enhanced cycle performance with contrarian stats
        cycle_win_rate = (profitable_in_cycle / executed_in_cycle) if executed_in_cycle > 0 else 0.5
        contrarian_win_rate = (contrarian_wins / contrarian_trades) if contrarian_trades > 0 else 0.0
    
        # ... existing code ...
    
        print(f"✅ Enhanced cycle complete: {executed_in_cycle}/{cycle_size} executed, "
              f"{profitable_in_cycle} profitable ({cycle_win_rate:.1%} cycle win rate)")
    
        if contrarian_trades > 0:
            print(f"🔄 Contrarian trades: {contrarian_trades}, wins: {contrarian_wins} "
                  f"({contrarian_win_rate:.1%} contrarian win rate)")

    def update_ml_predictions(self):
        """Enhanced ML predictions with ensemble + FIXED LOGGING"""  # ✅ FIXED: Proper indentation
        if not self.ml_integration:
            print("⚠️ ML integration not available")
            return

        try:
            # Get data for ML
            data_result = self.trade_executor.get_recent_transactions_hybrid(
                limit=500)

            if data_result and data_result['count'] >= 100:
                df = data_result['data']
                print(
                    f"🤖 Generating enhanced ML predictions from {len(df)} transactions...")

                # FIXED: Add detailed logging
                try:
                    prediction = self.ml_integration.get_ensemble_prediction_with_reality_check(df)
                    print(f"🔍 ML Prediction Generated: {type(prediction)} with keys: {list(prediction.keys()) if isinstance(prediction, dict) else 'NOT DICT'}")

                    if isinstance(
                            prediction, dict) and 'predicted_profitable' in prediction:
                        self.ml_predictions = prediction
                        self.ml_prediction_count += 1
                        self.current_confidence = prediction.get(
                            'confidence', 0.5)

                        # ENHANCED LOGGING - Log every prediction!
                        profitable = prediction['predicted_profitable']
                        probability = prediction.get(
                            'probability_profitable', 0.5)
                        confidence = prediction['confidence']
                        model_count = prediction.get('model_count', 1)
                        agreement = prediction.get('model_agreement', 0)
                        recommendation = prediction.get(
                            'recommendation', 'HOLD')

                        print(f"✅ ML Prediction #{self.ml_prediction_count}:")
                        print(
                            f"   • Profitable: {'YES' if profitable else 'NO'} ({probability:.1%} prob)")
                        print(f"   • Recommendation: {recommendation}")
                        print(f"   • Confidence: {confidence:.2f}")
                        print(
                            f"   • Models: {model_count}, Agreement: {agreement:.2f}")

                        # Reality check info
                        if 'reality_check' in prediction:
                            reality = prediction['reality_check']
                            if reality.get('issues'):
                                print(
                                    f"   • Reality Issues: {', '.join(reality['issues'])}")
                            else:
                                print(f"   • Reality Check: ✅ PASSED")

                    elif 'error' in prediction:
                        print(f"❌ ML Prediction Error: {prediction['error']}")
                        self.ml_predictions = {}
                    else:
                        print(f"⚠️ ML Prediction incomplete: {prediction}")
                        self.ml_predictions = {}

                except Exception as pred_error:
                    print(f"❌ ML Prediction Generation Error: {pred_error}")
                    import traceback
                    print(
                        f"🔍 Prediction Error Traceback: {traceback.format_exc()}")
                    self.ml_predictions = {}

                # Check for retraining
                try:
                    if self.ml_integration.should_retrain() and len(df) >= 200:
                        print("🔄 Starting enhanced model retraining...")
                        threading.Thread(
                            target=self._retrain_enhanced_models,
                            args=(df,),
                            daemon=True
                        ).start()
                    else:
                        print(
                            f"🔍 Retrain check: should_retrain={self.ml_integration.should_retrain()}, data_size={len(df)}")
                except Exception as retrain_error:
                    print(f"⚠️ Retrain check error: {retrain_error}")

            else:
                available = data_result['count'] if data_result else 0
                print(
                    f"⚠️ Need more data for enhanced ML ({available}/100 transactions)")
                if data_result:
                    print(
                        f"🔍 Data result: source={data_result.get('source')}, count={data_result.get('count')}")
                else:
                    print(f"🔍 Data result is None")

        except Exception as e:
            print(f"❌ Enhanced ML prediction error: {e}")
            import traceback
            print(f"🔍 ML Update Error Traceback: {traceback.format_exc()}")

    def _retrain_enhanced_models(self, df):
        """Enhanced model retraining with ensemble"""
        try:
            print("🔄 Retraining enhanced ensemble models...")
            results = self.ml_integration.train_models(df)

            if results.get('success'):
                successful_models = results.get('successful_models', [])
                print(
                    f"✅ Enhanced retraining complete! Models: {successful_models}")

                # Log detailed performance
                performance = self.ml_integration.get_model_performance()
                for model_name in successful_models:
                    if model_name in performance:
                        metrics = performance[model_name]
                        accuracy = metrics.get('accuracy', 0)
                        r2 = metrics.get('r2', 0)
                        weight = metrics.get('ensemble_weight', 0)
                        print(
                            f"   • {model_name}: Acc {accuracy:.1f}%, R² {r2:.3f}, Weight {weight:.2f}")

                self.last_ml_training = datetime.now()
            else:
                print(f"⚠️ Enhanced retraining failed: {results.get('error')}")

        except Exception as e:
            print(f"❌ Enhanced retraining error: {e}")

    def get_current_position(self, symbol='SOL/USDT'):
        """Get current position for symbol"""
        try:
            # Implement based on your exchange API
            # This is placeholder - adapt to your setup
            if hasattr(self.trade_executor,
                       'exchange') and self.trade_executor.exchange:
                balance = self.trade_executor.exchange.fetch_balance()
                position = balance.get(symbol.replace('/USDT', ''), {})
                return {
                    'symbol': symbol,
                    'amount': position.get('total', 0),
                    'value_usdt': position.get('total', 0) * self.latest_market_data.get('price', 0),
                    'side': 'long' if position.get('total', 0) > 0 else None
                }
            return None
        except Exception as e:
            print(f"Position check error: {e}")
            return None

    def get_ml_predictions_for_gpt(self):
        """Convert ML predictions to format suitable for GPT"""
        ml_predictions = []

        if self.ml_predictions:
            ml_predictions.append({
                'model_name': 'ML_Ensemble',
                'prediction': self.ml_predictions.get('direction', 'hold').upper(),
                'confidence': self.ml_predictions.get('confidence', 0.5),
                'reasoning': f"Ensemble prediction with {self.ml_predictions.get('model_count', 0)} models, agreement: {self.ml_predictions.get('model_agreement', 0):.2f}"
            })

        return ml_predictions

    def make_trading_decision_with_gpt(self):
        """Enhanced trading decision with GPT analysis"""
        try:
            if not self.latest_market_data:
                return None

            # Get ML predictions in GPT format
            ml_predictions = self.get_ml_predictions_for_gpt()

            # GPT Analysis
            gpt_analysis = None
            if self.gpt_enabled and self.gpt_analyzer:
                try:
                    current_position = self.get_current_position()
                    gpt_analysis = self.gpt_analyzer.analyze_market_signal(
                        market_data=self.latest_market_data,
                        ml_predictions=ml_predictions,
                        current_position=current_position
                    )

                    self.gpt_analysis_count += 1
                    self.last_gpt_analysis = gpt_analysis

                    # Log GPT analysis
                    if self.gpt_analysis_count % 3 == 1:  # Log every 3rd analysis
                        print(format_gpt_analysis_for_logging(gpt_analysis))

                except Exception as e:
                    print(f"⚠️ GPT analysis failed: {e}")
                    gpt_analysis = None

            # Combine ML and GPT decisions
            final_decision = self._combine_ml_gpt_decisions(
                ml_predictions, gpt_analysis, self.latest_market_data
            )

            return final_decision

        except Exception as e:
            print(f"❌ Trading decision error: {e}")
            return None

    def _combine_ml_gpt_decisions(
            self, ml_predictions, gpt_analysis, market_data):
        """Combine ML models and GPT analysis into final decision"""
        try:
            # Default decision
            decision = {
                'action': 'HOLD',
                'confidence': 0.5,
                'reasoning': 'Default conservative approach',
                'source': 'combined'
            }

            # ML Consensus
            ml_consensus = {'action': 'HOLD', 'confidence': 0.5}
            if self.ml_predictions:
                ml_direction = self.ml_predictions.get('direction', 'hold')
                ml_confidence = self.ml_predictions.get('confidence', 0.5)
                ml_consensus = {
                    'action': ml_direction.upper(),
                    'confidence': ml_confidence
                }

            # GPT Analysis
            if gpt_analysis and not gpt_analysis.get('error'):
                gpt_action = gpt_analysis.get('action', 'HOLD')
                gpt_confidence = gpt_analysis.get('confidence', 0.5)
                gpt_risk = gpt_analysis.get('risk_level', 'MEDIUM')

                # Decision Logic
                if ml_consensus['action'] == gpt_action:
                    # Agreement between ML and GPT
                    decision['action'] = gpt_action
                    decision['confidence'] = min(
                        0.9, (ml_consensus['confidence'] + gpt_confidence) / 2)
                    decision['reasoning'] = f"ML-GPT consensus: {gpt_action}"

                elif gpt_confidence > 0.8 and gpt_risk != 'HIGH':
                    # High confidence GPT overrides ML
                    decision['action'] = gpt_action
                    decision['confidence'] = gpt_confidence * \
                        0.9  # Slight discount
                    decision[
                        'reasoning'] = f"High-confidence GPT: {gpt_analysis.get('reasoning', '')[:100]}"

                elif ml_consensus['confidence'] > 0.7:
                    # Strong ML consensus
                    decision['action'] = ml_consensus['action']
                    decision['confidence'] = ml_consensus['confidence'] * 0.8
                    decision['reasoning'] = "Strong ML consensus overrides GPT"

                else:
                    # Conflicting signals - be conservative
                    decision['action'] = 'HOLD'
                    decision['confidence'] = 0.3
                    decision['reasoning'] = "Conflicting ML-GPT signals, staying conservative"

            else:
                # Only ML available
                decision = {
                    'action': ml_consensus['action'],
                    'confidence': ml_consensus['confidence'],
                    'reasoning': 'ML-only decision (GPT unavailable)',
                    'source': 'ml_only'
                }

            # Add metadata
            decision['ml_predictions_count'] = len(ml_predictions)
            decision['gpt_available'] = gpt_analysis is not None
            decision['timestamp'] = market_data.get('timestamp', '')

            # Log final decision
            if decision['confidence'] > 0.6 or decision['action'] != 'HOLD':
                print(
                    f"🎯 Final Decision: {decision['action']} (conf: {decision['confidence']:.2f}) - {decision['reasoning'][:50]}...")

            return decision

        except Exception as e:
            print(f"❌ Decision combination error: {e}")
            return {
                'action': 'HOLD',
                'confidence': 0.2,
                'reasoning': f'Error in decision logic: {str(e)}',
                'source': 'error_fallback'
            }

    def calculate_adaptive_parameters(self) -> Dict:
        """Calculate adaptive trading parameters based on ML confidence and market conditions"""
        base_cycle_size = settings.get("trades_per_cycle", 50)
        base_delay = settings.get("cycle_delay_seconds", 30)

        # ML confidence factor
        confidence_factor = 1.0
        if self.ml_predictions and settings.get("adaptive_trading", True):
            confidence = self.current_confidence

            if confidence > 0.7:
                confidence_factor = settings.get(
                    "high_confidence_multiplier", 1.5)
            elif confidence < 0.3:
                confidence_factor = settings.get(
                    "low_confidence_multiplier", 0.7)

        # GPT confidence factor
        gpt_factor = 1.0
        if self.last_gpt_analysis and not self.last_gpt_analysis.get('error'):
            gpt_confidence = self.last_gpt_analysis.get('confidence', 0.5)
            if gpt_confidence > 0.8:
                gpt_factor = 1.2
            elif gpt_confidence < 0.3:
                gpt_factor = 0.8

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
        final_factor = confidence_factor * gpt_factor * \
            volatility_factor * performance_factor

        self.adaptive_cycle_size = max(
            20, min(80, int(base_cycle_size * final_factor)))
        self.adaptive_delay = max(15, min(60, int(base_delay / final_factor)))

        return {
            'cycle_size': self.adaptive_cycle_size,
            'delay': self.adaptive_delay,
            'confidence_factor': confidence_factor,
            'gpt_factor': gpt_factor,
            'volatility_factor': volatility_factor,
            'performance_factor': performance_factor,
            'final_factor': final_factor
        }

    def should_execute_trade_enhanced(self) -> bool:
        """Enhanced trading decision with ML, GPT integration AND CONTRARIAN LOGIC"""
        if not self.latest_market_data:
            return True  # Fallback

        # Get combined ML+GPT decision
        trading_decision = self.make_trading_decision_with_gpt()

        if not trading_decision:
            # Fallback to original logic
            return self._should_execute_trade_original()

        # Use combined decision
        action = trading_decision.get('action', 'HOLD')
        confidence = trading_decision.get('confidence', 0.5)

        # 🎯 CONTRARIAN TRADING LOGIC - NEW!
        if self.ml_predictions:
            ml_direction = self.ml_predictions.get('direction', 'neutral')
            ml_confidence = self.ml_predictions.get('confidence', 0)
            rsi = self.latest_market_data.get('rsi', 50)
            price_change_24h = self.latest_market_data.get('price_change_24h', 0)
        
            # CONTRARIAN STRATEGY - Bet against ML in extreme market conditions
            if ml_direction == 'unprofitable' and ml_confidence > 0.85:
                contrarian_score = 0
                contrarian_reasons = []
            
                # RSI Extremes
                if rsi > 95:  # Extremely overbought
                    contrarian_score += 0.4
                    contrarian_reasons.append(f"RSI extremely overbought ({rsi:.1f})")
                elif rsi < 5:  # Extremely oversold  
                    contrarian_score += 0.4
                    contrarian_reasons.append(f"RSI extremely oversold ({rsi:.1f})")
                elif rsi > 90:  # Very overbought
                    contrarian_score += 0.2
                    contrarian_reasons.append(f"RSI very overbought ({rsi:.1f})")
                elif rsi < 10:  # Very oversold
                    contrarian_score += 0.2
                    contrarian_reasons.append(f"RSI very oversold ({rsi:.1f})")
                
                # Price momentum extremes
                if abs(price_change_24h) > 10:  # Extreme 24h move
                    contrarian_score += 0.2
                    contrarian_reasons.append(f"Extreme 24h move ({price_change_24h:.1f}%)")
                elif abs(price_change_24h) > 5:  # Large 24h move
                    contrarian_score += 0.1
                    contrarian_reasons.append(f"Large 24h move ({price_change_24h:.1f}%)")
                
                # ML confidence bonus
                if ml_confidence > 0.95:  # Ultra-confident ML
                    contrarian_score += 0.2
                    contrarian_reasons.append(f"Ultra-confident ML ({ml_confidence:.1%})")
                elif ml_confidence > 0.9:  # Very confident ML
                    contrarian_score += 0.1
                    contrarian_reasons.append(f"Very confident ML ({ml_confidence:.1%})")
                
                # CONTRARIAN EXECUTION
                if contrarian_score >= 0.5:  # High contrarian confidence
                    print(f"🔄 CONTRARIAN TRADE TRIGGERED!")
                    print(f"   • ML: {ml_direction.upper()} ({ml_confidence:.1%} confidence)")
                    print(f"   • Contrarian Score: {contrarian_score:.2f}")
                    print(f"   • Reasons: {', '.join(contrarian_reasons)}")
                    print(f"   • Strategy: BETTING AGAINST ML PREDICTION")
                    return True
                elif contrarian_score >= 0.3:  # Medium contrarian confidence
                    contrarian_probability = contrarian_score * 1.5  # Boost probability
                    if random.random() < contrarian_probability:
                        print(f"🎲 CONTRARIAN GAMBLE: Score {contrarian_score:.2f}, Prob {contrarian_probability:.2f}")
                        print(f"   • Reasons: {', '.join(contrarian_reasons)}")
                        return True
                    
            # REGULAR ML SKIP LOGIC (if not contrarian)
            if ml_direction == 'unprofitable' and ml_confidence > 0.8:
                # Check if contrarian didn't trigger
                rsi = self.latest_market_data.get('rsi', 50)
                if not (rsi > 90 or rsi < 10):  # Not extreme conditions
                    print(f"🚫 ML Skip: {ml_direction} with {ml_confidence:.1%} confidence")
                    return False

        # Market volatility check
        vol_threshold = settings.get("market_volatility_threshold", 0.05)
        if self.market_volatility > vol_threshold:
            confidence *= 0.8

        # 🎯 LOWERED ML THRESHOLD (was 0.3, now 0.2)
        ml_threshold = settings.get("ml_confidence_threshold", 0.2)  # Lowered!

        if action == 'BUY' or action == 'SELL':
            if confidence > 0.7:
                return True
            elif confidence > ml_threshold:
                # Probabilistic execution based on confidence
                return random.random() < confidence
            else:
                return random.random() < 0.6  # INCREASED from 0.3!
        else:  # HOLD
            return random.random() < 0.6  # INCREASED from 0.4!

    def _should_execute_trade_original(self) -> bool:
        """Original trading decision logic as fallback - ENHANCED VERSION"""
        # Base market signals
        signals = self.trading_signals.analyze_market_conditions(
            self.latest_market_data)
        base_confidence = signals.get('confidence', 0.5)

        # Enhanced ML integration
        enhanced_confidence = base_confidence
        ml_factor = 1.0

        if self.ml_integration and self.ml_predictions and settings.get("ml_enabled", True):
            try:
                ml_confidence = self.ml_predictions.get('confidence', 0.5)
                ml_direction = self.ml_predictions.get('direction', 'neutral')
                model_agreement = self.ml_predictions.get('model_agreement', 0.5)

                # 🎯 AGGRESSIVE ML SIGNALS - LOWERED THRESHOLDS
                if ml_confidence > 0.6 and model_agreement > 0.7:  # Lowered from 0.7 and 0.8
                    if ml_direction == 'up':
                        enhanced_confidence = min(base_confidence + 0.4, 1.0)
                        ml_factor = 1.4  # Increased from 1.3
                    else:
                        enhanced_confidence = max(base_confidence - 0.15, 0.15)  # Less harsh penalty
                        ml_factor = 0.85  # Less harsh from 0.8

                # 🚀 MODERATE ML SIGNALS - MORE AGGRESSIVE  
                elif ml_confidence > 0.4:  # Lowered from 0.5
                    if ml_direction == 'up':
                        enhanced_confidence = min(base_confidence + 0.25, 0.9)  # Increased boost
                        ml_factor = 1.2  # Increased from 1.1
                    else:
                        enhanced_confidence = max(base_confidence - 0.05, 0.25)  # Much less penalty
                        ml_factor = 0.95  # Much less harsh from 0.9

                # Log ML enhancement
                if abs(enhanced_confidence - base_confidence) > 0.05:  # Lowered threshold
                    print(f"🧠 ML Enhanced: {base_confidence:.2f} → {enhanced_confidence:.2f} "
                          f"(ML: {ml_direction}, {ml_confidence:.2f}, agree: {model_agreement:.2f})")

            except Exception as e:
                print(f"⚠️ ML decision enhancement error: {e}")

        # 🎯 CONTRARIAN CHECK IN FALLBACK TOO
        rsi = self.latest_market_data.get('rsi', 50)
        if rsi > 90 or rsi < 10:  # Extreme RSI conditions
            contrarian_boost = 0.2
            enhanced_confidence = min(enhanced_confidence + contrarian_boost, 1.0)
            print(f"🔄 Fallback Contrarian Boost: RSI={rsi:.1f}, boost={contrarian_boost}")

        # Market volatility check - LESS HARSH
        vol_threshold = settings.get("market_volatility_threshold", 0.03)  # Increased
        if self.market_volatility > vol_threshold:
            enhanced_confidence *= 0.85  # Less harsh from 0.8

        # Final decision with LOWERED confidence threshold
        ml_threshold = settings.get("ml_confidence_threshold", 0.5)  # Lowered

        if enhanced_confidence > 0.5:  # Lowered from 0.6
            return True
        elif enhanced_confidence > ml_threshold:
            # Probabilistic execution based on confidence
            return random.random() < (enhanced_confidence * ml_factor)
        else:
            # 🚀 INCREASED FALLBACK PROBABILITY 
            fallback_prob = settings.get("fallback_trade_probability", 0.6)  # Increased from 0.4
            return random.random() < fallback_prob

    def execute_enhanced_trade_cycle(self):
        """Enhanced trading cycle with adaptive parameters"""
        # Calculate adaptive parameters
        adaptive_params = self.calculate_adaptive_parameters()
        cycle_size = adaptive_params['cycle_size']

        print(f"\n🚀 Enhanced Cycle - {cycle_size} trades (adaptive)")
        print(f"   • ML Confidence: {self.current_confidence:.2f}")
        print(f"   • Market Volatility: {self.market_volatility:.4f}")
        print(
            f"   • Confidence Factor: {adaptive_params['confidence_factor']:.2f}")
        if self.gpt_enabled:
            print(f"   • GPT Factor: {adaptive_params['gpt_factor']:.2f}")
        print(f"   • Final Factor: {adaptive_params['final_factor']:.2f}")

        # Update ML predictions
        if self.ml_integration:
            self.update_ml_predictions()

        executed_in_cycle = 0
        profitable_in_cycle = 0

        for i in range(cycle_size):
            try:
                print(
                    f"🔹 Transaction {self.state['count'] + 1} (#{i + 1}/{cycle_size})")

                # Enhanced trading decision
                if self.should_execute_trade_enhanced():
                    # Execute trade with current market data
                    trade_result = self.trade_executor.execute_trade(
                        settings, self.latest_market_data)

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
        cycle_win_rate = (profitable_in_cycle /
                          executed_in_cycle) if executed_in_cycle > 0 else 0.5
        self.cycle_performance.append(cycle_win_rate)

        # Keep only last 10 cycles
        if len(self.cycle_performance) > 10:
            self.cycle_performance = self.cycle_performance[-10:]

        self.recent_win_rate = sum(
            self.cycle_performance) / len(self.cycle_performance)

        print(f"✅ Enhanced cycle complete: {executed_in_cycle}/{cycle_size} executed, "
              f"{profitable_in_cycle} profitable ({cycle_win_rate:.1%} cycle win rate)")

    def check_enhanced_status(self):
        """Enhanced status checking with ML and GPT insights"""
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
                    predicted_price = self.ml_predictions.get(
                        'predicted_price', 0)
                    print(
                        f"   • ML Forecast: {direction.upper()} → ${predicted_price:.4f} ({confidence:.2f})")

            except Exception as e:
                print(f"   • ML Status Error: {e}")

        # GPT Status
        if self.gpt_enabled:
            try:
                gpt_stats = self.gpt_analyzer.get_performance_stats() if self.gpt_analyzer else {}
                print(f"   • GPT Analyses: {self.gpt_analysis_count}")
                print(
                    f"   • GPT Success Rate: {gpt_stats.get('success_rate', 'N/A')}")

                if self.last_gpt_analysis:
                    gpt_action = self.last_gpt_analysis.get(
                        'action', 'UNKNOWN')
                    gpt_confidence = self.last_gpt_analysis.get(
                        'confidence', 0)
                    gpt_outlook = self.last_gpt_analysis.get(
                        'market_outlook', 'NEUTRAL')
                    print(
                        f"   • GPT Latest: {gpt_action} ({gpt_confidence:.2f}) - {gpt_outlook}")

            except Exception as e:
                print(f"   • GPT Status Error: {e}")

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
        """Log performance statistics including GPT"""
        try:
            if self.gpt_enabled and self.gpt_analyzer:
                gpt_stats = self.gpt_analyzer.get_performance_stats()
                print(f"""
📊 GPT Performance:
   Total Analyses: {gpt_stats['total_analyses']}
   Success Rate: {gpt_stats['success_rate']}
   Model: {gpt_stats['model_used']}
""")
        except Exception as e:
            print(f"Stats logging error: {e}")

    def start(self):
        """Start enhanced trading bot"""
        print("🚀 Starting OPTIMIZED DexBot with Enhanced ML, GPT & Adaptive Trading...")
        print(f"⏰ Start: {datetime.now()}")
        print(
            f"🎯 Settings: {settings['trades_per_cycle']} trades/{settings['cycle_delay_seconds']}s, ML: {settings.get('ml_enabled', True)}, GPT: {self.gpt_enabled}")

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
        self.market_service = create_market_data_service(
            self.on_market_data_update)

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
                        direction = self.ml_predictions.get(
                            'direction', 'unknown')
                        confidence = self.ml_predictions.get('confidence', 0)
                        predicted_price = self.ml_predictions.get(
                            'predicted_price', 0)
                        model_count = self.ml_predictions.get('model_count', 0)
                        print(
                            f"   • ML Ensemble: {direction.upper()} → ${predicted_price:.4f}")
                        print(
                            f"   • ML Confidence: {confidence:.2f} ({model_count} models)")
                    except Exception as e:
                        print(f"   • ML Display Error: {e}")

                # GPT status
                if self.gpt_enabled and self.last_gpt_analysis:
                    try:
                        gpt_action = self.last_gpt_analysis.get(
                            'action', 'UNKNOWN')
                        gpt_confidence = self.last_gpt_analysis.get(
                            'confidence', 0)
                        gpt_outlook = self.last_gpt_analysis.get(
                            'market_outlook', 'NEUTRAL')
                        print(
                            f"   • GPT Analysis: {gpt_action} ({gpt_confidence:.2f}) - {gpt_outlook}")
                        print(
                            f"   • GPT Analyses Count: {self.gpt_analysis_count}")
                    except Exception as e:
                        print(f"   • GPT Display Error: {e}")

                # Log performance stats every 5 cycles
                if cycle % 5 == 0:
                    self.log_performance_stats()

                # Adaptive delay between cycles
                print(
                    f"⏳ Enhanced break: {self.adaptive_delay}s before next cycle...")
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
                print(
                    f"💾 Final state saved: {self.state['count']} transactions")

            # Final enhanced status
            try:
                print(f"\n🏁 Enhanced Bot Session Complete:")
                print(f"   • Total transactions: {self.state['count']:,}")
                print(f"   • Recent win rate: {self.recent_win_rate:.1%}")
                print(
                    f"   • ML predictions generated: {self.ml_prediction_count}")

                if self.gpt_enabled:
                    print(
                        f"   • GPT analyses performed: {self.gpt_analysis_count}")
                    if self.gpt_analyzer:
                        gpt_stats = self.gpt_analyzer.get_performance_stats()
                        print(
                            f"   • GPT success rate: {gpt_stats.get('success_rate', 'N/A')}")

                if self.latest_market_data:
                    final_price = self.latest_market_data.get('price', 0)
                    print(f"   • Final SOL price: ${final_price:.4f}")

            except Exception as e:
                print(f"🏁 Session complete (status error: {e})")

if __name__ == "__main__":
    bot = OptimizedTradingBot()
    bot.start()
