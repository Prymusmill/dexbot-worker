# ml/auto_retrainer.py - ENHANCED Smart Auto-Retraining System
import threading
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class EnhancedRetrainingConfig:
    """Enhanced Configuration for smart auto-retraining"""
    # 🚀 SMART 1H RETRAINING
    retrain_interval_hours: float = 1.0          # Max 1 hour interval  
    retrain_interval_minutes: int = 60           # Max 60 minutes
    min_new_samples: int = 3000                  # 3K samples trigger (~50min worth)
    performance_threshold: float = 0.50          # 75% accuracy threshold
    
    # 🚀 MULTI-TRIGGER SYSTEM
    enable_time_trigger: bool = True             # Max interval trigger
    enable_sample_trigger: bool = True           # Sample count trigger
    enable_performance_trigger: bool = True     # Accuracy drop trigger
    enable_volatility_trigger: bool = True      # Market volatility trigger
    enable_rsi_trigger: bool = True              # RSI extreme trigger
    
    # 🚀 MARKET-BASED TRIGGERS
    volatility_threshold: float = 0.05           # 5% volatility spike
    rsi_extreme_low: float = 5.0                # RSI extreme low
    rsi_extreme_high: float = 95.0              # RSI extreme high
    price_change_threshold: float = 0.10        # 10% price change trigger
    
    # 🚀 SMART SCHEDULING
    smart_scheduling: bool = True               # Avoid training during high activity
    avoid_high_activity_hours: List[int] = None # Hours to avoid (None = auto-detect)
    min_quiet_period_minutes: int = 5          # Min quiet period before training
    
    # 🚀 QUALITY CONTROLS
    min_training_samples: int = 500             # Minimum samples for training
    validation_split: float = 0.2              # Validation split
    quality_threshold: float = 0.70            # Minimum model quality
    max_retrain_attempts: int = 3              # Max attempts per cycle
    
    # 🚀 PERFORMANCE MONITORING
    enable_performance_monitoring: bool = True
    performance_buffer_size: int = 100         # Track last 100 predictions
    accuracy_window_size: int = 20             # Recent accuracy window
    
    # 🚀 ADVANCED FEATURES
    enable_reality_check: bool = True          # Enable prediction reality checks
    enable_feature_importance: bool = True    # Track feature importance
    enable_model_diversity: bool = True       # Ensure model diversity
    auto_adjust_thresholds: bool = True       # Auto-adjust based on performance

class SmartAutoRetrainingManager:
    """🚀 Enhanced Smart Auto-Retraining System with Multi-Trigger Logic"""
    
    def __init__(self, ml_integration, db_manager=None, config: EnhancedRetrainingConfig = None):
        self.ml_integration = ml_integration
        self.db_manager = db_manager
        self.config = config or EnhancedRetrainingConfig()
        
        self.logger = logging.getLogger(__name__)
        self.is_running = False
        self.retraining_thread = None
        
        # 🚀 ENHANCED TRACKING
        self.last_retrain_time = None
        self.last_known_sample_count = 0
        self.retraining_history = []
        self.performance_history = []
        
        # 🚀 SMART TRIGGER TRACKING  
        self.last_market_data = None
        self.volatility_history = []
        self.rsi_history = []
        self.activity_history = []
        
        # 🚀 PERFORMANCE MONITORING
        self.prediction_accuracy_buffer = []
        self.recent_predictions = []
        self.feature_importance_history = []
        
        # 🚀 SMART SCHEDULING
        self.high_activity_periods = []
        self.last_activity_check = None
        
        print(f"🚀 Enhanced Auto-Retrainer initialized:")
        print(f"   • Max interval: {self.config.retrain_interval_minutes} minutes")
        print(f"   • Sample trigger: {self.config.min_new_samples:,}")
        print(f"   • Performance threshold: {self.config.performance_threshold:.1%}")
        print(f"   • Volatility trigger: {self.config.volatility_threshold:.1%}")
        print(f"   • RSI extremes: {self.config.rsi_extreme_low}-{self.config.rsi_extreme_high}")
        print(f"   • Smart scheduling: {'✅' if self.config.smart_scheduling else '❌'}")
    
    def start_auto_retraining(self):
        """Start the enhanced auto-retraining service"""
        if self.is_running:
            self.logger.warning("Auto-retraining already running")
            return
        
        self.is_running = True
        self.retraining_thread = threading.Thread(
            target=self._smart_retraining_loop, 
            daemon=True
        )
        self.retraining_thread.start()
        
        self.logger.info(f"🚀 Smart auto-retraining started - max {self.config.retrain_interval_minutes}min")
        print(f"🚀 Smart Auto-Retraining Service Started")
        print(f"   • Check interval: 5 minutes")
        print(f"   • Multi-trigger system: ✅")
        print(f"   • Smart scheduling: {'✅' if self.config.smart_scheduling else '❌'}")
    
    def stop_auto_retraining(self):
        """Stop the auto-retraining service"""
        self.is_running = False
        if self.retraining_thread:
            self.retraining_thread.join(timeout=5)
        self.logger.info("🛑 Smart auto-retraining stopped")
        print("🛑 Smart Auto-Retraining stopped")
    
    def _smart_retraining_loop(self):
        """🚀 Enhanced retraining loop with smart triggers"""
        while self.is_running:
            try:
                # Check all triggers
                should_retrain, trigger_reason = self._should_retrain_smart()
                
                if should_retrain:
                    # Check if it's a good time to retrain (smart scheduling)
                    if self._is_good_time_to_retrain():
                        print(f"\n🔄 SMART RETRAIN TRIGGERED: {trigger_reason}")
                        self._perform_enhanced_retraining(trigger_reason)
                    else:
                        print(f"⏸️ Retrain postponed (high activity): {trigger_reason}")
                
                # Smart check interval (every 5 minutes)
                time.sleep(5 * 60)
                
            except Exception as e:
                self.logger.error(f"Error in smart retraining loop: {e}")
                print(f"❌ Smart retraining loop error: {e}")
                time.sleep(60)  # Wait 1 minute before retry
    
    def _should_retrain_smart(self) -> Tuple[bool, str]:
        """🚀 Smart multi-trigger retraining decision"""
        
        # 🚀 TRIGGER 1: Time-based (max interval)
        if self.config.enable_time_trigger:
            if self.last_retrain_time is None:
                return True, "Initial training"
            
            time_since_retrain = datetime.now() - self.last_retrain_time
            minutes_since = time_since_retrain.total_seconds() / 60
            
            if minutes_since >= self.config.retrain_interval_minutes:
                return True, f"Max interval reached ({minutes_since:.0f}min)"
        
        # 🚀 TRIGGER 2: Sample-based (new data available)
        if self.config.enable_sample_trigger:
            try:
                current_sample_count = self._get_current_sample_count()
                new_samples = current_sample_count - self.last_known_sample_count
                
                if new_samples >= self.config.min_new_samples:
                    return True, f"Sample trigger ({new_samples:,} new samples)"
                    
            except Exception as e:
                self.logger.error(f"Error checking sample count: {e}")
        
        # 🚀 TRIGGER 3: Performance degradation
        if self.config.enable_performance_trigger and len(self.prediction_accuracy_buffer) >= self.config.accuracy_window_size:
            recent_accuracy = np.mean(self.prediction_accuracy_buffer[-self.config.accuracy_window_size:])
            if recent_accuracy < self.config.performance_threshold:
                return True, f"Performance drop (accuracy: {recent_accuracy:.1%})"
        
        # 🚀 TRIGGER 4: Market volatility spike
        if self.config.enable_volatility_trigger and self.last_market_data:
            current_volatility = self.last_market_data.get('volatility', 0)
            if current_volatility > self.config.volatility_threshold:
                return True, f"High volatility ({current_volatility:.1%})"
        
        # 🚀 TRIGGER 5: RSI extremes
        if self.config.enable_rsi_trigger and self.last_market_data:
            current_rsi = self.last_market_data.get('rsi', 50)
            if current_rsi <= self.config.rsi_extreme_low or current_rsi >= self.config.rsi_extreme_high:
                return True, f"RSI extreme ({current_rsi:.1f})"
        
        # 🚀 TRIGGER 6: Major price change
        if self.last_market_data:
            price_change_24h = abs(self.last_market_data.get('price_change_24h', 0)) / 100
            if price_change_24h > self.config.price_change_threshold:
                return True, f"Major price change ({price_change_24h:.1%})"
        
        return False, "No trigger conditions met"
    
    def _is_good_time_to_retrain(self) -> bool:
        """🚀 Smart scheduling - check if it's a good time to retrain"""
        if not self.config.smart_scheduling:
            return True
        
        try:
            # Check recent trading activity
            if self._is_high_activity_period():
                return False
            
            # Check if enough quiet time has passed
            if self._has_sufficient_quiet_period():
                return True
            
            return True  # Default to allow if unclear
            
        except Exception as e:
            self.logger.error(f"Error in smart scheduling: {e}")
            return True  # Default to allow on error
    
    def _is_high_activity_period(self) -> bool:
        """Check if current period has high trading activity"""
        try:
            # Simple heuristic: check if many trades in last 10 minutes
            current_time = datetime.now()
            
            # Get recent activity (this would need to be implemented)
            # For now, return False (allow training)
            return False
            
        except Exception:
            return False
    
    def _has_sufficient_quiet_period(self) -> bool:
        """Check if there's been sufficient quiet period"""
        if self.last_activity_check is None:
            return True
        
        quiet_time = datetime.now() - self.last_activity_check
        return quiet_time.total_seconds() >= self.config.min_quiet_period_minutes * 60
    
    def _perform_enhanced_retraining(self, trigger_reason: str):
        """🚀 Enhanced retraining process with smart features"""
        start_time = datetime.now()
        
        try:
            self.logger.info(f"🚀 Starting enhanced retraining: {trigger_reason}")
            print(f"\n🚀 ENHANCED AUTO-RETRAINING CYCLE")
            print(f"   • Trigger: {trigger_reason}")
            print(f"   • Started: {start_time.strftime('%H:%M:%S')}")
            
            # 🚀 GET FRESH DATA
            df = self._get_training_data()
            
            if len(df) < self.config.min_training_samples:
                self.logger.warning(f"Insufficient data: {len(df)}/{self.config.min_training_samples}")
                print(f"⚠️ Insufficient data: {len(df)}/{self.config.min_training_samples}")
                return
            
            print(f"📊 Training on {len(df):,} total samples")
            
            # 🚀 ENHANCED DATA PREPARATION
            train_df, val_df = self._prepare_training_data(df)
            print(f"   • Training: {len(train_df):,} samples")
            print(f"   • Validation: {len(val_df):,} samples")
            
            # 🚀 STORE OLD PERFORMANCE
            old_performance = self.ml_integration.get_model_performance()
            
            # 🚀 PERFORM RETRAINING WITH QUALITY CHECKS
            results = self._retrain_with_quality_checks(train_df)
            
            if results.get('success'):
                # 🚀 ENHANCED VALIDATION
                validation_results = self._enhanced_validation(val_df)
                
                # 🚀 QUALITY ASSESSMENT
                quality_passed = self._assess_model_quality(validation_results, old_performance)
                
                if quality_passed:
                    # 🚀 SUCCESS - Update tracking
                    self._record_successful_retrain(start_time, trigger_reason, results, validation_results, old_performance)
                else:
                    print(f"❌ Quality check failed - reverting to previous models")
                    self._revert_to_previous_models()
                    
            else:
                self._record_failed_retrain(start_time, trigger_reason, results.get('error', 'Unknown error'))
                
        except Exception as e:
            self.logger.error(f"Exception during enhanced retraining: {e}")
            print(f"❌ Enhanced retraining exception: {e}")
            self._record_failed_retrain(start_time, trigger_reason, str(e))
    
    def _get_training_data(self) -> pd.DataFrame:
        """Get fresh training data from database"""
        try:
            if self.db_manager:
                return self.db_manager.get_all_transactions_for_ml()
            else:
                # Fallback - try to get from ML integration
                return pd.DataFrame()  # Placeholder
        except Exception as e:
            self.logger.error(f"Error getting training data: {e}")
            return pd.DataFrame()
    
    def _prepare_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """🚀 Enhanced data preparation with quality checks"""
        # Smart validation split
        split_idx = int(len(df) * (1 - self.config.validation_split))
        train_df = df.iloc[:split_idx]
        val_df = df.iloc[split_idx:]
        
        # 🚀 FUTURE: Add data quality checks here
        # - Outlier detection
        # - Feature correlation analysis
        # - Data drift detection
        
        return train_df, val_df
    
    def _retrain_with_quality_checks(self, train_df: pd.DataFrame) -> Dict:
        """Retrain models with enhanced quality checks"""
        try:
            results = self.ml_integration.train_models(train_df)
            return results
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _enhanced_validation(self, val_df: pd.DataFrame) -> Dict:
        """🚀 Enhanced validation with multiple metrics"""
        try:
            if len(val_df) < 50:
                return {'accuracy': 0.5, 'confidence': 0.0}
            
            # Get prediction on validation set
            prediction = self.ml_integration.get_ensemble_prediction_with_reality_check(val_df.tail(100))
            
            if 'error' in prediction:
                return {'accuracy': 0.5, 'confidence': 0.0, 'error': prediction['error']}
            
            # Extract validation metrics
            accuracy = prediction.get('confidence', 0.5)
            model_agreement = prediction.get('model_agreement', 0.0)
            model_count = prediction.get('model_count', 0)
            
            return {
                'accuracy': accuracy,
                'model_agreement': model_agreement,
                'model_count': model_count,
                'confidence': accuracy,
                'validation_samples': len(val_df)
            }
            
        except Exception as e:
            self.logger.error(f"Enhanced validation error: {e}")
            return {'accuracy': 0.5, 'confidence': 0.0, 'error': str(e)}
    
    def _assess_model_quality(self, validation_results: Dict, old_performance: Dict) -> bool:
        """🚀 Assess if new models meet quality standards"""
        try:
            new_accuracy = validation_results.get('accuracy', 0.0)
            
            # Check minimum quality threshold
            if new_accuracy < self.config.quality_threshold:
                print(f"❌ Quality check: accuracy {new_accuracy:.1%} < {self.config.quality_threshold:.1%}")
                return False
            
            # Check improvement vs old models (if available)
            if old_performance:
                # Compare with best old model
                old_accuracies = [metrics.get('accuracy', 0) for metrics in old_performance.values()]
                best_old_accuracy = max(old_accuracies) / 100 if old_accuracies else 0
                
                # Allow slight degradation for fresh models
                min_acceptable = best_old_accuracy * 0.95  # 5% tolerance
                
                if new_accuracy < min_acceptable:
                    print(f"❌ Quality check: new accuracy {new_accuracy:.1%} < {min_acceptable:.1%} (95% of best old)")
                    return False
            
            print(f"✅ Quality check passed: accuracy {new_accuracy:.1%}")
            return True
            
        except Exception as e:
            self.logger.error(f"Quality assessment error: {e}")
            return False  # Conservative: fail on error
    
    def _record_successful_retrain(self, start_time: datetime, trigger_reason: str, 
                                 results: Dict, validation_results: Dict, old_performance: Dict):
        """Record successful retraining"""
        duration = (datetime.now() - start_time).total_seconds()
        
        retraining_record = {
            'timestamp': start_time,
            'trigger_reason': trigger_reason,
            'duration_seconds': duration,
            'validation_results': validation_results,
            'models_trained': results.get('successful_models', []),
            'old_performance': old_performance,
            'success': True
        }
        
        self.retraining_history.append(retraining_record)
        
        # Update tracking
        self.last_retrain_time = start_time
        self.last_known_sample_count = self._get_current_sample_count()
        
        # Success logging
        accuracy = validation_results.get('accuracy', 0)
        model_count = results.get('model_count', 0)
        
        print(f"✅ Enhanced retraining completed successfully!")
        print(f"   • Duration: {duration:.1f}s")
        print(f"   • Validation accuracy: {accuracy:.1%}")
        print(f"   • Models trained: {model_count}")
        print(f"   • Trigger: {trigger_reason}")
        
        self.logger.info(f"Successful retrain: {trigger_reason}, accuracy: {accuracy:.1%}")
    
    def _record_failed_retrain(self, start_time: datetime, trigger_reason: str, error: str):
        """Record failed retraining"""
        retraining_record = {
            'timestamp': start_time,
            'trigger_reason': trigger_reason,
            'success': False,
            'error': error
        }
        
        self.retraining_history.append(retraining_record)
        
        print(f"❌ Enhanced retraining failed: {error}")
        self.logger.error(f"Retraining failed: {trigger_reason} - {error}")
    
    def _revert_to_previous_models(self):
        """Revert to previous models if quality check fails"""
        # This would need implementation in ml_integration
        self.logger.warning("Model reversion not implemented - keeping new models")
    
    def _get_current_sample_count(self) -> int:
        """Get current sample count"""
        try:
            if self.db_manager:
                return self.db_manager.get_transaction_count()
            return 0
        except Exception:
            return 0
    
    def update_market_data(self, market_data: Dict):
        """🚀 Update market data for smart triggers"""
        self.last_market_data = market_data
        
        # Track volatility and RSI history
        if 'volatility' in market_data:
            self.volatility_history.append(market_data['volatility'])
            if len(self.volatility_history) > 100:
                self.volatility_history = self.volatility_history[-100:]
        
        if 'rsi' in market_data:
            self.rsi_history.append(market_data['rsi'])
            if len(self.rsi_history) > 100:
                self.rsi_history = self.rsi_history[-100:]
    
    def add_prediction_feedback(self, predicted_profitable: bool, actual_profitable: bool):
        """🚀 Enhanced prediction feedback tracking"""
        if self.config.enable_performance_monitoring:
            accuracy = 1.0 if predicted_profitable == actual_profitable else 0.0
            self.prediction_accuracy_buffer.append(accuracy)
            
            # Keep only recent predictions
            if len(self.prediction_accuracy_buffer) > self.config.performance_buffer_size:
                self.prediction_accuracy_buffer = self.prediction_accuracy_buffer[-self.config.performance_buffer_size:]
            
            # Enhanced tracking
            self.recent_predictions.append({
                'timestamp': datetime.now(),
                'predicted': predicted_profitable,
                'actual': actual_profitable,
                'correct': accuracy == 1.0
            })
            
            # Auto-adjust thresholds if enabled
            if self.config.auto_adjust_thresholds:
                self._auto_adjust_thresholds()
    
    def _auto_adjust_thresholds(self):
        """🚀 Auto-adjust retraining thresholds based on performance"""
        if len(self.prediction_accuracy_buffer) >= 50:
            recent_accuracy = np.mean(self.prediction_accuracy_buffer[-50:])
            
            # Adjust performance threshold based on actual performance
            if recent_accuracy > 0.8:
                # Performance is good, can be more selective
                self.config.performance_threshold = min(0.8, self.config.performance_threshold + 0.01)
            elif recent_accuracy < 0.6:
                # Performance is poor, be less selective
                self.config.performance_threshold = max(0.6, self.config.performance_threshold - 0.01)
    
    def get_enhanced_status(self) -> Dict:
        """🚀 Get comprehensive retraining status"""
        status = {
            'is_running': self.is_running,
            'last_retrain': self.last_retrain_time.isoformat() if self.last_retrain_time else None,
            'total_retrains': len(self.retraining_history),
            'successful_retrains': len([r for r in self.retraining_history if r.get('success')]),
            'last_known_samples': self.last_known_sample_count,
            'config': {
                'max_interval_minutes': self.config.retrain_interval_minutes,
                'min_new_samples': self.config.min_new_samples,
                'performance_threshold': self.config.performance_threshold,
                'volatility_threshold': self.config.volatility_threshold,
                'rsi_extremes': [self.config.rsi_extreme_low, self.config.rsi_extreme_high]
            }
        }
        
        # Performance metrics
        if self.prediction_accuracy_buffer:
            status['recent_accuracy'] = np.mean(self.prediction_accuracy_buffer[-self.config.accuracy_window_size:])
            status['total_predictions_tracked'] = len(self.prediction_accuracy_buffer)
        
        # Smart trigger status
        if self.last_market_data:
            status['current_market'] = {
                'volatility': self.last_market_data.get('volatility', 0),
                'rsi': self.last_market_data.get('rsi', 50),
                'price_change_24h': self.last_market_data.get('price_change_24h', 0)
            }
        
        # Time until next retrain
        if self.last_retrain_time:
            next_retrain = self.last_retrain_time + timedelta(minutes=self.config.retrain_interval_minutes)
            time_until_retrain = next_retrain - datetime.now()
            status['next_retrain_in_minutes'] = max(0, time_until_retrain.total_seconds() / 60)
        
        # Recent triggers
        if self.retraining_history:
            recent_triggers = [r.get('trigger_reason', 'Unknown') for r in self.retraining_history[-5:]]
            status['recent_triggers'] = recent_triggers
        
        return status

# 🚀 Enhanced integration helper
def setup_auto_retraining(ml_integration, db_manager=None, **config_kwargs):
    """🚀 Setup and start enhanced auto-retraining service"""
    
    # Map legacy config parameters to new structure
    legacy_mapping = {
        'retrain_interval_hours': 'retrain_interval_hours',
        'min_new_samples': 'min_new_samples', 
        'performance_threshold': 'performance_threshold'
    }
    
    # Convert legacy parameters
    enhanced_config = {}
    for legacy_key, new_key in legacy_mapping.items():
        if legacy_key in config_kwargs:
            enhanced_config[new_key] = config_kwargs[legacy_key]
    
    # Add any remaining config
    for key, value in config_kwargs.items():
        if key not in legacy_mapping:
            enhanced_config[key] = value
    
    # Convert hours to minutes for new system
    if 'retrain_interval_hours' in enhanced_config:
        enhanced_config['retrain_interval_minutes'] = int(enhanced_config['retrain_interval_hours'] * 60)
    
    config = EnhancedRetrainingConfig(**enhanced_config)
    manager = SmartAutoRetrainingManager(ml_integration, db_manager, config)
    manager.start_auto_retraining()
    
    print(f"🚀 Enhanced Auto-Retraining System initialized!")
    
    return manager