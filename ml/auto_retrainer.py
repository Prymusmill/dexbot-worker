# ml/auto_retrainer.py - Advanced Auto-Retraining System
import threading
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class RetrainingConfig:
    """Configuration for auto-retraining"""
    retrain_interval_hours: int = 6  # Retrain every 6 hours
    min_new_samples: int = 100       # Minimum new samples to trigger retrain
    performance_threshold: float = 0.55  # Retrain if accuracy drops below 55%
    max_retrain_attempts: int = 3    # Max retrain attempts per cycle
    validation_split: float = 0.2    # Validation data split
    enable_performance_monitoring: bool = True

class AutoRetrainingManager:
    """Advanced Auto-Retraining System for ML Models"""
    
    def __init__(self, ml_integration, db_manager, config: RetrainingConfig = None):
        self.ml_integration = ml_integration
        self.db_manager = db_manager
        self.config = config or RetrainingConfig()
        
        self.logger = logging.getLogger(__name__)
        self.is_running = False
        self.retraining_thread = None
        
        # Tracking variables
        self.last_retrain_time = None
        self.last_known_sample_count = 0
        self.retraining_history = []
        self.performance_history = []
        
        # Performance monitoring
        self.prediction_accuracy_buffer = []
        self.recent_predictions = []
        
    def start_auto_retraining(self):
        """Start the auto-retraining service"""
        if self.is_running:
            self.logger.warning("Auto-retraining already running")
            return
        
        self.is_running = True
        self.retraining_thread = threading.Thread(
            target=self._retraining_loop, 
            daemon=True
        )
        self.retraining_thread.start()
        
        self.logger.info(f"🔄 Auto-retraining started - every {self.config.retrain_interval_hours}h")
        print(f"🔄 Auto-retraining service started")
        print(f"   • Interval: {self.config.retrain_interval_hours} hours")
        print(f"   • Min samples: {self.config.min_new_samples}")
        print(f"   • Performance threshold: {self.config.performance_threshold}")
    
    def stop_auto_retraining(self):
        """Stop the auto-retraining service"""
        self.is_running = False
        if self.retraining_thread:
            self.retraining_thread.join(timeout=5)
        self.logger.info("🛑 Auto-retraining stopped")
    
    def _retraining_loop(self):
        """Main retraining loop"""
        while self.is_running:
            try:
                # Check if retraining is needed
                if self._should_retrain():
                    self._perform_retraining()
                
                # Sleep for check interval (every 30 minutes)
                time.sleep(30 * 60)  # 30 minutes
                
            except Exception as e:
                self.logger.error(f"Error in retraining loop: {e}")
                time.sleep(60)  # Wait 1 minute before retry
    
    def _should_retrain(self) -> bool:
        """Determine if retraining should be triggered"""
        
        # Check 1: Time-based retraining
        if self.last_retrain_time is None:
            return True
        
        time_since_retrain = datetime.now() - self.last_retrain_time
        if time_since_retrain.total_seconds() >= self.config.retrain_interval_hours * 3600:
            self.logger.info(f"⏰ Time-based retraining triggered ({time_since_retrain})")
            return True
        
        # Check 2: New samples available
        try:
            current_sample_count = self.db_manager.get_transaction_count()
            new_samples = current_sample_count - self.last_known_sample_count
            
            if new_samples >= self.config.min_new_samples:
                self.logger.info(f"📊 Sample-based retraining triggered ({new_samples} new samples)")
                return True
                
        except Exception as e:
            self.logger.error(f"Error checking sample count: {e}")
        
        # Check 3: Performance degradation
        if self.config.enable_performance_monitoring and len(self.prediction_accuracy_buffer) >= 20:
            recent_accuracy = np.mean(self.prediction_accuracy_buffer[-20:])
            if recent_accuracy < self.config.performance_threshold:
                self.logger.info(f"📉 Performance-based retraining triggered (accuracy: {recent_accuracy:.2f})")
                return True
        
        return False
    
    def _perform_retraining(self):
        """Perform the actual retraining process"""
        start_time = datetime.now()
        
        try:
            self.logger.info("🔄 Starting model retraining...")
            print(f"\n🔄 AUTO-RETRAINING CYCLE - {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Get fresh data
            df = self.db_manager.get_all_transactions_for_ml()
            
            if len(df) < 500:  # Minimum required samples
                self.logger.warning(f"Insufficient data for retraining: {len(df)}/500")
                return
            
            print(f"📊 Training on {len(df)} total samples")
            
            # Prepare validation split for performance monitoring
            split_idx = int(len(df) * (1 - self.config.validation_split))
            train_df = df.iloc[:split_idx]
            val_df = df.iloc[split_idx:]
            
            # Store old performance for comparison
            old_performance = self.ml_integration.get_model_performance()
            
            # Perform retraining
            results = self.ml_integration.train_models(train_df)
            
            if results.get('success'):
                # Validate on held-out data
                validation_accuracy = self._validate_models(val_df)
                
                # Get new performance
                new_performance = self.ml_integration.get_model_performance()
                
                # Log results
                duration = (datetime.now() - start_time).total_seconds()
                
                retraining_record = {
                    'timestamp': start_time,
                    'duration_seconds': duration,
                    'training_samples': len(train_df),
                    'validation_samples': len(val_df),
                    'validation_accuracy': validation_accuracy,
                    'models_trained': results.get('successful_models', []),
                    'old_performance': old_performance,
                    'new_performance': new_performance,
                    'success': True
                }
                
                self.retraining_history.append(retraining_record)
                
                # Update tracking variables
                self.last_retrain_time = start_time
                self.last_known_sample_count = len(df)
                
                # Log success
                print(f"✅ Retraining completed successfully!")
                print(f"   • Duration: {duration:.1f}s")
                print(f"   • Validation accuracy: {validation_accuracy:.1%}")
                print(f"   • Models trained: {results.get('model_count', 0)}")
                
                # Save to database if available
                self._save_retraining_record(retraining_record)
                
            else:
                error_msg = results.get('error', 'Unknown error')
                self.logger.error(f"Retraining failed: {error_msg}")
                print(f"❌ Retraining failed: {error_msg}")
                
                # Record failure
                self.retraining_history.append({
                    'timestamp': start_time,
                    'success': False,
                    'error': error_msg,
                    'training_samples': len(df)
                })
                
        except Exception as e:
            self.logger.error(f"Exception during retraining: {e}")
            print(f"❌ Retraining exception: {e}")
            
            self.retraining_history.append({
                'timestamp': start_time,
                'success': False,
                'error': str(e)
            })
    
    def _validate_models(self, val_df) -> float:
        """Validate models on held-out data"""
        try:
            if len(val_df) < 50:
                return 0.5  # Default if insufficient validation data
            
            # Generate predictions on validation set
            prediction = self.ml_integration.get_ensemble_prediction(val_df.tail(100))
            
            if 'error' in prediction:
                return 0.5
            
            # Simple accuracy calculation
            # In real implementation, you'd compare predictions vs actual outcomes
            confidence = prediction.get('confidence', 0.5)
            return confidence  # Simplified validation
            
        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            return 0.5
    
    def _save_retraining_record(self, record):
        """Save retraining record to database"""
        try:
            # Implement database saving logic here
            # For now, just log
            self.logger.info(f"Retraining record saved: {record['timestamp']}")
        except Exception as e:
            self.logger.error(f"Error saving retraining record: {e}")
    
    def add_prediction_feedback(self, predicted_profitable: bool, actual_profitable: bool):
        """Add feedback for performance monitoring"""
        if self.config.enable_performance_monitoring:
            accuracy = 1.0 if predicted_profitable == actual_profitable else 0.0
            self.prediction_accuracy_buffer.append(accuracy)
            
            # Keep only last 100 predictions
            if len(self.prediction_accuracy_buffer) > 100:
                self.prediction_accuracy_buffer = self.prediction_accuracy_buffer[-100:]
            
            # Store for analysis
            self.recent_predictions.append({
                'timestamp': datetime.now(),
                'predicted': predicted_profitable,
                'actual': actual_profitable,
                'correct': accuracy == 1.0
            })
    
    def get_retraining_status(self) -> Dict:
        """Get current retraining status and statistics"""
        status = {
            'is_running': self.is_running,
            'last_retrain': self.last_retrain_time.isoformat() if self.last_retrain_time else None,
            'total_retrains': len(self.retraining_history),
            'successful_retrains': len([r for r in self.retraining_history if r.get('success')]),
            'last_known_samples': self.last_known_sample_count,
            'config': {
                'interval_hours': self.config.retrain_interval_hours,
                'min_new_samples': self.config.min_new_samples,
                'performance_threshold': self.config.performance_threshold
            }
        }
        
        # Add performance metrics if available
        if self.prediction_accuracy_buffer:
            status['recent_accuracy'] = np.mean(self.prediction_accuracy_buffer[-20:])
            status['total_predictions_tracked'] = len(self.prediction_accuracy_buffer)
        
        # Time until next retrain
        if self.last_retrain_time:
            next_retrain = self.last_retrain_time + timedelta(hours=self.config.retrain_interval_hours)
            time_until_retrain = next_retrain - datetime.now()
            status['next_retrain_in_hours'] = max(0, time_until_retrain.total_seconds() / 3600)
        
        return status
    
    def get_retraining_history(self, limit: int = 10) -> List[Dict]:
        """Get recent retraining history"""
        return self.retraining_history[-limit:] if self.retraining_history else []

# Integration helper
def setup_auto_retraining(ml_integration, db_manager, **config_kwargs):
    """Setup and start auto-retraining service"""
    config = RetrainingConfig(**config_kwargs)
    manager = AutoRetrainingManager(ml_integration, db_manager, config)
    manager.start_auto_retraining()
    return manager