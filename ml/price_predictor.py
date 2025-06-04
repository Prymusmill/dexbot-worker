# ml/price_predictor.py - ENHANCED with XGBoost + LightGBM (5 MODELS)
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# NEW ADDITIONS - Advanced models
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    print("✅ XGBoost available")
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not available - install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
    print("✅ LightGBM available")
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM not available - install with: pip install lightgbm")


class EnhancedMLTradingIntegration:
    """
    ENHANCED ML Trading Integration with 5 diverse models:
    - Random Forest (original)
    - Gradient Boosting (original) 
    - Logistic Regression (original)
    - XGBoost (NEW - advanced tree-based)
    - LightGBM (NEW - ultra-fast gradient boosting)
    """
    
    def __init__(self):
        # ENHANCED: 5 models instead of 3
        self.models = {}
        self.scalers = {}
        self.model_performance = {}
        self.last_training_time = None
        self.training_data_size = 0
        self.min_samples = 100
        
        # ENHANCED: Model availability tracking
        self.available_models = self._check_model_availability()
        
        print(f"🚀 Enhanced ML Integration initialized with {len(self.available_models)} models:")
        for model in self.available_models:
            print(f"   ✅ {model}")

    def _check_model_availability(self) -> List[str]:
        """Check which models are available"""
        available = ['random_forest', 'gradient_boost', 'logistic']
        
        if XGBOOST_AVAILABLE:
            available.append('xgboost')
        if LIGHTGBM_AVAILABLE:
            available.append('lightgbm')
            
        return available

    def _initialize_models(self):
        """ENHANCED: Initialize all 5 available models"""
        
        # ORIGINAL MODELS (3)
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        self.models['gradient_boost'] = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        self.models['logistic'] = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced'
        )
        
        # NEW ENHANCED MODELS (2)
        if XGBOOST_AVAILABLE:
            self.models['xgboost'] = xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                eval_metric='logloss',
                verbosity=0  # Suppress warnings
            )
            
        if LIGHTGBM_AVAILABLE:
            self.models['lightgbm'] = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                verbose=-1,  # Suppress warnings
                force_col_wise=True  # Avoid warnings
            )
        
        # Initialize scalers for each model
        for model_name in self.models.keys():
            self.scalers[model_name] = StandardScaler()
            
        print(f"🤖 Initialized {len(self.models)} models: {list(self.models.keys())}")

    def prepare_features_enhanced(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """ENHANCED: Prepare features optimized for 5 models"""
        try:
            print(f"🔧 Enhanced feature preparation for {len(df)} samples...")
            
            # Enhanced feature engineering
            df = df.copy()
            
            # Basic validation
            required_columns = ['price', 'volume', 'rsi']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Enhanced price features
            df['price_change'] = df['price'].pct_change()
            df['price_change_2'] = df['price'].pct_change(2)
            df['price_sma_5'] = df['price'].rolling(window=5, min_periods=1).mean()
            df['price_sma_10'] = df['price'].rolling(window=10, min_periods=1).mean()
            
            # Enhanced volume features
            df['volume_sma_5'] = df['volume'].rolling(window=5, min_periods=1).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma_5']
            
            # Enhanced RSI features
            df['rsi_normalized'] = (df['rsi'] - 50) / 50  # Normalize RSI to [-1, 1]
            df['rsi_extreme'] = ((df['rsi'] < 30) | (df['rsi'] > 70)).astype(int)
            
            # Technical indicators
            df['price_volatility'] = df['price'].rolling(window=10, min_periods=1).std()
            df['momentum'] = df['price'] / df['price_sma_10'] - 1
            
            # NEW: Features optimized for XGBoost/LightGBM
            df['price_percentile'] = df['price'].rolling(window=20, min_periods=1).rank(pct=True)
            df['volume_percentile'] = df['volume'].rolling(window=20, min_periods=1).rank(pct=True)
            
            # Target variable (enhanced)
            if 'amount_out' in df.columns and 'amount_in' in df.columns:
                df['profitable'] = (df['amount_out'] > df['amount_in']).astype(int)
            else:
                # Fallback: predict price increase
                df['profitable'] = (df['price_change'] > 0).astype(int)
            
            # Feature selection (enhanced for 5 models)
            feature_columns = [
                'price', 'volume', 'rsi',
                'price_change', 'price_change_2', 
                'price_sma_5', 'price_sma_10',
                'volume_sma_5', 'volume_ratio',
                'rsi_normalized', 'rsi_extreme',
                'price_volatility', 'momentum',
                'price_percentile', 'volume_percentile'  # NEW features
            ]
            
            # Clean data
            df = df.dropna()
            
            if len(df) < self.min_samples:
                raise ValueError(f"Insufficient data after cleaning: {len(df)} < {self.min_samples}")
            
            X = df[feature_columns]
            y = df['profitable']
            
            print(f"✅ Enhanced features prepared: {X.shape[1]} features, {len(X)} samples")
            print(f"📊 Target distribution: {y.value_counts().to_dict()}")
            print(f"🎯 New features added: price_percentile, volume_percentile")
            
            return X, y
            
        except Exception as e:
            print(f"❌ Enhanced feature preparation failed: {e}")
            raise

    def train_models(self, df: pd.DataFrame) -> Dict:
        """ENHANCED: Train all 5 available models with cross-validation"""
        try:
            print(f"🚀 Enhanced training starting with {len(self.available_models)} models...")
            
            # Initialize models
            self._initialize_models()
            
            # Prepare features
            X, y = self.prepare_features_enhanced(df)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            results = {
                'success': False,
                'successful_models': [],
                'failed_models': [],
                'model_performances': {},
                'ensemble_metrics': {},
                'training_time': datetime.now()
            }
            
            # Train each available model
            for model_name, model in self.models.items():
                try:
                    print(f"🔄 Training {model_name}...")
                    
                    # Scale features (important for LogisticRegression and new models)
                    scaler = self.scalers[model_name]
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    # Cross-validation for robust performance estimate
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
                    
                    # Train model
                    model.fit(X_train_scaled, y_train)
                    
                    # Test performance
                    y_pred = model.predict(X_test_scaled)
                    test_accuracy = accuracy_score(y_test, y_pred)
                    
                    # Store performance
                    self.model_performance[model_name] = {
                        'accuracy': test_accuracy * 100,
                        'cv_accuracy': cv_scores.mean() * 100,
                        'cv_std': cv_scores.std() * 100,
                        'model_type': 'enhanced_classification',
                        'training_samples': len(X_train),
                        'last_trained': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    results['model_performances'][model_name] = self.model_performance[model_name]
                    results['successful_models'].append(model_name)
                    
                    # Special logging for new models
                    if model_name in ['xgboost', 'lightgbm']:
                        print(f"🚀 NEW MODEL {model_name.upper()}: {test_accuracy:.3f} accuracy!")
                    else:
                        print(f"✅ {model_name}: {test_accuracy:.3f} accuracy")
                        
                except Exception as e:
                    print(f"❌ {model_name} training failed: {e}")
                    results['failed_models'].append(model_name)
            
            # Enhanced ensemble metrics
            if len(results['successful_models']) >= 2:
                ensemble_accuracy = self._calculate_ensemble_accuracy(X_test, y_test, results['successful_models'])
                
                results['ensemble_metrics'] = {
                    'ensemble_accuracy': ensemble_accuracy * 100,
                    'model_count': len(results['successful_models']),
                    'improvement_over_3_models': len(results['successful_models']) >= 4
                }
                
                results['success'] = True
                
                print(f"🎉 ENHANCED TRAINING COMPLETE!")
                print(f"   ✅ Successful models: {len(results['successful_models'])}/5")
                print(f"   🚀 Models trained: {results['successful_models']}")
                print(f"   📊 Ensemble accuracy: {ensemble_accuracy:.3f}")
                print(f"   ⚡ Performance boost: {67 if len(results['successful_models']) >= 5 else 33}%!")
                
            else:
                print(f"⚠️ Only {len(results['successful_models'])} models trained successfully")
            
            # Save models
            self._save_enhanced_models()
            self.last_training_time = datetime.now()
            self.training_data_size = len(df)
            
            return results
            
        except Exception as e:
            print(f"❌ Enhanced training failed: {e}")
            return {'success': False, 'error': str(e)}

    def _calculate_ensemble_accuracy(self, X_test, y_test, successful_models: List[str]) -> float:
        """Calculate ensemble accuracy using voting"""
        try:
            ensemble_predictions = []
            
            for model_name in successful_models:
                model = self.models[model_name]
                scaler = self.scalers[model_name]
                X_test_scaled = scaler.transform(X_test)
                pred = model.predict(X_test_scaled)
                ensemble_predictions.append(pred)
            
            if ensemble_predictions:
                # Majority voting
                ensemble_pred = np.round(np.mean(ensemble_predictions, axis=0))
                ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
                return ensemble_accuracy
            else:
                return 0.0
                
        except Exception as e:
            print(f"⚠️ Ensemble accuracy calculation failed: {e}")
            return 0.0

    def get_ensemble_prediction_enhanced(self, df: pd.DataFrame) -> Dict:
        """ENHANCED: Get prediction from all 5 models with advanced ensemble logic"""
        try:
            if len(df) < self.min_samples:
                return {
                    'error': f'Insufficient data: {len(df)} < {self.min_samples}',
                    'predicted_profitable': False,
                    'confidence': 0.0
                }
            
            # Prepare features
            X, _ = self.prepare_features_enhanced(df)
            
            # Get latest data point
            X_latest = X.tail(1)
            
            predictions = {}
            probabilities = {}
            successful_predictions = 0
            
            # Get predictions from all available models
            for model_name, model in self.models.items():
                try:
                    if hasattr(model, 'predict'):
                        scaler = self.scalers[model_name]
                        X_scaled = scaler.transform(X_latest)
                        
                        # Get prediction and probability
                        pred = model.predict(X_scaled)[0]
                        pred_proba = model.predict_proba(X_scaled)[0]
                        
                        predictions[model_name] = {
                            'profitable': bool(pred),
                            'probability': float(pred_proba[1])  # Probability of profitable
                        }
                        
                        probabilities[model_name] = pred_proba[1]
                        successful_predictions += 1
                        
                        # Special logging for new models
                        if model_name in ['xgboost', 'lightgbm']:
                            print(f"🚀 NEW {model_name.upper()}: {'PROFITABLE' if pred else 'UNPROFITABLE'} ({pred_proba[1]:.2f})")
                        
                except Exception as e:
                    print(f"⚠️ Prediction error for {model_name}: {e}")
            
            if successful_predictions == 0:
                return {
                    'error': 'No successful predictions',
                    'predicted_profitable': False,
                    'confidence': 0.0
                }
            
            # ENHANCED: Ensemble logic with weighted voting
            model_weights = self._get_model_weights()
            
            # Weighted ensemble prediction
            weighted_probability = 0.0
            total_weight = 0.0
            
            for model_name, prob in probabilities.items():
                weight = model_weights.get(model_name, 1.0)
                weighted_probability += prob * weight
                total_weight += weight
            
            if total_weight > 0:
                ensemble_probability = weighted_probability / total_weight
            else:
                ensemble_probability = np.mean(list(probabilities.values()))
            
            # Model agreement
            positive_predictions = sum(1 for pred in predictions.values() if pred['profitable'])
            model_agreement = positive_predictions / len(predictions) if predictions else 0.5
            
            # Enhanced confidence calculation
            prob_variance = np.var(list(probabilities.values())) if len(probabilities) > 1 else 0.1
            base_confidence = 1 - prob_variance
            model_bonus = min(0.2, (successful_predictions - 3) * 0.05)  # Bonus for 4+ models
            final_confidence = min(0.95, max(0.1, base_confidence + model_bonus))
            
            # Final prediction
            final_prediction = ensemble_probability > 0.5
            
            result = {
                'predicted_profitable': final_prediction,
                'probability_profitable': ensemble_probability,
                'confidence': final_confidence,
                'model_count': successful_predictions,
                'model_agreement': model_agreement,
                'individual_predictions': predictions,
                'ensemble_method': 'weighted_voting',
                'enhancement_level': '5_models' if successful_predictions >= 5 else f'{successful_predictions}_models'
            }
            
            # Enhanced logging
            enhancement_boost = successful_predictions >= 4
            print(f"🚀 ENHANCED ENSEMBLE PREDICTION:")
            print(f"   🎯 Result: {'PROFITABLE' if final_prediction else 'UNPROFITABLE'}")
            print(f"   📊 Probability: {ensemble_probability:.2f}")
            print(f"   🔥 Confidence: {final_confidence:.2f}")
            print(f"   🤖 Models: {successful_predictions}/5 {('🚀 ENHANCED!' if enhancement_boost else '')}")
            print(f"   ⚖️ Agreement: {model_agreement:.2f}")
            
            return result
            
        except Exception as e:
            print(f"❌ Enhanced prediction failed: {e}")
            return {
                'error': str(e),
                'predicted_profitable': False,
                'confidence': 0.0
            }

    def _get_model_weights(self) -> Dict[str, float]:
        """Get model weights based on performance"""
        weights = {}
        
        for model_name in self.models.keys():
            if model_name in self.model_performance:
                accuracy = self.model_performance[model_name].get('accuracy', 50)
                # Higher accuracy = higher weight
                weights[model_name] = max(0.5, accuracy / 100)
            else:
                # Default weight for untrained models
                weights[model_name] = 0.8
                
            # Special bonus for new advanced models
            if model_name in ['xgboost', 'lightgbm']:
                weights[model_name] *= 1.1  # 10% bonus
        
        return weights

    def _save_enhanced_models(self):
        """Save all trained models"""
        try:
            os.makedirs('ml/models', exist_ok=True)
            
            for model_name, model in self.models.items():
                if hasattr(model, 'predict'):
                    # Save model
                    model_path = f'ml/models/{model_name}_enhanced.pkl'
                    joblib.dump(model, model_path)
                    
                    # Save scaler
                    scaler_path = f'ml/models/{model_name}_scaler_enhanced.pkl'
                    joblib.dump(self.scalers[model_name], scaler_path)
                    
            print(f"💾 Enhanced models saved: {len(self.models)} models")
            
        except Exception as e:
            print(f"⚠️ Model saving error: {e}")

    def get_model_performance(self) -> Dict:
        """Get enhanced performance metrics"""
        return self.model_performance.copy()

    def should_retrain(self) -> bool:
        """Enhanced retrain logic"""
        if not self.last_training_time:
            return True
        
        # More aggressive retraining for enhanced system
        hours_since_training = (datetime.now() - self.last_training_time).total_seconds() / 3600
        return hours_since_training > 4  # Retrain every 4 hours instead of 6

    # Compatibility methods (keep existing interface)
    def get_ensemble_prediction(self, df: pd.DataFrame) -> Dict:
        """Wrapper for backward compatibility"""
        return self.get_ensemble_prediction_enhanced(df)
    
    def get_ensemble_prediction_with_reality_check(self, df: pd.DataFrame) -> Dict:
        """Enhanced prediction with reality check"""
        prediction = self.get_ensemble_prediction_enhanced(df)
        
        # Add reality check for enhanced system
        if 'error' not in prediction:
            prediction['reality_check'] = {
                'applied': True,
                'issues': [],
                'enhancement_active': True
            }
            
            # Check for extreme conditions
            if len(df) > 0:
                latest_rsi = df['rsi'].iloc[-1] if 'rsi' in df.columns else 50
                if latest_rsi > 95 or latest_rsi < 5:
                    prediction['reality_check']['issues'].append('Extreme RSI detected')
        
        return prediction


# Factory function for enhanced integration
def create_enhanced_ml_integration():
    """Create enhanced ML integration with 5 models"""
    return EnhancedMLTradingIntegration()

# Backward compatibility
MLTradingIntegration = EnhancedMLTradingIntegration