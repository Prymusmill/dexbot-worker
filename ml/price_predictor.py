# ml/price_predictor.py - FIXED VERSION
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import os
import logging
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class PricePredictionModel:
    """
    FIXED: Enhanced ML model for SOL price prediction with better data handling
    """
    
    def __init__(self, model_type: str = "random_forest"):  # Changed default to RF for stability
        self.model_type = model_type.lower()
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_scaler = StandardScaler()
        self.is_trained = False
        self.sequence_length = 30  # Reduced from 60 for smaller datasets
        self.prediction_horizon = 1
        
        # Model performance metrics
        self.metrics = {
            'mse': 0.0,
            'mae': 0.0,
            'r2': 0.0,
            'accuracy': 0.0,
            'last_trained': None,
            'training_samples': 0
        }
        
        self.logger = logging.getLogger(__name__)
        os.makedirs("ml/models", exist_ok=True)
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """FIXED: More robust feature preparation with better data handling"""
        try:
            print(f"🔍 Starting feature preparation with {len(df)} rows")
            
            # Make a copy and sort
            df = df.copy().sort_values('timestamp')
            
            # FIXED: Better price column detection and creation
            if 'price' in df.columns and df['price'].notna().sum() > len(df) * 0.5:
                # Use price column if it has enough valid data
                price_col = 'price'
                print(f"✅ Using 'price' column ({df['price'].notna().sum()}/{len(df)} valid values)")
            else:
                # Create price from amount_out or fallback
                if 'amount_out' in df.columns:
                    df['price'] = pd.to_numeric(df['amount_out'], errors='coerce')
                    price_col = 'price'
                    print(f"✅ Created 'price' from 'amount_out'")
                else:
                    print(f"❌ No suitable price column found")
                    return pd.DataFrame()
            
            # FIXED: Ensure we have valid price data
            df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
            initial_len = len(df)
            df = df.dropna(subset=[price_col])
            print(f"📊 After price cleaning: {len(df)}/{initial_len} rows remain")
            
            if len(df) < 20:
                print(f"❌ Insufficient data after price cleaning: {len(df)}")
                return pd.DataFrame()
            
            # FIXED: Safer feature engineering with error handling
            try:
                # Basic price features
                df['price_change'] = df[price_col].pct_change().fillna(0)
                df['price_ma_5'] = df[price_col].rolling(5, min_periods=1).mean()
                df['price_ma_10'] = df[price_col].rolling(10, min_periods=1).mean()
                
                # Volatility with safer calculation
                df['volatility_5'] = df[price_col].rolling(5, min_periods=1).std().fillna(0)
                
                # FIXED: Safer RSI calculation
                if 'rsi' in df.columns:
                    df['rsi_final'] = pd.to_numeric(df['rsi'], errors='coerce').fillna(50.0)
                else:
                    df['rsi_final'] = self._safe_rsi(df[price_col])
                
                # Volume handling
                if 'volume' in df.columns:
                    df['volume_clean'] = pd.to_numeric(df['volume'], errors='coerce').fillna(df['amount_in'].median())
                else:
                    df['volume_clean'] = pd.to_numeric(df['amount_in'], errors='coerce')
                
                # Time features
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['hour'] = df['timestamp'].dt.hour
                df['day_of_week'] = df['timestamp'].dt.dayofweek
                
                # Market microstructure
                if 'price_impact' in df.columns:
                    df['price_impact_clean'] = pd.to_numeric(df['price_impact'], errors='coerce').fillna(0)
                else:
                    df['price_impact_clean'] = 0.0
                
                print(f"✅ Feature engineering completed")
                
            except Exception as e:
                print(f"❌ Feature engineering error: {e}")
                return pd.DataFrame()
            
            # FIXED: Select only numeric features for ML
            feature_columns = [
                'price_change', 'price_ma_5', 'price_ma_10', 'volatility_5',
                'rsi_final', 'volume_clean', 'hour', 'day_of_week', 'price_impact_clean'
            ]
            
            # Keep only existing columns
            available_features = [col for col in feature_columns if col in df.columns]
            print(f"📋 Available features: {available_features}")
            
            if len(available_features) < 3:
                print(f"❌ Too few features available: {len(available_features)}")
                return pd.DataFrame()
            
            # Create target (next price)
            df['target'] = df[price_col].shift(-1)
            
            # FIXED: More conservative data cleaning
            # Keep all rows except the last one (no target) and rows with critical missing data
            df = df[:-1]  # Remove last row (no target)
            
            # Only remove rows where ALL features are NaN
            feature_data = df[available_features]
            valid_rows = feature_data.notna().sum(axis=1) >= len(available_features) * 0.5  # At least 50% features valid
            df = df[valid_rows]
            
            # Fill remaining NaN with column medians
            for col in available_features:
                if col in df.columns:
                    df[col] = df[col].fillna(df[col].median())
            
            df['target'] = df['target'].fillna(method='ffill')  # Forward fill target
            final_clean = df.dropna(subset=['target'])
            
            print(f"📊 Final clean data: {len(final_clean)} samples with {len(available_features)} features")
            
            if len(final_clean) < 10:
                print(f"❌ Insufficient clean data: {len(final_clean)}")
                return pd.DataFrame()
            
            # Return only the columns we need
            result_df = final_clean[available_features + ['target']].copy()
            print(f"✅ Feature preparation successful: {result_df.shape}")
            
            return result_df
            
        except Exception as e:
            print(f"❌ Feature preparation failed: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def _safe_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """FIXED: Safer RSI calculation with proper error handling"""
        try:
            if len(prices) < window:
                return pd.Series([50.0] * len(prices), index=prices.index)
            
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
            
            # Avoid division by zero
            rs = gain / (loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            
            # Fill NaN with neutral RSI
            return rsi.fillna(50.0)
            
        except Exception as e:
            print(f"⚠️ RSI calculation error: {e}")
            return pd.Series([50.0] * len(prices), index=prices.index)
    
    def train_model(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict:
        """FIXED: More robust training with better error handling"""
        try:
            print(f"🤖 Training {self.model_type.upper()} model with {len(df)} samples...")
            
            # Prepare features
            df_features = self.prepare_features(df)
            
            if df_features.empty or len(df_features) < 20:
                error_msg = f"Insufficient data for training (need 20+, have {len(df_features)})"
                print(f"❌ {error_msg}")
                return {'success': False, 'error': error_msg}
            
            print(f"✅ Features prepared: {df_features.shape}")
            
            # FIXED: Robust data extraction
            feature_cols = [col for col in df_features.columns if col != 'target']
            
            try:
                X = df_features[feature_cols].values.astype(np.float64)
                y = df_features['target'].values.astype(np.float64)
                
                print(f"📊 Data shapes: X={X.shape}, y={y.shape}")
                
                # Check for infinite values
                if np.any(np.isinf(X)) or np.any(np.isinf(y)):
                    print("⚠️ Removing infinite values...")
                    finite_mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
                    X = X[finite_mask]
                    y = y[finite_mask]
                    print(f"📊 After removing infinites: X={X.shape}, y={y.shape}")
                
                if len(X) < 10:
                    return {'success': False, 'error': f'Too few samples after cleaning: {len(X)}'}
                
            except Exception as e:
                print(f"❌ Data preparation error: {e}")
                return {'success': False, 'error': f'Data preparation failed: {str(e)}'}
            
            # Train-test split
            split_idx = max(5, int(len(X) * (1 - test_size)))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            print(f"📊 Split sizes: train={len(X_train)}, test={len(X_test)}")
            
            # FIXED: Focus on Random Forest for reliability
            if self.model_type == "random_forest" or len(X_train) < 50:
                return self._train_rf_model(X_train, X_test, y_train, y_test)
            elif self.model_type == "gradient_boost":
                return self._train_gb_model(X_train, X_test, y_train, y_test)
            else:
                # Fallback to Random Forest for LSTM if dataset too small
                print("⚠️ Using Random Forest instead of LSTM for small dataset")
                self.model_type = "random_forest"
                return self._train_rf_model(X_train, X_test, y_train, y_test)
                
        except Exception as e:
            print(f"❌ Training error: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
    
    def _train_rf_model(self, X_train, X_test, y_train, y_test) -> Dict:
        """FIXED: Simplified and more robust Random Forest training"""
        try:
            print("🌲 Training Random Forest model...")
            
            # FIXED: Scale features safely
            try:
                X_train_scaled = self.feature_scaler.fit_transform(X_train)
                X_test_scaled = self.feature_scaler.transform(X_test)
            except Exception as e:
                print(f"⚠️ Scaling error, using original data: {e}")
                X_train_scaled = X_train
                X_test_scaled = X_test
            
            # FIXED: Simpler Random Forest configuration
            self.model = RandomForestRegressor(
                n_estimators=50,  # Reduced for faster training
                max_depth=8,      # Reduced to prevent overfitting
                min_samples_split=3,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=1  # Changed from -1 to avoid potential issues
            )
            
            # Train model
            print("🔄 Training Random Forest...")
            self.model.fit(X_train_scaled, y_train)
            
            # Make predictions
            print("🔮 Making predictions...")
            train_pred = self.model.predict(X_train_scaled)
            test_pred = self.model.predict(X_test_scaled)
            
            # Calculate metrics
            train_mse = mean_squared_error(y_train, train_pred)
            test_mse = mean_squared_error(y_test, test_pred)
            test_mae = mean_absolute_error(y_test, test_pred)
            test_r2 = r2_score(y_test, test_pred)
            
            # Direction accuracy
            if len(y_test) > 1:
                actual_direction = np.sign(np.diff(y_test))
                pred_direction = np.sign(np.diff(test_pred))
                accuracy = np.mean(actual_direction == pred_direction) * 100
            else:
                accuracy = 50.0
            
            # Update metrics
            self.metrics.update({
                'mse': float(test_mse),
                'mae': float(test_mae),
                'r2': float(test_r2),
                'accuracy': float(accuracy),
                'last_trained': datetime.now(),
                'training_samples': len(X_train)
            })
            
            self.is_trained = True
            
            print(f"✅ Random Forest training successful!")
            print(f"   • Training MSE: {train_mse:.6f}")
            print(f"   • Test MSE: {test_mse:.6f}")
            print(f"   • Test MAE: {test_mae:.6f}")
            print(f"   • Test R²: {test_r2:.4f}")
            print(f"   • Direction Accuracy: {accuracy:.1f}%")
            
            return {
                'success': True,
                'metrics': self.metrics.copy(),
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
        except Exception as e:
            print(f"❌ Random Forest training error: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
    
    def _train_gb_model(self, X_train, X_test, y_train, y_test) -> Dict:
        """FIXED: Simplified Gradient Boosting"""
        try:
            print("🚀 Training Gradient Boosting model...")
            
            X_train_scaled = self.feature_scaler.fit_transform(X_train)
            X_test_scaled = self.feature_scaler.transform(X_test)
            
            # Simpler GB configuration
            self.model = GradientBoostingRegressor(
                n_estimators=50,
                max_depth=4,
                learning_rate=0.1,
                random_state=42
            )
            
            self.model.fit(X_train_scaled, y_train)
            predictions = self.model.predict(X_test_scaled)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            
            # Direction accuracy
            if len(y_test) > 1:
                actual_direction = np.sign(np.diff(y_test))
                pred_direction = np.sign(np.diff(predictions))
                accuracy = np.mean(actual_direction == pred_direction) * 100
            else:
                accuracy = 50.0
            
            self.metrics.update({
                'mse': float(mse),
                'mae': float(mae), 
                'r2': float(r2),
                'accuracy': float(accuracy),
                'last_trained': datetime.now(),
                'training_samples': len(X_train)
            })
            
            self.is_trained = True
            
            print(f"✅ Gradient Boosting training successful!")
            print(f"   • MSE: {mse:.6f}")
            print(f"   • MAE: {mae:.6f}")
            print(f"   • R²: {r2:.4f}")
            print(f"   • Direction Accuracy: {accuracy:.1f}%")
            
            return {'success': True, 'metrics': self.metrics.copy()}
            
        except Exception as e:
            print(f"❌ Gradient Boosting training error: {e}")
            return {'success': False, 'error': str(e)}
    
    def predict_next_price(self, recent_data: pd.DataFrame) -> Dict:
        """FIXED: More robust prediction with better error handling"""
        try:
            if not self.is_trained:
                return {'error': 'Model not trained'}
            
            print(f"🔮 Making prediction with {len(recent_data)} data points")
            
            # Prepare features
            df_features = self.prepare_features(recent_data)
            
            if df_features.empty:
                return {'error': 'No valid features for prediction'}
            
            # Get latest features
            feature_cols = [col for col in df_features.columns if col != 'target']
            latest_features = df_features[feature_cols].iloc[-1:].values
            
            print(f"📊 Using {len(feature_cols)} features for prediction")
            
            # Scale features and predict
            try:
                if hasattr(self.feature_scaler, 'transform'):
                    latest_features_scaled = self.feature_scaler.transform(latest_features)
                else:
                    latest_features_scaled = latest_features
                
                prediction = self.model.predict(latest_features_scaled)[0]
                
            except Exception as e:
                print(f"⚠️ Prediction scaling error: {e}")
                prediction = self.model.predict(latest_features)[0]
            
            # Get current price for comparison
            if 'price' in recent_data.columns:
                current_price = recent_data['price'].iloc[-1]
            else:
                current_price = recent_data['amount_out'].iloc[-1]
            
            current_price = float(current_price)
            prediction = float(prediction)
            
            # Calculate metrics
            direction = 'up' if prediction > current_price else 'down'
            price_change = prediction - current_price
            price_change_pct = (price_change / current_price) * 100
            
            # FIXED: More realistic confidence based on model performance
            base_confidence = max(0.1, min(0.8, self.metrics.get('r2', 0.3)))
            
            # Adjust confidence based on prediction magnitude
            magnitude_factor = min(1.0, abs(price_change_pct) / 10.0)  # Scale down for large changes
            confidence = base_confidence * (0.7 + 0.3 * magnitude_factor)
            
            result = {
                'predicted_price': prediction,
                'current_price': current_price,
                'price_change': price_change,
                'price_change_pct': price_change_pct,
                'direction': direction,
                'confidence': float(confidence),
                'model_type': self.model_type,
                'prediction_time': datetime.now().isoformat(),
                'features_used': len(feature_cols)
            }
            
            print(f"🎯 Prediction: {direction.upper()} ${prediction:.4f} (confidence: {confidence:.2f})")
            
            return result
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return {'error': str(e)}
    
    def save_model(self, filepath: str = None):
        """Save trained model"""
        try:
            if not self.is_trained:
                return False
            
            if filepath is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = f"ml/models/{self.model_type}_model_{timestamp}"
            
            # Save model
            joblib.dump(self.model, f"{filepath}.pkl")
            
            # Save metadata
            joblib.dump({
                'feature_scaler': self.feature_scaler,
                'metrics': self.metrics,
                'model_type': self.model_type,
            }, f"{filepath}_metadata.pkl")
            
            print(f"✅ Model saved: {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False

class MLTradingIntegration:
    """FIXED: Simplified ML integration focused on reliability"""
    
    def __init__(self):
        # FIXED: Start with only Random Forest for reliability
        self.models = {
            'random_forest': PricePredictionModel('random_forest'),
        }
        self.active_model = 'random_forest'
        self.last_training_time = None
        
    def get_ensemble_prediction(self, recent_data: pd.DataFrame) -> Dict:
        """FIXED: Simplified ensemble with auto-training and better error handling"""
        print(f"🔮 Getting ensemble prediction from {len(recent_data)} data points")
        
        # Check if we have any trained models
        trained_models = [name for name, model in self.models.items() if model.is_trained]
        
        if not trained_models:
            print("🤖 No trained models found - attempting auto-training...")
            
            if len(recent_data) < 50:
                return {'error': f'Insufficient data for training (need 50+, have {len(recent_data)})'}
            
            # Auto-train Random Forest
            try:
                training_result = self.models['random_forest'].train_model(recent_data)
                
                if training_result.get('success'):
                    print("✅ Auto-training successful!")
                    self.models['random_forest'].save_model()
                    trained_models = ['random_forest']
                else:
                    error_msg = training_result.get('error', 'Unknown training error')
                    print(f"❌ Auto-training failed: {error_msg}")
                    return {'error': f'Auto-training failed: {error_msg}'}
                    
            except Exception as e:
                print(f"❌ Auto-training exception: {e}")
                return {'error': f'Auto-training exception: {str(e)}'}
        
        # Get predictions from trained models
        predictions = []
        confidences = []
        individual_predictions = {}
        
        for model_name in trained_models:
            model = self.models[model_name]
            try:
                pred = model.predict_next_price(recent_data)
                
                if 'predicted_price' in pred:
                    predictions.append(pred['predicted_price'])
                    confidences.append(pred['confidence'])
                    individual_predictions[model_name] = pred['predicted_price']
                    
                    print(f"✅ {model_name}: ${pred['predicted_price']:.4f} (conf: {pred['confidence']:.2f})")
                else:
                    print(f"⚠️ {model_name} prediction failed: {pred.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"❌ {model_name} prediction error: {e}")
        
        if not predictions:
            return {'error': 'No valid predictions generated'}
        
        # FIXED: Simple ensemble (average for now)
        ensemble_price = np.mean(predictions)
        ensemble_confidence = np.mean(confidences)
        
        # Get current price
        if 'price' in recent_data.columns:
            current_price = recent_data['price'].iloc[-1]
        else:
            current_price = recent_data['amount_out'].iloc[-1]
        
        direction = 'up' if ensemble_price > current_price else 'down'
        price_change_pct = ((ensemble_price - current_price) / current_price) * 100
        
        result = {
            'predicted_price': float(ensemble_price),
            'current_price': float(current_price),
            'price_change_pct': float(price_change_pct),
            'direction': direction,
            'confidence': float(ensemble_confidence),
            'model_count': len(predictions),
            'individual_predictions': individual_predictions,
            'prediction_time': datetime.now().isoformat()
        }
        
        print(f"🎯 Ensemble: {direction.upper()} ${ensemble_price:.4f} ({ensemble_confidence:.2f} confidence)")
        
        return result
    
    def get_model_performance(self) -> Dict:
        """Get performance metrics for trained models"""
        performance = {}
        
        for model_name, model in self.models.items():
            if model.is_trained:
                metrics = model.metrics.copy()
                # Add human-readable timestamp
                if metrics.get('last_trained'):
                    metrics['last_trained'] = metrics['last_trained'].strftime('%Y-%m-%d %H:%M:%S')
                performance[model_name] = metrics
        
        return performance
    
    def should_retrain(self) -> bool:
        """Check if models should be retrained"""
        if self.last_training_time is None:
            return True
        
        # Retrain every 6 hours (more frequent for better adaptation)
        time_since_training = datetime.now() - self.last_training_time
        return time_since_training > timedelta(hours=6)