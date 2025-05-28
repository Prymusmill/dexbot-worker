# ml/price_predictor.py
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
    Advanced ML model for SOL price prediction
    Supports multiple algorithms: LSTM, Random Forest, Gradient Boosting
    """
    
    def __init__(self, model_type: str = "lstm"):
        self.model_type = model_type.lower()
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_scaler = StandardScaler()
        self.is_trained = False
        self.sequence_length = 60  # 60 data points for prediction
        self.prediction_horizon = 1  # Predict 1 step ahead
        
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
        
        # Create models directory
        os.makedirs("ml/models", exist_ok=True)
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare enhanced features for ML model"""
        try:
            # Sort by timestamp
            df = df.sort_values('timestamp').copy()
            
            # Price features
            df['price_change'] = df['amount_out'].pct_change()
            df['price_ma_5'] = df['amount_out'].rolling(5).mean()
            df['price_ma_15'] = df['amount_out'].rolling(15).mean()
            df['price_ma_30'] = df['amount_out'].rolling(30).mean()
            
            # Volatility features
            df['volatility_5'] = df['amount_out'].rolling(5).std()
            df['volatility_15'] = df['amount_out'].rolling(15).std()
            
            # Technical indicators
            df['rsi'] = self._calculate_rsi(df['amount_out'])
            df['macd'], df['macd_signal'] = self._calculate_macd(df['amount_out'])
            df['bb_upper'], df['bb_lower'] = self._calculate_bollinger_bands(df['amount_out'])
            
            # Time-based features
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            
            # Market microstructure (if available)
            if 'price_impact' in df.columns:
                df['impact_ma'] = df['price_impact'].rolling(10).mean()
                df['impact_std'] = df['price_impact'].rolling(10).std()
            
            # Target variable (next price)
            df['target'] = df['amount_out'].shift(-1)
            
            # Drop NaN values
            df = df.dropna()
            
            print(f"✅ Features prepared: {len(df)} samples, {df.shape[1]} features")
            return df
            
        except Exception as e:
            print(f"❌ Error preparing features: {e}")
            return df
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, lower
    
    def create_sequences(self, data: np.array, target: np.array) -> Tuple[np.array, np.array]:
        """Create sequences for LSTM training"""
        X, y = [], []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(target[i])
        return np.array(X), np.array(y)
    
    def build_lstm_model(self, input_shape: Tuple) -> keras.Model:
        """Build LSTM neural network"""
        model = keras.Sequential([
            # First LSTM layer
            layers.LSTM(50, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.2),
            
            # Second LSTM layer
            layers.LSTM(50, return_sequences=True),
            layers.Dropout(0.2),
            
            # Third LSTM layer
            layers.LSTM(25),
            layers.Dropout(0.2),
            
            # Dense layers
            layers.Dense(25, activation='relu'),
            layers.Dense(1)
        ])
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_model(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict:
        """Train the ML model"""
        try:
            print(f"🤖 Training {self.model_type.upper()} model...")
            
            # Prepare features
            df_features = self.prepare_features(df)
            
            if len(df_features) < 100:
                print("⚠️ Insufficient data for training (need at least 100 samples)")
                return {'success': False, 'error': 'Insufficient data'}
            
            # Feature columns (exclude timestamp, target, and non-numeric columns)
            feature_columns = [col for col in df_features.columns 
                             if col not in ['timestamp', 'target', 'input_token', 'output_token', 'profitable']]
            
            X = df_features[feature_columns].values
            y = df_features['target'].values
            
            # Handle any remaining NaN values
            mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X, y = X[mask], y[mask]
            
            print(f"📊 Training data: {X.shape[0]} samples, {X.shape[1]} features")
            
            # Split data
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            if self.model_type == "lstm":
                return self._train_lstm_model(X_train, X_test, y_train, y_test)
            elif self.model_type == "random_forest":
                return self._train_rf_model(X_train, X_test, y_train, y_test)
            elif self.model_type == "gradient_boost":
                return self._train_gb_model(X_train, X_test, y_train, y_test)
            else:
                return {'success': False, 'error': f'Unknown model type: {self.model_type}'}
                
        except Exception as e:
            print(f"❌ Training error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _train_lstm_model(self, X_train, X_test, y_train, y_test) -> Dict:
        """Train LSTM model"""
        try:
            # Scale features
            X_train_scaled = self.feature_scaler.fit_transform(X_train)
            X_test_scaled = self.feature_scaler.transform(X_test)
            
            # Scale target
            y_train_scaled = self.scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
            y_test_scaled = self.scaler.transform(y_test.reshape(-1, 1)).flatten()
            
            # Create sequences
            X_train_seq, y_train_seq = self.create_sequences(X_train_scaled, y_train_scaled)
            X_test_seq, y_test_seq = self.create_sequences(X_test_scaled, y_test_scaled)
            
            if len(X_train_seq) == 0:
                return {'success': False, 'error': 'Not enough data for sequence creation'}
            
            # Build model
            self.model = self.build_lstm_model((X_train_seq.shape[1], X_train_seq.shape[2]))
            
            # Train model
            history = self.model.fit(
                X_train_seq, y_train_seq,
                epochs=50,
                batch_size=32,
                validation_data=(X_test_seq, y_test_seq),
                verbose=0,
                callbacks=[
                    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                    keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
                ]
            )
            
            # Evaluate model
            predictions = self.model.predict(X_test_seq)
            predictions_unscaled = self.scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
            y_test_unscaled = self.scaler.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()
            
            # Calculate metrics
            mse = mean_squared_error(y_test_unscaled, predictions_unscaled)
            mae = mean_absolute_error(y_test_unscaled, predictions_unscaled)
            r2 = r2_score(y_test_unscaled, predictions_unscaled)
            
            # Calculate direction accuracy
            actual_direction = np.sign(np.diff(y_test_unscaled))
            pred_direction = np.sign(np.diff(predictions_unscaled))
            accuracy = np.mean(actual_direction == pred_direction) * 100
            
            self.metrics.update({
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'accuracy': accuracy,
                'last_trained': datetime.now(),
                'training_samples': len(X_train_seq)
            })
            
            self.is_trained = True
            
            print(f"✅ LSTM model trained successfully!")
            print(f"   • MSE: {mse:.6f}")
            print(f"   • MAE: {mae:.6f}")
            print(f"   • R²: {r2:.4f}")
            print(f"   • Direction Accuracy: {accuracy:.1f}%")
            
            return {
                'success': True,
                'metrics': self.metrics,
                'training_history': history.history
            }
            
        except Exception as e:
            print(f"❌ LSTM training error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _train_rf_model(self, X_train, X_test, y_train, y_test) -> Dict:
        """Train Random Forest model"""
        try:
            # Scale features
            X_train_scaled = self.feature_scaler.fit_transform(X_train)
            X_test_scaled = self.feature_scaler.transform(X_test)
            
            # Create model
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            # Train model
            self.model.fit(X_train_scaled, y_train)
            
            # Predictions
            predictions = self.model.predict(X_test_scaled)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            
            # Direction accuracy
            actual_direction = np.sign(np.diff(y_test))
            pred_direction = np.sign(np.diff(predictions))
            accuracy = np.mean(actual_direction == pred_direction) * 100
            
            self.metrics.update({
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'accuracy': accuracy,
                'last_trained': datetime.now(),
                'training_samples': len(X_train)
            })
            
            self.is_trained = True
            
            print(f"✅ Random Forest model trained successfully!")
            print(f"   • MSE: {mse:.6f}")
            print(f"   • MAE: {mae:.6f}")
            print(f"   • R²: {r2:.4f}")
            print(f"   • Direction Accuracy: {accuracy:.1f}%")
            
            return {'success': True, 'metrics': self.metrics}
            
        except Exception as e:
            print(f"❌ Random Forest training error: {e}")
            return {'success': False, 'error': str(e)}
    
    def predict_next_price(self, recent_data: pd.DataFrame) -> Dict:
        """Predict next price based on recent data"""
        try:
            if not self.is_trained:
                return {'error': 'Model not trained'}
            
            # Prepare features
            df_features = self.prepare_features(recent_data)
            
            if len(df_features) == 0:
                return {'error': 'No valid features generated'}
            
            # Get feature columns
            feature_columns = [col for col in df_features.columns 
                             if col not in ['timestamp', 'target', 'input_token', 'output_token', 'profitable']]
            
            # Get latest features
            latest_features = df_features[feature_columns].iloc[-1:].values
            
            if self.model_type == "lstm":
                # For LSTM, we need sequence data
                if len(df_features) >= self.sequence_length:
                    sequence_data = self.feature_scaler.transform(
                        df_features[feature_columns].iloc[-self.sequence_length:].values
                    )
                    sequence_data = sequence_data.reshape(1, self.sequence_length, -1)
                    
                    prediction_scaled = self.model.predict(sequence_data, verbose=0)
                    prediction = self.scaler.inverse_transform(prediction_scaled.reshape(-1, 1))[0, 0]
                else:
                    return {'error': f'Need at least {self.sequence_length} data points for LSTM prediction'}
            else:
                # For traditional ML models
                latest_features_scaled = self.feature_scaler.transform(latest_features)
                prediction = self.model.predict(latest_features_scaled)[0]
            
            # Calculate confidence based on recent model performance
            confidence = max(0.1, min(0.9, self.metrics.get('r2', 0.5)))
            
            # Direction prediction
            current_price = recent_data['amount_out'].iloc[-1]
            direction = 'up' if prediction > current_price else 'down'
            price_change_pct = ((prediction - current_price) / current_price) * 100
            
            return {
                'predicted_price': float(prediction),
                'current_price': float(current_price),
                'price_change': float(prediction - current_price),
                'price_change_pct': float(price_change_pct),
                'direction': direction,
                'confidence': float(confidence),
                'model_type': self.model_type,
                'prediction_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return {'error': str(e)}
    
    def save_model(self, filepath: str = None):
        """Save trained model"""
        try:
            if not self.is_trained:
                print("⚠️ No trained model to save")
                return False
            
            if filepath is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = f"ml/models/{self.model_type}_model_{timestamp}"
            
            if self.model_type == "lstm":
                self.model.save(f"{filepath}.h5")
            else:
                joblib.dump(self.model, f"{filepath}.pkl")
            
            # Save scalers and metadata
            joblib.dump({
                'feature_scaler': self.feature_scaler,
                'target_scaler': self.scaler,
                'metrics': self.metrics,
                'model_type': self.model_type,
                'sequence_length': self.sequence_length
            }, f"{filepath}_metadata.pkl")
            
            print(f"✅ Model saved: {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False
    
    def load_model(self, filepath: str):
        """Load trained model"""
        try:
            # Load metadata
            metadata = joblib.load(f"{filepath}_metadata.pkl")
            self.feature_scaler = metadata['feature_scaler']
            self.scaler = metadata['target_scaler']
            self.metrics = metadata['metrics']
            self.model_type = metadata['model_type']
            self.sequence_length = metadata.get('sequence_length', 60)
            
            # Load model
            if self.model_type == "lstm":
                self.model = keras.models.load_model(f"{filepath}.h5")
            else:
                self.model = joblib.load(f"{filepath}.pkl")
            
            self.is_trained = True
            print(f"✅ Model loaded: {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

class MLTradingIntegration:
    """Integration class for ML predictions with trading system"""
    
    def __init__(self):
        self.models = {
            'lstm': PricePredictionModel('lstm'),
            'random_forest': PricePredictionModel('random_forest'),
            'gradient_boost': PricePredictionModel('gradient_boost')
        }
        self.active_model = 'lstm'
        self.prediction_cache = {}
        self.last_training_time = None
        
    def train_all_models(self, df: pd.DataFrame):
        """Train all available models"""
        results = {}
        
        for model_name, model in self.models.items():
            print(f"\n🤖 Training {model_name} model...")
            result = model.train_model(df)
            results[model_name] = result
            
            if result.get('success'):
                model.save_model()
        
        self.last_training_time = datetime.now()
        return results
    
    def get_ensemble_prediction(self, recent_data: pd.DataFrame) -> Dict:
        """Get ensemble prediction from multiple models"""
        predictions = []
        confidences = []
        
        for model_name, model in self.models.items():
            if model.is_trained:
                pred = model.predict_next_price(recent_data)
                if 'predicted_price' in pred:
                    predictions.append(pred['predicted_price'])
                    confidences.append(pred['confidence'])
        
        if not predictions:
            return {'error': 'No trained models available'}
        
        # Ensemble prediction (weighted average)
        weights = np.array(confidences)
        weights = weights / weights.sum()
        
        ensemble_price = np.average(predictions, weights=weights)
        ensemble_confidence = np.mean(confidences)
        
        current_price = recent_data['amount_out'].iloc[-1]
        direction = 'up' if ensemble_price > current_price else 'down'
        price_change_pct = ((ensemble_price - current_price) / current_price) * 100
        
        return {
            'predicted_price': float(ensemble_price),
            'current_price': float(current_price),
            'price_change_pct': float(price_change_pct),
            'direction': direction,
            'confidence': float(ensemble_confidence),
            'model_count': len(predictions),
            'individual_predictions': dict(zip(self.models.keys(), predictions)),
            'prediction_time': datetime.now().isoformat()
        }
    
    def should_retrain(self) -> bool:
        """Check if models should be retrained"""
        if self.last_training_time is None:
            return True
        
        # Retrain every 24 hours
        time_since_training = datetime.now() - self.last_training_time
        return time_since_training > timedelta(hours=24)
    
    def get_model_performance(self) -> Dict:
        """Get performance metrics for all models"""
        performance = {}
        
        for model_name, model in self.models.items():
            if model.is_trained:
                performance[model_name] = model.metrics
        
        return performance