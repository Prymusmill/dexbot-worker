# ml/price_predictor.py - OPTIMIZED Multi-Model Ensemble
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import os
import logging
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

class OptimizedPricePredictionModel:
    """
    OPTIMIZED: Multi-algorithm ML model with advanced feature engineering
    """
    
    def __init__(self, model_type: str = "ensemble"):
        self.model_type = model_type.lower()
        self.models = {}
        self.scalers = {}
        self.feature_scaler = RobustScaler()  # More robust than StandardScaler
        self.is_trained = False
        
        # Enhanced model configurations
        self.model_configs = {
            'random_forest': {
                'model': RandomForestRegressor,
                'params': {
                    'n_estimators': [50, 100, 150],
                    'max_depth': [6, 8, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'gradient_boost': {
                'model': GradientBoostingRegressor,
                'params': {
                    'n_estimators': [50, 100, 150],
                    'max_depth': [3, 4, 6],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'extra_trees': {
                'model': ExtraTreesRegressor,
                'params': {
                    'n_estimators': [50, 100],
                    'max_depth': [6, 8, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
            },
            'ridge': {
                'model': Ridge,
                'params': {
                    'alpha': [0.1, 1.0, 10.0, 100.0],
                    'solver': ['auto', 'svd', 'cholesky']
                }
            },
            'elastic_net': {
                'model': ElasticNet,
                'params': {
                    'alpha': [0.1, 1.0, 10.0],
                    'l1_ratio': [0.1, 0.5, 0.7, 0.9]
                }
            }
        }
        
        # Performance tracking
        self.model_performance = {}
        self.ensemble_weights = {}
        
        # Feature importance tracking
        self.feature_importance = {}
        
        self.logger = logging.getLogger(__name__)
        os.makedirs("ml/models", exist_ok=True)
        
    def prepare_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ENHANCED: Advanced feature engineering with more indicators"""
        try:
            print(f"🔍 Advanced feature preparation with {len(df)} rows")
            
            df = df.copy().sort_values('timestamp')
            
            # Price column handling
            if 'price' in df.columns and df['price'].notna().sum() > len(df) * 0.5:
                price_col = 'price'
            else:
                df['price'] = pd.to_numeric(df['amount_out'], errors='coerce')
                price_col = 'price'
            
            df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
            df = df.dropna(subset=[price_col])
            
            if len(df) < 30:
                print(f"❌ Insufficient data: {len(df)}")
                return pd.DataFrame()
            
            # ENHANCED FEATURE ENGINEERING
            
            # === PRICE FEATURES ===
            df['price_change'] = df[price_col].pct_change().fillna(0)
            df['price_change_abs'] = abs(df['price_change'])
            
            # Multiple timeframe moving averages
            for period in [3, 5, 10, 15, 20]:
                df[f'sma_{period}'] = df[price_col].rolling(period, min_periods=1).mean()
                df[f'price_vs_sma_{period}'] = (df[price_col] - df[f'sma_{period}']) / df[f'sma_{period}']
            
            # Exponential moving averages
            df['ema_5'] = df[price_col].ewm(span=5).mean()
            df['ema_10'] = df[price_col].ewm(span=10).mean()
            df['ema_20'] = df[price_col].ewm(span=20).mean()
            
            # Price momentum features
            df['momentum_3'] = df[price_col] / df[price_col].shift(3) - 1
            df['momentum_5'] = df[price_col] / df[price_col].shift(5) - 1
            df['momentum_10'] = df[price_col] / df[price_col].shift(10) - 1
            
            # === VOLATILITY FEATURES ===
            for period in [5, 10, 15, 20]:
                df[f'volatility_{period}'] = df[price_col].rolling(period, min_periods=1).std()
                df[f'volatility_norm_{period}'] = df[f'volatility_{period}'] / df[price_col]
            
            # Bollinger Bands
            df['bb_middle'] = df[price_col].rolling(20, min_periods=1).mean()
            df['bb_std'] = df[price_col].rolling(20, min_periods=1).std()
            df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
            df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
            df['bb_position'] = (df[price_col] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            
            # === TECHNICAL INDICATORS ===
            
            # Enhanced RSI
            if 'rsi' in df.columns:
                df['rsi_clean'] = pd.to_numeric(df['rsi'], errors='coerce').fillna(50.0)
            else:
                df['rsi_clean'] = self._calculate_rsi(df[price_col])
            
            df['rsi_oversold'] = (df['rsi_clean'] < 30).astype(int)
            df['rsi_overbought'] = (df['rsi_clean'] > 70).astype(int)
            df['rsi_neutral'] = ((df['rsi_clean'] >= 40) & (df['rsi_clean'] <= 60)).astype(int)
            
            # MACD
            df['macd_line'], df['macd_signal'] = self._calculate_macd(df[price_col])
            df['macd_histogram'] = df['macd_line'] - df['macd_signal']
            df['macd_bullish'] = (df['macd_histogram'] > 0).astype(int)
            
            # Stochastic Oscillator
            df['stoch_k'], df['stoch_d'] = self._calculate_stochastic(df[price_col])
            
            # === VOLUME FEATURES ===
            if 'volume' in df.columns:
                df['volume_clean'] = pd.to_numeric(df['volume'], errors='coerce')
            else:
                df['volume_clean'] = pd.to_numeric(df['amount_in'], errors='coerce')
            
            df['volume_sma_5'] = df['volume_clean'].rolling(5, min_periods=1).mean()
            df['volume_sma_10'] = df['volume_clean'].rolling(10, min_periods=1).mean()
            df['volume_ratio'] = df['volume_clean'] / df['volume_sma_10']
            df['volume_spike'] = (df['volume_ratio'] > 1.5).astype(int)
            
            # === MARKET STRUCTURE FEATURES ===
            
            # Support/Resistance levels
            df['recent_high'] = df[price_col].rolling(10, min_periods=1).max()
            df['recent_low'] = df[price_col].rolling(10, min_periods=1).min()
            df['price_position'] = (df[price_col] - df['recent_low']) / (df['recent_high'] - df['recent_low'])
            
            # Trend strength
            df['trend_strength'] = abs(df['sma_5'] - df['sma_20']) / df['sma_20']
            df['trend_direction'] = np.where(df['sma_5'] > df['sma_20'], 1, -1)
            
            # Price acceleration
            df['price_acceleration'] = df['price_change'].diff()
            
            # === TIME-BASED FEATURES ===
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
            
            # === MARKET REGIME FEATURES ===
            
            # Volatility regime
            df['vol_regime'] = pd.qcut(df['volatility_20'].fillna(df['volatility_20'].median()), 
                                     q=3, labels=[0, 1, 2]).astype(int)
            
            # Trend regime
            df['trend_regime'] = np.where(df['trend_strength'] > df['trend_strength'].quantile(0.7), 2,
                                        np.where(df['trend_strength'] > df['trend_strength'].quantile(0.3), 1, 0))
            
            # === TARGET VARIABLE ===
            df['target'] = df[price_col].shift(-1)
            
            # === FEATURE SELECTION ===
            feature_columns = [
                # Price features
                'price_change', 'price_change_abs', 'price_vs_sma_5', 'price_vs_sma_10', 'price_vs_sma_20',
                'momentum_3', 'momentum_5', 'momentum_10',
                
                # Volatility features
                'volatility_5', 'volatility_10', 'volatility_norm_5', 'volatility_norm_10',
                'bb_position', 'bb_width',
                
                # Technical indicators
                'rsi_clean', 'rsi_oversold', 'rsi_overbought', 'rsi_neutral',
                'macd_histogram', 'macd_bullish', 'stoch_k', 'stoch_d',
                
                # Volume features
                'volume_ratio', 'volume_spike',
                
                # Market structure
                'price_position', 'trend_strength', 'trend_direction', 'price_acceleration',
                
                # Time features
                'hour', 'day_of_week', 'is_weekend', 'is_night',
                
                # Regime features
                'vol_regime', 'trend_regime'
            ]
            
            # Keep only existing columns
            available_features = [col for col in feature_columns if col in df.columns and not df[col].isna().all()]
            
            print(f"📋 Available advanced features: {len(available_features)}")
            
            if len(available_features) < 10:
                print(f"❌ Too few features: {len(available_features)}")
                return pd.DataFrame()
            
            # Clean data
            df = df[:-1]  # Remove last row (no target)
            feature_data = df[available_features]
            
            # Fill NaN with medians
            for col in available_features:
                df[col] = df[col].fillna(df[col].median())
            
            df['target'] = df['target'].fillna(method='ffill')
            final_clean = df.dropna(subset=['target'])
            
            result_df = final_clean[available_features + ['target']].copy()
            
            print(f"✅ Advanced features prepared: {result_df.shape}")
            return result_df
            
        except Exception as e:
            print(f"❌ Advanced feature preparation failed: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Enhanced RSI calculation"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
        rs = gain / (loss + 1e-10)
        return (100 - (100 / (1 + rs))).fillna(50.0)
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Enhanced MACD calculation"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean() 
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal).mean()
        return macd_line.fillna(0), macd_signal.fillna(0)
    
    def _calculate_stochastic(self, prices: pd.Series, window: int = 14) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator calculation"""
        high_roll = prices.rolling(window).max()
        low_roll = prices.rolling(window).min()
        stoch_k = 100 * (prices - low_roll) / (high_roll - low_roll + 1e-10)
        stoch_d = stoch_k.rolling(3).mean()
        return stoch_k.fillna(50), stoch_d.fillna(50)
    
    def train_ensemble_models(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict:
        """Train multiple models with hyperparameter optimization"""
        try:
            print(f"🤖 Training ensemble models with {len(df)} samples...")
            
            # Prepare features
            df_features = self.prepare_advanced_features(df)
            
            if df_features.empty or len(df_features) < 50:
                return {'success': False, 'error': f'Insufficient data: {len(df_features)}'}
            
            # Extract features and target
            feature_cols = [col for col in df_features.columns if col != 'target']
            X = df_features[feature_cols].values.astype(np.float64)
            y = df_features['target'].values.astype(np.float64)
            
            # Remove infinite values
            finite_mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
            X, y = X[finite_mask], y[finite_mask]
            
            if len(X) < 30:
                return {'success': False, 'error': f'Too few clean samples: {len(X)}'}
            
            # Train-test split
            split_idx = max(10, int(len(X) * (1 - test_size)))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Scale features
            X_train_scaled = self.feature_scaler.fit_transform(X_train)
            X_test_scaled = self.feature_scaler.transform(X_test)
            
            print(f"📊 Training on {len(X_train)} samples, testing on {len(X_test)}")
            
            # Train multiple models
            results = {}
            for model_name, config in self.model_configs.items():
                try:
                    print(f"🔄 Training {model_name}...")
                    
                    # Quick training for large datasets
                    if len(X_train) > 200:
                        # Use simplified parameters for speed
                        if model_name == 'random_forest':
                            model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=1)
                        elif model_name == 'gradient_boost':
                            model = GradientBoostingRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
                        elif model_name == 'extra_trees':
                            model = ExtraTreesRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=1)
                        elif model_name == 'ridge':
                            model = Ridge(alpha=1.0)
                        elif model_name == 'elastic_net':
                            model = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=2000)
                    else:
                        # Use GridSearchCV for smaller datasets
                        base_model = config['model'](random_state=42)
                        # Simplified parameter grid for speed
                        simplified_params = {}
                        for param, values in config['params'].items():
                            simplified_params[param] = values[:2]  # Take only first 2 values
                        
                        model = GridSearchCV(base_model, simplified_params, cv=3, scoring='neg_mean_squared_error', n_jobs=1)
                    
                    # Train model
                    model.fit(X_train_scaled, y_train)
                    
                    # Get best estimator if GridSearch
                    if hasattr(model, 'best_estimator_'):
                        best_model = model.best_estimator_
                    else:
                        best_model = model
                    
                    # Make predictions
                    train_pred = best_model.predict(X_train_scaled)
                    test_pred = best_model.predict(X_test_scaled)
                    
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
                    
                    # Store model and performance
                    self.models[model_name] = best_model
                    self.model_performance[model_name] = {
                        'train_mse': train_mse,
                        'test_mse': test_mse,
                        'test_mae': test_mae,
                        'test_r2': test_r2,
                        'accuracy': accuracy,
                        'training_samples': len(X_train)
                    }
                    
                    # Feature importance (if available)
                    if hasattr(best_model, 'feature_importances_'):
                        self.feature_importance[model_name] = dict(zip(feature_cols, best_model.feature_importances_))
                    
                    results[model_name] = {
                        'success': True,
                        'test_r2': test_r2,
                        'accuracy': accuracy,
                        'test_mae': test_mae
                    }
                    
                    print(f"✅ {model_name}: R²={test_r2:.3f}, Acc={accuracy:.1f}%, MAE={test_mae:.6f}")
                    
                except Exception as e:
                    print(f"❌ {model_name} training failed: {e}")
                    results[model_name] = {'success': False, 'error': str(e)}
            
            # Calculate ensemble weights based on performance
            self._calculate_ensemble_weights()
            
            # Mark as trained if at least one model succeeded
            successful_models = [name for name, result in results.items() if result.get('success')]
            if successful_models:
                self.is_trained = True
                print(f"✅ Ensemble training complete! Successful models: {successful_models}")
            else:
                print("❌ All models failed to train")
                return {'success': False, 'error': 'All models failed'}
            
            return {'success': True, 'results': results, 'successful_models': successful_models}
            
        except Exception as e:
            print(f"❌ Ensemble training error: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
    
    def _calculate_ensemble_weights(self):
        """Calculate weights for ensemble based on model performance"""
        weights = {}
        total_score = 0
        
        for model_name, performance in self.model_performance.items():
            # Weight based on R² score and accuracy
            r2_score = max(0, performance.get('test_r2', 0))
            accuracy = performance.get('accuracy', 0) / 100
            
            # Combined score (R² weighted more heavily)
            combined_score = 0.7 * r2_score + 0.3 * accuracy
            weights[model_name] = max(0.1, combined_score)  # Minimum weight of 0.1
            total_score += weights[model_name]
        
        # Normalize weights
        if total_score > 0:
            for model_name in weights:
                weights[model_name] /= total_score
        
        self.ensemble_weights = weights
        print(f"📊 Ensemble weights: {self.ensemble_weights}")
    
    def predict_ensemble(self, recent_data: pd.DataFrame) -> Dict:
        """Make ensemble prediction using all trained models"""
        try:
            if not self.is_trained or not self.models:
                return {'error': 'No trained models available'}
            
            # Prepare features
            df_features = self.prepare_advanced_features(recent_data)
            if df_features.empty:
                return {'error': 'No valid features generated'}
            
            # Get latest features
            feature_cols = [col for col in df_features.columns if col != 'target']
            latest_features = df_features[feature_cols].iloc[-1:].values
            
            # Scale features
            latest_features_scaled = self.feature_scaler.transform(latest_features)
            
            # Get predictions from all models
            predictions = {}
            for model_name, model in self.models.items():
                try:
                    pred = model.predict(latest_features_scaled)[0]
                    predictions[model_name] = float(pred)
                except Exception as e:
                    print(f"⚠️ {model_name} prediction failed: {e}")
            
            if not predictions:
                return {'error': 'No valid predictions generated'}
            
            # Calculate weighted ensemble prediction
            if self.ensemble_weights:
                weighted_sum = sum(pred * self.ensemble_weights.get(name, 0) 
                                 for name, pred in predictions.items())
                ensemble_prediction = weighted_sum
                ensemble_confidence = sum(self.ensemble_weights.get(name, 0) 
                                        for name in predictions.keys())
            else:
                # Simple average if no weights
                ensemble_prediction = np.mean(list(predictions.values()))
                ensemble_confidence = 0.5
            
            # Get current price
            if 'price' in recent_data.columns:
                current_price = recent_data['price'].iloc[-1]
            else:
                current_price = recent_data['amount_out'].iloc[-1]
            
            current_price = float(current_price)
            ensemble_prediction = float(ensemble_prediction)
            
            # Calculate metrics
            direction = 'up' if ensemble_prediction > current_price else 'down'
            price_change = ensemble_prediction - current_price
            price_change_pct = (price_change / current_price) * 100
            
            # Enhanced confidence calculation
            model_agreement = self._calculate_model_agreement(predictions, current_price)
            final_confidence = min(0.95, ensemble_confidence * 0.7 + model_agreement * 0.3)
            
            result = {
                'predicted_price': ensemble_prediction,
                'current_price': current_price,
                'price_change': price_change,
                'price_change_pct': price_change_pct,
                'direction': direction,
                'confidence': float(final_confidence),
                'model_count': len(predictions),
                'individual_predictions': predictions,
                'ensemble_weights': self.ensemble_weights,
                'model_agreement': model_agreement,
                'prediction_time': datetime.now().isoformat(),
                'features_used': len(feature_cols)
            }
            
            print(f"🎯 Ensemble: {direction.upper()} ${ensemble_prediction:.4f} "
                  f"(confidence: {final_confidence:.2f}, agreement: {model_agreement:.2f})")
            
            return result
            
        except Exception as e:
            print(f"❌ Ensemble prediction error: {e}")
            return {'error': str(e)}
    
    def _calculate_model_agreement(self, predictions: Dict, current_price: float) -> float:
        """Calculate how much models agree on direction"""
        if len(predictions) < 2:
            return 0.5
        
        directions = [1 if pred > current_price else -1 for pred in predictions.values()]
        agreement = abs(sum(directions)) / len(directions)
        return agreement

class MLTradingIntegration:
    """OPTIMIZED: Enhanced ML integration with multi-model ensemble"""
    
    def __init__(self):
        self.ensemble_model = OptimizedPricePredictionModel('ensemble')
        self.last_training_time = None
        self.training_in_progress = False
        
    def train_models(self, df: pd.DataFrame):
        """Train ensemble models"""
        if self.training_in_progress:
            return {'success': False, 'error': 'Training already in progress'}
        
        try:
            self.training_in_progress = True
            print(f"🤖 Starting ensemble training with {len(df)} samples...")
            
            result = self.ensemble_model.train_ensemble_models(df)
            
            if result.get('success'):
                self.last_training_time = datetime.now()
                print("✅ Ensemble training completed successfully!")
                
                # Save models
                self.ensemble_model.save_ensemble_models()
            else:
                print(f"❌ Ensemble training failed: {result.get('error')}")
            
            return result
            
        finally:
            self.training_in_progress = False
    
    def get_ensemble_prediction(self, recent_data: pd.DataFrame) -> Dict:
        """Get enhanced ensemble prediction"""
        # Auto-train if no models or insufficient data
        if not self.ensemble_model.is_trained and len(recent_data) >= 100:
            print("🤖 Auto-training ensemble models...")
            training_result = self.train_models(recent_data)
            if not training_result.get('success'):
                return {'error': f'Auto-training failed: {training_result.get("error")}'}
        
        if not self.ensemble_model.is_trained:
            return {'error': f'No trained models (need 100+ samples, have {len(recent_data)})'}
        
        return self.ensemble_model.predict_ensemble(recent_data)
    
    def get_model_performance(self) -> Dict:
        """Get detailed performance metrics for all models"""
        if not self.ensemble_model.model_performance:
            return {}
        
        performance = {}
        for model_name, metrics in self.ensemble_model.model_performance.items():
            performance[model_name] = {
                'accuracy': metrics.get('accuracy', 0),
                'r2': metrics.get('test_r2', 0),
                'mae': metrics.get('test_mae', 0),
                'training_samples': metrics.get('training_samples', 0),
                'last_trained': self.last_training_time.strftime('%Y-%m-%d %H:%M:%S') if self.last_training_time else 'Never',
                'ensemble_weight': self.ensemble_model.ensemble_weights.get(model_name, 0)
            }
        
        return performance
    
    def should_retrain(self) -> bool:
        """Check if models should be retrained"""
        if self.last_training_time is None:
            return True
        
        # Retrain every 4 hours for optimal performance
        time_since_training = datetime.now() - self.last_training_time
        return time_since_training > timedelta(hours=4)
    
    def get_feature_importance(self) -> Dict:
        """Get feature importance from ensemble models"""
        return self.ensemble_model.feature_importance

    def save_ensemble_models(self):
        """Save all trained models"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            for model_name, model in self.models.items():
                filepath = f"ml/models/ensemble_{model_name}_{timestamp}"
                joblib.dump(model, f"{filepath}.pkl")
            
            # Save metadata
            metadata = {
                'feature_scaler': self.feature_scaler,
                'model_performance': self.model_performance,
                'ensemble_weights': self.ensemble_weights,
                'feature_importance': self.feature_importance,
                'training_time': datetime.now().isoformat()
            }
            
            joblib.dump(metadata, f"ml/models/ensemble_metadata_{timestamp}.pkl")
            print(f"✅ Ensemble models saved: {timestamp}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving ensemble models: {e}")
            return False