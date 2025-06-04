# ml/price_predictor.py - COMPLETE REWRITE: ULTRA ROBUST ENHANCED ML
import pandas as pd
import numpy as np
import logging
import os
import json
import pickle
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import traceback

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Core ML imports
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    import joblib
    SKLEARN_AVAILABLE = True
    print("✅ scikit-learn loaded successfully")
except ImportError as e:
    print(f"❌ scikit-learn import failed: {e}")
    SKLEARN_AVAILABLE = False

# Advanced ML imports (optional)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    print("✅ XGBoost available")
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not available")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
    print("✅ LightGBM available")
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM not available")

# Database imports
try:
    from sqlalchemy import create_engine
    SQLALCHEMY_AVAILABLE = True
    print("✅ SQLAlchemy available")
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    print("⚠️ SQLAlchemy not available - PostgreSQL reading may have issues")


@dataclass
class ModelPerformance:
    """Model performance metrics"""
    name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    cross_val_score: float
    training_samples: int
    feature_count: int
    training_time: float
    ensemble_weight: float = 0.0


class RobustDataLoader:
    """Ultra robust data loading with multiple fallback methods"""
    
    def __init__(self, db_manager=None):
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
        
    def load_training_data(self, min_samples: int = 100) -> Optional[pd.DataFrame]:
        """Load data with multiple fallback methods"""
        print(f"🔍 ROBUST DATA LOADING: Attempting to load {min_samples}+ samples...")
        
        # Method 1: PostgreSQL with SQLAlchemy (PREFERRED)
        df = self._load_from_postgresql_sqlalchemy(min_samples)
        if df is not None and len(df) >= min_samples:
            print(f"✅ LOADED FROM POSTGRESQL (SQLAlchemy): {len(df)} records")
            return df
            
        # Method 2: PostgreSQL with psycopg2 (FALLBACK 1)
        df = self._load_from_postgresql_psycopg2(min_samples)
        if df is not None and len(df) >= min_samples:
            print(f"✅ LOADED FROM POSTGRESQL (psycopg2): {len(df)} records")
            return df
            
        # Method 3: CSV fallback (FALLBACK 2)
        df = self._load_from_csv(min_samples)
        if df is not None and len(df) >= min_samples:
            print(f"✅ LOADED FROM CSV: {len(df)} records")
            return df
            
        print(f"❌ ALL DATA LOADING METHODS FAILED - need {min_samples}+ samples")
        return None
    
    def _load_from_postgresql_sqlalchemy(self, min_samples: int) -> Optional[pd.DataFrame]:
        """Load using SQLAlchemy (most robust for pandas)"""
        if not SQLALCHEMY_AVAILABLE or not self.db_manager:
            return None
            
        try:
            # Get DATABASE_URL from environment
            database_url = os.getenv('DATABASE_URL')
            if not database_url:
                print("⚠️ DATABASE_URL not found in environment")
                return None
                
            # Create SQLAlchemy engine
            engine = create_engine(database_url)
            
            # Query with proper SQL
            query = """
                SELECT timestamp, price, volume, rsi, amount_in, amount_out,
                       price_impact, profitable, input_token, output_token
                FROM transactions
                WHERE price IS NOT NULL AND price > 0 
                  AND rsi IS NOT NULL AND rsi BETWEEN 0 AND 100
                  AND volume IS NOT NULL AND volume > 0
                ORDER BY timestamp DESC
                LIMIT 5000;
            """
            
            print("🔍 Executing SQLAlchemy query...")
            df = pd.read_sql_query(query, engine)
            
            if len(df) > 0:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                print(f"✅ SQLAlchemy loaded {len(df)} valid records")
                return df
            else:
                print("⚠️ SQLAlchemy query returned no data")
                return None
                
        except Exception as e:
            print(f"⚠️ SQLAlchemy loading failed: {e}")
            return None
    
    def _load_from_postgresql_psycopg2(self, min_samples: int) -> Optional[pd.DataFrame]:
        """Load using direct psycopg2 (fallback)"""
        if not self.db_manager:
            return None
            
        try:
            print("🔍 Attempting direct psycopg2 loading...")
            df = self.db_manager.get_all_transactions_for_ml()
            
            if len(df) > 0:
                # Validate required columns
                required_cols = ['price', 'volume', 'rsi', 'amount_in', 'amount_out']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    print(f"⚠️ Missing columns in psycopg2 data: {missing_cols}")
                    return None
                    
                # Clean data
                df = df.dropna(subset=required_cols)
                df = df[df['price'] > 0]
                df = df[(df['rsi'] >= 0) & (df['rsi'] <= 100)]
                df = df[df['volume'] > 0]
                
                print(f"✅ psycopg2 loaded {len(df)} clean records")
                return df
            else:
                print("⚠️ psycopg2 returned no data")
                return None
                
        except Exception as e:
            print(f"⚠️ psycopg2 loading failed: {e}")
            return None
    
    def _load_from_csv(self, min_samples: int) -> Optional[pd.DataFrame]:
        """Load from CSV backup"""
        csv_path = "data/memory.csv"
        
        if not os.path.exists(csv_path):
            print("⚠️ CSV file not found")
            return None
            
        try:
            print("🔍 Attempting CSV loading...")
            df = pd.read_csv(csv_path)
            
            if len(df) > 0:
                # Ensure required columns exist
                required_cols = ['price', 'volume', 'rsi', 'amount_in', 'amount_out']
                
                # Add missing columns with defaults if needed
                for col in required_cols:
                    if col not in df.columns:
                        if col == 'volume':
                            df[col] = df.get('amount_in', 0.02)
                        elif col == 'rsi':
                            df[col] = 50.0
                        else:
                            df[col] = 0.0
                
                # Clean data
                df = df.dropna(subset=required_cols)
                df = df[df['price'] > 0]
                
                # Add profitable column if missing
                if 'profitable' not in df.columns:
                    df['profitable'] = df['amount_out'] > df['amount_in']
                
                print(f"✅ CSV loaded {len(df)} records")
                return df.tail(min_samples * 2)  # Get recent data
            else:
                print("⚠️ CSV file is empty")
                return None
                
        except Exception as e:
            print(f"⚠️ CSV loading failed: {e}")
            return None


class AdvancedFeatureEngineer:
    """Advanced feature engineering with robust error handling"""
    
    @staticmethod
    def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced features for ML"""
        print("🔧 FEATURE ENGINEERING: Creating advanced features...")
        
        try:
            # Work on a copy
            df = df.copy()
            
            # Sort by timestamp
            df = df.sort_values('timestamp')
            
            # 1. Price-based features
            df['price_change'] = df['price'].pct_change()
            df['price_ma_5'] = df['price'].rolling(5, min_periods=1).mean()
            df['price_ma_20'] = df['price'].rolling(20, min_periods=1).mean()
            df['price_volatility'] = df['price'].rolling(10, min_periods=1).std()
            
            # 2. RSI-based features
            df['rsi_normalized'] = (df['rsi'] - 50) / 50  # Normalize RSI to [-1, 1]
            df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
            df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
            df['rsi_momentum'] = df['rsi'].diff()
            
            # 3. Volume features
            df['volume_ma'] = df['volume'].rolling(10, min_periods=1).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            
            # 4. Time-based features
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            
            # 5. Lag features (previous values)
            for lag in [1, 2, 3]:
                df[f'price_lag_{lag}'] = df['price'].shift(lag)
                df[f'rsi_lag_{lag}'] = df['rsi'].shift(lag)
                df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
            
            # 6. Technical indicators
            df['price_to_ma_ratio'] = df['price'] / df['price_ma_20']
            df['rsi_divergence'] = df['rsi'] - df['rsi'].rolling(5, min_periods=1).mean()
            
            # 7. Profit target (for classification)
            if 'profitable' not in df.columns:
                df['profitable'] = df['amount_out'] > df['amount_in']
            
            # Remove rows with NaN (from rolling/lag operations)
            initial_len = len(df)
            df = df.dropna()
            final_len = len(df)
            
            print(f"✅ Feature engineering complete: {initial_len} → {final_len} samples")
            print(f"📊 Created {len(df.columns)} total features")
            
            return df
            
        except Exception as e:
            print(f"❌ Feature engineering failed: {e}")
            traceback.print_exc()
            return df  # Return original if feature engineering fails


class RobustModelTrainer:
    """Ultra robust model training with comprehensive error handling"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.performance = {}
        self.feature_names = []
        
    def get_model_configs(self) -> Dict[str, Any]:
        """Get model configurations with conservative parameters"""
        configs = {}
        
        # Random Forest (always available)
        if SKLEARN_AVAILABLE:
            configs['random_forest'] = {
                'model': RandomForestClassifier(
                    n_estimators=50,  # Reduced for speed
                    max_depth=10,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1
                ),
                'requires_scaling': False
            }
            
            configs['gradient_boost'] = {
                'model': GradientBoostingClassifier(
                    n_estimators=50,
                    max_depth=6,
                    learning_rate=0.1,
                    min_samples_split=10,
                    random_state=42
                ),
                'requires_scaling': False
            }
            
            configs['logistic'] = {
                'model': LogisticRegression(
                    random_state=42,
                    max_iter=500,
                    solver='liblinear'
                ),
                'requires_scaling': True
            }
        
        # XGBoost (if available)
        if XGBOOST_AVAILABLE:
            configs['xgboost'] = {
                'model': xgb.XGBClassifier(
                    n_estimators=50,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    eval_metric='logloss'
                ),
                'requires_scaling': False
            }
        
        # LightGBM (if available)
        if LIGHTGBM_AVAILABLE:
            configs['lightgbm'] = {
                'model': lgb.LGBMClassifier(
                    n_estimators=50,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    verbose=-1
                ),
                'requires_scaling': False
            }
        
        print(f"✅ Model configs ready: {list(configs.keys())}")
        return configs
    
    def train_single_model(self, name: str, config: Dict, X_train: np.ndarray, 
                          y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Optional[ModelPerformance]:
        """Train a single model with comprehensive error handling"""
        print(f"🔧 Training {name}...")
        
        start_time = datetime.now()
        
        try:
            model = config['model']
            requires_scaling = config['requires_scaling']
            
            # Apply scaling if required
            scaler = None
            if requires_scaling:
                scaler = RobustScaler()  # More robust than StandardScaler
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
            else:
                X_train_scaled = X_train
                X_val_scaled = X_val
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_val_scaled)
            y_pred_proba = model.predict_proba(X_val_scaled)[:, 1] if hasattr(model, 'predict_proba') else y_pred
            
            # Calculate metrics
            accuracy = accuracy_score(y_val, y_pred)
            
            # Cross-validation score (on smaller subset for speed)
            cv_size = min(1000, len(X_train))
            cv_X = X_train_scaled[:cv_size]
            cv_y = y_train[:cv_size]
            cv_scores = cross_val_score(model, cv_X, cv_y, cv=3, scoring='accuracy')
            cv_score = cv_scores.mean()
            
            # Classification report for precision/recall
            report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
            precision = report['weighted avg']['precision']
            recall = report['weighted avg']['recall']
            f1 = report['weighted avg']['f1-score']
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Store model and scaler
            self.models[name] = model
            if scaler:
                self.scalers[name] = scaler
            
            performance = ModelPerformance(
                name=name,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                cross_val_score=cv_score,
                training_samples=len(X_train),
                feature_count=X_train.shape[1],
                training_time=training_time
            )
            
            print(f"✅ {name}: Acc={accuracy:.3f}, CV={cv_score:.3f}, Time={training_time:.1f}s")
            return performance
            
        except Exception as e:
            print(f"❌ {name} training failed: {e}")
            return None
    
    def train_all_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, ModelPerformance]:
        """Train all available models"""
        print(f"🚀 TRAINING ALL MODELS: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"📊 Train: {len(X_train)}, Validation: {len(X_val)}")
        
        # Get model configurations
        model_configs = self.get_model_configs()
        
        # Train each model
        successful_models = {}
        for name, config in model_configs.items():
            performance = self.train_single_model(name, config, X_train, y_train, X_val, y_val)
            if performance:
                successful_models[name] = performance
                self.performance[name] = performance
        
        # Calculate ensemble weights based on cross-validation scores
        if successful_models:
            total_cv_score = sum(p.cross_val_score for p in successful_models.values())
            for name, performance in successful_models.items():
                performance.ensemble_weight = performance.cross_val_score / total_cv_score
        
        print(f"✅ TRAINING COMPLETE: {len(successful_models)}/{len(model_configs)} models successful")
        return successful_models


class MLTradingIntegration:
    """ULTRA ROBUST Enhanced ML Trading Integration - COMPLETE REWRITE"""
    
    def __init__(self, db_manager=None):
        print("🚀 INITIALIZING ENHANCED ML INTEGRATION...")
        
        # Core components
        self.db_manager = db_manager
        self.data_loader = RobustDataLoader(db_manager)
        self.feature_engineer = AdvancedFeatureEngineer()
        self.model_trainer = RobustModelTrainer()
        
        # Configuration
        self.min_samples = 100  # REDUCED from 500 for faster initial training
        self.model_dir = "ml/models"
        self.last_training_time = None
        self.training_data_hash = None
        
        # Performance tracking
        self.prediction_count = 0
        self.successful_predictions = 0
        
        # Ensure model directory exists
        os.makedirs(self.model_dir, exist_ok=True)
        
        print("✅ Enhanced ML Integration initialized")
        print(f"   • Min samples: {self.min_samples}")
        print(f"   • Model directory: {self.model_dir}")
        print(f"   • Database manager: {'✅' if self.db_manager else '❌'}")
    
    def should_retrain(self) -> bool:
        """Determine if models should be retrained"""
        try:
            # If no models exist, definitely retrain
            if not self.model_trainer.models:
                print("🔄 No models exist - retraining needed")
                return True
            
            # If never trained, retrain
            if not self.last_training_time:
                print("🔄 Never trained - retraining needed")
                return True
            
            # Time-based retraining (every 6 hours)
            time_since_training = datetime.now() - self.last_training_time
            if time_since_training > timedelta(hours=6):
                print(f"🔄 Time-based retrain needed ({time_since_training})")
                return True
            
            # Performance-based retraining
            if self.prediction_count > 50:
                success_rate = self.successful_predictions / self.prediction_count
                if success_rate < 0.6:
                    print(f"🔄 Performance-based retrain needed (success rate: {success_rate:.2f})")
                    return True
            
            return False
            
        except Exception as e:
            print(f"⚠️ Retrain check error: {e}")
            return False
    
    def train_models(self, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Train ML models with comprehensive error handling"""
        print("🚀 ENHANCED TRAINING STARTING...")
        
        try:
            # Load data if not provided
            if df is None:
                print("📊 Loading training data...")
                df = self.data_loader.load_training_data(self.min_samples)
                
                if df is None:
                    return {
                        'success': False,
                        'error': 'Failed to load sufficient training data',
                        'data_loaded': 0
                    }
            
            print(f"📊 Training data: {len(df)} samples")
            
            # Validate minimum samples
            if len(df) < self.min_samples:
                return {
                    'success': False,
                    'error': f'Insufficient data: {len(df)} < {self.min_samples}',
                    'data_loaded': len(df)
                }
            
            # Feature engineering
            print("🔧 Engineering features...")
            df_features = self.feature_engineer.engineer_features(df)
            
            if len(df_features) < self.min_samples // 2:
                return {
                    'success': False,
                    'error': 'Too much data lost in feature engineering',
                    'data_loaded': len(df),
                    'data_after_features': len(df_features)
                }
            
            # Prepare features and target
            target_col = 'profitable'
            feature_cols = [col for col in df_features.columns 
                          if col not in ['timestamp', 'profitable', 'input_token', 'output_token']]
            
            X = df_features[feature_cols].values
            y = df_features[target_col].values
            
            # Store feature names
            self.model_trainer.feature_names = feature_cols
            
            print(f"📊 Final training set: {X.shape[0]} samples, {X.shape[1]} features")
            print(f"📊 Target distribution: {np.bincount(y)}")
            
            # Train models
            successful_models = self.model_trainer.train_all_models(X, y)
            
            if successful_models:
                # Save models to database
                self._save_model_performance_to_db(successful_models)
                
                # Save models to disk
                self._save_models_to_disk()
                
                # Update tracking
                self.last_training_time = datetime.now()
                
                print(f"✅ TRAINING SUCCESS: {len(successful_models)} models trained")
                
                return {
                    'success': True,
                    'successful_models': list(successful_models.keys()),
                    'data_loaded': len(df),
                    'data_trained': len(df_features),
                    'features_count': len(feature_cols),
                    'model_performance': {name: {
                        'accuracy': perf.accuracy,
                        'cv_score': perf.cross_val_score,
                        'ensemble_weight': perf.ensemble_weight
                    } for name, perf in successful_models.items()}
                }
            else:
                return {
                    'success': False,
                    'error': 'No models trained successfully',
                    'data_loaded': len(df),
                    'attempted_models': len(self.model_trainer.get_model_configs())
                }
                
        except Exception as e:
            print(f"❌ TRAINING ERROR: {e}")
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def get_ensemble_prediction_with_reality_check(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get ensemble prediction with reality check"""
        self.prediction_count += 1
        
        try:
            print(f"🤖 GENERATING PREDICTION #{self.prediction_count}...")
            
            # Check if we have trained models
            if not self.model_trainer.models:
                print("⚠️ No trained models - attempting training...")
                training_result = self.train_models(df)
                
                if not training_result.get('success'):
                    return {
                        'error': 'No models available and training failed',
                        'training_error': training_result.get('error'),
                        'predicted_profitable': False,
                        'confidence': 0.1,
                        'recommendation': 'HOLD'
                    }
            
            # Prepare data for prediction
            df_features = self.feature_engineer.engineer_features(df.copy())
            
            if len(df_features) == 0:
                return {
                    'error': 'Feature engineering produced no data',
                    'predicted_profitable': False,
                    'confidence': 0.1,
                    'recommendation': 'HOLD'
                }
            
            # Get latest features
            latest_features = df_features.iloc[-1]
            feature_cols = self.model_trainer.feature_names
            
            # Check if we have required features
            missing_features = [col for col in feature_cols if col not in latest_features.index]
            if missing_features:
                print(f"⚠️ Missing features: {missing_features[:3]}...")
                # Use simplified prediction
                return self._simplified_prediction(latest_features)
            
            X_pred = latest_features[feature_cols].values.reshape(1, -1)
            
            # Get predictions from all models
            predictions = {}
            probabilities = {}
            
            for name, model in self.model_trainer.models.items():
                try:
                    # Apply scaling if needed
                    if name in self.model_trainer.scalers:
                        X_scaled = self.model_trainer.scalers[name].transform(X_pred)
                    else:
                        X_scaled = X_pred
                    
                    # Get prediction
                    pred = model.predict(X_scaled)[0]
                    
                    # Get probability if available
                    if hasattr(model, 'predict_proba'):
                        prob = model.predict_proba(X_scaled)[0, 1]
                    else:
                        prob = 0.5 + (pred - 0.5) * 0.3  # Convert to pseudo-probability
                    
                    predictions[name] = bool(pred)
                    probabilities[name] = prob
                    
                except Exception as e:
                    print(f"⚠️ Prediction error for {name}: {e}")
                    continue
            
            if not predictions:
                return {
                    'error': 'All model predictions failed',
                    'predicted_profitable': False,
                    'confidence': 0.1,
                    'recommendation': 'HOLD'
                }
            
            # Ensemble prediction using weights
            ensemble_prob = 0.0
            total_weight = 0.0
            
            for name, prob in probabilities.items():
                weight = self.model_trainer.performance.get(name, ModelPerformance(name, 0.5, 0, 0, 0, 0.5, 0, 0, 0)).ensemble_weight
                if weight > 0:
                    ensemble_prob += prob * weight
                    total_weight += weight
            
            if total_weight > 0:
                ensemble_prob /= total_weight
            else:
                ensemble_prob = sum(probabilities.values()) / len(probabilities)
            
            # Final prediction
            predicted_profitable = ensemble_prob > 0.5
            
            # Model agreement
            agree_count = sum(1 for pred in predictions.values() if pred == predicted_profitable)
            model_agreement = agree_count / len(predictions)
            
            # Confidence calculation
            base_confidence = abs(ensemble_prob - 0.5) * 2  # 0 to 1
            agreement_bonus = model_agreement * 0.3
            model_count_bonus = min(len(predictions) / 5, 0.2)
            
            confidence = min(base_confidence + agreement_bonus + model_count_bonus, 0.95)
            
            # Reality check
            reality_check = self._perform_reality_check(df_features, confidence, predicted_profitable)
            
            # Final confidence after reality check
            if reality_check.get('issues'):
                confidence *= 0.8
            
            # Generate recommendation
            if predicted_profitable and confidence > 0.7:
                recommendation = 'BUY'
            elif not predicted_profitable and confidence > 0.7:
                recommendation = 'SELL'
            else:
                recommendation = 'HOLD'
            
            self.successful_predictions += 1
            
            result = {
                'predicted_profitable': predicted_profitable,
                'probability_profitable': ensemble_prob,
                'confidence': confidence,
                'model_count': len(predictions),
                'model_agreement': model_agreement,
                'recommendation': recommendation,
                'direction': 'profitable' if predicted_profitable else 'unprofitable',
                'individual_predictions': {
                    name: {
                        'profitable': pred,
                        'probability': probabilities[name]
                    } for name, pred in predictions.items()
                },
                'reality_check': reality_check,
                'enhanced_metrics': {
                    'base_confidence': base_confidence,
                    'agreement_bonus': agreement_bonus,
                    'model_count_bonus': model_count_bonus
                }
            }
            
            print(f"✅ PREDICTION COMPLETE: {recommendation} ({confidence:.2f} confidence)")
            return result
            
        except Exception as e:
            print(f"❌ PREDICTION ERROR: {e}")
            traceback.print_exc()
            return {
                'error': str(e),
                'predicted_profitable': False,
                'confidence': 0.1,
                'recommendation': 'HOLD'
            }
    
    def _simplified_prediction(self, latest_features: pd.Series) -> Dict[str, Any]:
        """Simplified prediction when full feature set unavailable"""
        try:
            # Use basic indicators
            rsi = latest_features.get('rsi', 50)
            price_change = latest_features.get('price_change', 0)
            
            # Simple rules
            if rsi < 30 and price_change > -0.02:
                predicted_profitable = True
                confidence = 0.4
            elif rsi > 70 and price_change < 0.02:
                predicted_profitable = False
                confidence = 0.4
            else:
                predicted_profitable = price_change > 0
                confidence = 0.3
            
            return {
                'predicted_profitable': predicted_profitable,
                'confidence': confidence,
                'recommendation': 'HOLD',  # Conservative
                'method': 'simplified',
                'direction': 'profitable' if predicted_profitable else 'unprofitable'
            }
            
        except Exception as e:
            return {
                'error': f'Simplified prediction failed: {e}',
                'predicted_profitable': False,
                'confidence': 0.1,
                'recommendation': 'HOLD'
            }
    
    def _perform_reality_check(self, df: pd.DataFrame, confidence: float, predicted_profitable: bool) -> Dict[str, Any]:
        """Perform reality check on predictions"""
        issues = []
        
        try:
            # Check recent market volatility
            if 'price_volatility' in df.columns:
                recent_vol = df['price_volatility'].iloc[-5:].mean()
                if recent_vol > 0.05:  # High volatility
                    issues.append("High market volatility detected")
            
            # Check RSI extremes
            if 'rsi' in df.columns:
                recent_rsi = df['rsi'].iloc[-1]
                if recent_rsi > 80 or recent_rsi < 20:
                    issues.append(f"Extreme RSI detected: {recent_rsi:.1f}")
            
            # Check prediction confidence vs market conditions
            if confidence > 0.8 and len(df) < 200:
                issues.append("High confidence with limited data")
            
            return {
                'applied': True,
                'issues': issues,
                'confidence_adjustment': 0.8 if issues else 1.0
            }
            
        except Exception as e:
            return {
                'applied': False,
                'error': str(e),
                'confidence_adjustment': 0.9
            }
    
    def _save_model_performance_to_db(self, successful_models: Dict[str, ModelPerformance]):
        """Save model performance to database"""
        if not self.db_manager:
            print("⚠️ No database manager - skipping model performance save")
            return
        
        try:
            for name, performance in successful_models.items():
                model_info = {
                    'model_name': name,
                    'model_type': 'classification',
                    'accuracy': performance.accuracy * 100,  # Convert to percentage
                    'r2_score': performance.cross_val_score,
                    'mae': 1.0 - performance.accuracy,  # Simple error metric
                    'training_samples': performance.training_samples,
                    'model_file_path': f'{self.model_dir}/{name}.pkl',
                    'metrics': {
                        'precision': performance.precision,
                        'recall': performance.recall,
                        'f1_score': performance.f1_score,
                        'ensemble_weight': performance.ensemble_weight,
                        'training_time': performance.training_time,
                        'feature_count': performance.feature_count
                    }
                }
                
                self.db_manager.save_ml_model_info(model_info)
                print(f"✅ Saved {name} performance to database")
                
        except Exception as e:
            print(f"⚠️ Error saving model performance to DB: {e}")
    
    def _save_models_to_disk(self):
        """Save trained models to disk"""
        try:
            for name, model in self.model_trainer.models.items():
                model_path = os.path.join(self.model_dir, f'{name}.pkl')
                joblib.dump(model, model_path)
                
                # Save scaler if exists
                if name in self.model_trainer.scalers:
                    scaler_path = os.path.join(self.model_dir, f'{name}_scaler.pkl')
                    joblib.dump(self.model_trainer.scalers[name], scaler_path)
                
            print(f"✅ Models saved to {self.model_dir}")
            
        except Exception as e:
            print(f"⚠️ Error saving models to disk: {e}")
    
    def get_model_performance(self) -> Dict[str, Dict]:
        """Get model performance metrics"""
        try:
            if self.db_manager:
                # Try to get from database first
                return self.db_manager.get_ml_model_performance()
            else:
                # Fallback to in-memory performance
                return {
                    name: {
                        'accuracy': perf.accuracy * 100,
                        'model_type': 'classification',
                        'training_samples': perf.training_samples,
                        'last_trained': self.last_training_time.strftime('%Y-%m-%d %H:%M:%S') if self.last_training_time else 'Never',
                        'r2': perf.cross_val_score,
                        'mae': 1.0 - perf.accuracy,
                        'ensemble_weight': perf.ensemble_weight
                    } for name, perf in self.model_trainer.performance.items()
                }
                
        except Exception as e:
            print(f"⚠️ Error getting model performance: {e}")
            return {}
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get overall performance statistics"""
        success_rate = self.successful_predictions / self.prediction_count if self.prediction_count > 0 else 0
        
        return {
            'total_predictions': self.prediction_count,
            'successful_predictions': self.successful_predictions,
            'success_rate': f"{success_rate:.1%}",
            'models_trained': len(self.model_trainer.models),
            'last_training': self.last_training_time.strftime('%Y-%m-%d %H:%M:%S') if self.last_training_time else 'Never',
            'min_samples': self.min_samples
        }