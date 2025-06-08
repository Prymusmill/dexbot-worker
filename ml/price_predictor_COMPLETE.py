# ml/price_predictor.py - COMPLETE ENHANCED DIRECTIONAL ML PREDICTOR
import pandas as pd
import numpy as np
import logging
import os
import json
import pickle
import warnings
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import traceback

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Core ML imports
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
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
    print("⚠️ SQLAlchemy not available")


def safe_float_conversion(value: Any) -> float:
    """Bezpieczna konwersja wartości na float z obsługą dat"""
    if pd.isna(value) or value is None:
        return np.nan
    
    # Obsługa obiektów datetime i date
    if isinstance(value, (datetime, date)):
        epoch = datetime(1970, 1, 1).date() if isinstance(value, date) else datetime(1970, 1, 1)
        if isinstance(value, date) and not isinstance(value, datetime):
            value = datetime.combine(value, datetime.min.time())
        return (value - epoch).total_seconds() / 86400
    
    # Obsługa pandas Timestamp
    if hasattr(value, 'timestamp'):
        try:
            return value.timestamp() / 86400
        except:
            pass
    
    # Obsługa stringów dat
    if isinstance(value, str):
        try:
            parsed_date = pd.to_datetime(value)
            epoch = pd.Timestamp('1970-01-01')
            return (parsed_date - epoch).total_seconds() / 86400
        except:
            try:
                return float(value)
            except:
                return np.nan
    
    # Standardowa konwersja numeryczna
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan


@dataclass
class DirectionalModelPerformance:
    """Model performance metrics for directional trading"""
    name: str
    accuracy: float
    long_precision: float
    short_precision: float
    hold_precision: float
    long_recall: float
    short_recall: float
    hold_recall: float
    f1_score: float
    cross_val_score: float
    training_samples: int
    feature_count: int
    training_time: float
    ensemble_weight: float = 0.0
    directional_accuracy: Dict[str, float] = None


class DirectionalDataLoader:
    """Enhanced data loader for directional trading ML"""
    
    def __init__(self, db_manager=None):
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
        
    def load_directional_training_data(self, min_samples: int = 100) -> Optional[pd.DataFrame]:
        """Load data optimized for directional trading ML"""
        print(f"🎯 DIRECTIONAL DATA LOADING: Attempting to load {min_samples}+ samples...")
        
        # Method 1: PostgreSQL with SQLAlchemy
        df = self._load_from_postgresql_sqlalchemy(min_samples)
        if df is not None and len(df) >= min_samples:
            print(f"✅ LOADED FROM POSTGRESQL (SQLAlchemy): {len(df)} records")
            return self._validate_and_clean_directional_data(df)
            
        # Method 2: PostgreSQL with psycopg2
        df = self._load_from_postgresql_psycopg2(min_samples)
        if df is not None and len(df) >= min_samples:
            print(f"✅ LOADED FROM POSTGRESQL (psycopg2): {len(df)} records")
            return self._validate_and_clean_directional_data(df)
            
        # Method 3: CSV fallback
        df = self._load_from_csv_directional(min_samples)
        if df is not None and len(df) >= min_samples:
            print(f"✅ LOADED FROM CSV: {len(df)} records")
            return self._validate_and_clean_directional_data(df)
            
        print(f"❌ ALL DIRECTIONAL DATA LOADING METHODS FAILED - need {min_samples}+ samples")
        return None
    
    def _validate_and_clean_directional_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean data for directional trading"""
        try:
            original_len = len(df)
            
            # Convert timestamp if it's a string
            if 'timestamp' in df.columns and df['timestamp'].dtype == 'object':
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            
            # Required columns
            required_cols = ['price', 'rsi']
            optional_cols = ['volume', 'volatility', 'price_change_24h']
            
            # Handle required columns
            for col in required_cols:
                if col not in df.columns:
                    print(f"⚠️ Missing required column {col}, creating default...")
                    if col == 'price':
                        df[col] = 100.0
                    elif col == 'rsi':
                        df[col] = 50.0
                    else:
                        df[col] = 0.0
                
                # Ensure numeric type
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Handle optional columns
            for col in optional_cols:
                if col not in df.columns:
                    if col == 'volume':
                        df[col] = 1000.0
                    elif col == 'volatility':
                        df[col] = 0.02
                    elif col == 'price_change_24h':
                        df[col] = 0.0
                else:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Handle directional columns
            if 'action' in df.columns and 'direction' in df.columns:
                print("✅ Found directional trading data format")
                df = self._process_directional_format(df)
            else:
                print("🔄 Converting legacy format to directional")
                df = self._convert_legacy_to_directional(df)
            
            # Clean invalid values
            df = df[df['price'] > 0]
            df = df[(df['rsi'] >= 0) & (df['rsi'] <= 100)]
            
            # Fill remaining NaN values
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = df[col].fillna(df[col].median())
            
            print(f"✅ Directional data validation complete: {original_len} → {len(df)} samples")
            return df
            
        except Exception as e:
            print(f"⚠️ Directional data validation error: {e}")
            return df
    
    def _process_directional_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process data already in directional format"""
        try:
            # Ensure we have the target column for ML
            if 'direction' not in df.columns:
                if 'action' in df.columns:
                    # Map action to direction
                    action_to_direction = {
                        'LONG': 'long',
                        'SHORT': 'short', 
                        'HOLD': 'hold',
                        'CLOSE': 'hold'
                    }
                    df['direction'] = df['action'].map(action_to_direction)
                    df['direction'] = df['direction'].fillna('hold')
                else:
                    # Create direction from profitability
                    df['direction'] = df['profitable'].apply(lambda x: 'long' if x else 'short')
            
            # Ensure direction is string type
            df['direction'] = df['direction'].astype(str)
            
            # Clean up direction values
            direction_mapping = {
                'long': 'long',
                'short': 'short',
                'hold': 'hold',
                'LONG': 'long',
                'SHORT': 'short',
                'HOLD': 'hold',
                'profitable': 'long',
                'unprofitable': 'short'
            }
            
            df['direction'] = df['direction'].map(direction_mapping).fillna('hold')
            
            return df
            
        except Exception as e:
            print(f"⚠️ Error processing directional format: {e}")
            return df
    
    def _convert_legacy_to_directional(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert legacy format to directional trading format"""
        try:
            print("🔄 Converting legacy trading data to directional format...")
            
            # Create directional labels based on profitability and market conditions
            df['direction'] = 'hold'  # Default
            
            if 'profitable' in df.columns:
                # Basic conversion: profitable = long, unprofitable = short
                df.loc[df['profitable'] == True, 'direction'] = 'long'
                df.loc[df['profitable'] == False, 'direction'] = 'short'
            
            # Enhanced: Use RSI and momentum for better directional labeling
            if 'rsi' in df.columns:
                # RSI-based refinement
                df.loc[(df['rsi'] < 30) & (df['profitable'] == True), 'direction'] = 'long'
                df.loc[(df['rsi'] > 70) & (df['profitable'] == True), 'direction'] = 'short'
                
                # Neutral RSI = hold more often
                df.loc[(df['rsi'] >= 40) & (df['rsi'] <= 60), 'direction'] = 'hold'
            
            # Price momentum based refinement
            if 'price_change_24h' in df.columns:
                df.loc[(df['price_change_24h'] < -3) & (df['profitable'] == True), 'direction'] = 'long'
                df.loc[(df['price_change_24h'] > 5) & (df['profitable'] == True), 'direction'] = 'short'
            
            # Balance the dataset
            direction_counts = df['direction'].value_counts()
            print(f"🎯 Direction distribution after conversion: {dict(direction_counts)}")
            
            # If too imbalanced, create some balance
            if len(direction_counts) < 3:
                random_indices = np.random.choice(df.index, size=min(len(df)//4, 50), replace=False)
                df.loc[random_indices, 'direction'] = 'hold'
            
            return df
            
        except Exception as e:
            print(f"⚠️ Error converting legacy to directional: {e}")
            # Fallback: random balanced distribution
            directions = np.random.choice(['long', 'short', 'hold'], size=len(df), p=[0.4, 0.4, 0.2])
            df['direction'] = directions
            return df
    
    def _load_from_postgresql_sqlalchemy(self, min_samples: int) -> Optional[pd.DataFrame]:
        """Load using SQLAlchemy"""
        if not SQLALCHEMY_AVAILABLE or not self.db_manager:
            return None
            
        try:
            database_url = os.getenv('DATABASE_URL')
            if not database_url:
                return None
                
            engine = create_engine(database_url)
            
            query = """
                SELECT timestamp, price, volume, rsi, amount_in, amount_out,
                       price_impact, profitable, input_token, output_token
                FROM transactions
                WHERE price IS NOT NULL AND price > 0 
                  AND rsi IS NOT NULL AND rsi BETWEEN 0 AND 100
                  AND volume IS NOT NULL AND volume > 0
                ORDER BY timestamp DESC
                LIMIT 3000;
            """
            
            df = pd.read_sql_query(query, engine)
            engine.dispose()
            
            return df if len(df) > 0 else None
                
        except Exception as e:
            print(f"⚠️ SQLAlchemy loading failed: {e}")
            return None
    
    def _load_from_postgresql_psycopg2(self, min_samples: int) -> Optional[pd.DataFrame]:
        """Load using direct psycopg2"""
        if not self.db_manager:
            return None
            
        try:
            df = self.db_manager.get_all_transactions_for_ml()
            return df if len(df) > 0 else None
                
        except Exception as e:
            print(f"⚠️ psycopg2 loading failed: {e}")
            return None
    
    def _load_from_csv_directional(self, min_samples: int) -> Optional[pd.DataFrame]:
        """Load from CSV with directional support"""
        csv_path = "data/memory.csv"
        
        if not os.path.exists(csv_path):
            return None
            
        try:
            df = pd.read_csv(csv_path)
            
            if len(df) > 0:
                df = df.tail(min_samples * 2)
                
                if 'direction' in df.columns or 'action' in df.columns:
                    print("✅ Found directional CSV format")
                    return df
                else:
                    print("🔄 Found legacy CSV format")
                    return df
            else:
                return None
                
        except Exception as e:
            print(f"⚠️ CSV loading failed: {e}")
            return None


class DirectionalFeatureEngineer:
    """Enhanced feature engineering for directional trading"""
    
    @staticmethod
    def engineer_directional_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create features optimized for LONG/SHORT/HOLD predictions"""
        
        if len(df) < 5:
            print(f"⚠️ Dataset too small for directional feature engineering: {len(df)} samples")
            return DirectionalFeatureEngineer._create_minimal_directional_features(df)
        
        print(f"🎯 DIRECTIONAL FEATURE ENGINEERING: Processing {len(df)} samples...")
        
        try:
            df = df.copy()
            initial_len = len(df)
            
            # Ensure timestamp column for sorting
            if 'timestamp' in df.columns:
                df = df.sort_values('timestamp')
            
            # Essential data validation
            essential_cols = ['price', 'rsi']
            for col in essential_cols:
                if col not in df.columns:
                    default_val = {'price': 100.0, 'rsi': 50.0}[col]
                    df[col] = default_val
                    print(f"✅ Created missing column {col} with default {default_val}")
            
            # Modern pandas cleaning
            for col in essential_cols:
                if df[col].isna().any():
                    df[col] = df[col].ffill().bfill()
                    if col == 'price':
                        df[col] = df[col].fillna(100.0)
                    elif col == 'rsi':
                        df[col] = df[col].fillna(50.0)
            
            print(f"✅ Essential data cleaning complete")
            
            # Directional price features
            try:
                # Price change and momentum
                df['price_change_1'] = df['price'].pct_change(1).fillna(0)
                df['price_change_3'] = df['price'].pct_change(3).fillna(0)
                df['price_change_5'] = df['price'].pct_change(5).fillna(0)
                
                # Price moving averages
                df['price_ma_3'] = df['price'].rolling(window=3, min_periods=1).mean()
                df['price_ma_5'] = df['price'].rolling(window=5, min_periods=1).mean()
                df['price_ma_10'] = df['price'].rolling(window=min(10, len(df)), min_periods=1).mean()
                
                # Price position relative to moving averages
                df['price_vs_ma3'] = (df['price'] - df['price_ma_3']) / df['price_ma_3']
                df['price_vs_ma5'] = (df['price'] - df['price_ma_5']) / df['price_ma_5']
                df['price_vs_ma10'] = (df['price'] - df['price_ma_10']) / df['price_ma_10']
                
                print(f"✅ Price features created")
                
            except Exception as e:
                print(f"⚠️ Price feature engineering error: {e}")
            
            # RSI-based directional features
            try:
                # RSI momentum
                df['rsi_change_1'] = df['rsi'].diff(1).fillna(0)
                df['rsi_change_3'] = df['rsi'].diff(3).fillna(0)
                
                # RSI moving averages
                df['rsi_ma_3'] = df['rsi'].rolling(window=3, min_periods=1).mean()
                df['rsi_ma_5'] = df['rsi'].rolling(window=5, min_periods=1).mean()
                
                # RSI zones
                df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
                df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
                df['rsi_neutral'] = ((df['rsi'] >= 40) & (df['rsi'] <= 60)).astype(int)
                
                # RSI divergence signals
                df['rsi_vs_ma3'] = df['rsi'] - df['rsi_ma_3']
                df['rsi_vs_ma5'] = df['rsi'] - df['rsi_ma_5']
                
                print(f"✅ RSI features created")
                
            except Exception as e:
                print(f"⚠️ RSI feature engineering error: {e}")
            
            # Volume features (if available)
            if 'volume' in df.columns:
                try:
                    df['volume_ma_3'] = df['volume'].rolling(window=3, min_periods=1).mean()
                    df['volume_ma_5'] = df['volume'].rolling(window=5, min_periods=1).mean()
                    df['volume_ratio_3'] = df['volume'] / df['volume_ma_3']
                    df['volume_ratio_5'] = df['volume'] / df['volume_ma_5']
                    
                    # Volume-price relationship
                    df['volume_price_ratio'] = df['volume'] / df['price']
                    
                    print(f"✅ Volume features created")
                    
                except Exception as e:
                    print(f"⚠️ Volume feature engineering error: {e}")
            
            # Volatility features
            try:
                # Price volatility
                df['price_volatility_3'] = df['price'].rolling(window=3, min_periods=1).std().fillna(0)
                df['price_volatility_5'] = df['price'].rolling(window=5, min_periods=1).std().fillna(0)
                
                # RSI volatility
                df['rsi_volatility_3'] = df['rsi'].rolling(window=3, min_periods=1).std().fillna(0)
                df['rsi_volatility_5'] = df['rsi'].rolling(window=5, min_periods=1).std().fillna(0)
                
                print(f"✅ Volatility features created")
                
            except Exception as e:
                print(f"⚠️ Volatility feature engineering error: {e}")
            
            # Directional signals
            try:
                # Long signals
                df['long_signal_rsi'] = ((df['rsi'] < 35) & (df['rsi_change_1'] > 0)).astype(int)
                df['long_signal_price'] = ((df['price_change_1'] < -0.02) & (df['price_vs_ma5'] < -0.01)).astype(int)
                
                # Short signals
                df['short_signal_rsi'] = ((df['rsi'] > 65) & (df['rsi_change_1'] < 0)).astype(int)
                df['short_signal_price'] = ((df['price_change_1'] > 0.02) & (df['price_vs_ma5'] > 0.01)).astype(int)
                
                # Hold signals
                df['hold_signal_rsi'] = df['rsi_neutral']
                df['hold_signal_price'] = (abs(df['price_change_1']) < 0.005).astype(int)
                
                # Combined signals
                df['long_signal_combined'] = df['long_signal_rsi'] + df['long_signal_price']
                df['short_signal_combined'] = df['short_signal_rsi'] + df['short_signal_price']
                df['hold_signal_combined'] = df['hold_signal_rsi'] + df['hold_signal_price']
                
                print(f"✅ Directional signals created")
                
            except Exception as e:
                print(f"⚠️ Directional signal engineering error: {e}")
            
            # Time-based features (if timestamp available)
            if 'timestamp' in df.columns:
                try:
                    df['hour'] = df['timestamp'].dt.hour.fillna(12)
                    df['day_of_week'] = df['timestamp'].dt.dayofweek.fillna(1)
                    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
                    
                    # Cyclical encoding
                    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
                    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
                    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
                    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
                    
                    print(f"✅ Time features created")
                    
                except Exception as e:
                    print(f"⚠️ Time feature engineering error: {e}")
            
            # Clean up infinite values
            df = df.replace([np.inf, -np.inf], np.nan)
            
            # Fill any remaining NaN values
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col != 'direction':  # Don't fill target column
                    df[col] = df[col].fillna(df[col].median())
            
            final_len = len(df)
            feature_count = len([col for col in df.columns if col != 'direction'])
            
            print(f"✅ DIRECTIONAL FEATURE ENGINEERING COMPLETE:")
            print(f"   📊 Samples: {initial_len} → {final_len}")
            print(f"   🔧 Features: {feature_count}")
            print(f"   🎯 Target: direction (long/short/hold)")
            
            return df
            
        except Exception as e:
            print(f"❌ DIRECTIONAL FEATURE ENGINEERING FAILED: {e}")
            traceback.print_exc()
            return DirectionalFeatureEngineer._create_minimal_directional_features(df)
    
    @staticmethod
    def _create_minimal_directional_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create minimal features for very small datasets"""
        print("🔧 Creating minimal directional features for small dataset...")
        
        try:
            df = df.copy()
            
            # Ensure basic columns exist
            if 'price' not in df.columns:
                df['price'] = 100.0
            if 'rsi' not in df.columns:
                df['rsi'] = 50.0
            if 'direction' not in df.columns:
                df['direction'] = 'hold'
            
            # Basic features only
            df['price_normalized'] = (df['price'] - df['price'].mean()) / (df['price'].std() + 1e-8)
            df['rsi_normalized'] = (df['rsi'] - 50) / 50
            
            # Simple signals
            df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
            df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
            
            # Fill any NaN values
            df = df.fillna(0)
            
            print(f"✅ Minimal features created for {len(df)} samples")
            return df
            
        except Exception as e:
            print(f"❌ Minimal feature creation failed: {e}")
            return df


class DirectionalMLTradingIntegration:
    """Complete Enhanced Directional ML Trading Integration"""
    
    def __init__(self, db_manager=None):
        self.db_manager = db_manager
        self.models = {}
        self.ensemble_weights = {}
        self.feature_columns = []
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.is_trained = False
        self.last_training_time = None
        self.training_data_size = 0
        self.model_performance = {}
        self.data_loader = DirectionalDataLoader(db_manager)
        
        # Configuration
        self.min_samples_for_training = int(os.getenv('ML_MIN_SAMPLES', 100))
        self.retrain_hours = float(os.getenv('ML_RETRAIN_HOURS', 4.0))
        self.confidence_threshold = float(os.getenv('DIRECTIONAL_CONFIDENCE_THRESHOLD', 0.6))
        
        # Directional biases
        self.long_bias = float(os.getenv('LONG_BIAS', 0.4))
        self.short_bias = float(os.getenv('SHORT_BIAS', 0.4))
        self.hold_bias = float(os.getenv('HOLD_BIAS', 0.2))
        
        print(f"🎯 DirectionalMLTradingIntegration initialized")
        print(f"   📊 Min samples: {self.min_samples_for_training}")
        print(f"   ⏰ Retrain hours: {self.retrain_hours}")
        print(f"   🎯 Biases - Long: {self.long_bias}, Short: {self.short_bias}, Hold: {self.hold_bias}")
    
    def should_retrain(self) -> bool:
        """Check if model should be retrained"""
        if not self.is_trained:
            return True
        
        if self.last_training_time is None:
            return True
        
        hours_since_training = (datetime.now() - self.last_training_time).total_seconds() / 3600
        return hours_since_training >= self.retrain_hours
    
    def train_directional_models(self, force_retrain: bool = False) -> bool:
        """Train directional trading models"""
        if not force_retrain and not self.should_retrain():
            print(f"⏰ Model training not needed (last trained {self.last_training_time})")
            return True
        
        print(f"🎯 STARTING DIRECTIONAL MODEL TRAINING...")
        start_time = datetime.now()
        
        try:
            # Load training data
            df = self.data_loader.load_directional_training_data(self.min_samples_for_training)
            
            if df is None or len(df) < self.min_samples_for_training:
                print(f"❌ Insufficient training data: {len(df) if df is not None else 0} < {self.min_samples_for_training}")
                return False
            
            print(f"✅ Loaded {len(df)} training samples")
            
            # Feature engineering
            df_features = DirectionalFeatureEngineer.engineer_directional_features(df)
            
            if len(df_features) < 10:
                print(f"❌ Too few samples after feature engineering: {len(df_features)}")
                return False
            
            # Prepare features and target
            target_col = 'direction'
            if target_col not in df_features.columns:
                print(f"❌ Target column '{target_col}' not found")
                return False
            
            # Separate features and target
            feature_cols = [col for col in df_features.columns if col != target_col]
            X = df_features[feature_cols].copy()
            y = df_features[target_col].copy()
            
            # Handle datetime columns in features
            for col in X.columns:
                if X[col].dtype == 'object':
                    sample_val = X[col].dropna().iloc[0] if not X[col].dropna().empty else None
                    if isinstance(sample_val, (datetime, date)):
                        X[col] = X[col].apply(safe_float_conversion)
                    else:
                        # Try to convert to numeric
                        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
            
            # Ensure all features are numeric
            X = X.select_dtypes(include=[np.number])
            
            # Fill any remaining NaN values
            X = X.fillna(0)
            
            # Check for infinite values
            X = X.replace([np.inf, -np.inf], 0)
            
            print(f"✅ Prepared features: {X.shape}")
            print(f"✅ Target distribution: {y.value_counts().to_dict()}")
            
            # Encode target labels
            y_encoded = self.label_encoder.fit_transform(y)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Store feature columns
            self.feature_columns = X.columns.tolist()
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
            
            print(f"✅ Data split: Train {X_train.shape[0]}, Test {X_test.shape[0]}")
            
            # Train multiple models
            self.models = {}
            self.model_performance = {}
            
            # Random Forest
            if SKLEARN_AVAILABLE:
                try:
                    rf_model = RandomForestClassifier(
                        n_estimators=100,
                        max_depth=10,
                        random_state=42,
                        n_jobs=-1
                    )
                    rf_model.fit(X_train, y_train)
                    
                    # Evaluate
                    y_pred = rf_model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    self.models['random_forest'] = rf_model
                    self.model_performance['random_forest'] = {
                        'accuracy': accuracy,
                        'model': rf_model
                    }
                    
                    print(f"✅ Random Forest trained: {accuracy:.3f} accuracy")
                    
                except Exception as e:
                    print(f"⚠️ Random Forest training failed: {e}")
            
            # Gradient Boosting
            if SKLEARN_AVAILABLE:
                try:
                    gb_model = GradientBoostingClassifier(
                        n_estimators=100,
                        max_depth=6,
                        random_state=42
                    )
                    gb_model.fit(X_train, y_train)
                    
                    # Evaluate
                    y_pred = gb_model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    self.models['gradient_boosting'] = gb_model
                    self.model_performance['gradient_boosting'] = {
                        'accuracy': accuracy,
                        'model': gb_model
                    }
                    
                    print(f"✅ Gradient Boosting trained: {accuracy:.3f} accuracy")
                    
                except Exception as e:
                    print(f"⚠️ Gradient Boosting training failed: {e}")
            
            # Logistic Regression
            if SKLEARN_AVAILABLE:
                try:
                    lr_model = LogisticRegression(
                        random_state=42,
                        max_iter=1000
                    )
                    lr_model.fit(X_train, y_train)
                    
                    # Evaluate
                    y_pred = lr_model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    self.models['logistic_regression'] = lr_model
                    self.model_performance['logistic_regression'] = {
                        'accuracy': accuracy,
                        'model': lr_model
                    }
                    
                    print(f"✅ Logistic Regression trained: {accuracy:.3f} accuracy")
                    
                except Exception as e:
                    print(f"⚠️ Logistic Regression training failed: {e}")
            
            # XGBoost (if available)
            if XGBOOST_AVAILABLE:
                try:
                    xgb_model = xgb.XGBClassifier(
                        n_estimators=100,
                        max_depth=6,
                        random_state=42
                    )
                    xgb_model.fit(X_train, y_train)
                    
                    # Evaluate
                    y_pred = xgb_model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    self.models['xgboost'] = xgb_model
                    self.model_performance['xgboost'] = {
                        'accuracy': accuracy,
                        'model': xgb_model
                    }
                    
                    print(f"✅ XGBoost trained: {accuracy:.3f} accuracy")
                    
                except Exception as e:
                    print(f"⚠️ XGBoost training failed: {e}")
            
            # Calculate ensemble weights
            self._calculate_ensemble_weights()
            
            # Update training status
            self.is_trained = True
            self.last_training_time = datetime.now()
            self.training_data_size = len(df_features)
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            print(f"🎉 DIRECTIONAL MODEL TRAINING COMPLETE!")
            print(f"   ⏰ Training time: {training_time:.2f} seconds")
            print(f"   📊 Training samples: {self.training_data_size}")
            print(f"   🤖 Models trained: {len(self.models)}")
            print(f"   🎯 Best accuracy: {max([perf['accuracy'] for perf in self.model_performance.values()]):.3f}")
            
            # Save models
            self._save_models()
            
            return True
            
        except Exception as e:
            print(f"❌ DIRECTIONAL MODEL TRAINING FAILED: {e}")
            traceback.print_exc()
            return False
    
    def _calculate_ensemble_weights(self):
        """Calculate ensemble weights based on model performance"""
        if not self.model_performance:
            return
        
        # Calculate weights based on accuracy
        total_accuracy = sum([perf['accuracy'] for perf in self.model_performance.values()])
        
        self.ensemble_weights = {}
        for model_name, perf in self.model_performance.items():
            if total_accuracy > 0:
                self.ensemble_weights[model_name] = perf['accuracy'] / total_accuracy
            else:
                self.ensemble_weights[model_name] = 1.0 / len(self.model_performance)
        
        print(f"✅ Ensemble weights calculated: {self.ensemble_weights}")
    
    def predict_directional_action(self, current_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict directional trading action"""
        if not self.is_trained or not self.models:
            print("⚠️ Models not trained, using fallback prediction")
            return self._fallback_directional_prediction(current_data)
        
        try:
            # Prepare features
            features_df = pd.DataFrame([current_data])
            
            # Add minimal feature engineering
            if 'price' in features_df.columns and 'rsi' in features_df.columns:
                features_df['price_normalized'] = (features_df['price'] - 100) / 100
                features_df['rsi_normalized'] = (features_df['rsi'] - 50) / 50
                features_df['rsi_oversold'] = (features_df['rsi'] < 30).astype(int)
                features_df['rsi_overbought'] = (features_df['rsi'] > 70).astype(int)
            
            # Select only available features
            available_features = [col for col in self.feature_columns if col in features_df.columns]
            
            if not available_features:
                print("⚠️ No matching features found, using fallback")
                return self._fallback_directional_prediction(current_data)
            
            # Prepare feature vector
            X = features_df[available_features].fillna(0)
            
            # Handle datetime columns
            for col in X.columns:
                if X[col].dtype == 'object':
                    X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Get predictions from all models
            predictions = {}
            probabilities = {}
            
            for model_name, model in self.models.items():
                try:
                    pred = model.predict(X_scaled)[0]
                    prob = model.predict_proba(X_scaled)[0]
                    
                    # Convert back to direction labels
                    direction = self.label_encoder.inverse_transform([pred])[0]
                    
                    predictions[model_name] = direction
                    probabilities[model_name] = prob
                    
                except Exception as e:
                    print(f"⚠️ Prediction failed for {model_name}: {e}")
                    continue
            
            if not predictions:
                print("⚠️ All model predictions failed, using fallback")
                return self._fallback_directional_prediction(current_data)
            
            # Ensemble prediction
            ensemble_prediction = self._ensemble_predict(predictions, probabilities)
            
            return ensemble_prediction
            
        except Exception as e:
            print(f"⚠️ Directional prediction error: {e}")
            return self._fallback_directional_prediction(current_data)
    
    def _ensemble_predict(self, predictions: Dict[str, str], probabilities: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Combine predictions from multiple models"""
        try:
            # Count votes for each direction
            direction_votes = {'long': 0, 'short': 0, 'hold': 0}
            direction_confidence = {'long': 0, 'short': 0, 'hold': 0}
            
            for model_name, direction in predictions.items():
                weight = self.ensemble_weights.get(model_name, 1.0)
                direction_votes[direction] += weight
                
                # Add probability-weighted confidence
                if model_name in probabilities:
                    prob_array = probabilities[model_name]
                    if len(prob_array) >= 3:  # Ensure we have probabilities for all classes
                        direction_confidence['long'] += prob_array[0] * weight
                        direction_confidence['short'] += prob_array[1] * weight
                        direction_confidence['hold'] += prob_array[2] * weight
            
            # Apply directional biases
            direction_votes['long'] *= (1 + self.long_bias)
            direction_votes['short'] *= (1 + self.short_bias)
            direction_votes['hold'] *= (1 + self.hold_bias)
            
            # Get final prediction
            final_direction = max(direction_votes, key=direction_votes.get)
            max_confidence = max(direction_confidence.values())
            
            # Check confidence threshold
            if max_confidence < self.confidence_threshold:
                final_direction = 'hold'  # Default to hold if not confident
            
            return {
                'action': final_direction.upper(),
                'direction': final_direction,
                'confidence': max_confidence,
                'votes': direction_votes,
                'model_predictions': predictions,
                'ensemble_used': True
            }
            
        except Exception as e:
            print(f"⚠️ Ensemble prediction error: {e}")
            return {
                'action': 'HOLD',
                'direction': 'hold',
                'confidence': 0.5,
                'votes': {'long': 0, 'short': 0, 'hold': 1},
                'model_predictions': {},
                'ensemble_used': False
            }
    
    def _fallback_directional_prediction(self, current_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback prediction when ML models are not available"""
        try:
            rsi = current_data.get('rsi', 50)
            price_change = current_data.get('price_change_24h', 0)
            
            # Simple rule-based prediction
            if rsi < 30 and price_change < -2:
                direction = 'long'
                confidence = 0.7
            elif rsi > 70 and price_change > 2:
                direction = 'short'
                confidence = 0.7
            else:
                direction = 'hold'
                confidence = 0.6
            
            return {
                'action': direction.upper(),
                'direction': direction,
                'confidence': confidence,
                'votes': {direction: 1},
                'model_predictions': {'fallback': direction},
                'ensemble_used': False
            }
            
        except Exception as e:
            print(f"⚠️ Fallback prediction error: {e}")
            return {
                'action': 'HOLD',
                'direction': 'hold',
                'confidence': 0.5,
                'votes': {'hold': 1},
                'model_predictions': {},
                'ensemble_used': False
            }
    
    def _save_models(self):
        """Save trained models to disk"""
        try:
            os.makedirs('models', exist_ok=True)
            
            # Save each model
            for model_name, model in self.models.items():
                model_path = f'models/{model_name}_directional.pkl'
                joblib.dump(model, model_path)
                print(f"✅ Saved {model_name} to {model_path}")
            
            # Save metadata
            metadata = {
                'feature_columns': self.feature_columns,
                'label_encoder_classes': self.label_encoder.classes_.tolist(),
                'ensemble_weights': self.ensemble_weights,
                'training_time': self.last_training_time.isoformat() if self.last_training_time else None,
                'training_data_size': self.training_data_size,
                'model_performance': self.model_performance
            }
            
            with open('models/directional_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Save scaler
            joblib.dump(self.scaler, 'models/directional_scaler.pkl')
            
            print(f"✅ Model metadata and scaler saved")
            
        except Exception as e:
            print(f"⚠️ Model saving error: {e}")
    
    def load_models(self) -> bool:
        """Load trained models from disk"""
        try:
            if not os.path.exists('models/directional_metadata.json'):
                print("⚠️ No saved models found")
                return False
            
            # Load metadata
            with open('models/directional_metadata.json', 'r') as f:
                metadata = json.load(f)
            
            self.feature_columns = metadata.get('feature_columns', [])
            self.ensemble_weights = metadata.get('ensemble_weights', {})
            self.training_data_size = metadata.get('training_data_size', 0)
            self.model_performance = metadata.get('model_performance', {})
            
            if metadata.get('training_time'):
                self.last_training_time = datetime.fromisoformat(metadata['training_time'])
            
            # Load label encoder
            label_classes = metadata.get('label_encoder_classes', ['hold', 'long', 'short'])
            self.label_encoder.classes_ = np.array(label_classes)
            
            # Load scaler
            if os.path.exists('models/directional_scaler.pkl'):
                self.scaler = joblib.load('models/directional_scaler.pkl')
            
            # Load models
            self.models = {}
            model_files = [f for f in os.listdir('models') if f.endswith('_directional.pkl')]
            
            for model_file in model_files:
                model_name = model_file.replace('_directional.pkl', '')
                model_path = f'models/{model_file}'
                
                try:
                    model = joblib.load(model_path)
                    self.models[model_name] = model
                    print(f"✅ Loaded {model_name} from {model_path}")
                except Exception as e:
                    print(f"⚠️ Failed to load {model_name}: {e}")
            
            if self.models:
                self.is_trained = True
                print(f"✅ Loaded {len(self.models)} directional models")
                return True
            else:
                print("⚠️ No models could be loaded")
                return False
                
        except Exception as e:
            print(f"⚠️ Model loading error: {e}")
            return False
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get current model status"""
        return {
            'is_trained': self.is_trained,
            'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
            'training_data_size': self.training_data_size,
            'models_count': len(self.models),
            'model_names': list(self.models.keys()),
            'feature_count': len(self.feature_columns),
            'ensemble_weights': self.ensemble_weights,
            'model_performance': self.model_performance,
            'should_retrain': self.should_retrain()
        }


# Export the main class
__all__ = ["DirectionalMLTradingIntegration"]

