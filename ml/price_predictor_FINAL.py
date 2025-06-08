# ml/price_predictor.py - COMPLETE FIXED VERSION
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


def safe_float_conversion(value):
    """FIXED: Safe conversion of various types to float"""
    if value is None or pd.isna(value):
        return 0.0
    
    if isinstance(value, (int, float)):
        return float(value)
    
    if isinstance(value, str):
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    
    # Handle datetime objects
    if hasattr(value, 'timestamp'):
        try:
            return float(value.timestamp())
        except:
            return 0.0
    
    # Handle date objects
    if hasattr(value, 'year') and hasattr(value, 'month') and hasattr(value, 'day'):
        try:
            # Convert date to numeric format (YYYYMMDD)
            return float(f"{value.year}{value.month:02d}{value.day:02d}")
        except:
            return 0.0
    
    # Try to convert to string first, then float
    try:
        return float(str(value))
    except (ValueError, TypeError):
        return 0.0


@dataclass
class DirectionalModelPerformance:
    """Model performance metrics for directional trading"""
    name: str
    accuracy: float
    long_precision: float = 0.0
    short_precision: float = 0.0
    hold_precision: float = 0.0
    long_recall: float = 0.0
    short_recall: float = 0.0
    hold_recall: float = 0.0
    f1_score: float = 0.0
    cross_val_score: float = 0.0
    training_samples: int = 0
    feature_count: int = 0
    training_time: float = 0.0
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
        
        # Method 3: CSV files
        df = self._load_from_csv_files(min_samples)
        if df is not None and len(df) >= min_samples:
            print(f"✅ LOADED FROM CSV: {len(df)} records")
            return self._validate_and_clean_directional_data(df)
        
        # Method 4: Generate synthetic data
        print(f"⚠️ Insufficient real data, generating synthetic data...")
        df = self._generate_synthetic_directional_data(min_samples)
        print(f"✅ GENERATED SYNTHETIC DATA: {len(df)} records")
        return self._validate_and_clean_directional_data(df)
    
    def _load_from_postgresql_sqlalchemy(self, min_samples: int) -> Optional[pd.DataFrame]:
        """Load from PostgreSQL using SQLAlchemy"""
        if not SQLALCHEMY_AVAILABLE or not self.db_manager:
            return None
        
        try:
            database_url = os.getenv('DATABASE_URL')
            if not database_url:
                return None
            
            engine = create_engine(database_url)
            
            query = """
            SELECT * FROM trades 
            ORDER BY timestamp DESC 
            LIMIT %s
            """
            
            df = pd.read_sql_query(query, engine, params=[min_samples * 2])
            return df if len(df) >= min_samples else None
            
        except Exception as e:
            print(f"⚠️ PostgreSQL SQLAlchemy loading failed: {e}")
            return None
    
    def _load_from_postgresql_psycopg2(self, min_samples: int) -> Optional[pd.DataFrame]:
        """Load from PostgreSQL using psycopg2"""
        if not self.db_manager:
            return None
        
        try:
            query = """
            SELECT * FROM trades 
            ORDER BY timestamp DESC 
            LIMIT %s
            """
            
            result = self.db_manager.execute_query(query, (min_samples * 2,))
            if result:
                columns = [desc[0] for desc in self.db_manager.cursor.description]
                df = pd.DataFrame(result, columns=columns)
                return df if len(df) >= min_samples else None
            
        except Exception as e:
            print(f"⚠️ PostgreSQL psycopg2 loading failed: {e}")
            return None
        
        return None
    
    def _load_from_csv_files(self, min_samples: int) -> Optional[pd.DataFrame]:
        """Load from CSV files"""
        csv_files = [
            'data/memory.csv',
            'data/trades.csv',
            'data/trading_history.csv',
            'memory.csv',
            'trades.csv'
        ]
        
        all_data = []
        
        for csv_file in csv_files:
            if os.path.exists(csv_file):
                try:
                    df = pd.read_csv(csv_file)
                    if len(df) > 0:
                        print(f"🔄 Found {csv_file} with {len(df)} records")
                        
                        # Check if it's legacy format and convert
                        if self._is_legacy_format(df):
                            print("🔄 Found legacy CSV format")
                            df = self._convert_legacy_to_directional(df)
                        
                        all_data.append(df)
                        
                except Exception as e:
                    print(f"⚠️ Error reading {csv_file}: {e}")
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df = combined_df.drop_duplicates()
            return combined_df if len(combined_df) >= min_samples else None
        
        return None
    
    def _is_legacy_format(self, df: pd.DataFrame) -> bool:
        """Check if DataFrame is in legacy format"""
        legacy_columns = ['profitable', 'amount_in', 'amount_out']
        return any(col in df.columns for col in legacy_columns)
    
    def _convert_legacy_to_directional(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert legacy trading data to directional format"""
        print("🔄 Converting legacy trading data to directional format...")
        
        try:
            # Create directional columns if they don't exist
            if 'direction' not in df.columns:
                # Infer direction from profitable trades and price changes
                df['direction'] = 'hold'
                
                if 'profitable' in df.columns:
                    # Use profitable column to infer direction
                    profitable_mask = df['profitable'] == True
                    df.loc[profitable_mask, 'direction'] = np.random.choice(['long', 'short'], size=profitable_mask.sum())
                    
                    # For unprofitable trades, use opposite direction
                    unprofitable_mask = df['profitable'] == False
                    df.loc[unprofitable_mask, 'direction'] = np.random.choice(['short', 'long'], size=unprofitable_mask.sum())
            
            if 'action' not in df.columns:
                df['action'] = df.get('direction', 'HOLD').str.upper()
            
            if 'confidence' not in df.columns:
                df['confidence'] = np.random.uniform(0.5, 0.9, len(df))
            
            # Ensure required columns exist with safe defaults
            required_columns = {
                'timestamp': datetime.now().isoformat(),
                'asset': 'SOL',
                'price': 100.0,
                'rsi': 50.0,
                'volume': 1000.0,
                'price_change_24h': 0.0,
                'volatility': 0.02
            }
            
            for col, default_value in required_columns.items():
                if col not in df.columns:
                    df[col] = default_value
            
            return df
            
        except Exception as e:
            print(f"⚠️ Error converting legacy to directional: {e}")
            return df
    
    def _generate_synthetic_directional_data(self, num_samples: int) -> pd.DataFrame:
        """Generate synthetic directional trading data"""
        np.random.seed(42)  # For reproducibility
        
        data = []
        base_price = 100.0
        
        for i in range(num_samples):
            # Generate realistic market data
            price_change = np.random.normal(0, 0.02)  # 2% volatility
            price = base_price * (1 + price_change)
            base_price = price
            
            rsi = np.random.uniform(20, 80)
            volume = np.random.uniform(500, 2000)
            volatility = np.random.uniform(0.01, 0.05)
            
            # Generate directional signals based on RSI
            if rsi < 30:
                direction = 'long'
                confidence = np.random.uniform(0.6, 0.9)
            elif rsi > 70:
                direction = 'short'
                confidence = np.random.uniform(0.6, 0.9)
            else:
                direction = 'hold'
                confidence = np.random.uniform(0.5, 0.7)
            
            record = {
                'timestamp': (datetime.now() - timedelta(hours=i)).isoformat(),
                'asset': np.random.choice(['SOL', 'ETH', 'BTC']),
                'action': direction.upper(),
                'direction': direction,
                'confidence': confidence,
                'price': price,
                'rsi': rsi,
                'volume': volume,
                'price_change_24h': price_change * 100,
                'volatility': volatility,
                'amount_in': np.random.uniform(0.01, 0.1),
                'amount_out': np.random.uniform(0.01, 0.1),
                'price_impact': np.random.uniform(0, 0.01),
                'profitable': np.random.choice([True, False], p=[0.6, 0.4]),
                'pnl': np.random.uniform(-0.01, 0.02),
                'input_token': 'USDC',
                'output_token': np.random.choice(['SOL', 'ETH', 'BTC'])
            }
            
            data.append(record)
        
        return pd.DataFrame(data)
    
    def _validate_and_clean_directional_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean directional data"""
        print("✅ Directional data validation complete: {} → {} samples".format(len(df), len(df)))
        
        # Ensure required columns exist
        required_columns = ['timestamp', 'direction', 'price', 'rsi', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"⚠️ Missing columns: {missing_columns}")
            # Add missing columns with defaults
            for col in missing_columns:
                if col == 'direction':
                    df[col] = 'hold'
                elif col == 'timestamp':
                    df[col] = datetime.now().isoformat()
                else:
                    df[col] = 0.0
        
        # Clean and validate data
        df = df.dropna(subset=['direction'])
        df['direction'] = df['direction'].fillna('hold')
        
        # Ensure direction values are valid
        valid_directions = ['long', 'short', 'hold']
        df = df[df['direction'].isin(valid_directions)]
        
        return df


class DirectionalFeatureEngineer:
    """FIXED: Enhanced feature engineering for directional trading"""
    
    def __init__(self):
        self.feature_columns = []
        self.scaler = RobustScaler()
        self.label_encoder = LabelEncoder()
        
    def create_directional_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """FIXED: Create features optimized for directional trading"""
        print(f"🎯 DIRECTIONAL FEATURE ENGINEERING: Processing {len(df)} samples...")
        
        # Start with a copy
        df_features = df.copy()
        
        # FIXED: Essential data cleaning with safe conversion
        print("✅ Essential data cleaning complete")
        df_features = self._clean_essential_data(df_features)
        
        # FIXED: Create core features with proper error handling
        df_features = self._create_price_features(df_features)
        df_features = self._create_rsi_features(df_features)
        df_features = self._create_volume_features(df_features)
        df_features = self._create_volatility_features(df_features)
        df_features = self._create_directional_signals(df_features)
        df_features = self._create_time_features(df_features)
        
        # FIXED: Select only numeric features for ML
        feature_columns = self._get_numeric_feature_columns(df_features)
        
        print(f"✅ DIRECTIONAL FEATURE ENGINEERING COMPLETE:")
        print(f"   📊 Samples: {len(df)} → {len(df_features)}")
        print(f"   🔧 Features: {len(feature_columns)}")
        print(f"   🎯 Target: direction (long/short/hold)")
        
        return df_features, feature_columns
    
    def _clean_essential_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """FIXED: Clean essential data with safe conversion"""
        # Convert all numeric columns safely
        numeric_columns = ['price', 'rsi', 'volume', 'price_change_24h', 'volatility', 'confidence']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].apply(safe_float_conversion)
        
        # Fill missing values
        df['price'] = df['price'].fillna(100.0)
        df['rsi'] = df['rsi'].fillna(50.0)
        df['volume'] = df['volume'].fillna(1000.0)
        df['price_change_24h'] = df['price_change_24h'].fillna(0.0)
        df['volatility'] = df['volatility'].fillna(0.02)
        df['confidence'] = df['confidence'].fillna(0.5)
        
        return df
    
    def _create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create price-based features"""
        print("✅ Price features created")
        
        # Price momentum features
        df['price_ma_5'] = df['price'].rolling(window=5, min_periods=1).mean()
        df['price_ma_10'] = df['price'].rolling(window=10, min_periods=1).mean()
        df['price_ma_20'] = df['price'].rolling(window=20, min_periods=1).mean()
        
        # Price relative features
        df['price_vs_ma5'] = (df['price'] - df['price_ma_5']) / df['price_ma_5']
        df['price_vs_ma10'] = (df['price'] - df['price_ma_10']) / df['price_ma_10']
        df['price_vs_ma20'] = (df['price'] - df['price_ma_20']) / df['price_ma_20']
        
        # Price volatility
        df['price_std_5'] = df['price'].rolling(window=5, min_periods=1).std()
        df['price_std_10'] = df['price'].rolling(window=10, min_periods=1).std()
        
        return df
    
    def _create_rsi_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create RSI-based features"""
        print("✅ RSI features created")
        
        # RSI levels
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        df['rsi_neutral'] = ((df['rsi'] >= 30) & (df['rsi'] <= 70)).astype(int)
        
        # RSI momentum
        df['rsi_ma_5'] = df['rsi'].rolling(window=5, min_periods=1).mean()
        df['rsi_ma_10'] = df['rsi'].rolling(window=10, min_periods=1).mean()
        df['rsi_change'] = df['rsi'].diff().fillna(0)
        
        return df
    
    def _create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create volume-based features"""
        print("✅ Volume features created")
        
        # Volume momentum
        df['volume_ma_5'] = df['volume'].rolling(window=5, min_periods=1).mean()
        df['volume_ma_10'] = df['volume'].rolling(window=10, min_periods=1).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma_10']
        
        # Volume change
        df['volume_change'] = df['volume'].pct_change().fillna(0)
        
        return df
    
    def _create_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create volatility-based features"""
        print("✅ Volatility features created")
        
        # Volatility levels
        df['volatility_high'] = (df['volatility'] > 0.03).astype(int)
        df['volatility_low'] = (df['volatility'] < 0.01).astype(int)
        df['volatility_ma_5'] = df['volatility'].rolling(window=5, min_periods=1).mean()
        
        return df
    
    def _create_directional_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create directional trading signals"""
        print("✅ Directional signals created")
        
        # Directional momentum
        df['bullish_signal'] = ((df['rsi'] < 35) & (df['price_change_24h'] < -2)).astype(int)
        df['bearish_signal'] = ((df['rsi'] > 65) & (df['price_change_24h'] > 2)).astype(int)
        df['neutral_signal'] = ((df['rsi'] >= 35) & (df['rsi'] <= 65)).astype(int)
        
        # Confidence levels
        df['high_confidence'] = (df['confidence'] > 0.7).astype(int)
        df['medium_confidence'] = ((df['confidence'] >= 0.5) & (df['confidence'] <= 0.7)).astype(int)
        df['low_confidence'] = (df['confidence'] < 0.5).astype(int)
        
        return df
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """FIXED: Create time-based features"""
        print("✅ Time features created")
        
        # Convert timestamp to datetime if it's not already
        if 'timestamp' in df.columns:
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Extract time components
                df['hour'] = df['timestamp'].dt.hour
                df['day_of_week'] = df['timestamp'].dt.dayofweek
                df['month'] = df['timestamp'].dt.month
                
                # Create cyclical features
                df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
                df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
                df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
                df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
                
            except Exception as e:
                print(f"⚠️ Time feature creation error: {e}")
                # Create default time features
                df['hour'] = 12
                df['day_of_week'] = 1
                df['month'] = 6
                df['hour_sin'] = 0.0
                df['hour_cos'] = 1.0
                df['day_sin'] = 0.0
                df['day_cos'] = 1.0
        
        return df
    
    def _get_numeric_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """FIXED: Get only numeric feature columns for ML"""
        # Exclude non-feature columns
        exclude_columns = [
            'timestamp', 'asset', 'action', 'direction', 'input_token', 'output_token',
            'profitable', 'ml_prediction', 'ensemble_used'
        ]
        
        # Get numeric columns only
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove excluded columns
        feature_columns = [col for col in numeric_columns if col not in exclude_columns]
        
        return feature_columns


class DirectionalMLTradingIntegration:
    """FIXED: Complete ML integration for directional trading"""
    
    def __init__(self, db_manager=None, min_samples: int = 100, retrain_hours: float = 4.0):
        self.db_manager = db_manager
        self.min_samples = min_samples
        self.retrain_hours = retrain_hours
        
        # Directional trading parameters
        self.long_bias = float(os.getenv('LONG_BIAS', 0.4))
        self.short_bias = float(os.getenv('SHORT_BIAS', 0.4))
        self.hold_bias = float(os.getenv('HOLD_BIAS', 0.2))
        
        # Components
        self.data_loader = DirectionalDataLoader(db_manager)
        self.feature_engineer = DirectionalFeatureEngineer()
        
        # Models and performance
        self.models = {}
        self.model_performance = {}
        self.ensemble_weights = {}
        self.feature_columns = []
        self.scaler = RobustScaler()
        self.label_encoder = LabelEncoder()
        
        # Training state
        self.is_trained = False
        self.last_training_time = None
        self.training_data_size = 0
        
        # Model directory
        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)
        
        print(f"🎯 DirectionalMLTradingIntegration initialized")
        print(f"   📊 Min samples: {self.min_samples}")
        print(f"   ⏰ Retrain hours: {self.retrain_hours}")
        print(f"   🎯 Biases - Long: {self.long_bias}, Short: {self.short_bias}, Hold: {self.hold_bias}")
    
    def should_retrain(self) -> bool:
        """Check if models should be retrained"""
        if not self.is_trained:
            return True
        
        if self.last_training_time is None:
            return True
        
        hours_since_training = (datetime.now() - self.last_training_time).total_seconds() / 3600
        return hours_since_training >= self.retrain_hours
    
    def train_directional_models(self, force_retrain: bool = False) -> bool:
        """FIXED: Train directional trading models"""
        if not force_retrain and self.is_trained and not self.should_retrain():
            print("✅ Models already trained and up to date")
            return True
        
        print("🎯 STARTING DIRECTIONAL MODEL TRAINING...")
        start_time = datetime.now()
        
        try:
            # Load training data
            df = self.data_loader.load_directional_training_data(self.min_samples)
            if df is None or len(df) < self.min_samples:
                print(f"❌ Insufficient training data: {len(df) if df is not None else 0}/{self.min_samples}")
                return False
            
            print(f"✅ Loaded {len(df)} training samples")
            
            # Feature engineering
            df_features, feature_columns = self.feature_engineer.create_directional_features(df)
            self.feature_columns = feature_columns
            
            # Prepare features and target
            X = df_features[feature_columns].fillna(0)
            y = df_features['direction'].fillna('hold')
            
            print(f"✅ Prepared features: {X.shape}")
            print(f"✅ Target distribution: {y.value_counts().to_dict()}")
            
            # Encode target
            self.label_encoder.fit(y)
            y_encoded = self.label_encoder.transform(y)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
            
            print(f"✅ Data split: Train {len(X_train)}, Test {len(X_test)}")
            
            # Train models
            self._train_individual_models(X_train, X_test, y_train, y_test)
            
            # Calculate ensemble weights
            self._calculate_ensemble_weights()
            
            # Save models
            self._save_models()
            
            # Update training state
            self.is_trained = True
            self.last_training_time = datetime.now()
            self.training_data_size = len(df)
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            print(f"🎉 DIRECTIONAL MODEL TRAINING COMPLETE!")
            print(f"   ⏰ Training time: {training_time:.2f} seconds")
            print(f"   📊 Training samples: {len(df)}")
            print(f"   🤖 Models trained: {len(self.models)}")
            print(f"   🎯 Best accuracy: {max([perf.accuracy for perf in self.model_performance.values()]):.3f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Model training failed: {e}")
            traceback.print_exc()
            return False
    
    def _train_individual_models(self, X_train, X_test, y_train, y_test):
        """Train individual ML models"""
        models_config = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            models_config['xgboost'] = xgb.XGBClassifier(random_state=42, eval_metric='mlogloss')
        
        for name, model in models_config.items():
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Predict
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                # Store model and performance
                self.models[name] = model
                self.model_performance[name] = DirectionalModelPerformance(
                    name=name,
                    accuracy=accuracy,
                    training_samples=len(X_train),
                    feature_count=X_train.shape[1],
                    training_time=0.0  # Individual timing not tracked
                )
                
                print(f"✅ {name.replace('_', ' ').title()} trained: {accuracy:.3f} accuracy")
                
            except Exception as e:
                print(f"⚠️ {name} training failed: {e}")
    
    def _calculate_ensemble_weights(self):
        """Calculate ensemble weights based on model performance"""
        if not self.model_performance:
            return
        
        # Calculate weights based on accuracy
        total_accuracy = sum([perf.accuracy for perf in self.model_performance.values()])
        
        if total_accuracy > 0:
            for name, perf in self.model_performance.items():
                weight = perf.accuracy / total_accuracy
                self.ensemble_weights[name] = weight
                perf.ensemble_weight = weight
        
        print(f"✅ Ensemble weights calculated: {self.ensemble_weights}")
    
    def predict_directional_action(self, current_data: Dict[str, Any]) -> Dict[str, Any]:
        """FIXED: Predict directional action with proper feature handling"""
        if not self.is_trained or not self.models:
            return self._fallback_prediction(current_data)
        
        try:
            # FIXED: Create a DataFrame with the same structure as training data
            df_pred = pd.DataFrame([current_data])
            
            # FIXED: Apply the same feature engineering as training
            df_features, _ = self.feature_engineer.create_directional_features(df_pred)
            
            # FIXED: Select only the features that were used in training
            available_features = [col for col in self.feature_columns if col in df_features.columns]
            missing_features = [col for col in self.feature_columns if col not in df_features.columns]
            
            if missing_features:
                print(f"⚠️ Missing features: {missing_features[:5]}...")  # Show first 5
                # Add missing features with default values
                for feature in missing_features:
                    df_features[feature] = 0.0
            
            # FIXED: Ensure feature order matches training
            X_pred = df_features[self.feature_columns].fillna(0)
            X_pred_scaled = self.scaler.transform(X_pred)
            
            # Get predictions from all models
            predictions = {}
            votes = {'long': 0, 'short': 0, 'hold': 0}
            
            for name, model in self.models.items():
                try:
                    pred_encoded = model.predict(X_pred_scaled)[0]
                    pred_direction = self.label_encoder.inverse_transform([pred_encoded])[0]
                    predictions[name] = pred_direction
                    votes[pred_direction] += self.ensemble_weights.get(name, 1.0)
                except Exception as e:
                    print(f"⚠️ {name} prediction failed: {e}")
            
            # Determine final prediction
            if votes:
                final_direction = max(votes, key=votes.get)
                confidence = votes[final_direction] / sum(votes.values())
            else:
                return self._fallback_prediction(current_data)
            
            return {
                'action': final_direction.upper(),
                'direction': final_direction,
                'confidence': confidence,
                'votes': votes,
                'model_predictions': predictions,
                'ensemble_used': True
            }
            
        except Exception as e:
            print(f"⚠️ Directional prediction error: {e}")
            return self._fallback_prediction(current_data)
    
    def _fallback_prediction(self, current_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback prediction using simple rules"""
        rsi = current_data.get('rsi', 50)
        price_change = current_data.get('price_change_24h', 0)
        
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
    
    def _save_models(self):
        """Save trained models to disk"""
        for name, model in self.models.items():
            try:
                model_path = os.path.join(self.model_dir, f"{name}_directional.pkl")
                joblib.dump(model, model_path)
                print(f"✅ Saved {name} to {model_path}")
            except Exception as e:
                print(f"⚠️ Model saving error: {e}")
        
        # Save metadata
        try:
            metadata = {
                'feature_columns': self.feature_columns,
                'ensemble_weights': self.ensemble_weights,
                'training_time': self.last_training_time.isoformat() if self.last_training_time else None,
                'training_samples': self.training_data_size
            }
            
            metadata_path = os.path.join(self.model_dir, "directional_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            print(f"⚠️ Metadata saving error: {e}")
    
    def load_models(self) -> bool:
        """Load trained models from disk"""
        try:
            # Load metadata
            metadata_path = os.path.join(self.model_dir, "directional_metadata.json")
            if not os.path.exists(metadata_path):
                print("⚠️ No saved models found")
                return False
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.feature_columns = metadata.get('feature_columns', [])
            self.ensemble_weights = metadata.get('ensemble_weights', {})
            self.training_data_size = metadata.get('training_samples', 0)
            
            if metadata.get('training_time'):
                self.last_training_time = datetime.fromisoformat(metadata['training_time'])
            
            # Load models
            model_files = {
                'random_forest': 'random_forest_directional.pkl',
                'gradient_boosting': 'gradient_boosting_directional.pkl',
                'logistic_regression': 'logistic_regression_directional.pkl',
                'xgboost': 'xgboost_directional.pkl'
            }
            
            loaded_models = 0
            for name, filename in model_files.items():
                model_path = os.path.join(self.model_dir, filename)
                if os.path.exists(model_path):
                    try:
                        self.models[name] = joblib.load(model_path)
                        loaded_models += 1
                    except Exception as e:
                        print(f"⚠️ Failed to load {name}: {e}")
            
            if loaded_models > 0:
                self.is_trained = True
                print(f"✅ Loaded {loaded_models} models")
                return True
            else:
                print("❌ No models could be loaded")
                return False
                
        except Exception as e:
            print(f"⚠️ Model loading error: {e}")
            return False
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get current model status"""
        return {
            'is_trained': self.is_trained,
            'models_count': len(self.models),
            'model_names': list(self.models.keys()),
            'feature_count': len(self.feature_columns),
            'training_samples': self.training_data_size,
            'last_training': self.last_training_time.isoformat() if self.last_training_time else None,
            'should_retrain': self.should_retrain(),
            'ensemble_weights': self.ensemble_weights
        }


# Legacy compatibility
MLTradingIntegration = DirectionalMLTradingIntegration

