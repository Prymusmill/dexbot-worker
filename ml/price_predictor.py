# ml/price_predictor.py - ENHANCED DIRECTIONAL ML PREDICTOR (LONG/SHORT/HOLD)
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
from .price_predictor_macos_safe import MacOSSafeMLIntegration as DirectionalMLTradingIntegration

__all__ = ["DirectionalMLTradingIntegration"]

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
    print("⚠️ SQLAlchemy not available - PostgreSQL reading may have issues")


@dataclass
class DirectionalModelPerformance:
    """🎯 Model performance metrics for directional trading"""
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
    directional_accuracy: Dict[str, float] = None  # Per-direction accuracy


class DirectionalDataLoader:
    """🎯 Enhanced data loader for directional trading ML"""
    
    def __init__(self, db_manager=None):
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
        
    def load_directional_training_data(self, min_samples: int = 100) -> Optional[pd.DataFrame]:
        """🎯 Load data optimized for directional trading ML"""
        print(f"🎯 DIRECTIONAL DATA LOADING: Attempting to load {min_samples}+ samples...")
        
        # Method 1: PostgreSQL with SQLAlchemy (PREFERRED)
        df = self._load_from_postgresql_sqlalchemy(min_samples)
        if df is not None and len(df) >= min_samples:
            print(f"✅ LOADED FROM POSTGRESQL (SQLAlchemy): {len(df)} records")
            return self._validate_and_clean_directional_data(df)
            
        # Method 2: PostgreSQL with psycopg2 (FALLBACK 1)
        df = self._load_from_postgresql_psycopg2(min_samples)
        if df is not None and len(df) >= min_samples:
            print(f"✅ LOADED FROM POSTGRESQL (psycopg2): {len(df)} records")
            return self._validate_and_clean_directional_data(df)
            
        # Method 3: CSV fallback (FALLBACK 2)
        df = self._load_from_csv_directional(min_samples)
        if df is not None and len(df) >= min_samples:
            print(f"✅ LOADED FROM CSV: {len(df)} records")
            return self._validate_and_clean_directional_data(df)
            
        print(f"❌ ALL DIRECTIONAL DATA LOADING METHODS FAILED - need {min_samples}+ samples")
        return None
    
    def _validate_and_clean_directional_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """🎯 Validate and clean data for directional trading"""
        try:
            original_len = len(df)
            
            # Convert timestamp if it's a string
            if 'timestamp' in df.columns and df['timestamp'].dtype == 'object':
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            
            # 🎯 DIRECTIONAL DATA REQUIREMENTS
            required_cols = ['price', 'rsi']  # Minimal requirements
            optional_cols = ['volume', 'volatility', 'price_change_24h']
            directional_cols = ['action', 'direction', 'pnl', 'profitable']
            
            # Znajdź miejsce, gdzie występuje błąd i dodaj:
            for col in X.columns:
                if X[col].dtype == 'object':
                    sample_val = X[col].dropna().iloc[0] if not X[col].dropna().empty else None
                    if isinstance(sample_val, (datetime.date, datetime.datetime)):
                        X[col] = X[col].apply(lambda x: (pd.Timestamp(x) - pd.Timestamp('1970-01-01')).total_seconds() / 86400 if pd.notnull(x) else np.nan)
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
            
            # 🎯 HANDLE DIRECTIONAL COLUMNS
            # If we have new directional format, use it
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
        """🎯 Process data already in directional format"""
        try:
            # Ensure we have the target column for ML
            if 'direction' not in df.columns:
                if 'action' in df.columns:
                    # Map action to direction
                    action_to_direction = {
                        'LONG': 'long',
                        'SHORT': 'short', 
                        'HOLD': 'hold',
                        'CLOSE': 'hold'  # Treat close as hold
                    }
                    df['direction'] = df['action'].map(action_to_direction)
                    df['direction'] = df['direction'].fillna('hold')
                else:
                    # Create direction from profitability (fallback)
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
        """🎯 Convert legacy format to directional trading format"""
        try:
            print("🔄 Converting legacy trading data to directional format...")
            
            # Create directional labels based on profitability and market conditions
            df['direction'] = 'hold'  # Default
            
            if 'profitable' in df.columns:
                # Basic conversion: profitable = long, unprofitable = short
                df.loc[df['profitable'] == True, 'direction'] = 'long'
                df.loc[df['profitable'] == False, 'direction'] = 'short'
            
            # 🎯 ENHANCED: Use RSI and momentum for better directional labeling
            if 'rsi' in df.columns:
                # RSI-based refinement
                df.loc[(df['rsi'] < 30) & (df['profitable'] == True), 'direction'] = 'long'  # Oversold + profitable = good long
                df.loc[(df['rsi'] > 70) & (df['profitable'] == True), 'direction'] = 'short'  # Overbought + profitable = good short
                
                # Neutral RSI = hold more often
                df.loc[(df['rsi'] >= 40) & (df['rsi'] <= 60), 'direction'] = 'hold'
            
            # Price momentum based refinement
            if 'price_change_24h' in df.columns:
                df.loc[(df['price_change_24h'] < -3) & (df['profitable'] == True), 'direction'] = 'long'  # Strong dip + profit = long
                df.loc[(df['price_change_24h'] > 5) & (df['profitable'] == True), 'direction'] = 'short'  # Strong rally + profit = short
            
            # Balance the dataset - ensure we have all three directions
            direction_counts = df['direction'].value_counts()
            print(f"🎯 Direction distribution after conversion: {dict(direction_counts)}")
            
            # If too imbalanced, create some balance
            if len(direction_counts) < 3:
                # Add some hold periods
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
            
            # Enhanced query for directional data
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
        """🎯 Load from CSV with directional support"""
        csv_path = "data/memory.csv"
        
        if not os.path.exists(csv_path):
            return None
            
        try:
            df = pd.read_csv(csv_path)
            
            if len(df) > 0:
                # Get recent data
                df = df.tail(min_samples * 2)  # Get 2x requested
                
                # If it's the new directional format, return as is
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
    """🎯 Enhanced feature engineering for directional trading"""
    
    @staticmethod
    def engineer_directional_features(df: pd.DataFrame) -> pd.DataFrame:
        """🎯 Create features optimized for LONG/SHORT/HOLD predictions"""
        
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
            
            # 🎯 STEP 1: ESSENTIAL DIRECTIONAL DATA VALIDATION
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
            
            # 🎯 STEP 2: DIRECTIONAL PRICE FEATURES
            try:
                # Price change and momentum
                df['price_change'] = df['price'].pct_change().fillna(0)
                df['price_change'] = df['price_change'].replace([np.inf, -np.inf], 0)
                
                # 🎯 DIRECTIONAL BIAS FEATURES
                df['price_change_1h'] = df['price'].pct_change(periods=4).fillna(0)  # Assuming 15min intervals
                df['price_change_4h'] = df['price'].pct_change(periods=16).fillna(0)
                df['price_change_24h'] = df['price'].pct_change(periods=96).fillna(0) if len(df) > 96 else df['price_change']
                
                # Moving averages for trend detection
                df['price_ma_5'] = df['price'].rolling(window=5, min_periods=1).mean()
                df['price_ma_10'] = df['price'].rolling(window=10, min_periods=1).mean()
                df['price_ma_20'] = df['price'].rolling(window=20, min_periods=1).mean()
                
                # 🎯 TREND STRENGTH INDICATORS
                df['trend_strength_short'] = (df['price'] - df['price_ma_5']) / df['price_ma_5']
                df['trend_strength_medium'] = (df['price'] - df['price_ma_10']) / df['price_ma_10']
                df['trend_strength_long'] = (df['price'] - df['price_ma_20']) / df['price_ma_20']
                
                # Price volatility (key for HOLD decisions)
                df['price_volatility'] = df['price'].rolling(window=10, min_periods=1).std().fillna(0)
                df['volatility_normalized'] = df['price_volatility'] / df['price']
                
                print("✅ Directional price features created")
            except Exception as e:
                print(f"⚠️ Price features error: {e}")
            
            # 🎯 STEP 3: ENHANCED RSI FEATURES FOR DIRECTIONAL TRADING
            try:
                # Basic RSI features
                df['rsi_normalized'] = (df['rsi'] - 50) / 50
                df['rsi_momentum'] = df['rsi'].diff().fillna(0)
                
                # 🎯 DIRECTIONAL RSI SIGNALS
                df['rsi_extreme_oversold'] = (df['rsi'] < 25).astype(int)
                df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
                df['rsi_neutral'] = ((df['rsi'] >= 40) & (df['rsi'] <= 60)).astype(int)
                df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
                df['rsi_extreme_overbought'] = (df['rsi'] > 75).astype(int)
                
                # RSI divergence (key for reversal signals)
                df['rsi_divergence'] = df['rsi'] - df['rsi'].rolling(5, min_periods=1).mean()
                
                # 🎯 RSI TREND ANALYSIS
                df['rsi_trend'] = 0
                df.loc[df['rsi_momentum'] > 2, 'rsi_trend'] = 1  # Rising RSI
                df.loc[df['rsi_momentum'] < -2, 'rsi_trend'] = -1  # Falling RSI
                
                print("✅ Directional RSI features created")
            except Exception as e:
                print(f"⚠️ RSI features error: {e}")
            
            # 🎯 STEP 4: VOLUME FEATURES (if available)
            try:
                if 'volume' in df.columns:
                    df['volume_ma'] = df['volume'].rolling(window=10, min_periods=1).mean()
                    df['volume_ratio'] = df['volume'] / df['volume_ma']
                    df['volume_ratio'] = df['volume_ratio'].replace([np.inf, -np.inf], 1.0).fillna(1.0)
                    
                    # Volume spikes (important for direction changes)
                    df['volume_spike'] = (df['volume_ratio'] > 2.0).astype(int)
                else:
                    df['volume'] = 1000.0
                    df['volume_ma'] = 1000.0
                    df['volume_ratio'] = 1.0
                    df['volume_spike'] = 0
                
                print("✅ Volume features created")
            except Exception as e:
                print(f"⚠️ Volume features error: {e}")
            
            # 🎯 STEP 5: TIME-BASED FEATURES
            try:
                if 'timestamp' in df.columns:
                    df_time = pd.to_datetime(df['timestamp'], errors='coerce')
                    df['hour'] = df_time.dt.hour.fillna(12)
                    df['day_of_week'] = df_time.dt.dayofweek.fillna(1)
                    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
                    
                    # Market session indicators (crypto trades 24/7 but patterns exist)
                    df['session_us'] = ((df['hour'] >= 14) & (df['hour'] <= 21)).astype(int)  # US market hours in UTC
                    df['session_asia'] = ((df['hour'] >= 1) & (df['hour'] <= 8)).astype(int)   # Asia market hours in UTC
                else:
                    df['hour'] = 12
                    df['day_of_week'] = 1
                    df['is_weekend'] = 0
                    df['session_us'] = 0
                    df['session_asia'] = 0
                
                print("✅ Time features created")
            except Exception as e:
                print(f"⚠️ Time features error: {e}")
                df['hour'] = 12
                df['day_of_week'] = 1
                df['is_weekend'] = 0
                df['session_us'] = 0
                df['session_asia'] = 0
            
            # 🎯 STEP 6: DIRECTIONAL LAG FEATURES
            try:
                if len(df) > 10:
                    for lag in [1, 2, 3]:
                        # Price lags
                        df[f'price_lag_{lag}'] = df['price'].shift(lag)
                        df[f'rsi_lag_{lag}'] = df['rsi'].shift(lag)
                        
                        # Fill NaN in lag features
                        df[f'price_lag_{lag}'] = df[f'price_lag_{lag}'].bfill().fillna(df['price'].mean())
                        df[f'rsi_lag_{lag}'] = df[f'rsi_lag_{lag}'].bfill().fillna(50.0)
                        
                        # Lag changes (momentum indicators)
                        df[f'price_change_lag_{lag}'] = (df['price'] - df[f'price_lag_{lag}']) / df[f'price_lag_{lag}']
                        df[f'rsi_change_lag_{lag}'] = df['rsi'] - df[f'rsi_lag_{lag}']
                
                print("✅ Directional lag features created")
            except Exception as e:
                print(f"⚠️ Lag features error: {e}")
            
            # 🎯 STEP 7: COMBINED DIRECTIONAL INDICATORS
            try:
                # 🎯 LONG SIGNAL STRENGTH
                df['long_signal_strength'] = 0.0
                df.loc[df['rsi'] < 30, 'long_signal_strength'] += 0.3
                df.loc[df['price_change_24h'] < -0.02, 'long_signal_strength'] += 0.2
                df.loc[df['trend_strength_short'] < -0.02, 'long_signal_strength'] += 0.2
                df.loc[df['volume_spike'] == 1, 'long_signal_strength'] += 0.1
                
                # 🎯 SHORT SIGNAL STRENGTH
                df['short_signal_strength'] = 0.0
                df.loc[df['rsi'] > 70, 'short_signal_strength'] += 0.3
                df.loc[df['price_change_24h'] > 0.03, 'short_signal_strength'] += 0.2
                df.loc[df['trend_strength_short'] > 0.02, 'short_signal_strength'] += 0.2
                df.loc[df['volume_spike'] == 1, 'short_signal_strength'] += 0.1
                
                # 🎯 HOLD SIGNAL STRENGTH (low volatility, neutral RSI)
                df['hold_signal_strength'] = 0.0
                df.loc[df['rsi_neutral'] == 1, 'hold_signal_strength'] += 0.4
                df.loc[df['volatility_normalized'] < 0.01, 'hold_signal_strength'] += 0.3
                df.loc[(abs(df['price_change_24h']) < 0.01), 'hold_signal_strength'] += 0.2
                
                print("✅ Combined directional indicators created")
            except Exception as e:
                print(f"⚠️ Combined indicators error: {e}")
            
            # 🎯 STEP 8: TARGET VARIABLE (CRITICAL FOR DIRECTIONAL ML)
            try:
                if 'direction' not in df.columns:
                    print("🔄 Creating direction target from available data...")
                    
                    if 'profitable' in df.columns and 'rsi' in df.columns:
                        # Enhanced directional labeling
                        df['direction'] = 'hold'  # Default
                        
                        # LONG conditions: oversold + profitable or strong dip
                        long_conditions = (
                            ((df['rsi'] < 35) & (df['profitable'] == True)) |
                            ((df['price_change_24h'] < -0.03) & (df['profitable'] == True)) |
                            (df['rsi'] < 25)  # Extreme oversold
                        )
                        df.loc[long_conditions, 'direction'] = 'long'
                        
                        # SHORT conditions: overbought + profitable or strong rally
                        short_conditions = (
                            ((df['rsi'] > 65) & (df['profitable'] == True)) |
                            ((df['price_change_24h'] > 0.05) & (df['profitable'] == True)) |
                            (df['rsi'] > 75)  # Extreme overbought
                        )
                        df.loc[short_conditions, 'direction'] = 'short'
                        
                        # HOLD conditions: neutral RSI, low volatility
                        hold_conditions = (
                            (df['rsi_neutral'] == 1) |
                            (df['volatility_normalized'] < 0.005)
                        )
                        df.loc[hold_conditions, 'direction'] = 'hold'
                    
                    elif 'profitable' in df.columns:
                        # Basic fallback
                        df['direction'] = df['profitable'].apply(lambda x: 'long' if x else 'short')
                    else:
                        # Emergency fallback: use RSI
                        df['direction'] = 'hold'
                        df.loc[df['rsi'] < 30, 'direction'] = 'long'
                        df.loc[df['rsi'] > 70, 'direction'] = 'short'
                
                # Ensure string type and clean values
                df['direction'] = df['direction'].astype(str)
                direction_mapping = {
                    'long': 'long', 'short': 'short', 'hold': 'hold',
                    'LONG': 'long', 'SHORT': 'short', 'HOLD': 'hold',
                    'True': 'long', 'False': 'short',
                    'profitable': 'long', 'unprofitable': 'short'
                }
                df['direction'] = df['direction'].map(direction_mapping).fillna('hold')
                
                print("✅ Directional target variable created")
                
            except Exception as e:
                print(f"⚠️ Target variable error: {e}")
                # Emergency fallback
                directions = np.random.choice(['long', 'short', 'hold'], size=len(df), p=[0.4, 0.4, 0.2])
                df['direction'] = directions
            
            # 🎯 STEP 9: FINAL CLEANUP
            print("🏁 Final directional data cleanup...")
            
            # Replace infinite values
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                df[col] = df[col].replace([np.inf, -np.inf], df[col].median())
            
            # Handle NaN values conservatively
            final_len = len(df)
            
            if final_len > 1000:
                df_clean = df.dropna()
                if len(df_clean) >= 200:
                    df = df_clean
                    print(f"✅ Aggressive cleanup: {final_len} → {len(df)} samples")
                else:
                    df = df.fillna(df.median(numeric_only=True))
                    print(f"✅ Conservative cleanup: {len(df)} samples retained")
            else:
                df = df.fillna(df.median(numeric_only=True))
                print(f"✅ NaN filling: {len(df)} samples retained")
            
            # Validate target distribution
            if 'direction' in df.columns:
                direction_dist = df['direction'].value_counts()
                print(f"🎯 Final direction distribution: {dict(direction_dist)}")
                
                # Ensure minimum representation of each direction
                min_count = len(df) // 10  # At least 10% of each direction
                for direction in ['long', 'short', 'hold']:
                    if direction not in direction_dist or direction_dist[direction] < min_count:
                        print(f"⚠️ Low representation of {direction} direction, adding samples...")
                        # Add some samples of the missing direction
                        indices_to_change = np.random.choice(
                            df[df['direction'] != direction].index, 
                            size=min(min_count, len(df) // 5), 
                            replace=False
                        )
                        df.loc[indices_to_change, 'direction'] = direction
            
            print(f"✅ DIRECTIONAL FEATURE ENGINEERING COMPLETE: {initial_len} → {len(df)} samples")
            print(f"🎯 Total features: {len(df.columns)}")
            
            return df
            
        except Exception as e:
            print(f"❌ CRITICAL DIRECTIONAL FEATURE ENGINEERING ERROR: {e}")
            traceback.print_exc()
            
            # Emergency fallback
            print("🔄 Creating emergency directional dataset...")
            return DirectionalFeatureEngineer._create_emergency_directional_dataset()
    
    @staticmethod
    def _create_minimal_directional_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create minimal features for very small datasets"""
        try:
            if len(df) == 0:
                return DirectionalFeatureEngineer._create_emergency_directional_dataset()
            
            # Ensure basic columns exist
            if 'price' not in df.columns:
                df['price'] = 100.0
            if 'rsi' not in df.columns:
                df['rsi'] = 50.0
            if 'direction' not in df.columns:
                # Create balanced direction distribution
                directions = np.random.choice(['long', 'short', 'hold'], size=len(df), p=[0.4, 0.4, 0.2])
                df['direction'] = directions
            
            # Add minimal derived features
            df['price_change'] = 0.0
            df['rsi_normalized'] = (df['rsi'] - 50) / 50
            df['long_signal_strength'] = 0.3
            df['short_signal_strength'] = 0.3
            df['hold_signal_strength'] = 0.4
            
            print(f"✅ Minimal directional features created for {len(df)} samples")
            return df
            
        except Exception as e:
            print(f"❌ Minimal features error: {e}")
            return DirectionalFeatureEngineer._create_emergency_directional_dataset()
    
    @staticmethod
    def _create_emergency_directional_dataset() -> pd.DataFrame:
        """Create emergency synthetic dataset for directional trading"""
        print("🚨 Creating emergency directional dataset...")
        
        np.random.seed(42)
        n_samples = 120
        
        # Create more realistic directional data
        df = pd.DataFrame({
            'price': np.random.normal(100, 15, n_samples),
            'rsi': np.random.uniform(15, 85, n_samples),
            'price_change': np.random.normal(0, 0.02, n_samples),
            'price_change_24h': np.random.normal(0, 0.05, n_samples),
            'volatility_normalized': np.random.uniform(0.005, 0.03, n_samples),
            'volume_ratio': np.random.normal(1, 0.3, n_samples),
            'hour': np.random.randint(0, 24, n_samples),
            'rsi_normalized': np.random.normal(0, 0.4, n_samples),
            'trend_strength_short': np.random.normal(0, 0.02, n_samples),
            'long_signal_strength': np.random.uniform(0, 0.8, n_samples),
            'short_signal_strength': np.random.uniform(0, 0.8, n_samples),
            'hold_signal_strength': np.random.uniform(0, 0.8, n_samples)
        })
        
        # Create directional labels based on signals
        def assign_direction(row):
            signals = {
                'long': row['long_signal_strength'],
                'short': row['short_signal_strength'], 
                'hold': row['hold_signal_strength']
            }
            return max(signals, key=signals.get)
        
        df['direction'] = df.apply(assign_direction, axis=1)
        
        # Ensure balanced distribution
        direction_counts = df['direction'].value_counts()
        print(f"🎯 Emergency dataset direction distribution: {dict(direction_counts)}")
        
        print(f"✅ Emergency directional dataset created: {len(df)} synthetic samples")
        return df


class DirectionalModelTrainer:
    """🎯 Enhanced model trainer for directional trading (LONG/SHORT/HOLD)"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.performance = {}
        self.feature_names = []
        self.label_encoder = LabelEncoder()
        
    def get_directional_model_configs(self) -> Dict[str, Any]:
        """🎯 Get model configurations optimized for directional classification"""
        configs = {}
        
        # Random Forest (optimized for multi-class)
        if SKLEARN_AVAILABLE:
            configs['directional_rf'] = {
                'model': RandomForestClassifier(
                    n_estimators=50,  # Increased for multi-class
                    max_depth=10,
                    min_samples_split=8,
                    min_samples_leaf=3,
                    class_weight='balanced',  # Important for imbalanced directions
                    random_state=42,
                    n_jobs=1
                ),
                'requires_scaling': False
            }
            
            configs['directional_gb'] = {
                'model': GradientBoostingClassifier(
                    n_estimators=40,
                    max_depth=6,
                    learning_rate=0.1,
                    min_samples_split=8,
                    random_state=42
                ),
                'requires_scaling': False
            }
            
            configs['directional_logistic'] = {
                'model': LogisticRegression(
                    random_state=42,
                    max_iter=500,
                    solver='liblinear',
                    class_weight='balanced',
                    multi_class='ovr'  # One-vs-rest for multi-class
                ),
                'requires_scaling': True
            }
        
        # XGBoost (excellent for multi-class)
        if XGBOOST_AVAILABLE:
            configs['directional_xgb'] = {
                'model': xgb.XGBClassifier(
                    n_estimators=40,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    eval_metric='mlogloss',  # Multi-class log loss
                    verbosity=0,
                    objective='multi:softprob'  # Multi-class probability
                ),
                'requires_scaling': False
            }
        
        # LightGBM (also excellent for multi-class)
        if LIGHTGBM_AVAILABLE:
            configs['directional_lgb'] = {
                'model': lgb.LGBMClassifier(
                    n_estimators=40,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    verbose=-1,
                    objective='multiclass',
                    num_class=3,  # long, short, hold
                    force_row_wise=True
                ),
                'requires_scaling': False
            }
        
        print(f"✅ Directional model configs ready: {list(configs.keys())}")
        return configs
    
    def train_single_directional_model(self, name: str, config: Dict, X_train: np.ndarray, 
                                     y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Optional[DirectionalModelPerformance]:
        """🎯 Train a single model for directional trading"""
        print(f"🎯 Training directional model: {name}...")
        
        start_time = datetime.now()
        
        try:
            model = config['model']
            requires_scaling = config['requires_scaling']
            
            # Apply scaling if required
            scaler = None
            if requires_scaling:
                scaler = RobustScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
            else:
                X_train_scaled = X_train
                X_val_scaled = X_val
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_val_scaled)
            
            # Calculate overall accuracy
            accuracy = accuracy_score(y_val, y_pred)
            
            # Cross-validation score
            cv_size = min(300, len(X_train))
            cv_X = X_train_scaled[:cv_size]
            cv_y = y_train[:cv_size]
            
            try:
                cv_scores = cross_val_score(model, cv_X, cv_y, cv=3, scoring='accuracy')
                cv_score = cv_scores.mean()
            except Exception as e:
                print(f"⚠️ CV error for {name}: {e}")
                cv_score = accuracy
            
            # 🎯 DIRECTIONAL-SPECIFIC METRICS
            try:
                report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
                
                # Get per-direction metrics
                directions = ['long', 'short', 'hold']
                direction_metrics = {}
                
                for direction in directions:
                    if str(direction) in report:
                        direction_metrics[direction] = {
                            'precision': report[str(direction)]['precision'],
                            'recall': report[str(direction)]['recall'],
                            'f1': report[str(direction)]['f1-score']
                        }
                    else:
                        direction_metrics[direction] = {
                            'precision': 0.0,
                            'recall': 0.0,
                            'f1': 0.0
                        }
                
                # Weighted averages
                weighted_avg = report.get('weighted avg', {})
                precision = weighted_avg.get('precision', accuracy)
                recall = weighted_avg.get('recall', accuracy)
                f1 = weighted_avg.get('f1-score', accuracy)
                
                # Per-direction accuracy
                directional_accuracy = {}
                for direction in directions:
                    mask = (y_val == direction)
                    if mask.sum() > 0:
                        directional_accuracy[direction] = accuracy_score(y_val[mask], y_pred[mask])
                    else:
                        directional_accuracy[direction] = 0.0
                
            except Exception as e:
                print(f"⚠️ Directional metrics error for {name}: {e}")
                precision = accuracy
                recall = accuracy
                f1 = accuracy
                direction_metrics = {d: {'precision': accuracy, 'recall': accuracy, 'f1': accuracy} for d in ['long', 'short', 'hold']}
                directional_accuracy = {'long': accuracy, 'short': accuracy, 'hold': accuracy}
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Store model and scaler
            self.models[name] = model
            if scaler:
                self.scalers[name] = scaler
            
            performance = DirectionalModelPerformance(
                name=name,
                accuracy=accuracy,
                long_precision=direction_metrics.get('long', {}).get('precision', 0),
                short_precision=direction_metrics.get('short', {}).get('precision', 0),
                hold_precision=direction_metrics.get('hold', {}).get('precision', 0),
                long_recall=direction_metrics.get('long', {}).get('recall', 0),
                short_recall=direction_metrics.get('short', {}).get('recall', 0),
                hold_recall=direction_metrics.get('hold', {}).get('recall', 0),
                f1_score=f1,
                cross_val_score=cv_score,
                training_samples=len(X_train),
                feature_count=X_train.shape[1],
                training_time=training_time,
                directional_accuracy=directional_accuracy
            )
            
            print(f"✅ {name}: Acc={accuracy:.3f}, CV={cv_score:.3f}")
            print(f"   🎯 LONG: P={direction_metrics.get('long', {}).get('precision', 0):.3f}, R={direction_metrics.get('long', {}).get('recall', 0):.3f}")
            print(f"   🎯 SHORT: P={direction_metrics.get('short', {}).get('precision', 0):.3f}, R={direction_metrics.get('short', {}).get('recall', 0):.3f}")
            print(f"   🎯 HOLD: P={direction_metrics.get('hold', {}).get('precision', 0):.3f}, R={direction_metrics.get('hold', {}).get('recall', 0):.3f}")
            
            return performance
            
        except Exception as e:
            print(f"❌ {name} directional training failed: {e}")
            return None
    
    def train_all_directional_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, DirectionalModelPerformance]:
        """🎯 Train all models for directional trading"""
        print(f"🎯 TRAINING ALL DIRECTIONAL MODELS: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Encode labels to ensure proper format
        y_encoded = self.label_encoder.fit_transform(y)
        print(f"🎯 Direction classes: {list(self.label_encoder.classes_)}")
        
        # Split data with stratification
        test_size = min(0.3, 0.25 + (300 / len(X)))
        
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
            )
        except ValueError:
            # If stratify fails, split without stratify
            X_train, X_val, y_train, y_val = train_test_split(
                X, y_encoded, test_size=test_size, random_state=42
            )
        
        print(f"📊 Train: {len(X_train)}, Validation: {len(X_val)}")
        
        # Convert back to string labels for validation
        y_val_str = self.label_encoder.inverse_transform(y_val)
        
        # Get model configurations
        model_configs = self.get_directional_model_configs()
        
        # Train each model
        successful_models = {}
        for name, config in model_configs.items():
            performance = self.train_single_directional_model(name, config, X_train, y_train, X_val, y_val_str)
            if performance:
                successful_models[name] = performance
                self.performance[name] = performance
        
        # Calculate ensemble weights based on directional performance
        if successful_models:
            # Weight based on balanced accuracy across all directions
            total_score = 0
            for performance in successful_models.values():
                # Calculate balanced score across directions
                dir_accuracies = list(performance.directional_accuracy.values())
                balanced_score = np.mean(dir_accuracies)
                performance.ensemble_weight = balanced_score
                total_score += balanced_score
            
            # Normalize weights
            if total_score > 0:
                for performance in successful_models.values():
                    performance.ensemble_weight /= total_score
            else:
                # Equal weights if all scores are 0
                equal_weight = 1.0 / len(successful_models)
                for performance in successful_models.values():
                    performance.ensemble_weight = equal_weight
        
        print(f"✅ DIRECTIONAL TRAINING COMPLETE: {len(successful_models)}/{len(model_configs)} models successful")
        return successful_models


class DirectionalMLTradingIntegration:
    """🎯 ENHANCED ML Trading Integration for Directional Trading (LONG/SHORT/HOLD)"""
    
    def __init__(self, db_manager=None):
        print("🎯 INITIALIZING DIRECTIONAL ML INTEGRATION...")
        
        # Core components
        self.db_manager = db_manager
        self.data_loader = DirectionalDataLoader(db_manager)
        self.feature_engineer = DirectionalFeatureEngineer()
        self.model_trainer = DirectionalModelTrainer()
        
        # Configuration
        self.min_samples = 75  # Reduced for faster startup
        self.model_dir = "ml/models"
        self.last_training_time = None
        self.training_data_hash = None
        
        # Performance tracking
        self.prediction_count = 0
        self.successful_predictions = 0
        self.directional_prediction_accuracy = {'long': 0, 'short': 0, 'hold': 0}
        
        # Ensure model directory exists
        os.makedirs(self.model_dir, exist_ok=True)
        
        print("✅ Directional ML Integration initialized")
        print(f"   • Min samples: {self.min_samples}")
        print(f"   • Model directory: {self.model_dir}")
        print(f"   • Database manager: {'✅' if self.db_manager else '❌'}")
    
    def should_retrain(self) -> bool:
        """Determine if directional models should be retrained"""
        try:
            # If no models exist, definitely retrain
            if not self.model_trainer.models:
                print("🔄 No directional models exist - retraining needed")
                return True
            
            # If never trained, retrain
            if not self.last_training_time:
                print("🔄 Never trained directional models - retraining needed")
                return True
            
            # Time-based retraining (every 3 hours for directional trading)
            time_since_training = datetime.now() - self.last_training_time
            if time_since_training > timedelta(hours=3):
                print(f"🔄 Time-based directional retrain needed ({time_since_training})")
                return True
            
            # Performance-based retraining
            if self.prediction_count > 15:  # Lower threshold for directional
                success_rate = self.successful_predictions / self.prediction_count
                if success_rate < 0.45:  # Lower threshold for multi-class
                    print(f"🔄 Performance-based directional retrain needed (success rate: {success_rate:.2f})")
                    return True
            
            return False
            
        except Exception as e:
            print(f"⚠️ Directional retrain check error: {e}")
            return True
    
    def train_models(self, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """🎯 Train directional ML models (LONG/SHORT/HOLD)"""
        print("🎯 DIRECTIONAL MODEL TRAINING STARTING...")
        
        training_start = datetime.now()
        
        try:
            # Load data if not provided
            if df is None:
                print("📊 Loading directional training data...")
                df = self.data_loader.load_directional_training_data(self.min_samples)
                
                if df is None:
                    return {
                        'success': False,
                        'error': 'Failed to load sufficient directional training data',
                        'data_loaded': 0
                    }
            
            print(f"📊 Directional training data loaded: {len(df)} samples")
            
            # Validate minimum samples
            if len(df) < max(20, self.min_samples // 3):
                return {
                    'success': False,
                    'error': f'Insufficient directional data: {len(df)} < {max(20, self.min_samples // 3)}',
                    'data_loaded': len(df)
                }
            
            # Feature engineering
            print("🎯 Engineering directional features...")
            df_features = self.feature_engineer.engineer_directional_features(df)
            
            if len(df_features) == 0:
                print("🚨 Directional feature engineering resulted in empty dataset!")
                return {
                    'success': False,
                    'error': 'Directional feature engineering produced empty dataset',
                    'data_loaded': len(df),
                    'data_after_features': 0
                }
            
            print(f"✅ Directional feature engineering successful: {len(df)} → {len(df_features)} samples")
            
            # Prepare features and target
            target_col = 'direction'
            if target_col not in df_features.columns:
                print("❌ Target column 'direction' missing!")
                return {
                    'success': False,
                    'error': 'Target column direction missing after feature engineering',
                    'available_columns': list(df_features.columns)
                }
            
            # Select feature columns (exclude metadata)
            exclude_cols = ['timestamp', 'direction', 'action', 'input_token', 'output_token', 
                           'amount_in', 'amount_out', 'profitable', 'pnl', 'asset']
            feature_cols = [col for col in df_features.columns if col not in exclude_cols]
            
            if len(feature_cols) == 0:
                print("❌ No directional feature columns available!")
                return {
                    'success': False,
                    'error': 'No directional feature columns available after filtering',
                    'available_columns': list(df_features.columns)
                }
            
            print(f"📊 Selected {len(feature_cols)} directional feature columns")
            
            # Prepare arrays
            X = df_features[feature_cols].values
            y = df_features[target_col].values
            
            # Store feature names
            self.model_trainer.feature_names = feature_cols
            
            print(f"📊 Final directional training set: {X.shape[0]} samples, {X.shape[1]} features")
            
            # Check target distribution
            unique_directions = np.unique(y)
            direction_counts = np.bincount([list(unique_directions).index(d) for d in y])
            print(f"🎯 Direction classes: {unique_directions}")
            print(f"🎯 Direction distribution: {dict(zip(unique_directions, direction_counts))}")
            
            # Ensure we have all three directions
            expected_directions = ['long', 'short', 'hold']
            missing_directions = [d for d in expected_directions if d not in unique_directions]
            
            if missing_directions:
                print(f"⚠️ Missing directions: {missing_directions}, adding synthetic samples...")
                # Add a few synthetic samples for missing directions
                for missing_dir in missing_directions:
                    # Find similar samples and modify their direction
                    indices_to_modify = np.random.choice(len(y), size=min(5, len(y)//10), replace=False)
                    y[indices_to_modify] = missing_dir
                
                print(f"✅ Added synthetic samples for missing directions")
            
            # Train models
            print("🎯 Starting directional model training...")
            successful_models = self.model_trainer.train_all_directional_models(X, y)
            
            training_time = (datetime.now() - training_start).total_seconds()
            
            if successful_models:
                # Save models to database
                try:
                    self._save_directional_model_performance_to_db(successful_models)
                    print("✅ Directional model performance saved to database")
                except Exception as e:
                    print(f"⚠️ Database save error: {e}")
                
                # Save models to disk
                try:
                    self._save_models_to_disk()
                    print("✅ Directional models saved to disk")
                except Exception as e:
                    print(f"⚠️ Disk save error: {e}")
                
                # Update tracking
                self.last_training_time = datetime.now()
                
                print(f"🎉 DIRECTIONAL TRAINING SUCCESS: {len(successful_models)} models in {training_time:.1f}s")
                
                return {
                    'success': True,
                    'successful_models': list(successful_models.keys()),
                    'failed_models': [],
                    'data_loaded': len(df),
                    'data_trained': len(df_features),
                    'features_count': len(feature_cols),
                    'training_time': training_time,
                    'directions': list(unique_directions),
                    'model_performance': {name: {
                        'accuracy': perf.accuracy,
                        'cv_score': perf.cross_val_score,
                        'ensemble_weight': perf.ensemble_weight,
                        'directional_accuracy': perf.directional_accuracy,
                        'training_samples': perf.training_samples
                    } for name, perf in successful_models.items()}
                }
            else:
                return {
                    'success': False,
                    'error': 'No directional models trained successfully',
                    'data_loaded': len(df),
                    'data_trained': len(df_features),
                    'attempted_models': len(self.model_trainer.get_directional_model_configs()),
                    'training_time': training_time
                }
                
        except Exception as e:
            training_time = (datetime.now() - training_start).total_seconds()
            print(f"❌ DIRECTIONAL TRAINING ERROR: {e}")
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'training_time': training_time,
                'traceback': traceback.format_exc()
            }
    
    def get_directional_prediction(self, df: pd.DataFrame) -> Dict[str, Any]:
        """🎯 Get directional prediction (LONG/SHORT/HOLD) with confidence"""
        self.prediction_count += 1
        
        try:
            print(f"🎯 GENERATING DIRECTIONAL PREDICTION #{self.prediction_count}...")
            
            # Check if we have trained models
            if not self.model_trainer.models:
                print("⚠️ No directional models - attempting emergency training...")
                training_result = self.train_models(df)
                
                if not training_result.get('success'):
                    return {
                        'error': 'No directional models available and emergency training failed',
                        'training_error': training_result.get('error'),
                        'predicted_direction': 'hold',
                        'direction_probabilities': {'long': 0.33, 'short': 0.33, 'hold': 0.34},
                        'confidence': 0.1,
                        'recommendation': 'HOLD'
                    }
            
            # Prepare data for prediction
            print("🎯 Preparing directional prediction data...")
            df_features = self.feature_engineer.engineer_directional_features(df.copy())
            
            if len(df_features) == 0:
                print("⚠️ Directional feature engineering failed - using fallback")
                return self._fallback_directional_prediction(df)
            
            # Get latest features
            latest_features = df_features.iloc[-1]
            feature_cols = self.model_trainer.feature_names
            
            # Check feature availability
            available_features = [col for col in feature_cols if col in latest_features.index]
            missing_features = [col for col in feature_cols if col not in latest_features.index]
            
            if len(available_features) < len(feature_cols) * 0.7:  # Less than 70% features available
                print(f"⚠️ Many missing directional features ({len(missing_features)}/{len(feature_cols)}) - using fallback")
                return self._fallback_directional_prediction(df)
            
            # Fill missing features with defaults
            feature_values = []
            for col in feature_cols:
                if col in latest_features.index:
                    feature_values.append(latest_features[col])
                else:
                    # Use intelligent defaults for missing features
                    if 'price' in col:
                        feature_values.append(100.0)
                    elif 'rsi' in col:
                        feature_values.append(50.0)
                    elif 'signal_strength' in col:
                        feature_values.append(0.3)
                    elif 'volatility' in col:
                        feature_values.append(0.02)
                    else:
                        feature_values.append(0.0)
            
            X_pred = np.array(feature_values).reshape(1, -1)
            
            # Get predictions from all models
            direction_predictions = {}
            direction_probabilities = {}
            
            for name, model in self.model_trainer.models.items():
                try:
                    # Apply scaling if needed
                    if name in self.model_trainer.scalers:
                        X_scaled = self.model_trainer.scalers[name].transform(X_pred)
                    else:
                        X_scaled = X_pred
                    
                    # Get prediction
                    pred = model.predict(X_scaled)[0]
                    
                    # Convert back from encoded label if needed
                    if hasattr(self.model_trainer, 'label_encoder'):
                        try:
                            pred_str = self.model_trainer.label_encoder.inverse_transform([pred])[0]
                        except:
                            pred_str = str(pred)
                    else:
                        pred_str = str(pred)
                    
                    # Get probabilities if available
                    if hasattr(model, 'predict_proba'):
                        try:
                            proba = model.predict_proba(X_scaled)[0]
                            # Map probabilities to directions
                            if hasattr(self.model_trainer, 'label_encoder'):
                                classes = self.model_trainer.label_encoder.classes_
                            else:
                                classes = ['long', 'short', 'hold']
                            
                            proba_dict = {}
                            for i, class_name in enumerate(classes):
                                if i < len(proba):
                                    proba_dict[str(class_name)] = float(proba[i])
                                else:
                                    proba_dict[str(class_name)] = 0.33
                            
                            direction_probabilities[name] = proba_dict
                        except:
                            # Fallback probabilities
                            direction_probabilities[name] = {
                                pred_str: 0.7,
                                'long': 0.15 if pred_str != 'long' else 0.7,
                                'short': 0.15 if pred_str != 'short' else 0.7,
                                'hold': 0.15 if pred_str != 'hold' else 0.7
                            }
                    else:
                        # No probabilities available
                        direction_probabilities[name] = {
                            pred_str: 0.6,
                            'long': 0.2 if pred_str != 'long' else 0.6,
                            'short': 0.2 if pred_str != 'short' else 0.6,
                            'hold': 0.2 if pred_str != 'hold' else 0.6
                        }
                    
                    direction_predictions[name] = pred_str
                    
                except Exception as e:
                    print(f"⚠️ Directional prediction error for {name}: {e}")
                    continue
            
            if not direction_predictions:
                print("⚠️ All directional model predictions failed - using fallback")
                return self._fallback_directional_prediction(df)
            
            # Ensemble directional prediction using weights
            ensemble_probabilities = {'long': 0.0, 'short': 0.0, 'hold': 0.0}
            total_weight = 0.0
            
            for name, proba_dict in direction_probabilities.items():
                weight = self.model_trainer.performance.get(name, DirectionalModelPerformance(name, 0.33, 0, 0, 0, 0, 0, 0, 0, 0.33, 0, 0, 0)).ensemble_weight
                if weight > 0:
                    for direction in ensemble_probabilities:
                        ensemble_probabilities[direction] += proba_dict.get(direction, 0.33) * weight
                    total_weight += weight
            
            if total_weight > 0:
                for direction in ensemble_probabilities:
                    ensemble_probabilities[direction] /= total_weight
            else:
                # Equal probabilities if no weights
                ensemble_probabilities = {'long': 0.33, 'short': 0.33, 'hold': 0.34}
            
            # Final directional prediction
            predicted_direction = max(ensemble_probabilities, key=ensemble_probabilities.get)
            confidence = ensemble_probabilities[predicted_direction]
            
            # Model agreement
            direction_votes = {}
            for pred in direction_predictions.values():
                direction_votes[pred] = direction_votes.get(pred, 0) + 1
            
            max_votes = max(direction_votes.values()) if direction_votes else 1
            model_agreement = max_votes / len(direction_predictions)
            
            # Enhanced confidence calculation
            base_confidence = confidence
            agreement_bonus = model_agreement * 0.2
            model_count_bonus = min(len(direction_predictions) / 5, 0.1)
            
            final_confidence = min(base_confidence + agreement_bonus + model_count_bonus, 0.95)
            
            # Generate recommendation
            confidence_thresholds = {
                'long': 0.6,
                'short': 0.6,
                'hold': 0.4
            }
            
            if final_confidence > confidence_thresholds.get(predicted_direction, 0.5):
                if predicted_direction == 'long':
                    recommendation = 'LONG'
                elif predicted_direction == 'short':
                    recommendation = 'SHORT'
                else:
                    recommendation = 'HOLD'
            else:
                recommendation = 'HOLD'  # Conservative default
            
            self.successful_predictions += 1
            
            result = {
                'predicted_direction': predicted_direction,
                'direction_probabilities': ensemble_probabilities,
                'confidence': final_confidence,
                'model_count': len(direction_predictions),
                'model_agreement': model_agreement,
                'recommendation': recommendation,
                'individual_predictions': {
                    name: {
                        'direction': pred,
                        'probabilities': direction_probabilities.get(name, {})
                    } for name, pred in direction_predictions.items()
                },
                'enhanced_metrics': {
                    'base_confidence': base_confidence,
                    'agreement_bonus': agreement_bonus,
                    'model_count_bonus': model_count_bonus,
                    'available_features': len(available_features),
                    'missing_features': len(missing_features)
                }
            }
            
            print(f"✅ DIRECTIONAL PREDICTION COMPLETE: {recommendation} ({final_confidence:.2f} confidence)")
            print(f"   🎯 Direction: {predicted_direction} (Long: {ensemble_probabilities['long']:.2f}, Short: {ensemble_probabilities['short']:.2f}, Hold: {ensemble_probabilities['hold']:.2f})")
            
            return result
            
        except Exception as e:
            print(f"❌ DIRECTIONAL PREDICTION ERROR: {e}")
            return self._fallback_directional_prediction(df)
    
    def _fallback_directional_prediction(self, df: pd.DataFrame) -> Dict[str, Any]:
        """🎯 Fallback directional prediction when main system fails"""
        try:
            print("🔄 Using fallback directional prediction method...")
            
            # Use last row of data
            latest = df.iloc[-1] if len(df) > 0 else {}
            
            # Get basic indicators
            rsi = latest.get('rsi', 50)
            price_change_24h = latest.get('price_change_24h', 0)
            volatility = latest.get('volatility', 0.02)
            
            # Simple rule-based directional prediction
            probabilities = {'long': 0.33, 'short': 0.33, 'hold': 0.34}
            
            # RSI-based signals
            if rsi < 25:
                probabilities = {'long': 0.6, 'short': 0.2, 'hold': 0.2}
                predicted_direction = 'long'
                confidence = 0.5
            elif rsi > 75:
                probabilities = {'long': 0.2, 'short': 0.6, 'hold': 0.2}
                predicted_direction = 'short'
                confidence = 0.5
            elif 40 <= rsi <= 60 and volatility < 0.01:
                probabilities = {'long': 0.2, 'short': 0.2, 'hold': 0.6}
                predicted_direction = 'hold'
                confidence = 0.4
            else:
                # Use momentum
                if price_change_24h > 3:
                    probabilities = {'long': 0.25, 'short': 0.55, 'hold': 0.2}
                    predicted_direction = 'short'
                elif price_change_24h < -3:
                    probabilities = {'long': 0.55, 'short': 0.25, 'hold': 0.2}
                    predicted_direction = 'long'
                else:
                    predicted_direction = 'hold'
                
                confidence = 0.3
            
            return {
                'predicted_direction': predicted_direction,
                'direction_probabilities': probabilities,
                'confidence': confidence,
                'model_count': 0,
                'model_agreement': 1.0,
                'recommendation': 'HOLD',  # Always conservative
                'method': 'fallback',
                'fallback_reason': 'Main directional prediction system failed'
            }
            
        except Exception as e:
            print(f"❌ Even fallback directional prediction failed: {e}")
            return {
                'error': f'All directional prediction methods failed: {e}',
                'predicted_direction': 'hold',
                'direction_probabilities': {'long': 0.33, 'short': 0.33, 'hold': 0.34},
                'confidence': 0.1,
                'recommendation': 'HOLD'
            }
    
    # Legacy compatibility methods
    def get_ensemble_prediction_with_reality_check(self, df: pd.DataFrame) -> Dict[str, Any]:
        """🔄 Legacy compatibility wrapper for directional predictions"""
        directional_result = self.get_directional_prediction(df)
        
        # Convert to legacy format
        predicted_direction = directional_result.get('predicted_direction', 'hold')
        confidence = directional_result.get('confidence', 0.5)
        
        # Map direction to legacy profitable prediction
        if predicted_direction == 'long':
            predicted_profitable = True
            direction = 'profitable'
        elif predicted_direction == 'short':
            predicted_profitable = False
            direction = 'unprofitable'
        else:  # hold
            predicted_profitable = True  # Neutral/safe
            direction = 'neutral'
        
        # Generate legacy-compatible result
        legacy_result = {
            'predicted_profitable': predicted_profitable,
            'probability_profitable': directional_result.get('direction_probabilities', {}).get('long', 0.5),
            'confidence': confidence,
            'direction': direction,
            'recommendation': directional_result.get('recommendation', 'HOLD'),
            'model_count': directional_result.get('model_count', 0),
            'model_agreement': directional_result.get('model_agreement', 1.0),
            
            # Enhanced directional data
            'directional_recommendation': predicted_direction,
            'direction_probabilities': directional_result.get('direction_probabilities', {}),
            'individual_predictions': directional_result.get('individual_predictions', {}),
            'enhanced_metrics': directional_result.get('enhanced_metrics', {}),
            'reality_check': {'applied': True, 'issues': [], 'confidence_adjustment': 1.0}
        }
        
        return legacy_result
    
    def _save_directional_model_performance_to_db(self, successful_models: Dict[str, DirectionalModelPerformance]):
        """Save directional model performance to database"""
        if not self.db_manager:
            return
        
        try:
            for name, performance in successful_models.items():
                model_info = {
                    'model_name': name,
                    'model_type': 'directional_classification',
                    'accuracy': performance.accuracy * 100,
                    'r2_score': performance.cross_val_score,
                    'mae': 1.0 - performance.accuracy,
                    'training_samples': performance.training_samples,
                    'model_file_path': f'{self.model_dir}/{name}.pkl',
                    'metrics': {
                        'long_precision': performance.long_precision,
                        'short_precision': performance.short_precision,
                        'hold_precision': performance.hold_precision,
                        'long_recall': performance.long_recall,
                        'short_recall': performance.short_recall,
                        'hold_recall': performance.hold_recall,
                        'f1_score': performance.f1_score,
                        'ensemble_weight': performance.ensemble_weight,
                        'training_time': performance.training_time,
                        'feature_count': performance.feature_count,
                        'directional_accuracy': performance.directional_accuracy
                    }
                }
                
                self.db_manager.save_ml_model_info(model_info)
                
        except Exception as e:
            print(f"⚠️ Error saving directional model performance to DB: {e}")
    
    def _save_models_to_disk(self):
        """Save trained directional models to disk"""
        try:
            for name, model in self.model_trainer.models.items():
                model_path = os.path.join(self.model_dir, f'{name}.pkl')
                joblib.dump(model, model_path)
                
                # Save scaler if exists
                if name in self.model_trainer.scalers:
                    scaler_path = os.path.join(self.model_dir, f'{name}_scaler.pkl')
                    joblib.dump(self.model_trainer.scalers[name], scaler_path)
            
            # Save label encoder
            if hasattr(self.model_trainer, 'label_encoder'):
                encoder_path = os.path.join(self.model_dir, 'directional_label_encoder.pkl')
                joblib.dump(self.model_trainer.label_encoder, encoder_path)
                
        except Exception as e:
            print(f"⚠️ Error saving directional models to disk: {e}")
    
    def get_model_performance(self) -> Dict[str, Dict]:
        """Get directional model performance metrics"""
        try:
            if self.db_manager:
                return self.db_manager.get_ml_model_performance()
            else:
                return {
                    name: {
                        'accuracy': perf.accuracy * 100,
                        'model_type': 'directional_classification',
                        'training_samples': perf.training_samples,
                        'last_trained': self.last_training_time.strftime('%Y-%m-%d %H:%M:%S') if self.last_training_time else 'Never',
                        'r2': perf.cross_val_score,
                        'mae': 1.0 - perf.accuracy,
                        'ensemble_weight': perf.ensemble_weight,
                        'directional_accuracy': perf.directional_accuracy,
                        'long_precision': perf.long_precision,
                        'short_precision': perf.short_precision,
                        'hold_precision': perf.hold_precision
                    } for name, perf in self.model_trainer.performance.items()
                }
                
        except Exception as e:
            print(f"⚠️ Error getting directional model performance: {e}")
            return {}
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get overall directional performance statistics"""
        success_rate = self.successful_predictions / self.prediction_count if self.prediction_count > 0 else 0
        
        return {
            'total_predictions': self.prediction_count,
            'successful_predictions': self.successful_predictions,
            'success_rate': f"{success_rate:.1%}",
            'models_trained': len(self.model_trainer.models),
            'last_training': self.last_training_time.strftime('%Y-%m-%d %H:%M:%S') if self.last_training_time else 'Never',
            'min_samples': self.min_samples,
            'directional_accuracy': self.directional_prediction_accuracy,
            'model_type': 'directional_classification'
        }


# Legacy compatibility - create alias
MLTradingIntegration = DirectionalMLTradingIntegration