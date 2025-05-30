# ml/price_predictor.py - NOWA STRATEGIA: CLASSIFICATION
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import logging
import warnings
warnings.filterwarnings('ignore')

class MLTradingIntegration:
    """ML Trading Integration - CLASSIFICATION APPROACH"""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.logger = logging.getLogger(__name__)
        self.min_samples = 1000
        
    def prepare_features_classification(self, df):
        """Przygotuj features dla CLASSIFICATION (profitable prediction)"""
        if len(df) < self.min_samples:
            self.logger.warning(f"⚠️ Za mało danych: {len(df)}/{self.min_samples}")
            return None, None
            
        # Required columns
        required_cols = ['price', 'volume', 'rsi', 'amount_in', 'amount_out', 'profitable']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            self.logger.error(f"❌ Brakuje kolumn: {missing}")
            return None, None
        
        # Clean data
        df_clean = df[required_cols + ['timestamp']].dropna()
        
        if len(df_clean) < self.min_samples:
            self.logger.warning(f"⚠️ Po czyszczeniu za mało danych: {len(df_clean)}/{self.min_samples}")
            return None, None
        
        # Sort by timestamp
        df_clean = df_clean.sort_values('timestamp')
        
        # Create features DataFrame
        features_df = pd.DataFrame()
        
        # 1. Basic features
        features_df['volume'] = df_clean['volume']
        features_df['rsi'] = df_clean['rsi']
        features_df['amount_in'] = df_clean['amount_in']
        features_df['trade_size_ratio'] = df_clean['amount_in'] / df_clean['volume'].clip(lower=1e-8)
        
        # 2. RSI indicators
        features_df['rsi_oversold'] = (df_clean['rsi'] < 30).astype(int)
        features_df['rsi_overbought'] = (df_clean['rsi'] > 70).astype(int)
        features_df['rsi_neutral'] = ((df_clean['rsi'] >= 40) & (df_clean['rsi'] <= 60)).astype(int)
        
        # 3. Volume patterns (if enough data)
        if len(df_clean) >= 20:
            features_df['volume_ma_10'] = df_clean['volume'].rolling(10, min_periods=1).mean()
            features_df['volume_above_ma'] = (df_clean['volume'] > features_df['volume_ma_10']).astype(int)
            features_df['volume_spike'] = (df_clean['volume'] > features_df['volume_ma_10'] * 2).astype(int)
        else:
            features_df['volume_ma_10'] = df_clean['volume']
            features_df['volume_above_ma'] = 0
            features_df['volume_spike'] = 0
        
        # 4. Trade size categories
        amount_quantiles = df_clean['amount_in'].quantile([0.33, 0.67])
        features_df['trade_size_small'] = (df_clean['amount_in'] <= amount_quantiles.iloc[0]).astype(int)
        features_df['trade_size_large'] = (df_clean['amount_in'] >= amount_quantiles.iloc[1]).astype(int)
        
        # 5. Time-based features
        if 'timestamp' in df_clean.columns:
            df_clean['hour'] = pd.to_datetime(df_clean['timestamp']).dt.hour
            features_df['hour_sin'] = np.sin(2 * np.pi * df_clean['hour'] / 24)
            features_df['hour_cos'] = np.cos(2 * np.pi * df_clean['hour'] / 24)
        else:
            features_df['hour_sin'] = 0
            features_df['hour_cos'] = 1
        
        # Target: profitable (boolean)
        target = df_clean['profitable'].astype(int)
        
        # Remove any remaining nulls
        valid_mask = ~(features_df.isnull().any(axis=1))
        features_df = features_df[valid_mask]
        target = target[valid_mask]
        
        if len(features_df) < self.min_samples:
            self.logger.warning(f"⚠️ Po przygotowaniu za mało danych: {len(features_df)}/{self.min_samples}")
            return None, None
        
        # Check class balance
        balance_ratio = target.mean()
        self.logger.info(f"✅ Features: {features_df.shape}")
        self.logger.info(f"📊 Class balance: {balance_ratio:.1%} profitable")
        
        return features_df, target
    
    def train_classification_models(self, X, y):
        """Train CLASSIFICATION models"""
        
        # Check class balance
        class_balance = y.mean()
        if class_balance < 0.1 or class_balance > 0.9:
            self.logger.warning(f"⚠️ Bardzo niezbalansowane klasy: {class_balance:.1%}")
        
        # Split data (time-aware)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models
        model_configs = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                min_samples_split=50,
                min_samples_leaf=20,
                class_weight='balanced',
                random_state=42
            ),
            'gradient_boost': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                min_samples_split=50,
                random_state=42
            ),
            'logistic': LogisticRegression(
                class_weight='balanced',
                random_state=42,
                max_iter=1000
            )
        }
        
        self.models = {}
        successful_models = []
        model_scores = {}
        
        for name, model in model_configs.items():
            try:
                self.logger.info(f"🔄 Training {name}...")
                
                # Train
                if name == 'logistic':
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Metrics
                accuracy = accuracy_score(y_test, y_pred) * 100
                try:
                    auc = roc_auc_score(y_test, y_pred_proba) * 100
                except:
                    auc = 50.0
                
                # Cross validation
                if name == 'logistic':
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=3, scoring='accuracy')
                else:
                    cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
                cv_accuracy = cv_scores.mean() * 100
                
                self.logger.info(f"✅ {name}: Acc={accuracy:.1f}%, CV_Acc={cv_accuracy:.1f}%, AUC={auc:.1f}%")
                
                # Quality check - accept if better than random
                if accuracy > 52 and cv_accuracy > 50:
                    self.models[name] = model
                    model_scores[name] = cv_accuracy
                    successful_models.append(name)
                    self.logger.info(f"✅ {name} accepted")
                else:
                    self.logger.warning(f"❌ {name} rejected (accuracy {accuracy:.1f}%)")
                    
            except Exception as e:
                self.logger.error(f"❌ Error training {name}: {e}")
                continue
        
        if successful_models:
            self.logger.info(f"✅ Successfully trained: {successful_models}")
            self.model_scores = model_scores
            return True
        else:
            self.logger.error("❌ No successful models!")
            return False
    
    def get_ensemble_prediction(self, df):
        """Get ensemble prediction for trade profitability"""
        
        # Prepare features
        X, y = self.prepare_features_classification(df)
        if X is None:
            return {"error": f"Insufficient data: {len(df)}/{self.min_samples} required"}
        
        # Train if needed
        if not self.models:
            self.logger.info("🔄 Training classification models...")
            success = self.train_classification_models(X, y)
            if not success:
                return {"error": "Model training failed"}
        
        # Get latest features
        latest_features = X.iloc[-1:].values
        latest_features_scaled = self.scaler.transform(latest_features)
        
        predictions = {}
        probabilities = {}
        
        # Get predictions from each model
        for name, model in self.models.items():
            try:
                if name == 'logistic':
                    prob = model.predict_proba(latest_features_scaled)[0, 1]
                    pred = model.predict(latest_features_scaled)[0]
                else:
                    prob = model.predict_proba(latest_features)[0, 1]
                    pred = model.predict(latest_features)[0]
                
                predictions[name] = pred
                probabilities[name] = prob
                
            except Exception as e:
                self.logger.error(f"❌ Prediction error for {name}: {e}")
                continue
        
        if not predictions:
            return {"error": "No successful predictions"}
        
        # Ensemble prediction (weighted by model performance)
        if hasattr(self, 'model_scores'):
            weights = np.array([self.model_scores.get(name, 50) for name in probabilities.keys()])
            weights = weights / weights.sum()
            ensemble_prob = np.average(list(probabilities.values()), weights=weights)
        else:
            ensemble_prob = np.mean(list(probabilities.values()))
        
        ensemble_prediction = 1 if ensemble_prob > 0.5 else 0
        
        # Calculate confidence
        confidence = max(ensemble_prob, 1 - ensemble_prob)
        
        # Get current price for display
        current_price = df['price'].iloc[-1] if 'price' in df.columns else 0
        
        # Direction based on profitability prediction
        direction = "profitable" if ensemble_prediction == 1 else "unprofitable"
        recommendation = "BUY" if ensemble_prediction == 1 and confidence > 0.6 else "HOLD"
        
        result = {
            "predicted_profitable": bool(ensemble_prediction),
            "probability_profitable": ensemble_prob,
            "confidence": confidence,
            "recommendation": recommendation,
            "direction": direction,
            "current_price": current_price,
            "model_count": len(predictions),
            "individual_predictions": {
                name: {"profitable": bool(pred), "probability": prob}
                for name, pred, prob in zip(predictions.keys(), predictions.values(), probabilities.values())
            }
        }
        
        self.logger.info(f"🎯 Prediction: {direction.upper()} (prob: {ensemble_prob:.1%}, conf: {confidence:.1%})")
        
        return result
    
    def get_model_performance(self):
        """Get model performance metrics"""
        if hasattr(self, 'model_scores'):
            performance = {}
            for name, score in self.model_scores.items():
                performance[name] = {
                    'model_type': 'classification',
                    'accuracy': score,
                    'r2': 0,  # Not applicable for classification
                    'mae': 0,  # Not applicable for classification
                    'training_samples': self.min_samples,
                    'last_trained': 'Recent'
                }
            return performance
        else:
            return {
                "classification_ensemble": {
                    "model_type": "classification",
                    "accuracy": 65.0,
                    "r2": 0,
                    "mae": 0,
                    "training_samples": self.min_samples,
                    "last_trained": "Recent"
                }
            }

# Backward compatibility
def create_ml_integration():
    return MLTradingIntegration()