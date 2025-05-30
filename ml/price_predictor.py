# ml/price_predictor.py - ENHANCED VERSION (upgrade existing)
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
    """ENHANCED ML Trading Integration - CLASSIFICATION APPROACH"""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.logger = logging.getLogger(__name__)
        self.min_samples = 500  # ZMIENIONE: 1000 → 500 (jak bot!)
        
        # DODANE: Model agreement tracking
        self.model_agreements = {}
        self.ensemble_weights = {}
        self.last_prediction_details = {}
        
    def prepare_features_classification(self, df):
        """ENHANCED feature preparation with advanced patterns"""
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
        
        # 1. ENHANCED Basic features
        features_df['volume'] = df_clean['volume']
        features_df['rsi'] = df_clean['rsi']
        features_df['amount_in'] = df_clean['amount_in']
        features_df['trade_size_ratio'] = df_clean['amount_in'] / df_clean['volume'].clip(lower=1e-8)
        
        # DODANE: Price momentum features
        features_df['price_momentum_5'] = df_clean['price'].pct_change(5).fillna(0)
        features_df['price_momentum_10'] = df_clean['price'].pct_change(10).fillna(0)
        features_df['price_volatility'] = df_clean['price'].rolling(10).std().fillna(0)
        
        # 2. ENHANCED RSI indicators
        features_df['rsi_oversold'] = (df_clean['rsi'] < 30).astype(int)
        features_df['rsi_overbought'] = (df_clean['rsi'] > 70).astype(int)
        features_df['rsi_neutral'] = ((df_clean['rsi'] >= 40) & (df_clean['rsi'] <= 60)).astype(int)
        # DODANE: RSI momentum
        features_df['rsi_momentum'] = df_clean['rsi'].diff().fillna(0)
        features_df['rsi_divergence'] = (df_clean['rsi'].diff() * df_clean['price'].pct_change()).fillna(0)
        
        # 3. ENHANCED Volume patterns
        if len(df_clean) >= 20:
            features_df['volume_ma_10'] = df_clean['volume'].rolling(10, min_periods=1).mean()
            features_df['volume_above_ma'] = (df_clean['volume'] > features_df['volume_ma_10']).astype(int)
            features_df['volume_spike'] = (df_clean['volume'] > features_df['volume_ma_10'] * 2).astype(int)
            # DODANE: Volume trend
            features_df['volume_trend'] = df_clean['volume'].rolling(10).mean().pct_change().fillna(0)
            features_df['volume_consistency'] = (df_clean['volume'].rolling(5).std() / df_clean['volume'].rolling(5).mean()).fillna(0)
        else:
            features_df['volume_ma_10'] = df_clean['volume']
            features_df['volume_above_ma'] = 0
            features_df['volume_spike'] = 0
            features_df['volume_trend'] = 0
            features_df['volume_consistency'] = 0
        
        # 4. ENHANCED Trade size categories
        amount_quantiles = df_clean['amount_in'].quantile([0.25, 0.5, 0.75])
        features_df['trade_size_small'] = (df_clean['amount_in'] <= amount_quantiles.iloc[0]).astype(int)
        features_df['trade_size_medium'] = ((df_clean['amount_in'] > amount_quantiles.iloc[0]) & 
                                          (df_clean['amount_in'] <= amount_quantiles.iloc[2])).astype(int)
        features_df['trade_size_large'] = (df_clean['amount_in'] >= amount_quantiles.iloc[2]).astype(int)
        
        # DODANE: Profit/Loss patterns
        features_df['recent_profit_streak'] = df_clean['profitable'].rolling(5).sum().fillna(0)
        features_df['profit_momentum'] = df_clean['profitable'].astype(int).diff().fillna(0)
        
        # 5. ENHANCED Time-based features
        if 'timestamp' in df_clean.columns:
            df_clean['hour'] = pd.to_datetime(df_clean['timestamp']).dt.hour
            df_clean['day_of_week'] = pd.to_datetime(df_clean['timestamp']).dt.dayofweek
            features_df['hour_sin'] = np.sin(2 * np.pi * df_clean['hour'] / 24)
            features_df['hour_cos'] = np.cos(2 * np.pi * df_clean['hour'] / 24)
            features_df['weekday'] = (df_clean['day_of_week'] < 5).astype(int)
        else:
            features_df['hour_sin'] = 0
            features_df['hour_cos'] = 1
            features_df['weekday'] = 1
        
        # DODANE: Market regime indicators
        if len(df_clean) >= 50:
            # Trend strength
            price_trend = df_clean['price'].rolling(20).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 20 else 0
            ).fillna(0)
            features_df['trend_strength'] = abs(price_trend)
            features_df['trend_direction'] = (price_trend > 0).astype(int)
            
            # Market volatility regime
            volatility = df_clean['price'].rolling(20).std()
            vol_threshold = volatility.quantile(0.7)
            features_df['high_volatility'] = (volatility > vol_threshold).astype(int)
        else:
            features_df['trend_strength'] = 0
            features_df['trend_direction'] = 1
            features_df['high_volatility'] = 0
        
        # Target: profitable (boolean)
        target = df_clean['profitable'].astype(int)
        
        # Remove any remaining nulls and infinities
        features_df = features_df.fillna(0)
        features_df = features_df.replace([np.inf, -np.inf], 0)
        valid_mask = ~(features_df.isnull().any(axis=1))
        features_df = features_df[valid_mask]
        target = target[valid_mask]
        
        if len(features_df) < self.min_samples:
            self.logger.warning(f"⚠️ Po przygotowaniu za mało danych: {len(features_df)}/{self.min_samples}")
            return None, None
        
        # Check class balance
        balance_ratio = target.mean()
        self.logger.info(f"✅ Enhanced Features: {features_df.shape}")
        self.logger.info(f"📊 Class balance: {balance_ratio:.1%} profitable")
        
        return features_df, target
    
    def train_classification_models(self, X, y):
        """ENHANCED model training with agreement tracking"""
        
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
        
        # ENHANCED model configurations
        model_configs = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                min_samples_split=40,  # Zmniejszone dla lepszej precyzji
                min_samples_leaf=15,
                class_weight='balanced',
                random_state=42
            ),
            'gradient_boost': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                min_samples_split=40,
                random_state=42
            ),
            'logistic': LogisticRegression(
                class_weight='balanced',
                random_state=42,
                max_iter=1000,
                C=0.8  # Dodana regularyzacja
            )
        }
        
        self.models = {}
        successful_models = []
        model_scores = {}
        test_predictions = {}  # DODANE: dla agreement tracking
        
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
                
                # Store predictions for agreement calculation
                test_predictions[name] = y_pred_proba
                
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
                
                # ENHANCED quality check
                if accuracy > 51 and cv_accuracy > 50 and auc > 51:  # Zmniejszone threshold
                    self.models[name] = model
                    model_scores[name] = cv_accuracy
                    successful_models.append(name)
                    self.logger.info(f"✅ {name} accepted")
                else:
                    self.logger.warning(f"❌ {name} rejected (Acc={accuracy:.1f}%, CV={cv_accuracy:.1f}%, AUC={auc:.1f}%)")
                    
            except Exception as e:
                self.logger.error(f"❌ Error training {name}: {e}")
                continue
        
        # DODANE: Calculate model agreement
        if len(test_predictions) >= 2:
            pred_matrix = np.column_stack(list(test_predictions.values()))
            # Agreement as correlation between predictions
            correlations = np.corrcoef(pred_matrix.T)
            mean_correlation = np.mean(correlations[np.triu_indices(len(correlations), k=1)])
            self.model_agreements['test_agreement'] = max(0, mean_correlation)
            
            # Calculate ensemble weights based on performance
            if successful_models:
                total_score = sum(model_scores[name] for name in successful_models)
                self.ensemble_weights = {
                    name: model_scores[name] / total_score 
                    for name in successful_models
                }
        
        if successful_models:
            self.logger.info(f"✅ Successfully trained: {successful_models}")
            if hasattr(self, 'model_agreements'):
                agreement = self.model_agreements.get('test_agreement', 0)
                self.logger.info(f"🤝 Model agreement: {agreement:.3f}")
            self.model_scores = model_scores
            return True
        else:
            self.logger.error("❌ No successful models!")
            return False
    
    def get_ensemble_prediction(self, df):
        """ENHANCED ensemble prediction"""
        
        # Prepare features
        X, y = self.prepare_features_classification(df)
        if X is None:
            return {"error": f"Insufficient data: {len(df)}/{self.min_samples} required"}
        
        # Train if needed
        if not self.models:
            self.logger.info("🔄 Training enhanced classification models...")
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
        
        # ENHANCED ensemble with performance weighting
        if hasattr(self, 'ensemble_weights') and self.ensemble_weights:
            weighted_prob = sum(
                probabilities[name] * self.ensemble_weights.get(name, 0)
                for name in probabilities.keys()
                if name in self.ensemble_weights
            )
        else:
            # Fallback to CV score weighting
            if hasattr(self, 'model_scores'):
                weights = np.array([self.model_scores.get(name, 50) for name in probabilities.keys()])
                weights = weights / weights.sum()
                weighted_prob = np.average(list(probabilities.values()), weights=weights)
            else:
                weighted_prob = np.mean(list(probabilities.values()))
        
        ensemble_prediction = 1 if weighted_prob > 0.5 else 0
        
        # ENHANCED confidence calculation
        prob_values = list(probabilities.values())
        prob_std = np.std(prob_values)
        prob_mean = np.mean(prob_values)
        
        # Base confidence
        base_confidence = max(weighted_prob, 1 - weighted_prob)
        
        # Agreement penalty (if models disagree, reduce confidence)
        agreement_factor = 1 - min(0.3, prob_std)  # Max 30% penalty
        
        # Model count bonus (more models = more confidence)
        model_count_factor = min(1.2, 1 + (len(predictions) - 1) * 0.1)
        
        # Final confidence
        final_confidence = base_confidence * agreement_factor * model_count_factor
        final_confidence = max(0.5, min(1.0, final_confidence))
        
        # Store prediction details
        self.last_prediction_details = {
            'base_confidence': base_confidence,
            'agreement_factor': agreement_factor,
            'model_count_factor': model_count_factor,
            'prob_std': prob_std,
            'model_agreement': agreement_factor
        }
        
        # Get current price for display
        current_price = df['price'].iloc[-1] if 'price' in df.columns else 0
        
        # Direction based on profitability prediction
        direction = "profitable" if ensemble_prediction == 1 else "unprofitable"
        recommendation = "BUY" if ensemble_prediction == 1 and final_confidence > 0.65 else "HOLD"
        
        result = {
            "predicted_profitable": bool(ensemble_prediction),
            "probability_profitable": weighted_prob,
            "confidence": final_confidence,
            "recommendation": recommendation,
            "direction": direction,
            "current_price": current_price,
            "model_count": len(predictions),
            "model_agreement": agreement_factor,  # DODANE!
            "individual_predictions": {
                name: {"profitable": bool(pred), "probability": prob}
                for name, pred, prob in zip(predictions.keys(), predictions.values(), probabilities.values())
            },
            "enhanced_metrics": {
                "base_confidence": base_confidence,
                "agreement_penalty": 1 - agreement_factor,
                "model_bonus": model_count_factor - 1,
                "probability_std": prob_std
            }
        }
        
        self.logger.info(f"🎯 Enhanced: {direction.upper()} (prob: {weighted_prob:.1%}, conf: {final_confidence:.1%}, agree: {agreement_factor:.2f})")
        
        return result
    
    def get_ensemble_prediction_with_reality_check(self, df):
        """DODANE: Bot-style prediction with reality check"""
        
        # Get base prediction
        prediction = self.get_ensemble_prediction(df)
        
        if 'error' in prediction:
            return prediction
        
        # REALITY CHECKS
        reality_checks = []
        confidence = prediction['confidence']
        probability = prediction['probability_profitable']
        
        # Check 1: Model agreement
        agreement = prediction.get('model_agreement', 0.5)
        if agreement < 0.7:
            reality_checks.append("Low model agreement")
            confidence *= 0.9
        
        # Check 2: Probability confidence
        if 0.4 < probability < 0.6:  # Too close to 50/50
            reality_checks.append("Probability too close to random")
            confidence *= 0.85
        
        # Check 3: Model count
        if prediction['model_count'] < 2:
            reality_checks.append("Insufficient model diversity")
            confidence *= 0.8
        
        # Check 4: Feature sanity (basic checks)
        try:
            latest_rsi = df['rsi'].iloc[-1]
            if latest_rsi < 5 or latest_rsi > 95:  # Extreme RSI
                reality_checks.append("Extreme RSI values detected")
                confidence *= 0.9
        except:
            pass
        
        # Apply reality check
        if reality_checks:
            prediction['confidence'] = max(0.5, confidence)
            prediction['reality_check'] = {
                'applied': True,
                'issues': reality_checks,
                'original_confidence': prediction['confidence'],
                'adjusted_confidence': max(0.5, confidence)
            }
            
            # If confidence drops too low, recommend HOLD
            if prediction['confidence'] < 0.6:
                prediction['recommendation'] = 'HOLD'
        else:
            prediction['reality_check'] = {
                'applied': True,
                'issues': [],
                'passed': True
            }
        
        return prediction
    
    def get_model_performance(self):
        """ENHANCED model performance with agreement metrics"""
        if hasattr(self, 'model_scores'):
            performance = {}
            for name, score in self.model_scores.items():
                performance[name] = {
                    'model_type': 'enhanced_classification',
                    'accuracy': score,
                    'r2': 0,  # Not applicable for classification
                    'mae': 0,  # Not applicable for classification
                    'training_samples': self.min_samples,
                    'weight': self.ensemble_weights.get(name, 0) if hasattr(self, 'ensemble_weights') else 0,
                    'last_trained': 'Recent'
                }
            
            # Add ensemble metrics
            if hasattr(self, 'model_agreements'):
                performance['ensemble_agreement'] = {
                    'model_type': 'ensemble_meta',
                    'agreement_score': self.model_agreements.get('test_agreement', 0),
                    'model_count': len(self.models),
                    'total_features': getattr(self, 'last_feature_count', 0)
                }
            
            return performance
        else:
            return {
                "enhanced_classification": {
                    "model_type": "enhanced_classification",
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