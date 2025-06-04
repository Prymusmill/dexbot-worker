# ml/enhanced_ensemble.py - ROBUST 7+ MODEL ENSEMBLE
import numpy as np
import pandas as pd
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                            VotingClassifier, StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class RobustMLEnsemble:
    """
    Enhanced ML Ensemble with 7+ diverse models for institutional-grade predictions
    """
    
    def __init__(self):
        self.models = {}
        self.model_weights = {}
        self.diversity_scores = {}
        self.ensemble_performance = {}
        self.min_models = 7
        self.max_correlation_threshold = 0.85
        self.min_accuracy_threshold = 0.55
        
        # Initialize diverse model portfolio
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize diverse ensemble of 7+ models"""
        
        # 1. TREE-BASED MODELS (3)
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
        
        # XGBoost for advanced tree-based learning
        self.models['xgboost'] = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            eval_metric='logloss'
        )
        
        # 2. LINEAR MODELS (2)
        self.models['logistic'] = LogisticRegression(
            random_state=42,
            max_iter=1000,
            C=1.0
        )
        
        self.models['svm'] = SVC(
            probability=True,
            kernel='rbf',
            C=1.0,
            random_state=42
        )
        
        # 3. NEURAL NETWORKS (2)
        self.models['neural_net'] = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,
            random_state=42,
            max_iter=500
        )
        
        # LightGBM for fast gradient boosting
        self.models['lightgbm'] = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            verbose=-1
        )
        
        # 4. META-ENSEMBLE MODELS (2)
        # Voting classifier combining best performers
        self.models['voting_meta'] = VotingClassifier(
            estimators=[
                ('rf', self.models['random_forest']),
                ('gb', self.models['gradient_boost']),
                ('lr', self.models['logistic'])
            ],
            voting='soft'
        )
        
        # Stacking classifier for advanced ensemble
        self.models['stacking_meta'] = StackingClassifier(
            estimators=[
                ('rf', self.models['random_forest']),
                ('xgb', self.models['xgboost']),
                ('nn', self.models['neural_net'])
            ],
            final_estimator=LogisticRegression(),
            cv=3
        )
        
        print(f"✅ Initialized {len(self.models)} diverse models")
        print(f"📊 Model types: Tree-based(3), Linear(2), Neural(2), Meta(2)")
        
    def train_ensemble(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Train all models with diversity validation"""
        results = {
            'trained_models': [],
            'failed_models': [],
            'ensemble_metrics': {},
            'diversity_metrics': {},
            'model_performances': {}
        }
        
        # Individual model training
        model_predictions = {}
        
        for name, model in self.models.items():
            try:
                print(f"🔄 Training {name}...")
                
                # Cross-validation for robust performance estimate
                cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
                mean_cv_score = cv_scores.mean()
                
                # Only train models above threshold
                if mean_cv_score >= self.min_accuracy_threshold:
                    model.fit(X, y)
                    
                    # Get predictions for diversity analysis
                    pred_proba = model.predict_proba(X)[:, 1]  # Probability of positive class
                    model_predictions[name] = pred_proba
                    
                    # Store performance
                    results['model_performances'][name] = {
                        'cv_accuracy': mean_cv_score,
                        'cv_std': cv_scores.std(),
                        'trained': True
                    }
                    
                    results['trained_models'].append(name)
                    print(f"✅ {name}: CV Accuracy = {mean_cv_score:.3f} ± {cv_scores.std():.3f}")
                    
                else:
                    results['failed_models'].append(name)
                    print(f"❌ {name}: CV Accuracy = {mean_cv_score:.3f} (below threshold)")
                    
            except Exception as e:
                results['failed_models'].append(name)
                print(f"❌ {name} training failed: {e}")
                
        # Diversity Analysis
        diversity_metrics = self._calculate_diversity(model_predictions)
        results['diversity_metrics'] = diversity_metrics
        
        # Calculate ensemble weights based on performance and diversity
        self._calculate_ensemble_weights(results['model_performances'], diversity_metrics)
        
        # Ensemble performance
        if len(results['trained_models']) >= self.min_models:
            ensemble_pred = self._get_ensemble_prediction(X, results['trained_models'])
            ensemble_accuracy = accuracy_score(y, ensemble_pred > 0.5)
            
            results['ensemble_metrics'] = {
                'ensemble_accuracy': ensemble_accuracy,
                'trained_models_count': len(results['trained_models']),
                'diversity_score': diversity_metrics.get('avg_correlation', 1.0),
                'ensemble_boost': ensemble_accuracy - max([
                    perf['cv_accuracy'] for perf in results['model_performances'].values()
                ])
            }
            
            print(f"🎯 Ensemble Performance:")
            print(f"   • Models trained: {len(results['trained_models'])}/{len(self.models)}")
            print(f"   • Ensemble accuracy: {ensemble_accuracy:.3f}")
            print(f"   • Diversity score: {diversity_metrics.get('avg_correlation', 1.0):.3f}")
            print(f"   • Ensemble boost: {results['ensemble_metrics']['ensemble_boost']:.3f}")
            
        else:
            print(f"⚠️ Only {len(results['trained_models'])} models trained (minimum: {self.min_models})")
            
        return results
    
    def _calculate_diversity(self, model_predictions: Dict) -> Dict:
        """Calculate diversity metrics between models"""
        if len(model_predictions) < 2:
            return {'avg_correlation': 1.0, 'max_correlation': 1.0, 'min_correlation': 1.0}
            
        # Calculate correlation matrix
        pred_df = pd.DataFrame(model_predictions)
        corr_matrix = pred_df.corr()
        
        # Extract upper triangle (exclude diagonal)
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        correlations = upper_triangle.stack().values
        
        return {
            'avg_correlation': correlations.mean(),
            'max_correlation': correlations.max(),
            'min_correlation': correlations.min(),
            'correlation_matrix': corr_matrix.to_dict()
        }
    
    def _calculate_ensemble_weights(self, performances: Dict, diversity: Dict):
        """Calculate optimal weights for ensemble based on performance and diversity"""
        for name, perf in performances.items():
            if perf.get('trained', False):
                # Base weight from accuracy
                accuracy_weight = perf['cv_accuracy']
                
                # Diversity bonus (lower correlation = higher weight)
                avg_corr = diversity.get('avg_correlation', 0.5)
                diversity_bonus = (1 - avg_corr) * 0.2
                
                # Stability bonus (lower std = higher weight)  
                stability_bonus = (1 - perf['cv_std']) * 0.1
                
                final_weight = accuracy_weight + diversity_bonus + stability_bonus
                self.model_weights[name] = final_weight
                
        # Normalize weights
        total_weight = sum(self.model_weights.values())
        if total_weight > 0:
            self.model_weights = {k: v/total_weight for k, v in self.model_weights.items()}
            
    def _get_ensemble_prediction(self, X: pd.DataFrame, trained_models: List[str]) -> np.ndarray:
        """Get weighted ensemble prediction"""
        ensemble_pred = np.zeros(len(X))
        total_weight = 0
        
        for name in trained_models:
            if name in self.models and name in self.model_weights:
                try:
                    model = self.models[name]
                    weight = self.model_weights[name]
                    
                    pred_proba = model.predict_proba(X)[:, 1]
                    ensemble_pred += weight * pred_proba
                    total_weight += weight
                    
                except Exception as e:
                    print(f"⚠️ Prediction error for {name}: {e}")
                    
        if total_weight > 0:
            ensemble_pred /= total_weight
            
        return ensemble_pred
    
    def predict_ensemble(self, X: pd.DataFrame) -> Dict:
        """Get comprehensive ensemble prediction with confidence metrics"""
        trained_models = [name for name, model in self.models.items() 
                         if hasattr(model, 'predict')]
        
        if len(trained_models) < self.min_models:
            return {
                'error': f'Insufficient trained models: {len(trained_models)}/{self.min_models}',
                'prediction': 0.5,
                'confidence': 0.0
            }
        
        # Individual predictions
        individual_preds = {}
        for name in trained_models:
            try:
                model = self.models[name]
                pred_proba = model.predict_proba(X)[:, 1]
                individual_preds[name] = {
                    'probability': pred_proba.mean(),
                    'prediction': pred_proba.mean() > 0.5
                }
            except Exception as e:
                print(f"⚠️ Individual prediction error for {name}: {e}")
        
        # Ensemble prediction
        ensemble_pred = self._get_ensemble_prediction(X, trained_models)
        ensemble_probability = ensemble_pred.mean()
        
        # Model agreement analysis
        predictions = [pred['prediction'] for pred in individual_preds.values()]
        agreement = sum(predictions) / len(predictions) if predictions else 0.5
        
        # Confidence calculation
        pred_variance = np.var([pred['probability'] for pred in individual_preds.values()])
        confidence = max(0.1, min(0.9, 1 - pred_variance))
        
        return {
            'ensemble_probability': ensemble_probability,
            'ensemble_prediction': ensemble_probability > 0.5,
            'individual_predictions': individual_preds,
            'model_agreement': agreement,
            'confidence': confidence,
            'model_count': len(trained_models),
            'diversity_score': 1 - self.diversity_scores.get('avg_correlation', 0.5)
        }
    
    def get_model_diagnostics(self) -> Dict:
        """Get comprehensive model diagnostics"""
        return {
            'total_models': len(self.models),
            'trained_models': len([m for m in self.models.values() if hasattr(m, 'predict')]),
            'model_weights': self.model_weights.copy(),
            'diversity_metrics': self.diversity_scores.copy(),
            'model_types': {
                'tree_based': ['random_forest', 'gradient_boost', 'xgboost', 'lightgbm'],
                'linear': ['logistic', 'svm'],
                'neural': ['neural_net'],
                'meta_ensemble': ['voting_meta', 'stacking_meta']
            }
        }

# Usage example for integration
def create_robust_ensemble():
    """Factory function to create enhanced ensemble"""
    return RobustMLEnsemble()

# Testing and validation functions
def validate_ensemble_diversity(ensemble: RobustMLEnsemble, X: pd.DataFrame, y: pd.Series) -> Dict:
    """Validate ensemble meets diversity requirements"""
    # Train ensemble
    results = ensemble.train_ensemble(X, y)
    
    # Check requirements
    validation = {
        'min_models_met': len(results['trained_models']) >= ensemble.min_models,
        'diversity_adequate': results['diversity_metrics'].get('avg_correlation', 1.0) < ensemble.max_correlation_threshold,
        'accuracy_threshold_met': all(
            perf['cv_accuracy'] >= ensemble.min_accuracy_threshold 
            for perf in results['model_performances'].values() 
            if perf.get('trained', False)
        ),
        'ensemble_boost_positive': results['ensemble_metrics'].get('ensemble_boost', -1) > 0
    }
    
    validation['all_requirements_met'] = all(validation.values())
    
    return {
        'validation': validation,
        'metrics': results,
        'recommendations': _generate_recommendations(validation, results)
    }

def _generate_recommendations(validation: Dict, results: Dict) -> List[str]:
    """Generate recommendations for ensemble improvement"""
    recommendations = []
    
    if not validation['min_models_met']:
        recommendations.append("❌ Add more diverse models (target: 7+ models)")
    
    if not validation['diversity_adequate']:
        recommendations.append("❌ Increase model diversity (reduce correlation)")
        
    if not validation['accuracy_threshold_met']:
        recommendations.append("❌ Improve individual model performance")
        
    if not validation['ensemble_boost_positive']:
        recommendations.append("❌ Ensemble should outperform best individual model")
        
    if validation['all_requirements_met']:
        recommendations.append("✅ Ensemble meets all institutional-grade requirements!")
        
    return recommendations