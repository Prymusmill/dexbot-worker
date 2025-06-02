# ml/claude_analyzer.py - Claude Pro ML Analysis for DexBot
import anthropic
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
import os
import logging


class ClaudeMLAnalyzer:
    """Claude Pro powered ML analysis for trading decisions"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable required")

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.logger = logging.getLogger(__name__)

    def analyze_ml_performance(
            self, performance_data: Dict, recent_trades: pd.DataFrame = None) -> Dict:
        """Deep analysis of ML model performance using Claude"""
        try:
            # Prepare comprehensive data for Claude
            analysis_data = self._prepare_performance_data(
                performance_data, recent_trades)

            prompt = f"""
            I need you to analyze the performance of my crypto trading ML ensemble system. You're an expert quantitative analyst with deep knowledge of machine learning and crypto markets.

            CURRENT ML MODEL PERFORMANCE:
            {json.dumps(analysis_data.get('models', {}), indent=2)}

            RECENT TRADING RESULTS:
            - Total Trades: {analysis_data.get('total_trades', 0):,}
            - Win Rate: {analysis_data.get('win_rate', 0):.1f}%
            - Average PnL: ${analysis_data.get('avg_pnl', 0):.6f}
            - Profitable Trades: {analysis_data.get('profitable_trades', 0):,}
            - Recent Performance Trend: {analysis_data.get('performance_trend', 'Unknown')}

            MARKET CONDITIONS:
            - Current RSI: {analysis_data.get('current_rsi', 50):.1f}
            - Market Volatility: {analysis_data.get('volatility', 0.01):.4f}
            - Price Trend (24h): {analysis_data.get('price_change_24h', 0):+.2f}%
            - Trading Pair: SOL/USDC

            TECHNICAL SETUP:
            - Ensemble Models: RandomForest, GradientBoosting, ExtraTrees, Ridge, ElasticNet
            - Features: 30+ technical indicators, market regime, temporal features
            - Training Data: 31,000+ historical transactions
            - Reality Check System: Active (caps confidence, checks market conditions)

            Please provide a comprehensive analysis including:

            1. **Performance Assessment** (Score 1-10 with reasoning)
            2. **Model Strengths & Weaknesses** (specific to each model type)
            3. **Win Rate Analysis** (47.5% - is this good for crypto trading?)
            4. **Market Regime Impact** (how current conditions affect performance)
            5. **Overfitting Risk Assessment** (31k samples, 5 models)
            6. **Specific Improvement Recommendations** (actionable steps)
            7. **Risk Management Assessment** (current Reality Check effectiveness)
            8. **Parameter Optimization Suggestions** (confidence thresholds, etc.)

            Format your response as structured analysis with clear sections. Be specific and actionable.
            """

            response = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=2000,
                temperature=0.1,  # Low temperature for analytical consistency
                messages=[{"role": "user", "content": prompt}]
            )

            analysis = response.content[0].text

            # Extract key metrics and recommendations
            recommendations = self._extract_recommendations(analysis)
            performance_score = self._extract_performance_score(analysis)

            return {
                'success': True,
                'analysis': analysis,
                'performance_score': performance_score,
                'recommendations': recommendations,
                'timestamp': datetime.now().isoformat(),
                'analyzer': 'claude-3-sonnet'
            }

        except Exception as e:
            self.logger.error(f"Claude ML analysis failed: {e}")
            return {'success': False, 'error': str(e)}

    def analyze_trading_opportunity(
            self, market_data: Dict, ml_prediction: Dict, recent_performance: Dict = None) -> Dict:
        """Analyze specific trading opportunity with Claude"""
        try:
            prompt = f"""
            I need you to analyze a specific crypto trading opportunity. You're an expert crypto trader who understands both technical analysis and machine learning predictions.

            CURRENT MARKET DATA (SOL/USDC):
            - Price: ${market_data.get('price', 0):.4f}
            - RSI: {market_data.get('rsi', 50):.1f}
            - SMA20: ${market_data.get('sma_20', 0):.4f}
            - SMA50: ${market_data.get('sma_50', 0):.4f}
            - Volatility: {market_data.get('volatility', 0.01):.4f}
            - 24h Price Change: {market_data.get('price_change_24h', 0):+.2f}%
            - Bid/Ask Spread: ${market_data.get('spread', 0):.4f}

            ML ENSEMBLE PREDICTION:
            - Predicted Direction: {ml_prediction.get('direction', 'unknown').upper()}
            - Confidence: {ml_prediction.get('confidence', 0):.2f}
            - Predicted Price: ${ml_prediction.get('predicted_price', 0):.4f}
            - Expected Change: {ml_prediction.get('price_change_pct', 0):+.2f}%
            - Model Agreement: {ml_prediction.get('model_agreement', 0):.2f}
            - Active Models: {ml_prediction.get('model_count', 0)}
            - Reality Check Applied: {ml_prediction.get('reality_checked', False)}

            RECENT SYSTEM PERFORMANCE:
            - Win Rate: {recent_performance.get('win_rate', 47.5):.1f}% (last 100 trades)
            - ML Model Accuracy: {recent_performance.get('ml_accuracy', 0):.1f}%

            TRADING PARAMETERS:
            - Trade Size: $0.02 (micro-trading)
            - Max Risk: 1.5% per trade
            - Current Confidence Threshold: 0.75 (very conservative)

            Please analyze this opportunity and provide:

            1. **Trading Recommendation**: BUY/SELL/HOLD with strength (1-10)
            2. **Risk Assessment**: LOW/MEDIUM/HIGH with specific risks
            3. **Confidence Analysis**: Is ML confidence realistic? Any red flags?
            4. **Technical Confirmation**: Do technical indicators support ML prediction?
            5. **Market Regime Check**: How do current conditions affect trade quality?
            6. **Entry/Exit Strategy**: Specific price levels and timing
            7. **Position Sizing**: Recommend adjustment to $0.02 base size
            8. **Stop Loss/Take Profit**: Specific levels based on volatility

            Be conservative and focus on risk management. Format as structured recommendation.
            """

            response = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1200,
                temperature=0.15,
                messages=[{"role": "user", "content": prompt}]
            )

            analysis = response.content[0].text

            # Extract key decision points
            recommendation = self._extract_trading_recommendation(analysis)
            risk_level = self._extract_risk_level(analysis)

            return {
                'success': True,
                'analysis': analysis,
                'recommendation': recommendation,
                'risk_level': risk_level,
                'timestamp': datetime.now().isoformat(),
                'market_price': market_data.get('price', 0),
                'ml_confidence': ml_prediction.get('confidence', 0)
            }

        except Exception as e:
            self.logger.error(f"Claude trading analysis failed: {e}")
            return {'success': False, 'error': str(e)}

    def analyze_anomalies(self, recent_data: pd.DataFrame,
                          ml_predictions: List[Dict]) -> Dict:
        """Detect anomalies in trading data and ML predictions"""
        try:
            # Prepare anomaly data
            anomaly_data = self._prepare_anomaly_data(
                recent_data, ml_predictions)

            prompt = f"""
            I need you to analyze potential anomalies in my crypto trading system. You're an expert in anomaly detection and trading system monitoring.

            RECENT TRADING DATA SUMMARY:
            {json.dumps(anomaly_data, indent=2)}

            Please identify and analyze:

            1. **Price Anomalies**: Unusual price movements or patterns
            2. **ML Prediction Anomalies**: Inconsistent or suspicious predictions
            3. **Performance Anomalies**: Unexpected win/loss patterns
            4. **Technical Indicator Anomalies**: RSI, volatility, or other indicators behaving unusually
            5. **Market Regime Changes**: Shifts that might affect model performance
            6. **Data Quality Issues**: Missing data, outliers, or inconsistencies

            For each anomaly found:
            - Severity: LOW/MEDIUM/HIGH
            - Likely Cause
            - Impact on Trading
            - Recommended Action

            Be thorough and conservative in your assessment.
            """

            response = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1500,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}]
            )

            return {
                'success': True,
                'anomaly_analysis': response.content[0].text,
                'timestamp': datetime.now().isoformat(),
                'data_points_analyzed': len(recent_data) if recent_data is not None else 0
            }

        except Exception as e:
            self.logger.error(f"Claude anomaly analysis failed: {e}")
            return {'success': False, 'error': str(e)}

    def _prepare_performance_data(
            self, performance_data: Dict, recent_trades: pd.DataFrame = None) -> Dict:
        """Prepare comprehensive performance data for analysis"""
        data = {
            'models': performance_data,
            'total_trades': 31574,  # From your logs
            'win_rate': 47.5,  # From your logs
            'profitable_trades': int(31574 * 0.475),
            'avg_pnl': 0.0001,  # Estimated
            'current_rsi': 75.0,  # From recent logs
            'volatility': 0.02,
            'price_change_24h': 0.5,
            'performance_trend': 'stable'
        }

        if recent_trades is not None and len(recent_trades) > 0:
            # Calculate recent performance metrics
            data['recent_win_rate'] = (
                recent_trades['profitable'].sum() / len(recent_trades)) * 100
            data['recent_avg_pnl'] = recent_trades['pnl'].mean(
            ) if 'pnl' in recent_trades.columns else 0

        return data

    def _prepare_anomaly_data(
            self, recent_data: pd.DataFrame, ml_predictions: List[Dict]) -> Dict:
        """Prepare data for anomaly detection analysis"""
        if recent_data is None or len(recent_data) == 0:
            return {'error': 'No recent data available'}

        # Basic statistics
        data = {
            'price_stats': {
                'mean': recent_data['price'].mean() if 'price' in recent_data.columns else 0,
                'std': recent_data['price'].std() if 'price' in recent_data.columns else 0,
                'min': recent_data['price'].min() if 'price' in recent_data.columns else 0,
                'max': recent_data['price'].max() if 'price' in recent_data.columns else 0,
            },
            'rsi_stats': {
                'mean': recent_data['rsi'].mean() if 'rsi' in recent_data.columns else 50,
                'unusual_values': (recent_data['rsi'] >= 99).sum() if 'rsi' in recent_data.columns else 0,
            },
            'ml_predictions_count': len(ml_predictions),
            'recent_trades': len(recent_data)
        }

        return data

    def _extract_recommendations(self, analysis: str) -> List[str]:
        """Extract actionable recommendations from Claude's analysis"""
        recommendations = []
        lines = analysis.split('\n')

        in_recommendations = False
        for line in lines:
            line = line.strip()
            if 'recommendation' in line.lower(
            ) or 'improve' in line.lower() or 'suggest' in line.lower():
                in_recommendations = True
            elif in_recommendations and line and not line.startswith('#'):
                if any(word in line.lower() for word in [
                       'should', 'could', 'recommend', 'consider', 'try']):
                    recommendations.append(line)
                    if len(recommendations) >= 5:
                        break

        return recommendations

    def _extract_performance_score(self, analysis: str) -> Optional[int]:
        """Extract performance score from Claude's analysis"""
        import re
        score_pattern = r'(?:score|rating|assessment).*?(\d+)(?:/10|out of 10)'
        match = re.search(score_pattern, analysis.lower())
        return int(match.group(1)) if match else None

    def _extract_trading_recommendation(self, analysis: str) -> str:
        """Extract trading recommendation from Claude's analysis"""
        analysis_lower = analysis.lower()
        if 'strong buy' in analysis_lower or 'buy' in analysis_lower:
            return 'BUY'
        elif 'strong sell' in analysis_lower or 'sell' in analysis_lower:
            return 'SELL'
        else:
            return 'HOLD'

    def _extract_risk_level(self, analysis: str) -> str:
        """Extract risk level from Claude's analysis"""
        analysis_lower = analysis.lower()
        if 'high risk' in analysis_lower:
            return 'HIGH'
        elif 'medium risk' in analysis_lower:
            return 'MEDIUM'
        elif 'low risk' in analysis_lower:
            return 'LOW'
        else:
            return 'MEDIUM'  # Default

# Integration with existing ML system


class ClaudeEnhancedMLIntegration:
    """Enhanced ML integration with Claude analysis"""

    def __init__(self, anthropic_api_key: str = None):
        from ml.price_predictor import MLTradingIntegration
        self.ml_integration = MLTradingIntegration()

        if anthropic_api_key:
            self.claude_analyzer = ClaudeMLAnalyzer(anthropic_api_key)
        else:
            self.claude_analyzer = None

        self.logger = logging.getLogger(__name__)

    def get_claude_enhanced_prediction(
            self, recent_data: pd.DataFrame, market_data: Dict) -> Dict:
        """Get ML prediction enhanced with Claude analysis"""
        # Get base ML prediction
        ml_prediction = self.ml_integration.get_ensemble_prediction_with_reality_check(
            recent_data)

        if 'error' in ml_prediction or not self.claude_analyzer:
            return ml_prediction

        try:
            # Get Claude's analysis of this trading opportunity
            claude_analysis = self.claude_analyzer.analyze_trading_opportunity(
                market_data, ml_prediction, {
                    'win_rate': 47.5, 'ml_accuracy': 50}
            )

            if claude_analysis.get('success'):
                ml_prediction['claude_analysis'] = claude_analysis

                # Adjust confidence based on Claude's risk assessment
                if claude_analysis.get('risk_level') == 'HIGH':
                    # Reduce confidence for high risk
                    ml_prediction['confidence'] *= 0.7
                elif claude_analysis.get('risk_level') == 'LOW':
                    # Slightly increase for low risk
                    ml_prediction['confidence'] *= 1.1

                self.logger.info(f"🤖 Claude: {claude_analysis.get('recommendation', 'N/A')} "
                                 f"(Risk: {claude_analysis.get('risk_level', 'N/A')})")

        except Exception as e:
            self.logger.error(f"Claude analysis error: {e}")
            ml_prediction['claude_analysis'] = {'error': str(e)}

        return ml_prediction

    def get_performance_insights(self) -> Dict:
        """Get Claude-powered performance insights"""
        if not self.claude_analyzer:
            return {'error': 'Claude analyzer not available'}

        try:
            performance_data = self.ml_integration.get_model_performance()
            return self.claude_analyzer.analyze_ml_performance(
                performance_data)
        except Exception as e:
            return {'error': f'Performance analysis failed: {e}'}

# Usage example


def setup_claude_enhanced_trading():
    """Setup Claude-enhanced trading system"""
    # Set your Anthropic API key in environment or pass directly
    # Set this in your environment
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

    if not ANTHROPIC_API_KEY:
        print("⚠️ ANTHROPIC_API_KEY not set - Claude analysis disabled")
        return None

    enhanced_ml = ClaudeEnhancedMLIntegration(ANTHROPIC_API_KEY)
    print("✅ Claude-enhanced ML system initialized")

    return enhanced_ml
