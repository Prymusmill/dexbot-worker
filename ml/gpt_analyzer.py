import openai
import os
from typing import Dict, List, Optional
import json
from datetime import datetime
import json
from datetime import datetime

def safe_json_dumps(data):
    """Safely serialize data to JSON, handling datetime objects"""
    def datetime_handler(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    return json.dumps(data, default=datetime_handler, indent=2)

class GPTTradingAnalyzer:
    def __init__(self):
        self.client = openai.OpenAI(
            api_key=os.getenv('OPENAI_API_KEY')
        )
        self.analysis_count = 0
        self.successful_analyses = 0
        self.model_used = "gpt-4"
        
    def analyze_market_data(self, market_data: Dict) -> Dict:
        """Basic market data analysis"""
        try:
            self.analysis_count += 1
            
            prompt = f"""
            Analyze this trading data and provide recommendations:
            
            Market Data: {safe_json_dumps(market_data)}
            
            Provide analysis in JSON format:
            {{
                "action": "BUY/SELL/HOLD",
                "confidence": 0.0-1.0,
                "reasoning": "explanation",
                "risk_level": "LOW/MEDIUM/HIGH",
                "market_outlook": "BULLISH/BEARISH/NEUTRAL"
            }}
            """
            
            response = self.client.chat.completions.create(
                model=self.model_used,
                messages=[
                    {"role": "system", "content": "You are an expert cryptocurrency trading analyst specializing in SOL/USDT trading."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            result = json.loads(response.choices[0].message.content)
            result['timestamp'] = datetime.now().isoformat()
            self.successful_analyses += 1
            
            return result
            
        except Exception as e:
            print(f"GPT Analysis error: {e}")
            return {
                "action": "HOLD", 
                "confidence": 0.5, 
                "reasoning": "Analysis failed due to error",
                "risk_level": "MEDIUM",
                "market_outlook": "NEUTRAL",
                "error": str(e)
            }
    
    def analyze_market_signal(self, market_data: Dict, ml_predictions: List[Dict] = None, current_position: Dict = None) -> Dict:
        """Enhanced market signal analysis with ML predictions and position data"""
        try:
            self.analysis_count += 1
            
            # Build comprehensive prompt
            prompt_parts = [
                "Analyze this comprehensive trading scenario and provide detailed recommendations:",
                "",
                f"Current Market Data: {safe_json_dumps(market_data)}"
            ]
            
            if ml_predictions:
                prompt_parts.extend([
                    "",
                    f"ML Model Predictions: {safe_json_dumps(ml_predictions)}"
                ])
            
            if current_position:
                prompt_parts.extend([
                    "",
                    f"Current Position: {safe_json_dumps(current_position)}"
                ])
            
            prompt_parts.extend([
                "",
                "Consider all available data and provide analysis in JSON format:",
                "{",
                '  "action": "BUY/SELL/HOLD",',
                '  "confidence": 0.0-1.0,',
                '  "reasoning": "detailed explanation of decision",',
                '  "risk_level": "LOW/MEDIUM/HIGH",',
                '  "market_outlook": "BULLISH/BEARISH/NEUTRAL",',
                '  "ml_agreement": "HIGH/MEDIUM/LOW/NONE",',
                '  "position_advice": "specific advice about current position",',
                '  "key_factors": ["factor1", "factor2", "factor3"]',
                "}"
            ])
            
            full_prompt = "\n".join(prompt_parts)
            
            response = self.client.chat.completions.create(
                model=self.model_used,
                messages=[
                    {
                        "role": "system", 
                        "content": """You are an expert cryptocurrency trading analyst with deep knowledge of:
                        - Technical analysis and market indicators
                        - Machine learning prediction interpretation
                        - Risk management and position sizing
                        - SOL/USDT market dynamics
                        
                        Always provide actionable, well-reasoned trading advice based on all available data."""
                    },
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.2,
                max_tokens=500
            )
            
            result = json.loads(response.choices[0].message.content)
            result['timestamp'] = datetime.now().isoformat()
            result['analysis_id'] = self.analysis_count
            self.successful_analyses += 1
            
            return result
            
        except Exception as e:
            print(f"GPT Market Signal Analysis error: {e}")
            return {
                "action": "HOLD",
                "confidence": 0.3,
                "reasoning": f"Analysis failed: {str(e)}",
                "risk_level": "HIGH",
                "market_outlook": "UNCERTAIN",
                "ml_agreement": "NONE",
                "position_advice": "Maintain current position due to analysis error",
                "key_factors": ["analysis_error"],
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "analysis_id": self.analysis_count
            }
    
    def get_performance_stats(self) -> Dict:
        """Get GPT analyzer performance statistics"""
        success_rate = (self.successful_analyses / self.analysis_count * 100) if self.analysis_count > 0 else 0
        
        return {
            'total_analyses': self.analysis_count,
            'successful_analyses': self.successful_analyses,
            'success_rate': f"{success_rate:.1f}%",
            'model_used': self.model_used,
            'error_rate': f"{100 - success_rate:.1f}%"
        }
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.analysis_count = 0
        self.successful_analyses = 0
    
    def set_model(self, model_name: str):
        """Change GPT model (gpt-4, gpt-3.5-turbo, etc.)"""
        self.model_used = model_name
        print(f"GPT model changed to: {model_name}")

def setup_gpt_enhanced_trading():
    """Initialize GPT trading analyzer"""
    try:
        analyzer = GPTTradingAnalyzer()
        # Test connection
        test_data = {"price": 100, "rsi": 50, "volume": 1000}
        test_result = analyzer.analyze_market_data(test_data)
        
        if not test_result.get('error'):
            print("✅ GPT analyzer initialized and tested successfully")
            return analyzer
        else:
            print(f"⚠️ GPT analyzer test failed: {test_result.get('error')}")
            return analyzer
            
    except Exception as e:
        print(f"❌ GPT analyzer initialization failed: {e}")
        return None

def format_gpt_analysis_for_logging(analysis: Dict) -> str:
    """Format GPT analysis results for console logging"""
    if not analysis:
        return "❌ GPT Analysis: No data"
    
    if analysis.get('error'):
        return f"❌ GPT Analysis Failed: {analysis.get('reasoning', 'Unknown error')}"
    
    action = analysis.get('action', 'UNKNOWN')
    confidence = analysis.get('confidence', 0)
    risk_level = analysis.get('risk_level', 'UNKNOWN')
    market_outlook = analysis.get('market_outlook', 'NEUTRAL')
    reasoning = analysis.get('reasoning', 'No reasoning provided')
    
    # Truncate reasoning for logging
    short_reasoning = reasoning[:80] + "..." if len(reasoning) > 80 else reasoning
    
    # Create formatted log message
    log_message = f"🤖 GPT Analysis #{analysis.get('analysis_id', '?')}:\n"
    log_message += f"   • Action: {action} (confidence: {confidence:.2f})\n"
    log_message += f"   • Risk: {risk_level} | Outlook: {market_outlook}\n"
    log_message += f"   • Reasoning: {short_reasoning}"
    
    # Add ML agreement if available
    if 'ml_agreement' in analysis and analysis['ml_agreement'] != 'NONE':
        log_message += f"\n   • ML Agreement: {analysis['ml_agreement']}"
    
    # Add key factors if available
    if 'key_factors' in analysis and analysis['key_factors']:
        factors = ', '.join(analysis['key_factors'][:3])  # Show first 3 factors
        log_message += f"\n   • Key Factors: {factors}"
    
    return log_message

def test_gpt_analyzer():
    """Test function for GPT analyzer"""
    print("🧪 Testing GPT Analyzer...")
    
    analyzer = setup_gpt_enhanced_trading()
    if not analyzer:
        print("❌ GPT Analyzer setup failed")
        return False
    
    # Test data
    test_market_data = {
        "price": 147.85,
        "rsi": 62.5,
        "volume": 1500000,
        "price_change_24h": 2.15,
        "volatility": 0.035
    }
    
    test_ml_predictions = [
        {
            "model_name": "RandomForest",
            "prediction": "BUY",
            "confidence": 0.72,
            "reasoning": "Strong upward momentum detected"
        }
    ]
    
    # Run test
    result = analyzer.analyze_market_signal(
        market_data=test_market_data,
        ml_predictions=test_ml_predictions
    )
    
    print("📊 Test Result:")
    print(format_gpt_analysis_for_logging(result))
    
    # Show stats
    stats = analyzer.get_performance_stats()
    print(f"\n📈 Performance: {stats}")
    
    return True

if __name__ == "__main__":
    # Run test if executed directly
    test_gpt_analyzer()