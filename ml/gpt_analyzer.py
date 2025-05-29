import openai
import os
from typing import Dict, List, Optional
import json

class GPTTradingAnalyzer:
    def __init__(self):
        self.client = openai.OpenAI(
            api_key=os.getenv('OPENAI_API_KEY')
        )
    
    def analyze_market_data(self, market_data: Dict) -> Dict:
        try:
            prompt = f"""
            Analyze this trading data and provide recommendations:
            
            Market Data: {json.dumps(market_data, indent=2)}
            
            Provide analysis in JSON format:
            {{
                "action": "BUY/SELL/HOLD",
                "confidence": 0.0-1.0,
                "reasoning": "explanation",
                "risk_level": "LOW/MEDIUM/HIGH"
            }}
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4",  # lub gpt-3.5-turbo dla tańszej opcji
                messages=[
                    {"role": "system", "content": "You are an expert trading analyst."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            print(f"GPT Analysis error: {e}")
            return {"action": "HOLD", "confidence": 0.5, "reasoning": "Analysis failed"}

def setup_gpt_enhanced_trading():
    return GPTTradingAnalyzer()