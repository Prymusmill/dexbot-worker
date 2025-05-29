# config/settings.py - CONSERVATIVE OPTIMIZATION for better win rate
SETTINGS = {
    # Trading mode
    "mode": "simulation",
    
    # CONSERVATIVE: Slightly reduced frequency for better quality
    "max_trades_per_minute": 40,  # Reduced from 50
    "trades_per_cycle": 35,       # Reduced from 50  
    "cycle_delay_seconds": 45,    # Increased from 30
    
    # Trading parameters
    "trade_amount_usd": 0.02,
    "pair": "SOL/USDC",
    
    # ENHANCED: More conservative strategy
    "strategy": "ml_conservative",    
    "slippage_tolerance": 0.005,
    "risk_limit": 0.02,              # Reduced from 0.03
    
    # Memory and data
    "memory_limit": 3000,            # Increased for more data
    
    # OPTIMIZED: More conservative ML settings
    "ml_enabled": True,              
    "ml_confidence_threshold": 0.6,  # INCREASED from 0.3 - much more selective
    "ml_retrain_hours": 2,           # More frequent retraining from 4h
    "adaptive_trading": True,        
    
    # ENHANCED: Conservative trading controls
    "high_confidence_multiplier": 1.2,  # Reduced from 1.5
    "low_confidence_multiplier": 0.5,   # Reduced from 0.7 - much more conservative  
    "market_volatility_threshold": 0.03, # Reduced from 0.05 - more sensitive
    
    # NEW: Performance-based controls
    "min_win_rate_threshold": 0.45,     # Stop aggressive trading if win rate < 45%
    "model_accuracy_threshold": 0.4,    # Only use models with >40% accuracy
    "max_model_confidence": 0.8,        # Cap confidence at 80% to prevent overconfidence
    "recent_performance_weight": 0.7,   # Weight recent performance heavily
    
    # NEW: Market regime awareness  
    "rsi_overbought_threshold": 75,     # Reduce trading when RSI > 75
    "rsi_oversold_threshold": 25,       # Reduce trading when RSI < 25
    "trend_strength_threshold": 0.02,   # Minimum trend strength for trend trades
}