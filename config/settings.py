# ZASTĄP W config/settings.py - AGGRESSIVE TRADING SETTINGS

SETTINGS = {
    # Trading mode
    "mode": "simulation",

    # AGGRESSIVE: Increased trading frequency 
    "max_trades_per_minute": 40,  # Increased from 30
    "trades_per_cycle": 30,       # Increased from 25
    "cycle_delay_seconds": 45,    # Decreased from 60 - faster cycles

    # Trading parameters
    "trade_amount_usd": 0.02,
    "pair": "SOL/USDC",

    # AGGRESSIVE: Less conservative strategy
    "strategy": "ml_aggressive_contrarian",  # NEW STRATEGY!
    "slippage_tolerance": 0.003,
    "risk_limit": 0.02,  # Increased from 0.015

    # Memory and data
    "memory_limit": 5000,

    # 🎯 AGGRESSIVE ML SETTINGS - MAJOR CHANGES!
    "ml_enabled": True,
    "ml_confidence_threshold": 0.5,   # LOWERED from 0.75 - MORE TRADES!
    "ml_retrain_hours": 1.5,
    "adaptive_trading": True,

    # 🚀 AGGRESSIVE TRADING CONTROLS
    "high_confidence_multiplier": 1.3,  # Increased from 1.1
    "low_confidence_multiplier": 0.5,   # Increased from 0.3
    "market_volatility_threshold": 0.03, # Lowered from 0.02 - trade in higher vol

    # 🎯 CONTRARIAN TRADING SETTINGS - NEW!
    "contrarian_enabled": True,          # Enable contrarian logic
    "contrarian_rsi_extreme": 95,        # RSI threshold for extreme contrarian
    "contrarian_rsi_high": 85,           # RSI threshold for moderate contrarian  
    "contrarian_ml_confidence": 0.85,    # ML confidence for contrarian trigger
    "contrarian_score_threshold": 0.5,   # Minimum score for contrarian trade

    # ENHANCED: More aggressive performance controls
    "min_win_rate_threshold": 0.45,     # Lowered from 0.5
    "model_accuracy_threshold": 0.45,   # Lowered from 0.5 - use more models
    "max_model_confidence": 0.85,       # Increased from 0.75
    "recent_performance_weight": 0.7,   # Decreased from 0.8

    # ENHANCED: More sensitive market regime awareness  
    "rsi_overbought_threshold": 80,     # Increased from 70 - allow more overbought trades
    "rsi_oversold_threshold": 20,       # Decreased from 30 - allow more oversold trades
    "trend_strength_threshold": 0.01,   # Lowered from 0.015 - more sensitive

    # 🎯 FALLBACK & PROBABILITY SETTINGS - NEW!
    "fallback_trade_probability": 0.6,  # Increased from 0.4
    "hold_trade_probability": 0.6,      # Increased from 0.4  
    "low_confidence_probability": 0.6,  # Increased from 0.3
    "contrarian_probability_boost": 1.5, # Multiply contrarian score by this

    # ERROR TOLERANCE SETTINGS (unchanged)
    "feature_engineering_retries": 3,
    "model_training_retries": 2,
    "prediction_fallback_confidence": 0.3,
    "min_data_quality_threshold": 0.8,
    "max_feature_correlation": 0.95,

    # DATA VALIDATION SETTINGS (unchanged)
    "min_training_samples": 150,
    "max_feature_missing_rate": 0.1,
    "price_change_outlier_threshold": 0.5,
    "volume_outlier_threshold": 10,
}