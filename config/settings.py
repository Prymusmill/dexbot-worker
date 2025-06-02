# config/settings.py - ULTRA CONSERVATIVE with Error Tolerance
SETTINGS = {
    # Trading mode
    "mode": "simulation",

    # ULTRA CONSERVATIVE: Reduced frequency for better stability
    "max_trades_per_minute": 30,  # Reduced from 40
    "trades_per_cycle": 25,       # Reduced from 35
    "cycle_delay_seconds": 60,    # Increased from 45 - more time between cycles

    # Trading parameters
    "trade_amount_usd": 0.02,
    "pair": "SOL/USDC",

    # ENHANCED: Ultra conservative strategy
    "strategy": "ml_ultra_conservative",
    "slippage_tolerance": 0.003,    # Reduced slippage tolerance
    "risk_limit": 0.015,            # Reduced from 0.02

    # Memory and data
    "memory_limit": 5000,           # Increased for more ML data

    # ULTRA CONSERVATIVE ML settings
    "ml_enabled": True,
    "ml_confidence_threshold": 0.75,  # MUCH HIGHER from 0.6 - very selective
    "ml_retrain_hours": 1.5,          # More frequent retraining
    "adaptive_trading": True,

    # ULTRA CONSERVATIVE trading controls
    "high_confidence_multiplier": 1.1,  # Reduced from 1.2
    "low_confidence_multiplier": 0.3,   # Much more conservative from 0.5
    "market_volatility_threshold": 0.02,  # More sensitive from 0.03

    # ENHANCED: Stricter performance controls
    "min_win_rate_threshold": 0.5,     # Higher threshold from 0.45
    "model_accuracy_threshold": 0.5,    # Higher from 0.4 - only use good models
    "max_model_confidence": 0.75,       # Lower cap from 0.8
    "recent_performance_weight": 0.8,   # Weight recent performance more heavily

    # ENHANCED: Stricter market regime awareness
    "rsi_overbought_threshold": 70,     # More sensitive from 75
    "rsi_oversold_threshold": 30,       # More sensitive from 25
    "trend_strength_threshold": 0.015,  # Lower from 0.02 - stricter trend requirement

    # NEW: Error tolerance settings
    "feature_engineering_retries": 3,   # Retry feature engineering on failure
    "model_training_retries": 2,        # Retry model training on failure
    "prediction_fallback_confidence": 0.3,  # Fallback confidence when ML fails
    "min_data_quality_threshold": 0.8,  # Require 80% valid data
    "max_feature_correlation": 0.95,    # Remove highly correlated features

    # NEW: Data validation settings
    "min_training_samples": 150,        # Minimum samples for training
    "max_feature_missing_rate": 0.1,    # Max 10% missing values per feature
    "price_change_outlier_threshold": 0.5,  # Remove extreme price changes
    "volume_outlier_threshold": 10,     # Remove extreme volume outliers
}
