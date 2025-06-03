# config/settings.py - ENHANCED with Smart 1-Hour ML Retraining
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

    # 🚀 ENHANCED ML SETTINGS - SMART 1H RETRAINING
    "ml_enabled": True,
    "ml_confidence_threshold": 0.75,  # MUCH HIGHER from 0.6 - very selective
    "ml_retrain_hours": 1.0,          # 🚀 NEW: 1 hour instead of 1.5h
    "adaptive_trading": True,

    # 🚀 SMART RETRAINING SYSTEM - NEW!
    "adaptive_retraining": True,                    # Enable smart retraining logic
    "retrain_max_interval_minutes": 60,            # Never wait longer than 1h
    "retrain_min_samples_trigger": 3000,           # Trigger at 3K new samples (~50min)
    "retrain_accuracy_threshold": 0.50,            # Retrain if accuracy drops below 75%
    "retrain_volatility_trigger": 0.05,            # Retrain on 5%+ volatility spike
    "retrain_rsi_extremes": [5, 95],               # Retrain on RSI extremes
    "retrain_smart_scheduling": True,              # Avoid training during high activity
    "retrain_force_on_market_change": True,        # Force retrain on major market shifts

    # 🚀 ADVANCED ML QUALITY CONTROLS - NEW!
    "ml_model_quality_threshold": 0.50,            # Minimum model accuracy to use
    "ml_ensemble_min_agreement": 0.60,             # Minimum model agreement (60%)
    "ml_prediction_confidence_min": 0.65,          # Don't use predictions below 65%
    "ml_reality_check_enabled": True,              # Enable reality checks on predictions
    "ml_outlier_detection": True,                  # Detect and handle outliers
    "ml_feature_importance_tracking": True,        # Track which features matter most

    # 🚀 MULTI-TRIGGER RETRAINING CONDITIONS - NEW!
    "retrain_triggers": {
        "time_based": True,                         # Max 1h interval
        "sample_based": True,                       # 3K samples trigger  
        "performance_based": True,                  # Accuracy drop trigger
        "market_based": True,                       # Volatility/RSI trigger
        "adaptive_based": True,                     # Smart scheduling
    },

    # ULTRA CONSERVATIVE trading controls
    "high_confidence_multiplier": 1.1,  # Reduced from 1.2
    "low_confidence_multiplier": 0.3,   # Much more conservative from 0.5
    "market_volatility_threshold": 0.02,  # More sensitive from 0.03

    # ENHANCED: Stricter performance controls
    "min_win_rate_threshold": 0.5,     # Higher threshold from 0.45
    "model_accuracy_threshold": 0.70,   # 🚀 INCREASED: Higher from 0.4 - only use good models
    "max_model_confidence": 0.85,       # 🚀 INCREASED: Higher cap from 0.75
    "recent_performance_weight": 0.8,   # Weight recent performance more heavily

    # ENHANCED: Stricter market regime awareness
    "rsi_overbought_threshold": 70,     # More sensitive from 75
    "rsi_oversold_threshold": 30,       # More sensitive from 25
    "trend_strength_threshold": 0.015,  # Lower from 0.02 - stricter trend requirement

    # 🚀 CONTRARIAN TRADING SETTINGS - NEW!
    "contrarian_trading_enabled": True,             # Enable contrarian logic
    "contrarian_ml_confidence_threshold": 0.85,     # ML confidence needed for contrarian
    "contrarian_rsi_extreme_threshold": [10, 90],   # RSI levels for contrarian trades
    "contrarian_score_minimum": 0.3,               # Minimum contrarian score
    "contrarian_probability_multiplier": 1.5,       # Boost contrarian probability

    # ERROR TOLERANCE SETTINGS
    "feature_engineering_retries": 3,   # Retry feature engineering on failure
    "model_training_retries": 2,        # Retry model training on failure
    "prediction_fallback_confidence": 0.3,  # Fallback confidence when ML fails
    "min_data_quality_threshold": 0.8,  # Require 80% valid data
    "max_feature_correlation": 0.95,    # Remove highly correlated features

    # DATA VALIDATION SETTINGS
    "min_training_samples": 150,        # Minimum samples for training
    "max_feature_missing_rate": 0.1,    # Max 10% missing values per feature
    "price_change_outlier_threshold": 0.5,  # Remove extreme price changes
    "volume_outlier_threshold": 10,     # Remove extreme volume outliers

    # 🚀 PERFORMANCE MONITORING - NEW!
    "performance_tracking": {
        "track_ml_accuracy_trend": True,            # Monitor accuracy over time
        "track_prediction_quality": True,          # Monitor prediction quality
        "track_feature_importance": True,          # Monitor which features work best
        "track_contrarian_performance": True,      # Monitor contrarian trade success
        "alert_on_accuracy_drop": True,           # Alert when accuracy drops
        "alert_threshold": 0.65,                  # Alert below 65% accuracy
    },

    # 🚀 ADAPTIVE CYCLE MANAGEMENT - NEW!
    "adaptive_cycle_enabled": True,               # Enable adaptive cycle sizing
    "cycle_size_base": 25,                       # Base cycle size
    "cycle_size_ml_multiplier": 1.2,             # Increase with good ML
    "cycle_size_contrarian_multiplier": 0.8,     # Reduce for contrarian periods
    "cycle_delay_volatility_adjustment": True,    # Adjust delay based on volatility

    # 🚀 MARKET CONDITION DETECTION - NEW!
    "market_regime_detection": {
        "enabled": True,                          # Enable market regime detection
        "bull_market_threshold": 0.02,           # 2%+ daily gains
        "bear_market_threshold": -0.02,          # 2%+ daily losses  
        "sideways_volatility_max": 0.01,         # Max volatility for sideways
        "regime_lookback_hours": 24,             # Hours to analyze for regime
    },

    # 🚀 RISK MANAGEMENT ENHANCEMENTS - NEW!
    "enhanced_risk_management": {
        "max_consecutive_losses": 5,             # Stop after 5 consecutive losses
        "drawdown_protection_threshold": 0.05,  # Stop if 5% drawdown
        "volatility_position_sizing": True,     # Reduce size in high volatility
        "correlation_risk_limit": 0.8,          # Limit correlated positions
    },

    # 🚀 ADVANCED LOGGING & MONITORING - NEW!
    "advanced_logging": {
        "log_ml_decisions": True,                # Log all ML decisions
        "log_contrarian_triggers": True,        # Log contrarian trade triggers
        "log_retrain_events": True,             # Log when/why retraining happens
        "log_performance_metrics": True,        # Log detailed performance metrics
        "log_level": "INFO",                    # Logging level
    },

    # LEGACY COMPATIBILITY
    "ml_min_samples": 3000,                     # Legacy parameter for old code
    "ml_quality_threshold": 0.75,              # Legacy parameter mapping
}