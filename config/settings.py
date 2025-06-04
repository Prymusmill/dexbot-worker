# config/settings.py - ENHANCED MULTI-ASSET SETTINGS
SETTINGS = {
    # Trading mode
    "mode": "simulation",

    # ENHANCED: Multi-asset trading parameters
    "multi_asset_enabled": True,
    "supported_assets": ["SOL", "ETH", "BTC"],
    "default_asset": "SOL",
    
    # Portfolio allocation (must sum to 1.0)
    "portfolio_allocation": {
        "SOL": 0.40,   # 40%
        "ETH": 0.35,   # 35% 
        "BTC": 0.25    # 25%
    },
    
    # Asset switching parameters
    "asset_switch_threshold": 0.8,      # Confidence needed to override portfolio allocation
    "rebalance_threshold": 0.15,        # 15% deviation triggers rebalancing
    "min_trades_before_switch": 5,      # Minimum trades before considering asset switch

    # CONSERVATIVE: Core trading parameters
    "max_trades_per_minute": 25,        # Reduced from 30
    "trades_per_cycle": 30,             # Reduced from 50 for stability
    "cycle_delay_seconds": 45,          # Increased for more conservative approach

    # Trading execution
    "trade_amount_usd": 0.02,
    "pair": "MULTI-ASSET",              # Updated for multi-asset
    "slippage_tolerance": 0.003,
    "risk_limit": 0.015,

    # Memory and data management
    "memory_limit": 10000,              # Increased for multi-asset data

    # ENHANCED: ML Configuration
    "ml_enabled": True,
    "ml_confidence_threshold": 0.65,    # Lowered from 0.75 for more activity
    "ml_retrain_hours": 2.0,            # More frequent retraining
    "adaptive_trading": True,
    
    # Enhanced ML training parameters
    "ml_min_samples": 100,              # Reduced from 500 for faster startup
    "ml_max_features": 15,              # Limit features to prevent overfitting
    "retrain_min_samples_trigger": 200, # Trigger retraining after 200 new samples
    "retrain_accuracy_threshold": 0.55, # Retrain if accuracy drops below 55%
    "ml_prediction_timeout": 30,        # Timeout for ML predictions (seconds)

    # ENHANCED: Contrarian trading
    "contrarian_enabled": True,
    "contrarian_threshold": 0.3,        # Minimum score to consider contrarian trade
    "contrarian_max_percentage": 0.2,   # Max 20% of trades can be contrarian
    "extreme_rsi_threshold": 90,        # RSI level for extreme conditions
    
    # RSI thresholds for contrarian analysis
    "rsi_extreme_overbought": 95,
    "rsi_extreme_oversold": 5,
    "rsi_strong_overbought": 85,
    "rsi_strong_oversold": 15,

    # ENHANCED: Adaptive trading controls
    "high_confidence_multiplier": 1.2,  # Slightly reduced from 1.5
    "low_confidence_multiplier": 0.7,   # More conservative
    "market_volatility_threshold": 0.04, # More sensitive to volatility
    
    # Performance-based adaptive parameters
    "performance_adjustment_window": 5,  # Look at last 5 cycles for performance
    "good_performance_threshold": 0.6,   # 60%+ win rate = good
    "poor_performance_threshold": 0.4,   # <40% win rate = poor
    "performance_boost_factor": 1.15,    # Boost trading when performing well
    "performance_reduce_factor": 0.85,   # Reduce trading when performing poorly

    # ENHANCED: Risk management
    "max_consecutive_losses": 3,         # Stop after 3 consecutive losses
    "daily_loss_limit": 0.05,           # Stop if daily losses exceed 5%
    "position_size_scaling": True,       # Scale position size based on confidence
    "max_position_multiplier": 1.5,     # Max 1.5x normal position size
    "min_position_multiplier": 0.5,     # Min 0.5x normal position size

    # Market condition awareness
    "min_win_rate_threshold": 0.45,     # Stop trading if win rate drops below 45%
    "model_accuracy_threshold": 0.50,   # Only use models with 50%+ accuracy
    "recent_performance_weight": 0.7,   # Weight recent performance heavily

    # Technical analysis thresholds
    "rsi_overbought_threshold": 70,
    "rsi_oversold_threshold": 30,
    "trend_strength_threshold": 0.02,
    "volume_spike_threshold": 2.0,      # 2x average volume = spike

    # ENHANCED: Multi-asset specific settings
    "correlation_threshold": 0.8,       # High correlation threshold
    "diversification_bonus": 0.1,       # Bonus for trading uncorrelated assets
    "asset_momentum_weight": 0.3,       # Weight for asset momentum in selection
    "cross_asset_signal_weight": 0.2,   # Weight for cross-asset signals

    # Data quality and validation
    "min_data_quality_threshold": 0.8,
    "max_feature_correlation": 0.95,
    "feature_engineering_retries": 3,
    "model_training_retries": 2,
    "prediction_fallback_confidence": 0.3,

    # Data preprocessing
    "min_training_samples": 100,        # Reduced for faster startup
    "max_feature_missing_rate": 0.15,   # Allow 15% missing values
    "price_change_outlier_threshold": 0.3,
    "volume_outlier_threshold": 5,

    # ENHANCED: GPT Integration (if available)
    "gpt_enabled": True,
    "gpt_confidence_weight": 0.25,      # Weight GPT analysis at 25%
    "gpt_override_threshold": 0.85,     # GPT can override ML if confidence >85%
    "gpt_analysis_frequency": 3,        # Every 3rd cycle
    "gpt_model": "gpt-4o-mini",         # Cost-effective model

    # Auto-retraining settings
    "auto_retrain_enabled": True,
    "retrain_schedule_hours": [2, 8, 14, 20],  # Retrain at specific hours
    "retrain_on_poor_performance": True,
    "retrain_performance_lookback": 50,  # Look at last 50 trades for performance

    # ENHANCED: Monitoring and logging
    "detailed_logging": True,
    "log_ml_predictions": True,
    "log_contrarian_trades": True,
    "log_asset_switches": True,
    "performance_reporting_frequency": 10,  # Every 10 cycles

    # Market data settings
    "market_data_timeout": 30,          # Timeout for market data (seconds)
    "reconnect_attempts": 3,             # Max reconnection attempts
    "reconnect_delay": 5,                # Delay between reconnection attempts
    "data_staleness_threshold": 120,    # Consider data stale after 2 minutes

    # ENHANCED: Safety and circuit breakers
    "emergency_stop_loss": 0.1,         # Stop all trading if losses exceed 10%
    "max_daily_trades": 500,            # Hard limit on daily trades
    "cooling_off_period": 300,          # 5 minute cooling off after major loss
    "volatility_circuit_breaker": 0.15, # Stop trading if volatility exceeds 15%

    # Database settings
    "db_connection_timeout": 30,
    "db_query_timeout": 15,
    "db_retry_attempts": 3,

    # Development and testing
    "simulation_mode": True,
    "paper_trading": True,
    "backtest_mode": False,
    "debug_mode": False,
    "verbose_logging": False,

    # Strategy parameters
    "strategy": "enhanced_multi_asset_ml",
    "strategy_version": "2.0",
    
    # Execution timing
    "trade_execution_delay": 0.1,       # Delay between trade executions
    "cycle_status_frequency": 10,       # Log status every 10 trades
    "session_summary_frequency": 5,     # Summary every 5 cycles

    # ENHANCED: Performance optimization
    "parallel_ml_training": False,      # Disable for stability
    "cache_ml_predictions": True,       # Cache predictions for reuse
    "optimize_feature_selection": True, # Use only best features
    "dynamic_model_weighting": True,    # Weight models by recent performance

    # Market session awareness
    "trading_hours_enabled": False,     # Trade 24/7 for crypto
    "weekend_trading": True,            # Enable weekend trading
    "holiday_trading": True,            # Enable holiday trading

    # ENHANCED: Advanced features
    "sentiment_analysis": False,        # Disable for now (future feature)
    "news_integration": False,          # Disable for now (future feature)
    "social_media_signals": False,     # Disable for now (future feature)
    
    # API rate limiting
    "api_rate_limit": 100,              # Max API calls per minute
    "api_burst_limit": 10,              # Max burst API calls
    
    # Failsafe settings
    "max_memory_usage": 1024,           # Max memory usage in MB
    "max_cpu_usage": 80,                # Max CPU usage percentage
    "health_check_frequency": 60,       # Health check every minute
}

# ENHANCED: Validation function
def validate_settings():
    """Validate settings configuration"""
    errors = []
    
    # Portfolio allocation validation
    allocation = SETTINGS.get("portfolio_allocation", {})
    total_allocation = sum(allocation.values())
    if abs(total_allocation - 1.0) > 0.01:
        errors.append(f"Portfolio allocation must sum to 1.0, got {total_allocation}")
    
    # Asset consistency
    supported_assets = set(SETTINGS.get("supported_assets", []))
    allocation_assets = set(allocation.keys())
    if supported_assets != allocation_assets:
        errors.append(f"Supported assets {supported_assets} don't match allocation assets {allocation_assets}")
    
    # Threshold validations
    if SETTINGS.get("ml_confidence_threshold", 0) < 0.1 or SETTINGS.get("ml_confidence_threshold", 0) > 0.9:
        errors.append("ML confidence threshold must be between 0.1 and 0.9")
    
    if SETTINGS.get("contrarian_threshold", 0) < 0.1 or SETTINGS.get("contrarian_threshold", 0) > 0.8:
        errors.append("Contrarian threshold must be between 0.1 and 0.8")
    
    # Performance thresholds
    good_perf = SETTINGS.get("good_performance_threshold", 0.6)
    poor_perf = SETTINGS.get("poor_performance_threshold", 0.4)
    if good_perf <= poor_perf:
        errors.append("Good performance threshold must be greater than poor performance threshold")
    
    if errors:
        print("⚠️ Settings validation errors:")
        for error in errors:
            print(f"   • {error}")
        return False
    else:
        print("✅ Settings validation passed")
        return True

# Validate settings on import
if __name__ == "__main__":
    validate_settings()