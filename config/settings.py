# config/settings.py - ENHANCED DIRECTIONAL TRADING SETTINGS
SETTINGS = {
    # Trading mode
    "mode": "directional_simulation",

    # 🎯 ENHANCED: DIRECTIONAL TRADING PARAMETERS
    "directional_trading_enabled": True,
    "long_enabled": True,           # Enable LONG positions (buy, expect rise)
    "short_enabled": True,          # Enable SHORT positions (sell, expect fall) 
    "hold_enabled": True,           # Enable HOLD periods (stay in USDC)
    
    # 🎯 DIRECTIONAL CONFIDENCE THRESHOLDS
    "long_confidence_threshold": 0.65,   # Confidence needed to go LONG
    "short_confidence_threshold": 0.65,  # Confidence needed to go SHORT
    "hold_confidence_threshold": 0.5,    # Below this = HOLD in USDC
    
    # 🎯 DIRECTIONAL RISK MANAGEMENT
    "max_position_hold_time": 1800,      # Max 30 minutes per position
    "stop_loss_percentage": 3.0,         # 3% stop loss
    "take_profit_percentage": 5.0,       # 5% take profit
    "max_concurrent_positions": 1,       # Only 1 position at a time
    
    # 🎯 DIRECTIONAL BIAS SETTINGS
    "long_bias_multiplier": 1.0,         # No bias toward LONG
    "short_bias_multiplier": 1.0,        # No bias toward SHORT  
    "hold_bias_multiplier": 1.0,         # No bias toward HOLD
    "trend_following_weight": 0.3,       # Weight for trend following
    "mean_reversion_weight": 0.7,        # Weight for mean reversion

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

    # 🎯 CONSERVATIVE: Core trading parameters
    "max_trades_per_minute": 20,        # Reduced for directional trading
    "trades_per_cycle": 25,             # Fewer decisions per cycle
    "cycle_delay_seconds": 60,          # Longer delays for position management

    # Trading execution
    "trade_amount_usd": 0.02,
    "pair": "DIRECTIONAL-MULTI-ASSET",  # Updated for directional multi-asset
    "slippage_tolerance": 0.0005,       # Tighter slippage for directional trades
    "risk_limit": 0.02,                 # Slightly higher risk for directional

    # Memory and data management
    "memory_limit": 15000,              # Increased for directional trade history

    # ENHANCED: ML Configuration for Directional Trading
    "ml_enabled": True,
    "ml_confidence_threshold": 0.35,    # Lower threshold for more activity
    "ml_retrain_hours": 3.0,            # More frequent retraining
    "adaptive_trading": True,
    
    # 🎯 DIRECTIONAL ML PARAMETERS
    "ml_directional_mode": True,        # Train models to predict direction
    "ml_long_prediction_weight": 0.4,   # Weight for LONG predictions
    "ml_short_prediction_weight": 0.4,  # Weight for SHORT predictions
    "ml_hold_prediction_weight": 0.2,   # Weight for HOLD predictions
    
    # Enhanced ML training parameters
    "ml_min_samples": 75,               # Further reduced for faster startup
    "ml_max_features": 12,              # Reduced features for directional focus
    "retrain_min_samples_trigger": 150, # Trigger retraining after 150 new samples
    "retrain_accuracy_threshold": 0.52, # Lower threshold for directional trading
    "ml_prediction_timeout": 25,        # Shorter timeout

    # 🎯 ENHANCED: RSI-based Directional Signals
    "rsi_extreme_long_threshold": 25,   # RSI below this = strong LONG signal
    "rsi_strong_long_threshold": 35,    # RSI below this = moderate LONG signal
    "rsi_neutral_lower": 40,            # RSI neutral zone lower bound
    "rsi_neutral_upper": 60,            # RSI neutral zone upper bound
    "rsi_strong_short_threshold": 65,   # RSI above this = moderate SHORT signal
    "rsi_extreme_short_threshold": 75,  # RSI above this = strong SHORT signal
    
    # 🎯 MOMENTUM-based Directional Signals
    "momentum_long_threshold": -2.0,    # 24h change below this = LONG signal
    "momentum_short_threshold": 3.0,    # 24h change above this = SHORT signal
    "momentum_extreme_long": -5.0,      # Extreme downward momentum = strong LONG
    "momentum_extreme_short": 8.0,      # Extreme upward momentum = strong SHORT

    # ENHANCED: Adaptive trading controls
    "high_confidence_multiplier": 1.15, # Slightly conservative
    "low_confidence_multiplier": 0.85,  # More conservative
    "market_volatility_threshold": 0.02, # More sensitive to volatility
    
    # Performance-based adaptive parameters
    "performance_adjustment_window": 5,  # Look at last 5 cycles for performance
    "good_performance_threshold": 0.65,  # 65%+ win rate = good
    "poor_performance_threshold": 0.35,  # <35% win rate = poor
    "performance_boost_factor": 1.1,     # Conservative boost when performing well
    "performance_reduce_factor": 0.9,    # Conservative reduction when performing poorly

    # 🎯 ENHANCED: Directional Risk Management
    "max_consecutive_losses": 2,         # Stop after 2 consecutive losses
    "daily_loss_limit": 0.03,           # Stop if daily losses exceed 3%
    "position_size_scaling": True,       # Scale position size based on confidence
    "max_position_multiplier": 1.3,     # Max 1.3x normal position size
    "min_position_multiplier": 0.7,     # Min 0.7x normal position size
    
    # 🎯 DIRECTIONAL POSITION MANAGEMENT
    "force_close_on_reverse_signal": True,   # Close LONG if SHORT signal appears
    "position_timeout_seconds": 1200,        # Force close after 20 minutes
    "trailing_stop_enabled": False,          # Disable trailing stops for now
    "partial_close_enabled": False,          # Disable partial closes for now

    # Market condition awareness
    "min_win_rate_threshold": 0.3,      # Stop trading if win rate drops below 30%
    "model_accuracy_threshold": 0.4,    # Only use models with 40%+ accuracy
    "recent_performance_weight": 0.8,   # Weight recent performance heavily

    # 🎯 ENHANCED: Technical Analysis for Directional Trading
    "rsi_overbought_threshold": 70,
    "rsi_oversold_threshold": 30,
    "trend_strength_threshold": 0.015,
    "volume_spike_threshold": 1.8,      # 1.8x average volume = spike
    
    # Bollinger Bands for directional signals
    "bb_period": 20,
    "bb_std_dev": 2.0,
    "bb_squeeze_threshold": 0.015,      # Low volatility threshold
    
    # Moving Average signals
    "ma_fast_period": 10,
    "ma_slow_period": 25,
    "ma_crossover_weight": 0.3,

    # ENHANCED: Multi-asset specific settings
    "correlation_threshold": 0.8,       # High correlation threshold
    "diversification_bonus": 0.05,      # Bonus for trading uncorrelated assets
    "asset_momentum_weight": 0.25,      # Weight for asset momentum in selection
    "cross_asset_signal_weight": 0.15,  # Weight for cross-asset signals

    # Data quality and validation
    "min_data_quality_threshold": 0.75,
    "max_feature_correlation": 0.92,
    "feature_engineering_retries": 3,
    "model_training_retries": 2,
    "prediction_fallback_confidence": 0.25,

    # Data preprocessing
    "min_training_samples": 75,         # Reduced for faster startup
    "max_feature_missing_rate": 0.2,    # Allow 20% missing values
    "price_change_outlier_threshold": 0.4,
    "volume_outlier_threshold": 4,

    # ENHANCED: GPT Integration (if available)
    "gpt_enabled": False,               # Disabled for directional focus
    "gpt_confidence_weight": 0.15,      # Reduced weight
    "gpt_override_threshold": 0.9,      # Very high threshold for override
    "gpt_analysis_frequency": 5,        # Every 5th cycle
    "gpt_model": "gpt-4o-mini",         # Cost-effective model

    # Auto-retraining settings
    "auto_retrain_enabled": True,
    "retrain_schedule_hours": [3, 9, 15, 21],  # Retrain 4 times daily
    "retrain_on_poor_performance": True,
    "retrain_performance_lookback": 40,  # Look at last 40 trades for performance

    # ENHANCED: Monitoring and logging
    "detailed_logging": True,
    "log_ml_predictions": True,
    "log_directional_decisions": True,   # Log all directional decisions
    "log_position_changes": True,        # Log position opens/closes
    "log_asset_switches": True,
    "performance_reporting_frequency": 8,  # Every 8 cycles

    # Market data settings
    "market_data_timeout": 25,          # Timeout for market data (seconds)
    "reconnect_attempts": 3,             # Max reconnection attempts
    "reconnect_delay": 5,                # Delay between reconnection attempts
    "data_staleness_threshold": 90,     # Consider data stale after 1.5 minutes

    # 🎯 ENHANCED: Safety and Circuit Breakers for Directional Trading
    "emergency_stop_loss": 0.05,        # Stop all trading if losses exceed 5%
    "max_daily_trades": 300,            # Hard limit on daily trades
    "cooling_off_period": 600,          # 10 minute cooling off after major loss
    "volatility_circuit_breaker": 0.1,  # Stop trading if volatility exceeds 10%
    
    # 🎯 DIRECTIONAL-SPECIFIC CIRCUIT BREAKERS
    "max_consecutive_long_losses": 3,   # Stop LONG trades after 3 losses
    "max_consecutive_short_losses": 3,  # Stop SHORT trades after 3 losses
    "directional_performance_threshold": 0.25,  # Disable direction if win rate < 25%

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
    "strategy": "enhanced_directional_multi_asset",
    "strategy_version": "3.0",
    
    # Execution timing
    "trade_execution_delay": 0.05,      # Faster execution for directional trades
    "cycle_status_frequency": 8,        # Log status every 8 trades
    "session_summary_frequency": 6,     # Summary every 6 cycles

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
    "api_rate_limit": 80,               # Reduced for directional trading
    "api_burst_limit": 8,               # Reduced burst limit
    
    # Failsafe settings
    "max_memory_usage": 1024,           # Max memory usage in MB
    "max_cpu_usage": 75,                # Max CPU usage percentage
    "health_check_frequency": 90,       # Health check every 1.5 minutes

    # 🎯 DIRECTIONAL TRADING EXPERIMENTAL FEATURES
    "experimental_features_enabled": False,
    "neural_network_signals": False,    # Advanced NN-based signals
    "sentiment_weighted_signals": False, # Weight signals by market sentiment
    "cross_timeframe_analysis": False,  # Multi-timeframe signal analysis
    "dynamic_threshold_adjustment": False, # Adjust thresholds based on performance
}

# 🎯 ENHANCED: Validation function for directional trading
def validate_settings():
    """Validate directional trading settings configuration"""
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
    
    # 🎯 DIRECTIONAL THRESHOLD VALIDATIONS
    long_threshold = SETTINGS.get("long_confidence_threshold", 0.65)
    short_threshold = SETTINGS.get("short_confidence_threshold", 0.65)
    hold_threshold = SETTINGS.get("hold_confidence_threshold", 0.5)
    
    if not (0.1 <= long_threshold <= 0.9):
        errors.append(f"LONG confidence threshold must be between 0.1 and 0.9, got {long_threshold}")
    
    if not (0.1 <= short_threshold <= 0.9):
        errors.append(f"SHORT confidence threshold must be between 0.1 and 0.9, got {short_threshold}")
        
    if not (0.1 <= hold_threshold <= 0.9):
        errors.append(f"HOLD confidence threshold must be between 0.1 and 0.9, got {hold_threshold}")
    
    # RSI threshold validations
    rsi_extreme_long = SETTINGS.get("rsi_extreme_long_threshold", 25)
    rsi_extreme_short = SETTINGS.get("rsi_extreme_short_threshold", 75)
    
    if not (0 <= rsi_extreme_long <= 50):
        errors.append(f"RSI extreme LONG threshold must be between 0 and 50, got {rsi_extreme_long}")
        
    if not (50 <= rsi_extreme_short <= 100):
        errors.append(f"RSI extreme SHORT threshold must be between 50 and 100, got {rsi_extreme_short}")
    
    # Position management validations
    max_hold_time = SETTINGS.get("max_position_hold_time", 1800)
    if max_hold_time < 60 or max_hold_time > 3600:
        errors.append(f"Max position hold time must be between 60 and 3600 seconds, got {max_hold_time}")
    
    # Risk management validations
    stop_loss = SETTINGS.get("stop_loss_percentage", 3.0)
    take_profit = SETTINGS.get("take_profit_percentage", 5.0)
    
    if stop_loss <= 0 or stop_loss > 10:
        errors.append(f"Stop loss percentage must be between 0 and 10, got {stop_loss}")
        
    if take_profit <= 0 or take_profit > 20:
        errors.append(f"Take profit percentage must be between 0 and 20, got {take_profit}")
        
    if take_profit <= stop_loss:
        errors.append(f"Take profit ({take_profit}) must be greater than stop loss ({stop_loss})")
    
    # Performance thresholds
    good_perf = SETTINGS.get("good_performance_threshold", 0.65)
    poor_perf = SETTINGS.get("poor_performance_threshold", 0.35)
    if good_perf <= poor_perf:
        errors.append("Good performance threshold must be greater than poor performance threshold")
    
    # ML validation
    ml_confidence = SETTINGS.get("ml_confidence_threshold", 0.35)
    if ml_confidence < 0.1 or ml_confidence > 0.9:
        errors.append("ML confidence threshold must be between 0.1 and 0.9")
    
    if errors:
        print("⚠️ Settings validation errors:")
        for error in errors:
            print(f"   • {error}")
        return False
    else:
        print("✅ Directional trading settings validation passed")
        return True

# 🎯 DIRECTIONAL TRADING HELPER FUNCTIONS
def get_directional_thresholds():
    """Get directional trading thresholds"""
    return {
        'long_threshold': SETTINGS.get("long_confidence_threshold", 0.65),
        'short_threshold': SETTINGS.get("short_confidence_threshold", 0.65),
        'hold_threshold': SETTINGS.get("hold_confidence_threshold", 0.5),
        'rsi_extreme_long': SETTINGS.get("rsi_extreme_long_threshold", 25),
        'rsi_extreme_short': SETTINGS.get("rsi_extreme_short_threshold", 75)
    }

def get_risk_management_params():
    """Get risk management parameters"""
    return {
        'stop_loss_pct': SETTINGS.get("stop_loss_percentage", 3.0),
        'take_profit_pct': SETTINGS.get("take_profit_percentage", 5.0),
        'max_hold_time': SETTINGS.get("max_position_hold_time", 1800),
        'max_daily_loss': SETTINGS.get("daily_loss_limit", 0.03),
        'max_consecutive_losses': SETTINGS.get("max_consecutive_losses", 2)
    }

def is_directional_trading_enabled():
    """Check if directional trading is enabled"""
    return SETTINGS.get("directional_trading_enabled", True)

def get_enabled_directions():
    """Get which trading directions are enabled"""
    return {
        'long': SETTINGS.get("long_enabled", True),
        'short': SETTINGS.get("short_enabled", True),
        'hold': SETTINGS.get("hold_enabled", True)
    }

# Validate settings on import
if __name__ == "__main__":
    validate_settings()
    
    # Print directional trading summary
    print("\n🎯 DIRECTIONAL TRADING CONFIGURATION:")
    print(f"   • LONG enabled: {SETTINGS.get('long_enabled', True)}")
    print(f"   • SHORT enabled: {SETTINGS.get('short_enabled', True)}")
    print(f"   • HOLD enabled: {SETTINGS.get('hold_enabled', True)}")
    print(f"   • LONG threshold: {SETTINGS.get('long_confidence_threshold', 0.65)}")
    print(f"   • SHORT threshold: {SETTINGS.get('short_confidence_threshold', 0.65)}")
    print(f"   • HOLD threshold: {SETTINGS.get('hold_confidence_threshold', 0.5)}")
    print(f"   • Max position time: {SETTINGS.get('max_position_hold_time', 1800)}s")
    print(f"   • Stop loss: {SETTINGS.get('stop_loss_percentage', 3.0)}%")
    print(f"   • Take profit: {SETTINGS.get('take_profit_percentage', 5.0)}%")