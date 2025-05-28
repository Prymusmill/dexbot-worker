# config/settings.py - OPTIMIZED for higher frequency trading
SETTINGS = {
    # Trading mode
    "mode": "simulation",
    
    # OPTIMIZED: Increased frequency
    "max_trades_per_minute": 50,  # Increased from 30
    "trades_per_cycle": 50,       # NEW: Increased from 30
    "cycle_delay_seconds": 30,    # NEW: Reduced from 60
    
    # Trading parameters
    "trade_amount_usd": 0.02,
    "pair": "SOL/USDC",
    
    # Strategy settings
    "strategy": "ml_enhanced",    # NEW: ML-enhanced strategy
    "slippage_tolerance": 0.005,
    "risk_limit": 0.03,
    
    # Memory and data
    "memory_limit": 2000,         # Increased from 1000
    
    # ML-specific settings
    "ml_enabled": True,           # NEW: Enable ML predictions
    "ml_confidence_threshold": 0.3,  # NEW: Minimum confidence for ML trades
    "ml_retrain_hours": 4,        # NEW: Retrain every 4 hours
    "adaptive_trading": True,     # NEW: Adjust frequency based on ML confidence
    
    # Advanced trading controls
    "high_confidence_multiplier": 1.5,  # NEW: Trade more when ML confident
    "low_confidence_multiplier": 0.7,   # NEW: Trade less when ML uncertain
    "market_volatility_threshold": 0.05, # NEW: Reduce trading in high volatility
}