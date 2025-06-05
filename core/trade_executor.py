# core/trade_executor.py - ENHANCED with DIRECTIONAL TRADING (LONG/SHORT/HOLD)
import csv
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import random

MEMORY_FILE = "data/memory.csv"


@dataclass
class DirectionalTradeResult:
    """🎯 Enhanced trade result with directional trading support"""
    timestamp: str
    asset: str
    direction: str  # 'long', 'short', 'hold'
    action: str     # 'open_long', 'close_long', 'open_short', 'close_short', 'hold'
    amount_in: float
    amount_out: float
    price_entry: float
    price_exit: float
    price_impact: float
    market_price: float
    spread: float
    signal_strength: float
    strategy_used: str
    profitable: bool
    pnl: float
    pnl_percentage: float
    position_duration: float  # How long position was held
    risk_level: str  # 'low', 'medium', 'high'


class DirectionalPositionTracker:
    """🎯 Track open positions for directional trading"""
    
    def __init__(self):
        self.open_positions = {}  # asset -> position_info
        self.position_history = []
        
    def open_position(self, asset: str, direction: str, entry_price: float, 
                     amount: float, timestamp: str) -> bool:
        """Open new position"""
        if asset in self.open_positions:
            print(f"⚠️ Position already open for {asset}, closing first...")
            self.close_position(asset, entry_price, timestamp)
        
        self.open_positions[asset] = {
            'direction': direction,
            'entry_price': entry_price,
            'amount': amount,
            'timestamp': timestamp,
            'entry_time': datetime.now()
        }
        
        print(f"🎯 Opened {direction.upper()} position: {asset} @ ${entry_price:.4f}")
        return True
    
    def close_position(self, asset: str, exit_price: float, timestamp: str) -> Optional[Dict]:
        """Close existing position and calculate P&L"""
        if asset not in self.open_positions:
            return None
        
        position = self.open_positions[asset]
        direction = position['direction']
        entry_price = position['entry_price']
        amount = position['amount']
        entry_time = position['entry_time']
        
        # Calculate P&L based on direction
        if direction == 'long':
            # LONG: profit when price goes up
            pnl = (exit_price - entry_price) * amount
            pnl_percentage = ((exit_price - entry_price) / entry_price) * 100
        elif direction == 'short':
            # SHORT: profit when price goes down
            pnl = (entry_price - exit_price) * amount
            pnl_percentage = ((entry_price - exit_price) / entry_price) * 100
        else:
            pnl = 0
            pnl_percentage = 0
        
        # Position duration
        duration = (datetime.now() - entry_time).total_seconds()
        
        # Create position result
        result = {
            'asset': asset,
            'direction': direction,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'amount': amount,
            'pnl': pnl,
            'pnl_percentage': pnl_percentage,
            'duration': duration,
            'profitable': pnl > 0,
            'entry_timestamp': position['timestamp'],
            'exit_timestamp': timestamp
        }
        
        # Remove from open positions
        del self.open_positions[asset]
        
        # Add to history
        self.position_history.append(result)
        
        profit_status = "✅ PROFIT" if pnl > 0 else "❌ LOSS"
        print(f"🎯 Closed {direction.upper()} position: {asset} @ ${exit_price:.4f} → {profit_status} ${pnl:.6f} ({pnl_percentage:+.2f}%)")
        
        return result
    
    def get_open_position(self, asset: str) -> Optional[Dict]:
        """Get open position for asset"""
        return self.open_positions.get(asset)
    
    def has_open_position(self, asset: str) -> bool:
        """Check if asset has open position"""
        return asset in self.open_positions
    
    def get_all_open_positions(self) -> Dict:
        """Get all open positions"""
        return self.open_positions.copy()
    
    def get_position_summary(self) -> Dict:
        """Get position summary statistics"""
        if not self.position_history:
            return {'total': 0, 'profitable': 0, 'win_rate': 0, 'avg_pnl': 0}
        
        total = len(self.position_history)
        profitable = sum(1 for p in self.position_history if p['profitable'])
        win_rate = profitable / total if total > 0 else 0
        avg_pnl = sum(p['pnl'] for p in self.position_history) / total if total > 0 else 0
        
        return {
            'total_positions': total,
            'profitable_positions': profitable,
            'win_rate': win_rate,
            'average_pnl': avg_pnl,
            'open_positions': len(self.open_positions)
        }


class EnhancedDirectionalTradeExecutor:
    """🎯 Enhanced trade executor with FULL directional trading support"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.last_market_data = {}  # Per-asset market data
        self.db_manager = None
        self.db_available = False
        
        # 🎯 DIRECTIONAL TRADING COMPONENTS
        self.position_tracker = DirectionalPositionTracker()
        self.trading_mode = 'directional'  # 'directional' vs 'legacy'
        self.ml_integration = None
        
        # Enhanced settings
        self.default_position_size = 0.02  # $0.02 per position
        self.max_position_duration = 300   # 5 minutes max hold
        self.min_profit_threshold = 0.01   # 1% minimum profit to close
        self.max_loss_threshold = -0.03    # 3% maximum loss before stop
        
        # Initialize database connection
        self._init_database()

    def set_ml_integration(self, ml_integration):
        """🤖 Set ML integration for enhanced predictions"""
        self.ml_integration = ml_integration
        print("✅ ML integration connected to trade executor")
    
    def _init_database(self):
        """Initialize database connection with fallback"""
        try:
            from database.db_manager import get_db_manager
            self.db_manager = get_db_manager()
            self.db_available = True
            print("✅ PostgreSQL database connected successfully")

            # Try to migrate existing CSV data
            self._migrate_csv_if_needed()

        except Exception as e:
            print(f"⚠️ PostgreSQL connection failed: {e}")
            print("🔄 Continuing with CSV-only mode")
            self.db_available = False

    def _migrate_csv_if_needed(self):
        """Migrate existing CSV data to PostgreSQL on first run"""
        if not self.db_available or not os.path.exists(MEMORY_FILE):
            return

        try:
            # Check if database is empty
            count = self.db_manager.get_transaction_count()
            if count == 0:
                print("🔄 Migrating existing CSV data to PostgreSQL...")
                success = self.db_manager.migrate_from_csv(MEMORY_FILE)
                if success:
                    print("✅ CSV data migration completed")
                else:
                    print("⚠️ CSV data migration had issues")
            else:
                print(f"📊 PostgreSQL already has {count} transactions")

        except Exception as e:
            print(f"⚠️ Migration check failed: {e}")

    def update_market_data(self, market_data: Dict, asset: str = 'SOL'):
        """Update market data for specific asset"""
        self.last_market_data[asset] = market_data
        
        # Check for position management
        self._check_position_management(asset, market_data)

    def _check_position_management(self, asset: str, market_data: Dict):
        """🎯 Check if open positions need management (stop loss, take profit, time exit)"""
        if not self.position_tracker.has_open_position(asset):
            return
        
        position = self.position_tracker.get_open_position(asset)
        current_price = market_data.get('price', 0)
        
        if current_price <= 0:
            return
        
        direction = position['direction']
        entry_price = position['entry_price']
        entry_time = position['entry_time']
        
        # Calculate current P&L
        if direction == 'long':
            current_pnl_pct = ((current_price - entry_price) / entry_price) * 100
        elif direction == 'short':
            current_pnl_pct = ((entry_price - current_price) / entry_price) * 100
        else:
            return
        
        # Position duration check
        duration = (datetime.now() - entry_time).total_seconds()
        
        should_close = False
        close_reason = ""
        
        # Time-based exit (max duration)
        if duration > self.max_position_duration:
            should_close = True
            close_reason = "max_duration"
        
        # Take profit
        elif current_pnl_pct > self.min_profit_threshold:
            should_close = True
            close_reason = "take_profit"
        
        # Stop loss
        elif current_pnl_pct < self.max_loss_threshold:
            should_close = True
            close_reason = "stop_loss"
        
        if should_close:
            print(f"🎯 Auto-closing {asset} {direction} position: {close_reason} (P&L: {current_pnl_pct:+.2f}%)")
            self._close_position(asset, current_price, close_reason)

    def execute_directional_trade(self, settings: Dict, asset: str = 'SOL', 
                                 direction: str = None) -> Optional[DirectionalTradeResult]:
        """🎯 Execute directional trade (LONG/SHORT/HOLD)"""
        
        # Get market data for asset
        market_data = self.last_market_data.get(asset)
        if not market_data:
            print(f"⚠️ No market data available for {asset}")
            return self._simulate_directional_trade(settings, asset, direction)
        
        # Determine direction if not provided
        if not direction:
            direction = self._determine_trading_direction(asset, market_data)
        
        timestamp = datetime.utcnow().isoformat()
        current_price = market_data.get('price', 0)
        
        if current_price <= 0:
            return self._simulate_directional_trade(settings, asset, direction)
        
        # Check if we need to close existing position first
        if self.position_tracker.has_open_position(asset):
            return self._handle_existing_position(asset, direction, market_data, timestamp)
        
        # Open new position
        if direction == 'hold':
            return self._execute_hold_action(asset, market_data, timestamp)
        else:
            return self._open_new_position(asset, direction, market_data, timestamp, settings)

    def _determine_trading_direction(self, asset: str, market_data: Dict) -> str:
        """🎯 Determine trading direction based on signals and ML"""
        
        # Try to get directional signals from market data
        directional_signals = market_data.get('directional_signals', {})
        
        if directional_signals:
            recommended = max(directional_signals, key=directional_signals.get)
            confidence = directional_signals[recommended]
            
            if confidence > 0.6:
                print(f"🎯 Direction from signals: {recommended.upper()} ({confidence:.2f})")
                return recommended
        
        # Fallback to ML if available
        if self.ml_integration:
            try:
                # Get ML prediction
                df = self._create_prediction_dataframe(market_data)
                ml_prediction = self.ml_integration.get_directional_prediction(df)
                
                ml_direction = ml_prediction.get('predicted_direction', 'hold')
                ml_confidence = ml_prediction.get('confidence', 0.5)
                
                if ml_confidence > 0.6:
                    print(f"🤖 Direction from ML: {ml_direction.upper()} ({ml_confidence:.2f})")
                    return ml_direction
                    
            except Exception as e:
                print(f"⚠️ ML direction determination failed: {e}")
        
        # Fallback to technical analysis
        return self._technical_direction_fallback(market_data)

    def _technical_direction_fallback(self, market_data: Dict) -> str:
        """🎯 Fallback direction determination using technical analysis"""
        rsi = market_data.get('rsi', 50)
        price_change_24h = market_data.get('price_change_24h', 0)
        volatility = market_data.get('volatility', 0.02)
        
        # Simple technical rules
        if rsi < 25 and price_change_24h < -5:
            return 'long'  # Oversold and declining = buy the dip
        elif rsi > 75 and price_change_24h > 8:
            return 'short'  # Overbought and rising = short the top
        elif volatility < 0.01:
            return 'hold'  # Low volatility = wait
        else:
            # Random choice weighted by RSI
            if rsi < 45:
                return random.choice(['long', 'long', 'hold'])
            elif rsi > 55:
                return random.choice(['short', 'short', 'hold'])
            else:
                return 'hold'

    def _create_prediction_dataframe(self, market_data: Dict):
        """Create DataFrame for ML prediction"""
        import pandas as pd
        
        # Create simple DataFrame with current market data
        df = pd.DataFrame([{
            'price': market_data.get('price', 100),
            'rsi': market_data.get('rsi', 50),
            'volume': market_data.get('volume_24h', 1000),
            'volatility': market_data.get('volatility', 0.02),
            'price_change_24h': market_data.get('price_change_24h', 0),
            'timestamp': datetime.now()
        }])
        
        return df

    def _handle_existing_position(self, asset: str, new_direction: str, 
                                 market_data: Dict, timestamp: str) -> Optional[DirectionalTradeResult]:
        """🎯 Handle when we already have an open position"""
        
        current_position = self.position_tracker.get_open_position(asset)
        current_direction = current_position['direction']
        current_price = market_data.get('price', 0)
        
        # If same direction, hold the position
        if new_direction == current_direction:
            print(f"🎯 Maintaining {current_direction.upper()} position for {asset}")
            return self._create_hold_result(asset, market_data, timestamp, "maintain_position")
        
        # If different direction, close current and potentially open new
        print(f"🔄 Direction change for {asset}: {current_direction} → {new_direction}")
        
        # Close current position
        close_result = self.position_tracker.close_position(asset, current_price, timestamp)
        
        if close_result:
            # Save the close trade
            close_trade_result = self._create_directional_trade_result(
                timestamp=timestamp,
                asset=asset,
                direction=current_direction,
                action=f"close_{current_direction}",
                amount_in=close_result['amount'],
                amount_out=close_result['amount'] + close_result['pnl'],
                price_entry=close_result['entry_price'],
                price_exit=close_result['exit_price'],
                market_data=market_data,
                strategy="position_switch",
                pnl=close_result['pnl'],
                pnl_percentage=close_result['pnl_percentage'],
                duration=close_result['duration']
            )
            
            # Save close trade
            self._save_trade_data(close_trade_result, market_data)
        
        # Open new position if not hold
        if new_direction != 'hold':
            return self._open_new_position(asset, new_direction, market_data, timestamp, {})
        else:
            return self._execute_hold_action(asset, market_data, timestamp)

    def _open_new_position(self, asset: str, direction: str, market_data: Dict, 
                          timestamp: str, settings: Dict) -> DirectionalTradeResult:
        """🎯 Open new directional position"""
        
        amount_usd = settings.get('trade_amount_usd', self.default_position_size)
        current_price = market_data.get('price', 0)
        spread = market_data.get('spread', current_price * 0.001)
        volatility = market_data.get('volatility', 0.02)
        
        # Calculate entry price with realistic slippage
        if direction == 'long':
            # LONG: buy at ask price
            entry_price = current_price + (spread / 2)
            action = "open_long"
        elif direction == 'short':
            # SHORT: sell at bid price
            entry_price = current_price - (spread / 2)
            action = "open_short"
        else:
            return self._execute_hold_action(asset, market_data, timestamp)
        
        # Add volatility-based slippage
        slippage = random.uniform(-volatility, volatility) * 0.1
        entry_price *= (1 + slippage)
        
        # Open position in tracker
        self.position_tracker.open_position(asset, direction, entry_price, amount_usd, timestamp)
        
        # Create trade result for the opening
        trade_result = self._create_directional_trade_result(
            timestamp=timestamp,
            asset=asset,
            direction=direction,
            action=action,
            amount_in=amount_usd,
            amount_out=amount_usd,  # No immediate P&L on open
            price_entry=entry_price,
            price_exit=entry_price,
            market_data=market_data,
            strategy="directional_opening",
            pnl=0.0,  # No P&L on opening
            pnl_percentage=0.0,
            duration=0.0
        )
        
        # Save opening trade
        self._save_trade_data(trade_result, market_data)
        
        print(f"🎯 Opened {direction.upper()} position: {asset} @ ${entry_price:.4f}")
        
        return trade_result

    def _close_position(self, asset: str, exit_price: float, reason: str) -> Optional[DirectionalTradeResult]:
        """🎯 Close existing position"""
        
        timestamp = datetime.utcnow().isoformat()
        close_result = self.position_tracker.close_position(asset, exit_price, timestamp)
        
        if not close_result:
            return None
        
        # Get market data for the close
        market_data = self.last_market_data.get(asset, {})
        market_data['price'] = exit_price
        
        # Create trade result for the close
        trade_result = self._create_directional_trade_result(
            timestamp=timestamp,
            asset=asset,
            direction=close_result['direction'],
            action=f"close_{close_result['direction']}",
            amount_in=close_result['amount'],
            amount_out=close_result['amount'] + close_result['pnl'],
            price_entry=close_result['entry_price'],
            price_exit=close_result['exit_price'],
            market_data=market_data,
            strategy=f"close_{reason}",
            pnl=close_result['pnl'],
            pnl_percentage=close_result['pnl_percentage'],
            duration=close_result['duration']
        )
        
        # Save close trade
        self._save_trade_data(trade_result, market_data)
        
        return trade_result

    def _execute_hold_action(self, asset: str, market_data: Dict, timestamp: str) -> DirectionalTradeResult:
        """🎯 Execute hold action (no position change)"""
        
        trade_result = self._create_hold_result(asset, market_data, timestamp, "hold_signal")
        
        # Save hold action
        self._save_trade_data(trade_result, market_data)
        
        return trade_result

    def _create_hold_result(self, asset: str, market_data: Dict, timestamp: str, reason: str) -> DirectionalTradeResult:
        """Create result for hold action"""
        
        current_price = market_data.get('price', 0)
        
        return self._create_directional_trade_result(
            timestamp=timestamp,
            asset=asset,
            direction='hold',
            action='hold',
            amount_in=0.0,
            amount_out=0.0,
            price_entry=current_price,
            price_exit=current_price,
            market_data=market_data,
            strategy=reason,
            pnl=0.0,
            pnl_percentage=0.0,
            duration=0.0
        )

    def _create_directional_trade_result(self, timestamp: str, asset: str, direction: str,
                                       action: str, amount_in: float, amount_out: float,
                                       price_entry: float, price_exit: float,
                                       market_data: Dict, strategy: str,
                                       pnl: float, pnl_percentage: float,
                                       duration: float) -> DirectionalTradeResult:
        """Create comprehensive directional trade result"""
        
        # Calculate additional metrics
        price_impact = abs(price_exit - price_entry) / price_entry if price_entry > 0 else 0
        spread = market_data.get('spread', 0)
        signal_strength = market_data.get('signal_strength', 0.5)
        
        # Assess risk level
        volatility = market_data.get('volatility', 0.02)
        if volatility > 0.05:
            risk_level = 'high'
        elif volatility < 0.01:
            risk_level = 'low'
        else:
            risk_level = 'medium'
        
        return DirectionalTradeResult(
            timestamp=timestamp,
            asset=asset,
            direction=direction,
            action=action,
            amount_in=amount_in,
            amount_out=amount_out,
            price_entry=price_entry,
            price_exit=price_exit,
            price_impact=price_impact,
            market_price=market_data.get('price', price_exit),
            spread=spread,
            signal_strength=signal_strength,
            strategy_used=strategy,
            profitable=(pnl > 0),
            pnl=pnl,
            pnl_percentage=pnl_percentage,
            position_duration=duration,
            risk_level=risk_level
        )

    def _simulate_directional_trade(self, settings: Dict, asset: str, direction: str) -> DirectionalTradeResult:
        """🎯 Simulate directional trade when no market data available"""
        
        timestamp = datetime.utcnow().isoformat()
        amount_usd = settings.get('trade_amount_usd', self.default_position_size)
        
        # Simulate basic price movement
        base_price = 100.0
        
        if direction == 'long':
            # Simulate long position
            price_change = random.uniform(-0.02, 0.03)  # Slight long bias
            exit_price = base_price * (1 + price_change)
            pnl = amount_usd * price_change
        elif direction == 'short':
            # Simulate short position  
            price_change = random.uniform(-0.03, 0.02)  # Slight short bias
            exit_price = base_price * (1 - price_change)  # Inverse for short
            pnl = amount_usd * price_change
        else:  # hold
            price_change = 0
            exit_price = base_price
            pnl = 0
        
        # Create simulated market data
        simulated_market_data = {
            'price': exit_price,
            'spread': 0.001,
            'volatility': 0.02,
            'signal_strength': 0.5
        }
        
        trade_result = self._create_directional_trade_result(
            timestamp=timestamp,
            asset=asset,
            direction=direction,
            action=f"simulate_{direction}",
            amount_in=amount_usd,
            amount_out=amount_usd + pnl,
            price_entry=base_price,
            price_exit=exit_price,
            market_data=simulated_market_data,
            strategy="simulation",
            pnl=pnl,
            pnl_percentage=(pnl / amount_usd) * 100 if amount_usd > 0 else 0,
            duration=0.0
        )
        
        print(f"🎯 Simulated {direction.upper()}: {asset} P&L: ${pnl:.6f} ({trade_result.pnl_percentage:+.2f}%)")
        
        return trade_result

    # 🚀 LEGACY COMPATIBILITY
    def execute_trade(self, settings: Dict, market_data: Optional[Dict] = None, asset: str = 'SOL') -> Optional[DirectionalTradeResult]:
        """🔄 Legacy compatibility wrapper - converts to directional trading"""
        
        if market_data:
            self.update_market_data(market_data, asset)
        
        # Execute as directional trade
        return self.execute_directional_trade(settings, asset)

    def _save_trade_data(self, trade_result: DirectionalTradeResult, market_data: Optional[Dict] = None):
        """🎯 Enhanced save with directional data"""

        # 1. Try to save to PostgreSQL first
        if self.db_available and self.db_manager:
            try:
                trade_data = {
                    'timestamp': trade_result.timestamp,
                    'input_token': trade_result.asset,
                    'output_token': 'USDC',
                    'amount_in': trade_result.amount_in,
                    'amount_out': trade_result.amount_out,
                    'price_impact': trade_result.price_impact,
                    'direction': trade_result.direction,  # NEW: directional data
                    'action': trade_result.action,        # NEW: action data
                    'pnl': trade_result.pnl,             # NEW: P&L data
                    'strategy': trade_result.strategy_used
                }

                db_id = self.db_manager.save_transaction(trade_data, market_data)
                if db_id:
                    print(f"✅ Directional trade saved to PostgreSQL (ID: {db_id})")
                else:
                    print("⚠️ PostgreSQL save failed, continuing with CSV")

            except Exception as e:
                print(f"⚠️ PostgreSQL error: {e}")
                print("🔄 Falling back to CSV only")

        # 2. Always save to CSV as backup
        self._save_to_csv(trade_result, market_data)

    def _save_to_csv(self, trade_result: DirectionalTradeResult, market_data: Optional[Dict] = None):
        """🎯 Save directional trade to CSV with enhanced columns"""
        os.makedirs("data", exist_ok=True)
        file_exists = os.path.isfile(MEMORY_FILE)

        # 🎯 ENHANCED HEADERS for directional trading
        headers = [
            "timestamp",
            "asset",               # NEW: asset name
            "direction",          # NEW: long/short/hold
            "action",             # NEW: open_long/close_short/etc
            "amount_in",
            "amount_out",
            "pnl",                # NEW: profit/loss
            "pnl_percentage",     # NEW: P&L percentage
            "price_entry",        # NEW: entry price
            "price_exit",         # NEW: exit price
            "price_impact",
            "market_price",       # Renamed from 'price'
            "volume",
            "rsi",
            "strategy",           # NEW: strategy used
            "profitable",         # Enhanced: use directional profitable
            "position_duration",  # NEW: how long position held
            "risk_level"          # NEW: risk assessment
        ]

        # Get data with defaults
        current_price = market_data.get('price', trade_result.price_exit) if market_data else trade_result.market_price
        current_rsi = market_data.get('rsi', 50.0) if market_data else 50.0
        volume = market_data.get('volume_24h', 1000.0) if market_data else 1000.0

        # 🎯 ENHANCED ROW with directional data
        row = [
            trade_result.timestamp,           # 1
            trade_result.asset,               # 2 - NEW
            trade_result.direction,           # 3 - NEW
            trade_result.action,              # 4 - NEW
            trade_result.amount_in,           # 5
            trade_result.amount_out,          # 6
            trade_result.pnl,                 # 7 - NEW
            trade_result.pnl_percentage,      # 8 - NEW
            trade_result.price_entry,         # 9 - NEW
            trade_result.price_exit,          # 10 - NEW
            trade_result.price_impact,        # 11
            current_price,                    # 12
            volume,                          # 13
            current_rsi,                     # 14
            trade_result.strategy_used,       # 15 - NEW
            trade_result.profitable,          # 16
            trade_result.position_duration,   # 17 - NEW
            trade_result.risk_level           # 18 - NEW
        ]

        # Save to CSV
        try:
            with open(MEMORY_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists or os.stat(MEMORY_FILE).st_size == 0:
                    writer.writerow(headers)
                writer.writerow(row)

            print(f"✅ Directional trade saved to CSV")

        except Exception as e:
            print(f"❌ CSV backup failed: {e}")

    def get_recent_transactions_hybrid(self, limit: int = 100) -> Optional[Dict]:
        """Get recent transactions with directional support"""

        # Try PostgreSQL first
        if self.db_available and self.db_manager:
            try:
                df = self.db_manager.get_recent_transactions(limit)
                if len(df) > 0:
                    return {
                        'source': 'postgresql',
                        'data': df,
                        'count': len(df)
                    }
            except Exception as e:
                print(f"⚠️ PostgreSQL read error: {e}")

        # Fallback to CSV
        try:
            if os.path.exists(MEMORY_FILE):
                import pandas as pd
                df = pd.read_csv(MEMORY_FILE)
                if len(df) > 0:
                    df = df.tail(limit)  # Get last N records
                    return {
                        'source': 'csv',
                        'data': df,
                        'count': len(df)
                    }
        except Exception as e:
            print(f"⚠️ CSV read error: {e}")

        return None

    def get_position_status(self, asset: str = None) -> Dict:
        """🎯 Get current position status"""
        if asset:
            position = self.position_tracker.get_open_position(asset)
            return {'asset': asset, 'position': position}
        else:
            return {
                'all_positions': self.position_tracker.get_all_open_positions(),
                'summary': self.position_tracker.get_position_summary()
            }

    def get_database_status(self) -> Dict:
        """Get comprehensive database status"""
        status = {
            'postgresql_available': self.db_available,
            'csv_available': os.path.exists(MEMORY_FILE),
            'postgresql_count': 0,
            'csv_count': 0,
            'migration_needed': False,
            'trading_mode': self.trading_mode,
            'open_positions': len(self.position_tracker.open_positions)
        }

        # PostgreSQL stats
        if self.db_available and self.db_manager:
            try:
                status['postgresql_count'] = self.db_manager.get_transaction_count()
                db_stats = self.db_manager.get_database_stats()
                status['postgresql_stats'] = db_stats
            except Exception as e:
                print(f"⚠️ Error getting PostgreSQL stats: {e}")

        # CSV stats
        if os.path.exists(MEMORY_FILE):
            try:
                import pandas as pd
                df = pd.read_csv(MEMORY_FILE)
                status['csv_count'] = len(df)
            except Exception as e:
                print(f"⚠️ Error getting CSV stats: {e}")

        # Position summary
        status['position_summary'] = self.position_tracker.get_position_summary()

        # Check if migration needed
        if status['csv_count'] > 0 and status['postgresql_count'] == 0:
            status['migration_needed'] = True

        return status


# 🔄 LEGACY COMPATIBILITY FUNCTIONS
def simulate_trade(settings):
    """Wrapper function for legacy compatibility"""
    executor = EnhancedDirectionalTradeExecutor()
    return executor.execute_trade(settings)


# Global instance for sharing
_global_executor = EnhancedDirectionalTradeExecutor()


def get_trade_executor():
    """Get global trade executor instance"""
    return _global_executor