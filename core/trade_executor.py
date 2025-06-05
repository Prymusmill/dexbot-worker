# core/trade_executor.py - ENHANCED with PostgreSQL + CSV backup + REALISTIC P&L
import csv
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
import random

MEMORY_FILE = "data/memory.csv"


@dataclass
class TradeResult:
    timestamp: str
    input_token: str
    output_token: str
    amount_in: float
    amount_out: float
    price_impact: float
    market_price: float
    spread: float
    signal_strength: float
    strategy_used: str
    profitable: bool


class EnhancedTradeExecutor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.last_market_data = None
        self.db_manager = None
        self.db_available = False

        # Initialize database connection
        self._init_database()

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

    def update_market_data(self, market_data: Dict):
        """Aktualizuj dane rynkowe"""
        self.last_market_data = market_data

    def execute_trade(self, settings: Dict,
                      market_data: Optional[Dict] = None) -> Optional[TradeResult]:
        """Wykonaj transakcję z zapisem do PostgreSQL + CSV backup"""

        # Użyj przekazanych danych lub ostatnich dostępnych
        current_market_data = market_data or self.last_market_data

        if current_market_data and current_market_data.get('price', 0) > 0:
            return self._execute_market_based_trade(
                settings, current_market_data)
        else:
            # Fallback do symulacji jeśli brak danych rynkowych
            return self._simulate_trade(settings)

    def _execute_market_based_trade(
            self, settings: Dict, market_data: Dict) -> TradeResult:
        """FIXED: Execute trade with REALISTIC P&L simulation"""
        timestamp = datetime.utcnow().isoformat()
        input_token = "SOL"
        output_token = "USDC"
        amount_in = settings["trade_amount_usd"]

        # Pobierz dane rynkowe
        market_price = market_data['price']
        bid_price = market_data.get('bid', market_price)
        ask_price = market_data.get('ask', market_price)
        spread = market_data.get('spread', 0.001)
        rsi = market_data.get('rsi', 50)
        volatility = market_data.get('volatility', 0.01)

        # FIXED: REALISTIC PRICE IMPACT based on market conditions
        base_impact = spread / market_price

        # Market condition impacts
        if rsi > 80:  # Overbought - worse execution
            market_condition_impact = random.uniform(0.0005, 0.002)
        elif rsi < 20:  # Oversold - better execution  
            market_condition_impact = random.uniform(-0.001, 0.0005)
        else:  # Normal conditions
            market_condition_impact = random.uniform(-0.0005, 0.001)

        # Volatility impact (higher vol = worse execution)
        volatility_impact = volatility * random.uniform(0.1, 0.5)

        # Random execution noise
        noise_impact = random.uniform(-0.0003, 0.0003)

        # TOTAL REALISTIC PRICE IMPACT
        price_impact = base_impact + market_condition_impact + volatility_impact + noise_impact

        # FIXED: Realistic execution with actual profit/loss variation
        if random.random() < 0.4:  # 40% chance of buying at ask (worse price)
            execution_price = ask_price * (1 + abs(price_impact))
            amount_out = amount_in / execution_price
            # When selling back, get bid price minus spread
            final_usd_value = amount_out * bid_price * (1 - abs(price_impact) * 0.5)
        else:  # 60% chance of better execution
            execution_price = market_price * (1 + price_impact)
            amount_out = amount_in / execution_price
            # Better sell execution
            final_usd_value = amount_out * market_price * (1 + price_impact * 0.3)

        # Ensure some realistic variation
        final_usd_value *= random.uniform(0.9985, 1.0025)  # ±0.15% to ±0.25% variation

        # ENHANCED: Add trading fees simulation
        trading_fees = amount_in * 0.001  # 0.1% fees
        net_amount_out = final_usd_value - trading_fees

        # Calculate profitability
        profitable = net_amount_out > amount_in

        # Add some market-based profitability bias
        if rsi < 25:  # Very oversold - higher chance of profit
            if random.random() < 0.7:  # 70% chance to be profitable
                net_amount_out = amount_in * random.uniform(1.0005, 1.002)
                profitable = True
        elif rsi > 75:  # Very overbought - higher chance of loss
            if random.random() < 0.7:  # 70% chance to be unprofitable
                net_amount_out = amount_in * random.uniform(0.998, 0.9995)
                profitable = False

        # Informacje o strategii
        strategy_used = "enhanced_realistic_simulation_v2"

        # Signal strength based on market conditions
        if rsi < 30 or rsi > 70:
            signal_strength = random.uniform(0.6, 0.9)
        else:
            signal_strength = random.uniform(0.3, 0.7)

        trade_result = TradeResult(
            timestamp=timestamp,
            input_token=input_token,
            output_token=output_token,
            amount_in=amount_in,
            amount_out=net_amount_out,
            price_impact=price_impact,
            market_price=market_price,
            spread=spread,
            signal_strength=signal_strength,
            strategy_used=strategy_used,
            profitable=profitable
        )

        # ENHANCED: Zapisz do PostgreSQL + CSV
        self._save_trade_data(trade_result, market_data)

        # ENHANCED LOGGING with realistic profit/loss details
        pnl = net_amount_out - amount_in
        pnl_pct = (pnl / amount_in) * 100
        profit_status = "✅ PROFIT" if profitable else "❌ LOSS"

        print(f"{profit_status} Market Trade: {input_token}→{output_token}")
        print(f"   💰 ${amount_in:.4f} → ${net_amount_out:.4f} (P&L: {pnl:+.4f} / {pnl_pct:+.2f}%)")
        print(f"   📊 Price: ${market_price:.4f}, Impact: {price_impact:.4f}, RSI: {rsi:.1f}")

        return trade_result

    def _simulate_trade(self, settings: Dict) -> TradeResult:
        """FIXED: Fallback simulation with realistic variation"""
        timestamp = datetime.utcnow().isoformat()
        input_token = "SOL"
        output_token = "USDC"
        amount_in = settings["trade_amount_usd"]

        # IMPROVED: More realistic simulation
        price_impact = round(random.uniform(-0.005, 0.005), 5)  # ±0.5% range
        amount_out = amount_in * (1 + price_impact)
        
        # Add trading fees
        trading_fees = amount_in * 0.001
        amount_out -= trading_fees
        
        profitable = amount_out > amount_in

        trade_result = TradeResult(
            timestamp=timestamp,
            input_token=input_token,
            output_token=output_token,
            amount_in=amount_in,
            amount_out=amount_out,
            price_impact=price_impact,
            market_price=100.0,  # Default price
            spread=0.001,
            signal_strength=0.5,
            strategy_used="simulation_fallback_v2",
            profitable=profitable
        )

        # ENHANCED: Zapisz do PostgreSQL + CSV
        self._save_trade_data(trade_result)

        pnl = amount_out - amount_in
        pnl_pct = (pnl / amount_in) * 100
        profit_status = "✅ PROFIT" if profitable else "❌ LOSS"

        print(f"{profit_status} Simulation: {trade_result.input_token}→{trade_result.output_token}")
        print(f"   💰 ${amount_in:.4f} → ${amount_out:.4f} (P&L: {pnl:+.4f} / {pnl_pct:+.2f}%)")

        return trade_result

    def _save_trade_data(self, trade_result: TradeResult,
                         market_data: Optional[Dict] = None):
        """ENHANCED: Save to both PostgreSQL and CSV"""

        # 1. Try to save to PostgreSQL first
        if self.db_available and self.db_manager:
            try:
                trade_data = {
                    'timestamp': trade_result.timestamp,
                    'input_token': trade_result.input_token,
                    'output_token': trade_result.output_token,
                    'amount_in': trade_result.amount_in,
                    'amount_out': trade_result.amount_out,
                    'price_impact': trade_result.price_impact
                }

                db_id = self.db_manager.save_transaction(
                    trade_data, market_data)
                if db_id:
                    print(f"✅ Saved to PostgreSQL (ID: {db_id})")
                else:
                    print("⚠️ PostgreSQL save failed, continuing with CSV")

            except Exception as e:
                print(f"⚠️ PostgreSQL error: {e}")
                print("🔄 Falling back to CSV only")

        # 2. Always save to CSV as backup
        self._save_to_csv(trade_result, market_data)

    def _save_to_csv(self, trade_result: TradeResult,
                     market_data: Optional[Dict] = None):
        """Save to CSV file (backup method)"""
        os.makedirs("data", exist_ok=True)
        file_exists = os.path.isfile(MEMORY_FILE)

        # Headers - MUSZĄ być dokładnie 9 kolumn
        headers = [
            "timestamp",
            "input_token",
            "output_token",
            "amount_in",
            "amount_out",
            "price_impact",
            "price",
            "volume",
            "rsi"]

        # Pobierz dane z market_data lub użyj wartości domyślnych
        current_price = market_data.get(
            'price', trade_result.market_price) if market_data else trade_result.market_price
        current_rsi = market_data.get('rsi', 50.0) if market_data else 50.0
        volume = trade_result.amount_in  # Volume jako wartość transakcji

        # Row - MUSI mieć dokładnie 9 wartości (pasujących do headers)
        row = [
            trade_result.timestamp,           # 1
            trade_result.input_token,         # 2
            trade_result.output_token,        # 3
            trade_result.amount_in,           # 4
            trade_result.amount_out,          # 5
            trade_result.price_impact,        # 6
            current_price,                    # 7
            volume,                          # 8
            current_rsi                      # 9
        ]

        # Zapisz do CSV
        try:
            with open(MEMORY_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists or os.stat(MEMORY_FILE).st_size == 0:
                    writer.writerow(headers)
                writer.writerow(row)

            print(f"✅ Saved to CSV backup")

        except Exception as e:
            print(f"❌ CSV backup failed: {e}")

    def get_recent_transactions_hybrid(
            self, limit: int = 100) -> Optional[Dict]:
        """Get recent transactions from PostgreSQL with CSV fallback"""

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

    def get_database_status(self) -> Dict:
        """Get comprehensive database status"""
        status = {
            'postgresql_available': self.db_available,
            'csv_available': os.path.exists(MEMORY_FILE),
            'postgresql_count': 0,
            'csv_count': 0,
            'migration_needed': False
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

        # Check if migration needed
        if status['csv_count'] > 0 and status['postgresql_count'] == 0:
            status['migration_needed'] = True

        return status

# Funkcja kompatybilności wstecznej


def simulate_trade(settings):
    """Wrapper function dla kompatybilności z istniejącym kodem"""
    executor = EnhancedTradeExecutor()
    return executor.execute_trade(settings)


# Global instance for sharing market data
_global_executor = EnhancedTradeExecutor()


def get_trade_executor():
    """Pobierz globalną instancję trade executor"""
    return _global_executor