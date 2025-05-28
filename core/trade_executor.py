# core/trade_executor.py - Enhanced with real market data
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
        
    def update_market_data(self, market_data: Dict):
        """Aktualizuj dane rynkowe"""
        self.last_market_data = market_data
        
    def execute_trade(self, settings: Dict, market_data: Optional[Dict] = None) -> Optional[TradeResult]:
        """Wykonaj transakcję z uwzględnieniem danych rynkowych"""
        
        # Użyj przekazanych danych lub ostatnich dostępnych
        current_market_data = market_data or self.last_market_data
        
        if current_market_data and current_market_data.get('price', 0) > 0:
            return self._execute_market_based_trade(settings, current_market_data)
        else:
            # Fallback do symulacji jeśli brak danych rynkowych
            return self._simulate_trade(settings)
    
    def _execute_market_based_trade(self, settings: Dict, market_data: Dict) -> TradeResult:
        """Wykonaj transakcję opartą na rzeczywistych danych rynkowych"""
        timestamp = datetime.utcnow().isoformat()
        input_token = "SOL"
        output_token = "USDC"
        amount_in = settings["trade_amount_usd"]
        
        # Pobierz dane rynkowe
        market_price = market_data['price']
        bid_price = market_data.get('bid', market_price)
        ask_price = market_data.get('ask', market_price)
        spread = market_data.get('spread', 0.001)
        
        # Oblicz realistic price impact na podstawie spreadu i volatility
        volatility = market_data.get('volatility', 0.01)
        base_impact = spread / market_price  # Podstawowy impact z spreadu
        volatility_impact = random.uniform(-volatility, volatility) * 0.1  # 10% volatility impact
        
        price_impact = base_impact + volatility_impact
        
        # Symuluj wykonanie zlecenia
        # Buy order - płacimy ask price + impact
        execution_price = ask_price * (1 + price_impact)
        amount_out = amount_in / execution_price
        
        # Faktyczny P&L uwzględniając spread
        amount_out_usd = amount_out * bid_price  # Gdybyśmy od razu sprzedawali
        profitable = amount_out_usd > amount_in
        
        # Informacje o strategii (dla przyszłych ML models)
        signal_strength = random.uniform(0.1, 0.9)  # Placeholder for future ML signals
        strategy_used = "market_based_v1"
        
        trade_result = TradeResult(
            timestamp=timestamp,
            input_token=input_token,
            output_token=output_token,
            amount_in=amount_in,
            amount_out=amount_out_usd,  # Zapisujemy w USD dla łatwości analizy
            price_impact=price_impact,
            market_price=market_price,
            spread=spread,
            signal_strength=signal_strength,
            strategy_used=strategy_used,
            profitable=profitable
        )
        
        # Zapisz do CSV
        self._save_to_csv(trade_result, market_data)
        
        print(f"✅ Market Trade: {trade_result.input_token}→{trade_result.output_token}, "
              f"${trade_result.amount_in:.4f}→${trade_result.amount_out:.4f}, "
              f"Price: ${market_price:.4f}, Impact: {price_impact:.4f}")
        
        return trade_result
    
    def _simulate_trade(self, settings: Dict) -> TradeResult:
        """Fallback symulacja gdy brak danych rynkowych"""
        timestamp = datetime.utcnow().isoformat()
        input_token = "SOL"
        output_token = "USDC"
        amount_in = settings["trade_amount_usd"]

        # Podstawowa symulacja (twoja oryginalna logika)
        price_impact = round(random.uniform(-0.01, 0.01), 5)
        amount_out = round(amount_in * (1 + price_impact), 5)
        profitable = amount_out > amount_in

        trade_result = TradeResult(
            timestamp=timestamp,
            input_token=input_token,
            output_token=output_token,
            amount_in=amount_in,
            amount_out=amount_out,
            price_impact=price_impact,
            market_price=0.0,  # Brak danych rynkowych
            spread=0.0,
            signal_strength=0.5,
            strategy_used="simulation_fallback",
            profitable=profitable
        )

        self._save_to_csv(trade_result)
        
        print(f"✅ Simulation: {trade_result.input_token}→{trade_result.output_token}, "
              f"${trade_result.amount_in:.4f}→${trade_result.amount_out:.4f}, "
              f"Impact: {price_impact:.4f}")
        
        return trade_result
    
    def _save_to_csv(self, trade_result: TradeResult, market_data: Optional[Dict] = None):
        """Zapisz wynik transakcji do CSV"""
        os.makedirs("data", exist_ok=True)
        file_exists = os.path.isfile(MEMORY_FILE)
        
        # Extended row z dodatkowymi danymi
        row = [
            trade_result.timestamp,
            trade_result.input_token,
            trade_result.output_token,
            trade_result.amount_in,
            trade_result.amount_out,
            trade_result.price_impact
        ]
        
        # Nagłówki (zachowaj kompatybilność z starym formatem)
        headers = ["timestamp", "input_token", "output_token", "amount_in", "amount_out", "price_impact", "price", "rsi"]
        
        with open(MEMORY_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists or os.stat(MEMORY_FILE).st_size == 0:
                writer.writerow(headers)
            writer.writerow(row)

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