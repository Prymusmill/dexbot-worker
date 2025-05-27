# run_worker.py - Enhanced with real-time market data
import os
import sys
import time
import json
import csv
from datetime import datetime
from typing import Dict

# Wyłącz git checks
os.environ['GIT_PYTHON_REFRESH'] = 'quiet'
os.environ['GIT_PYTHON_GIT_EXECUTABLE'] = ''

# Import local modules
try:
    from config.settings import SETTINGS as settings
    from core.trade_executor import get_trade_executor
    from core.market_data import create_market_data_service, TradingSignals
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

STATE_FILE = "data/state.json"
MEMORY_FILE = "data/memory.csv"

class TradingBot:
    def __init__(self):
        self.trade_executor = get_trade_executor()
        self.market_service = None
        self.latest_market_data = None
        self.trading_signals = TradingSignals()
        self.state = {"count": 0}
        
    def on_market_data_update(self, market_data: Dict):
        """Callback wywoływany przy każdej aktualizacji danych rynkowych"""
        self.latest_market_data = market_data
        self.trade_executor.update_market_data(market_data)
        
        # Log market data co 30 sekund
        if hasattr(self, '_last_market_log'):
            if (datetime.now() - self._last_market_log).seconds >= 30:
                self._log_market_data(market_data)
                self._last_market_log = datetime.now()
        else:
            self._log_market_data(market_data)
            self._last_market_log = datetime.now()
    
    def _log_market_data(self, market_data: Dict):
        """Loguj dane rynkowe"""
        price = market_data.get('price', 0)
        rsi = market_data.get('rsi', 0)
        trend = 'up' if market_data.get('price_change_24h', 0) > 0 else 'down'
        
        print(f"📊 Market: SOL/USDC ${price:.4f}, RSI: {rsi:.1f}, 24h: {trend}")
    
    def should_execute_trade(self) -> bool:
        """Określ czy wykonać transakcję na podstawie sygnałów rynkowych"""
        if not self.latest_market_data:
            return True  # Fallback - wykonuj jak wcześniej
        
        # Analizuj sygnały rynkowe
        signals = self.trading_signals.analyze_market_conditions(self.latest_market_data)
        
        # Proste zasady wykonywania transakcji:
        # 1. Zawsze wykonuj jeśli confidence > 0.3
        # 2. Wykonuj losowo jeśli confidence < 0.3
        confidence = signals.get('confidence', 0.5)
        
        if confidence > 0.3:
            return True
        else:
            # 70% szans na wykonanie przy niskim confidence
            import random
            return random.random() < 0.7
    
    def execute_trade_cycle(self):
        """Wykonaj cykl 30 transakcji"""
        print(f"\n🔄 Cykl - wykonuję 30 transakcji...")
        
        for i in range(30):
            try:
                print(f"🔹 Transakcja {self.state['count'] + 1} (#{i+1}/30)")
                
                # Sprawdź czy wykonać transakcję
                if self.should_execute_trade():
                    # Wykonaj transakcję z aktualnymi danymi rynkowymi
                    self.trade_executor.execute_trade(settings, self.latest_market_data)
                    self.state["count"] += 1
                else:
                    print("⏸️ Pominięto transakcję - niekorzystne warunki rynkowe")
                
                # Sprawdź status co 10 transakcji
                if (i + 1) % 10 == 0:
                    self.check_file_status()
                
                # Przerwa między transakcjami
                time.sleep(0.25)
                
            except Exception as e:
                print(f"❌ Błąd podczas transakcji: {e}")
                continue
    
    def check_file_status(self):
        """Sprawdź status plików"""
        if os.path.exists(MEMORY_FILE):
            size = os.stat(MEMORY_FILE).st_size
            try:
                with open(MEMORY_FILE, "r") as f:
                    lines = sum(1 for _ in f)
                print(f"📁 {MEMORY_FILE}: {size:,} bajtów, {lines:,} linii")
            except:
                print(f"📁 {MEMORY_FILE}: {size:,} bajtów")
    
    def load_state(self):
        """Załaduj stan aplikacji"""
        try:
            if os.path.exists(STATE_FILE):
                with open(STATE_FILE, "r") as f:
                    self.state = json.load(f)
                    if "count" not in self.state:
                        self.state["count"] = 0
                print(f"📂 Załadowano stan: {self.state['count']} transakcji")
            else:
                print("📝 Tworzę nowy plik stanu")
                self.state = {"count": 0}
        except Exception as e:
            print(f"⚠️ Błąd wczytywania stanu: {e}")
            self.state = {"count": 0}
    
    def save_state(self):
        """Zapisz stan aplikacji"""
        try:
            os.makedirs("data", exist_ok=True)
            with open(STATE_FILE, "w") as f:
                json.dump(self.state, f)
            return True
        except Exception as e:
            print(f"❌ Błąd zapisu stanu: {e}")
            return False
    
    def start(self):
        """Uruchom bota tradingowego"""
        print("🚀 Uruchamiam Enhanced DexBot Worker z Real-time Market Data...")
        print(f"⏰ Start: {datetime.now()}")
        
        # Utwórz katalogi
        os.makedirs("data", exist_ok=True)
        os.makedirs("data/results", exist_ok=True)
        
        # Załaduj stan
        self.load_state()
        start_count = self.state["count"]
        
        # Uruchom market data service
        print("🌐 Łączę z Binance WebSocket...")
        self.market_service = create_market_data_service(self.on_market_data_update)
        
        if not self.market_service:
            print("⚠️ Nie udało się połączyć z market data - kontynuuję w trybie symulacji")
        else:
            print("✅ Połączony z Binance - używam real-time data")
            time.sleep(5)  # Daj czas na pierwsze dane
        
        print(f"🎯 Rozpoczynam od transakcji #{start_count + 1}")
        
        # Główna pętla
        cycle = 0
        try:
            while True:
                cycle += 1
                
                # Wykonaj cykl transakcji
                self.execute_trade_cycle()
                
                # Zapisz stan
                if self.save_state():
                    print(f"💾 Stan zapisany: {self.state['count']} transakcji")
                
                # Status podsumowujący
                total_executed = self.state["count"] - start_count
                print(f"\n📈 Statystyki sesji:")
                print(f"   • Łącznie wykonano: {total_executed} nowych transakcji")
                print(f"   • Całkowita liczba: {self.state['count']:,} transakcji")
                print(f"   • Cykli ukończonych: {cycle}")
                
                if self.latest_market_data:
                    price = self.latest_market_data.get('price', 0)
                    rsi = self.latest_market_data.get('rsi', 0)
                    print(f"   • Aktualna cena SOL: ${price:.4f}")
                    print(f"   • RSI: {rsi:.1f}")
                
                # Przerwa między cyklami
                print("⏳ Przerwa 60 sekund przed kolejnym cyklem...")
                time.sleep(60)
                
        except KeyboardInterrupt:
            print("\n🛑 Zatrzymano przez użytkownika")
        except Exception as e:
            print(f"\n💥 Nieoczekiwany błąd: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Zamknij market data service
            if self.market_service:
                self.market_service.stop_stream()
            
            # Zapisz stan na koniec
            if self.save_state():
                print(f"💾 Końcowy zapis stanu: {self.state['count']} transakcji")
            
            print(f"🏁 Worker zakończony. Łącznie: {self.state['count']:,} transakcji")

if __name__ == "__main__":
    bot = TradingBot()
    bot.start()