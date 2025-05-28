# run_worker.py - Enhanced with real-time market data and ML integration (FIXED)
import os
import sys
import time
import json
import csv
import pandas as pd
import threading
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

# ML Integration with safe fallback - FIXED: Initialize ML_AVAILABLE first
ML_AVAILABLE = False
try:
    from ml.price_predictor import MLTradingIntegration
    ML_AVAILABLE = True
    print("✅ ML modules available")
except ImportError as e:
    print(f"⚠️ ML modules not available: {e}")
    ML_AVAILABLE = False

STATE_FILE = "data/state.json"
MEMORY_FILE = "data/memory.csv"

class TradingBot:
    def __init__(self):
        self.trade_executor = get_trade_executor()
        self.market_service = None
        self.latest_market_data = None
        self.trading_signals = TradingSignals()
        self.state = {"count": 0}
        
        # Initialize ML attributes regardless of availability
        self.ml_predictions = {}
        self.ml_prediction_count = 0
        self.last_ml_training = None
        
        # ML Integration setup
        if ML_AVAILABLE:
            try:
                self.ml_integration = MLTradingIntegration()
                print("🤖 ML integration initialized")
            except Exception as e:
                print(f"⚠️ ML integration failed: {e}")
                self.ml_integration = None
        else:
            self.ml_integration = None
            
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
        
        # Safe ML info check
        ml_info = ""
        if self.ml_predictions:
            direction = self.ml_predictions.get('direction', 'unknown')
            confidence = self.ml_predictions.get('confidence', 0)
            ml_info = f", ML: {direction.upper()} ({confidence:.2f})"
        
        print(f"📊 Market: SOL/USDC ${price:.4f}, RSI: {rsi:.1f}, 24h: {trend}{ml_info}")
    
    def update_ml_predictions(self):
        """Update ML predictions based on recent data"""
        if not self.ml_integration:
            return
        
        try:
            # Get recent trading data for ML prediction
            if os.path.exists(MEMORY_FILE):
                df = pd.read_csv(MEMORY_FILE)
                
                if len(df) >= 100:  # Zawsze sprawdzaj jeśli masz dane
                    print("🤖 Forcing ML predictions update...", flush=True)
                    # Get ensemble prediction
                    prediction = self.ml_integration.get_ensemble_prediction(df.tail(200))
                    
                    if 'predicted_price' in prediction:
                        self.ml_predictions = prediction
                        self.ml_prediction_count += 1
                        
                        # Log prediction periodically
                        if self.ml_prediction_count % 10 == 1:  # Every 10th prediction
                            direction = prediction['direction']
                            confidence = prediction['confidence']
                            price_change = prediction['price_change_pct']
                            
                            print(f"🔮 ML Prediction #{self.ml_prediction_count}: {direction.upper()} "
                                  f"({price_change:+.2f}%, confidence: {confidence:.2f})")
                    
                    # Check if models need retraining
                    if self.ml_integration.should_retrain() and len(df) >= 500:
                        print("🤖 ML models need retraining - starting background process...")
                        threading.Thread(target=self._retrain_ml_models, args=(df,), daemon=True).start()
                else:
                    print(f"⚠️ Need more data for ML predictions ({len(df)}/100 transactions)")
        
        except Exception as e:
            print(f"⚠️ ML prediction error: {e}")

    def _retrain_ml_models(self, df):
        """Retrain ML models in background"""
        try:
            print("🔄 Retraining ML models (background process)...")
            results = self.ml_integration.train_all_models(df)
            
            successful_models = [name for name, result in results.items() 
                               if result.get('success')]
            
            if successful_models:
                print(f"✅ ML retraining complete. Successful models: {successful_models}")
                
                # Log performance metrics
                performance = self.ml_integration.get_model_performance()
                for model_name in successful_models:
                    if model_name in performance:
                        metrics = performance[model_name]
                        accuracy = metrics.get('accuracy', 0)
                        r2 = metrics.get('r2', 0)
                        print(f"   • {model_name}: Accuracy {accuracy:.1f}%, R² {r2:.3f}")
            else:
                print("⚠️ ML retraining failed for all models")
        
        except Exception as e:
            print(f"❌ ML retraining error: {e}")

    def should_execute_trade(self) -> bool:
        """Określ czy wykonać transakcję na podstawie sygnałów rynkowych i ML"""
        if not self.latest_market_data:
            return True  # Fallback - wykonuj jak wcześniej
        
        # Analizuj sygnały rynkowe
        signals = self.trading_signals.analyze_market_conditions(self.latest_market_data)
        base_confidence = signals.get('confidence', 0.5)
        
        # Enhance decision with ML predictions if available
        enhanced_confidence = base_confidence
        
        if self.ml_integration and self.ml_predictions and 'confidence' in self.ml_predictions:
            try:
                ml_confidence = self.ml_predictions['confidence']
                ml_direction = self.ml_predictions.get('direction', 'neutral')
                
                # Combine traditional signals with ML predictions
                if ml_confidence > 0.7:  # High ML confidence
                    if ml_direction == 'up':
                        enhanced_confidence = min(base_confidence + 0.3, 1.0)
                    else:
                        enhanced_confidence = max(base_confidence - 0.2, 0.0)
                else:
                    enhanced_confidence = base_confidence
                
                # Log enhanced decision making
                if abs(enhanced_confidence - base_confidence) > 0.1:
                    print(f"🧠 ML Enhanced Decision: {base_confidence:.2f} → {enhanced_confidence:.2f} "
                          f"(ML: {ml_direction}, {ml_confidence:.2f})")
            except Exception as e:
                print(f"⚠️ ML decision enhancement error: {e}")
                enhanced_confidence = base_confidence
        
        # Decision logic
        if enhanced_confidence > 0.4:
            return True
        else:
            # Reduced randomness when confidence is low
            import random
            return random.random() < 0.6
    
    def execute_trade_cycle(self):
        """Wykonaj cykl 30 transakcji z ML predictions"""
        print(f"\n🔄 Cykl - wykonuję 30 transakcji...")
    
        # ZMIANA: Sprawdzaj ML predictions częściej i z debugowaniem
        if self.ml_integration:
            print(f"🔍 DEBUG: Checking ML predictions. Current trades: {self.state['count']}", flush=True)
        
            # Sprawdź czy plik memory.csv istnieje
            if os.path.exists(MEMORY_FILE):
                try:
                    df = pd.read_csv(MEMORY_FILE)
                    print(f"📊 DEBUG: Memory file has {len(df)} rows", flush=True)
                    print(f"📋 DEBUG: Available columns: {list(df.columns)}", flush=True)
                
                    # ZAWSZE próbuj aktualizować ML jeśli mamy 100+ transakcji
                    if len(df) >= 100:
                        print("🤖 Forcing ML predictions update...", flush=True)
                        self.update_ml_predictions()
                    else:
                        print(f"⚠️ Need more data: {len(df)}/100 transactions in memory.csv", flush=True)
                    
                except Exception as e:
                    print(f"❌ Error reading memory.csv: {e}", flush=True)
            else:
                print(f"❌ Memory file not found: {MEMORY_FILE}", flush=True)
        else:
            print("⚠️ ML integration not available", flush=True)
        
        executed_in_cycle = 0
        
        for i in range(30):
            try:
                print(f"🔹 Transakcja {self.state['count'] + 1} (#{i+1}/30)")
                
                # Sprawdź czy wykonać transakcję (with ML enhancement)
                if self.should_execute_trade():
                    # Wykonaj transakcję z aktualnymi danymi rynkowymi
                    self.trade_executor.execute_trade(settings, self.latest_market_data)
                    self.state["count"] += 1
                    executed_in_cycle += 1
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
        
        print(f"✅ Cykl zakończony: {executed_in_cycle}/30 transakcji wykonanych")
    
    def check_file_status(self):
        """Sprawdź status plików z szczegółowym debugowaniem"""
        if os.path.exists(MEMORY_FILE):
            size = os.stat(MEMORY_FILE).st_size
            try:
                df = pd.read_csv(MEMORY_FILE)
                lines = len(df)
                
                print(f"📁 {MEMORY_FILE}: {size:,} bajtów, {lines:,} wierszy", flush=True)
                
                # Pokaż ostatnie 5 transakcji
                if len(df) > 0:
                    print(f"📊 Ostatnie 5 transakcji:", flush=True)
                    # Wybierz najważniejsze kolumny do wyświetlenia
                    display_cols = []
                    available_cols = df.columns.tolist()
                    
                    # Sprawdź które kolumny są dostępne i dodaj je do wyświetlania
                    priority_cols = ['timestamp', 'price', 'action', 'pnl', 'volume', 'rsi', 'sma_20']
                    for col in priority_cols:
                        if col in available_cols:
                            display_cols.append(col)
                    
                    # Jeśli nie ma priorytetowych kolumn, weź pierwsze 5
                    if not display_cols:
                        display_cols = available_cols[:min(5, len(available_cols))]
                    
                    try:
                        print(df[display_cols].tail(5).to_string(index=False), flush=True)
                    except Exception as e:
                        print(f"   Błąd wyświetlania kolumn {display_cols}: {e}", flush=True)
                        print(df.tail(5).to_string(index=False), flush=True)
                    
                    # Statystyki finansowe
                    if 'pnl' in df.columns:
                        total_pnl = df['pnl'].sum()
                        avg_pnl = df['pnl'].mean()
                        max_profit = df['pnl'].max()
                        max_loss = df['pnl'].min()
                        winning_trades = len(df[df['pnl'] > 0])
                        losing_trades = len(df[df['pnl'] < 0])
                        win_rate = (winning_trades / len(df)) * 100 if len(df) > 0 else 0
                        
                        print(f"💰 PnL Summary:", flush=True)
                        print(f"   • Total PnL: ${total_pnl:.6f}", flush=True)
                        print(f"   • Average PnL/trade: ${avg_pnl:.6f}", flush=True)
                        print(f"   • Best trade: ${max_profit:.6f}", flush=True)
                        print(f"   • Worst trade: ${max_loss:.6f}", flush=True)
                        print(f"   • Win rate: {win_rate:.1f}% ({winning_trades}W/{losing_trades}L)", flush=True)
                    
                    # Informacje o cenach
                    if 'price' in df.columns:
                        current_price = df['price'].iloc[-1]
                        price_change = df['price'].iloc[-1] - df['price'].iloc[-min(10, len(df))]
                        price_change_pct = (price_change / df['price'].iloc[-min(10, len(df))]) * 100
                        
                        print(f"📈 Price Analysis (last 10 trades):", flush=True)
                        print(f"   • Current price: ${current_price:.4f}", flush=True)
                        print(f"   • Price change: ${price_change:+.4f} ({price_change_pct:+.2f}%)", flush=True)
                    
                    # Dostępne kolumny
                    print(f"📋 Available columns: {', '.join(available_cols)}", flush=True)
                else:
                    print("⚠️ Plik memory.csv jest pusty", flush=True)
                    
            except Exception as e:
                print(f"❌ Błąd czytania {MEMORY_FILE}: {e}", flush=True)
                # Fallback do prostego sprawdzenia
                try:
                    with open(MEMORY_FILE, "r") as f:
                        lines = sum(1 for _ in f)
                    print(f"📁 {MEMORY_FILE}: {size:,} bajtów, {lines:,} linii (fallback)", flush=True)
                except Exception as e2:
                    print(f"📁 {MEMORY_FILE}: {size:,} bajtów (błąd liczenia linii: {e2})", flush=True)
        else:
            print(f"❌ Plik {MEMORY_FILE} nie istnieje", flush=True)
    
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
    
    def get_system_status(self):
        """Get comprehensive system status"""
        status = {
            'total_trades': self.state["count"],
            'market_connected': self.market_service is not None,
            'latest_price': self.latest_market_data.get('price', 0) if self.latest_market_data else 0,
            'ml_available': ML_AVAILABLE and self.ml_integration is not None,
            'ml_predictions_count': self.ml_prediction_count
        }
        
        if ML_AVAILABLE and self.ml_integration:
            try:
                performance = self.ml_integration.get_model_performance()
                status['ml_models'] = list(performance.keys())
                status['ml_last_training'] = self.last_ml_training
            except Exception as e:
                print(f"⚠️ Error getting ML performance: {e}")
                status['ml_models'] = []
        
        return status
    
    def debug_ml_status(self):
        """Debug ML system status z pełnymi szczegółami"""
        print("\n🔍 ML DEBUG STATUS:", flush=True)
        print(f"   • ML_AVAILABLE: {ML_AVAILABLE}", flush=True)
        print(f"   • ml_integration: {self.ml_integration is not None}", flush=True)
        print(f"   • Memory file exists: {os.path.exists(MEMORY_FILE)}", flush=True)
        
        if os.path.exists(MEMORY_FILE):
            try:
                df = pd.read_csv(MEMORY_FILE)
                print(f"   • Memory file rows: {len(df)}", flush=True)
                print(f"   • Memory file columns: {list(df.columns)}", flush=True)
                
                if len(df) > 0:
                    print(f"   • Date range: {df.iloc[0].get('timestamp', 'N/A')} → {df.iloc[-1].get('timestamp', 'N/A')}", flush=True)
                    
                    # Sprawdź jakość danych dla ML
                    required_ml_cols = ['price', 'volume', 'rsi']
                    missing_cols = [col for col in required_ml_cols if col not in df.columns]
                    if missing_cols:
                        print(f"   ⚠️ Missing ML columns: {missing_cols}", flush=True)
                    else:
                        print(f"   ✅ All required ML columns present", flush=True)
                    
                    # Sprawdź czy są puste wartości
                    null_counts = df.isnull().sum()
                    if null_counts.sum() > 0:
                        print(f"   ⚠️ Null values found: {dict(null_counts[null_counts > 0])}", flush=True)
                    else:
                        print(f"   ✅ No null values in data", flush=True)
                
                print(f"   • Last 3 rows preview:", flush=True)
                print(df.tail(3).to_string(index=False), flush=True)
                
            except Exception as e:
                print(f"   • Error reading memory: {e}", flush=True)
        
        if self.ml_integration:
            try:
                models = self.ml_integration.get_model_performance()
                print(f"   • Available models: {list(models.keys())}", flush=True)
                
                # Szczegóły o modelach
                for model_name, performance in models.items():
                    accuracy = performance.get('accuracy', 0)
                    r2 = performance.get('r2', 0)
                    print(f"     - {model_name}: Accuracy {accuracy:.1f}%, R² {r2:.3f}", flush=True)
                    
            except Exception as e:
                print(f"   • Error getting models: {e}", flush=True)
        
        # Sprawdź ML predictions
        if self.ml_predictions:
            print(f"   • Current ML predictions:", flush=True)
            for key, value in self.ml_predictions.items():
                print(f"     - {key}: {value}", flush=True)
            print(f"   • Total predictions made: {self.ml_prediction_count}", flush=True)
        else:
            print(f"   • No ML predictions yet", flush=True)
        
        # Sprawdź czy katalogi ML istnieją
        ml_dirs = ['ml', 'ml/models', 'data']
        for dir_path in ml_dirs:
            exists = os.path.exists(dir_path)
            print(f"   • Directory {dir_path}: {'✅' if exists else '❌'}", flush=True)
        
        print("", flush=True)  # Dodatkowa linia dla czytelności
    
    def start(self):
        """Uruchom bota tradingowego"""
        print("🚀 Uruchamiam Enhanced DexBot Worker z Real-time Market Data i ML...")
        print(f"⏰ Start: {datetime.now()}")
        
        # Utwórz katalogi
        os.makedirs("data", exist_ok=True)
        os.makedirs("data/results", exist_ok=True)
        if ML_AVAILABLE:
            os.makedirs("ml", exist_ok=True)
            os.makedirs("ml/models", exist_ok=True)
        
        # Załaduj stan
        self.load_state()
        start_count = self.state["count"]
        
        # Debug ML status na początku
        self.debug_ml_status()
        
        # Initial ML setup if available
        if ML_AVAILABLE and self.ml_integration and start_count > 500:
            print("🤖 Checking for existing ML models...")
            # Could add logic to load existing models here
            
        # Uruchom market data service
        print("🌐 Łączę z Binance WebSocket...")
        self.market_service = create_market_data_service(self.on_market_data_update)
        
        if not self.market_service:
            print("⚠️ Nie udało się połączyć z market data - kontynuuję w trybie symulacji")
        else:
            print("✅ Połączony z Binance - używam real-time data")
            time.sleep(5)  # Daj czas na pierwsze dane
        
        # Initial ML prediction update if enough data
        if ML_AVAILABLE and self.ml_integration and start_count >= 100:
            print("🤖 Generating initial ML predictions...")
            self.update_ml_predictions()
        
        # DODATKOWE: Wymuszenie ML update jeśli mamy dużo danych
        if ML_AVAILABLE and self.ml_integration:
            print("🤖 FORCING initial ML predictions check...")
            self.update_ml_predictions()
        
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
                print(f"\n📈 Statystyki sesji:", flush=True)
                print(f"   • Łącznie wykonano: {total_executed} nowych transakcji", flush=True)
                print(f"   • Całkowita liczba: {self.state['count']:,} transakcji", flush=True)
                print(f"   • Cykli ukończonych: {cycle}", flush=True)
                
                if self.latest_market_data:
                    price = self.latest_market_data.get('price', 0)
                    rsi = self.latest_market_data.get('rsi', 0)
                    print(f"   • Aktualna cena SOL: ${price:.4f}", flush=True)
                    print(f"   • RSI: {rsi:.1f}", flush=True)
                
                # ML status info
                if ML_AVAILABLE and self.ml_integration and self.ml_predictions:
                    try:
                        ml_direction = self.ml_predictions.get('direction', 'unknown')
                        ml_confidence = self.ml_predictions.get('confidence', 0)
                        predicted_price = self.ml_predictions.get('predicted_price', 0)
                        print(f"   • ML Prediction: {ml_direction.upper()} → ${predicted_price:.4f} ({ml_confidence:.2f})", flush=True)
                    except Exception as e:
                        print(f"   • ML Status: Error displaying prediction ({e})", flush=True)
                
                # System status every 10 cycles
                if cycle % 10 == 0:
                    try:
                        status = self.get_system_status()
                        print(f"\n🔍 System Status (Cycle {cycle}):", flush=True)
                        print(f"   • Market Data: {'✅ Connected' if status['market_connected'] else '❌ Disconnected'}", flush=True)
                        if status['ml_available']:
                            print(f"   • ML Models: {len(status.get('ml_models', []))} active", flush=True)
                            print(f"   • ML Predictions: {status['ml_predictions_count']} generated", flush=True)
                        else:
                            print(f"   • ML Status: ❌ Not available", flush=True)
                    except Exception as e:
                        print(f"   • Status Error: {e}", flush=True)
                
                # Przerwa między cyklami
                print("⏳ Przerwa 60 sekund przed kolejnym cyklem...", flush=True)
                time.sleep(60)
                
        except KeyboardInterrupt:
            print("\n🛑 Zatrzymano przez użytkownika", flush=True)
        except Exception as e:
            print(f"\n💥 Nieoczekiwany błąd: {e}", flush=True)
            import traceback
            traceback.print_exc()
        finally:
            # Zamknij market data service
            if self.market_service:
                self.market_service.stop_stream()
            
            # Zapisz stan na koniec
            if self.save_state():
                print(f"💾 Końcowy zapis stanu: {self.state['count']} transakcji", flush=True)
            
            # Final system status
            try:
                final_status = self.get_system_status()
                print(f"\n🏁 Worker zakończony:", flush=True)
                print(f"   • Łączna liczba transakcji: {final_status['total_trades']:,}", flush=True)
                if final_status['ml_available']:
                    print(f"   • ML predictions wygenerowanych: {final_status['ml_predictions_count']}", flush=True)
                print(f"   • Ostatnia cena SOL: ${final_status['latest_price']:.4f}", flush=True)
                
                # Final file status check
                print(f"\n📁 Final File Status:", flush=True)
                self.check_file_status()
                
            except Exception as e:
                print(f"🏁 Worker zakończony (status error: {e})", flush=True)

if __name__ == "__main__":
    bot = TradingBot()
    bot.start()