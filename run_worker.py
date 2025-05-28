# run_worker.py - Enhanced with real-time market data and ML integration
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

# ML Integration
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
        
        # ML Integration
        if ML_AVAILABLE:
            self.ml_integration = MLTradingIntegration()
            self.ml_predictions = {}
            self.last_ml_training = None
            self.ml_prediction_count = 0
            print("🤖 ML integration initialized")
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
        
        # Add ML prediction info if available
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
                
                if len(df) >= 100:  # Need minimum data for prediction
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
        if self.ml_predictions and 'confidence' in self.ml_predictions:
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
        else:
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
        
        # Update ML predictions before trading cycle (every 3rd cycle)
        if self.ml_integration and self.state["count"] % 90 == 0:  # Every ~90 trades
            print("🤖 Updating ML predictions...")
            self.update_ml_predictions()
        
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
    
    def get_system_status(self):
        """Get comprehensive system status"""
        status = {
            'total_trades': self.state["count"],
            'market_connected': self.market_service is not None,
            'latest_price': self.latest_market_data.get('price', 0) if self.latest_market_data else 0,
            'ml_available': ML_AVAILABLE,
            'ml_predictions_count': self.ml_prediction_count if ML_AVAILABLE else 0
        }
        
        if ML_AVAILABLE and self.ml_integration:
            performance = self.ml_integration.get_model_performance()
            status['ml_models'] = list(performance.keys())
            status['ml_last_training'] = self.last_ml_training
        
        return status
    
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
        
        # Initial ML setup if available
        if ML_AVAILABLE and start_count > 500:
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
        if ML_AVAILABLE and start_count >= 100:
            print("🤖 Generating initial ML predictions...")
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
                print(f"\n📈 Statystyki sesji:")
                print(f"   • Łącznie wykonano: {total_executed} nowych transakcji")
                print(f"   • Całkowita liczba: {self.state['count']:,} transakcji")
                print(f"   • Cykli ukończonych: {cycle}")
                
                if self.latest_market_data:
                    price = self.latest_market_data.get('price', 0)
                    rsi = self.latest_market_data.get('rsi', 0)
                    print(f"   • Aktualna cena SOL: ${price:.4f}")
                    print(f"   • RSI: {rsi:.1f}")
                
                # ML status info
                if ML_AVAILABLE and self.ml_predictions:
                    ml_direction = self.ml_predictions.get('direction', 'unknown')
                    ml_confidence = self.ml_predictions.get('confidence', 0)
                    predicted_price = self.ml_predictions.get('predicted_price', 0)
                    print(f"   • ML Prediction: {ml_direction.upper()} → ${predicted_price:.4f} ({ml_confidence:.2f})")
                
                # System status every 10 cycles
                if cycle % 10 == 0:
                    status = self.get_system_status()
                    print(f"\n🔍 System Status (Cycle {cycle}):")
                    print(f"   • Market Data: {'✅ Connected' if status['market_connected'] else '❌ Disconnected'}")
                    if ML_AVAILABLE:
                        print(f"   • ML Models: {len(status.get('ml_models', []))} active")
                        print(f"   • ML Predictions: {status['ml_predictions_count']} generated")
                
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
            
            # Final system status
            final_status = self.get_system_status()
            print(f"\n🏁 Worker zakończony:")
            print(f"   • Łączna liczba transakcji: {final_status['total_trades']:,}")
            if ML_AVAILABLE:
                print(f"   • ML predictions wygenerowanych: {final_status['ml_predictions_count']}")
            print(f"   • Ostatnia cena SOL: ${final_status['latest_price']:.4f}")

if __name__ == "__main__":
    bot = TradingBot()
    bot.start()