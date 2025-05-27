# run_worker.py - Fixed version for Railway
import os
import sys

# Wyłącz git checks
os.environ['GIT_PYTHON_REFRESH'] = 'quiet'
os.environ['GIT_PYTHON_GIT_EXECUTABLE'] = ''

import time
import json
import csv
from datetime import datetime

# Import local modules
try:
    from config.settings import SETTINGS as settings
    from core.trade_executor import simulate_trade
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("🔧 Sprawdź czy pliki config/settings.py i core/trade_executor.py istnieją")
    sys.exit(1)

STATE_FILE = "data/state.json"
MEMORY_FILE = "data/memory.csv"

def ensure_data_directory():
    """Upewnij się że katalog data istnieje"""
    try:
        os.makedirs("data", exist_ok=True)
        os.makedirs("data/results", exist_ok=True)
        print("✅ Katalogi data/ utworzone")
        return True
    except Exception as e:
        print(f"❌ Błąd tworzenia katalogów: {e}")
        return False

def load_state():
    """Załaduj stan aplikacji"""
    try:
        if not os.path.exists(STATE_FILE):
            print("📝 Tworzę nowy plik stanu")
            return {"count": 0}
        
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
            if "count" not in state:
                state["count"] = 0
            print(f"📂 Załadowano stan: {state['count']} transakcji")
            return state
    except Exception as e:
        print(f"⚠️ Błąd wczytywania stanu: {e}, tworzę nowy")
        return {"count": 0}

def save_state(state):
    """Zapisz stan aplikacji"""
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f)
        return True
    except Exception as e:
        print(f"❌ Błąd zapisu stanu: {e}")
        return False

def export_results():
    """Eksportuj ostatnie 100 transakcji"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    export_dir = "data/results"
    export_path = f"{export_dir}/results_{timestamp}.csv"

    if not os.path.exists(MEMORY_FILE):
        print("⚠️ Brak pliku memory.csv — eksport pominięty")
        return None

    try:
        with open(MEMORY_FILE, "r") as src:
            rows = list(csv.reader(src))
            if not rows:
                print("⚠️ Plik memory.csv jest pusty — eksport pominięty")
                return None
            
            header = rows[0]
            data_rows = rows[1:]
            last_100 = data_rows[-100:] if len(data_rows) > 100 else data_rows

        with open(export_path, "w", newline="") as dst:
            writer = csv.writer(dst)
            writer.writerow(header)
            writer.writerows(last_100)

        print(f"✅ Wyeksportowano {len(last_100)} wpisów do: {export_path}")
        return export_path
    except Exception as e:
        print(f"❌ Błąd eksportu danych: {e}")
        return None

def check_file_status():
    """Sprawdź status plików"""
    print("\n📊 Status plików:")
    
    if os.path.exists(MEMORY_FILE):
        size = os.stat(MEMORY_FILE).st_size
        print(f"📁 {MEMORY_FILE}: {size:,} bajtów")
        
        # Sprawdź liczbę linii
        try:
            with open(MEMORY_FILE, "r") as f:
                lines = sum(1 for _ in f)
            print(f"📋 Liczba linii: {lines:,}")
        except Exception as e:
            print(f"⚠️ Nie można odczytać linii: {e}")
    else:
        print(f"❌ {MEMORY_FILE}: nie istnieje")
    
    if os.path.exists(STATE_FILE):
        print(f"📁 {STATE_FILE}: istnieje")
    else:
        print(f"❌ {STATE_FILE}: nie istnieje")

def main():
    print("🚀 Uruchamiam Enhanced DexBot Worker...", flush=True)
    print(f"⏰ Start: {datetime.now()}")
    
    # Sprawdź i utwórz katalogi
    if not ensure_data_directory():
        print("❌ Nie można utworzyć katalogów, kończę")
        sys.exit(1)
    
    # Sprawdź status plików na start
    check_file_status()
    
    # Załaduj stan
    state = load_state()
    start_count = state["count"]
    
    print(f"🎯 Rozpoczynam od transakcji #{start_count + 1}")
    
    # Główna pętla
    cycle = 0
    try:
        while True:
            cycle += 1
            print(f"\n🔄 Cykl #{cycle} - wykonuję 30 transakcji...")
            
            # Wykonaj 30 transakcji
            for i in range(30):
                try:
                    print(f"🔹 Transakcja {state['count'] + 1} (Cykl {cycle}, #{i+1}/30)")
                    
                    # Wykonaj symulację
                    simulate_trade(settings)
                    state["count"] += 1
                    
                    # Sprawdź status co 10 transakcji
                    if (i + 1) % 10 == 0:
                        check_file_status()
                    
                    # Krótka przerwa między transakcjami
                    time.sleep(0.25)
                    
                except Exception as e:
                    print(f"❌ Błąd podczas transakcji: {e}")
                    continue
            
            # Zapisz stan po każdym cyklu
            if save_state(state):
                print(f"💾 Stan zapisany: {state['count']} transakcji")
            
            # Eksport co 100 transakcji
            if state["count"] % 100 == 0:
                print(f"\n📤 Eksport po {state['count']} transakcjach...")
                export_results()
            
            # Status podsumowujący
            total_executed = state["count"] - start_count
            print(f"\n📈 Statystyki sesji:")
            print(f"   • Łącznie wykonano: {total_executed} nowych transakcji")
            print(f"   • Całkowita liczba: {state['count']:,} transakcji")
            print(f"   • Cykli ukończonych: {cycle}")
            
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
        # Zapisz stan na koniec
        if save_state(state):
            print(f"💾 Końcowy zapis stanu: {state['count']} transakcji")
        
        # Końcowy status
        check_file_status()
        print(f"🏁 Worker zakończony. Łącznie: {state['count']:,} transakcji")

if __name__ == "__main__":
    main()