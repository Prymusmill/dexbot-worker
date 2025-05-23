import time
import os
import json
import csv
from datetime import datetime
from config.settings import SETTINGS as settings
from core.trade_executor import simulate_trade

STATE_FILE = "data/state.json"
MEMORY_FILE = "data/memory.csv"

def load_state():
    if not os.path.exists(STATE_FILE):
        return {"count": 0}
    try:
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
            if "count" not in state:
                state["count"] = 0
            return state
    except Exception:
        return {"count": 0}

def save_state(state):
    os.makedirs("data", exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)

def export_results():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    export_dir = "data/results"
    export_path = f"{export_dir}/results_{timestamp}.csv"

    if not os.path.exists(MEMORY_FILE):
        print("⚠️ Brak pliku memory.csv — eksport pominięty", flush=True)
        return None

    os.makedirs(export_dir, exist_ok=True)

    try:
        with open(MEMORY_FILE, "r") as src:
            rows = list(csv.reader(src))
            if not rows:
                print("⚠️ Plik memory.csv jest pusty — eksport pominięty", flush=True)
                return None
            header = rows[0]
            data_rows = rows[1:]
            last_100 = data_rows[-100:]

        with open(export_path, "w", newline="") as dst:
            writer = csv.writer(dst)
            writer.writerow(header)
            writer.writerows(last_100)

        print(f"✅ Wyeksportowano 100 wpisów do: {export_path}", flush=True)
        return export_path
    except Exception as e:
        print(f"❌ Błąd eksportu danych: {e}", flush=True)
        return None

if __name__ == "__main__":
    print("🚀 Uruchamiam bota DEX w trybie ciągłym...", flush=True)

    os.makedirs("data", exist_ok=True)
    state = load_state()

    while True:
        for _ in range(30):
            print(f"🔁 Symulacja {state['count'] + 1}", flush=True)
            simulate_trade(settings)

            if os.path.exists(MEMORY_FILE):
                size = os.stat(MEMORY_FILE).st_size
                print(f"📁 memory.csv istnieje – rozmiar: {size} bajtów", flush=True)
            else:
                print("❌ Plik memory.csv NIE istnieje!", flush=True)

            state["count"] += 1
            time.sleep(0.25)

        save_state(state)

        if state["count"] % 100 == 0:
            export_results()

        print("⏳ Oczekiwanie 60 sekund przed kolejną paczką...", flush=True)
        time.sleep(60)