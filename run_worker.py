# run_worker.py

import time
import os
import json
import csv
from datetime import datetime
from config.settings import load_settings
from core.trade_executor import simulate_trade

STATE_FILE = "data/state.json"
MEMORY_FILE = "data/memory.csv"

def load_state():
    if not os.path.exists(STATE_FILE):
        return {"count": 0}
    with open(STATE_FILE, "r") as f:
        return json.load(f)

def save_state(state):
    os.makedirs("data", exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)

def export_results():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    export_dir = "data/results"
    export_path = f"{export_dir}/results_{timestamp}.csv"

    if not os.path.exists(MEMORY_FILE):
        print("⚠️ Brak pliku memory.csv — eksport pominięty")
        return

    os.makedirs(export_dir, exist_ok=True)

    with open(MEMORY_FILE, "r") as src:
        rows = list(csv.reader(src))
        if not rows:
            print("⚠️ Plik memory.csv jest pusty — eksport pominięty")
            return
        header = rows[0]
        data_rows = rows[1:]

    last_100 = data_rows[-100:]

    with open(export_path, "w", newline="") as dst:
        writer = csv.writer(dst)
        writer.writerow(header)
        writer.writerows(last_100)

    print(f"✅ Wyeksportowano ostatnie 100 wierszy do: {export_path}")

if __name__ == "__main__":
    print("🚀 Uruchamiam bota DEX w trybie ciągłym...")

    os.makedirs("data", exist_ok=True)

    settings = load_settings()
    state = load_state()

    while True:
        for i in range(5):  # 🔁 wykonaj paczkę 5 symulacji
            print(f"🔁 Symulacja {state['count'] + 1}")
            simulate_trade(settings)
            state["count"] += 1
            time.sleep(1)

        save_state(state)

        if state["count"] % 200 == 0:
            export_results()

        print("⏳ Oczekiwanie 60 sekund przed kolejną paczką...")
        time.sleep(60)