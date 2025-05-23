import time
import os
import json
import csv
import subprocess
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
    return export_path  # 🧠 zwracamy ścieżkę do pliku

def commit_and_push(file_path):
    try:
        subprocess.run(["git", "config", "--global", "user.name", "dexbot"], check=True)
        subprocess.run(["git", "config", "--global", "user.email", "bot@dex.ai"], check=True)
        subprocess.run(["git", "add", file_path], check=True)
        subprocess.run(["git", "commit", "-m", "Auto export results"], check=True)
        subprocess.run(["git", "push", "origin", "main"], check=True)
        print("🚀 Plik wypchnięty do GitHuba.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Błąd podczas pushowania: {e}")

if __name__ == "__main__":
    print("🚀 Uruchamiam bota DEX w trybie ciągłym...")

    os.makedirs("data", exist_ok=True)
    state = load_state()

    while True:
        for i in range(30):  # 🔁 Paczka 5 symulacji
            print(f"🔁 Symulacja {state['count'] + 1}")
            simulate_trade(settings)
            state["count"] += 1
            time.sleep(0.25)

        save_state(state)

        if state["count"] % 100 == 0:
            exported_file = export_results()
            if exported_file:
                commit_and_push(exported_file)

        print("⏳ Oczekiwanie 60 sekund przed kolejną paczką...")
        time.sleep(60)