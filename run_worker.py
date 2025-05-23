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
EXPORT_DIR = "data/results"


def load_state():
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(STATE_FILE):
        print("🔧 Tworzę nowy plik state.json z count = 0")
        return {"count": 0}
    try:
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
            if "count" not in state:
                state["count"] = 0
            return state
    except Exception as e:
        print(f"⚠️ Błąd wczytywania state.json: {e}")
        return {"count": 0}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)
    print(f"💾 Zapisano stan: count = {state['count']}")


def export_results():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    export_path = f"{EXPORT_DIR}/results_{timestamp}.csv"

    if not os.path.exists(MEMORY_FILE):
        print("⚠️ Brak pliku memory.csv — eksport pominięty")
        return None

    os.makedirs(EXPORT_DIR, exist_ok=True)

    with open(MEMORY_FILE, "r") as src:
        rows = list(csv.reader(src))
        if not rows:
            print("⚠️ Plik memory.csv jest pusty")
            return None
        header, data_rows = rows[0], rows[1:]
        last_100 = data_rows[-100:]

    with open(export_path, "w", newline="") as dst:
        writer = csv.writer(dst)
        writer.writerow(header)
        writer.writerows(last_100)

    print(f"✅ Wyeksportowano 100 wpisów do: {export_path}")
    return export_path


def commit_and_push(file_path):
    try:
        subprocess.run(["git", "config", "--global", "user.name", "dexbot"], check=True)
        subprocess.run(["git", "config", "--global", "user.email", "bot@dex.ai"], check=True)
        subprocess.run(["git", "add", file_path], check=True)
        subprocess.run(["git", "commit", "-m", "Auto export results"], check=True)
        subprocess.run(["git", "push", "origin", "main"], check=True)
        print("🚀 Wyniki wypchnięte do GitHuba")
    except subprocess.CalledProcessError as e:
        print(f"❌ Błąd pushowania: {e}")


if __name__ == "__main__":
    print("🚀 Uruchamiam bota DEX w trybie ciągłym...")

    state = load_state()

    while True:
    for _ in range(30):
        print(f"🔁 Symulacja {state['count'] + 1}", flush=True)
        simulate_trade(settings)
        state["count"] += 1
        time.sleep(0.25)

    save_state(state)

    if state["count"] % 100 == 0:
        exported_file = export_results()
        if exported_file:
            commit_and_push(exported_file)

    print("⏳ Oczekiwanie 60 sekund przed kolejną paczką...", flush=True)
    time.sleep(60)

    with open(MEMORY_FILE, "w", newline="") as dst:
        writer = csv.writer(dst)
        writer.writerow(header)
        writer.writerows(last_100)