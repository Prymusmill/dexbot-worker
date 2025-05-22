import csv
import os
import random
from datetime import datetime

MEMORY_FILE = "data/memory.csv"

def simulate_trade(settings):
    os.makedirs("data", exist_ok=True)

    # Przykładowe dane transakcji
    timestamp = datetime.now().isoformat()
    input_token = "SOL"
    output_token = "USDC"
    trade_amount = float(settings.get("trade_amount_usd", 0.01))
    output_received = round(trade_amount * random.uniform(0.98, 1.02), 5)  # symulacja z niewielką zmiennością
    success = random.choice([True] * 9 + [False])  # 90% szans powodzenia

    row = [timestamp, input_token, output_token, trade_amount, output_received, success]

    write_header = not os.path.exists(MEMORY_FILE)
    with open(MEMORY_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(["timestamp", "input_token", "output_token", "trade_amount_usd", "output_received", "success"])
        writer.writerow(row)

    print(f"✅ Symulacja: {row}")