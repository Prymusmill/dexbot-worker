# core/trade_executor.py

import csv
from datetime import datetime
import random
import os

MEMORY_FILE = "data/memory.csv"

def simulate_trade(settings):
    timestamp = datetime.utcnow().isoformat()
    input_token = "SOL"
    output_token = "USDC"
    amount_in = settings["trade_amount_usd"]

    # Losowe dane wyjściowe
    price_impact = round(random.uniform(-0.01, 0.01), 5)
    amount_out = round(amount_in * (1 + price_impact), 5)
    profitable = amount_out > amount_in

    row = [timestamp, input_token, output_token, amount_in, amount_out, price_impact]

    # Zapis do memory.csv
    os.makedirs("data", exist_ok=True)
    file_exists = os.path.isfile(MEMORY_FILE)
    with open(MEMORY_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists or os.stat(MEMORY_FILE).st_size == 0:
            writer.writerow(["timestamp", "input_token", "output_token", "amount_in", "amount_out", "price_impact"])
        writer.writerow(row)

    print(f"✅ Symulacja: {row}")