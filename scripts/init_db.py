#!/usr/bin/env python3
"""
Skrypt do inicjalizacji bazy danych.
"""

import os
import sys
import time
import pandas as pd
from datetime import datetime

# Dodaj ścieżkę do modułu
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importuj moduły
try:
    from database.db_manager import get_db_manager
    print("✅ DB manager imported")
except ImportError as e:
    print(f"❌ Error importing DB manager: {e}")
    sys.exit(1)

def init_database():
    """Inicjalizuje bazę danych."""
    try:
        print("🔄 Initializing database...")
        
        # Pobierz menedżera bazy danych
        db_manager = get_db_manager()
        
        # Sprawdź, czy tabele istnieją
        print("🔍 Checking database tables...")
        
        # Sprawdź, czy są dane w bazie
        transaction_count = db_manager.get_transaction_count()
        print(f"📊 Found {transaction_count} transactions in database")
        
        # Jeśli nie ma danych, spróbuj zaimportować z CSV
        if transaction_count == 0:
            print("🔄 No transactions found, trying to import from CSV...")
            
            # Sprawdź, czy istnieje plik CSV
            csv_path = os.path.join('data', 'memory.csv')
            if os.path.exists(csv_path):
                print(f"📄 Found CSV file: {csv_path}")
                
                # Zaimportuj dane z CSV
                result = db_manager.migrate_from_csv(csv_path)
                if result:
                    print("✅ Successfully imported data from CSV")
                else:
                    print("⚠️ Failed to import data from CSV")
            else:
                print("⚠️ No CSV file found")
                
                # Utwórz pusty plik CSV
                create_empty_csv(csv_path)
        
        print("✅ Database initialization completed")
        return True
        
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        return False

def create_empty_csv(csv_path):
    """Tworzy pusty plik CSV z podstawową strukturą."""
    try:
        # Utwórz katalog, jeśli nie istnieje
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        # Utwórz pusty DataFrame z podstawowymi kolumnami
        df = pd.DataFrame({
            'timestamp': [datetime.now()],
            'price': [100.0],
            'volume': [1000.0],
            'rsi': [50.0],
            'direction': ['hold']
        })
        
        # Zapisz do pliku CSV
        df.to_csv(csv_path, index=False)
        print(f"✅ Created empty CSV file: {csv_path}")
        
    except Exception as e:
        print(f"❌ Error creating empty CSV file: {e}")

if __name__ == "__main__":
    # Inicjalizuj bazę danych
    success = init_database()
    
    # Zakończ z odpowiednim kodem wyjścia
    sys.exit(0 if success else 1)

