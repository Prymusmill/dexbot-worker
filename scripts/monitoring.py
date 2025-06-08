#!/usr/bin/env python3
"""
Skrypt do monitorowania aplikacji.
"""

import os
import sys
import time
import json
import logging
import psutil
import requests
from datetime import datetime
from prometheus_client import start_http_server, Gauge, Counter, Summary

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('logs', 'monitoring.log'))
    ]
)
logger = logging.getLogger('monitoring')

# Metryki Prometheus
CPU_USAGE = Gauge('dexbot_cpu_usage_percent', 'CPU usage in percent')
MEMORY_USAGE = Gauge('dexbot_memory_usage_bytes', 'Memory usage in bytes')
DISK_USAGE = Gauge('dexbot_disk_usage_percent', 'Disk usage in percent')
TRADES_TOTAL = Counter('dexbot_trades_total', 'Total number of trades')
PROFITABLE_TRADES = Counter('dexbot_profitable_trades_total', 'Number of profitable trades')
TRADE_DURATION = Summary('dexbot_trade_duration_seconds', 'Duration of trades in seconds')

# Konfiguracja
MONITORING_PORT = int(os.getenv('MONITORING_PORT', 8000))
CHECK_INTERVAL = int(os.getenv('CHECK_INTERVAL', 60))  # sekundy
ALERT_THRESHOLD_CPU = float(os.getenv('ALERT_THRESHOLD_CPU', 90.0))  # procent
ALERT_THRESHOLD_MEMORY = float(os.getenv('ALERT_THRESHOLD_MEMORY', 90.0))  # procent
ALERT_THRESHOLD_DISK = float(os.getenv('ALERT_THRESHOLD_DISK', 90.0))  # procent

def collect_system_metrics():
    """Zbiera metryki systemowe."""
    try:
        # CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        CPU_USAGE.set(cpu_percent)
        
        # Pamięć
        memory = psutil.virtual_memory()
        MEMORY_USAGE.set(memory.used)
        memory_percent = memory.percent
        
        # Dysk
        disk = psutil.disk_usage('/')
        DISK_USAGE.set(disk.percent)
        disk_percent = disk.percent
        
        logger.info(f"System metrics - CPU: {cpu_percent}%, Memory: {memory_percent}%, Disk: {disk_percent}%")
        
        # Sprawdź alerty
        if cpu_percent > ALERT_THRESHOLD_CPU:
            logger.warning(f"⚠️ HIGH CPU USAGE: {cpu_percent}%")
        
        if memory_percent > ALERT_THRESHOLD_MEMORY:
            logger.warning(f"⚠️ HIGH MEMORY USAGE: {memory_percent}%")
        
        if disk_percent > ALERT_THRESHOLD_DISK:
            logger.warning(f"⚠️ HIGH DISK USAGE: {disk_percent}%")
        
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'disk_percent': disk_percent
        }
        
    except Exception as e:
        logger.error(f"Error collecting system metrics: {e}")
        return {}

def collect_application_metrics():
    """Zbiera metryki aplikacji."""
    try:
        # Sprawdź, czy istnieje plik state.json
        state_file = os.path.join('data', 'state.json')
        if os.path.exists(state_file):
            with open(state_file, 'r') as f:
                state = json.load(f)
                
                # Aktualizuj metryki
                if 'total_trades' in state:
                    TRADES_TOTAL._value.set(state['total_trades'])
                
                if 'profitable_trades' in state:
                    PROFITABLE_TRADES._value.set(state['profitable_trades'])
                
                logger.info(f"Application metrics - Trades: {state.get('total_trades', 0)}, Profitable: {state.get('profitable_trades', 0)}")
                
                return state
        
        return {}
        
    except Exception as e:
        logger.error(f"Error collecting application metrics: {e}")
        return {}

def check_database_connection():
    """Sprawdza połączenie z bazą danych."""
    try:
        # Dodaj ścieżkę do modułu
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        
        # Importuj moduł
        from database.db_manager import get_db_manager
        
        # Pobierz menedżera bazy danych
        db_manager = get_db_manager()
        
        # Sprawdź połączenie
        transaction_count = db_manager.get_transaction_count()
        logger.info(f"Database connection OK - {transaction_count} transactions")
        
        return True
        
    except Exception as e:
        logger.error(f"Error checking database connection: {e}")
        return False

def main():
    """Główna funkcja monitorowania."""
    logger.info("🚀 Starting monitoring service...")
    
    # Uruchom serwer HTTP dla Prometheus
    start_http_server(MONITORING_PORT)
    logger.info(f"Prometheus metrics server started on port {MONITORING_PORT}")
    
    # Główna pętla monitorowania
    while True:
        try:
            # Zbierz metryki
            system_metrics = collect_system_metrics()
            app_metrics = collect_application_metrics()
            db_connection = check_database_connection()
            
            # Zapisz metryki do pliku
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'system': system_metrics,
                'application': app_metrics,
                'database_connection': db_connection
            }
            
            with open(os.path.join('logs', 'metrics.json'), 'w') as f:
                json.dump(metrics, f)
            
            # Czekaj na następne sprawdzenie
            time.sleep(CHECK_INTERVAL)
            
        except KeyboardInterrupt:
            logger.info("Monitoring service stopped by user")
            break
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
            time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    # Utwórz katalog logów, jeśli nie istnieje
    os.makedirs('logs', exist_ok=True)
    
    # Uruchom główną funkcję
    main()

