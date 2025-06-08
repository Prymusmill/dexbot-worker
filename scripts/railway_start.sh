#!/bin/bash
# railway_start.sh - Skrypt startowy dla Railway

# Wyświetl informacje o środowisku
echo "🚀 Starting dexbot-worker on Railway..."
echo "📅 Date: $(date)"
echo "🔧 Python version: $(python --version)"
echo "🌐 Environment: $RAILWAY_ENVIRONMENT"

# Sprawdź, czy zmienne środowiskowe są ustawione
if [ -z "$DATABASE_URL" ]; then
    echo "❌ ERROR: DATABASE_URL is not set"
    exit 1
fi

# Utwórz katalogi, jeśli nie istnieją
mkdir -p data/results logs

# Inicjalizuj bazę danych (jeśli potrzeba)
echo "🔄 Initializing database..."
python scripts/init_db.py

# Uruchom odpowiedni proces w zależności od SERVICE_TYPE
if [ "$SERVICE_TYPE" = "dashboard" ]; then
    echo "📊 Starting dashboard service..."
    exec streamlit run dashboard.py
elif [ "$SERVICE_TYPE" = "monitor" ]; then
    echo "👀 Starting monitoring service..."
    exec python scripts/monitoring.py
else
    # Domyślnie uruchom worker
    echo "🤖 Starting worker service..."
    exec python run_worker.py
fi

