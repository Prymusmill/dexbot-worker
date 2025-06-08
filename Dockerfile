# FIXED Dockerfile for Railway - Python 3.10+ compatible
FROM python:3.10-slim

# Ustawiamy zmienne środowiskowe
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Ustawiamy katalog roboczy
WORKDIR /app

# Instalujemy zależności systemowe
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Kopiujemy pliki requirements
COPY requirements.txt .

# FIXED: Instalujemy zależności Pythona z lepszym error handling
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt || \
    (echo "Failed to install requirements, trying without cryptography..." && \
     pip install --no-cache-dir pandas numpy scikit-learn joblib xgboost lightgbm flask psycopg2-binary sqlalchemy python-dotenv requests websocket-client psutil)

# Kopiujemy kod źródłowy
COPY . .

# Tworzymy katalogi, jeśli nie istnieją
RUN mkdir -p data/results tests logs models ml

# FIXED: Ustawiamy zmienne środowiskowe dla Railway (kompatybilne z Python 3.10+)
ENV PORT=8000
ENV ML_MIN_SAMPLES=100
ENV ML_MIN_SAMPLES_PER_CLASS=50
ENV ML_RETRAIN_HOURS=4.0
ENV TRADES_PER_CYCLE=25
ENV CYCLE_DELAY_SECONDS=60
ENV DIRECTIONAL_CONFIDENCE_THRESHOLD=0.6
ENV ENABLE_DIRECTIONAL_TRADING=true
ENV ENABLE_AUTO_RETRAINING=true
ENV LOG_LEVEL=INFO
ENV LONG_BIAS=0.4
ENV SHORT_BIAS=0.4
ENV HOLD_BIAS=0.2
ENV TRADE_AMOUNT_USD=0.02

# Eksponujemy port
EXPOSE 8000

# FIXED: Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# FIXED: Uruchamiamy skrypt startowy z fallback
CMD ["python", "run_worker.py"]