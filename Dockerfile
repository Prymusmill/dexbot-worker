# Używamy oficjalnego obrazu Python 3.9
FROM python:3.9-slim

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

# Instalujemy zależności Pythona
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Kopiujemy kod źródłowy
COPY . .

# Tworzymy katalogi, jeśli nie istnieją
RUN mkdir -p data/results tests logs

# Ustawiamy uprawnienia
RUN chmod +x scripts/*.py scripts/*.sh

# Ustawiamy zmienne środowiskowe dla Railway
ENV PORT=8000
ENV ML_MIN_SAMPLES=100
ENV ML_MIN_SAMPLES_PER_CLASS=50
ENV ML_RETRAIN_HOURS=4.0
ENV TRADES_PER_CYCLE=30
ENV CYCLE_DELAY_SECONDS=45
ENV DIRECTIONAL_CONFIDENCE_THRESHOLD=0.6
ENV ENABLE_DIRECTIONAL_TRADING=true
ENV ENABLE_AUTO_RETRAINING=true
ENV LOG_LEVEL=INFO

# Eksponujemy port
EXPOSE 8000

# Uruchamiamy skrypt startowy
CMD ["python", "run_worker.py"]

