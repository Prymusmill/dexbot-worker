"""
Moduł do augmentacji danych dla modeli ML.

Ten moduł zawiera funkcje do augmentacji danych, balansowania klas i generowania syntetycznych danych,
co pozwala na trenowanie modeli ML na większej ilości danych.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
import os
import logging
import random
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_historical_data(csv_path: Optional[str] = None, 
                         db_connection: Optional[Any] = None, 
                         min_samples: int = 100) -> pd.DataFrame:
    """
    Ładuje dane historyczne z pliku CSV lub bazy danych.
    
    Args:
        csv_path: Ścieżka do pliku CSV z danymi historycznymi
        db_connection: Połączenie do bazy danych
        min_samples: Minimalna liczba próbek do załadowania
        
    Returns:
        pd.DataFrame: DataFrame z danymi historycznymi
    """
    data = pd.DataFrame()
    
    # Próba załadowania danych z bazy danych
    if db_connection is not None:
        try:
            if hasattr(db_connection, 'get_all_transactions_for_ml'):
                data = db_connection.get_all_transactions_for_ml()
                logger.info(f"✅ Loaded {len(data)} samples from database")
            elif hasattr(db_connection, 'connection'):
                # Próba użycia bezpośredniego połączenia SQL
                query = """
                    SELECT * FROM transactions 
                    ORDER BY timestamp DESC 
                    LIMIT 1000
                """
                data = pd.read_sql_query(query, db_connection.connection)
                logger.info(f"✅ Loaded {len(data)} samples from database using direct SQL")
        except Exception as e:
            logger.error(f"❌ Error loading data from database: {e}")
    
    # Jeśli nie udało się załadować danych z bazy lub jest ich za mało, próba załadowania z CSV
    if len(data) < min_samples and csv_path is not None:
        try:
            if os.path.exists(csv_path):
                csv_data = pd.read_csv(csv_path)
                if 'timestamp' in csv_data.columns:
                    csv_data['timestamp'] = pd.to_datetime(csv_data['timestamp'])
                logger.info(f"✅ Loaded {len(csv_data)} samples from CSV")
                
                # Jeśli mamy już jakieś dane z bazy, połącz je z danymi z CSV
                if not data.empty:
                    # Sprawdź, czy kolumny się zgadzają
                    common_cols = list(set(data.columns) & set(csv_data.columns))
                    if common_cols:
                        data = pd.concat([data, csv_data[common_cols]], ignore_index=True)
                        logger.info(f"✅ Combined data: {len(data)} samples")
                    else:
                        data = csv_data
                else:
                    data = csv_data
        except Exception as e:
            logger.error(f"❌ Error loading data from CSV: {e}")
    
    # Jeśli nadal mamy za mało danych, generuj syntetyczne dane
    if len(data) < min_samples:
        logger.warning(f"⚠️ Not enough data: {len(data)}/{min_samples} samples")
        if not data.empty:
            # Generuj syntetyczne dane na podstawie istniejących
            synthetic_data = generate_synthetic_data(data, min_samples - len(data))
            data = pd.concat([data, synthetic_data], ignore_index=True)
            logger.info(f"✅ Generated synthetic data: {len(data)} samples total")
        else:
            # Jeśli nie mamy żadnych danych, generuj całkowicie losowe dane
            data = generate_random_data(min_samples)
            logger.info(f"✅ Generated random data: {len(data)} samples")
    
    return data

def balance_directional_classes(df: pd.DataFrame, 
                               target_col: str = 'direction', 
                               min_samples_per_class: int = 50) -> pd.DataFrame:
    """
    Balansuje klasy w danych treningowych.
    
    Args:
        df: DataFrame z danymi treningowymi
        target_col: Nazwa kolumny zawierającej etykiety klas
        min_samples_per_class: Minimalna liczba próbek dla każdej klasy
        
    Returns:
        pd.DataFrame: Zbalansowany DataFrame
    """
    if df.empty or target_col not in df.columns:
        logger.warning(f"⚠️ Cannot balance classes: DataFrame is empty or missing target column '{target_col}'")
        return df
    
    # Sprawdź rozkład klas
    class_counts = df[target_col].value_counts()
    logger.info(f"📊 Initial class distribution: {dict(class_counts)}")
    
    # Sprawdź, czy mamy wszystkie oczekiwane klasy
    expected_classes = ['long', 'short', 'hold']
    missing_classes = [cls for cls in expected_classes if cls not in class_counts.index]
    
    balanced_df = df.copy()
    
    # Dodaj brakujące klasy
    for cls in missing_classes:
        logger.warning(f"⚠️ Missing class: '{cls}', generating synthetic samples")
        # Generuj syntetyczne próbki dla brakującej klasy
        if not df.empty:
            synthetic_samples = generate_synthetic_samples_for_class(df, cls, min_samples_per_class)
            balanced_df = pd.concat([balanced_df, synthetic_samples], ignore_index=True)
        else:
            # Jeśli DataFrame jest pusty, generuj losowe próbki
            random_samples = generate_random_samples_for_class(cls, min_samples_per_class)
            balanced_df = pd.concat([balanced_df, random_samples], ignore_index=True)
    
    # Uzupełnij klasy, które mają za mało próbek
    for cls in class_counts.index:
        if class_counts[cls] < min_samples_per_class:
            samples_to_add = min_samples_per_class - class_counts[cls]
            logger.info(f"⚠️ Class '{cls}' has only {class_counts[cls]} samples, adding {samples_to_add} more")
            
            # Wybierz próbki z danej klasy
            class_samples = df[df[target_col] == cls]
            
            if len(class_samples) > 0:
                # Generuj syntetyczne próbki na podstawie istniejących
                synthetic_samples = generate_synthetic_samples_for_class(
                    df, cls, samples_to_add, base_samples=class_samples
                )
                balanced_df = pd.concat([balanced_df, synthetic_samples], ignore_index=True)
            else:
                # Jeśli nie ma próbek danej klasy, generuj losowe
                random_samples = generate_random_samples_for_class(cls, samples_to_add)
                balanced_df = pd.concat([balanced_df, random_samples], ignore_index=True)
    
    # Sprawdź końcowy rozkład klas
    final_class_counts = balanced_df[target_col].value_counts()
    logger.info(f"✅ Final class distribution: {dict(final_class_counts)}")
    
    return balanced_df

def generate_synthetic_data(df: pd.DataFrame, n_samples: int) -> pd.DataFrame:
    """
    Generuje syntetyczne dane na podstawie istniejących próbek.
    
    Args:
        df: DataFrame z danymi źródłowymi
        n_samples: Liczba próbek do wygenerowania
        
    Returns:
        pd.DataFrame: DataFrame z syntetycznymi danymi
    """
    if df.empty or n_samples <= 0:
        return pd.DataFrame()
    
    try:
        # Wybierz tylko kolumny numeryczne
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            logger.warning("⚠️ No numeric columns found for synthetic data generation")
            return pd.DataFrame()
        
        # Przygotuj dane do generowania
        X = df[numeric_cols].values
        
        # Normalizuj dane
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Użyj algorytmu k-najbliższych sąsiadów do generowania syntetycznych próbek
        synthetic_samples = []
        
        # Określ liczbę sąsiadów (k)
        k = min(5, len(df) - 1)
        if k <= 1:
            # Jeśli mamy za mało próbek, użyj prostszej metody
            for _ in range(n_samples):
                # Wybierz losową próbkę
                idx = np.random.randint(0, len(df))
                sample = df.iloc[idx].copy()
                
                # Dodaj losowe zaburzenia do wartości numerycznych
                for col in numeric_cols:
                    if col in sample:
                        noise = np.random.normal(0, 0.1 * df[col].std())
                        sample[col] += noise
                
                synthetic_samples.append(sample)
        else:
            # Użyj algorytmu k-NN
            nbrs = NearestNeighbors(n_neighbors=k).fit(X_scaled)
            
            for _ in range(n_samples):
                # Wybierz losową próbkę
                idx = np.random.randint(0, len(df))
                sample = X_scaled[idx]
                
                # Znajdź k najbliższych sąsiadów
                _, indices = nbrs.kneighbors([sample])
                
                # Wybierz losowego sąsiada
                neighbor_idx = indices[0][np.random.randint(1, k)]
                neighbor = X_scaled[neighbor_idx]
                
                # Generuj syntetyczną próbkę jako kombinację próbki i sąsiada
                alpha = np.random.random()
                synthetic = sample + alpha * (neighbor - sample)
                
                # Odwróć normalizację
                synthetic_denorm = scaler.inverse_transform([synthetic])[0]
                
                # Utwórz nowy wiersz DataFrame
                synthetic_row = df.iloc[idx].copy()
                for i, col in enumerate(numeric_cols):
                    synthetic_row[col] = synthetic_denorm[i]
                
                synthetic_samples.append(synthetic_row)
        
        # Utwórz DataFrame z syntetycznych próbek
        synthetic_df = pd.DataFrame(synthetic_samples)
        
        logger.info(f"✅ Generated {len(synthetic_df)} synthetic samples")
        return synthetic_df
        
    except Exception as e:
        logger.error(f"❌ Error generating synthetic data: {e}")
        return pd.DataFrame()

def generate_synthetic_samples_for_class(df: pd.DataFrame, 
                                        target_class: str, 
                                        n_samples: int,
                                        target_col: str = 'direction',
                                        base_samples: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Generuje syntetyczne próbki dla określonej klasy.
    
    Args:
        df: DataFrame z danymi źródłowymi
        target_class: Klasa, dla której generujemy próbki
        n_samples: Liczba próbek do wygenerowania
        target_col: Nazwa kolumny zawierającej etykiety klas
        base_samples: Opcjonalnie, próbki bazowe do generowania
        
    Returns:
        pd.DataFrame: DataFrame z syntetycznymi próbkami
    """
    if df.empty or n_samples <= 0:
        return pd.DataFrame()
    
    try:
        # Jeśli nie podano próbek bazowych, użyj wszystkich próbek
        if base_samples is None or base_samples.empty:
            base_samples = df.copy()
        
        # Wybierz tylko kolumny numeryczne
        numeric_cols = base_samples.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            logger.warning("⚠️ No numeric columns found for synthetic sample generation")
            return pd.DataFrame()
        
        # Generuj syntetyczne próbki
        synthetic_samples = []
        
        for _ in range(n_samples):
            # Wybierz losową próbkę
            idx = np.random.randint(0, len(base_samples))
            sample = base_samples.iloc[idx].copy()
            
            # Ustaw klasę docelową
            sample[target_col] = target_class
            
            # Dodaj losowe zaburzenia do wartości numerycznych
            for col in numeric_cols:
                if col in sample and col != target_col:
                    noise = np.random.normal(0, 0.1 * base_samples[col].std())
                    sample[col] += noise
            
            synthetic_samples.append(sample)
        
        # Utwórz DataFrame z syntetycznych próbek
        synthetic_df = pd.DataFrame(synthetic_samples)
        
        logger.info(f"✅ Generated {len(synthetic_df)} synthetic samples for class '{target_class}'")
        return synthetic_df
        
    except Exception as e:
        logger.error(f"❌ Error generating synthetic samples for class '{target_class}': {e}")
        return pd.DataFrame()

def generate_random_data(n_samples: int) -> pd.DataFrame:
    """
    Generuje całkowicie losowe dane.
    
    Args:
        n_samples: Liczba próbek do wygenerowania
        
    Returns:
        pd.DataFrame: DataFrame z losowymi danymi
    """
    try:
        # Utwórz podstawowe kolumny
        data = {
            'timestamp': [datetime.now() - timedelta(hours=i) for i in range(n_samples)],
            'price': np.random.uniform(10, 1000, n_samples),
            'volume': np.random.uniform(1000, 100000, n_samples),
            'rsi': np.random.uniform(0, 100, n_samples),
            'direction': np.random.choice(['long', 'short', 'hold'], n_samples)
        }
        
        # Dodaj dodatkowe kolumny
        data['price_change_24h'] = np.random.uniform(-10, 10, n_samples)
        data['volatility'] = np.random.uniform(0.01, 0.1, n_samples)
        
        # Utwórz DataFrame
        df = pd.DataFrame(data)
        
        logger.info(f"✅ Generated {n_samples} random samples")
        return df
        
    except Exception as e:
        logger.error(f"❌ Error generating random data: {e}")
        return pd.DataFrame()

def generate_random_samples_for_class(target_class: str, n_samples: int) -> pd.DataFrame:
    """
    Generuje losowe próbki dla określonej klasy.
    
    Args:
        target_class: Klasa, dla której generujemy próbki
        n_samples: Liczba próbek do wygenerowania
        
    Returns:
        pd.DataFrame: DataFrame z losowymi próbkami
    """
    try:
        # Utwórz podstawowe kolumny
        data = {
            'timestamp': [datetime.now() - timedelta(hours=i) for i in range(n_samples)],
            'price': np.random.uniform(10, 1000, n_samples),
            'volume': np.random.uniform(1000, 100000, n_samples),
            'rsi': np.random.uniform(0, 100, n_samples),
            'direction': [target_class] * n_samples
        }
        
        # Dostosuj wartości w zależności od klasy
        if target_class == 'long':
            # Dla klasy 'long' ustaw wyższe RSI i pozytywne zmiany cen
            data['rsi'] = np.random.uniform(60, 90, n_samples)
            data['price_change_24h'] = np.random.uniform(1, 10, n_samples)
        elif target_class == 'short':
            # Dla klasy 'short' ustaw niższe RSI i negatywne zmiany cen
            data['rsi'] = np.random.uniform(10, 40, n_samples)
            data['price_change_24h'] = np.random.uniform(-10, -1, n_samples)
        else:  # 'hold'
            # Dla klasy 'hold' ustaw neutralne RSI i małe zmiany cen
            data['rsi'] = np.random.uniform(40, 60, n_samples)
            data['price_change_24h'] = np.random.uniform(-1, 1, n_samples)
        
        # Dodaj dodatkowe kolumny
        data['volatility'] = np.random.uniform(0.01, 0.1, n_samples)
        
        # Utwórz DataFrame
        df = pd.DataFrame(data)
        
        logger.info(f"✅ Generated {n_samples} random samples for class '{target_class}'")
        return df
        
    except Exception as e:
        logger.error(f"❌ Error generating random samples for class '{target_class}': {e}")
        return pd.DataFrame()

def augment_time_series_data(df: pd.DataFrame, time_col: str = 'timestamp', value_col: str = 'price') -> pd.DataFrame:
    """
    Augmentuje dane szeregów czasowych.
    
    Args:
        df: DataFrame z danymi szeregów czasowych
        time_col: Nazwa kolumny zawierającej czas
        value_col: Nazwa kolumny zawierającej wartości
        
    Returns:
        pd.DataFrame: DataFrame z augmentowanymi danymi
    """
    if df.empty or time_col not in df.columns or value_col not in df.columns:
        return df
    
    try:
        # Sortuj dane według czasu
        df_sorted = df.sort_values(by=time_col).copy()
        
        # Utwórz kopię danych z różnymi transformacjami
        augmented_data = []
        
        # 1. Oryginalne dane
        augmented_data.append(df_sorted.copy())
        
        # 2. Dodaj szum do wartości
        noisy_df = df_sorted.copy()
        noisy_df[value_col] = noisy_df[value_col] * (1 + np.random.normal(0, 0.01, len(noisy_df)))
        augmented_data.append(noisy_df)
        
        # 3. Przesuń wartości w czasie
        shifted_df = df_sorted.copy()
        shifted_df[value_col] = shifted_df[value_col].shift(1).fillna(method='bfill')
        augmented_data.append(shifted_df)
        
        # 4. Wygładź wartości
        smoothed_df = df_sorted.copy()
        smoothed_df[value_col] = smoothed_df[value_col].rolling(window=3, min_periods=1).mean()
        augmented_data.append(smoothed_df)
        
        # Połącz wszystkie augmentowane dane
        augmented_df = pd.concat(augmented_data, ignore_index=True)
        
        logger.info(f"✅ Augmented time series data: {len(df)} → {len(augmented_df)} samples")
        return augmented_df
        
    except Exception as e:
        logger.error(f"❌ Error augmenting time series data: {e}")
        return df

def save_augmented_data(df: pd.DataFrame, csv_path: str) -> bool:
    """
    Zapisuje augmentowane dane do pliku CSV.
    
    Args:
        df: DataFrame z augmentowanymi danymi
        csv_path: Ścieżka do pliku CSV
        
    Returns:
        bool: True, jeśli zapis się powiódł, False w przeciwnym razie
    """
    if df.empty:
        logger.warning("⚠️ Cannot save empty DataFrame")
        return False
    
    try:
        # Utwórz katalog, jeśli nie istnieje
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        
        # Zapisz dane do pliku CSV
        df.to_csv(csv_path, index=False)
        
        logger.info(f"✅ Saved {len(df)} samples to {csv_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error saving augmented data: {e}")
        return False

