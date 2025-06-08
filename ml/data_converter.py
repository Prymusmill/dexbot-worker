"""
Moduł do konwersji danych dla modeli ML.

Ten moduł zawiera funkcje do konwersji różnych typów danych na format odpowiedni dla modeli ML,
w szczególności konwersję dat na format numeryczny.
"""

import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Union, List, Dict, Any, Optional

def preprocess_dataframe_for_ml(df: pd.DataFrame) -> pd.DataFrame:
    """
    Przygotowuje DataFrame do trenowania modeli ML.
    
    Args:
        df (pd.DataFrame): DataFrame do przetworzenia
        
    Returns:
        pd.DataFrame: Przetworzony DataFrame
    """
    if df is None or len(df) == 0:
        print("⚠️ Empty DataFrame provided for preprocessing")
        return pd.DataFrame()
    
    try:
        # Tworzenie kopii DataFrame, aby nie modyfikować oryginału
        df_processed = df.copy()
        
        # Konwersja kolumn typu datetime
        df_processed = convert_datetime_features(df_processed)
        
        # Konwersja kolumn typu object (string)
        df_processed = convert_object_features(df_processed)
        
        # Zastąpienie wartości nieskończonych
        df_processed = df_processed.replace([np.inf, -np.inf], np.nan)
        
        # Wypełnienie brakujących wartości
        df_processed = df_processed.fillna(df_processed.median(numeric_only=True))
        
        print(f"✅ DataFrame preprocessed for ML: {len(df_processed)} rows, {len(df_processed.columns)} columns")
        return df_processed
        
    except Exception as e:
        print(f"❌ Error preprocessing DataFrame for ML: {e}")
        # Zwróć oryginalny DataFrame, jeśli wystąpił błąd
        return df

def convert_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Konwertuje kolumny typu datetime na liczby.
    
    Args:
        df (pd.DataFrame): DataFrame do przetworzenia
        
    Returns:
        pd.DataFrame: DataFrame z przekonwertowanymi kolumnami datetime
    """
    try:
        df_converted = df.copy()
        datetime_cols = []
        
        # Znajdź kolumny typu datetime lub object, które mogą zawierać daty
        for col in df_converted.columns:
            if df_converted[col].dtype == 'datetime64[ns]':
                datetime_cols.append(col)
            elif df_converted[col].dtype == 'object':
                # Sprawdź, czy kolumna zawiera daty
                sample_val = df_converted[col].dropna().iloc[0] if not df_converted[col].dropna().empty else None
                if isinstance(sample_val, (date, datetime)):
                    datetime_cols.append(col)
        
        # Konwertuj znalezione kolumny datetime na liczby (dni od epoki)
        for col in datetime_cols:
            print(f"🔄 Converting datetime column: {col}")
            df_converted[col] = df_converted[col].apply(
                lambda x: date_to_numeric(x) if pd.notnull(x) else np.nan
            )
        
        if datetime_cols:
            print(f"✅ Converted {len(datetime_cols)} datetime columns to numeric")
        
        return df_converted
        
    except Exception as e:
        print(f"❌ Error converting datetime features: {e}")
        return df

def convert_object_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Konwertuje kolumny typu object (string) na format odpowiedni dla modeli ML.
    
    Args:
        df (pd.DataFrame): DataFrame do przetworzenia
        
    Returns:
        pd.DataFrame: DataFrame z przekonwertowanymi kolumnami object
    """
    try:
        df_converted = df.copy()
        object_cols = df_converted.select_dtypes(include=['object']).columns
        
        for col in object_cols:
            # Sprawdź, czy kolumna zawiera daty
            sample_val = df_converted[col].dropna().iloc[0] if not df_converted[col].dropna().empty else None
            
            if isinstance(sample_val, (date, datetime)):
                # Konwertuj daty na liczby
                df_converted[col] = df_converted[col].apply(
                    lambda x: date_to_numeric(x) if pd.notnull(x) else np.nan
                )
            elif isinstance(sample_val, str):
                # Jeśli to string, spróbuj przekonwertować na datetime
                try:
                    df_converted[col] = pd.to_datetime(df_converted[col], errors='coerce')
                    df_converted[col] = df_converted[col].apply(
                        lambda x: date_to_numeric(x) if pd.notnull(x) else np.nan
                    )
                except:
                    # Jeśli nie można przekonwertować na datetime, pozostaw jako jest
                    pass
        
        return df_converted
        
    except Exception as e:
        print(f"❌ Error converting object features: {e}")
        return df

def date_to_numeric(dt: Union[datetime, date, str]) -> float:
    """
    Konwertuje obiekt datetime lub date na liczbę (dni od epoki).
    
    Args:
        dt: Obiekt datetime, date lub string do konwersji
        
    Returns:
        float: Liczba dni od epoki (1970-01-01)
    """
    try:
        if isinstance(dt, str):
            dt = pd.to_datetime(dt)
        
        if isinstance(dt, datetime):
            return (dt - datetime(1970, 1, 1)).total_seconds() / 86400.0
        elif isinstance(dt, date):
            return (dt - date(1970, 1, 1)).days
        else:
            return float(dt)  # Próba konwersji na float
    except Exception as e:
        print(f"⚠️ Error converting date to numeric: {e}, value: {dt}, type: {type(dt)}")
        return 0.0

def safe_float_conversion(value: Any) -> float:
    """
    Bezpieczna konwersja różnych typów danych na float.
    
    Args:
        value: Wartość do konwersji
        
    Returns:
        float: Przekonwertowana wartość
    """
    try:
        if pd.isna(value) or value is None:
            return 0.0
        elif isinstance(value, (datetime, date)):
            return date_to_numeric(value)
        elif isinstance(value, str):
            # Próba konwersji string na datetime
            try:
                dt = pd.to_datetime(value)
                return date_to_numeric(dt)
            except:
                # Jeśli nie można przekonwertować na datetime, próba konwersji na float
                try:
                    return float(value)
                except:
                    return 0.0
        elif isinstance(value, (int, float, np.number)):
            return float(value)
        else:
            return 0.0
    except Exception as e:
        print(f"⚠️ Safe float conversion error: {e}, value: {value}, type: {type(value)}")
        return 0.0

def extract_time_features(df: pd.DataFrame, datetime_col: str = 'timestamp') -> pd.DataFrame:
    """
    Ekstrahuje cechy czasowe z kolumny datetime.
    
    Args:
        df (pd.DataFrame): DataFrame do przetworzenia
        datetime_col (str): Nazwa kolumny zawierającej datetime
        
    Returns:
        pd.DataFrame: DataFrame z dodatkowymi cechami czasowymi
    """
    try:
        if datetime_col not in df.columns:
            print(f"⚠️ Datetime column '{datetime_col}' not found in DataFrame")
            return df
        
        df_with_time = df.copy()
        
        # Upewnij się, że kolumna jest typu datetime
        if df_with_time[datetime_col].dtype != 'datetime64[ns]':
            df_with_time[datetime_col] = pd.to_datetime(df_with_time[datetime_col], errors='coerce')
        
        # Ekstrahuj cechy czasowe
        df_with_time['hour'] = df_with_time[datetime_col].dt.hour
        df_with_time['day_of_week'] = df_with_time[datetime_col].dt.dayofweek
        df_with_time['day_of_month'] = df_with_time[datetime_col].dt.day
        df_with_time['month'] = df_with_time[datetime_col].dt.month
        df_with_time['year'] = df_with_time[datetime_col].dt.year
        
        # Dodatkowe cechy
        df_with_time['is_weekend'] = df_with_time['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        df_with_time['is_month_start'] = df_with_time['day_of_month'].apply(lambda x: 1 if x <= 3 else 0)
        df_with_time['is_month_end'] = df_with_time['day_of_month'].apply(lambda x: 1 if x >= 28 else 0)
        
        # Cechy cykliczne (sinusoidalne)
        df_with_time['hour_sin'] = np.sin(2 * np.pi * df_with_time['hour'] / 24)
        df_with_time['hour_cos'] = np.cos(2 * np.pi * df_with_time['hour'] / 24)
        df_with_time['day_of_week_sin'] = np.sin(2 * np.pi * df_with_time['day_of_week'] / 7)
        df_with_time['day_of_week_cos'] = np.cos(2 * np.pi * df_with_time['day_of_week'] / 7)
        df_with_time['month_sin'] = np.sin(2 * np.pi * df_with_time['month'] / 12)
        df_with_time['month_cos'] = np.cos(2 * np.pi * df_with_time['month'] / 12)
        
        print(f"✅ Extracted time features from '{datetime_col}'")
        return df_with_time
        
    except Exception as e:
        print(f"❌ Error extracting time features: {e}")
        return df

