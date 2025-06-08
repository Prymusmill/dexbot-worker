"""
Testy dla modułu konwersji danych.
"""

import sys
import os
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta

# Dodaj ścieżkę do modułu
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importuj moduł do testowania
from ml.data_converter import (
    preprocess_dataframe_for_ml,
    convert_datetime_features,
    convert_object_features,
    date_to_numeric,
    safe_float_conversion,
    extract_time_features
)

class TestDataConverter(unittest.TestCase):
    """Testy dla modułu konwersji danych."""
    
    def setUp(self):
        """Przygotowanie danych testowych."""
        # Utwórz przykładowy DataFrame z różnymi typami danych
        self.test_data = {
            'timestamp': [datetime.now(), datetime.now() - timedelta(days=1), datetime.now() - timedelta(days=2)],
            'date_only': [date.today(), date.today() - timedelta(days=1), date.today() - timedelta(days=2)],
            'string_date': ['2025-06-01', '2025-06-02', '2025-06-03'],
            'numeric': [1.0, 2.0, 3.0],
            'string': ['a', 'b', 'c']
        }
        self.df = pd.DataFrame(self.test_data)
    
    def test_preprocess_dataframe_for_ml(self):
        """Test funkcji preprocess_dataframe_for_ml."""
        # Wywołaj funkcję
        processed_df = preprocess_dataframe_for_ml(self.df)
        
        # Sprawdź, czy wynikowy DataFrame ma tę samą liczbę wierszy
        self.assertEqual(len(processed_df), len(self.df))
        
        # Sprawdź, czy kolumny datetime zostały przekonwertowane na liczby
        self.assertTrue(pd.api.types.is_numeric_dtype(processed_df['timestamp']))
        self.assertTrue(pd.api.types.is_numeric_dtype(processed_df['date_only']))
        
        # Sprawdź, czy string_date został przekonwertowany na liczbę
        self.assertTrue(pd.api.types.is_numeric_dtype(processed_df['string_date']))
    
    def test_convert_datetime_features(self):
        """Test funkcji convert_datetime_features."""
        # Wywołaj funkcję
        converted_df = convert_datetime_features(self.df)
        
        # Sprawdź, czy kolumny datetime zostały przekonwertowane na liczby
        self.assertTrue(pd.api.types.is_numeric_dtype(converted_df['timestamp']))
        self.assertTrue(pd.api.types.is_numeric_dtype(converted_df['date_only']))
        
        # Sprawdź, czy inne kolumny pozostały niezmienione
        self.assertEqual(converted_df['numeric'].dtype, self.df['numeric'].dtype)
        self.assertEqual(converted_df['string'].dtype, self.df['string'].dtype)
    
    def test_convert_object_features(self):
        """Test funkcji convert_object_features."""
        # Utwórz DataFrame z kolumnami typu object
        test_data = {
            'string_date': ['2025-06-01', '2025-06-02', '2025-06-03'],
            'string': ['a', 'b', 'c']
        }
        test_df = pd.DataFrame(test_data)
        
        # Wywołaj funkcję
        converted_df = convert_object_features(test_df)
        
        # Sprawdź, czy string_date został przekonwertowany na datetime
        self.assertTrue(pd.api.types.is_numeric_dtype(converted_df['string_date']))
        
        # Sprawdź, czy string pozostał niezmieniony
        self.assertEqual(converted_df['string'].dtype, test_df['string'].dtype)
    
    def test_date_to_numeric(self):
        """Test funkcji date_to_numeric."""
        # Test dla datetime
        dt = datetime(2025, 6, 1, 12, 0, 0)
        numeric_dt = date_to_numeric(dt)
        self.assertIsInstance(numeric_dt, float)
        
        # Test dla date
        d = date(2025, 6, 1)
        numeric_d = date_to_numeric(d)
        self.assertIsInstance(numeric_d, float)
        
        # Test dla string
        s = '2025-06-01'
        numeric_s = date_to_numeric(s)
        self.assertIsInstance(numeric_s, float)
    
    def test_safe_float_conversion(self):
        """Test funkcji safe_float_conversion."""
        # Test dla różnych typów danych
        self.assertEqual(safe_float_conversion(None), 0.0)
        self.assertEqual(safe_float_conversion(np.nan), 0.0)
        self.assertIsInstance(safe_float_conversion(datetime.now()), float)
        self.assertIsInstance(safe_float_conversion(date.today()), float)
        self.assertEqual(safe_float_conversion('1.0'), 1.0)
        self.assertEqual(safe_float_conversion(1), 1.0)
        self.assertEqual(safe_float_conversion(1.0), 1.0)
    
    def test_extract_time_features(self):
        """Test funkcji extract_time_features."""
        # Wywołaj funkcję
        df_with_time = extract_time_features(self.df, 'timestamp')
        
        # Sprawdź, czy zostały dodane nowe kolumny
        self.assertIn('hour', df_with_time.columns)
        self.assertIn('day_of_week', df_with_time.columns)
        self.assertIn('day_of_month', df_with_time.columns)
        self.assertIn('month', df_with_time.columns)
        self.assertIn('year', df_with_time.columns)
        self.assertIn('is_weekend', df_with_time.columns)
        self.assertIn('hour_sin', df_with_time.columns)
        self.assertIn('hour_cos', df_with_time.columns)

if __name__ == '__main__':
    unittest.main()

