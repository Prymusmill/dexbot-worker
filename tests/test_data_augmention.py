"""
Testy dla modułu augmentacji danych.
"""

import sys
import os
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Dodaj ścieżkę do modułu
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importuj moduł do testowania
from ml.data_augmentation import (
    load_historical_data,
    balance_directional_classes,
    generate_synthetic_data,
    generate_synthetic_samples_for_class,
    generate_random_data,
    generate_random_samples_for_class,
    augment_time_series_data,
    save_augmented_data
)

class TestDataAugmentation(unittest.TestCase):
    """Testy dla modułu augmentacji danych."""
    
    def setUp(self):
        """Przygotowanie danych testowych."""
        # Utwórz przykładowy DataFrame z danymi
        self.test_data = {
            'timestamp': [datetime.now() - timedelta(hours=i) for i in range(10)],
            'price': [100 + i for i in range(10)],
            'volume': [1000 * (i + 1) for i in range(10)],
            'rsi': [50 + i for i in range(10)],
            'direction': ['long', 'short', 'long', 'short', 'hold', 'long', 'short', 'long', 'short', 'hold']
        }
        self.df = pd.DataFrame(self.test_data)
        
        # Utwórz tymczasowy plik CSV
        self.csv_path = 'test_data.csv'
        self.df.to_csv(self.csv_path, index=False)
    
    def tearDown(self):
        """Czyszczenie po testach."""
        # Usuń tymczasowy plik CSV
        if os.path.exists(self.csv_path):
            os.remove(self.csv_path)
    
    def test_load_historical_data(self):
        """Test funkcji load_historical_data."""
        # Wywołaj funkcję z plikiem CSV
        data = load_historical_data(csv_path=self.csv_path, min_samples=5)
        
        # Sprawdź, czy dane zostały załadowane
        self.assertGreaterEqual(len(data), 5)
        
        # Wywołaj funkcję z minimalną liczbą próbek większą niż w pliku CSV
        data = load_historical_data(csv_path=self.csv_path, min_samples=20)
        
        # Sprawdź, czy zostały wygenerowane dodatkowe próbki
        self.assertGreaterEqual(len(data), 20)
    
    def test_balance_directional_classes(self):
        """Test funkcji balance_directional_classes."""
        # Utwórz niezbalansowany DataFrame
        unbalanced_data = {
            'timestamp': [datetime.now() - timedelta(hours=i) for i in range(10)],
            'price': [100 + i for i in range(10)],
            'direction': ['long', 'long', 'long', 'long', 'long', 'long', 'long', 'short', 'short', 'hold']
        }
        unbalanced_df = pd.DataFrame(unbalanced_data)
        
        # Wywołaj funkcję
        balanced_df = balance_directional_classes(unbalanced_df, target_col='direction', min_samples_per_class=5)
        
        # Sprawdź, czy klasy są zbalansowane
        class_counts = balanced_df['direction'].value_counts()
        self.assertGreaterEqual(class_counts['long'], 5)
        self.assertGreaterEqual(class_counts['short'], 5)
        self.assertGreaterEqual(class_counts['hold'], 5)
    
    def test_generate_synthetic_data(self):
        """Test funkcji generate_synthetic_data."""
        # Wywołaj funkcję
        synthetic_df = generate_synthetic_data(self.df, n_samples=5)
        
        # Sprawdź, czy zostały wygenerowane próbki
        self.assertEqual(len(synthetic_df), 5)
        
        # Sprawdź, czy wygenerowane próbki mają te same kolumny
        self.assertEqual(set(synthetic_df.columns), set(self.df.columns))
    
    def test_generate_synthetic_samples_for_class(self):
        """Test funkcji generate_synthetic_samples_for_class."""
        # Wywołaj funkcję
        synthetic_df = generate_synthetic_samples_for_class(
            self.df, target_class='long', n_samples=5, target_col='direction'
        )
        
        # Sprawdź, czy zostały wygenerowane próbki
        self.assertEqual(len(synthetic_df), 5)
        
        # Sprawdź, czy wszystkie próbki mają odpowiednią klasę
        self.assertTrue((synthetic_df['direction'] == 'long').all())
    
    def test_generate_random_data(self):
        """Test funkcji generate_random_data."""
        # Wywołaj funkcję
        random_df = generate_random_data(n_samples=5)
        
        # Sprawdź, czy zostały wygenerowane próbki
        self.assertEqual(len(random_df), 5)
        
        # Sprawdź, czy wygenerowane próbki mają odpowiednie kolumny
        self.assertIn('timestamp', random_df.columns)
        self.assertIn('price', random_df.columns)
        self.assertIn('volume', random_df.columns)
        self.assertIn('rsi', random_df.columns)
        self.assertIn('direction', random_df.columns)
    
    def test_generate_random_samples_for_class(self):
        """Test funkcji generate_random_samples_for_class."""
        # Wywołaj funkcję
        random_df = generate_random_samples_for_class(target_class='short', n_samples=5)
        
        # Sprawdź, czy zostały wygenerowane próbki
        self.assertEqual(len(random_df), 5)
        
        # Sprawdź, czy wszystkie próbki mają odpowiednią klasę
        self.assertTrue((random_df['direction'] == 'short').all())
    
    def test_augment_time_series_data(self):
        """Test funkcji augment_time_series_data."""
        # Wywołaj funkcję
        augmented_df = augment_time_series_data(self.df, time_col='timestamp', value_col='price')
        
        # Sprawdź, czy zostały wygenerowane dodatkowe próbki
        self.assertGreater(len(augmented_df), len(self.df))
    
    def test_save_augmented_data(self):
        """Test funkcji save_augmented_data."""
        # Wywołaj funkcję
        result = save_augmented_data(self.df, csv_path='augmented_data.csv')
        
        # Sprawdź, czy zapis się powiódł
        self.assertTrue(result)
        
        # Sprawdź, czy plik został utworzony
        self.assertTrue(os.path.exists('augmented_data.csv'))
        
        # Usuń utworzony plik
        if os.path.exists('augmented_data.csv'):
            os.remove('augmented_data.csv')

if __name__ == '__main__':
    unittest.main()

