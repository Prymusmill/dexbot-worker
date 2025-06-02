# ml_debug.py - Diagnoza problemów z ML
import pandas as pd
import numpy as np
from database.db_manager import get_db_manager
import matplotlib.pyplot as plt
import seaborn as sns


def diagnose_ml_data():
    """Kompleksowa diagnoza danych ML"""
    print("🔍 DIAGNOZA DANYCH ML")
    print("=" * 50)

    # Pobierz dane
    try:
        db = get_db_manager()
        df = db.get_all_transactions_for_ml()
        print(f"✅ Załadowano {len(df)} rekordów")
    except Exception as e:
        print(f"❌ Błąd ładowania danych: {e}")
        return

    if len(df) == 0:
        print("❌ Brak danych!")
        return

    # PODSTAWOWE STATYSTYKI
    print("\n📊 PODSTAWOWE STATYSTYKI:")
    print(f"Liczba rekordów: {len(df):,}")
    print(f"Kolumny: {list(df.columns)}")
    print(f"Typy danych:\n{df.dtypes}")

    # SPRAWDŹ NULL VALUES
    print("\n🔍 NULL VALUES:")
    null_counts = df.isnull().sum()
    for col, count in null_counts.items():
        if count > 0:
            print(
                f"❌ {col}: {count} null values ({
                    count /
                    len(df) *
                    100:.1f}%)")

    # KLUCZOWE KOLUMNY
    key_columns = [
        'price',
        'volume',
        'rsi',
        'amount_in',
        'amount_out',
        'profitable']

    print("\n📈 STATYSTYKI KLUCZOWYCH KOLUMN:")
    for col in key_columns:
        if col in df.columns:
            print(f"\n{col.upper()}:")
            print(f"  Mean: {df[col].mean():.6f}")
            print(f"  Std:  {df[col].std():.6f}")
            print(f"  Min:  {df[col].min():.6f}")
            print(f"  Max:  {df[col].max():.6f}")
            print(f"  Nulls: {df[col].isnull().sum()}")
            print(f"  Unique: {df[col].nunique()}")

            # Sprawdź dziwne wartości
            if col == 'price' and df[col].min() <= 0:
                print(f"  ⚠️ PROBLEM: Ceny <= 0!")
            if col == 'rsi' and (df[col].min() < 0 or df[col].max() > 100):
                print(f"  ⚠️ PROBLEM: RSI poza zakresem 0-100!")

    # TARGET VARIABLE ANALYSIS
    print("\n🎯 ANALIZA TARGET VARIABLE (profitable):")
    if 'profitable' in df.columns:
        profitable_counts = df['profitable'].value_counts()
        print(
            f"True (profitable): {
                profitable_counts.get(
                    True,
                    0)} ({
                profitable_counts.get(
                    True,
                    0) /
                len(df) *
                100:.1f}%)")
        print(
            f"False (loss): {
                profitable_counts.get(
                    False,
                    0)} ({
                profitable_counts.get(
                    False,
                    0) /
                len(df) *
                100:.1f}%)")

        # Sprawdź balance
        balance_ratio = min(profitable_counts) / max(profitable_counts)
        print(f"Balance ratio: {balance_ratio:.3f}")
        if balance_ratio < 0.1:
            print("⚠️ PROBLEM: Bardzo niezrównoważone klasy!")

    # PRICE CHANGE ANALYSIS
    print("\n💰 ANALIZA ZMIAN CEN:")
    if 'price' in df.columns and len(df) > 1:
        df_sorted = df.sort_values('timestamp')
        price_changes = df_sorted['price'].pct_change().dropna()

        print(f"Średnia zmiana ceny: {price_changes.mean():.6f}")
        print(f"Std zmian cen: {price_changes.std():.6f}")
        print(f"Min zmiana: {price_changes.min():.6f}")
        print(f"Max zmiana: {price_changes.max():.6f}")

        # Sprawdź czy zmiany są zbyt małe
        if price_changes.std() < 0.001:
            print("⚠️ PROBLEM: Bardzo małe zmiany cen - trudne do przewidzenia!")

    # CORRELATION ANALYSIS
    print("\n🔗 ANALIZA KORELACJI:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()

        # Korelacja z profitable
        if 'profitable' in df.columns:
            df['profitable_numeric'] = df['profitable'].astype(int)
            correlations_with_target = df[numeric_cols].corrwith(
                df['profitable_numeric'])

            print("Korelacje z profitable:")
            for col, corr in correlations_with_target.items():
                if col != 'profitable_numeric' and not np.isnan(corr):
                    print(f"  {col}: {corr:.4f}")

    # DATA QUALITY ISSUES
    print("\n⚠️ PROBLEMY Z JAKOŚCIĄ DANYCH:")
    issues = []

    # Sprawdź duplikaty
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        issues.append(f"Duplikaty: {duplicates}")

    # Sprawdź outliers w price
    if 'price' in df.columns:
        Q1 = df['price'].quantile(0.25)
        Q3 = df['price'].quantile(0.75)
        IQR = Q3 - Q1
        outliers = (
            (df['price'] < (
                Q1 -
                1.5 *
                IQR)) | (
                df['price'] > (
                    Q3 +
                    1.5 *
                    IQR))).sum()
        if outliers > len(df) * 0.1:  # Więcej niż 10% outliers
            issues.append(
                f"Dużo outliers w price: {outliers} ({
                    outliers / len(df) * 100:.1f}%)")

    # Sprawdź constant values
    for col in numeric_cols:
        if df[col].nunique() == 1:
            issues.append(f"Stała wartość w {col}: {df[col].iloc[0]}")

    if issues:
        for issue in issues:
            print(f"❌ {issue}")
    else:
        print("✅ Brak oczywistych problemów z jakością danych")

    # REKOMENDACJE
    print("\n💡 REKOMENDACJE:")

    if len(df) < 1000:
        print("1. ⚠️ Za mało danych - poczekaj na więcej transakcji (min 1000)")

    if 'price' in df.columns and df['price'].std() < 0.01:
        print("2. ⚠️ Małe zmiany cen - rozważ użycie percentage change")

    if 'profitable' in df.columns:
        balance_ratio = df['profitable'].value_counts(
        ).min() / df['profitable'].value_counts().max()
        if balance_ratio < 0.3:
            print("3. ⚠️ Niezrównoważone klasy - użyj class_weight='balanced'")

    print("4. ✅ Użyj tylko najważniejszych features: price, volume, rsi")
    print("5. ✅ Dodaj cross-validation")
    print("6. ✅ Zwiększ regularization")

    return df


if __name__ == "__main__":
    diagnose_ml_data()
