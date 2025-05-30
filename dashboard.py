# dashboard.py - Complete dashboard with ML integration (FIXED CACHE)
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import numpy as np
import time
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(
    page_title="DexBot Trading Dashboard",
    page_icon="🚀",
    layout="wide"
)

# FIXED: Usunięto @st.cache_data żeby zawsze ładować najnowsze dane


def load_trading_data():
    try:
        # Najpierw spróbuj PostgreSQL
        try:
            from database.db_manager import get_db_manager
            db_manager = get_db_manager()
            df = db_manager.get_recent_transactions(limit=500)  # NOWE - 500 najnowszych

            if len(df) > 100:
                st.success(f"✅ Loaded {len(df)} transactions from PostgreSQL!")
                # Process data...
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['profitable'] = df['amount_out'] > df['amount_in']
                df['pnl'] = df['amount_out'] - df['amount_in']
                df['net_pnl'] = df['pnl'] - (df['amount_in'] * 0.001)
                df = df.sort_values('timestamp')
                df['cumulative_pnl'] = df['net_pnl'].cumsum()
                df['pnl_percentage'] = (
                    df['amount_out'] - df['amount_in']) / df['amount_in'] * 100
                df['fees_estimated'] = df['amount_in'] * 0.001
                df['date'] = df['timestamp'].dt.date
                df['hour'] = df['timestamp'].dt.hour
                return df
        except Exception as e:
            st.warning(f"⚠️ PostgreSQL failed: {e}, fallback to CSV")

        # Fallback to CSV
         # Fallback to CSV
        if not os.path.exists("data/memory.csv"):
            st.error("❌ No data source available!")
            return pd.DataFrame()
        
        df = pd.read_csv("data/memory.csv")
        # Add original CSV processing here
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['profitable'] = df['amount_out'] > df['amount_in']
        df['pnl'] = df['amount_out'] - df['amount_in']
        df['net_pnl'] = df['pnl'] - (df['amount_in'] * 0.001)
        df = df.sort_values('timestamp')
        df['cumulative_pnl'] = df['net_pnl'].cumsum()
        df['pnl_percentage'] = (df['amount_out'] - df['amount_in']) / df['amount_in'] * 100
        df['fees_estimated'] = df['amount_in'] * 0.001
        df['date'] = df['timestamp'].dt.date
        df['hour'] = df['timestamp'].dt.hour
        return df
    except Exception as e:
        st.error(f"❌ Błąd wczytywania danych: {e}")
        return pd.DataFrame()

def calculate_metrics(df):
    """Oblicz kluczowe metryki"""
    if df.empty:
        return {}

    total_trades = len(df)
    winning_trades = len(df[df['profitable']])
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0

    total_pnl = df['net_pnl'].sum()
    avg_trade_pnl = df['net_pnl'].mean()

    best_trade = df['net_pnl'].max()
    worst_trade = df['net_pnl'].min()

    # Max drawdown
    cumulative = df['cumulative_pnl']
    rolling_max = cumulative.expanding().max()
    drawdown = cumulative - rolling_max
    max_drawdown = drawdown.min()

    # Sharpe ratio (simplified)
    if df['net_pnl'].std() != 0:
        sharpe_ratio = df['net_pnl'].mean(
        ) / df['net_pnl'].std() * np.sqrt(len(df))
    else:
        sharpe_ratio = 0

    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_trade_pnl': avg_trade_pnl,
        'best_trade': best_trade,
        'worst_trade': worst_trade,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio
    }


def display_ml_predictions(df):
    """Display ML predictions section - FIXED"""
    st.header("🤖 Machine Learning Predictions")

    # DODANE: Info o aktualnym stanie danych
    st.info(f"📊 Załadowano {len(df)} transakcji. ML wymaga minimum 100.")

    try:
        # Try to load ML integration
        from ml.price_predictor import MLTradingIntegration
        ml_integration = MLTradingIntegration()
        st.success("✅ ML Integration loaded successfully")

        # Check if we have enough data
        if len(df) >= 100:
            with st.spinner("Generating ML prediction..."):
                # DODANE: Debug info
                st.write(
                    f"🔍 Próbuję wygenerować predykcję dla {len(df)} transakcji...")

                # Check required columns
                required_cols = ['price', 'volume', 'rsi']
                missing_cols = [
                    col for col in required_cols if col not in df.columns]

                if missing_cols:
                    st.error(f"❌ Brakuje kolumn ML: {missing_cols}")
                    st.write(f"📋 Dostępne kolumny: {list(df.columns)}")
                    return
                else:
                    st.success(
                        f"✅ Wszystkie kolumny ML obecne: {required_cols}")

                try:
                    if hasattr(ml_integration, 'get_ensemble_prediction_with_reality_check'):
                        prediction = ml_integration.get_ensemble_prediction_with_reality_check(df)
                    else:
                        prediction = ml_integration.get_ensemble_prediction(df)
                    st.write(f"🔍 Otrzymana predykcja: {prediction}")
                except Exception as e:
                    st.error(f"❌ Błąd generowania predykcji: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                    return

            if prediction and 'predicted_price' in prediction:
                st.success("🎉 ML Prediction wygenerowana pomyślnie!")

                # Main prediction metrics
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    delta_color = "normal" if prediction['price_change_pct'] >= 0 else "inverse"
                    st.metric(
                        "🔮 Przewidywana cena",
                        f"${prediction['predicted_price']:.4f}",
                        delta=f"{prediction['price_change_pct']:+.2f}%"
                    )

                with col2:
                    direction_color = "🟢" if prediction['direction'] == 'up' else "🔴"
                    st.metric(
                        "📈 Kierunek",
                        f"{direction_color} {prediction['direction'].upper()}",
                        delta=None
                    )

                with col3:
                    confidence_pct = prediction['confidence'] * 100
                    confidence_color = "normal" if confidence_pct >= 70 else "inverse"
                    st.metric(
                        "🎯 Pewność modelu",
                        f"{confidence_pct:.1f}%",
                        delta=None
                    )

                with col4:
                    st.metric(
                        "🤖 Modeli aktywnych",
                        prediction.get('model_count', 1),
                        delta=None
                    )

                # Individual model predictions
                if 'individual_predictions' in prediction:
                    st.subheader("🔍 Predykcje poszczególnych modeli")

                    pred_data = []
                    current_price = prediction.get(
                        'current_price', df['price'].iloc[-1] if 'price' in df.columns else 0)

                    for model_name, pred_price in prediction['individual_predictions'].items(
                    ):
                        change_pct = (
                            (pred_price - current_price) / current_price) * 100 if current_price > 0 else 0
                        pred_data.append({
                            'Model': model_name.replace('_', ' ').title(),
                            'Przewidywana cena': f"${pred_price:.4f}",
                            'Zmiana %': f"{change_pct:+.2f}%",
                            'Kierunek': '🟢 UP' if change_pct > 0 else '🔴 DOWN'
                        })

                    st.dataframe(
                        pd.DataFrame(pred_data),
                        use_container_width=True)

                # ML Model Performance
                st.subheader("📊 Performance Modeli ML")
                try:
                    performance = ml_integration.get_model_performance()
                    if performance:
                        perf_data = []
                        for model_name, metrics in performance.items():
                            perf_data.append({
                                'Model': model_name.replace('_', ' ').title(),
                                'Accuracy': f"{metrics.get('accuracy', 0):.1f}%",
                                'R² Score': f"{metrics.get('r2', 0):.3f}",
                                'MAE': f"{metrics.get('mae', 0):.6f}",
                                'Training Samples': f"{metrics.get('training_samples', 0):,}",
                                'Last Trained': metrics.get('last_trained', 'Never')
                            })

                        if perf_data:
                            st.dataframe(
                                pd.DataFrame(perf_data),
                                use_container_width=True)
                        else:
                            st.info("📊 Modele nie zostały jeszcze wytrenowane")
                    else:
                        st.info("📊 Brak danych o performance modeli")
                except Exception as e:
                    st.warning(f"⚠️ Nie można pobrać performance modeli: {e}")

                # Prediction confidence visualization
                st.subheader("📈 Wizualizacja Predykcji")

                # Create prediction vs actual chart
                try:
                    recent_data = df.tail(50)
                    if 'price' in df.columns:
                        fig_pred = go.Figure()

                        # Actual prices
                        fig_pred.add_trace(go.Scatter(
                            x=recent_data['timestamp'],
                            y=recent_data['price'],
                            mode='lines',
                            name='Actual Price',
                            line=dict(color='blue', width=2)
                        ))

                        # Predicted price (as a point)
                        fig_pred.add_trace(go.Scatter(
                            x=[datetime.now()],
                            y=[prediction['predicted_price']],
                            mode='markers',
                            name=f'ML Prediction ({prediction["direction"].upper()})',
                            marker=dict(
                                color='green' if prediction['direction'] == 'up' else 'red',
                                size=15,
                                symbol='triangle-up' if prediction['direction'] == 'up' else 'triangle-down'
                            )
                        ))

                        fig_pred.update_layout(
                            title="Aktualne Ceny vs ML Predykcja",
                            xaxis_title="Czas",
                            yaxis_title="Cena ($)",
                            height=400
                        )

                        st.plotly_chart(fig_pred, use_container_width=True)
                    else:
                        st.warning("⚠️ Brak kolumny 'price' do wykresu")

                except Exception as e:
                    st.warning(f"⚠️ Nie można utworzyć wykresu predykcji: {e}")

            else:
                st.error("❌ Nie udało się wygenerować predykcji ML")
                if prediction and 'error' in prediction:
                    st.error(f"Błąd: {prediction['error']}")
                st.write(f"🔍 Otrzymana odpowiedź: {prediction}")
        else:
            st.info(
                f"📊 Potrzeba więcej danych do predykcji ML (obecne: {len(df)}/100 transakcji)")

            # Show progress bar
            progress = min(len(df) / 100, 1.0)
            st.progress(progress)
            st.caption(
                f"Postęp do pierwszej predykcji ML: {len(df)}/100 transakcji")

    except ImportError as e:
        st.error(f"❌ Moduły ML nie są dostępne: {e}")
        st.info(
            "💡 Aby włączyć predykcje ML, upewnij się że system ma zainstalowane: scikit-learn, tensorflow")
    except Exception as e:
        st.error(f"❌ Błąd ML: {e}")
        import traceback
        st.code(traceback.format_exc())


def display_trading_performance(df, metrics):
    """Display comprehensive trading performance"""
    st.header("📊 Performance Tradingu")

    # Main metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Łączna liczba transakcji",
            f"{metrics['total_trades']:,}",
            delta=None
        )

    with col2:
        pnl_color = "normal" if metrics['total_pnl'] >= 0 else "inverse"
        st.metric(
            "Całkowity P&L",
            f"${metrics['total_pnl']:.4f}",
            delta=f"{metrics['avg_trade_pnl']:.4f} śr/trade"
        )

    with col3:
        st.metric(
            "Wskaźnik wygranych",
            f"{metrics['win_rate']:.1f}%",
            delta=None
        )

    with col4:
        last_trade_time = df['timestamp'].max()
        try:
            time_since = datetime.now() - last_trade_time.to_pydatetime().replace(tzinfo=None)
            minutes_ago = time_since.seconds // 60
        except Exception:
            minutes_ago = 0
        st.metric(
            "Ostatnia transakcja",
            f"{minutes_ago}min temu",
            delta=None
        )

    # Secondary metrics row
    col5, col6, col7, col8 = st.columns(4)

    with col5:
        st.metric(
            "Najlepsza transakcja",
            f"${metrics['best_trade']:.4f}",
            delta=None
        )

    with col6:
        st.metric(
            "Najgorsza transakcja",
            f"${metrics['worst_trade']:.4f}",
            delta=None
        )

    with col7:
        st.metric(
            "Max Drawdown",
            f"${metrics['max_drawdown']:.4f}",
            delta=None
        )

    with col8:
        st.metric(
            "Sharpe Ratio",
            f"{metrics['sharpe_ratio']:.2f}",
            delta=None
        )


def main():
    # Auto-refresh functionality - FIXED: zawsze czyść cache przy refresh
    auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (60s)", value=False)

    if auto_refresh:
        if 'refresh_counter' not in st.session_state:
            st.session_state.refresh_counter = 60

        st.sidebar.write(f"⏱️ Refresh za: {st.session_state.refresh_counter}s")

        if st.session_state.refresh_counter <= 0:
            st.session_state.refresh_counter = 60
            # REMOVED: st.cache_data.clear() - nie ma już cache
            st.rerun()
        else:
            st.session_state.refresh_counter -= 5
            time.sleep(5)
            st.rerun()

    # Header
    st.title("🚀 DexBot Trading Dashboard")
    st.markdown("---")

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Kontrola")

        # Status
        st.subheader("Status Systemu")
        st.success("🟢 Bot Aktywny")
        st.info("📊 Tryb: Real-time Trading")

        # Manual refresh button
        if st.button("🔄 Odśwież Dane"):
            # REMOVED: st.cache_data.clear() - nie ma już cache
            st.rerun()

        # File info
        if os.path.exists("data/memory.csv"):
            file_size = os.path.getsize("data/memory.csv")
            st.metric("Rozmiar pliku danych", f"{file_size / 1024:.1f} KB")

        # ML Controls
        st.subheader("🤖 ML Controls")
        show_ml = st.checkbox("Pokaż predykcje ML", value=True)

        if st.button("🔄 Retrain ML Models"):
            st.info("🤖 Model retraining będzie uruchomiony w tle...")

    # Load data - ZAWSZE najnowsze dane
    df = load_trading_data()

    if df.empty:
        st.warning("⚠️ Brak danych do wyświetlenia")
        st.info(
            "Worker może jeszcze nie zapisał danych lub wystąpił problem z plikiem.")
        return

    # Calculate metrics
    metrics = calculate_metrics(df)

    if not metrics:
        st.warning("⚠️ Nie można obliczyć metryk")
        return

    # Display trading performance
    display_trading_performance(df, metrics)

    st.markdown("---")

    # ML Predictions section
    if show_ml:
        display_ml_predictions(df)
        st.markdown("---")

    # Main chart - Cumulative P&L
    st.subheader("📈 Cumulative P&L w czasie")

    try:
        fig_pnl = go.Figure()
        fig_pnl.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['cumulative_pnl'],
                mode='lines',
                name='Cumulative P&L',
                line=dict(
                    color='green' if metrics['total_pnl'] >= 0 else 'red',
                    width=2)))

        fig_pnl.update_layout(
            title="Krzywa Equity",
            xaxis_title="Czas",
            yaxis_title="Cumulative P&L ($)",
            height=500
        )

        st.plotly_chart(fig_pnl, use_container_width=True)
    except Exception as e:
        st.error(f"Błąd wykresu P&L: {e}")

    # Additional analysis tabs
    tab1, tab2, tab3 = st.tabs(
        ["📊 Statystyki", "📋 Ostatnie Transakcje", "📈 Analiza Czasowa"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            # Win/Loss ratio pie chart
            try:
                win_loss_data = pd.DataFrame({
                    'Typ': ['Wygrane', 'Przegrane'],
                    'Liczba': [len(df[df['profitable']]), len(df[~df['profitable']])]
                })

                fig_pie = px.pie(
                    win_loss_data,
                    values='Liczba',
                    names='Typ',
                    title="Stosunek wygranych do przegranych",
                    color_discrete_map={'Wygrane': 'green', 'Przegrane': 'red'}
                )
                fig_pie.update_layout(height=400)
                st.plotly_chart(fig_pie, use_container_width=True)
            except Exception as e:
                st.error(f"Błąd wykresu kołowego: {e}")

        with col2:
            # P&L distribution histogram
            try:
                fig_hist = px.histogram(
                    df,
                    x='net_pnl',
                    nbins=50,
                    title="Rozkład P&L na transakcję",
                    color_discrete_sequence=['lightblue']
                )
                fig_hist.update_layout(height=400)
                st.plotly_chart(fig_hist, use_container_width=True)
            except Exception as e:
                st.error(f"Błąd histogramu: {e}")

    with tab2:
        # Recent trades table
        st.subheader("🕒 Ostatnie 20 transakcji")
        try:
            if len(df) > 0:
                recent_trades = df.tail(20)[['timestamp',
                                             'input_token',
                                             'output_token',
                                             'amount_in',
                                             'amount_out',
                                             'net_pnl',
                                             'profitable']].copy()
                recent_trades['timestamp'] = recent_trades['timestamp'].dt.strftime(
                    '%Y-%m-%d %H:%M:%S')

                # Style the dataframe
                def color_row(row):
                    if row['profitable']:
                        return [
                            'background-color: rgba(0, 255, 0, 0.1)'] * len(row)
                    else:
                        return [
                            'background-color: rgba(255, 0, 0, 0.1)'] * len(row)

                styled_df = recent_trades.style.apply(color_row, axis=1)
                st.dataframe(styled_df, use_container_width=True)

                # Summary of recent trades
                last_20_pnl = recent_trades['net_pnl'].sum()
                last_20_wins = len(recent_trades[recent_trades['profitable']])

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("P&L ostatnich 20", f"${last_20_pnl:.4f}")
                with col2:
                    st.metric("Wygrane w ostatnich 20", f"{last_20_wins}/20")
            else:
                st.warning("Brak transakcji do wyświetlenia")
        except Exception as e:
            st.error(f"Błąd wyświetlania transakcji: {e}")

    with tab3:
        # Time-based analysis
        st.subheader("Aktywność tradingowa w czasie")

        try:
            # Trades per hour
            if 'hour' in df.columns:
                hourly_trades = df.groupby(
                    'hour').size().reset_index(name='trade_count')

                fig_hourly = px.bar(
                    hourly_trades,
                    x='hour',
                    y='trade_count',
                    title="Liczba transakcji według godziny",
                    color='trade_count',
                    color_continuous_scale='viridis'
                )
                st.plotly_chart(fig_hourly, use_container_width=True)

            # Daily performance if we have multiple days
            if len(df['date'].unique()) > 1:
                daily_stats = df.groupby('date').agg({
                    'net_pnl': ['sum', 'count', 'mean'],
                    'profitable': 'sum'
                }).round(4)

                daily_stats.columns = [
                    'Daily_PnL',
                    'Trades_Count',
                    'Avg_PnL',
                    'Winning_Trades']
                daily_stats['Win_Rate'] = (
                    daily_stats['Winning_Trades'] /
                    daily_stats['Trades_Count'] *
                    100).round(1)
                daily_stats = daily_stats.reset_index()

                fig_daily = px.bar(
                    daily_stats,
                    x='date',
                    y='Daily_PnL',
                    title="Dzienny P&L",
                    color='Daily_PnL',
                    color_continuous_scale='RdYlGn'
                )
                st.plotly_chart(fig_daily, use_container_width=True)

                st.subheader("📊 Statystyki dzienne")
                st.dataframe(daily_stats, use_container_width=True)
        except Exception as e:
            st.error(f"Błąd analizy czasowej: {e}")

    # Footer info
    st.markdown("---")
    st.info("💡 Dashboard ładuje najnowsze dane przy każdym odświeżeniu. Włącz auto-refresh dla live updates.")

    # System info
    with st.expander("🔍 System Info"):
        st.write("**Trading Bot Status:**")
        st.write(f"- Całkowita liczba transakcji: {len(df):,}")
        st.write(
            f"- Ostatnia aktualizacja: {df['timestamp'].max() if not df.empty else 'N/A'}")
        st.write(f"- Rozmiar danych: {os.path.getsize('data/memory.csv') / 1024:.1f} KB" if os.path.exists(
            'data/memory.csv') else "- Plik danych niedostępny")

        # ML Status
        try:
            from ml.price_predictor import MLTradingIntegration
            st.write("**ML Status:** ✅ Dostępne")
        except ImportError:
            st.write("**ML Status:** ❌ Niedostępne")


if __name__ == "__main__":
    main()
