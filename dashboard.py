# dashboard_simple.py - Prostszy dashboard bez problematycznych wykresów
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import numpy as np
import time
import asyncio

# Konfiguracja strony
st.set_page_config(
    page_title="DexBot Trading Dashboard",
    page_icon="🚀",
    layout="wide"
)

@st.cache_data
def load_trading_data():
    """Załaduj dane z memory.csv"""
    try:
        if not os.path.exists("data/memory.csv"):
            st.error("❌ Plik data/memory.csv nie został znaleziony!")
            return pd.DataFrame()
        
        df = pd.read_csv("data/memory.csv")
        
        if len(df) == 0:
            st.warning("⚠️ Plik memory.csv jest pusty")
            return pd.DataFrame()
        
        # Konwersja timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Dodatkowe kolumny dla analiz
        df['profitable'] = df['amount_out'] > df['amount_in']
        df['pnl'] = df['amount_out'] - df['amount_in']
        df['pnl_percentage'] = (df['amount_out'] - df['amount_in']) / df['amount_in'] * 100
        df['fees_estimated'] = df['amount_in'] * 0.001  # Szacowane fees 0.1%
        df['net_pnl'] = df['pnl'] - df['fees_estimated']
        
        # Cumulative PnL
        df = df.sort_values('timestamp')
        df['cumulative_pnl'] = df['net_pnl'].cumsum()
        
        # Dodaj datę dla group by
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
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_trade_pnl': avg_trade_pnl,
        'best_trade': best_trade,
        'worst_trade': worst_trade,
        'max_drawdown': max_drawdown
    }

def main():
   # Header
   st.title("🚀 DexBot Trading Dashboard")
   st.markdown("---")
   
   # Auto-refresh w sidebar - lepsze rozwiązanie
   auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (60s)", value=True)
   
   if auto_refresh:
       # Refresh co 5 sekund dla smooth countdown
       if 'refresh_counter' not in st.session_state:
           st.session_state.refresh_counter = 60
       
       st.sidebar.write(f"⏱️ Refresh za: {st.session_state.refresh_counter}s")
       
       if st.session_state.refresh_counter <= 0:
           st.session_state.refresh_counter = 60
           st.cache_data.clear()
           st.rerun()
       else:
           st.session_state.refresh_counter -= 5
           time.sleep(5)
           st.rerun()
   
   # Sidebar
   with st.sidebar:
        st.header("⚙️ Kontrola")
        
        # Status
        st.subheader("Status Systemu")
        st.success("🟢 Bot Aktywny")
        st.info("📊 Tryb: Symulacja")
        
        # Refresh button
        if st.button("🔄 Odśwież Dane"):
            st.cache_data.clear()
            st.rerun()
        
        # File info
        if os.path.exists("data/memory.csv"):
            file_size = os.path.getsize("data/memory.csv")
            st.metric("Rozmiar pliku danych", f"{file_size / 1024:.1f} KB")
    
    # Wczytaj dane
    df = load_trading_data()
    
    if df.empty:
        st.warning("⚠️ Brak danych do wyświetlenia")
        st.info("Worker może jeszcze nie zapisał danych lub wystąpił problem z plikiem.")
        return
    
    # Oblicz metryki
    metrics = calculate_metrics(df)
    
    if not metrics:
        st.warning("⚠️ Nie można obliczyć metryk")
        return
    
    # Główne metryki
    st.header("📊 Kluczowe Metryki")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Łączna liczba transakcji",
            f"{metrics['total_trades']:,}",
            delta=None
        )
    
    with col2:
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
        time_since = datetime.now() - last_trade_time.to_pydatetime()
        st.metric(
            "Ostatnia transakcja",
            f"{time_since.seconds // 60}min temu",
            delta=None
        )
    
    # Druga linia metryk
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
        current_balance = metrics['total_pnl']
        st.metric(
            "Aktualny bilans",
            f"${current_balance:.4f}",
            delta=None
        )
    
    st.markdown("---")
    
    # Główny wykres - Cumulative P&L
    st.subheader("📈 Cumulative P&L w czasie")
    
    try:
        fig_pnl = go.Figure()
        fig_pnl.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['cumulative_pnl'],
            mode='lines',
            name='Cumulative P&L',
            line=dict(color='green' if metrics['total_pnl'] >= 0 else 'red', width=2)
        ))
        
        fig_pnl.update_layout(
            title="Krzywa Equity",
            xaxis_title="Czas",
            yaxis_title="Cumulative P&L ($)",
            height=400
        )
        
        st.plotly_chart(fig_pnl, use_container_width=True)
    except Exception as e:
        st.error(f"Błąd wykresu P&L: {e}")
    
    # Tabs dla dodatkowych analiz
    tab1, tab2 = st.tabs(["📊 Statystyki", "📋 Ostatnie Transakcje"])
    
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
                fig_pie.update_layout(height=300)
                st.plotly_chart(fig_pie, use_container_width=True)
            except Exception as e:
                st.error(f"Błąd wykresu kołowego: {e}")
        
        with col2:
            # Basic statistics table
            st.subheader("📋 Podstawowe statystyki")
            stats_data = {
                'Metryka': [
                    'Całkowite transakcje',
                    'Transakcje wygrane',
                    'Transakcje przegrane',
                    'Wskaźnik wygranych',
                    'Średni P&L na transakcję',
                    'Całkowity P&L'
                ],
                'Wartość': [
                    f"{metrics['total_trades']:,}",
                    f"{len(df[df['profitable']]):,}",
                    f"{len(df[~df['profitable']]):,}",
                    f"{metrics['win_rate']:.1f}%",
                    f"${metrics['avg_trade_pnl']:.4f}",
                    f"${metrics['total_pnl']:.4f}"
                ]
            }
            st.table(pd.DataFrame(stats_data))
    
    with tab2:
        # Ostatnie transakcje
        st.subheader("🕒 Ostatnie 20 transakcji")
        try:
            if len(df) > 0:
                recent_trades = df.tail(20)[['timestamp', 'input_token', 'output_token', 'amount_in', 'amount_out', 'net_pnl', 'profitable']].copy()
                recent_trades['timestamp'] = recent_trades['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                
                # Uproszczone kolorowanie
                st.dataframe(recent_trades, use_container_width=True)
                
                # Podsumowanie ostatnich 20
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
    
    # Auto refresh info
    st.markdown("---")
    st.info("💡 Kliknij 'Odśwież Dane' w menu po lewej aby zobaczyć najnowsze transakcje")

if __name__ == "__main__":
    main()