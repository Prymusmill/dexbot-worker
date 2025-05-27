# dashboard.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import numpy as np
from streamlit_autorefresh import st_autorefresh

# Konfiguracja strony
st.set_page_config(
    page_title="DexBot Trading Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Auto-refresh co 30 sekund (opcjonalnie)
st_autorefresh(interval=30000, key="datarefresh")

@st.cache_data
def load_trading_data():
    """Załaduj dane z memory.csv"""
    try:
        if not os.path.exists("data/memory.csv"):
            st.error("❌ Plik data/memory.csv nie został znaleziony!")
            return pd.DataFrame()
        
        df = pd.read_csv("data/memory.csv")
        
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
    
    # Sharpe ratio (uproszczona wersja)
    if df['net_pnl'].std() != 0:
        sharpe_ratio = df['net_pnl'].mean() / df['net_pnl'].std() * np.sqrt(len(df))
    else:
        sharpe_ratio = 0
    
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
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown
    }

def main():
    # Header
    st.title("🚀 DexBot Trading Dashboard")
    st.markdown("---")
    
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
            st.metric("Rozmiar pliku danych", f"{file_size / 1024 / 1024:.1f} MB")
        
        st.markdown("---")
        
        # Filtry czasowe
        st.subheader("🕒 Filtry Czasowe")
        time_filter = st.selectbox(
            "Okres analizy",
            ["Wszystko", "Ostatnie 24h", "Ostatnie 7 dni", "Ostatni miesiąc"]
        )
    
    # Wczytaj dane
    df = load_trading_data()
    
    if df.empty:
        st.warning("⚠️ Brak danych do wyświetlenia")
        return
    
    # Filtrowanie czasowe
    if time_filter != "Wszystko":
        now = datetime.now()
        if time_filter == "Ostatnie 24h":
            cutoff = now - timedelta(hours=24)
        elif time_filter == "Ostatnie 7 dni":
            cutoff = now - timedelta(days=7)
        elif time_filter == "Ostatni miesiąc":
            cutoff = now - timedelta(days=30)
        
        df = df[df['timestamp'] >= cutoff]
    
    # Oblicz metryki
    metrics = calculate_metrics(df)
    
    if not metrics:
        st.warning("⚠️ Brak danych dla wybranego okresu")
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
        st.metric(
            "Sharpe Ratio",
            f"{metrics['sharpe_ratio']:.2f}",
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
        last_trade_time = df['timestamp'].max()
        time_since = datetime.now() - last_trade_time.to_pydatetime()
        st.metric(
            "Ostatnia transakcja",
            f"{time_since.seconds // 60}min temu",
            delta=None
        )
    
    st.markdown("---")
    
    # Wykresy
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Performance", "📊 Analiza Transakcji", "⏰ Aktywność w Czasie", "🎯 Risk Analysis"])
    
    with tab1:
        st.subheader("Cumulative P&L w czasie")
        
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
        
        # Histogram P&L
        col1, col2 = st.columns(2)
        
        with col1:
            fig_hist = px.histogram(
                df, 
                x='net_pnl', 
                nbins=50,
                title="Rozkład P&L na transakcję",
                color_discrete_sequence=['lightblue']
            )
            fig_hist.update_layout(height=300)
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Win/Loss ratio pie chart
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
    
    with tab2:
        st.subheader("Szczegółowa analiza transakcji")
        
        # Statystyki price impact
        col1, col2 = st.columns(2)
        
        with col1:
            fig_price_impact = px.histogram(
                df,
                x='price_impact',
                nbins=30,
                title="Rozkład Price Impact",
                color_discrete_sequence=['orange']
            )
            st.plotly_chart(fig_price_impact, use_container_width=True)
        
        with col2:
            # Scatter plot: price_impact vs pnl
            fig_scatter = px.scatter(
                df.sample(min(1000, len(df))),  # Sample dla wydajności
                x='price_impact',
                y='net_pnl',
                title="Price Impact vs P&L",
                color='profitable',
                color_discrete_map={True: 'green', False: 'red'}
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Ostatnie transakcje
        st.subheader("🕒 Ostatnie 20 transakcji")
        recent_trades = df.tail(20)[['timestamp', 'input_token', 'output_token', 'amount_in', 'amount_out', 'price_impact', 'net_pnl', 'profitable']].copy()
        recent_trades['timestamp'] = recent_trades['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Kolorowanie wiersży
        def color_row(row):
            if row['profitable']:
                return ['background-color: rgba(0, 255, 0, 0.1)'] * len(row)
            else:
                return ['background-color: rgba(255, 0, 0, 0.1)'] * len(row)
        
        styled_df = recent_trades.style.apply(color_row, axis=1)
        st.dataframe(styled_df, use_container_width=True)
    
    with tab3:
        st.subheader("Aktywność tradingowa w czasie")
        
        # Trades per hour
        hourly_trades = df.groupby('hour').size().reset_index(name='trade_count')
        
        fig_hourly = px.bar(
            hourly_trades,
            x='hour',
            y='trade_count',
            title="Liczba transakcji według godziny",
            color='trade_count',
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig_hourly, use_container_width=True)
        
        # Daily performance
        if len(df['date'].unique()) > 1:
            daily_stats = df.groupby('date').agg({
                'net_pnl': ['sum', 'count', 'mean'],
                'profitable': 'sum'
            }).round(4)
            
            daily_stats.columns = ['Daily_PnL', 'Trades_Count', 'Avg_PnL', 'Winning_Trades']
            daily_stats['Win_Rate'] = (daily_stats['Winning_Trades'] / daily_stats['Trades_Count'] * 100).round(1)
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
    
    with tab4:
        st.subheader("Analiza ryzyka")
        
        # Drawdown analysis
        cumulative = df['cumulative_pnl']
        rolling_max = cumulative.expanding().max()
        drawdown = cumulative - rolling_max
        
        fig_dd = go.Figure()
        
        fig_dd.add_trace(go.Scatter(
            x=df['timestamp'],
            y=drawdown,
            fill='tonexty',
            mode='lines',
            name='Drawdown',
            line=dict(color='red')
        ))
        
        fig_dd.update_layout(
            title="Analiza Drawdown",
            xaxis_title="Czas",
            yaxis_title="Drawdown ($)",
            height=400
        )
        
        st.plotly_chart(fig_dd, use_container_width=True)
        
        # Risk metrics table
        risk_metrics = {
            'Metryka': [
                'Max Drawdown',
                'Volatility (std)',
                'Sharpe Ratio',
                'Worst Trade',
                'Best Trade',
                'Avg Trade P&L',
                '95% VaR (dzienne)'
            ],
            'Wartość': [
                f"${metrics['max_drawdown']:.4f}",
                f"${df['net_pnl'].std():.4f}",
                f"{metrics['sharpe_ratio']:.2f}",
                f"${metrics['worst_trade']:.4f}",
                f"${metrics['best_trade']:.4f}",
                f"${metrics['avg_trade_pnl']:.4f}",
                f"${df['net_pnl'].quantile(0.05):.4f}"
            ]
        }
        
        st.subheader("📋 Tabela metryk ryzyka")
        st.table(pd.DataFrame(risk_metrics))

if __name__ == "__main__":
    main()