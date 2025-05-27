import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import numpy as np

st.set_page_config(
    page_title="DexBot Trading Dashboard",
    page_icon="🚀",
    layout="wide"
)

@st.cache_data
def load_trading_data():
    try:
        if not os.path.exists("data/memory.csv"):
            st.error("❌ Plik data/memory.csv nie został znaleziony!")
            return pd.DataFrame()
        
        df = pd.read_csv("data/memory.csv")
        
        if len(df) == 0:
            st.warning("⚠️ Plik memory.csv jest pusty")
            return pd.DataFrame()
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['profitable'] = df['amount_out'] > df['amount_in']
        df['pnl'] = df['amount_out'] - df['amount_in']
        df['net_pnl'] = df['pnl'] - (df['amount_in'] * 0.001)
        df = df.sort_values('timestamp')
        df['cumulative_pnl'] = df['net_pnl'].cumsum()
        
        return df
    except Exception as e:
        st.error(f"❌ Błąd wczytywania danych: {e}")
        return pd.DataFrame()

def main():
    st.title("🚀 DexBot Trading Dashboard")
    
    with st.sidebar:
        st.header("⚙️ Kontrola")
        if st.button("🔄 Odśwież Dane"):
            st.cache_data.clear()
            st.rerun()

    df = load_trading_data()
    
    if df.empty:
        st.warning("⚠️ Brak danych")
        return
    
    st.metric("Łączna liczba transakcji", len(df))
    st.metric("Całkowity P&L", f"${df['net_pnl'].sum():.4f}")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['cumulative_pnl'], mode='lines'))
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()