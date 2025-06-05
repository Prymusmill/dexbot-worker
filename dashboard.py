# dashboard.py - ENHANCED DIRECTIONAL TRADING DASHBOARD (LONG/SHORT/HOLD)
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
    page_title="🎯 DexBot Directional Trading Dashboard",
    page_icon="🎯",
    layout="wide"
)

# 🎯 DIRECTIONAL TRADING DATA LOADING
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_directional_trading_data():
    """🎯 Load directional trading data with enhanced metrics"""
    try:
        # Try PostgreSQL first
        try:
            from database.db_manager import get_db_manager
            db_manager = get_db_manager()

            # Load recent transactions with directional data
            df = db_manager.get_recent_transactions(limit=500)

            if len(df) > 50:
                st.success(f"✅ Loaded {len(df)} directional trades from PostgreSQL!")

                # Enhanced processing for directional data
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # 🎯 DIRECTIONAL DATA PROCESSING
                # Check if we have new directional format
                if 'action' in df.columns or 'direction' in df.columns:
                    df = process_directional_format(df)
                else:
                    df = convert_legacy_to_directional_format(df)
                
                df = df.sort_values('timestamp')
                df['date'] = df['timestamp'].dt.date
                df['hour'] = df['timestamp'].dt.hour

                return df

        except Exception as e:
            st.error(f"⚠️ PostgreSQL failed: {e}")

        # Fallback to CSV
        if not os.path.exists("data/memory.csv"):
            st.error("❌ No directional trading data available!")
            return pd.DataFrame()

        df = pd.read_csv("data/memory.csv")
        df = df.tail(500)  # Last 500 trades
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Process directional CSV data
        if 'action' in df.columns or 'direction' in df.columns:
            df = process_directional_format(df)
        else:
            df = convert_legacy_to_directional_format(df)
        
        df = df.sort_values('timestamp')
        df['date'] = df['timestamp'].dt.date
        df['hour'] = df['timestamp'].dt.hour

        return df

    except Exception as e:
        st.error(f"❌ Error loading directional trading data: {e}")
        return pd.DataFrame()


def process_directional_format(df: pd.DataFrame) -> pd.DataFrame:
    """🎯 Process data already in directional format"""
    try:
        # Ensure we have direction column
        if 'direction' not in df.columns and 'action' in df.columns:
            action_to_direction = {
                'LONG': 'long',
                'SHORT': 'short',
                'HOLD': 'hold',
                'CLOSE': 'hold'
            }
            df['direction'] = df['action'].map(action_to_direction).fillna('hold')
        
        # Ensure we have P&L columns
        if 'pnl' not in df.columns:
            if 'amount_out' in df.columns and 'amount_in' in df.columns:
                df['pnl'] = df['amount_out'] - df['amount_in']
            else:
                df['pnl'] = 0.0
        
        if 'pnl_percentage' not in df.columns:
            if 'amount_in' in df.columns and df['amount_in'].sum() > 0:
                df['pnl_percentage'] = (df['pnl'] / df['amount_in']) * 100
            else:
                df['pnl_percentage'] = 0.0
        
        # Calculate profitable from P&L
        df['profitable'] = df['pnl'] > 0
        
        # Calculate cumulative P&L
        df['cumulative_pnl'] = df['pnl'].cumsum()
        
        # Duration (if available)
        if 'duration_seconds' not in df.columns:
            df['duration_seconds'] = 60.0  # Default 1 minute
        
        return df
        
    except Exception as e:
        st.error(f"Error processing directional format: {e}")
        return df


def convert_legacy_to_directional_format(df: pd.DataFrame) -> pd.DataFrame:
    """🎯 Convert legacy format to directional"""
    try:
        st.info("🔄 Converting legacy data to directional format...")
        
        # Calculate P&L from legacy format
        if 'amount_out' in df.columns and 'amount_in' in df.columns:
            df['pnl'] = df['amount_out'] - df['amount_in']
            df['pnl_percentage'] = (df['pnl'] / df['amount_in']) * 100
        else:
            df['pnl'] = 0.0
            df['pnl_percentage'] = 0.0
        
        df['profitable'] = df['pnl'] > 0
        df['cumulative_pnl'] = df['pnl'].cumsum()
        
        # Create directional labels from profitability and RSI
        df['direction'] = 'hold'  # Default
        
        if 'rsi' in df.columns:
            # Enhanced directional conversion
            df.loc[(df['rsi'] < 30) & (df['profitable'] == True), 'direction'] = 'long'
            df.loc[(df['rsi'] > 70) & (df['profitable'] == True), 'direction'] = 'short'
            df.loc[(df['rsi'] >= 40) & (df['rsi'] <= 60), 'direction'] = 'hold'
        else:
            # Simple conversion
            df.loc[df['profitable'] == True, 'direction'] = 'long'
            df.loc[df['profitable'] == False, 'direction'] = 'short'
        
        # Add missing columns
        df['action'] = df['direction'].str.upper()
        df['duration_seconds'] = 60.0
        
        return df
        
    except Exception as e:
        st.error(f"Error converting legacy format: {e}")
        return df


def calculate_directional_metrics(df):
    """🎯 Calculate enhanced metrics for directional trading"""
    if df.empty:
        return {}

    total_trades = len(df)
    
    # 🎯 DIRECTIONAL TRADE BREAKDOWN
    direction_counts = df['direction'].value_counts() if 'direction' in df.columns else {}
    long_trades = direction_counts.get('long', 0)
    short_trades = direction_counts.get('short', 0)
    hold_periods = direction_counts.get('hold', 0)
    
    # P&L calculations
    total_pnl = df['pnl'].sum() if 'pnl' in df.columns else 0
    
    # 🎯 DIRECTIONAL PERFORMANCE
    directional_performance = {}
    
    for direction in ['long', 'short', 'hold']:
        direction_df = df[df['direction'] == direction] if 'direction' in df.columns else pd.DataFrame()
        
        if len(direction_df) > 0:
            direction_profitable = len(direction_df[direction_df['profitable']]) if 'profitable' in direction_df.columns else 0
            direction_win_rate = direction_profitable / len(direction_df)
            direction_pnl = direction_df['pnl'].sum() if 'pnl' in direction_df.columns else 0
            direction_avg_pnl = direction_pnl / len(direction_df)
            direction_avg_duration = direction_df['duration_seconds'].mean() if 'duration_seconds' in direction_df.columns else 0
        else:
            direction_win_rate = 0
            direction_pnl = 0
            direction_avg_pnl = 0
            direction_avg_duration = 0
        
        directional_performance[direction] = {
            'trades': len(direction_df),
            'wins': direction_profitable if len(direction_df) > 0 else 0,
            'win_rate': direction_win_rate,
            'total_pnl': direction_pnl,
            'avg_pnl': direction_avg_pnl,
            'avg_duration': direction_avg_duration
        }
    
    # Overall metrics
    profitable_trades = len(df[df['profitable']]) if 'profitable' in df.columns else 0
    overall_win_rate = (profitable_trades / total_trades) * 100 if total_trades > 0 else 0
    avg_trade_pnl = total_pnl / total_trades if total_trades > 0 else 0
    
    # Best and worst trades
    best_trade = df['pnl'].max() if 'pnl' in df.columns and len(df) > 0 else 0
    worst_trade = df['pnl'].min() if 'pnl' in df.columns and len(df) > 0 else 0
    
    # Max drawdown
    if 'cumulative_pnl' in df.columns and len(df) > 0:
        cumulative = df['cumulative_pnl']
        rolling_max = cumulative.expanding().max()
        drawdown = cumulative - rolling_max
        max_drawdown = drawdown.min()
    else:
        max_drawdown = 0
    
    # Sharpe ratio (simplified)
    if 'pnl' in df.columns and len(df) > 1 and df['pnl'].std() != 0:
        sharpe_ratio = df['pnl'].mean() / df['pnl'].std() * np.sqrt(len(df))
    else:
        sharpe_ratio = 0

    return {
        'total_trades': total_trades,
        'long_trades': long_trades,
        'short_trades': short_trades,
        'hold_periods': hold_periods,
        'overall_win_rate': overall_win_rate,
        'total_pnl': total_pnl,
        'avg_trade_pnl': avg_trade_pnl,
        'best_trade': best_trade,
        'worst_trade': worst_trade,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'directional_performance': directional_performance
    }


@st.cache_data(ttl=600)  # Cache for 10 minutes
def display_directional_ml_predictions(df):
    """🎯 Display directional ML predictions (LONG/SHORT/HOLD)"""
    st.header("🎯 Directional ML Predictions (LONG/SHORT/HOLD)")

    # Limit data for dashboard ML
    if len(df) > 1500:
        df_ml = df.tail(1500)
        st.info(f"📊 Using last 1500 of {len(df)} transactions for directional ML prediction")
    else:
        df_ml = df
        st.info(f"📊 Using all {len(df_ml)} transactions for directional ML.")

    try:
        # Try to load directional ML integration
        from ml.price_predictor import DirectionalMLTradingIntegration
        ml_integration = DirectionalMLTradingIntegration()
        
        st.success("✅ Directional ML Integration loaded successfully")

        # Check if we have enough data
        if len(df_ml) >= 75:  # Lower threshold for directional
            with st.spinner("🎯 Generating directional ML prediction..."):
                st.write(f"🎯 Generating LONG/SHORT/HOLD prediction for {len(df_ml)} transactions...")

                # Check required columns
                required_cols = ['price', 'rsi']
                missing_cols = [col for col in required_cols if col not in df_ml.columns]

                if missing_cols:
                    st.error(f"❌ Missing columns for directional ML: {missing_cols}")
                    st.write(f"📋 Available columns: {list(df_ml.columns)}")
                    return
                else:
                    st.success(f"✅ All required columns present: {required_cols}")

                try:
                    # Generate directional prediction
                    prediction = ml_integration.get_directional_prediction(df_ml)
                    
                    st.write(f"🎯 Directional prediction generated successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error generating directional prediction: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                    return

            # Display directional prediction results
            if prediction and 'predicted_direction' in prediction:
                st.success("🎉 Directional ML Prediction generated successfully!")

                # 🎯 MAIN DIRECTIONAL PREDICTION METRICS
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    # Show directional prediction
                    predicted_direction = prediction['predicted_direction']
                    direction_probs = prediction.get('direction_probabilities', {})
                    
                    direction_colors = {"long": "🟢", "short": "🔴", "hold": "⚪"}
                    direction_color = direction_colors.get(predicted_direction, "⚪")
                    
                    st.metric(
                        "🎯 Predicted Direction",
                        f"{direction_color} {predicted_direction.upper()}",
                        delta=f"{direction_probs.get(predicted_direction, 0):.1%} confidence"
                    )

                with col2:
                    # Show recommendation
                    recommendation = prediction.get('recommendation', 'HOLD')
                    confidence = prediction.get('confidence', 0)
                    
                    rec_colors = {"LONG": "🟢", "SHORT": "🔴", "HOLD": "⚪"}
                    rec_color = rec_colors.get(recommendation, "⚪")
                    
                    st.metric(
                        "📈 Recommendation",
                        f"{rec_color} {recommendation}",
                        delta=f"Confidence: {confidence:.2f}"
                    )

                with col3:
                    # Model info
                    model_count = prediction.get('model_count', 0)
                    model_agreement = prediction.get('model_agreement', 0)
                    
                    st.metric(
                        "🤖 Model Info",
                        f"{model_count} models",
                        delta=f"{model_agreement:.1%} agreement"
                    )

                with col4:
                    # Direction probabilities summary
                    long_prob = direction_probs.get('long', 0)
                    short_prob = direction_probs.get('short', 0)
                    hold_prob = direction_probs.get('hold', 0)
                    
                    st.metric(
                        "🎯 Direction Breakdown",
                        f"L:{long_prob:.1%} S:{short_prob:.1%}",
                        delta=f"H:{hold_prob:.1%}"
                    )

                # 🎯 DIRECTIONAL PROBABILITIES VISUALIZATION
                st.subheader("🎯 Direction Probability Breakdown")
                
                direction_data = pd.DataFrame({
                    'Direction': ['LONG 🟢', 'SHORT 🔴', 'HOLD ⚪'],
                    'Probability': [
                        direction_probs.get('long', 0),
                        direction_probs.get('short', 0), 
                        direction_probs.get('hold', 0)
                    ],
                    'Color': ['green', 'red', 'gray']
                })
                
                fig_direction = px.bar(
                    direction_data, 
                    x='Direction', 
                    y='Probability',
                    color='Color',
                    color_discrete_map={'green': '#00ff00', 'red': '#ff0000', 'gray': '#888888'},
                    title="🎯 Directional Trading Probability Distribution"
                )
                fig_direction.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_direction, use_container_width=True)

                # 🎯 INDIVIDUAL MODEL PREDICTIONS
                if 'individual_predictions' in prediction:
                    st.subheader("🔍 Individual Model Predictions")

                    model_data = []
                    for model_name, pred_info in prediction['individual_predictions'].items():
                        direction = pred_info.get('direction', 'unknown')
                        probabilities = pred_info.get('probabilities', {})
                        
                        model_data.append({
                            'Model': model_name.replace('_', ' ').title(),
                            'Predicted Direction': f"{direction_colors.get(direction, '❓')} {direction.upper()}",
                            'LONG Prob': f"{probabilities.get('long', 0):.1%}",
                            'SHORT Prob': f"{probabilities.get('short', 0):.1%}",
                            'HOLD Prob': f"{probabilities.get('hold', 0):.1%}",
                            'Confidence': f"{max(probabilities.values()) if probabilities else 0:.1%}"
                        })

                    if model_data:
                        st.dataframe(pd.DataFrame(model_data), use_container_width=True)

                # 🎯 ENHANCED METRICS DISPLAY
                if 'enhanced_metrics' in prediction:
                    st.subheader("📊 Enhanced Directional Metrics")
                    
                    enhanced = prediction['enhanced_metrics']
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Base Confidence", f"{enhanced.get('base_confidence', 0):.3f}")
                    with col2:
                        st.metric("Agreement Bonus", f"{enhanced.get('agreement_bonus', 0):.3f}")
                    with col3:
                        st.metric("Available Features", f"{enhanced.get('available_features', 0)}")
                    with col4:
                        st.metric("Missing Features", f"{enhanced.get('missing_features', 0)}")

                # 🎯 DIRECTIONAL CONFIDENCE GAUGE
                st.subheader("📈 Directional Confidence Gauge")
                
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = confidence * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': f"Directional ML Confidence - {predicted_direction.upper()}"},
                    delta = {'reference': 50},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': {"long": "green", "short": "red", "hold": "gray"}.get(predicted_direction, "blue")},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgray"},
                            {'range': [30, 60], 'color': "yellow"},
                            {'range': [60, 80], 'color': "orange"},
                            {'range': [80, 100], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 70
                        }
                    }
                ))

                fig_gauge.update_layout(height=400)
                st.plotly_chart(fig_gauge, use_container_width=True)

                # ML Model Performance
                try:
                    st.subheader("📊 Directional Model Performance")
                    performance = ml_integration.get_model_performance()
                    if performance:
                        perf_data = []
                        for model_name, metrics in performance.items():
                            perf_data.append({
                                'Model': model_name.replace('_', ' ').title(),
                                'Accuracy': f"{metrics.get('accuracy', 0):.1f}%",
                                'Type': metrics.get('model_type', 'directional_classification'),
                                'Training Samples': f"{metrics.get('training_samples', 0):,}",
                                'Last Trained': metrics.get('last_trained', 'Never'),
                                'LONG Precision': f"{metrics.get('long_precision', 0):.2f}",
                                'SHORT Precision': f"{metrics.get('short_precision', 0):.2f}",
                                'HOLD Precision': f"{metrics.get('hold_precision', 0):.2f}"
                            })

                        if perf_data:
                            st.dataframe(pd.DataFrame(perf_data), use_container_width=True)
                        else:
                            st.info("📊 Directional models not yet trained")
                    else:
                        st.info("📊 No directional model performance data available")
                except Exception as e:
                    st.warning(f"⚠️ Cannot retrieve directional model performance: {e}")

            elif prediction and 'error' in prediction:
                st.error(f"❌ Directional ML Error: {prediction['error']}")
            else:
                st.error("❌ Failed to generate directional prediction")
                st.write(f"🔍 Response received: {prediction}")
                
        else:
            st.info(f"📊 Need more data for directional ML predictions (current: {len(df_ml)}/75 transactions)")
            
            # Show progress bar
            progress = min(len(df_ml) / 75, 1.0)
            st.progress(progress)
            st.caption(f"Progress to first directional ML prediction: {len(df_ml)}/75 transactions")

    except ImportError as e:
        st.error(f"❌ Directional ML modules not available: {e}")
        st.info("💡 To enable directional ML predictions, ensure the system has: scikit-learn, xgboost, lightgbm")
    except Exception as e:
        st.error(f"❌ Directional ML error: {e}")
        import traceback
        st.code(traceback.format_exc())


def display_directional_trading_performance(df, metrics):
    """🎯 Display comprehensive directional trading performance"""
    st.header("🎯 Directional Trading Performance")

    # 🎯 MAIN DIRECTIONAL METRICS
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Trades",
            f"{metrics['total_trades']:,}",
            delta=None
        )

    with col2:
        total_pnl = metrics['total_pnl']
        pnl_color = "normal" if total_pnl >= 0 else "inverse"
        st.metric(
            "Total P&L",
            f"${total_pnl:.6f}",
            delta=f"${metrics['avg_trade_pnl']:.6f} avg"
        )

    with col3:
        st.metric(
            "Overall Win Rate",
            f"{metrics['overall_win_rate']:.1f}%",
            delta=None
        )

    with col4:
        if len(df) > 0:
            last_trade_time = df['timestamp'].max()
            try:
                time_since = datetime.now() - last_trade_time.to_pydatetime().replace(tzinfo=None)
                minutes_ago = time_since.seconds // 60
            except Exception:
                minutes_ago = 0
            st.metric(
                "Last Trade",
                f"{minutes_ago}min ago",
                delta=None
            )

    # 🎯 DIRECTIONAL BREAKDOWN
    st.subheader("🎯 Trading Direction Breakdown")
    
    dir_col1, dir_col2, dir_col3 = st.columns(3)
    
    directional_perf = metrics.get('directional_performance', {})
    
    with dir_col1:
        long_perf = directional_perf.get('long', {})
        long_trades = long_perf.get('trades', 0)
        long_win_rate = long_perf.get('win_rate', 0) * 100
        long_pnl = long_perf.get('total_pnl', 0)
        
        st.metric(
            "🟢 LONG Trades",
            f"{long_trades}",
            delta=f"{long_win_rate:.1f}% win rate"
        )
        st.write(f"P&L: ${long_pnl:.6f}")
        
        if long_trades > 0:
            avg_duration = long_perf.get('avg_duration', 0) / 60  # Convert to minutes
            st.write(f"Avg Duration: {avg_duration:.1f}m")

    with dir_col2:
        short_perf = directional_perf.get('short', {})
        short_trades = short_perf.get('trades', 0)
        short_win_rate = short_perf.get('win_rate', 0) * 100
        short_pnl = short_perf.get('total_pnl', 0)
        
        st.metric(
            "🔴 SHORT Trades",
            f"{short_trades}",
            delta=f"{short_win_rate:.1f}% win rate"
        )
        st.write(f"P&L: ${short_pnl:.6f}")
        
        if short_trades > 0:
            avg_duration = short_perf.get('avg_duration', 0) / 60
            st.write(f"Avg Duration: {avg_duration:.1f}m")

    with dir_col3:
        hold_perf = directional_perf.get('hold', {})
        hold_periods = hold_perf.get('trades', 0)
        hold_win_rate = hold_perf.get('win_rate', 0) * 100
        hold_pnl = hold_perf.get('total_pnl', 0)
        
        st.metric(
            "⚪ HOLD Periods",
            f"{hold_periods}",
            delta=f"{hold_win_rate:.1f}% success"
        )
        st.write(f"P&L: ${hold_pnl:.6f}")
        
        if hold_periods > 0:
            avg_duration = hold_perf.get('avg_duration', 0) / 60
            st.write(f"Avg Duration: {avg_duration:.1f}m")

    # 🎯 SECONDARY METRICS
    col5, col6, col7, col8 = st.columns(4)

    with col5:
        st.metric(
            "Best Trade",
            f"${metrics['best_trade']:.6f}",
            delta=None
        )

    with col6:
        st.metric(
            "Worst Trade",
            f"${metrics['worst_trade']:.6f}",
            delta=None
        )

    with col7:
        st.metric(
            "Max Drawdown",
            f"${metrics['max_drawdown']:.6f}",
            delta=None
        )

    with col8:
        st.metric(
            "Sharpe Ratio",
            f"{metrics['sharpe_ratio']:.2f}",
            delta=None
        )


def main():
    # Auto-refresh functionality
    auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (90s)", value=False)

    if auto_refresh:
        if 'refresh_counter' not in st.session_state:
            st.session_state.refresh_counter = 90

        st.sidebar.write(f"⏱️ Refresh in: {st.session_state.refresh_counter}s")

        if st.session_state.refresh_counter <= 0:
            st.session_state.refresh_counter = 90
            st.rerun()
        else:
            st.session_state.refresh_counter -= 5
            time.sleep(5)
            st.rerun()

    # Header
    st.title("🎯 DexBot Directional Trading Dashboard")
    st.markdown("💰 **LONG/SHORT/HOLD Trading Analytics**")
    st.markdown("---")

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Directional Controls")

        # Status
        st.subheader("🎯 System Status")
        st.success("🟢 Directional Bot Active")
        st.info("📊 Mode: LONG/SHORT/HOLD Trading")

        # Manual refresh button
        if st.button("🔄 Refresh Data"):
            st.rerun()

        # File info
        if os.path.exists("data/memory.csv"):
            file_size = os.path.getsize("data/memory.csv")
            st.metric("Data File Size", f"{file_size / 1024:.1f} KB")

        # 🎯 DIRECTIONAL ML CONTROLS
        st.subheader("🎯 Directional ML Controls")
        show_ml = st.checkbox("Show Directional ML Predictions", value=False)

        if st.button("🔄 Retrain Directional Models"):
            st.info("🎯 Directional model retraining initiated...")

        # Directional Trading Info
        st.subheader("🎯 Trading Directions")
        st.info("🟢 LONG: Buy asset, profit on price rise")
        st.info("🔴 SHORT: Sell asset, profit on price fall")
        st.info("⚪ HOLD: Stay in USDC, wait for opportunity")

        # Performance thresholds
        st.subheader("📊 Performance Thresholds")
        st.metric("LONG Threshold", "65%")
        st.metric("SHORT Threshold", "65%")
        st.metric("HOLD Threshold", "50%")

    # Load directional trading data
    df = load_directional_trading_data()

    if df.empty:
        st.warning("⚠️ No directional trading data available")
        st.info("The directional trading bot may not have started yet or there's an issue with data access.")
        return

    # Calculate directional metrics
    metrics = calculate_directional_metrics(df)

    if not metrics:
        st.warning("⚠️ Cannot calculate directional metrics")
        return

    # Display directional trading performance
    display_directional_trading_performance(df, metrics)

    st.markdown("---")

    # 🎯 DIRECTIONAL ML PREDICTIONS
    if show_ml:
        display_directional_ml_predictions(df)
        st.markdown("---")

    # 🎯 MAIN CHART - Cumulative P&L with directional markers
    st.subheader("📈 Directional Trading P&L Over Time")

    try:
        fig_pnl = go.Figure()
        
        # Main P&L line
        fig_pnl.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['cumulative_pnl'],
                mode='lines',
                name='Cumulative P&L',
                line=dict(
                    color='blue',
                    width=2
                )
            )
        )
        
        # Add directional markers if available
        if 'direction' in df.columns:
            # LONG trades
            long_trades = df[df['direction'] == 'long']
            if len(long_trades) > 0:
                fig_pnl.add_trace(
                    go.Scatter(
                        x=long_trades['timestamp'],
                        y=long_trades['cumulative_pnl'],
                        mode='markers',
                        name='LONG Trades',
                        marker=dict(color='green', size=8, symbol='triangle-up')
                    )
                )
            
            # SHORT trades  
            short_trades = df[df['direction'] == 'short']
            if len(short_trades) > 0:
                fig_pnl.add_trace(
                    go.Scatter(
                        x=short_trades['timestamp'],
                        y=short_trades['cumulative_pnl'],
                        mode='markers',
                        name='SHORT Trades',
                        marker=dict(color='red', size=8, symbol='triangle-down')
                    )
                )
            
            # HOLD periods
            hold_periods = df[df['direction'] == 'hold']
            if len(hold_periods) > 0:
                fig_pnl.add_trace(
                    go.Scatter(
                        x=hold_periods['timestamp'],
                        y=hold_periods['cumulative_pnl'],
                        mode='markers',
                        name='HOLD Periods',
                        marker=dict(color='gray', size=6, symbol='circle')
                    )
                )

        fig_pnl.update_layout(
            title="🎯 Directional Trading Equity Curve",
            xaxis_title="Time",
            yaxis_title="Cumulative P&L ($)",
            height=500
        )

        st.plotly_chart(fig_pnl, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating directional P&L chart: {e}")

    # 🎯 DIRECTIONAL ANALYSIS TABS
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🎯 Directional Stats", "📋 Recent Trades", "📈 Time Analysis", "🏆 Performance"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            # Direction distribution pie chart
            try:
                if 'direction' in df.columns:
                    direction_counts = df['direction'].value_counts()
                    
                    direction_data = pd.DataFrame({
                        'Direction': [f"{d.upper()} {{'long': '🟢', 'short': '🔴', 'hold': '⚪'}[d]}" for d in direction_counts.index],
                        'Count': direction_counts.values
                    })

                    fig_pie = px.pie(
                        direction_data,
                        values='Count',
                        names='Direction',
                        title="🎯 Trading Direction Distribution",
                        color_discrete_map={
                            'LONG 🟢': 'green',
                            'SHORT 🔴': 'red', 
                            'HOLD ⚪': 'lightgray'
                        }
                    )
                    fig_pie.update_layout(height=400)
                    st.plotly_chart(fig_pie, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating direction pie chart: {e}")

        with col2:
            # P&L distribution by direction
            try:
                if 'direction' in df.columns and 'pnl' in df.columns:
                    fig_box = px.box(
                        df,
                        x='direction',
                        y='pnl',
                        title="🎯 P&L Distribution by Direction",
                        color='direction',
                        color_discrete_map={
                            'long': 'green',
                            'short': 'red',
                            'hold': 'gray'
                        }
                    )
                    fig_box.update_layout(height=400)
                    st.plotly_chart(fig_box, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating P&L distribution chart: {e}")

    with tab2:
        # Recent directional trades table
        st.subheader("🕒 Last 20 Directional Trades")
        try:
            if len(df) > 0:
                recent_cols = ['timestamp', 'direction', 'pnl', 'pnl_percentage', 'duration_seconds', 'profitable']
                available_cols = [col for col in recent_cols if col in df.columns]
                
                if available_cols:
                    recent_trades = df.tail(20)[available_cols].copy()
                    recent_trades['timestamp'] = recent_trades['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Add direction emojis
                    if 'direction' in recent_trades.columns:
                        direction_map = {'long': '🟢', 'short': '🔴', 'hold': '⚪'}
                        'Direction': [f"{d.upper()} {direction_map.get(d, '❓')}" for d in direction_counts.index],
                        recent_trades['direction'] = recent_trades['direction'].map(direction_map)
                    
                    # Format duration
                    if 'duration_seconds' in recent_trades.columns:
                        recent_trades['duration_minutes'] = (recent_trades['duration_seconds'] / 60).round(1)
                        recent_trades = recent_trades.drop('duration_seconds', axis=1)

                    # Style the dataframe
                    def color_row(row):
                        if row.get('profitable', False):
                            return ['background-color: rgba(0, 255, 0, 0.1)'] * len(row)
                        else:
                            return ['background-color: rgba(255, 0, 0, 0.1)'] * len(row)

                    styled_df = recent_trades.style.apply(color_row, axis=1)
                    st.dataframe(styled_df, use_container_width=True)

                    # Summary of recent trades
                    if 'pnl' in recent_trades.columns:
                        recent_pnl = recent_trades['pnl'].sum()
                        recent_wins = len(recent_trades[recent_trades.get('profitable', False)])

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Last 20 P&L", f"${recent_pnl:.6f}")
                        with col2:
                            st.metric("Wins in Last 20", f"{recent_wins}/20")
                else:
                    st.warning("No trade data columns available")
            else:
                st.warning("No trades to display")
        except Exception as e:
            st.error(f"Error displaying recent trades: {e}")

    with tab3:
        # Time-based directional analysis
        st.subheader("🕐 Directional Trading Activity Over Time")

        try:
            # Trades per hour with direction breakdown
            if 'hour' in df.columns and 'direction' in df.columns:
                hourly_direction = df.groupby(['hour', 'direction']).size().reset_index(name='count')
                
                fig_hourly = px.bar(
                    hourly_direction,
                    x='hour',
                    y='count',
                    color='direction',
                    title="🎯 Directional Trades by Hour",
                    color_discrete_map={
                        'long': 'green',
                        'short': 'red',
                        'hold': 'gray'
                    }
                )
                st.plotly_chart(fig_hourly, use_container_width=True)

            # Daily performance if we have multiple days
            if 'date' in df.columns and len(df['date'].unique()) > 1:
                daily_stats = df.groupby(['date', 'direction']).agg({
                    'pnl': ['sum', 'count', 'mean'],
                    'profitable': 'sum'
                }).round(6)

                daily_stats.columns = ['Daily_PnL', 'Trades_Count', 'Avg_PnL', 'Winning_Trades']
                daily_stats['Win_Rate'] = (daily_stats['Winning_Trades'] / daily_stats['Trades_Count'] * 100).round(1)
                daily_stats = daily_stats.reset_index()

                fig_daily = px.bar(
                    daily_stats,
                    x='date',
                    y='Daily_PnL',
                    color='direction',
                    title="📅 Daily P&L by Direction",
                    color_discrete_map={
                        'long': 'green',
                        'short': 'red',
                        'hold': 'gray'
                    }
                )
                st.plotly_chart(fig_daily, use_container_width=True)

                st.subheader("📊 Daily Directional Statistics")
                st.dataframe(daily_stats, use_container_width=True)
        except Exception as e:
            st.error(f"Error in time analysis: {e}")

    with tab4:
        # Performance comparison
        st.subheader("🏆 Directional Performance Comparison")
        
        try:
            directional_perf = metrics.get('directional_performance', {})
            
            perf_data = []
            for direction, perf in directional_perf.items():
                emoji = {'long': '🟢', 'short': '🔴', 'hold': '⚪'}[direction]
                perf_data.append({
                    'Direction': f"{emoji} {direction.upper()}",
                    'Total Trades': perf.get('trades', 0),
                    'Wins': perf.get('wins', 0),
                    'Win Rate (%)': f"{perf.get('win_rate', 0) * 100:.1f}%",
                    'Total P&L': f"${perf.get('total_pnl', 0):.6f}",
                    'Avg P&L': f"${perf.get('avg_pnl', 0):.6f}",
                    'Avg Duration (min)': f"{perf.get('avg_duration', 0) / 60:.1f}"
                })
            
            if perf_data:
                st.dataframe(pd.DataFrame(perf_data), use_container_width=True)
                
                # Performance visualization
                fig_perf = px.bar(
                    pd.DataFrame(perf_data),
                    x='Direction',
                    y=[float(x.replace('$', '').replace('%', '')) for x in pd.DataFrame(perf_data)['Win Rate (%)']],
                    title="🏆 Win Rate by Direction",
                    color='Direction'
                )
                st.plotly_chart(fig_perf, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error creating performance comparison: {e}")

    # Footer info
    st.markdown("---")
    st.info("🎯 Directional Trading Dashboard - LONG/SHORT/HOLD Analytics")

    # System info
    with st.expander("🔍 Directional System Info"):
        st.write("**Directional Trading Bot Status:**")
        st.write(f"- Total directional trades: {len(df):,}")
        st.write(f"- LONG trades: {metrics.get('long_trades', 0):,}")
        st.write(f"- SHORT trades: {metrics.get('short_trades', 0):,}")
        st.write(f"- HOLD periods: {metrics.get('hold_periods', 0):,}")
        st.write(f"- Last update: {df['timestamp'].max() if not df.empty else 'N/A'}")
        st.write(f"- Data file size: {os.path.getsize('data/memory.csv') / 1024:.1f} KB" if os.path.exists('data/memory.csv') else "- Data file not available")

        # Directional ML Status
        try:
            from ml.price_predictor import DirectionalMLTradingIntegration
            st.write("**Directional ML Status:** ✅ Available")
        except ImportError:
            st.write("**Directional ML Status:** ❌ Not Available")

        # Trading Direction Configuration
        st.write("**Direction Thresholds:**")
        st.write("- LONG threshold: 65%")
        st.write("- SHORT threshold: 65%")
        st.write("- HOLD threshold: 50%")


if __name__ == "__main__":
    main()