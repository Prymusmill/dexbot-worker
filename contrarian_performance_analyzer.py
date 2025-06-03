# contrarian_performance_analyzer.py - Analyze contrarian trade performance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Connect to database
try:
    from database.db_manager import get_db_manager
    db = get_db_manager()
    print("✅ Connected to database")
except Exception as e:
    print(f"❌ Database connection failed: {e}")
    exit(1)

def analyze_contrarian_performance():
    """Analyze which contrarian conditions work best"""
    print("🔍 CONTRARIAN PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    # Get recent transactions with market data
    query = """
    SELECT timestamp, amount_in, amount_out, price, rsi, 
           (amount_out > amount_in) as profitable,
           price_impact, volume
    FROM transactions 
    WHERE timestamp >= NOW() - INTERVAL '24 hours'
    AND rsi IS NOT NULL 
    AND price IS NOT NULL
    ORDER BY timestamp DESC;
    """
    
    try:
        df = pd.read_sql_query(query, db.connection)
        print(f"📊 Analyzing {len(df)} recent transactions")
    except Exception as e:
        print(f"❌ Query failed: {e}")
        return
    
    if len(df) == 0:
        print("⚠️ No data found")
        return
    
    # Calculate profit/loss
    df['pnl'] = df['amount_out'] - df['amount_in']
    df['pnl_pct'] = (df['amount_out'] - df['amount_in']) / df['amount_in'] * 100
    
    # Identify contrarian trades (extreme RSI conditions)
    df['is_contrarian'] = ((df['rsi'] < 20) | (df['rsi'] > 80))
    df['rsi_extreme'] = df['rsi'].apply(lambda x: 
        'oversold' if x < 20 else 'overbought' if x > 80 else 'normal')
    
    # ANALYSIS 1: Contrarian vs Normal Performance
    print("\n📊 CONTRARIAN vs NORMAL PERFORMANCE:")
    contrarian_trades = df[df['is_contrarian']]
    normal_trades = df[~df['is_contrarian']]
    
    if len(contrarian_trades) > 0:
        contrarian_win_rate = contrarian_trades['profitable'].mean()
        contrarian_avg_pnl = contrarian_trades['pnl_pct'].mean()
        print(f"🔄 Contrarian trades: {len(contrarian_trades)}")
        print(f"   Win rate: {contrarian_win_rate:.1%}")
        print(f"   Avg PnL: {contrarian_avg_pnl:.3f}%")
    
    if len(normal_trades) > 0:
        normal_win_rate = normal_trades['profitable'].mean()
        normal_avg_pnl = normal_trades['pnl_pct'].mean()
        print(f"📈 Normal trades: {len(normal_trades)}")
        print(f"   Win rate: {normal_win_rate:.1%}")
        print(f"   Avg PnL: {normal_avg_pnl:.3f}%")
    
    # ANALYSIS 2: RSI Threshold Performance
    print("\n📊 RSI THRESHOLD ANALYSIS:")
    rsi_ranges = [
        (0, 10, 'Extreme Oversold'),
        (10, 20, 'Very Oversold'), 
        (20, 30, 'Oversold'),
        (70, 80, 'Overbought'),
        (80, 90, 'Very Overbought'),
        (90, 100, 'Extreme Overbought')
    ]
    
    for min_rsi, max_rsi, label in rsi_ranges:
        range_trades = df[(df['rsi'] >= min_rsi) & (df['rsi'] < max_rsi)]
        if len(range_trades) > 0:
            win_rate = range_trades['profitable'].mean()
            avg_pnl = range_trades['pnl_pct'].mean()
            print(f"{label} (RSI {min_rsi}-{max_rsi}): {len(range_trades)} trades, "
                  f"{win_rate:.1%} win rate, {avg_pnl:.3f}% avg PnL")
    
    # ANALYSIS 3: Best and Worst Conditions
    print("\n🎯 BEST CONTRARIAN CONDITIONS:")
    best_conditions = contrarian_trades.nlargest(10, 'pnl_pct')
    for _, trade in best_conditions.iterrows():
        print(f"   RSI: {trade['rsi']:.1f}, PnL: {trade['pnl_pct']:.3f}%, "
              f"Price: ${trade['price']:.2f}")
    
    print("\n💥 WORST CONTRARIAN CONDITIONS:")
    worst_conditions = contrarian_trades.nsmallest(10, 'pnl_pct')
    for _, trade in worst_conditions.iterrows():
        print(f"   RSI: {trade['rsi']:.1f}, PnL: {trade['pnl_pct']:.3f}%, "
              f"Price: ${trade['price']:.2f}")
    
    # ANALYSIS 4: Recommendations
    print("\n💡 OPTIMIZATION RECOMMENDATIONS:")
    
    if len(contrarian_trades) > 0:
        # Find optimal RSI thresholds
        oversold_trades = contrarian_trades[contrarian_trades['rsi'] < 50]
        overbought_trades = contrarian_trades[contrarian_trades['rsi'] > 50]
        
        if len(oversold_trades) > 0:
            best_oversold_threshold = oversold_trades[oversold_trades['profitable']]['rsi'].max()
            print(f"1. 📉 Optimal oversold threshold: RSI < {best_oversold_threshold:.0f}")
        
        if len(overbought_trades) > 0:
            best_overbought_threshold = overbought_trades[overbought_trades['profitable']]['rsi'].min()
            print(f"2. 📈 Optimal overbought threshold: RSI > {best_overbought_threshold:.0f}")
        
        # Volume analysis
        high_volume = contrarian_trades[contrarian_trades['volume'] > contrarian_trades['volume'].median()]
        if len(high_volume) > 0:
            high_vol_win_rate = high_volume['profitable'].mean()
            low_vol_win_rate = contrarian_trades[contrarian_trades['volume'] <= contrarian_trades['volume'].median()]['profitable'].mean()
            print(f"3. 📊 High volume contrarian win rate: {high_vol_win_rate:.1%}")
            print(f"   Low volume contrarian win rate: {low_vol_win_rate:.1%}")
            
        # Price impact analysis
        if 'price_impact' in df.columns:
            low_impact = contrarian_trades[abs(contrarian_trades['price_impact']) < 0.01]
            if len(low_impact) > 0:
                low_impact_win_rate = low_impact['profitable'].mean()
                print(f"4. 🎯 Low price impact (<1%) win rate: {low_impact_win_rate:.1%}")
    
    return df

if __name__ == "__main__":
    try:
        df = analyze_contrarian_performance()
        print(f"\n✅ Analysis complete! Found patterns to optimize contrarian strategy.")
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()