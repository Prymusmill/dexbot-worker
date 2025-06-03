# quick_analysis.py - Railway PostgreSQL Analysis
import os

def run_quick_analysis():
    """Run analysis using existing database connection"""
    try:
        # Use existing database manager
        from database.db_manager import get_db_manager
        db = get_db_manager()
        print("✅ Connected to database")
        
        # Query 1: Overall 24h performance
        print("\n📊 OVERALL PERFORMANCE (24h):")
        cursor = db.connection.cursor()
        
        cursor.execute("""
        SELECT 
            COUNT(*) as total_trades,
            AVG(CASE WHEN amount_out > amount_in THEN 1.0 ELSE 0.0 END) as win_rate,
            AVG(amount_out - amount_in) as avg_pnl
        FROM transactions 
        WHERE timestamp >= NOW() - INTERVAL '24 hours';
        """)
        
        result = cursor.fetchone()
        print(f"Total Trades: {result[0]}")
        print(f"Win Rate: {result[1]:.1%}" if result[1] else "Win Rate: N/A")
        print(f"Avg P&L: ${result[2]:.6f}" if result[2] else "Avg P&L: N/A")
        
        # Query 2: RSI performance
        print("\n📊 RSI EXTREME PERFORMANCE:")
        cursor.execute("""
        SELECT 
            CASE 
                WHEN rsi < 20 THEN 'Oversold'
                WHEN rsi > 80 THEN 'Overbought'  
                ELSE 'Normal'
            END as condition,
            COUNT(*) as trades,
            AVG(CASE WHEN amount_out > amount_in THEN 1.0 ELSE 0.0 END) as win_rate,
            AVG(amount_out - amount_in) as avg_pnl
        FROM transactions 
        WHERE timestamp >= NOW() - INTERVAL '24 hours'
        AND rsi IS NOT NULL
        GROUP BY CASE 
            WHEN rsi < 20 THEN 'Oversold'
            WHEN rsi > 80 THEN 'Overbought'  
            ELSE 'Normal'
        END;
        """)
        
        results = cursor.fetchall()
        for row in results:
            condition, trades, win_rate, avg_pnl = row
            print(f"{condition}: {trades} trades, {win_rate:.1%} win rate, ${avg_pnl:.6f} avg P&L")
        
        # Query 3: Best RSI ranges
        print("\n📊 RSI RANGE ANALYSIS:")
        cursor.execute("""
        SELECT 
            FLOOR(rsi/10)*10 as rsi_range,
            COUNT(*) as trades,
            AVG(CASE WHEN amount_out > amount_in THEN 1.0 ELSE 0.0 END) as win_rate
        FROM transactions 
        WHERE timestamp >= NOW() - INTERVAL '24 hours'
        AND rsi IS NOT NULL
        GROUP BY FLOOR(rsi/10)*10
        ORDER BY rsi_range;
        """)
        
        results = cursor.fetchall()
        for row in results:
            rsi_range, trades, win_rate = row
            range_label = f"{int(rsi_range)}-{int(rsi_range)+9}"
            print(f"RSI {range_label}: {trades} trades, {win_rate:.1%} win rate")
        
        # Query 4: Recent optimization impact
        print("\n🔧 RECENT OPTIMIZATION IMPACT (6h):")
        cursor.execute("""
        SELECT 
            COUNT(*) as recent_trades,
            AVG(CASE WHEN amount_out > amount_in THEN 1.0 ELSE 0.0 END) as recent_win_rate,
            SUM(CASE WHEN rsi < 20 OR rsi > 80 THEN 1 ELSE 0 END) as extreme_rsi_trades
        FROM transactions 
        WHERE timestamp >= NOW() - INTERVAL '6 hours'
        AND rsi IS NOT NULL;
        """)
        
        result = cursor.fetchone()
        if result[0] > 0:
            print(f"Recent trades: {result[0]}")
            print(f"Recent win rate: {result[1]:.1%}")
            print(f"Extreme RSI trades: {result[2]} ({result[2]/result[0]:.1%})")
        
        cursor.close()
        print("\n✅ Analysis complete!")
        
        # Recommendations based on data
        print("\n💡 OPTIMIZATION RECOMMENDATIONS:")
        print("1. Monitor if extreme RSI trades (< 20, > 80) have better win rates")
        print("2. Check if recent optimization (0.7→0.5 threshold) improved performance")
        print("3. Consider adjusting position sizes for high-confidence scenarios")
        
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🔍 QUICK CONTRARIAN ANALYSIS")
    print("=" * 50)
    run_quick_analysis()