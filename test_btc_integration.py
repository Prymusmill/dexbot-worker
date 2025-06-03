# test_btc_integration.py - Test 3-asset system
import sys
import time
from datetime import datetime

try:
    from core.multi_asset_data import create_multi_asset_service, MultiAssetSignals
    print("✅ Multi-asset modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def on_asset_update(asset_symbol: str, market_data: dict):
    """Callback function for asset updates"""
    try:
        price = market_data.get('price', 0)
        rsi = market_data.get('rsi', 50)
        price_change_24h = market_data.get('price_change_24h', 0)
        
        change_icon = "📈" if price_change_24h > 0 else "📉" if price_change_24h < 0 else "➡️"
        
        print(f"📊 {asset_symbol}: ${price:.2f}, RSI: {rsi:.1f}, "
              f"24h: {change_icon} {price_change_24h:+.2f}%")
              
    except Exception as e:
        print(f"❌ Error processing {asset_symbol} update: {e}")

def test_three_assets():
    """Test SOL + ETH + BTC integration"""
    print("🧪 TEST: 3-Asset Integration (SOL + ETH + BTC)")
    print("=" * 60)
    
    service = create_multi_asset_service(['SOL', 'ETH', 'BTC'], on_asset_update)
    
    if not service:
        print("❌ Failed to create 3-asset service")
        return False
        
    try:
        print("⏱️ Running 3-asset test for 60 seconds...")
        
        # Monitor for 60 seconds
        for i in range(6):  # 6 x 10 seconds = 60 seconds
            time.sleep(10)
            
            print(f"\n📊 Status Update ({(i+1)*10}s):")
            summary = service.get_market_summary()
            
            print(f"   Connected: {summary['connected_assets']}/{summary['total_assets']} assets")
            
            for asset, data in summary['assets'].items():
                if data:
                    price = data['price']
                    rsi = data['rsi']
                    change = data['price_change_24h']
                    
                    # RSI status
                    rsi_status = ""
                    if rsi > 70:
                        rsi_status = " 📈 Overbought"
                    elif rsi < 30:
                        rsi_status = " 📉 Oversold"
                    
                    print(f"   {asset}: ${price:.2f}, RSI: {rsi:.1f}{rsi_status}, 24h: {change:+.1f}%")
                else:
                    print(f"   {asset}: No data")
        
        # Final portfolio analysis
        print(f"\n🎯 PORTFOLIO ANALYSIS:")
        signals_analyzer = MultiAssetSignals()
        all_data = service.get_all_data()
        
        if len(all_data) >= 3:
            signals = signals_analyzer.analyze_multi_asset_conditions(all_data)
            best_asset = signals_analyzer.get_best_asset_to_trade(signals)
            
            print(f"   Best trading asset: {best_asset or 'None'}")
            
            for asset, signal in signals.items():
                action = signal.get('action', 'hold')
                confidence = signal.get('confidence', 0)
                print(f"   {asset}: {action.upper()} (confidence: {confidence:.2f})")
        
        return True
        
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        return True
        
    finally:
        service.stop_tracking()
        print("✅ 3-asset test completed")

if __name__ == "__main__":
    try:
        print("🚀 3-ASSET INTEGRATION TEST")
        print("=" * 60)
        print(f"⏰ Start time: {datetime.now()}")
        
        result = test_three_assets()
        
        if result:
            print("\n🎉 3-ASSET INTEGRATION SUCCESS!")
            print("✅ SOL + ETH + BTC ready for deployment")
        else:
            print("\n❌ 3-asset integration failed")
            
    except KeyboardInterrupt:
        print("\n🛑 Testing stopped by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()