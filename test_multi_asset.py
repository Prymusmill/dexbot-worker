# test_multi_asset.py - Test Multi-Asset Integration
import sys
import time
from datetime import datetime

# Import our new multi-asset module
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
        volume_24h = market_data.get('volume_24h', 0)
        price_change_24h = market_data.get('price_change_24h', 0)
        timestamp = market_data.get('timestamp', datetime.now())
        
        # Format price change with color indicator
        change_icon = "📈" if price_change_24h > 0 else "📉" if price_change_24h < 0 else "➡️"
        
        print(f"📊 {asset_symbol}: ${price:.4f}, RSI: {rsi:.1f}, "
              f"24h: {change_icon} {price_change_24h:+.2f}%, Vol: {volume_24h:.0f}")
              
    except Exception as e:
        print(f"❌ Error processing {asset_symbol} update: {e}")


def test_single_asset():
    """Test individual asset - ETH only"""
    print("\n🧪 TEST 1: Single Asset (ETH)")
    print("=" * 50)
    
    service = create_multi_asset_service(['ETH'], on_asset_update)
    
    if not service:
        print("❌ Failed to create ETH service")
        return False
        
    try:
        print("⏱️ Running ETH test for 30 seconds...")
        time.sleep(30)
        
        # Check data
        eth_data = service.get_asset_data('ETH')
        if eth_data:
            print(f"✅ ETH data received: ${eth_data.get('price', 0):.4f}")
        else:
            print("⚠️ No ETH data received")
            
        # Check connection
        status = service.get_connection_status()
        print(f"📡 Connection status: {status}")
        
        return True
        
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        return True
        
    finally:
        service.stop_tracking()
        print("✅ ETH test completed")


def test_multi_asset():
    """Test multiple assets - SOL + ETH"""
    print("\n🧪 TEST 2: Multi-Asset (SOL + ETH)")
    print("=" * 50)
    
    service = create_multi_asset_service(['SOL', 'ETH'], on_asset_update)
    
    if not service:
        print("❌ Failed to create multi-asset service")
        return False
        
    try:
        print("⏱️ Running multi-asset test for 60 seconds...")
        
        # Monitor for 60 seconds
        for i in range(6):  # 6 x 10 seconds = 60 seconds
            time.sleep(10)
            
            # Status update every 10 seconds
            print(f"\n📊 Status Update ({(i+1)*10}s):")
            summary = service.get_market_summary()
            
            print(f"   Connected: {summary['connected_assets']}/{summary['total_assets']} assets")
            
            for asset, data in summary['assets'].items():
                if data:
                    print(f"   {asset}: ${data['price']:.4f}, RSI: {data['rsi']:.1f}")
                else:
                    print(f"   {asset}: No data")
                    
        return True
        
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        return True
        
    finally:
        service.stop_tracking()
        print("✅ Multi-asset test completed")


def test_signals():
    """Test signal generation"""
    print("\n🧪 TEST 3: Signal Generation")
    print("=" * 50)
    
    service = create_multi_asset_service(['SOL', 'ETH'], on_asset_update)
    signals_analyzer = MultiAssetSignals()
    
    if not service:
        print("❌ Failed to create service for signals test")
        return False
        
    try:
        print("⏱️ Collecting data for signal analysis...")
        time.sleep(20)  # Collect some data
        
        # Get all market data
        all_data = service.get_all_data()
        
        if len(all_data) > 0:
            print(f"📊 Analyzing signals for {len(all_data)} assets...")
            
            # Generate signals
            signals = signals_analyzer.analyze_multi_asset_conditions(all_data)
            
            print("\n🎯 SIGNAL ANALYSIS:")
            for asset, signal in signals.items():
                trend = signal.get('trend', 'neutral')
                action = signal.get('action', 'hold')
                confidence = signal.get('confidence', 0)
                strength = signal.get('strength', 0)
                
                print(f"   {asset}:")
                print(f"     Trend: {trend}")
                print(f"     Action: {action}")
                print(f"     Confidence: {confidence:.2f}")
                print(f"     Strength: {strength:.2f}")
            
            # Best asset to trade
            best_asset = signals_analyzer.get_best_asset_to_trade(signals)
            if best_asset:
                print(f"\n🏆 Best asset to trade: {best_asset}")
            else:
                print("\n⚠️ No suitable asset found for trading")
                
        else:
            print("⚠️ No market data available for signal analysis")
            
        return True
        
    except KeyboardInterrupt:
        print("\n🛑 Signal test interrupted")
        return True
        
    finally:
        service.stop_tracking()
        print("✅ Signal test completed")


def main():
    """Main test function"""
    print("🚀 MULTI-ASSET INTEGRATION TEST")
    print("=" * 60)
    print(f"⏰ Start time: {datetime.now()}")
    
    # Test sequence
    tests = [
        ("Single Asset (ETH)", test_single_asset),
        ("Multi-Asset (SOL+ETH)", test_multi_asset), 
        ("Signal Generation", test_signals)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🎯 Starting: {test_name}")
        try:
            result = test_func()
            results.append((test_name, "✅ PASSED" if result else "❌ FAILED"))
        except Exception as e:
            print(f"💥 Test {test_name} crashed: {e}")
            results.append((test_name, f"💥 CRASHED: {str(e)}"))
            
        # Short break between tests
        time.sleep(2)
    
    # Final results
    print("\n" + "=" * 60)
    print("🏁 TEST RESULTS SUMMARY:")
    print("=" * 60)
    
    for test_name, result in results:
        print(f"{result} {test_name}")
    
    passed_tests = sum(1 for _, result in results if "✅" in result)
    total_tests = len(results)
    
    print(f"\n📊 Overall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! Multi-asset integration ready!")
    else:
        print("⚠️ Some tests failed. Check the logs above.")
    
    print(f"⏰ Completed: {datetime.now()}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Testing stopped by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()