# core/multi_asset_data.py - Multi-Asset Market Data Manager
import websocket
import json
import threading
import time
import requests
import numpy as np
from datetime import datetime
from typing import Dict, List, Callable, Optional
import logging

from core.market_data import BinanceMarketData, TradingSignals


class MultiAssetMarketData:
    """
    Zarządza danymi rynkowymi dla wielu aktywów jednocześnie
    Rozszerzenie obecnego single-asset systemu
    """
    
    def __init__(self, on_data_update: Callable[[str, Dict], None]):
        self.on_data_update = on_data_update  # Callback function
        self.assets = {}
        self.latest_data = {}
        self.running = False
        
        # Obsługiwane aktywa
        self.supported_assets = {
            'SOL': 'SOLUSDC',
            'ETH': 'ETHUSDC', 
            'BTC': 'BTCUSDC'
        }
        
        self.logger = logging.getLogger(__name__)
        print("📊 MultiAssetMarketData initialized")

    def add_asset(self, asset_symbol: str):
        """Dodaj nowy asset do trackowania"""
        if asset_symbol not in self.supported_assets:
            print(f"❌ Asset {asset_symbol} not supported")
            return False
            
        if asset_symbol in self.assets:
            print(f"⚠️ Asset {asset_symbol} already tracked")
            return True
            
        try:
            # Stwórz callback dla konkretnego assetu
            def asset_callback(market_data):
                self._on_asset_update(asset_symbol, market_data)
            
            # Inicjalizuj market data dla assetu
            binance_symbol = self.supported_assets[asset_symbol]
            asset_stream = BinanceMarketData(asset_callback)
            
            # Uruchom stream
            if asset_stream.start_stream(binance_symbol.lower()):
                self.assets[asset_symbol] = asset_stream
                self.latest_data[asset_symbol] = {}
                print(f"✅ {asset_symbol} stream started ({binance_symbol})")
                return True
            else:
                print(f"❌ Failed to start {asset_symbol} stream")
                return False
                
        except Exception as e:
            print(f"❌ Error adding {asset_symbol}: {e}")
            return False

    def _on_asset_update(self, asset_symbol: str, market_data: Dict):
        """Callback dla aktualizacji danych konkretnego assetu"""
        try:
            # Dodaj symbol assetu do danych
            market_data['asset_symbol'] = asset_symbol
            
            # Zapisz najnowsze dane
            self.latest_data[asset_symbol] = market_data
            
            # Wywołaj główny callback
            self.on_data_update(asset_symbol, market_data)
            
        except Exception as e:
            print(f"❌ Error processing {asset_symbol} update: {e}")

    def start_tracking(self, assets: List[str]):
        """Uruchom trackowanie dla listy assetów"""
        print(f"🌐 Starting multi-asset tracking: {assets}")
        
        success_count = 0
        for asset in assets:
            if self.add_asset(asset):
                success_count += 1
                time.sleep(1)  # Delay między połączeniami
        
        if success_count > 0:
            self.running = True
            print(f"✅ Multi-asset tracking started: {success_count}/{len(assets)} assets")
            return True
        else:
            print("❌ Failed to start any asset streams")
            return False

    def stop_tracking(self):
        """Zatrzymaj wszystkie streamy"""
        print("🛑 Stopping multi-asset tracking...")
        
        for asset_symbol, stream in self.assets.items():
            try:
                stream.stop_stream()
                print(f"✅ {asset_symbol} stream stopped")
            except Exception as e:
                print(f"⚠️ Error stopping {asset_symbol}: {e}")
        
        self.assets.clear()
        self.latest_data.clear()
        self.running = False
        print("✅ Multi-asset tracking stopped")

    def get_asset_data(self, asset_symbol: str) -> Optional[Dict]:
        """Pobierz najnowsze dane dla konkretnego assetu"""
        return self.latest_data.get(asset_symbol)

    def get_all_data(self) -> Dict[str, Dict]:
        """Pobierz dane wszystkich assetów"""
        return self.latest_data.copy()

    def get_asset_price(self, asset_symbol: str) -> float:
        """Pobierz aktualną cenę assetu"""
        data = self.get_asset_data(asset_symbol)
        return data.get('price', 0.0) if data else 0.0

    def get_asset_rsi(self, asset_symbol: str) -> float:
        """Pobierz aktualny RSI assetu"""
        data = self.get_asset_data(asset_symbol)
        return data.get('rsi', 50.0) if data else 50.0

    def is_asset_connected(self, asset_symbol: str) -> bool:
        """Sprawdź czy asset jest połączony"""
        if asset_symbol not in self.assets:
            return False
        return self.assets[asset_symbol].running

    def get_connection_status(self) -> Dict[str, bool]:
        """Status połączenia wszystkich assetów"""
        status = {}
        for asset_symbol in self.assets:
            status[asset_symbol] = self.is_asset_connected(asset_symbol)
        return status

    def get_market_summary(self) -> Dict:
        """Podsumowanie rynku wszystkich assetów"""
        summary = {
            'total_assets': len(self.assets),
            'connected_assets': sum(1 for asset in self.assets if self.is_asset_connected(asset)),
            'assets': {}
        }
        
        for asset_symbol, data in self.latest_data.items():
            if data:
                summary['assets'][asset_symbol] = {
                    'price': data.get('price', 0),
                    'price_change_24h': data.get('price_change_24h', 0),
                    'rsi': data.get('rsi', 50),
                    'volume_24h': data.get('volume_24h', 0),
                    'last_update': data.get('timestamp')
                }
        
        return summary


class MultiAssetSignals:
    """
    Generuje sygnały tradingowe dla wielu assetów
    Uwzględnia korelacje między assetami
    """
    
    def __init__(self):
        self.trading_signals = TradingSignals()
        
    def analyze_multi_asset_conditions(self, all_market_data: Dict[str, Dict]) -> Dict[str, Dict]:
        """Analizuj warunki dla wszystkich assetów z uwzględnieniem korelacji"""
        signals = {}
        
        for asset_symbol, market_data in all_market_data.items():
            if not market_data:
                continue
                
            # Podstawowa analiza dla assetu
            asset_signals = self.trading_signals.analyze_market_conditions(market_data)
            
            # Dodaj informacje o asset
            asset_signals['asset_symbol'] = asset_symbol
            asset_signals['price'] = market_data.get('price', 0)
            asset_signals['rsi'] = market_data.get('rsi', 50)
            
            signals[asset_symbol] = asset_signals
        
        # Analiza korelacji (prosty przykład)
        signals = self._apply_correlation_analysis(signals, all_market_data)
        
        return signals
    
    def _apply_correlation_analysis(self, signals: Dict, all_market_data: Dict) -> Dict:
        """Zastosuj analizę korelacji do sygnałów"""
        # Prosty przykład: jeśli wszystkie assety mają ten sam trend, zmniejsz confidence
        trends = [signals[asset].get('trend', 'neutral') for asset in signals]
        
        if len(set(trends)) == 1 and len(trends) > 1:
            # Wszystkie assety mają ten sam trend - możliwa korelacja
            for asset in signals:
                current_confidence = signals[asset].get('confidence', 0.5)
                signals[asset]['confidence'] = current_confidence * 0.8
                signals[asset]['reasons'].append('High correlation detected - reduced confidence')
        
        return signals

    def get_best_asset_to_trade(self, signals: Dict[str, Dict]) -> Optional[str]:
        """Wybierz najlepszy asset do tradowania"""
        if not signals:
            return None
            
        # Sortuj assety według confidence i strength
        asset_scores = []
        for asset_symbol, signal in signals.items():
            confidence = signal.get('confidence', 0)
            strength = abs(signal.get('strength', 0))
            score = confidence * strength
            
            asset_scores.append((asset_symbol, score, signal))
        
        # Sortuj malejąco według score
        asset_scores.sort(key=lambda x: x[1], reverse=True)
        
        if asset_scores and asset_scores[0][1] > 0.3:  # Minimum threshold
            best_asset = asset_scores[0][0]
            print(f"🎯 Best asset to trade: {best_asset} (score: {asset_scores[0][1]:.2f})")
            return best_asset
        
        return None


# Helper function dla łatwego użycia
def create_multi_asset_service(assets: List[str], callback_function: Callable[[str, Dict], None]):
    """Stwórz i uruchom multi-asset market data service"""
    multi_service = MultiAssetMarketData(callback_function)
    
    if multi_service.start_tracking(assets):
        print(f"✅ Multi-asset service uruchomiony dla: {assets}")
        return multi_service
    else:
        print(f"❌ Nie udało się uruchomić multi-asset service")
        return None


# Przykład użycia
if __name__ == "__main__":
    def example_callback(asset_symbol: str, market_data: Dict):
        price = market_data.get('price', 0)
        rsi = market_data.get('rsi', 0)
        print(f"📊 {asset_symbol}: ${price:.2f}, RSI: {rsi:.1f}")
    
    # Test multi-asset
    service = create_multi_asset_service(['SOL', 'ETH'], example_callback)
    
    if service:
        try:
            time.sleep(30)  # Run for 30 seconds
            print(f"📈 Market Summary:")
            summary = service.get_market_summary()
            for asset, data in summary['assets'].items():
                print(f"  {asset}: ${data['price']:.2f} ({data['price_change_24h']:+.1f}%)")
                
        except KeyboardInterrupt:
            print("🛑 Stopping test...")
        finally:
            service.stop_tracking()