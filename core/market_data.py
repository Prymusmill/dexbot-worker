# core/market_data.py - Complete version with enhanced technical indicators
import websocket
import json
import threading
import time
import requests
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Callable, Optional
import logging

class BinanceMarketData:
    def __init__(self, on_price_update: Callable[[Dict], None]):
        self.on_price_update = on_price_update
        self.ws = None
        self.running = False
        self.current_price = 0.0
        self.price_history = []
        self.volume_24h = 0.0
        self.price_change_24h = 0.0
        self.bid_price = 0.0
        self.ask_price = 0.0
        self.last_update = None
        
        # Technical indicators storage
        self.sma_20 = 0.0
        self.sma_50 = 0.0
        self.rsi = 50.0
        self.volatility = 0.0
        
        # Enhanced technical analysis components
        self.data_manager = None
        self.signal_generator = None
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize technical analysis components after import
        self._init_technical_components()
        
    def _init_technical_components(self):
        """Initialize technical analysis components"""
        try:
            from core.technical_indicators import MarketDataManager, SignalGenerator
            self.data_manager = MarketDataManager()
            self.signal_generator = SignalGenerator()
            print("✅ Technical analysis components initialized")
        except ImportError:
            print("⚠️ Technical indicators not available - using basic analysis")
            self.data_manager = None
            self.signal_generator = None
        
    def get_historical_data(self, symbol: str = "SOLUSDC", interval: str = "1m", limit: int = 100):
        """Pobierz dane historyczne z Binance REST API"""
        try:
            url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Konwertuj na format [timestamp, open, high, low, close, volume]
            candles = []
            for candle in data:
                candles.append({
                    'timestamp': datetime.fromtimestamp(candle[0] / 1000),
                    'open': float(candle[1]),
                    'high': float(candle[2]), 
                    'low': float(candle[3]),
                    'close': float(candle[4]),
                    'volume': float(candle[5])
                })
            
            print(f"✅ Pobrano {len(candles)} świec historycznych dla {symbol}")
            
            # Initialize price history with historical data
            self.price_history = [candle['close'] for candle in candles[-50:]]
            self.current_price = candles[-1]['close']
            
            # Add to data manager if available
            if self.data_manager:
                for candle in candles[-50:]:  # Last 50 candles
                    self.data_manager.add_data_point(
                        price=candle['close'],
                        volume=candle['volume'],
                        high=candle['high'],
                        low=candle['low'],
                        timestamp=candle['timestamp']
                    )
            
            return candles
            
        except Exception as e:
            print(f"❌ Błąd pobierania danych historycznych: {e}")
            return []
    
    def calculate_technical_indicators(self, prices: List[float]):
        """Oblicz wskaźniki techniczne using custom implementation"""
        try:
            if len(prices) < 20:
                return
                
            # Basic indicators using numpy (fallback)
            if len(prices) >= 20:
                self.sma_20 = float(np.mean(prices[-20:]))
            if len(prices) >= 50:
                self.sma_50 = float(np.mean(prices[-50:]))
                
            # Basic RSI calculation
            if len(prices) >= 15:
                self.rsi = self._calculate_basic_rsi(prices)
            
            # Volatility (standard deviation)
            if len(prices) >= 20:
                self.volatility = float(np.std(prices[-20:]))
                
            # Try to use advanced technical indicators if available
            if self.data_manager:
                try:
                    from core.technical_indicators import TechnicalIndicators
                    
                    # Use professional indicators
                    self.sma_20 = TechnicalIndicators.sma(prices, 20)
                    self.sma_50 = TechnicalIndicators.sma(prices, 50)
                    self.rsi = TechnicalIndicators.rsi(prices, 14)
                    
                    print(f"📊 Professional Indicators - SMA20: {self.sma_20:.4f}, SMA50: {self.sma_50:.4f}, RSI: {self.rsi:.1f}")
                    return
                except ImportError:
                    pass
            
            print(f"📊 Basic Indicators - SMA20: {self.sma_20:.4f}, RSI: {self.rsi:.1f}, Vol: {self.volatility:.4f}")
            
        except Exception as e:
            print(f"⚠️ Błąd obliczania wskaźników: {e}")
    
    def _calculate_basic_rsi(self, prices: List[float], period: int = 14) -> float:
        """Basic RSI calculation using numpy"""
        try:
            if len(prices) < period + 1:
                return 50.0
            
            prices_array = np.array(prices)
            deltas = np.diff(prices_array)
            
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])
            
            if avg_loss == 0:
                return 100.0
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return float(rsi)
        except:
            return 50.0
    
    def on_message(self, ws, message):
        """Enhanced message handler with technical analysis"""
        try:
            data = json.loads(message)
            
            # Binance ticker stream format
            if 'c' in data:  # Current price
                self.current_price = float(data['c'])
                self.volume_24h = float(data['v'])
                self.price_change_24h = float(data['P'])
                self.bid_price = float(data.get('b', self.current_price))
                self.ask_price = float(data.get('a', self.current_price))
                self.last_update = datetime.now()
                
                # Dodaj do historii cen
                self.price_history.append(self.current_price)
                
                # Zachowaj tylko ostatnie 200 cen
                if len(self.price_history) > 200:
                    self.price_history = self.price_history[-200:]
                
                # Add to enhanced data manager if available
                if self.data_manager:
                    self.data_manager.add_data_point(
                        price=self.current_price,
                        volume=self.volume_24h,
                        high=self.current_price,  # Simplified for ticker data
                        low=self.current_price,
                        timestamp=self.last_update
                    )
                
                # Oblicz wskaźniki techniczne co 10 nowych cen
                if len(self.price_history) % 10 == 0:
                    self.calculate_technical_indicators(self.price_history)
                
                # Prepare basic market data
                market_data = {
                    'price': self.current_price,
                    'bid': self.bid_price,
                    'ask': self.ask_price,
                    'volume_24h': self.volume_24h,
                    'price_change_24h': self.price_change_24h,
                    'timestamp': self.last_update,
                    'sma_20': self.sma_20,
                    'sma_50': self.sma_50,
                    'rsi': self.rsi,
                    'volatility': self.volatility,
                    'spread': self.ask_price - self.bid_price,
                    'price_history': self.price_history.copy()
                }
                
                # Enhanced analysis if technical components available
                if self.data_manager and self.signal_generator:
                    try:
                        # Get professional technical indicators
                        indicators = self.data_manager.get_current_indicators()
                        
                        # Generate trading signals
                        analysis_data = self.data_manager.get_data_for_analysis()
                        analysis_data.update({
                            'price': self.current_price,
                            'volatility': self.volatility,
                            'price_change_24h': self.price_change_24h
                        })
                        
                        signals = self.signal_generator.generate_signals(analysis_data)
                        
                        # Add enhanced data to market_data
                        market_data.update({
                            'indicators': indicators,
                            'signals': signals,
                            'analysis_enhanced': True
                        })
                        
                        # Log enhanced signals periodically
                        if len(self.price_history) % 30 == 0:  # Every 30 updates
                            action = signals.get('action', 'hold')
                            confidence = signals.get('confidence', 0.0)
                            print(f"🎯 Signal: {action.upper()} (confidence: {confidence:.2f})")
                        
                    except Exception as e:
                        print(f"⚠️ Enhanced analysis error: {e}")
                        market_data['analysis_enhanced'] = False
                else:
                    market_data['analysis_enhanced'] = False
                
                # Wywołaj callback
                self.on_price_update(market_data)
                
        except Exception as e:
            print(f"❌ Błąd przetwarzania wiadomości: {e}")
    
    def on_error(self, ws, error):
        """Obsłuż błąd WebSocket"""
        print(f"❌ WebSocket error: {error}")
    
    def on_close(self, ws, close_status_code, close_msg):
        """Obsłuż zamknięcie WebSocket"""
        print("🔌 WebSocket connection closed")
        self.running = False
    
    def on_open(self, ws):
        """Obsłuż otwarcie WebSocket"""
        print("✅ WebSocket connection opened")
        self.running = True
    
    def start_stream(self, symbol: str = "solusdc"):
        """Uruchom strumień danych w czasie rzeczywistym"""
        try:
            # Pobierz dane historyczne na start
            historical = self.get_historical_data(symbol.upper())
            if historical:
                self.calculate_technical_indicators(self.price_history)
            
            # URL dla Binance WebSocket - 24hr ticker
            ws_url = f"wss://stream.binance.com:9443/ws/{symbol}@ticker"
            
            print(f"🔗 Łączę z Binance WebSocket: {symbol.upper()}")
            
            self.ws = websocket.WebSocketApp(
                ws_url,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close,
                on_open=self.on_open
            )
            
            # Uruchom w osobnym wątku
            ws_thread = threading.Thread(target=self.ws.run_forever)
            ws_thread.daemon = True
            ws_thread.start()
            
            return True
            
        except Exception as e:
            print(f"❌ Błąd uruchamiania WebSocket: {e}")
            return False
    
    def stop_stream(self):
        """Zatrzymaj strumień danych"""
        self.running = False
        if self.ws:
            self.ws.close()
    
    def get_current_data(self) -> Dict:
        """Pobierz aktualne dane rynkowe"""
        base_data = {
            'price': self.current_price,
            'bid': self.bid_price,
            'ask': self.ask_price,
            'volume_24h': self.volume_24h,
            'price_change_24h': self.price_change_24h,
            'last_update': self.last_update,
            'sma_20': self.sma_20,
            'sma_50': self.sma_50,
            'rsi': self.rsi,
            'volatility': self.volatility,
            'spread': self.ask_price - self.bid_price if self.ask_price and self.bid_price else 0.0,
            'is_connected': self.running,
            'price_history': self.price_history.copy()
        }
        
        # Add enhanced data if available
        if self.data_manager and self.signal_generator:
            try:
                indicators = self.data_manager.get_current_indicators()
                analysis_data = self.data_manager.get_data_for_analysis()
                analysis_data.update({
                    'price': self.current_price,
                    'volatility': self.volatility,
                    'price_change_24h': self.price_change_24h
                })
                signals = self.signal_generator.generate_signals(analysis_data)
                
                base_data.update({
                    'indicators': indicators,
                    'signals': signals,
                    'analysis_enhanced': True
                })
            except Exception as e:
                print(f"⚠️ Error getting enhanced data: {e}")
                base_data['analysis_enhanced'] = False
        else:
            base_data['analysis_enhanced'] = False
        
        return base_data

class TradingSignals:
    """Enhanced trading signals with professional analysis"""
    
    @staticmethod
    def analyze_market_conditions(market_data: Dict) -> Dict:
        """Analizuj warunki rynkowe i generuj sygnały"""
        signals = {
            'trend': 'neutral',
            'strength': 0.0,
            'action': 'hold',
            'confidence': 0.0,
            'reasons': []
        }
        
        try:
            price = market_data['price']
            sma_20 = market_data.get('sma_20', 0)
            sma_50 = market_data.get('sma_50', 0)
            rsi = market_data.get('rsi', 50)
            volatility = market_data.get('volatility', 0.01)
            price_change_24h = market_data.get('price_change_24h', 0)
            
            # Check for enhanced signals first
            if market_data.get('analysis_enhanced') and 'signals' in market_data:
                enhanced_signals = market_data['signals']
                signals.update({
                    'action': enhanced_signals.get('action', 'hold'),
                    'confidence': enhanced_signals.get('confidence', 0.0),
                    'strength': enhanced_signals.get('strength', 0.0),
                    'reasons': enhanced_signals.get('reasons', []),
                    'risk_level': enhanced_signals.get('risk_level', 'medium')
                })
                
                # Determine trend from action
                action = signals['action']
                if 'buy' in action:
                    signals['trend'] = 'bullish'
                elif 'sell' in action:
                    signals['trend'] = 'bearish'
                else:
                    signals['trend'] = 'neutral'
                
                return signals
            
            # Fallback to basic analysis
            # Analiza trendu (SMA crossover)
            if sma_20 > 0 and sma_50 > 0:
                if sma_20 > sma_50:
                    signals['trend'] = 'bullish'
                    signals['strength'] += 0.3
                    signals['reasons'].append('SMA20 > SMA50 (bullish)')
                else:
                    signals['trend'] = 'bearish'
                    signals['strength'] -= 0.3
                    signals['reasons'].append('SMA20 < SMA50 (bearish)')
            
            # Analiza RSI
            if rsi < 30:
                signals['action'] = 'buy'
                signals['strength'] += 0.4
                signals['reasons'].append(f'RSI oversold ({rsi:.1f})')
            elif rsi > 70:
                signals['action'] = 'sell'
                signals['strength'] -= 0.4
                signals['reasons'].append(f'RSI overbought ({rsi:.1f})')
            
            # Analiza momentum (price change 24h)
            if price_change_24h > 2.0:
                signals['strength'] += 0.2
                signals['reasons'].append(f'Strong 24h momentum (+{price_change_24h:.1f}%)')
            elif price_change_24h < -2.0:
                signals['strength'] -= 0.2
                signals['reasons'].append(f'Negative 24h momentum ({price_change_24h:.1f}%)')
            
            # Analiza volatility
            if volatility > 0.5:  # High volatility
                signals['strength'] *= 0.8  # Reduce confidence
                signals['reasons'].append('High volatility - reduced confidence')
            
            # Finalne sygnały
            signals['confidence'] = min(abs(signals['strength']), 1.0)
            
            if signals['strength'] > 0.5:
                signals['action'] = 'buy'
            elif signals['strength'] < -0.5:
                signals['action'] = 'sell'
            else:
                signals['action'] = 'hold'
                
        except Exception as e:
            print(f"⚠️ Błąd analizy sygnałów: {e}")
        
        return signals

# Helper funkcja do użycia w innych plikach
def create_market_data_service(callback_function):
    """Stwórz i uruchom serwis danych rynkowych"""
    market_service = BinanceMarketData(callback_function)
    
    if market_service.start_stream("solusdc"):
        print("✅ Market data service uruchomiony")
        return market_service
    else:
        print("❌ Nie udało się uruchomić market data service")
        return None