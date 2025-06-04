# core/multi_asset_data.py - Enhanced Multi-Asset Market Data Manager
import websocket
import json
import threading
import time
import requests
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Callable, Optional
import logging


class MultiAssetMarketData:
    """Enhanced multi-asset market data manager with portfolio optimization"""
    
    def __init__(self, assets: List[str], callback: Callable[[str, Dict], None]):
        self.assets = assets
        self.callback = callback
        self.connections = {}
        self.asset_data = {}
        self.running = False
        
        # Enhanced tracking
        self.price_histories = {asset: [] for asset in assets}
        self.connection_status = {asset: False for asset in assets}
        self.last_updates = {asset: None for asset in assets}
        
        # Asset mapping for Binance symbols
        self.symbol_mapping = {
            'SOL': 'SOLUSDC',
            'ETH': 'ETHUSDC', 
            'BTC': 'BTCUSDC'
        }
        
        self.logger = logging.getLogger(__name__)
        print(f"✅ Multi-asset manager initialized for: {assets}")
    
    def start_tracking(self) -> bool:
        """Start tracking all assets"""
        print(f"🚀 Starting multi-asset tracking for {len(self.assets)} assets...")
        
        success_count = 0
        
        for asset in self.assets:
            try:
                if self._start_asset_stream(asset):
                    success_count += 1
                    time.sleep(1)  # Stagger connections
            except Exception as e:
                print(f"❌ Failed to start {asset} stream: {e}")
        
        self.running = success_count > 0
        
        if self.running:
            print(f"✅ Multi-asset tracking started: {success_count}/{len(self.assets)} assets connected")
            
            # Start monitoring thread
            monitor_thread = threading.Thread(target=self._monitor_connections, daemon=True)
            monitor_thread.start()
            
            return True
        else:
            print("❌ Failed to start any asset streams")
            return False
    
    def _start_asset_stream(self, asset: str) -> bool:
        """Start individual asset stream"""
        try:
            # Get historical data first
            symbol = self.symbol_mapping.get(asset, f"{asset}USDC")
            historical_data = self._get_historical_data(symbol)
            
            if historical_data:
                # Initialize price history
                self.price_histories[asset] = [candle['close'] for candle in historical_data[-50:]]
                
                # Initialize current data
                latest = historical_data[-1]
                self.asset_data[asset] = {
                    'price': latest['close'],
                    'volume_24h': latest['volume'],
                    'price_change_24h': 0.0,  # Will be updated by WebSocket
                    'timestamp': latest['timestamp'],
                    'rsi': self._calculate_initial_rsi(asset),
                    'volatility': self._calculate_initial_volatility(asset),
                    'sma_20': np.mean(self.price_histories[asset][-20:]) if len(self.price_histories[asset]) >= 20 else latest['close'],
                    'spread': latest['close'] * 0.001,  # Estimated spread
                    'bid': latest['close'] * 0.999,
                    'ask': latest['close'] * 1.001,
                    'confidence': 0.5  # Initial confidence
                }
                
                print(f"📊 {asset} historical data loaded: {len(historical_data)} candles")
            
            # Start WebSocket connection
            ws_url = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@ticker"
            
            ws = websocket.WebSocketApp(
                ws_url,
                on_message=lambda ws, msg: self._on_message(asset, ws, msg),
                on_error=lambda ws, error: self._on_error(asset, ws, error),
                on_close=lambda ws, close_status_code, close_msg: self._on_close(asset, ws, close_status_code, close_msg),
                on_open=lambda ws: self._on_open(asset, ws)
            )
            
            # Start in separate thread
            ws_thread = threading.Thread(target=ws.run_forever, daemon=True)
            ws_thread.start()
            
            self.connections[asset] = ws
            time.sleep(2)  # Allow connection to establish
            
            return self.connection_status.get(asset, False)
            
        except Exception as e:
            print(f"❌ Error starting {asset} stream: {e}")
            return False
    
    def _get_historical_data(self, symbol: str, interval: str = "1m", limit: int = 100) -> List[Dict]:
        """Get historical data from Binance API"""
        try:
            url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
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
            
            return candles
            
        except Exception as e:
            print(f"⚠️ Historical data error for {symbol}: {e}")
            return []
    
    def _calculate_initial_rsi(self, asset: str, period: int = 14) -> float:
        """Calculate initial RSI from historical data"""
        try:
            prices = self.price_histories[asset]
            if len(prices) < period + 1:
                return 50.0
            
            # Calculate price changes
            changes = []
            for i in range(1, len(prices)):
                changes.append(prices[i] - prices[i-1])
            
            if len(changes) < period:
                return 50.0
            
            # Separate gains and losses
            gains = [max(change, 0) for change in changes]
            losses = [max(-change, 0) for change in changes]
            
            # Calculate average gains and losses
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])
            
            if avg_loss == 0:
                return 99.0
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return max(1.0, min(99.0, rsi))
            
        except Exception as e:
            print(f"⚠️ RSI calculation error for {asset}: {e}")
            return 50.0
    
    def _calculate_initial_volatility(self, asset: str) -> float:
        """Calculate initial volatility from historical data"""
        try:
            prices = self.price_histories[asset]
            if len(prices) < 20:
                return 0.01
            
            # Calculate price changes
            changes = []
            for i in range(1, len(prices)):
                change = (prices[i] - prices[i-1]) / prices[i-1]
                changes.append(change)
            
            return float(np.std(changes[-20:]))
            
        except Exception as e:
            print(f"⚠️ Volatility calculation error for {asset}: {e}")
            return 0.01
    
    def _on_open(self, asset: str, ws):
        """Handle WebSocket connection open"""
        print(f"✅ {asset} WebSocket connected")
        self.connection_status[asset] = True
    
    def _on_message(self, asset: str, ws, message):
        """Handle WebSocket message for specific asset"""
        try:
            data = json.loads(message)
            
            if 'c' in data:  # Current price from ticker stream
                current_price = float(data['c'])
                volume_24h = float(data['v'])
                price_change_24h = float(data['P'])
                bid_price = float(data.get('b', current_price))
                ask_price = float(data.get('a', current_price))
                
                # Update price history
                self.price_histories[asset].append(current_price)
                if len(self.price_histories[asset]) > 200:
                    self.price_histories[asset] = self.price_histories[asset][-200:]
                
                # Calculate technical indicators
                rsi = self._calculate_rsi(asset)
                volatility = self._calculate_volatility(asset)
                sma_20 = self._calculate_sma(asset, 20)
                
                # Update asset data
                self.asset_data[asset] = {
                    'price': current_price,
                    'bid': bid_price,
                    'ask': ask_price,
                    'volume_24h': volume_24h,
                    'price_change_24h': price_change_24h,
                    'timestamp': datetime.now(),
                    'rsi': rsi,
                    'volatility': volatility,
                    'sma_20': sma_20,
                    'spread': ask_price - bid_price,
                    'price_history': self.price_histories[asset].copy(),
                    'confidence': self._calculate_confidence(asset)
                }
                
                self.last_updates[asset] = datetime.now()
                
                # Call callback
                self.callback(asset, self.asset_data[asset])
                
        except Exception as e:
            print(f"❌ Message processing error for {asset}: {e}")
    
    def _calculate_rsi(self, asset: str, period: int = 14) -> float:
        """Calculate RSI for asset"""
        try:
            prices = self.price_histories[asset]
            if len(prices) < period + 1:
                return 50.0
            
            # Get recent prices
            recent_prices = prices[-(period + 10):]
            changes = []
            
            for i in range(1, len(recent_prices)):
                changes.append(recent_prices[i] - recent_prices[i-1])
            
            if len(changes) < period:
                return 50.0
            
            # Separate gains and losses
            gains = [max(change, 0) for change in changes]
            losses = [max(-change, 0) for change in changes]
            
            # Calculate averages
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])
            
            if avg_loss == 0:
                return 99.0
            if avg_gain == 0:
                return 1.0
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return max(1.0, min(99.0, rsi))
            
        except Exception as e:
            print(f"⚠️ RSI error for {asset}: {e}")
            return 50.0
    
    def _calculate_volatility(self, asset: str) -> float:
        """Calculate volatility for asset"""
        try:
            prices = self.price_histories[asset]
            if len(prices) < 20:
                return 0.01
            
            # Calculate percentage changes
            changes = []
            for i in range(1, len(prices[-20:])):
                change = (prices[-20:][i] - prices[-20:][i-1]) / prices[-20:][i-1]
                changes.append(change)
            
            return float(np.std(changes))
            
        except Exception as e:
            return 0.01
    
    def _calculate_sma(self, asset: str, period: int) -> float:
        """Calculate Simple Moving Average for asset"""
        try:
            prices = self.price_histories[asset]
            if len(prices) < period:
                return prices[-1] if prices else 0.0
            
            return float(np.mean(prices[-period:]))
            
        except Exception as e:
            return 0.0
    
    def _calculate_confidence(self, asset: str) -> float:
        """Calculate trading confidence for asset based on multiple factors"""
        try:
            if asset not in self.asset_data:
                return 0.5
            
            data = self.asset_data[asset]
            confidence = 0.5
            
            # RSI factor
            rsi = data.get('rsi', 50)
            if 30 <= rsi <= 70:
                confidence += 0.2  # Good RSI range
            elif rsi < 25 or rsi > 75:
                confidence += 0.1  # Extreme RSI (potential reversal)
            
            # Volatility factor
            volatility = data.get('volatility', 0.01)
            if 0.01 <= volatility <= 0.05:
                confidence += 0.2  # Good volatility range
            elif volatility > 0.1:
                confidence -= 0.1  # Too volatile
            
            # Volume factor
            volume = data.get('volume_24h', 0)
            if volume > 1000000:  # High volume threshold
                confidence += 0.1
            
            # Price momentum factor
            price_change = data.get('price_change_24h', 0)
            if abs(price_change) < 5:
                confidence += 0.1  # Stable price
            elif abs(price_change) > 15:
                confidence -= 0.1  # Too volatile
            
            return max(0.1, min(0.9, confidence))
            
        except Exception as e:
            return 0.5
    
    def _on_error(self, asset: str, ws, error):
        """Handle WebSocket error"""
        print(f"❌ {asset} WebSocket error: {error}")
        self.connection_status[asset] = False
    
    def _on_close(self, asset: str, ws, close_status_code, close_msg):
        """Handle WebSocket close"""
        print(f"🔌 {asset} WebSocket closed")
        self.connection_status[asset] = False
    
    def _monitor_connections(self):
        """Monitor and reconnect dead connections"""
        while self.running:
            try:
                time.sleep(30)  # Check every 30 seconds
                
                for asset in self.assets:
                    # Check if connection is stale
                    last_update = self.last_updates.get(asset)
                    if last_update and (datetime.now() - last_update).seconds > 120:
                        print(f"⚠️ {asset} connection stale, reconnecting...")
                        self._restart_asset_stream(asset)
                    
                    # Check connection status
                    if not self.connection_status.get(asset, False):
                        print(f"⚠️ {asset} disconnected, attempting reconnect...")
                        self._restart_asset_stream(asset)
                        
            except Exception as e:
                print(f"⚠️ Connection monitor error: {e}")
    
    def _restart_asset_stream(self, asset: str):
        """Restart individual asset stream"""
        try:
            # Close existing connection
            if asset in self.connections:
                try:
                    self.connections[asset].close()
                except:
                    pass
                del self.connections[asset]
            
            # Reset status
            self.connection_status[asset] = False
            
            # Restart after delay
            time.sleep(2)
            self._start_asset_stream(asset)
            
        except Exception as e:
            print(f"❌ Failed to restart {asset} stream: {e}")
    
    def stop_tracking(self):
        """Stop all asset tracking"""
        print("🛑 Stopping multi-asset tracking...")
        self.running = False
        
        for asset, ws in self.connections.items():
            try:
                ws.close()
                print(f"✅ {asset} stream stopped")
            except:
                pass
        
        self.connections.clear()
        self.connection_status = {asset: False for asset in self.assets}
        print("✅ All asset streams stopped")
    
    def get_connection_status(self) -> Dict[str, bool]:
        """Get connection status for all assets"""
        return self.connection_status.copy()
    
    def get_asset_data(self, asset: str) -> Optional[Dict]:
        """Get current data for specific asset"""
        return self.asset_data.get(asset)
    
    def get_all_asset_data(self) -> Dict[str, Dict]:
        """Get current data for all assets"""
        return self.asset_data.copy()
    
    def get_asset_price(self, asset: str) -> float:
        """Get current price for asset"""
        data = self.asset_data.get(asset)
        return data.get('price', 0.0) if data else 0.0
    
    def get_asset_rsi(self, asset: str) -> float:
        """Get current RSI for asset"""
        data = self.asset_data.get(asset)
        return data.get('rsi', 50.0) if data else 50.0
    
    def get_asset_confidence(self, asset: str) -> float:
        """Get trading confidence for asset"""
        data = self.asset_data.get(asset)
        return data.get('confidence', 0.5) if data else 0.5


class MultiAssetSignals:
    """Enhanced multi-asset signal analysis and portfolio optimization"""
    
    def __init__(self):
        self.correlation_window = 50
        self.signal_weights = {
            'rsi': 0.3,
            'volatility': 0.2, 
            'momentum': 0.25,
            'volume': 0.15,
            'technical': 0.1
        }
    
    def analyze_multi_asset_conditions(self, asset_data: Dict[str, Dict]) -> Dict[str, Dict]:
        """Analyze conditions across all assets"""
        signals = {}
        
        try:
            for asset, data in asset_data.items():
                signals[asset] = self._analyze_single_asset(asset, data, asset_data)
            
            # Add correlation analysis
            self._add_correlation_analysis(signals, asset_data)
            
            return signals
            
        except Exception as e:
            print(f"❌ Multi-asset analysis error: {e}")
            return {}
    
    def _analyze_single_asset(self, asset: str, data: Dict, all_data: Dict[str, Dict]) -> Dict:
        """Analyze signals for single asset"""
        try:
            signal = {
                'asset': asset,
                'action': 'HOLD',
                'confidence': 0.5,
                'scores': {},
                'reasons': []
            }
            
            # RSI analysis
            rsi = data.get('rsi', 50)
            rsi_score = self._calculate_rsi_score(rsi)
            signal['scores']['rsi'] = rsi_score
            
            if rsi < 30:
                signal['reasons'].append(f"Oversold RSI ({rsi:.1f})")
            elif rsi > 70:
                signal['reasons'].append(f"Overbought RSI ({rsi:.1f})")
            
            # Volatility analysis
            volatility = data.get('volatility', 0.01)
            vol_score = self._calculate_volatility_score(volatility)
            signal['scores']['volatility'] = vol_score
            
            # Momentum analysis
            price_change = data.get('price_change_24h', 0)
            momentum_score = self._calculate_momentum_score(price_change)
            signal['scores']['momentum'] = momentum_score
            
            if abs(price_change) > 5:
                direction = "Strong" if abs(price_change) > 10 else "Moderate"
                trend = "upward" if price_change > 0 else "downward"
                signal['reasons'].append(f"{direction} {trend} momentum ({price_change:+.1f}%)")
            
            # Volume analysis
            volume = data.get('volume_24h', 0)
            volume_score = self._calculate_volume_score(volume, asset)
            signal['scores']['volume'] = volume_score
            
            # Technical analysis
            tech_score = self._calculate_technical_score(data)
            signal['scores']['technical'] = tech_score
            
            # Calculate weighted confidence
            total_score = 0
            for indicator, score in signal['scores'].items():
                weight = self.signal_weights.get(indicator, 0.2)
                total_score += score * weight
            
            signal['confidence'] = max(0.1, min(0.9, total_score))
            
            # Determine action
            if signal['confidence'] > 0.7:
                signal['action'] = 'BUY'
            elif signal['confidence'] < 0.3:
                signal['action'] = 'SELL'
            else:
                signal['action'] = 'HOLD'
            
            return signal
            
        except Exception as e:
            print(f"⚠️ Single asset analysis error for {asset}: {e}")
            return {'asset': asset, 'action': 'HOLD', 'confidence': 0.5, 'scores': {}, 'reasons': []}
    
    def _calculate_rsi_score(self, rsi: float) -> float:
        """Calculate RSI-based score (0.0 to 1.0)"""
        if rsi < 25:
            return 0.8  # Strong oversold signal
        elif rsi < 35:
            return 0.6  # Moderate oversold
        elif rsi > 75:
            return 0.2  # Strong overbought signal
        elif rsi > 65:
            return 0.4  # Moderate overbought
        else:
            return 0.5  # Neutral
    
    def _calculate_volatility_score(self, volatility: float) -> float:
        """Calculate volatility-based score"""
        if volatility < 0.005:
            return 0.3  # Too low volatility
        elif volatility < 0.02:
            return 0.7  # Good volatility for trading
        elif volatility < 0.05:
            return 0.6  # Moderate volatility
        elif volatility < 0.1:
            return 0.4  # High volatility
        else:
            return 0.2  # Too high volatility
    
    def _calculate_momentum_score(self, price_change: float) -> float:
        """Calculate momentum-based score"""
        abs_change = abs(price_change)
        
        if abs_change < 1:
            return 0.4  # Low momentum
        elif abs_change < 3:
            return 0.6  # Moderate momentum
        elif abs_change < 7:
            return 0.7  # Good momentum
        elif abs_change < 15:
            return 0.6  # High momentum
        else:
            return 0.3  # Extreme momentum (risky)
    
    def _calculate_volume_score(self, volume: float, asset: str) -> float:
        """Calculate volume-based score"""
        # Volume thresholds by asset (approximate)
        volume_thresholds = {
            'BTC': 1000000000,  # 1B USDC
            'ETH': 500000000,   # 500M USDC
            'SOL': 100000000    # 100M USDC
        }
        
        threshold = volume_thresholds.get(asset, 50000000)
        
        if volume > threshold * 2:
            return 0.8  # Very high volume
        elif volume > threshold:
            return 0.7  # High volume
        elif volume > threshold * 0.5:
            return 0.6  # Moderate volume
        elif volume > threshold * 0.2:
            return 0.4  # Low volume
        else:
            return 0.2  # Very low volume
    
    def _calculate_technical_score(self, data: Dict) -> float:
        """Calculate technical analysis score"""
        try:
            score = 0.5
            
            # Price vs SMA
            price = data.get('price', 0)
            sma_20 = data.get('sma_20', price)
            
            if price > sma_20 * 1.02:
                score += 0.2  # Price above SMA
            elif price < sma_20 * 0.98:
                score -= 0.2  # Price below SMA
            
            # Spread analysis
            spread = data.get('spread', 0)
            if spread < price * 0.002:  # Low spread
                score += 0.1
            
            return max(0.1, min(0.9, score))
            
        except Exception as e:
            return 0.5
    
    def _add_correlation_analysis(self, signals: Dict[str, Dict], asset_data: Dict[str, Dict]):
        """Add correlation analysis between assets"""
        try:
            assets = list(asset_data.keys())
            
            if len(assets) < 2:
                return
            
            # Calculate correlations (simplified)
            correlations = {}
            
            for i, asset1 in enumerate(assets):
                for j, asset2 in enumerate(assets[i+1:], i+1):
                    try:
                        data1 = asset_data[asset1]
                        data2 = asset_data[asset2]
                        
                        # Simple correlation based on price changes
                        change1 = data1.get('price_change_24h', 0)
                        change2 = data2.get('price_change_24h', 0)
                        
                        # Simplified correlation indicator
                        correlation = 1 if (change1 > 0) == (change2 > 0) else -1
                        correlations[f"{asset1}-{asset2}"] = correlation
                        
                    except Exception as e:
                        continue
            
            # Adjust confidence based on correlations
            for asset, signal in signals.items():
                try:
                    # If highly correlated with other assets, reduce unique signal strength
                    correlation_penalty = 0
                    for corr_pair, corr_value in correlations.items():
                        if asset in corr_pair and abs(corr_value) > 0.8:
                            correlation_penalty += 0.05
                    
                    signal['confidence'] = max(0.1, signal['confidence'] - correlation_penalty)
                    signal['correlation_penalty'] = correlation_penalty
                    
                except Exception as e:
                    continue
                    
        except Exception as e:
            print(f"⚠️ Correlation analysis error: {e}")
    
    def get_best_asset_to_trade(self, signals: Dict[str, Dict]) -> Optional[str]:
        """Get the best asset to trade based on signals"""
        try:
            if not signals:
                return None
            
            # Find asset with highest confidence and BUY signal
            best_asset = None
            best_score = 0
            
            for asset, signal in signals.items():
                confidence = signal.get('confidence', 0)
                action = signal.get('action', 'HOLD')
                
                # Prioritize BUY signals
                if action == 'BUY' and confidence > best_score:
                    best_score = confidence
                    best_asset = asset
            
            # If no BUY signals, get highest confidence HOLD
            if not best_asset:
                for asset, signal in signals.items():
                    confidence = signal.get('confidence', 0)
                    action = signal.get('action', 'HOLD')
                    
                    if action == 'HOLD' and confidence > best_score:
                        best_score = confidence
                        best_asset = asset
            
            return best_asset
            
        except Exception as e:
            print(f"⚠️ Best asset selection error: {e}")
            return None
    
    def get_portfolio_recommendations(self, signals: Dict[str, Dict], 
                                    current_allocation: Dict[str, float],
                                    target_allocation: Dict[str, float]) -> Dict[str, str]:
        """Get portfolio rebalancing recommendations"""
        try:
            recommendations = {}
            
            for asset in signals.keys():
                signal = signals[asset]
                current_pct = current_allocation.get(asset, 0)
                target_pct = target_allocation.get(asset, 0)
                confidence = signal.get('confidence', 0.5)
                
                deficit = target_pct - current_pct
                
                if deficit > 0.1 and confidence > 0.6:
                    recommendations[asset] = 'INCREASE'
                elif deficit < -0.1 and confidence < 0.4:
                    recommendations[asset] = 'DECREASE'
                else:
                    recommendations[asset] = 'MAINTAIN'
            
            return recommendations
            
        except Exception as e:
            print(f"⚠️ Portfolio recommendations error: {e}")
            return {}


def create_multi_asset_service(assets: List[str], callback: Callable[[str, Dict], None]) -> Optional[MultiAssetMarketData]:
    """Create and start multi-asset market data service"""
    try:
        print(f"🚀 Creating multi-asset service for: {assets}")
        
        service = MultiAssetMarketData(assets, callback)
        
        if service.start_tracking():
            print("✅ Multi-asset service created successfully")
            return service
        else:
            print("❌ Failed to create multi-asset service")
            return None
            
    except Exception as e:
        print(f"❌ Multi-asset service creation error: {e}")
        return None