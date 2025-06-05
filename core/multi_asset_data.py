# core/multi_asset_data.py - ENHANCED DIRECTIONAL MULTI-ASSET DATA MANAGER
import websocket
import json
import threading
import time
import requests
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Callable, Optional, Tuple
import logging


class DirectionalMultiAssetData:
    """🎯 Enhanced multi-asset market data manager with FULL directional trading signals"""
    
    def __init__(self, assets: List[str], callback: Callable[[str, Dict], None]):
        self.assets = assets
        self.callback = callback
        self.connections = {}
        self.asset_data = {}
        self.running = False
        
        # Enhanced tracking for directional trading
        self.price_histories = {asset: [] for asset in assets}
        self.connection_status = {asset: False for asset in assets}
        self.last_updates = {asset: None for asset in assets}
        
        # 🎯 DIRECTIONAL SIGNAL TRACKING
        self.directional_signals = {asset: {'long': 0.0, 'short': 0.0, 'hold': 0.0} for asset in assets}
        self.signal_history = {asset: [] for asset in assets}
        self.correlation_matrix = {}
        
        # 🚀 ENHANCED: ML INTEGRATION
        self.ml_integration = None
        self.ml_predictions = {asset: {} for asset in assets}
        
        # Asset mapping for Binance symbols
        self.symbol_mapping = {
            'SOL': 'SOLUSDC',
            'ETH': 'ETHUSDC', 
            'BTC': 'BTCUSDC'
        }
        
        self.logger = logging.getLogger(__name__)
        print(f"🎯 Directional multi-asset manager initialized for: {assets}")
    
    def set_ml_integration(self, ml_integration):
        """🤖 Set ML integration for enhanced directional predictions"""
        self.ml_integration = ml_integration
        print("✅ ML integration connected to multi-asset data manager")
    
    def start_tracking(self) -> bool:
        """Start tracking all assets with directional analysis"""
        print(f"🎯 Starting directional multi-asset tracking for {len(self.assets)} assets...")
        
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
            print(f"✅ Directional multi-asset tracking started: {success_count}/{len(self.assets)} assets connected")
            
            # Start monitoring and analysis threads
            monitor_thread = threading.Thread(target=self._monitor_connections, daemon=True)
            monitor_thread.start()
            
            analysis_thread = threading.Thread(target=self._continuous_directional_analysis, daemon=True)
            analysis_thread.start()
            
            return True
        else:
            print("❌ Failed to start any asset streams")
            return False
    
    def _start_asset_stream(self, asset: str) -> bool:
        """Start individual asset stream with directional analysis setup"""
        try:
            # Get historical data first
            symbol = self.symbol_mapping.get(asset, f"{asset}USDC")
            historical_data = self._get_historical_data(symbol)
            
            if historical_data:
                # Initialize price history
                self.price_histories[asset] = [candle['close'] for candle in historical_data[-50:]]
                
                # Initialize current data with directional analysis
                latest = historical_data[-1]
                
                # Calculate initial directional signals
                initial_signals = self._calculate_initial_directional_signals(asset, historical_data)
                
                self.asset_data[asset] = {
                    'price': latest['close'],
                    'volume_24h': latest['volume'],
                    'price_change_24h': 0.0,
                    'timestamp': latest['timestamp'],
                    'rsi': self._calculate_initial_rsi(asset),
                    'volatility': self._calculate_initial_volatility(asset),
                    'sma_20': np.mean(self.price_histories[asset][-20:]) if len(self.price_histories[asset]) >= 20 else latest['close'],
                    'spread': latest['close'] * 0.001,
                    'bid': latest['close'] * 0.999,
                    'ask': latest['close'] * 1.001,
                    'confidence': 0.5,
                    
                    # 🎯 DIRECTIONAL SIGNALS
                    'directional_signals': initial_signals,
                    'recommended_direction': self._determine_recommended_direction(initial_signals),
                    'signal_strength': max(initial_signals.values()),
                    'trend_direction': self._calculate_trend_direction(asset),
                    'momentum_direction': self._calculate_momentum_direction(asset),
                    
                    # 🚀 ENHANCED: TRADING READINESS
                    'trading_recommendation': self._generate_trading_recommendation(initial_signals, latest['close']),
                    'position_suggestion': self._calculate_position_suggestion(initial_signals),
                    'risk_level': self._assess_risk_level(asset)
                }
                
                # Store directional signals
                self.directional_signals[asset] = initial_signals
                
                print(f"🎯 {asset} directional data loaded: {len(historical_data)} candles")
                print(f"   📊 Initial signals - LONG: {initial_signals['long']:.2f}, SHORT: {initial_signals['short']:.2f}, HOLD: {initial_signals['hold']:.2f}")
            
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
            print(f"❌ Error starting {asset} directional stream: {e}")
            return False
    
    def _calculate_initial_directional_signals(self, asset: str, historical_data: List[Dict]) -> Dict[str, float]:
        """🎯 Calculate initial directional signals from historical data"""
        try:
            if len(historical_data) < 10:
                return {'long': 0.33, 'short': 0.33, 'hold': 0.34}
            
            prices = [candle['close'] for candle in historical_data[-20:]]
            volumes = [candle['volume'] for candle in historical_data[-20:]]
            
            # Calculate technical indicators
            rsi = self._calculate_rsi_from_prices(prices)
            price_change_24h = ((prices[-1] - prices[-24]) / prices[-24] * 100) if len(prices) >= 24 else 0
            volatility = np.std([p / prices[i-1] - 1 for i, p in enumerate(prices[1:])]) if len(prices) > 1 else 0.02
            volume_trend = (np.mean(volumes[-5:]) / np.mean(volumes[-10:-5])) if len(volumes) >= 10 else 1.0
            
            # 🎯 DIRECTIONAL SIGNAL CALCULATION
            long_signal = 0.0
            short_signal = 0.0
            hold_signal = 0.0
            
            # 🚀 ENHANCED: RSI-based signals for directional trading
            if rsi < 20:
                long_signal += 0.5  # Strong oversold = LONG opportunity
                print(f"   🎯 {asset} STRONG LONG signal: RSI {rsi:.1f}")
            elif rsi < 30:
                long_signal += 0.35
            elif rsi < 35:
                long_signal += 0.2
            elif rsi > 80:
                short_signal += 0.5  # Strong overbought = SHORT opportunity
                print(f"   🎯 {asset} STRONG SHORT signal: RSI {rsi:.1f}")
            elif rsi > 70:
                short_signal += 0.35
            elif rsi > 65:
                short_signal += 0.2
            elif 40 <= rsi <= 60:
                hold_signal += 0.3  # Neutral zone = HOLD
            
            # 🚀 ENHANCED: Momentum-based signals for profit on trends
            if price_change_24h < -8:
                long_signal += 0.4  # Strong dip = buy the dip
                print(f"   🎯 {asset} DIP BUYING opportunity: {price_change_24h:.1f}%")
            elif price_change_24h < -4:
                long_signal += 0.25
            elif price_change_24h < -2:
                long_signal += 0.1
            elif price_change_24h > 12:
                short_signal += 0.4  # Strong rally = short the top
                print(f"   🎯 {asset} SHORT opportunity: {price_change_24h:.1f}%")
            elif price_change_24h > 6:
                short_signal += 0.25
            elif price_change_24h > 3:
                short_signal += 0.1
            elif -1 <= price_change_24h <= 1:
                hold_signal += 0.25  # Sideways movement = wait
            
            # Volatility-based signals
            if volatility < 0.005:
                hold_signal += 0.3  # Very low volatility = wait for breakout
            elif volatility < 0.01:
                hold_signal += 0.15
            elif volatility > 0.08:
                # Very high volatility = risky but opportunities exist
                long_signal *= 0.7
                short_signal *= 0.7
                hold_signal += 0.2
            elif volatility > 0.04:
                # High volatility = proceed with caution
                long_signal *= 0.85
                short_signal *= 0.85
                hold_signal += 0.1
            
            # Volume confirmation
            if volume_trend > 2.0:  # High volume confirms signals
                long_signal += 0.1
                short_signal += 0.1
            elif volume_trend < 0.7:  # Low volume = hold
                hold_signal += 0.15
            
            # Normalize to probabilities
            total = long_signal + short_signal + hold_signal
            if total > 0:
                long_signal /= total
                short_signal /= total
                hold_signal /= total
            else:
                long_signal = short_signal = hold_signal = 0.33
            
            # Ensure minimum values
            long_signal = max(0.1, long_signal)
            short_signal = max(0.1, short_signal)
            hold_signal = max(0.1, hold_signal)
            
            # Re-normalize
            total = long_signal + short_signal + hold_signal
            long_signal /= total
            short_signal /= total
            hold_signal /= total
            
            return {
                'long': float(long_signal),
                'short': float(short_signal),
                'hold': float(hold_signal)
            }
            
        except Exception as e:
            print(f"⚠️ Error calculating initial directional signals for {asset}: {e}")
            return {'long': 0.33, 'short': 0.33, 'hold': 0.34}
    
    def _generate_trading_recommendation(self, signals: Dict[str, float], price: float) -> Dict[str, any]:
        """🚀 Generate concrete trading recommendation"""
        max_signal = max(signals.values())
        direction = max(signals, key=signals.get)
        
        recommendation = {
            'action': direction.upper(),
            'confidence': max_signal,
            'entry_price': price,
            'stop_loss': 0.0,
            'take_profit': 0.0,
            'position_size': 1.0
        }
        
        # Calculate risk management levels
        if direction == 'long':
            recommendation['stop_loss'] = price * 0.97  # 3% stop loss
            recommendation['take_profit'] = price * 1.06  # 6% take profit
        elif direction == 'short':
            recommendation['stop_loss'] = price * 1.03  # 3% stop loss
            recommendation['take_profit'] = price * 0.94  # 6% take profit
        
        # Adjust position size based on confidence
        if max_signal > 0.7:
            recommendation['position_size'] = 1.0
        elif max_signal > 0.6:
            recommendation['position_size'] = 0.8
        elif max_signal > 0.5:
            recommendation['position_size'] = 0.6
        else:
            recommendation['position_size'] = 0.4
        
        return recommendation
    
    def _calculate_position_suggestion(self, signals: Dict[str, float]) -> str:
        """🎯 Calculate position suggestion for bot"""
        long_prob = signals.get('long', 0)
        short_prob = signals.get('short', 0)
        hold_prob = signals.get('hold', 0)
        
        if long_prob > 0.6:
            return 'OPEN_LONG'
        elif short_prob > 0.6:
            return 'OPEN_SHORT'
        elif hold_prob > 0.6:
            return 'STAY_NEUTRAL'
        elif long_prob > short_prob:
            return 'LEAN_LONG'
        elif short_prob > long_prob:
            return 'LEAN_SHORT'
        else:
            return 'WAIT'
    
    def _assess_risk_level(self, asset: str) -> str:
        """📊 Assess risk level for trading"""
        try:
            if asset not in self.asset_data:
                return 'medium'
            
            data = self.asset_data[asset]
            volatility = data.get('volatility', 0.02)
            price_change = abs(data.get('price_change_24h', 0))
            
            if volatility > 0.1 or price_change > 15:
                return 'high'
            elif volatility < 0.01 and price_change < 2:
                return 'low'
            else:
                return 'medium'
                
        except Exception as e:
            return 'medium'
    
    def _calculate_rsi_from_prices(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI from price list"""
        try:
            if len(prices) < period + 1:
                return 50.0
            
            changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
            gains = [max(change, 0) for change in changes]
            losses = [max(-change, 0) for change in changes]
            
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
            return 50.0
    
    def _determine_recommended_direction(self, signals: Dict[str, float]) -> str:
        """🎯 Determine recommended trading direction from signals"""
        try:
            max_signal = max(signals.values())
            
            # Only recommend if signal is strong enough
            if max_signal < 0.4:
                return 'hold'
            
            return max(signals, key=signals.get)
            
        except Exception as e:
            return 'hold'
    
    def _calculate_trend_direction(self, asset: str) -> str:
        """Calculate overall trend direction"""
        try:
            prices = self.price_histories[asset]
            if len(prices) < 20:
                return 'neutral'
            
            sma_5 = np.mean(prices[-5:])
            sma_20 = np.mean(prices[-20:])
            
            if sma_5 > sma_20 * 1.02:
                return 'bullish'
            elif sma_5 < sma_20 * 0.98:
                return 'bearish'
            else:
                return 'neutral'
                
        except Exception as e:
            return 'neutral'
    
    def _calculate_momentum_direction(self, asset: str) -> str:
        """Calculate momentum direction"""
        try:
            prices = self.price_histories[asset]
            if len(prices) < 10:
                return 'neutral'
            
            recent_change = (prices[-1] - prices[-5]) / prices[-5] * 100 if len(prices) >= 5 else 0
            
            if recent_change > 2:
                return 'positive'
            elif recent_change < -2:
                return 'negative'
            else:
                return 'neutral'
                
        except Exception as e:
            return 'neutral'
    
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
            return self._calculate_rsi_from_prices(prices, period)
        except Exception as e:
            print(f"⚠️ RSI calculation error for {asset}: {e}")
            return 50.0
    
    def _calculate_initial_volatility(self, asset: str) -> float:
        """Calculate initial volatility from historical data"""
        try:
            prices = self.price_histories[asset]
            if len(prices) < 20:
                return 0.01
            
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
        print(f"✅ {asset} directional WebSocket connected")
        self.connection_status[asset] = True
    
    def _on_message(self, asset: str, ws, message):
        """🎯 Enhanced message handler with directional analysis + ML integration"""
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
                
                # 🎯 CALCULATE ENHANCED DIRECTIONAL SIGNALS
                directional_signals = self._calculate_live_directional_signals(asset, current_price, rsi, price_change_24h, volatility, volume_24h)
                
                # 🚀 ENHANCED: ML INTEGRATION
                ml_enhanced_signals = self._integrate_ml_predictions(asset, directional_signals)
                
                # Generate trading recommendation
                trading_rec = self._generate_trading_recommendation(ml_enhanced_signals, current_price)
                position_suggestion = self._calculate_position_suggestion(ml_enhanced_signals)
                risk_level = self._assess_risk_level(asset)
                
                # Update asset data with directional information
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
                    'confidence': self._calculate_confidence(asset),
                    
                    # 🎯 ENHANCED DIRECTIONAL DATA
                    'directional_signals': ml_enhanced_signals,
                    'recommended_direction': self._determine_recommended_direction(ml_enhanced_signals),
                    'signal_strength': max(ml_enhanced_signals.values()),
                    'trend_direction': self._calculate_trend_direction(asset),
                    'momentum_direction': self._calculate_momentum_direction(asset),
                    'cross_asset_correlation': self._get_cross_asset_correlation(asset),
                    
                    # 🚀 ENHANCED: TRADING INTEGRATION
                    'trading_recommendation': trading_rec,
                    'position_suggestion': position_suggestion,
                    'risk_level': risk_level,
                    'ml_enhanced': self.ml_integration is not None
                }
                
                # Store directional signals
                self.directional_signals[asset] = ml_enhanced_signals
                
                # Add to signal history for trend analysis
                self.signal_history[asset].append({
                    'timestamp': datetime.now(),
                    'signals': ml_enhanced_signals.copy(),
                    'price': current_price,
                    'rsi': rsi
                })
                
                # Keep only recent signal history
                if len(self.signal_history[asset]) > 100:
                    self.signal_history[asset] = self.signal_history[asset][-100:]
                
                self.last_updates[asset] = datetime.now()
                
                # Call callback with enhanced data
                self.callback(asset, self.asset_data[asset])
                
        except Exception as e:
            print(f"❌ Message processing error for {asset}: {e}")
    
    def _integrate_ml_predictions(self, asset: str, base_signals: Dict[str, float]) -> Dict[str, float]:
        """🚀 ENHANCED: Integrate ML predictions with technical signals"""
        try:
            if not self.ml_integration:
                return base_signals
            
            # Get ML prediction for this asset if available
            ml_prediction = self.ml_predictions.get(asset, {})
            
            if not ml_prediction:
                return base_signals
            
            # Extract ML directional data
            ml_direction = ml_prediction.get('predicted_direction', 'hold')
            ml_confidence = ml_prediction.get('confidence', 0.5)
            ml_probabilities = ml_prediction.get('direction_probabilities', {})
            
            # Blend technical and ML signals
            blend_factor = min(ml_confidence * 0.7, 0.6)  # Max 60% ML influence
            tech_factor = 1.0 - blend_factor
            
            enhanced_signals = {}
            for direction in ['long', 'short', 'hold']:
                tech_signal = base_signals.get(direction, 0.33)
                ml_signal = ml_probabilities.get(direction, 0.33)
                
                # Weighted blend
                enhanced_signals[direction] = (tech_signal * tech_factor) + (ml_signal * blend_factor)
            
            # Normalize
            total = sum(enhanced_signals.values())
            if total > 0:
                for direction in enhanced_signals:
                    enhanced_signals[direction] /= total
            
            # Log significant ML influence
            if blend_factor > 0.3:
                print(f"   🤖 {asset} ML influence: {ml_direction} ({ml_confidence:.2f}) blend: {blend_factor:.2f}")
            
            return enhanced_signals
            
        except Exception as e:
            print(f"⚠️ ML integration error for {asset}: {e}")
            return base_signals
    
    def update_ml_prediction(self, asset: str, ml_prediction: Dict):
        """🤖 Update ML prediction for specific asset"""
        self.ml_predictions[asset] = ml_prediction
        print(f"🤖 ML prediction updated for {asset}: {ml_prediction.get('predicted_direction', 'unknown')}")
    
    def _calculate_live_directional_signals(self, asset: str, price: float, rsi: float, 
                                          price_change_24h: float, volatility: float, volume_24h: float) -> Dict[str, float]:
        """🎯 Calculate live directional signals with enhanced logic"""
        try:
            long_signal = 0.0
            short_signal = 0.0
            hold_signal = 0.0
            
            # 🎯 RSI-BASED DIRECTIONAL SIGNALS (enhanced for profit on reversals)
            if rsi <= 15:
                long_signal += 0.6  # Very strong oversold = LONG opportunity
            elif rsi <= 25:
                long_signal += 0.45
            elif rsi <= 35:
                long_signal += 0.25
            elif rsi >= 85:
                short_signal += 0.6  # Very strong overbought = SHORT opportunity
            elif rsi >= 75:
                short_signal += 0.45
            elif rsi >= 65:
                short_signal += 0.25
            elif 45 <= rsi <= 55:
                hold_signal += 0.3  # Neutral zone
            
            # 🎯 MOMENTUM-BASED SIGNALS (enhanced for trend following)
            if price_change_24h <= -10:
                long_signal += 0.5  # Strong dip = LONG opportunity (buy the dip)
            elif price_change_24h <= -5:
                long_signal += 0.3
            elif price_change_24h <= -2:
                long_signal += 0.15
            elif price_change_24h >= 15:
                short_signal += 0.5  # Strong rally = SHORT opportunity (sell the top)
            elif price_change_24h >= 8:
                short_signal += 0.3
            elif price_change_24h >= 3:
                short_signal += 0.15
            elif -1 <= price_change_24h <= 1:
                hold_signal += 0.25  # Sideways movement
            
            # 🎯 VOLATILITY-BASED SIGNALS
            if volatility < 0.005:
                hold_signal += 0.3  # Very low volatility = wait for breakout
            elif volatility < 0.01:
                hold_signal += 0.15
            elif volatility > 0.1:
                # Very high volatility = risky but can be profitable
                if rsi < 30 or rsi > 70:  # Only trade high vol with extreme RSI
                    long_signal *= 0.8
                    short_signal *= 0.8
                else:
                    hold_signal += 0.3  # Too risky without extreme RSI
            elif volatility > 0.05:
                # High volatility = proceed with caution
                long_signal *= 0.9
                short_signal *= 0.9
                hold_signal += 0.1
            
            # 🎯 VOLUME CONFIRMATION
            prices = self.price_histories[asset]
            if len(prices) >= 10:
                # Calculate volume trend
                recent_avg_volume = 1000000.0  # Default if no history
                try:
                    recent_volumes = [self.asset_data.get(asset, {}).get('volume_24h', volume_24h) for _ in range(5)]
                    if recent_volumes:
                        recent_avg_volume = np.mean(recent_volumes)
                except:
                    pass
                
                volume_ratio = volume_24h / recent_avg_volume if recent_avg_volume > 0 else 1.0
                
                if volume_ratio > 2.5:  # Very high volume = confirms signals
                    long_signal += 0.15
                    short_signal += 0.15
                elif volume_ratio > 1.5:  # High volume
                    long_signal += 0.1
                    short_signal += 0.1
                elif volume_ratio < 0.5:  # Low volume = wait
                    hold_signal += 0.2
            
            # 🎯 TREND CONFIRMATION
            trend = self._calculate_trend_direction(asset)
            if trend == 'bullish':
                long_signal += 0.15
                short_signal *= 0.85  # Reduce counter-trend signals
            elif trend == 'bearish':
                short_signal += 0.15
                long_signal *= 0.85  # Reduce counter-trend signals
            else:  # neutral
                hold_signal += 0.1
            
            # 🎯 CROSS-ASSET CORRELATION ADJUSTMENT
            correlation_adjustment = self._get_correlation_adjustment(asset)
            long_signal *= correlation_adjustment.get('long', 1.0)
            short_signal *= correlation_adjustment.get('short', 1.0)
            hold_signal *= correlation_adjustment.get('hold', 1.0)
            
            # Normalize to probabilities
            total = long_signal + short_signal + hold_signal
            if total > 0:
                long_signal /= total
                short_signal /= total
                hold_signal /= total
            else:
                long_signal = short_signal = hold_signal = 0.33
            
            # Ensure minimum values and smooth changes
            min_val = 0.08
            long_signal = max(min_val, long_signal)
            short_signal = max(min_val, short_signal)
            hold_signal = max(min_val, hold_signal)
            
            # Re-normalize
            total = long_signal + short_signal + hold_signal
            long_signal /= total
            short_signal /= total
            hold_signal /= total
            
            # Smooth signal changes (prevent erratic switching)
            previous_signals = self.directional_signals.get(asset, {})
            if previous_signals:
                smoothing_factor = 0.7  # 70% new, 30% old
                long_signal = smoothing_factor * long_signal + (1 - smoothing_factor) * previous_signals.get('long', long_signal)
                short_signal = smoothing_factor * short_signal + (1 - smoothing_factor) * previous_signals.get('short', short_signal)
                hold_signal = smoothing_factor * hold_signal + (1 - smoothing_factor) * previous_signals.get('hold', hold_signal)
                
                # Re-normalize after smoothing
                total = long_signal + short_signal + hold_signal
                long_signal /= total
                short_signal /= total
                hold_signal /= total
            
            return {
                'long': float(long_signal),
                'short': float(short_signal),
                'hold': float(hold_signal)
            }
            
        except Exception as e:
            print(f"⚠️ Live directional signals error for {asset}: {e}")
            return self.directional_signals.get(asset, {'long': 0.33, 'short': 0.33, 'hold': 0.34})
    
    def _get_correlation_adjustment(self, asset: str) -> Dict[str, float]:
        """🎯 Get correlation-based signal adjustments"""
        try:
            adjustments = {'long': 1.0, 'short': 1.0, 'hold': 1.0}
            
            # Check correlation with other assets
            for other_asset in self.assets:
                if other_asset == asset:
                    continue
                    
                if other_asset in self.directional_signals:
                    other_signals = self.directional_signals[other_asset]
                    
                    # If other assets are strongly trending in one direction,
                    # reduce confidence in opposite direction for this asset
                    if other_signals['long'] > 0.6:
                        adjustments['short'] *= 0.9
                        adjustments['long'] *= 1.05
                    elif other_signals['short'] > 0.6:
                        adjustments['long'] *= 0.9
                        adjustments['short'] *= 1.05
                    elif other_signals['hold'] > 0.6:
                        adjustments['hold'] *= 1.1
            
            return adjustments
            
        except Exception as e:
            return {'long': 1.0, 'short': 1.0, 'hold': 1.0}
    
    def _get_cross_asset_correlation(self, asset: str) -> Dict[str, float]:
        """Calculate cross-asset correlation"""
        try:
            correlations = {}
            
            for other_asset in self.assets:
                if other_asset == asset:
                    continue
                    
                if other_asset in self.asset_data:
                    other_data = self.asset_data[other_asset]
                    current_data = self.asset_data.get(asset, {})
                    
                    # Simple correlation based on price changes
                    current_change = current_data.get('price_change_24h', 0)
                    other_change = other_data.get('price_change_24h', 0)
                    
                    # Correlation indicator (-1 to 1)
                    if abs(current_change) > 0.1 and abs(other_change) > 0.1:
                        if (current_change > 0) == (other_change > 0):
                            correlations[other_asset] = min(abs(current_change), abs(other_change)) / max(abs(current_change), abs(other_change))
                        else:
                            correlations[other_asset] = -min(abs(current_change), abs(other_change)) / max(abs(current_change), abs(other_change))
                    else:
                        correlations[other_asset] = 0.0
            
            return correlations
            
        except Exception as e:
            return {}
    
    def _calculate_rsi(self, asset: str, period: int = 14) -> float:
        """Calculate RSI for asset"""
        try:
            prices = self.price_histories[asset]
            return self._calculate_rsi_from_prices(prices, period)
        except Exception as e:
            print(f"⚠️ RSI error for {asset}: {e}")
            return 50.0
    
    def _calculate_volatility(self, asset: str) -> float:
        """Calculate volatility for asset"""
        try:
            prices = self.price_histories[asset]
            if len(prices) < 20:
                return 0.01
            
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
        """🎯 Calculate enhanced trading confidence with directional factors"""
        try:
            if asset not in self.asset_data:
                return 0.5
            
            data = self.asset_data[asset]
            confidence = 0.5
            
            # RSI factor
            rsi = data.get('rsi', 50)
            if 25 <= rsi <= 75:
                confidence += 0.2
            elif rsi < 20 or rsi > 80:
                confidence += 0.25  # Extreme RSI = high confidence for reversal
            
            # Volatility factor
            volatility = data.get('volatility', 0.01)
            if 0.01 <= volatility <= 0.04:
                confidence += 0.2
            elif volatility > 0.08:
                confidence -= 0.15  # Very high volatility reduces confidence
            
            # Volume factor
            volume = data.get('volume_24h', 0)
            if volume > 1000000:
                confidence += 0.1
            
            # Price momentum factor
            price_change = data.get('price_change_24h', 0)
            if abs(price_change) < 3:
                confidence += 0.1  # Stable price
            elif abs(price_change) > 12:
                confidence -= 0.1  # Very volatile
            
            # 🎯 DIRECTIONAL SIGNAL FACTOR
            signals = data.get('directional_signals', {})
            if signals:
                max_signal = max(signals.values())
                if max_signal > 0.6:
                    confidence += 0.15  # Strong directional signal
                elif max_signal < 0.4:
                    confidence -= 0.1   # Weak/unclear signals
            
            # Trend confirmation factor
            trend = data.get('trend_direction', 'neutral')
            momentum = data.get('momentum_direction', 'neutral')
            if trend != 'neutral' and momentum != 'neutral':
                if (trend == 'bullish' and momentum == 'positive') or (trend == 'bearish' and momentum == 'negative'):
                    confidence += 0.1  # Trend and momentum aligned
            
            # 🚀 ML enhancement factor
            if data.get('ml_enhanced', False):
                confidence += 0.05  # Bonus for ML integration
            
            return max(0.1, min(0.95, confidence))
            
        except Exception as e:
            return 0.5
    
    def _continuous_directional_analysis(self):
        """🎯 Continuous background analysis for directional signals"""
        while self.running:
            try:
                time.sleep(30)  # Analyze every 30 seconds
                
                # Update correlation matrix
                self._update_correlation_matrix()
                
                # Analyze signal trends
                self._analyze_signal_trends()
                
                # Log directional summary
                self._log_directional_summary()
                
            except Exception as e:
                print(f"⚠️ Continuous analysis error: {e}")
                time.sleep(10)
    
    def _update_correlation_matrix(self):
        """Update cross-asset correlation matrix"""
        try:
            for asset1 in self.assets:
                for asset2 in self.assets:
                    if asset1 != asset2 and asset1 in self.asset_data and asset2 in self.asset_data:
                        corr = self._calculate_asset_correlation(asset1, asset2)
                        self.correlation_matrix[f"{asset1}-{asset2}"] = corr
        except Exception as e:
            print(f"⚠️ Correlation matrix update error: {e}")
    
    def _calculate_asset_correlation(self, asset1: str, asset2: str) -> float:
        """Calculate correlation between two assets"""
        try:
            if asset1 not in self.signal_history or asset2 not in self.signal_history:
                return 0.0
            
            history1 = self.signal_history[asset1][-20:]  # Last 20 signals
            history2 = self.signal_history[asset2][-20:]
            
            if len(history1) < 10 or len(history2) < 10:
                return 0.0
            
            # Compare recommended directions
            matches = 0
            total = min(len(history1), len(history2))
            
            for i in range(total):
                signals1 = history1[i]['signals']
                signals2 = history2[i]['signals']
                
                direction1 = max(signals1, key=signals1.get)
                direction2 = max(signals2, key=signals2.get)
                
                if direction1 == direction2:
                    matches += 1
            
            return matches / total if total > 0 else 0.0
            
        except Exception as e:
            return 0.0
    
    def _analyze_signal_trends(self):
        """Analyze directional signal trends"""
        try:
            for asset in self.assets:
                if asset not in self.signal_history:
                    continue
                
                history = self.signal_history[asset][-10:]  # Last 10 signals
                
                if len(history) < 5:
                    continue
                
                # Analyze trend in each direction
                long_trend = [h['signals']['long'] for h in history]
                short_trend = [h['signals']['short'] for h in history]
                hold_trend = [h['signals']['hold'] for h in history]
                
                # Simple trend analysis (increasing/decreasing)
                long_increasing = long_trend[-1] > long_trend[0]
                short_increasing = short_trend[-1] > short_trend[0]
                hold_increasing = hold_trend[-1] > hold_trend[0]
                
                # Store trend information
                if asset in self.asset_data:
                    self.asset_data[asset]['signal_trends'] = {
                        'long_increasing': long_increasing,
                        'short_increasing': short_increasing,
                        'hold_increasing': hold_increasing
                    }
                    
        except Exception as e:
            print(f"⚠️ Signal trend analysis error: {e}")
    
    def _log_directional_summary(self):
        """Log periodic directional summary with trading focus"""
        try:
            print(f"\n🎯 DIRECTIONAL TRADING SUMMARY ({datetime.now().strftime('%H:%M:%S')})")
            
            best_long = None
            best_short = None
            best_long_score = 0
            best_short_score = 0
            
            for asset in self.assets:
                if asset in self.asset_data:
                    data = self.asset_data[asset]
                    signals = data.get('directional_signals', {})
                    recommended = data.get('recommended_direction', 'hold')
                    strength = data.get('signal_strength', 0)
                    position_suggestion = data.get('position_suggestion', 'WAIT')
                    price = data.get('price', 0)
                    rsi = data.get('rsi', 50)
                    
                    direction_emoji = {'long': '🟢', 'short': '🔴', 'hold': '⚪'}.get(recommended, '❓')
                    
                    print(f"   {direction_emoji} {asset}: {recommended.upper()} ({strength:.2f}) → {position_suggestion} - "
                          f"${price:.4f}, RSI: {rsi:.1f}")
                    print(f"      L:{signals.get('long', 0):.2f} S:{signals.get('short', 0):.2f} H:{signals.get('hold', 0):.2f}")
                    
                    # Track best opportunities
                    if signals.get('long', 0) > best_long_score:
                        best_long_score = signals.get('long', 0)
                        best_long = asset
                    
                    if signals.get('short', 0) > best_short_score:
                        best_short_score = signals.get('short', 0)
                        best_short = asset
            
            # Trading recommendations
            if best_long and best_long_score > 0.6:
                print(f"   🚀 BEST LONG OPPORTUNITY: {best_long} ({best_long_score:.2f})")
            
            if best_short and best_short_score > 0.6:
                print(f"   📉 BEST SHORT OPPORTUNITY: {best_short} ({best_short_score:.2f})")
            
            # Correlation summary
            if self.correlation_matrix:
                high_corr = {k: v for k, v in self.correlation_matrix.items() if abs(v) > 0.7}
                if high_corr:
                    print(f"   📊 High correlations: {high_corr}")
            
        except Exception as e:
            print(f"⚠️ Directional summary error: {e}")
    
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
        print("🛑 Stopping directional multi-asset tracking...")
        self.running = False
        
        for asset, ws in self.connections.items():
            try:
                ws.close()
                print(f"✅ {asset} stream stopped")
            except:
                pass
        
        self.connections.clear()
        self.connection_status = {asset: False for asset in self.assets}
        print("✅ All directional asset streams stopped")
    
    # 🚀 PUBLIC API FOR BOT INTEGRATION
    def get_connection_status(self) -> Dict[str, bool]:
        """Get connection status for all assets"""
        return self.connection_status.copy()
    
    def get_asset_data(self, asset: str) -> Optional[Dict]:
        """Get current data for specific asset"""
        return self.asset_data.get(asset)
    
    def get_all_asset_data(self) -> Dict[str, Dict]:
        """Get current data for all assets"""
        return self.asset_data.copy()
    
    def get_directional_signals(self, asset: str = None) -> Dict:
        """🎯 Get directional signals for asset(s)"""
        if asset:
            return self.directional_signals.get(asset, {})
        else:
            return self.directional_signals.copy()
    
    def get_best_directional_opportunity(self) -> Optional[Tuple[str, str, float]]:
        """🎯 Get the best directional trading opportunity"""
        try:
            best_asset = None
            best_direction = None
            best_strength = 0.0
            
            for asset, signals in self.directional_signals.items():
                for direction, strength in signals.items():
                    if direction != 'hold' and strength > best_strength and strength > 0.5:
                        best_strength = strength
                        best_direction = direction
                        best_asset = asset
            
            return (best_asset, best_direction, best_strength) if best_asset else None
            
        except Exception as e:
            print(f"⚠️ Best opportunity error: {e}")
            return None
    
    def get_trading_recommendation(self, asset: str) -> Optional[Dict]:
        """🚀 Get concrete trading recommendation for asset"""
        if asset in self.asset_data:
            return self.asset_data[asset].get('trading_recommendation')
        return None
    
    def get_position_suggestion(self, asset: str) -> str:
        """🎯 Get position suggestion for asset"""
        if asset in self.asset_data:
            return self.asset_data[asset].get('position_suggestion', 'WAIT')
        return 'WAIT'
    
    def get_correlation_matrix(self) -> Dict[str, float]:
        """Get cross-asset correlation matrix"""
        return self.correlation_matrix.copy()


class DirectionalMultiAssetSignals:
    """🎯 Enhanced multi-asset signal analysis for directional trading"""
    
    def __init__(self):
        self.correlation_window = 50
        self.signal_weights = {
            'rsi': 0.25,
            'volatility': 0.15, 
            'momentum': 0.3,
            'volume': 0.15,
            'technical': 0.15
        }
    
    def analyze_directional_multi_asset_conditions(self, asset_data: Dict[str, Dict]) -> Dict[str, Dict]:
        """🎯 Analyze directional conditions across all assets"""
        signals = {}
        
        try:
            for asset, data in asset_data.items():
                signals[asset] = self._analyze_single_asset_directional(asset, data, asset_data)
            
            # Add cross-asset directional analysis
            self._add_cross_asset_directional_analysis(signals, asset_data)
            
            return signals
            
        except Exception as e:
            print(f"❌ Directional multi-asset analysis error: {e}")
            return {}
    
    def _analyze_single_asset_directional(self, asset: str, data: Dict, all_data: Dict[str, Dict]) -> Dict:
        """🎯 Analyze directional signals for single asset"""
        try:
            signal = {
                'asset': asset,
                'recommended_direction': 'hold',
                'direction_scores': {'long': 0.0, 'short': 0.0, 'hold': 0.0},
                'confidence': 0.5,
                'reasons': [],
                'trading_action': 'WAIT'
            }
            
            # Get existing directional signals if available
            if 'directional_signals' in data:
                existing_signals = data['directional_signals']
                signal['direction_scores'] = existing_signals
                signal['recommended_direction'] = max(existing_signals, key=existing_signals.get)
                signal['confidence'] = max(existing_signals.values())
                
                # Map to trading action
                if signal['recommended_direction'] == 'long' and signal['confidence'] > 0.6:
                    signal['trading_action'] = 'OPEN_LONG'
                elif signal['recommended_direction'] == 'short' and signal['confidence'] > 0.6:
                    signal['trading_action'] = 'OPEN_SHORT'
                else:
                    signal['trading_action'] = 'HOLD_POSITION'
            else:
                # Calculate from scratch
                signal = self._calculate_directional_scores(asset, data, signal)
            
            # Add reasoning
            signal['reasons'] = self._generate_directional_reasoning(data, signal['direction_scores'])
            
            return signal
            
        except Exception as e:
            print(f"⚠️ Single asset directional analysis error for {asset}: {e}")
            return {
                'asset': asset,
                'recommended_direction': 'hold',
                'direction_scores': {'long': 0.33, 'short': 0.33, 'hold': 0.34},
                'confidence': 0.3,
                'reasons': [f'Analysis error: {e}'],
                'trading_action': 'WAIT'
            }
    
    def _calculate_directional_scores(self, asset: str, data: Dict, signal: Dict) -> Dict:
        """Calculate directional scores from market data"""
        try:
            rsi = data.get('rsi', 50)
            price_change_24h = data.get('price_change_24h', 0)
            volatility = data.get('volatility', 0.02)
            volume_24h = data.get('volume_24h', 0)
            
            long_score = 0.0
            short_score = 0.0
            hold_score = 0.0
            
            # RSI analysis for directional trading
            if rsi < 20:
                long_score += 0.5  # Strong oversold = LONG
            elif rsi < 30:
                long_score += 0.3
            elif rsi > 80:
                short_score += 0.5  # Strong overbought = SHORT
            elif rsi > 70:
                short_score += 0.3
            elif 40 <= rsi <= 60:
                hold_score += 0.3
            
            # Momentum analysis for trend following
            if price_change_24h < -8:
                long_score += 0.4  # Strong dip = buy opportunity
            elif price_change_24h < -3:
                long_score += 0.2
            elif price_change_24h > 10:
                short_score += 0.4  # Strong rally = short opportunity
            elif price_change_24h > 5:
                short_score += 0.2
            elif -1 <= price_change_24h <= 1:
                hold_score += 0.2
            
            # Volatility analysis
            if volatility < 0.01:
                hold_score += 0.2
            elif volatility > 0.05:
                hold_score += 0.1
            
            # Normalize scores
            total = long_score + short_score + hold_score
            if total > 0:
                signal['direction_scores'] = {
                    'long': long_score / total,
                    'short': short_score / total,
                    'hold': hold_score / total
                }
            else:
                signal['direction_scores'] = {'long': 0.33, 'short': 0.33, 'hold': 0.34}
            
            signal['recommended_direction'] = max(signal['direction_scores'], key=signal['direction_scores'].get)
            signal['confidence'] = max(signal['direction_scores'].values())
            
            # Set trading action
            if signal['recommended_direction'] == 'long' and signal['confidence'] > 0.6:
                signal['trading_action'] = 'OPEN_LONG'
            elif signal['recommended_direction'] == 'short' and signal['confidence'] > 0.6:
                signal['trading_action'] = 'OPEN_SHORT'
            else:
                signal['trading_action'] = 'HOLD_POSITION'
            
            return signal
            
        except Exception as e:
            signal['direction_scores'] = {'long': 0.33, 'short': 0.33, 'hold': 0.34}
            signal['recommended_direction'] = 'hold'
            signal['confidence'] = 0.3
            signal['trading_action'] = 'WAIT'
            return signal
    
    def _generate_directional_reasoning(self, data: Dict, scores: Dict[str, float]) -> List[str]:
        """Generate reasoning for directional recommendation"""
        reasons = []
        
        try:
            rsi = data.get('rsi', 50)
            price_change_24h = data.get('price_change_24h', 0)
            volatility = data.get('volatility', 0.02)
            
            # RSI reasoning
            if rsi < 25:
                reasons.append(f"Strong oversold RSI ({rsi:.1f}) → LONG opportunity")
            elif rsi > 75:
                reasons.append(f"Strong overbought RSI ({rsi:.1f}) → SHORT opportunity")
            elif 40 <= rsi <= 60:
                reasons.append(f"Neutral RSI ({rsi:.1f}) → HOLD position")
            
            # Momentum reasoning
            if price_change_24h < -5:
                reasons.append(f"Strong decline ({price_change_24h:.1f}%) → Buy the dip")
            elif price_change_24h > 8:
                reasons.append(f"Strong rally ({price_change_24h:.1f}%) → Short the top")
            
            # Volatility reasoning
            if volatility < 0.01:
                reasons.append("Low volatility → Wait for breakout")
            elif volatility > 0.05:
                reasons.append("High volatility → Increased risk")
            
            # Add top direction reasoning
            top_direction = max(scores, key=scores.get)
            top_score = scores[top_direction]
            
            if top_score > 0.6:
                reasons.append(f"Strong {top_direction.upper()} signal ({top_score:.2f})")
            elif top_score < 0.4:
                reasons.append(f"Weak signals → Default to HOLD")
            
            return reasons[:3]  # Return top 3 reasons
            
        except Exception as e:
            return [f"Reasoning error: {e}"]
    
    def _add_cross_asset_directional_analysis(self, signals: Dict[str, Dict], asset_data: Dict[str, Dict]):
        """🎯 Add cross-asset directional analysis"""
        try:
            assets = list(asset_data.keys())
            
            if len(assets) < 2:
                return
            
            # Analyze directional consensus
            directional_consensus = {'long': 0, 'short': 0, 'hold': 0}
            
            for asset, signal in signals.items():
                recommended = signal.get('recommended_direction', 'hold')
                directional_consensus[recommended] += 1
            
            # Calculate consensus strength
            total_assets = len(assets)
            consensus_direction = max(directional_consensus, key=directional_consensus.get)
            consensus_strength = directional_consensus[consensus_direction] / total_assets
            
            # Apply consensus adjustment to individual signals
            for asset, signal in signals.items():
                try:
                    current_direction = signal.get('recommended_direction', 'hold')
                    current_confidence = signal.get('confidence', 0.5)
                    
                    # Boost confidence if aligned with consensus
                    if current_direction == consensus_direction and consensus_strength > 0.6:
                        signal['confidence'] = min(current_confidence * 1.15, 0.95)
                        signal['reasons'].append(f"Cross-asset consensus supports {consensus_direction.upper()}")
                    
                    # Add consensus information
                    signal['cross_asset_consensus'] = {
                        'direction': consensus_direction,
                        'strength': consensus_strength,
                        'breakdown': directional_consensus
                    }
                    
                except Exception as e:
                    continue
                    
        except Exception as e:
            print(f"⚠️ Cross-asset directional analysis error: {e}")
    
    def get_best_directional_asset_to_trade(self, signals: Dict[str, Dict]) -> Optional[str]:
        """🎯 Get the best asset for directional trading"""
        try:
            if not signals:
                return None
            
            best_asset = None
            best_score = 0
            
            for asset, signal in signals.items():
                confidence = signal.get('confidence', 0)
                direction = signal.get('recommended_direction', 'hold')
                
                # Prioritize non-HOLD directions with high confidence
                if direction != 'hold' and confidence > best_score:
                    best_score = confidence
                    best_asset = asset
            
            # If no strong directional signals, get highest confidence overall
            if not best_asset:
                for asset, signal in signals.items():
                    confidence = signal.get('confidence', 0)
                    if confidence > best_score:
                        best_score = confidence
                        best_asset = asset
            
            return best_asset
            
        except Exception as e:
            print(f"⚠️ Best directional asset selection error: {e}")
            return None
    
    def get_directional_portfolio_recommendations(self, signals: Dict[str, Dict], 
                                                current_allocation: Dict[str, float],
                                                target_allocation: Dict[str, float]) -> Dict[str, str]:
        """🎯 Get directional portfolio rebalancing recommendations"""
        try:
            recommendations = {}
            
            for asset in signals.keys():
                signal = signals[asset]
                current_pct = current_allocation.get(asset, 0)
                target_pct = target_allocation.get(asset, 0)
                confidence = signal.get('confidence', 0.5)
                direction = signal.get('recommended_direction', 'hold')
                trading_action = signal.get('trading_action', 'WAIT')
                
                deficit = target_pct - current_pct
                
                # Directional-based recommendations
                if trading_action == 'OPEN_LONG':
                    recommendations[asset] = 'INCREASE_LONG_POSITION'
                elif trading_action == 'OPEN_SHORT':
                    recommendations[asset] = 'INCREASE_SHORT_POSITION'
                elif trading_action == 'HOLD_POSITION':
                    recommendations[asset] = 'MAINTAIN_POSITION'
                elif deficit > 0.1 and confidence > 0.5:
                    recommendations[asset] = 'REBALANCE_UP'
                elif deficit < -0.1:
                    recommendations[asset] = 'REBALANCE_DOWN'
                else:
                    recommendations[asset] = 'NO_ACTION'
            
            return recommendations
            
        except Exception as e:
            print(f"⚠️ Directional portfolio recommendations error: {e}")
            return {}


def create_directional_multi_asset_service(assets: List[str], callback: Callable[[str, Dict], None]) -> Optional[DirectionalMultiAssetData]:
    """🎯 Create and start directional multi-asset market data service"""
    try:
        print(f"🎯 Creating directional multi-asset service for: {assets}")
        
        service = DirectionalMultiAssetData(assets, callback)
        
        if service.start_tracking():
            print("✅ Directional multi-asset service created successfully")
            return service
        else:
            print("❌ Failed to create directional multi-asset service")
            return None
            
    except Exception as e:
        print(f"❌ Directional multi-asset service creation error: {e}")
        return None


# Legacy compatibility aliases
MultiAssetMarketData = DirectionalMultiAssetData
MultiAssetSignals = DirectionalMultiAssetSignals
create_multi_asset_service = create_directional_multi_asset_service