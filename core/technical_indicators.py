# core/technical_indicators.py
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta


class TechnicalIndicators:
    """
    Professional technical indicators implementation using pandas/numpy
    No external TA libraries required - pure Python calculations
    """

    @staticmethod
    def sma(prices: List[float], period: int) -> float:
        """Simple Moving Average"""
        if len(prices) < period:
            return 0.0
        return np.mean(prices[-period:])

    @staticmethod
    def ema(prices: List[float], period: int) -> float:
        """Exponential Moving Average"""
        if len(prices) < period:
            return 0.0

        prices_array = np.array(prices)
        alpha = 2.0 / (period + 1.0)
        ema_values = np.zeros_like(prices_array)
        ema_values[0] = prices_array[0]

        for i in range(1, len(prices_array)):
            ema_values[i] = alpha * prices_array[i] + \
                (1 - alpha) * ema_values[i - 1]

        return float(ema_values[-1])

    @staticmethod
    def rsi(prices: List[float], period: int = 14) -> float:
        """Relative Strength Index"""
        if len(prices) < period + 1:
            return 50.0

        prices_series = pd.Series(prices)
        delta = prices_series.diff()

        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0

    @staticmethod
    def macd(prices: List[float], fast: int = 12,
             slow: int = 26, signal: int = 9) -> Dict[str, float]:
        """MACD (Moving Average Convergence Divergence)"""
        if len(prices) < slow:
            return {'macd': 0.0, 'signal': 0.0, 'histogram': 0.0}

        ema_fast = TechnicalIndicators.ema(prices, fast)
        ema_slow = TechnicalIndicators.ema(prices, slow)
        macd_line = ema_fast - ema_slow

        # Calculate signal line (EMA of MACD)
        if len(prices) < slow + signal:
            signal_line = macd_line
        else:
            # For simplicity, use SMA instead of EMA for signal line
            macd_values = []
            for i in range(slow, len(prices) + 1):
                ema_f = TechnicalIndicators.ema(prices[:i], fast)
                ema_s = TechnicalIndicators.ema(prices[:i], slow)
                macd_values.append(ema_f - ema_s)

            signal_line = TechnicalIndicators.sma(macd_values, signal)

        histogram = macd_line - signal_line

        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }

    @staticmethod
    def bollinger_bands(
            prices: List[float], period: int = 20, std_dev: float = 2.0) -> Dict[str, float]:
        """Bollinger Bands"""
        if len(prices) < period:
            current_price = prices[-1] if prices else 0.0
            return {
                'upper': current_price * 1.02,
                'middle': current_price,
                'lower': current_price * 0.98
            }

        sma = TechnicalIndicators.sma(prices, period)
        std = np.std(prices[-period:])

        return {
            'upper': sma + (std * std_dev),
            'middle': sma,
            'lower': sma - (std * std_dev)
        }

    @staticmethod
    def stochastic_oscillator(highs: List[float], lows: List[float], closes: List[float],
                              k_period: int = 14, d_period: int = 3) -> Dict[str, float]:
        """Stochastic Oscillator %K and %D"""
        if len(closes) < k_period:
            return {'%K': 50.0, '%D': 50.0}

        # Calculate %K
        recent_high = max(highs[-k_period:])
        recent_low = min(lows[-k_period:])
        current_close = closes[-1]

        if recent_high == recent_low:
            k_percent = 50.0
        else:
            k_percent = ((current_close - recent_low) /
                         (recent_high - recent_low)) * 100

        # Calculate %D (SMA of %K)
        # For simplicity, return %K as %D if we don't have enough data
        d_percent = k_percent  # In real implementation, this would be SMA of %K values

        return {'%K': k_percent, '%D': d_percent}

    @staticmethod
    def atr(highs: List[float], lows: List[float],
            closes: List[float], period: int = 14) -> float:
        """Average True Range"""
        if len(closes) < 2:
            return 0.0

        true_ranges = []
        for i in range(1, len(closes)):
            high_low = highs[i] - lows[i]
            high_close_prev = abs(highs[i] - closes[i - 1])
            low_close_prev = abs(lows[i] - closes[i - 1])

            true_range = max(high_low, high_close_prev, low_close_prev)
            true_ranges.append(true_range)

        if len(true_ranges) < period:
            return np.mean(true_ranges) if true_ranges else 0.0

        return np.mean(true_ranges[-period:])

    @staticmethod
    def volume_profile(
            prices: List[float], volumes: List[float], bins: int = 20) -> Dict:
        """Volume Profile Analysis"""
        if len(prices) != len(volumes) or len(prices) < bins:
            return {'vwap': prices[-1] if prices else 0.0,
                    'poc': prices[-1] if prices else 0.0}

        # Calculate VWAP (Volume Weighted Average Price)
        total_volume = sum(volumes)
        if total_volume == 0:
            vwap = np.mean(prices)
        else:
            vwap = sum(p * v for p, v in zip(prices, volumes)) / total_volume

        # Find Point of Control (POC) - price level with highest volume
        price_min, price_max = min(prices), max(prices)
        price_ranges = np.linspace(price_min, price_max, bins)
        volume_at_price = [0] * (bins - 1)

        for i, price in enumerate(prices):
            bin_index = min(
                int((price - price_min) / (price_max - price_min) * (bins - 1)), bins - 2)
            volume_at_price[bin_index] += volumes[i]

        poc_index = volume_at_price.index(max(volume_at_price))
        poc = (price_ranges[poc_index] + price_ranges[poc_index + 1]) / 2

        return {'vwap': vwap, 'poc': poc}


class SignalGenerator:
    """
    Advanced trading signal generation using technical indicators
    """

    def __init__(self):
        self.indicators = TechnicalIndicators()

    def analyze_trend(self, prices: List[float]) -> Dict[str, any]:
        """Comprehensive trend analysis"""
        if len(prices) < 50:
            return {'trend': 'neutral', 'strength': 0.0, 'confidence': 0.0}

        # Calculate multiple timeframe SMAs
        sma_10 = self.indicators.sma(prices, 10)
        sma_20 = self.indicators.sma(prices, 20)
        sma_50 = self.indicators.sma(prices, 50)

        current_price = prices[-1]

        # Trend determination
        trend_score = 0
        reasons = []

        # SMA alignment
        if sma_10 > sma_20 > sma_50:
            trend_score += 3
            reasons.append("SMA alignment bullish (10>20>50)")
        elif sma_10 < sma_20 < sma_50:
            trend_score -= 3
            reasons.append("SMA alignment bearish (10<20<50)")

        # Price vs SMAs
        if current_price > sma_20:
            trend_score += 1
            reasons.append("Price above SMA20")
        else:
            trend_score -= 1
            reasons.append("Price below SMA20")

        # Recent momentum (last 5 periods)
        if len(prices) >= 5:
            recent_change = (prices[-1] - prices[-5]) / prices[-5] * 100
            if recent_change > 1.0:
                trend_score += 1
                reasons.append(
                    f"Strong recent momentum (+{recent_change:.1f}%)")
            elif recent_change < -1.0:
                trend_score -= 1
                reasons.append(f"Weak recent momentum ({recent_change:.1f}%)")

        # Determine final trend
        if trend_score >= 2:
            trend = 'bullish'
        elif trend_score <= -2:
            trend = 'bearish'
        else:
            trend = 'neutral'

        strength = min(abs(trend_score) / 5.0, 1.0)
        confidence = strength * 0.8  # Conservative confidence

        return {
            'trend': trend,
            'strength': strength,
            'confidence': confidence,
            'score': trend_score,
            'reasons': reasons,
            'sma_10': sma_10,
            'sma_20': sma_20,
            'sma_50': sma_50
        }

    def generate_signals(self, market_data: Dict) -> Dict[str, any]:
        """Generate comprehensive trading signals"""

        signals = {
            'action': 'hold',
            'confidence': 0.0,
            'strength': 0.0,
            'reasons': [],
            'indicators': {},
            'risk_level': 'medium'
        }

        try:
            # Get price history from market data
            price_history = market_data.get('price_history', [])
            current_price = market_data.get('price', 0.0)

            if len(price_history) < 20:
                signals['reasons'].append("Insufficient data for analysis")
                return signals

            # Calculate all indicators
            rsi = self.indicators.rsi(price_history)
            macd_data = self.indicators.macd(price_history)
            bb_data = self.indicators.bollinger_bands(price_history)
            trend_analysis = self.analyze_trend(price_history)

            # Store indicators for dashboard
            signals['indicators'] = {
                'rsi': rsi,
                'macd': macd_data,
                'bollinger': bb_data,
                'trend': trend_analysis
            }

            # Signal generation logic
            signal_score = 0

            # RSI signals
            if rsi < 30:
                signal_score += 2
                signals['reasons'].append(f"RSI oversold ({rsi:.1f})")
            elif rsi > 70:
                signal_score -= 2
                signals['reasons'].append(f"RSI overbought ({rsi:.1f})")
            elif 40 <= rsi <= 60:
                signal_score += 0.5
                signals['reasons'].append("RSI neutral - good for entry")

            # MACD signals
            if macd_data['histogram'] > 0 and macd_data['macd'] > macd_data['signal']:
                signal_score += 1.5
                signals['reasons'].append("MACD bullish crossover")
            elif macd_data['histogram'] < 0 and macd_data['macd'] < macd_data['signal']:
                signal_score -= 1.5
                signals['reasons'].append("MACD bearish crossover")

            # Bollinger Bands signals
            if current_price <= bb_data['lower']:
                signal_score += 1
                signals['reasons'].append("Price at lower Bollinger Band")
            elif current_price >= bb_data['upper']:
                signal_score -= 1
                signals['reasons'].append("Price at upper Bollinger Band")

            # Trend signals
            if trend_analysis['trend'] == 'bullish' and trend_analysis['confidence'] > 0.6:
                signal_score += 1
                signals['reasons'].append("Strong bullish trend")
            elif trend_analysis['trend'] == 'bearish' and trend_analysis['confidence'] > 0.6:
                signal_score -= 1
                signals['reasons'].append("Strong bearish trend")

            # Market volatility check
            volatility = market_data.get('volatility', 0.01)
            if volatility > 0.05:  # High volatility
                signal_score *= 0.7  # Reduce confidence
                signals['risk_level'] = 'high'
                signals['reasons'].append(
                    "High volatility - reduced confidence")
            elif volatility < 0.01:  # Low volatility
                signals['risk_level'] = 'low'

            # Final signal determination
            signals['strength'] = min(abs(signal_score) / 5.0, 1.0)
            signals['confidence'] = signals['strength'] * 0.8

            if signal_score >= 2.5:
                signals['action'] = 'strong_buy'
            elif signal_score >= 1.0:
                signals['action'] = 'buy'
            elif signal_score <= -2.5:
                signals['action'] = 'strong_sell'
            elif signal_score <= -1.0:
                signals['action'] = 'sell'
            else:
                signals['action'] = 'hold'

        except Exception as e:
            print(f"⚠️ Error generating signals: {e}")
            signals['reasons'].append(f"Analysis error: {str(e)}")

        return signals

# Helper class for data management


class MarketDataManager:
    """Manages historical market data for technical analysis"""

    def __init__(self, max_history: int = 200):
        self.max_history = max_history
        self.price_history = []
        self.volume_history = []
        self.high_history = []
        self.low_history = []
        self.timestamp_history = []

    def add_data_point(self, price: float, volume: float = 0.0,
                       high: float = None, low: float = None,
                       timestamp: datetime = None):
        """Add new market data point"""
        self.price_history.append(price)
        self.volume_history.append(volume)
        self.high_history.append(high or price)
        self.low_history.append(low or price)
        self.timestamp_history.append(timestamp or datetime.now())

        # Maintain max history length
        if len(self.price_history) > self.max_history:
            self.price_history = self.price_history[-self.max_history:]
            self.volume_history = self.volume_history[-self.max_history:]
            self.high_history = self.high_history[-self.max_history:]
            self.low_history = self.low_history[-self.max_history:]
            self.timestamp_history = self.timestamp_history[-self.max_history:]

    def get_current_indicators(self) -> Dict:
        """Get all current technical indicators"""
        if len(self.price_history) < 10:
            return {}

        indicators = TechnicalIndicators()

        return {
            'sma_10': indicators.sma(self.price_history, 10),
            'sma_20': indicators.sma(self.price_history, 20),
            'sma_50': indicators.sma(self.price_history, 50),
            'ema_12': indicators.ema(self.price_history, 12),
            'ema_26': indicators.ema(self.price_history, 26),
            'rsi': indicators.rsi(self.price_history),
            'macd': indicators.macd(self.price_history),
            'bollinger': indicators.bollinger_bands(self.price_history),
            'atr': indicators.atr(self.high_history, self.low_history, self.price_history),
            'stochastic': indicators.stochastic_oscillator(
                self.high_history, self.low_history, self.price_history
            ),
            'volume_profile': indicators.volume_profile(self.price_history, self.volume_history)
        }

    def get_data_for_analysis(self) -> Dict:
        """Get formatted data for signal analysis"""
        return {
            'price_history': self.price_history.copy(),
            'volume_history': self.volume_history.copy(),
            'high_history': self.high_history.copy(),
            'low_history': self.low_history.copy(),
            'current_price': self.price_history[-1] if self.price_history else 0.0,
            'data_points': len(self.price_history)
        }
