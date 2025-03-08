#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import random
import statistics
import time
import anthropic
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import logging
import os
import warnings
warnings.filterwarnings("ignore")

# Local imports
from utils.logger import logger
from config import config

class TechnicalIndicators:
    """Class for calculating technical indicators"""
    
    @staticmethod
    def calculate_rsi(prices: List[float], period: int = 14) -> float:
        """
        Calculate Relative Strength Index
        """
        if len(prices) < period + 1:
            return 50.0  # Default to neutral if not enough data
            
        # Calculate price changes
        deltas = np.diff(prices)
        
        # Get gains and losses
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Initial average gain and loss
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        # Calculate for remaining periods
        for i in range(period, len(deltas)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            
        # Calculate RS and RSI
        if avg_loss == 0:
            return 100.0
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def calculate_macd(prices: List[float], fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[float, float, float]:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        Returns (macd_line, signal_line, histogram)
        """
        if len(prices) < slow_period + signal_period:
            return 0.0, 0.0, 0.0  # Default if not enough data
            
        # Convert to numpy array for efficiency
        prices_array = np.array(prices)
        
        # Calculate EMAs
        ema_fast = TechnicalIndicators.calculate_ema(prices_array, fast_period)
        ema_slow = TechnicalIndicators.calculate_ema(prices_array, slow_period)
        
        # Calculate MACD line
        macd_line = ema_fast - ema_slow
        
        # Calculate Signal line (EMA of MACD line)
        signal_line = TechnicalIndicators.calculate_ema(macd_line, signal_period)
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        return macd_line[-1], signal_line[-1], histogram[-1]
    
    @staticmethod
    def calculate_ema(values: np.ndarray, period: int) -> np.ndarray:
        """
        Calculate Exponential Moving Average
        """
        if len(values) < 2 * period:
            # Pad with the first value if we don't have enough data
            padding = np.full(2 * period - len(values), values[0])
            values = np.concatenate((padding, values))
            
        alpha = 2 / (period + 1)
        
        # Calculate EMA
        ema = np.zeros_like(values)
        ema[0] = values[0]  # Initialize with first value
        
        for i in range(1, len(values)):
            ema[i] = alpha * values[i] + (1 - alpha) * ema[i-1]
            
        return ema
    
    @staticmethod
    def calculate_bollinger_bands(prices: List[float], period: int = 20, num_std: float = 2.0) -> Tuple[float, float, float]:
        """
        Calculate Bollinger Bands
        Returns (upper_band, middle_band, lower_band)
        """
        if len(prices) < period:
            return prices[-1] + num_std * 0.02 * prices[-1], prices[-1], prices[-1] - num_std * 0.02 * prices[-1]
            
        # Calculate middle band (SMA)
        middle_band = sum(prices[-period:]) / period
        
        # Calculate standard deviation
        std = statistics.stdev(prices[-period:])
        
        # Calculate upper and lower bands
        upper_band = middle_band + (std * num_std)
        lower_band = middle_band - (std * num_std)
        
        return upper_band, middle_band, lower_band
    
    @staticmethod
    def calculate_stochastic_oscillator(prices: List[float], highs: List[float], lows: List[float], k_period: int = 14, d_period: int = 3) -> Tuple[float, float]:
        """
        Calculate Stochastic Oscillator
        Returns (%K, %D)
        """
        if len(prices) < k_period or len(highs) < k_period or len(lows) < k_period:
            return 50.0, 50.0  # Default to mid-range if not enough data
            
        # Get last k_period prices, highs, lows
        recent_prices = prices[-k_period:]
        recent_highs = highs[-k_period:]
        recent_lows = lows[-k_period:]
        
        # Calculate %K
        current_close = recent_prices[-1]
        highest_high = max(recent_highs)
        lowest_low = min(recent_lows)
        
        if highest_high == lowest_low:
            k = 50.0  # Default if there's no range
        else:
            k = 100 * ((current_close - lowest_low) / (highest_high - lowest_low))
            
        # Calculate %D (SMA of %K)
        if len(prices) < k_period + d_period - 1:
            d = k  # Not enough data for proper %D
        else:
            # We need historical %K values to calculate %D
            k_values = []
            for i in range(d_period):
                idx = -(i + 1)  # Start from most recent and go backwards
                
                c = prices[idx]
                h = max(highs[idx-k_period+1:idx+1])
                l = min(lows[idx-k_period+1:idx+1])
                
                if h == l:
                    k_values.append(50.0)
                else:
                    k_values.append(100 * ((c - l) / (h - l)))
                    
            d = sum(k_values) / d_period
            
        return k, d
    
    @staticmethod
    def calculate_volume_profile(volumes: List[float], prices: List[float], num_levels: int = 10) -> Dict[str, float]:
        """
        Calculate Volume Profile - shows volume concentration at different price levels
        Returns a dictionary mapping price levels to volume percentages
        """
        if not volumes or not prices or len(volumes) != len(prices):
            return {}
            
        # Create price bins
        min_price = min(prices)
        max_price = max(prices)
        
        if min_price == max_price:
            return {str(min_price): 100.0}
            
        # Create price levels
        bin_size = (max_price - min_price) / num_levels
        levels = [min_price + i * bin_size for i in range(num_levels + 1)]
        
        # Initialize volume profile
        volume_profile = {f"{round(levels[i], 2)}-{round(levels[i+1], 2)}": 0 for i in range(num_levels)}
        
        # Distribute volumes across price levels
        total_volume = sum(volumes)
        if total_volume == 0:
            return {key: 0.0 for key in volume_profile}
            
        for price, volume in zip(prices, volumes):
            # Find the bin this price belongs to
            for i in range(num_levels):
                if levels[i] <= price < levels[i+1] or (i == num_levels - 1 and price == levels[i+1]):
                    key = f"{round(levels[i], 2)}-{round(levels[i+1], 2)}"
                    volume_profile[key] += volume
                    break
                    
        # Convert to percentages
        for key in volume_profile:
            volume_profile[key] = (volume_profile[key] / total_volume) * 100
            
        return volume_profile
    
    @staticmethod
    def calculate_obv(prices: List[float], volumes: List[float]) -> float:
        """
        Calculate On-Balance Volume (OBV)
        """
        if len(prices) < 2 or len(volumes) < 2:
            return 0.0
            
        obv = volumes[0]
        
        for i in range(1, len(prices)):
            if prices[i] > prices[i-1]:
                obv += volumes[i]
            elif prices[i] < prices[i-1]:
                obv -= volumes[i]
                
        return obv
    
    @staticmethod
    def calculate_adx(highs: List[float], lows: List[float], prices: List[float], period: int = 14) -> float:
        """
        Calculate Average Directional Index (ADX)
        """
        if len(prices) < 2 * period:
            return 25.0  # Default to moderate trend strength
            
        # Calculate +DM and -DM
        plus_dm = []
        minus_dm = []
        
        for i in range(1, len(highs)):
            h_diff = highs[i] - highs[i-1]
            l_diff = lows[i-1] - lows[i]
            
            if h_diff > l_diff and h_diff > 0:
                plus_dm.append(h_diff)
            else:
                plus_dm.append(0)
                
            if l_diff > h_diff and l_diff > 0:
                minus_dm.append(l_diff)
            else:
                minus_dm.append(0)
                
        # Calculate True Range
        tr = []
        for i in range(1, len(prices)):
            tr1 = abs(highs[i] - lows[i])
            tr2 = abs(highs[i] - prices[i-1])
            tr3 = abs(lows[i] - prices[i-1])
            tr.append(max(tr1, tr2, tr3))
            
        # Calculate ATR (Average True Range)
        atr = sum(tr[:period]) / period
        
        # Calculate +DI and -DI
        plus_di = sum(plus_dm[:period]) / atr
        minus_di = sum(minus_dm[:period]) / atr
        
        # Calculate DX (Directional Index)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) > 0 else 0
        
        # Calculate ADX (smoothed DX)
        adx = dx
        
        for i in range(period, len(tr)):
            # Update ATR
            atr = ((period - 1) * atr + tr[i]) / period
            
            # Update +DI and -DI
            plus_di = ((period - 1) * plus_di + plus_dm[i]) / period
            minus_di = ((period - 1) * minus_di + minus_dm[i]) / period
            
            # Update DX
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) > 0 else 0
            
            # Smooth ADX
            adx = ((period - 1) * adx + dx) / period
            
        return adx
    
    @staticmethod
    def calculate_ichimoku(prices: List[float], highs: List[float], lows: List[float], 
                         tenkan_period: int = 9, kijun_period: int = 26, 
                         senkou_b_period: int = 52) -> Dict[str, float]:
        """
        Calculate Ichimoku Cloud components
        Returns key Ichimoku components
        """
        if len(prices) < senkou_b_period or len(highs) < senkou_b_period or len(lows) < senkou_b_period:
            return {
                "tenkan_sen": prices[-1] if prices else 0, 
                "kijun_sen": prices[-1] if prices else 0,
                "senkou_span_a": prices[-1] if prices else 0, 
                "senkou_span_b": prices[-1] if prices else 0
            }
            
        # Calculate Tenkan-sen (Conversion Line)
        high_tenkan = max(highs[-tenkan_period:])
        low_tenkan = min(lows[-tenkan_period:])
        tenkan_sen = (high_tenkan + low_tenkan) / 2
        
        # Calculate Kijun-sen (Base Line)
        high_kijun = max(highs[-kijun_period:])
        low_kijun = min(lows[-kijun_period:])
        kijun_sen = (high_kijun + low_kijun) / 2
        
        # Calculate Senkou Span A (Leading Span A)
        senkou_span_a = (tenkan_sen + kijun_sen) / 2
        
        # Calculate Senkou Span B (Leading Span B)
        high_senkou = max(highs[-senkou_b_period:])
        low_senkou = min(lows[-senkou_b_period:])
        senkou_span_b = (high_senkou + low_senkou) / 2
        
        return {
            "tenkan_sen": tenkan_sen,
            "kijun_sen": kijun_sen,
            "senkou_span_a": senkou_span_a,
            "senkou_span_b": senkou_span_b
        }
    
    @staticmethod 
    def calculate_pivot_points(high: float, low: float, close: float, pivot_type: str = "standard") -> Dict[str, float]:
        """
        Calculate pivot points for support and resistance levels
        Supports standard, fibonacci, and woodie pivot types
        """
        if pivot_type == "fibonacci":
            pivot = (high + low + close) / 3
            r1 = pivot + 0.382 * (high - low)
            r2 = pivot + 0.618 * (high - low)
            r3 = pivot + 1.0 * (high - low)
            s1 = pivot - 0.382 * (high - low)
            s2 = pivot - 0.618 * (high - low)
            s3 = pivot - 1.0 * (high - low)
        elif pivot_type == "woodie":
            pivot = (high + low + 2 * close) / 4
            r1 = 2 * pivot - low
            r2 = pivot + (high - low)
            s1 = 2 * pivot - high
            s2 = pivot - (high - low)
            r3 = r1 + (high - low)
            s3 = s1 - (high - low)
        else:  # standard
            pivot = (high + low + close) / 3
            r1 = 2 * pivot - low
            r2 = pivot + (high - low)
            r3 = r2 + (high - low)
            s1 = 2 * pivot - high
            s2 = pivot - (high - low)
            s3 = s2 - (high - low)
            
        return {
            "pivot": pivot,
            "r1": r1,
            "r2": r2,
            "r3": r3,
            "s1": s1,
            "s2": s2,
            "s3": s3
        }
                
    @staticmethod
    def analyze_technical_indicators(prices: List[float], volumes: List[float], highs: List[float] = None, lows: List[float] = None, timeframe: str = "1h") -> Dict[str, Any]:
        """
        Analyze multiple technical indicators and return results with interpretations
        Adjusted for different timeframes (1h, 24h, 7d)
        """
        if not prices or len(prices) < 26:
            return {"error": "Insufficient price data for technical analysis"}
            
        # Use closing prices for highs/lows if not provided
        if highs is None:
            highs = prices
        if lows is None:
            lows = prices
            
        # Adjust indicator parameters based on timeframe
        if timeframe == "24h":
            rsi_period = 14
            macd_fast, macd_slow, macd_signal = 12, 26, 9
            bb_period, bb_std = 20, 2.0
            stoch_k, stoch_d = 14, 3
        elif timeframe == "7d":
            rsi_period = 14
            macd_fast, macd_slow, macd_signal = 12, 26, 9
            bb_period, bb_std = 20, 2.0
            stoch_k, stoch_d = 14, 3
            # For weekly, we maintain similar parameters but apply them to weekly data
        else:  # 1h default
            rsi_period = 14
            macd_fast, macd_slow, macd_signal = 12, 26, 9
            bb_period, bb_std = 20, 2.0
            stoch_k, stoch_d = 14, 3
            
        # Calculate indicators
        rsi = TechnicalIndicators.calculate_rsi(prices, period=rsi_period)
        macd_line, signal_line, histogram = TechnicalIndicators.calculate_macd(
            prices, fast_period=macd_fast, slow_period=macd_slow, signal_period=macd_signal
        )
        upper_band, middle_band, lower_band = TechnicalIndicators.calculate_bollinger_bands(
            prices, period=bb_period, num_std=bb_std
        )
        k, d = TechnicalIndicators.calculate_stochastic_oscillator(
            prices, highs, lows, k_period=stoch_k, d_period=stoch_d
        )
        obv = TechnicalIndicators.calculate_obv(prices, volumes) if volumes else 0
        
        # Calculate additional indicators for longer timeframes
        additional_indicators = {}
        if timeframe in ["24h", "7d"]:
            # Calculate ADX for trend strength
            adx = TechnicalIndicators.calculate_adx(highs, lows, prices)
            additional_indicators["adx"] = adx
            
            # Calculate Ichimoku Cloud for longer-term trend analysis
            ichimoku = TechnicalIndicators.calculate_ichimoku(prices, highs, lows)
            additional_indicators["ichimoku"] = ichimoku
            
            # Calculate Pivot Points for key support/resistance levels
            # Use recent high, low, close for pivot calculation
            if len(prices) >= 5:
                high = max(highs[-5:])
                low = min(lows[-5:])
                close = prices[-1]
                pivot_type = "fibonacci" if timeframe == "7d" else "standard"
                pivots = TechnicalIndicators.calculate_pivot_points(high, low, close, pivot_type)
                additional_indicators["pivot_points"] = pivots
        
        # Interpret RSI with timeframe context
        if timeframe == "1h":
            if rsi > 70:
                rsi_signal = "overbought"
            elif rsi < 30:
                rsi_signal = "oversold"
            else:
                rsi_signal = "neutral"
        elif timeframe == "24h":
            # Slightly wider thresholds for daily
            if rsi > 75:
                rsi_signal = "overbought"
            elif rsi < 25:
                rsi_signal = "oversold"
            else:
                rsi_signal = "neutral"
        else:  # 7d
            # Even wider thresholds for weekly
            if rsi > 80:
                rsi_signal = "overbought"
            elif rsi < 20:
                rsi_signal = "oversold"
            else:
                rsi_signal = "neutral"
            
        # Interpret MACD
        if macd_line > signal_line and histogram > 0:
            macd_signal = "bullish"
        elif macd_line < signal_line and histogram < 0:
            macd_signal = "bearish"
        else:
            macd_signal = "neutral"
            
        # Interpret Bollinger Bands
        current_price = prices[-1]
        if current_price > upper_band:
            bb_signal = "overbought"
        elif current_price < lower_band:
            bb_signal = "oversold"
        else:
            # Check for Bollinger Band squeeze
            previous_bandwidth = (upper_band - lower_band) / middle_band
            if previous_bandwidth < 0.1:  # Tight bands indicate potential breakout
                bb_signal = "squeeze"
            else:
                bb_signal = "neutral"
                
        # Interpret Stochastic
        if k > 80 and d > 80:
            stoch_signal = "overbought"
        elif k < 20 and d < 20:
            stoch_signal = "oversold"
        elif k > d:
            stoch_signal = "bullish"
        elif k < d:
            stoch_signal = "bearish"
        else:
            stoch_signal = "neutral"
            
        # Add ADX interpretation for longer timeframes
        adx_signal = "neutral"
        if timeframe in ["24h", "7d"] and "adx" in additional_indicators:
            adx_value = additional_indicators["adx"]
            if adx_value > 30:
                adx_signal = "strong_trend"
            elif adx_value > 20:
                adx_signal = "moderate_trend"
            else:
                adx_signal = "weak_trend"
                
        # Add Ichimoku interpretation for longer timeframes
        ichimoku_signal = "neutral"
        if timeframe in ["24h", "7d"] and "ichimoku" in additional_indicators:
            ichimoku_data = additional_indicators["ichimoku"]
            if (current_price > ichimoku_data["senkou_span_a"] and 
                current_price > ichimoku_data["senkou_span_b"]):
                ichimoku_signal = "bullish"
            elif (current_price < ichimoku_data["senkou_span_a"] and 
                  current_price < ichimoku_data["senkou_span_b"]):
                ichimoku_signal = "bearish"
            else:
                ichimoku_signal = "neutral"
                
        # Determine overall signal
        signals = {
            "bullish": 0,
            "bearish": 0,
            "neutral": 0,
            "overbought": 0,
            "oversold": 0
        }
        
        # Count signals
        for signal in [rsi_signal, macd_signal, bb_signal, stoch_signal]:
            signals[signal] += 1
            
        # Add additional signals for longer timeframes
        if timeframe in ["24h", "7d"]:
            if adx_signal == "strong_trend" and macd_signal == "bullish":
                signals["bullish"] += 1
            elif adx_signal == "strong_trend" and macd_signal == "bearish":
                signals["bearish"] += 1
                
            if ichimoku_signal == "bullish":
                signals["bullish"] += 1
            elif ichimoku_signal == "bearish":
                signals["bearish"] += 1
            
        # Determine trend strength and direction
        if signals["bullish"] + signals["oversold"] > signals["bearish"] + signals["overbought"]:
            if signals["bullish"] > signals["oversold"]:
                trend = "strong_bullish" if signals["bullish"] >= 2 else "moderate_bullish"
            else:
                trend = "potential_reversal_bullish"
        elif signals["bearish"] + signals["overbought"] > signals["bullish"] + signals["oversold"]:
            if signals["bearish"] > signals["overbought"]:
                trend = "strong_bearish" if signals["bearish"] >= 2 else "moderate_bearish"
            else:
                trend = "potential_reversal_bearish"
        else:
            trend = "neutral"
            
        # Calculate trend strength (0-100)
        bullish_strength = signals["bullish"] * 25 + signals["oversold"] * 15
        bearish_strength = signals["bearish"] * 25 + signals["overbought"] * 15
        
        if trend in ["strong_bullish", "moderate_bullish", "potential_reversal_bullish"]:
            trend_strength = bullish_strength
        elif trend in ["strong_bearish", "moderate_bearish", "potential_reversal_bearish"]:
            trend_strength = bearish_strength
        else:
            trend_strength = 50  # Neutral
            
        # Calculate price volatility
        if len(prices) > 20:
            recent_prices = prices[-20:]
            volatility = np.std(recent_prices) / np.mean(recent_prices) * 100
        else:
            volatility = 5.0  # Default moderate volatility
        
        # Return all indicators and interpretations
        result = {
            "indicators": {
                "rsi": rsi,
                "macd": {
                    "macd_line": macd_line,
                    "signal_line": signal_line,
                    "histogram": histogram
                },
                "bollinger_bands": {
                    "upper": upper_band,
                    "middle": middle_band,
                    "lower": lower_band
                },
                "stochastic": {
                    "k": k,
                    "d": d
                },
                "obv": obv
            },
            "signals": {
                "rsi": rsi_signal,
                "macd": macd_signal,
                "bollinger_bands": bb_signal,
                "stochastic": stoch_signal
            },
            "overall_trend": trend,
            "trend_strength": trend_strength,
            "volatility": volatility,
            "timeframe": timeframe
        }
        
        # Add additional indicators for longer timeframes
        if timeframe in ["24h", "7d"]:
            result["indicators"].update(additional_indicators)
            result["signals"].update({
                "adx": adx_signal,
                "ichimoku": ichimoku_signal
            })
            
        return result


class StatisticalModels:
    """Class for statistical forecasting models"""
    
    @staticmethod
    def arima_forecast(prices: List[float], forecast_steps: int = 1, timeframe: str = "1h") -> Dict[str, Any]:
        """
        ARIMA forecasting model adjusted for different timeframes
        """
        try:
            # Adjust minimum data requirements based on timeframe
            min_data_points = 30
            if timeframe == "24h":
                min_data_points = 60  # Need more data for daily forecasts
            elif timeframe == "7d":
                min_data_points = 90  # Need even more data for weekly forecasts
                
            if len(prices) < min_data_points:
                raise ValueError(f"Insufficient data for ARIMA model with {timeframe} timeframe")
                
            # Adjust ARIMA parameters based on timeframe
            if timeframe == "1h":
                order = (5, 1, 0)  # Default for 1-hour
            elif timeframe == "24h":
                order = (5, 1, 1)  # Add MA component for daily
            else:  # 7d
                order = (7, 1, 1)  # More AR terms for weekly
                
            model = ARIMA(prices, order=order)
            model_fit = model.fit()
            
            # Make forecast
            forecast = model_fit.forecast(steps=forecast_steps)
            
            # Calculate confidence intervals (simple approach)
            residuals = model_fit.resid
            resid_std = np.std(residuals)
            
            # Adjust confidence interval width based on timeframe
            if timeframe == "1h":
                ci_multiplier_95 = 1.96
                ci_multiplier_80 = 1.28
            elif timeframe == "24h":
                ci_multiplier_95 = 2.20  # Wider for daily
                ci_multiplier_80 = 1.50
            else:  # 7d
                ci_multiplier_95 = 2.50  # Even wider for weekly
                ci_multiplier_80 = 1.80
                
            confidence_intervals = []
            for f in forecast:
                confidence_intervals.append({
                    "95": [f - ci_multiplier_95 * resid_std, f + ci_multiplier_95 * resid_std],
                    "80": [f - ci_multiplier_80 * resid_std, f + ci_multiplier_80 * resid_std]
                })
                
            return {
                "forecast": forecast.tolist(),
                "confidence_intervals": confidence_intervals,
                "model_info": {
                    "order": order,
                    "aic": model_fit.aic,
                    "timeframe": timeframe
                }
            }
        except Exception as e:
            logger.log_error(f"ARIMA Forecast ({timeframe})", str(e))
            # Return simple moving average forecast as fallback
            return StatisticalModels.moving_average_forecast(prices, forecast_steps, timeframe)
    
    @staticmethod
    def moving_average_forecast(prices: List[float], forecast_steps: int = 1, timeframe: str = "1h", window: int = None) -> Dict[str, Any]:
        """
        Simple moving average forecast (fallback method)
        Adjusted for different timeframes
        """
        # Set appropriate window size based on timeframe
        if window is None:
            if timeframe == "1h":
                window = 5
            elif timeframe == "24h":
                window = 7
            else:  # 7d
                window = 4
                
        if len(prices) < window:
            return {
                "forecast": [prices[-1]] * forecast_steps,
                "confidence_intervals": [{
                    "95": [prices[-1] * 0.95, prices[-1] * 1.05],
                    "80": [prices[-1] * 0.97, prices[-1] * 1.03]
                }] * forecast_steps,
                "model_info": {
                    "method": "last_price_fallback",
                    "timeframe": timeframe
                }
            }
            
        # Calculate moving average
        ma = sum(prices[-window:]) / window
        
        # Calculate standard deviation for confidence intervals
        std = np.std(prices[-window:])
        
        # Adjust confidence intervals based on timeframe
        if timeframe == "1h":
            ci_multiplier_95 = 1.96
            ci_multiplier_80 = 1.28
        elif timeframe == "24h":
            ci_multiplier_95 = 2.20
            ci_multiplier_80 = 1.50
        else:  # 7d
            ci_multiplier_95 = 2.50
            ci_multiplier_80 = 1.80
            
        # Generate forecast (all same value for MA)
        forecast = [ma] * forecast_steps
        
        # Generate confidence intervals
        confidence_intervals = []
        for _ in range(forecast_steps):
            confidence_intervals.append({
                "95": [ma - ci_multiplier_95 * std, ma + ci_multiplier_95 * std],
                "80": [ma - ci_multiplier_80 * std, ma + ci_multiplier_80 * std]
            })
            
        return {
            "forecast": forecast,
            "confidence_intervals": confidence_intervals,
            "model_info": {
                "method": "moving_average",
                "window": window,
                "timeframe": timeframe
            }
        }
    
    @staticmethod
    def weighted_average_forecast(prices: List[float], volumes: List[float] = None, 
                                forecast_steps: int = 1, timeframe: str = "1h") -> Dict[str, Any]:
        """
        Volume-weighted average price forecast or linearly weighted forecast
        Adjusted for different timeframes
        """
        # Adjust window size based on timeframe
        if timeframe == "1h":
            window = 10
        elif timeframe == "24h":
            window = 14
        else:  # 7d
            window = 8
            
        if len(prices) < window:
            return StatisticalModels.moving_average_forecast(prices, forecast_steps, timeframe)
            
        # If volumes available, use volume-weighted average
        if volumes and len(volumes) == len(prices):
            # Get last window periods
            recent_prices = prices[-window:]
            recent_volumes = volumes[-window:]
            
            # Calculate VWAP
            vwap = sum(p * v for p, v in zip(recent_prices, recent_volumes)) / sum(recent_volumes) if sum(recent_volumes) > 0 else sum(recent_prices) / len(recent_prices)
            
            forecast = [vwap] * forecast_steps
            method = "volume_weighted"
        else:
            # Use linearly weighted average (more weight to recent prices)
            # Adjust weights based on timeframe for recency bias
            if timeframe == "1h":
                weights = list(range(1, window + 1))  # Linear weights
            elif timeframe == "24h":
                # Exponential weights for daily (more recency bias)
                weights = [1.5 ** i for i in range(1, window + 1)]
            else:  # 7d
                # Even more recency bias for weekly
                weights = [2.0 ** i for i in range(1, window + 1)]
                
            recent_prices = prices[-window:]
            
            weighted_avg = sum(p * w for p, w in zip(recent_prices, weights)) / sum(weights)
            
            forecast = [weighted_avg] * forecast_steps
            method = "weighted_average"
            
        # Calculate standard deviation for confidence intervals
        std = np.std(prices[-window:])
        
        # Adjust confidence intervals based on timeframe
        if timeframe == "1h":
            ci_multiplier_95 = 1.96
            ci_multiplier_80 = 1.28
        elif timeframe == "24h":
            ci_multiplier_95 = 2.20
            ci_multiplier_80 = 1.50
        else:  # 7d
            ci_multiplier_95 = 2.50
            ci_multiplier_80 = 1.80
            
        # Generate confidence intervals
        confidence_intervals = []
        for f in forecast:
            confidence_intervals.append({
                "95": [f - ci_multiplier_95 * std, f + ci_multiplier_95 * std],
                "80": [f - ci_multiplier_80 * std, f + ci_multiplier_80 * std]
            })
            
        return {
            "forecast": forecast,
            "confidence_intervals": confidence_intervals,
            "model_info": {
                "method": method,
                "window": window,
                "timeframe": timeframe
            }
        }
        
    @staticmethod
    def holt_winters_forecast(prices: List[float], forecast_steps: int = 1, timeframe: str = "1h") -> Dict[str, Any]:
        """
        Holt-Winters exponential smoothing forecast
        Good for data with trend and seasonality
        """
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            
            # Minimum data requirements based on timeframe
            min_data_points = {
                "1h": 48,   # 2 days of hourly data
                "24h": 30,  # 1 month of daily data
                "7d": 16    # 4 months of weekly data
            }
            
            if len(prices) < min_data_points.get(timeframe, 48):
                return StatisticalModels.weighted_average_forecast(prices, None, forecast_steps, timeframe)
                
            # Determine seasonal_periods based on timeframe
            if timeframe == "1h":
                seasonal_periods = 24  # 24 hours in a day
            elif timeframe == "24h":
                seasonal_periods = 7   # 7 days in a week
            else:  # 7d
                seasonal_periods = 4   # 4 weeks in a month
                
            # Create and fit model
            model = ExponentialSmoothing(
                prices, 
                trend='add',
                seasonal='add', 
                seasonal_periods=seasonal_periods
            )
            model_fit = model.fit()
            
            # Generate forecast
            forecast = model_fit.forecast(forecast_steps)
            
            # Calculate confidence intervals
            residuals = model_fit.resid
            resid_std = np.std(residuals)
            
            # Adjust confidence interval width based on timeframe
            if timeframe == "1h":
                ci_multiplier_95 = 1.96
                ci_multiplier_80 = 1.28
            elif timeframe == "24h":
                ci_multiplier_95 = 2.20
                ci_multiplier_80 = 1.50
            else:  # 7d
                ci_multiplier_95 = 2.50
                ci_multiplier_80 = 1.80
                
            confidence_intervals = []
            for f in forecast:
                confidence_intervals.append({
                    "95": [f - ci_multiplier_95 * resid_std, f + ci_multiplier_95 * resid_std],
                    "80": [f - ci_multiplier_80 * resid_std, f + ci_multiplier_80 * resid_std]
                })
                
            return {
                "forecast": forecast.tolist(),
                "confidence_intervals": confidence_intervals,
                "model_info": {
                    "method": "holt_winters",
                    "seasonal_periods": seasonal_periods,
                    "timeframe": timeframe
                }
            }
        except Exception as e:
            logger.log_error(f"Holt-Winters Forecast ({timeframe})", str(e))
            return StatisticalModels.weighted_average_forecast(prices, None, forecast_steps, timeframe)


class MachineLearningModels:
    """Class for machine learning forecasting models"""
    
    @staticmethod
    def create_features(prices: List[float], volumes: List[float] = None, timeframe: str = "1h") -> pd.DataFrame:
        """
        Create features for ML models from price and volume data
        Adjusted for different timeframes
        """
        # Adjust window sizes based on timeframe
        if timeframe == "1h":
            window_sizes = [5, 10, 20]
            max_lag = 6
        elif timeframe == "24h":
            window_sizes = [7, 14, 30]
            max_lag = 10
        else:  # 7d
            window_sizes = [4, 8, 12]
            max_lag = 8
            
        df = pd.DataFrame({'price': prices})
        
        if volumes:
            df['volume'] = volumes[:len(prices)]
        
        # Add lagged features
        for lag in range(1, max_lag):
            df[f'price_lag_{lag}'] = df['price'].shift(lag)
            
        # Add moving averages
        for window in window_sizes:
            df[f'ma_{window}'] = df['price'].rolling(window=window).mean()
            
        # Add price momentum
        for window in window_sizes:
            df[f'momentum_{window}'] = df['price'] - df[f'ma_{window}']
            
        # Add relative price change
        for lag in range(1, max_lag):
            df[f'price_change_{lag}'] = (df['price'] / df[f'price_lag_{lag}'] - 1) * 100
            
        # Add volatility
        for window in window_sizes:
            df[f'volatility_{window}'] = df['price'].rolling(window=window).std()
            
        # Add volume features if available
        if volumes:
            for window in window_sizes:
                df[f'volume_ma_{window}'] = df['volume'].rolling(window=window).mean()
                df[f'volume_change_{window}'] = (df['volume'] / df[f'volume_ma_{window}'] - 1) * 100
                
        # Add timeframe-specific features
        if timeframe == "24h":
            # Add day-of-week effect for daily data (if we have enough data)
            if len(df) >= 7:
                # Create day of week encoding (0-6, where 0 is Monday)
                # This is a placeholder - in real implementation you would use actual dates
                df['day_of_week'] = np.arange(len(df)) % 7
                
                # One-hot encode day of week
                for day in range(7):
                    df[f'day_{day}'] = (df['day_of_week'] == day).astype(int)
        elif timeframe == "7d":
            # Add week-of-month or week-of-year features
            if len(df) >= 4:
                # Create week of month encoding (0-3)
                # This is a placeholder - in real implementation you would use actual dates
                df['week_of_month'] = np.arange(len(df)) % 4
                
                # One-hot encode week of month
                for week in range(4):
                    df[f'week_{week}'] = (df['week_of_month'] == week).astype(int)
                    
        # Add additional technical indicators
        if len(prices) >= 14:
            # RSI
            delta = df['price'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            
            rs = avg_gain / avg_loss
            df['rsi_14'] = 100 - (100 / (1 + rs))
            
            # MACD components
            ema_12 = df['price'].ewm(span=12, adjust=False).mean()
            ema_26 = df['price'].ewm(span=26, adjust=False).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
                
        # Drop NaN values
        df = df.dropna()
        
        return df
    
    @staticmethod
    def random_forest_forecast(prices: List[float], volumes: List[float] = None, 
                             forecast_steps: int = 1, timeframe: str = "1h") -> Dict[str, Any]:
        """
        Random Forest regression forecast
        Adjusted for different timeframes
        """
        try:
            # Minimum data requirements based on timeframe
            min_data_points = {
                "1h": 48,   # 2 days of hourly data
                "24h": 30,  # 1 month of daily data
                "7d": 16    # 4 months of weekly data
            }
            
            if len(prices) < min_data_points.get(timeframe, 30):
                raise ValueError(f"Insufficient data for Random Forest model with {timeframe} timeframe")
                
            # Create features with timeframe-specific settings
            df = MachineLearningModels.create_features(prices, volumes, timeframe)
            
            if len(df) < min_data_points.get(timeframe, 30) // 2:
                raise ValueError(f"Insufficient features after preprocessing for {timeframe} timeframe")
                
            # Prepare training data
            X = df.drop('price', axis=1)
            y = df['price']
            
            # Create and train model with timeframe-specific parameters
            if timeframe == "1h":
                model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            elif timeframe == "24h":
                model = RandomForestRegressor(n_estimators=150, max_depth=15, random_state=42)
            else:  # 7d
                model = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42)
                
            model.fit(X, y)
            
            # Prepare forecast data
            forecast_data = []
            last_known = df.iloc[-1:].copy()
            
            for _ in range(forecast_steps):
                # Make prediction for next step
                pred = model.predict(last_known.drop('price', axis=1))[0]
                
                # Update last_known for next step
                new_row = last_known.copy()
                new_row['price'] = pred
                
                # Update lags
                max_lag = 10 if timeframe == "24h" else 8 if timeframe == "7d" else 6
                for lag in range(max_lag - 1, 0, -1):
                    if lag == 1:
                        new_row[f'price_lag_{lag}'] = last_known['price'].values[0]
                    else:
                        new_row[f'price_lag_{lag}'] = last_known[f'price_lag_{lag-1}'].values[0]
                        
                # Add prediction to results
                forecast_data.append(pred)
                
                # Update last_known for next iteration
                last_known = new_row
                
            # Calculate confidence intervals based on feature importance and model uncertainty
            feature_importance = model.feature_importances_.sum()
            
            # Higher importance = more confident = narrower intervals
            # Adjust confidence scale based on timeframe
            if timeframe == "1h":
                base_confidence_scale = 1.0
            elif timeframe == "24h":
                base_confidence_scale = 1.2  # Slightly less confident for daily
            else:  # 7d
                base_confidence_scale = 1.5  # Even less confident for weekly
                
            confidence_scale = max(0.5, min(2.0, base_confidence_scale / feature_importance))
            std = np.std(prices[-20:]) * confidence_scale
            
            # Adjust CI width based on timeframe
            if timeframe == "1h":
                ci_multiplier_95 = 1.96
                ci_multiplier_80 = 1.28
            elif timeframe == "24h":
                ci_multiplier_95 = 2.20
                ci_multiplier_80 = 1.50
            else:  # 7d
                ci_multiplier_95 = 2.50
                ci_multiplier_80 = 1.80
            
            confidence_intervals = []
            for f in forecast_data:
                confidence_intervals.append({
                    "95": [f - ci_multiplier_95 * std, f + ci_multiplier_95 * std],
                    "80": [f - ci_multiplier_80 * std, f + ci_multiplier_80 * std]
                })
            
            # Get feature importance for top features
            feature_importance_dict = dict(zip(X.columns, model.feature_importances_))
            top_features = dict(sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)[:5])
                
            return {
                "forecast": forecast_data,
                "confidence_intervals": confidence_intervals,
                "feature_importance": top_features,
                "model_info": {
                    "method": "random_forest",
                    "n_estimators": model.n_estimators,
                    "max_depth": model.max_depth,
                    "timeframe": timeframe
                }
            }
        except Exception as e:
            logger.log_error(f"Random Forest Forecast ({timeframe})", str(e))
            # Fallback to moving average
            return StatisticalModels.moving_average_forecast(prices, forecast_steps, timeframe)
    
    @staticmethod
    def linear_regression_forecast(prices: List[float], volumes: List[float] = None, 
                                 forecast_steps: int = 1, timeframe: str = "1h") -> Dict[str, Any]:
        """
        Linear regression forecast
        Adjusted for different timeframes
        """
        try:
            # Minimum data requirements based on timeframe
            min_data_points = {
                "1h": 24,   # 1 day of hourly data
                "24h": 14,  # 2 weeks of daily data
                "7d": 8     # 2 months of weekly data
            }
            
            if len(prices) < min_data_points.get(timeframe, 20):
                raise ValueError(f"Insufficient data for Linear Regression model with {timeframe} timeframe")
                
            # Create features with smaller window sizes for linear regression
            if timeframe == "1h":
                window_sizes = [3, 5, 10]
            elif timeframe == "24h":
                window_sizes = [3, 7, 14]
            else:  # 7d
                window_sizes = [2, 4, 8]
                
            # Create DataFrame
            df = pd.DataFrame({'price': prices})
            
            if volumes:
                df['volume'] = volumes[:len(prices)]
            
            # Add lagged features
            max_lag = 5
            for lag in range(1, max_lag + 1):
                df[f'price_lag_{lag}'] = df['price'].shift(lag)
                
            # Add moving averages
            for window in window_sizes:
                df[f'ma_{window}'] = df['price'].rolling(window=window).mean()
                
            # Add price momentum
            for window in window_sizes:
                df[f'momentum_{window}'] = df['price'] - df[f'ma_{window}']
                
            # Add volume features if available
            if volumes:
                for window in window_sizes:
                    df[f'volume_ma_{window}'] = df['volume'].rolling(window=window).mean()
                    
            # Drop NaN values
            df = df.dropna()
            
            if len(df) < 15:
                raise ValueError("Insufficient features after preprocessing")
                
            # Prepare training data
            X = df.drop('price', axis=1)
            y = df['price']
            
            # Create and train model
            model = LinearRegression()
            model.fit(X, y)
            
            # Prepare forecast data
            forecast_data = []
            last_known = df.iloc[-1:].copy()
            
            for _ in range(forecast_steps):
                # Make prediction for next step
                pred = model.predict(last_known.drop('price', axis=1))[0]
                
                # Update last_known for next step
                new_row = last_known.copy()
                new_row['price'] = pred
                
                # Update lags
                for lag in range(max_lag, 0, -1):
                    if lag == 1:
                        new_row[f'price_lag_{lag}'] = last_known['price'].values[0]
                    else:
                        new_row[f'price_lag_{lag}'] = last_known[f'price_lag_{lag-1}'].values[0]
                        
                # Add prediction to results
                forecast_data.append(pred)
                
                # Update last_known for next iteration
                last_known = new_row
                
            # Calculate confidence intervals based on model's prediction error
            y_pred = model.predict(X)
            mse = np.mean((y - y_pred) ** 2)
            std = np.sqrt(mse)
            
            # Adjust confidence intervals based on timeframe
            if timeframe == "1h":
                ci_multiplier_95 = 1.96
                ci_multiplier_80 = 1.28
            elif timeframe == "24h":
                ci_multiplier_95 = 2.20
                ci_multiplier_80 = 1.50
            else:  # 7d
                ci_multiplier_95 = 2.50
                ci_multiplier_80 = 1.80
                
            confidence_intervals = []
            for f in forecast_data:
                confidence_intervals.append({
                    "95": [f - ci_multiplier_95 * std, f + ci_multiplier_95 * std],
                    "80": [f - ci_multiplier_80 * std, f + ci_multiplier_80 * std]
                })
                
            # Get top coefficients
            coefficients = dict(zip(X.columns, model.coef_))
            top_coefficients = dict(sorted(coefficients.items(), key=lambda x: abs(x[1]), reverse=True)[:5])
                
            return {
                "forecast": forecast_data,
                "confidence_intervals": confidence_intervals,
                "coefficients": top_coefficients,
                "model_info": {
                    "method": "linear_regression",
                    "r2_score": model.score(X, y),
                    "timeframe": timeframe
                }
            }
        except Exception as e:
            logger.log_error(f"Linear Regression Forecast ({timeframe})", str(e))
            # Fallback to moving average
            return StatisticalModels.moving_average_forecast(prices, forecast_steps, timeframe)
    
    @staticmethod
    def lstm_forecast(prices: List[float], forecast_steps: int = 1, timeframe: str = "1h") -> Dict[str, Any]:
        """
        LSTM neural network forecast for time series
        """
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense
            from tensorflow.keras.callbacks import EarlyStopping
            
            # Minimum data requirements based on timeframe
            min_data_points = {
                "1h": 168,  # 7 days of hourly data
                "24h": 60,  # 2 months of daily data
                "7d": 24    # 6 months of weekly data
            }
            
            if len(prices) < min_data_points.get(timeframe, 100):
                return MachineLearningModels.random_forest_forecast(prices, None, forecast_steps, timeframe)
                
            # Prepare data for LSTM (with lookback window)
            if timeframe == "1h":
                lookback = 24  # 1 day
            elif timeframe == "24h":
                lookback = 14  # 2 weeks
            else:  # 7d
                lookback = 8   # 2 months
                
            # Scale data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_prices = scaler.fit_transform(np.array(prices).reshape(-1, 1))
            
            # Create dataset with lookback
            X, y = [], []
            for i in range(len(scaled_prices) - lookback):
                X.append(scaled_prices[i:i+lookback, 0])
                y.append(scaled_prices[i+lookback, 0])
                
            X, y = np.array(X), np.array(y)
            X = X.reshape(X.shape[0], X.shape[1], 1)
            
            # Build LSTM model
            model = Sequential()
            model.add(LSTM(50, return_sequences=True, input_shape=(lookback, 1)))
            model.add(LSTM(50))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mean_squared_error')
            
            # Train model with early stopping
            early_stopping = EarlyStopping(
                monitor='loss',
                patience=10,
                restore_best_weights=True
            )
            
            model.fit(
                X, y,
                epochs=50,
                batch_size=32,
                verbose=0,
                callbacks=[early_stopping]
            )
            
            # Generate predictions
            last_window = scaled_prices[-lookback:].reshape(1, lookback, 1)
            forecast_scaled = []
            
            for _ in range(forecast_steps):
                next_pred = model.predict(last_window, verbose=0)[0, 0]
                forecast_scaled.append(next_pred)
                # Update window for next prediction
                last_window = np.append(last_window[:, 1:, :], [[next_pred]], axis=1)
                
            # Inverse transform to get actual price predictions
            forecast_data = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1)).flatten().tolist()
            
            # Calculate confidence intervals based on model's training error
            y_pred = model.predict(X, verbose=0).flatten()
            y_true = y
            
            mse = np.mean((y_true - y_pred) ** 2)
            std = np.sqrt(mse)
            std_unscaled = std * (scaler.data_max_[0] - scaler.data_min_[0])
            
            # Adjust confidence intervals based on timeframe
            if timeframe == "1h":
                ci_multiplier_95 = 1.96
                ci_multiplier_80 = 1.28
            elif timeframe == "24h":
                ci_multiplier_95 = 2.20
                ci_multiplier_80 = 1.50
            else:  # 7d
                ci_multiplier_95 = 2.50
                ci_multiplier_80 = 1.80
                
            confidence_intervals = []
            for f in forecast_data:
                confidence_intervals.append({
                    "95": [f - ci_multiplier_95 * std_unscaled, f + ci_multiplier_95 * std_unscaled],
                    "80": [f - ci_multiplier_80 * std_unscaled, f + ci_multiplier_80 * std_unscaled]
                })
                
            return {
                "forecast": forecast_data,
                "confidence_intervals": confidence_intervals,
                "model_info": {
                    "method": "lstm",
                    "lookback": lookback,
                    "lstm_units": 50,
                    "timeframe": timeframe
                }
            }
        except Exception as e:
            logger.log_error(f"LSTM Forecast ({timeframe})", str(e))
            # Fallback to random forest
            return MachineLearningModels.random_forest_forecast(prices, None, forecast_steps, timeframe)


class ClaudeEnhancedPrediction:
    """Class for generating Claude AI enhanced predictions"""
    
    def __init__(self, api_key: str, model: str = "claude-3-7-sonnet-20250219"):
        """Initialize Claude client"""
        self.client = anthropic.Client(api_key=api_key)
        self.model = model
        
    def generate_enhanced_prediction(self, 
                                     token: str, 
                                     current_price: float,
                                     technical_analysis: Dict[str, Any],
                                     statistical_forecast: Dict[str, Any],
                                     ml_forecast: Dict[str, Any],
                                     timeframe: str = "1h",
                                     price_history_24h: List[Dict[str, Any]] = None,
                                     market_conditions: Dict[str, Any] = None,
                                     recent_predictions: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate an enhanced prediction using Claude AI
        Supports 1h, 24h, and 7d timeframes
        """
        try:
            # Format technical signals
            tech_signals = technical_analysis.get("signals", {})
            overall_trend = technical_analysis.get("overall_trend", "neutral")
            trend_strength = technical_analysis.get("trend_strength", 50)
            
            # Format forecasts
            stat_forecast = statistical_forecast.get("forecast", [current_price])[0]
            stat_confidence = statistical_forecast.get("confidence_intervals", [{"80": [current_price*0.98, current_price*1.02]}])[0]
            
            ml_forecast = ml_forecast.get("forecast", [current_price])[0]
            ml_confidence = ml_forecast.get("confidence_intervals", [{"80": [current_price*0.98, current_price*1.02]}])[0]
            
            # Calculate average forecast
            avg_forecast = (stat_forecast + ml_forecast) / 2
            
            # Prepare historical context
            historical_context = ""
            if price_history_24h:
                # Get min and max prices over 24h
                prices = [entry.get("price", 0) for entry in price_history_24h]
                volumes = [entry.get("volume", 0) for entry in price_history_24h]
                
                if prices:
                    min_price = min(prices)
                    max_price = max(prices)
                    avg_price = sum(prices) / len(prices)
                    total_volume = sum(volumes)
                    
                    # Adjust display based on timeframe
                    if timeframe == "1h":
                        period_desc = "24-Hour"
                    elif timeframe == "24h":
                        period_desc = "7-Day"
                    else:  # 7d
                        period_desc = "30-Day"
                        
                    historical_context = f"""
{period_desc} Price Data:
- Current: ${current_price}
- Average: ${avg_price}
- High: ${max_price}
- Low: ${min_price}
- Range: ${max_price - min_price} ({((max_price - min_price) / min_price) * 100:.2f}%)
- Total Volume: ${total_volume}
"""
            
            # Market conditions context
            market_context = ""
            if market_conditions:
                market_context = f"""
Market Conditions:
- Overall market trend: {market_conditions.get('market_trend', 'unknown')}
- BTC dominance: {market_conditions.get('btc_dominance', 'unknown')}
- Market volatility: {market_conditions.get('market_volatility', 'unknown')}
- Sector performance: {market_conditions.get('sector_performance', 'unknown')}
"""
                
            # Accuracy context
            accuracy_context = ""
            if recent_predictions:
                correct_predictions = [p for p in recent_predictions if p.get("was_correct")]
                accuracy_rate = len(correct_predictions) / len(recent_predictions) if recent_predictions else 0
                
                accuracy_context = f"""
Recent Prediction Performance:
- Accuracy rate for {timeframe} predictions: {accuracy_rate * 100:.1f}%
- Total predictions: {len(recent_predictions)}
- Correct predictions: {len(correct_predictions)}
"""
                
            # Get additional technical indicators for longer timeframes
            additional_indicators = ""
            if timeframe in ["24h", "7d"] and "indicators" in technical_analysis:
                indicators = technical_analysis["indicators"]
                
                # Add ADX if available
                if "adx" in indicators:
                    additional_indicators += f"- ADX: {indicators['adx']:.2f}\n"
                    
                # Add Ichimoku Cloud if available
                if "ichimoku" in indicators:
                    ichimoku = indicators["ichimoku"]
                    additional_indicators += "- Ichimoku Cloud:\n"
                    additional_indicators += f"  - Tenkan-sen: {ichimoku['tenkan_sen']:.4f}\n"
                    additional_indicators += f"  - Kijun-sen: {ichimoku['kijun_sen']:.4f}\n"
                    additional_indicators += f"  - Senkou Span A: {ichimoku['senkou_span_a']:.4f}\n"
                    additional_indicators += f"  - Senkou Span B: {ichimoku['senkou_span_b']:.4f}\n"
                    
                # Add Pivot Points if available
                if "pivot_points" in indicators:
                    pivots = indicators["pivot_points"]
                    additional_indicators += "- Pivot Points:\n"
                    additional_indicators += f"  - Pivot: {pivots['pivot']:.4f}\n"
                    additional_indicators += f"  - R1: {pivots['r1']:.4f}, R2: {pivots['r2']:.4f}\n"
                    additional_indicators += f"  - S1: {pivots['s1']:.4f}, S2: {pivots['s2']:.4f}\n"
            
            # Calculate optimal confidence interval for FOMO generation
            # We want tight but realistic bounds - narrower than statistical models suggest
            # but not so narrow that we're always wrong
            current_volatility = technical_analysis.get("volatility", 5.0)
            
            # Scale confidence interval based on volatility, trend strength, and timeframe
            # Higher volatility = wider interval
            # Stronger trend = narrower interval (more confident)
            # Longer timeframe = wider interval
            volatility_factor = min(1.5, max(0.5, current_volatility / 10))
            trend_factor = max(0.7, min(1.3, 1.2 - (trend_strength / 100)))
            
            # Timeframe factor - wider intervals for longer timeframes
            if timeframe == "1h":
                timeframe_factor = 1.0
            elif timeframe == "24h":
                timeframe_factor = 1.5
            else:  # 7d
                timeframe_factor = 2.0
                
            # Calculate confidence bounds
            bound_factor = volatility_factor * trend_factor * timeframe_factor
            lower_bound = avg_forecast * (1 - 0.015 * bound_factor)
            upper_bound = avg_forecast * (1 + 0.015 * bound_factor)
            
            # Ensure bounds are narrow enough to create FOMO but realistic for the timeframe
            price_range_pct = (upper_bound - lower_bound) / current_price * 100
            
            # Adjust max range based on timeframe
            max_range_pct = {
                "1h": 3.0,   # 3% for 1 hour
                "24h": 8.0,  # 8% for 24 hours
                "7d": 15.0   # 15% for 7 days
            }.get(timeframe, 3.0)
            
            if price_range_pct > max_range_pct:
                # Too wide - recalculate to create FOMO
                center = (upper_bound + lower_bound) / 2
                margin = (current_price * max_range_pct / 200)  # half of max_range_pct
                upper_bound = center + margin
                lower_bound = center - margin
                
            # Timeframe-specific guidance for FOMO generation
            fomo_guidance = {
                "1h": "Focus on immediate catalysts and short-term technical breakouts for this 1-hour prediction.",
                "24h": "Emphasize day-trading patterns and 24-hour potential for this daily prediction.",
                "7d": "Highlight medium-term trend confirmation and key weekly support/resistance levels."
            }.get(timeframe, "")
            
            # Prepare the prompt for Claude with timeframe-specific adjustments
            prompt = f"""
You are a sophisticated crypto market prediction expert. I need your analysis to make a precise {timeframe} prediction for {token}.

## Technical Analysis
- RSI Signal: {tech_signals.get('rsi', 'neutral')}
- MACD Signal: {tech_signals.get('macd', 'neutral')}
- Bollinger Bands: {tech_signals.get('bollinger_bands', 'neutral')}
- Stochastic Oscillator: {tech_signals.get('stochastic', 'neutral')}
- Overall Trend: {overall_trend}
- Trend Strength: {trend_strength}/100
{additional_indicators}

## Statistical Models
- Forecast: ${stat_forecast:.4f}
- 80% Confidence: [${stat_confidence['80'][0]:.4f}, ${stat_confidence['80'][1]:.4f}]

## Machine Learning Models
- ML Forecast: ${ml_forecast:.4f}
- 80% Confidence: [${ml_confidence['80'][0]:.4f}, ${ml_confidence['80'][1]:.4f}]

## Current Market Data
- Current Price: ${current_price}
- Predicted Range: [${lower_bound:.4f}, ${upper_bound:.4f}]

{historical_context}
{market_context}
{accuracy_context}

## Prediction Task
1. Predict the EXACT price of {token} in {timeframe} with a confidence level between 65-85%.
2. Provide a narrow price range to create FOMO, but ensure it's realistic given the data and {timeframe} timeframe.
3. State the percentage change you expect.
4. Give a concise rationale (2-3 sentences maximum).
5. Assign a sentiment: BULLISH, BEARISH, or NEUTRAL.

{fomo_guidance}

Your prediction must follow this EXACT JSON format:
{{
  "prediction": {{
    "price": [exact price prediction],
    "confidence": [confidence percentage],
    "lower_bound": [lower price bound],
    "upper_bound": [upper price bound],
    "percent_change": [expected percentage change],
    "timeframe": "{timeframe}"
  }},
  "rationale": [brief explanation],
  "sentiment": [BULLISH/BEARISH/NEUTRAL],
  "key_factors": [list of 2-3 main factors influencing this prediction]
}}

Your prediction should be precise, data-driven, and conservative enough to be accurate while narrow enough to generate excitement.
IMPORTANT: Provide ONLY the JSON response, no additional text.
"""
            
            # Make the API call to Claude
            logger.logger.debug(f"Requesting Claude {timeframe} prediction for {token}")
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Process the response
            result_text = response.content[0].text.strip()
            
            # Extract JSON
            result_text = result_text.replace("```json", "").replace("```", "").strip()
            result = json.loads(result_text)
            
            # Add the model weightings that produced this prediction
            result["model_weights"] = {
                "technical_analysis": 0.25,
                "statistical_models": 0.25,
                "machine_learning": 0.25,
                "claude_enhanced": 0.25
            }
            
            # Add the inputs that generated this prediction
            result["inputs"] = {
                "current_price": current_price,
                "technical_analysis": {
                    "overall_trend": overall_trend,
                    "trend_strength": trend_strength,
                    "signals": tech_signals
                },
                "statistical_forecast": {
                    "prediction": stat_forecast,
                    "confidence": stat_confidence['80']
                },
                "ml_forecast": {
                    "prediction": ml_forecast,
                    "confidence": ml_confidence['80']
                },
                "timeframe": timeframe
            }
            
            logger.logger.debug(f"Claude {timeframe} prediction generated for {token}: {result['prediction']['price']}")
            return result
            
        except Exception as e:
            logger.log_error(f"Claude Enhanced Prediction ({timeframe})", str(e))
            
            # Generate fallback prediction
            fallback = {
                "prediction": {
                    "price": (stat_forecast + ml_forecast) / 2,
                    "confidence": 70,
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "percent_change": ((avg_forecast / current_price) - 1) * 100,
                    "timeframe": timeframe
                },
                "rationale": f"Based on combined technical and statistical analysis for {timeframe} forecast.",
                "sentiment": "NEUTRAL" if overall_trend == "neutral" else "BULLISH" if "bullish" in overall_trend else "BEARISH",
                "key_factors": ["Technical indicators", "Statistical models", "Recent price action"],
                "model_weights": {
                    "technical_analysis": 0.4,
                    "statistical_models": 0.4,
                    "machine_learning": 0.2,
                    "claude_enhanced": 0.0
                }
            }
            
            return fallback


class PredictionEngine:
    """Main prediction engine class that combines all approaches"""
    
    def __init__(self, database, claude_api_key=None):
        """Initialize the prediction engine"""
        self.db = database
        self.claude_model = "claude-3-7-sonnet-20250219"
        
        # Initialize Claude if API key provided
        if claude_api_key:
            self.claude = ClaudeEnhancedPrediction(api_key=claude_api_key, model=self.claude_model)
        else:
            self.claude = None
            
    def generate_prediction(self, token: str, market_data: Dict[str, Any], timeframe: str = "1h") -> Dict[str, Any]:
        """
        Generate a comprehensive prediction for a token
        Supports 1h, 24h, and 7d timeframes
        """
        try:
            # Validate timeframe
            if timeframe not in ["1h", "24h", "7d"]:
                logger.logger.warning(f"Invalid timeframe: {timeframe}. Using 1h as default.")
                timeframe = "1h"
                
            # Extract token data
            token_data = market_data.get(token, {})
            if not token_data:
                raise ValueError(f"No market data available for {token}")
                
            current_price = token_data.get('current_price', 0)
            
            # Get historical price data from database
            # Adjust the hours parameter based on timeframe
            if timeframe == "1h":
                historical_hours = 48  # 2 days of data for 1h predictions
            elif timeframe == "24h":
                historical_hours = 168  # 7 days of data for 24h predictions
            else:  # 7d
                historical_hours = 720  # 30 days of data for 7d predictions
                
            historical_data = self.db.get_recent_market_data(token, hours=historical_hours)
            
            if not historical_data:
                logger.logger.warning(f"No historical data found for {token}")
                
            # Extract price and volume history
            prices = [current_price]  # Start with current price
            volumes = [token_data.get('volume', 0)]  # Start with current volume
            highs = [current_price]
            lows = [current_price]
            
            # Add historical data
            for entry in reversed(historical_data):  # Oldest to newest
                prices.insert(0, entry['price'])
                volumes.insert(0, entry['volume'])
                # Use price as high/low if not available
                highs.insert(0, entry['price'])
                lows.insert(0, entry['price'])
                
            # Ensure we have at least some data
            if len(prices) < 5:
                # Duplicate the last price a few times
                prices = [prices[0]] * (5 - len(prices)) + prices
                volumes = [volumes[0]] * (5 - len(volumes)) + volumes
                highs = [highs[0]] * (5 - len(highs)) + highs
                lows = [lows[0]] * (5 - len(lows)) + lows
                
            # Generate technical analysis with timeframe parameter
            tech_analysis = TechnicalIndicators.analyze_technical_indicators(
                prices, volumes, highs, lows, timeframe
            )
            
            # Generate statistical forecast
            # Choose the best statistical model based on timeframe
            if timeframe == "1h":
                stat_forecast = StatisticalModels.arima_forecast(prices, forecast_steps=1, timeframe=timeframe)
            elif timeframe == "24h":
                try:
                    # Try Holt-Winters for daily data
                    stat_forecast = StatisticalModels.holt_winters_forecast(prices, forecast_steps=1, timeframe=timeframe)
                except:
                    # Fallback to ARIMA
                    stat_forecast = StatisticalModels.arima_forecast(prices, forecast_steps=1, timeframe=timeframe)
            else:  # 7d
                # For weekly, use weighted average forecast
                stat_forecast = StatisticalModels.weighted_average_forecast(prices, volumes, forecast_steps=1, timeframe=timeframe)
            
            # Generate machine learning forecast
            # Choose the best ML model based on timeframe and data availability
            if timeframe == "1h" and len(prices) >= 48:
                try:
                    # Try RandomForest for hourly with sufficient data
                    ml_forecast = MachineLearningModels.random_forest_forecast(
                        prices, volumes, forecast_steps=1, timeframe=timeframe
                    )
                except:
                    # Fallback to linear regression
                    ml_forecast = MachineLearningModels.linear_regression_forecast(
                        prices, volumes, forecast_steps=1, timeframe=timeframe
                    )
            elif timeframe == "24h" and len(prices) >= 60:
                try:
                    # Try LSTM for daily if we have enough data
                    ml_forecast = MachineLearningModels.lstm_forecast(
                        prices, forecast_steps=1, timeframe=timeframe
                    )
                except:
                    # Fallback to random forest
                    ml_forecast = MachineLearningModels.random_forest_forecast(
                        prices, volumes, forecast_steps=1, timeframe=timeframe
                    )
            else:
                # Default to linear regression for others
                ml_forecast = MachineLearningModels.linear_regression_forecast(
                    prices, volumes, forecast_steps=1, timeframe=timeframe
                )
            
            # Get market conditions
            market_conditions = self._generate_market_conditions(market_data, token)
            
            # Get recent predictions and their accuracy for this timeframe
            recent_predictions = self._get_recent_prediction_performance(token, timeframe)
            
            # Generate Claude-enhanced prediction if available
            if self.claude:
                logger.logger.debug(f"Generating Claude-enhanced {timeframe} prediction for {token}")
                prediction = self.claude.generate_enhanced_prediction(
                    token=token,
                    current_price=current_price,
                    technical_analysis=tech_analysis,
                    statistical_forecast=stat_forecast,
                    ml_forecast=ml_forecast,
                    timeframe=timeframe,
                    price_history_24h=historical_data,
                    market_conditions=market_conditions,
                    recent_predictions=recent_predictions
                )
            else:
                # Combine predictions manually if Claude not available
                logger.logger.debug(f"Generating manually combined {timeframe} prediction for {token}")
                prediction = self._combine_predictions(
                    token=token,
                    current_price=current_price,
                    technical_analysis=tech_analysis,
                    statistical_forecast=stat_forecast,
                    ml_forecast=ml_forecast,
                    market_conditions=market_conditions,
                    timeframe=timeframe
                )
                
            # Apply FOMO-inducing adjustments to the prediction
            prediction = self._apply_fomo_enhancement(prediction, current_price, tech_analysis, timeframe)
            
            # Store the prediction in the database
            self._store_prediction(token, prediction, timeframe)
            
            return prediction
            
        except Exception as e:
            logger.log_error(f"Prediction Generation - {token} ({timeframe})", str(e))
            
            # Return a simple fallback prediction
            return self._generate_fallback_prediction(token, market_data.get(token, {}), timeframe)
    
    def _generate_market_conditions(self, market_data: Dict[str, Any], excluded_token: str) -> Dict[str, Any]:
        """Generate overall market condition assessment"""
        # Remove the token itself from analysis
        filtered_data = {token: data for token, data in market_data.items() if token != excluded_token}
        
        if not filtered_data:
            return {"market_trend": "unknown", "btc_dominance": "unknown"}
            
        # Calculate market trend
        price_changes = [data.get('price_change_percentage_24h', 0) for data in filtered_data.values() 
                         if data.get('price_change_percentage_24h') is not None]
        
        if not price_changes:
            market_trend = "unknown"
        else:
            avg_change = sum(price_changes) / len(price_changes)
            if avg_change > 3:
                market_trend = "strongly bullish"
            elif avg_change > 1:
                market_trend = "bullish"
            elif avg_change < -3:
                market_trend = "strongly bearish"
            elif avg_change < -1:
                market_trend = "bearish"
            else:
                market_trend = "neutral"
                
        # Calculate BTC dominance if available
        btc_dominance = "unknown"
        if "BTC" in market_data:
            btc_market_cap = market_data["BTC"].get('market_cap', 0)
            total_market_cap = sum(data.get('market_cap', 0) for data in market_data.values() 
                                 if data.get('market_cap') is not None)
            if total_market_cap > 0:
                btc_dominance = f"{(btc_market_cap / total_market_cap) * 100:.1f}%"
                
        # Calculate market volatility
        if len(price_changes) > 1:
            market_volatility = f"{np.std(price_changes):.2f}"
        else:
            market_volatility = "unknown"
            
        # Group tokens by category (simple approach)
        layer1s = ["ETH", "SOL", "AVAX", "NEAR", "POL"]
        defi = ["UNI", "AAVE"]
        
        # Calculate sector performance
        sector_performance = {}
        
        # Layer 1s
        layer1_changes = [data.get('price_change_percentage_24h', 0) for token, data in filtered_data.items() 
                         if token in layer1s and data.get('price_change_percentage_24h') is not None]
        if layer1_changes:
            sector_performance["layer1"] = sum(layer1_changes) / len(layer1_changes)
            
        # DeFi
        defi_changes = [data.get('price_change_percentage_24h', 0) for token, data in filtered_data.items() 
                       if token in defi and data.get('price_change_percentage_24h') is not None]
        if defi_changes:
            sector_performance["defi"] = sum(defi_changes) / len(defi_changes)
            
        return {
            "market_trend": market_trend,
            "btc_dominance": btc_dominance,
            "market_volatility": market_volatility,
            "sector_performance": sector_performance
        }
    
    def _get_recent_prediction_performance(self, token: str, timeframe: str) -> List[Dict[str, Any]]:
        """Get recent prediction performance for token and timeframe"""
        try:
            # Get prediction performance from database
            performance = self.db.get_prediction_performance(token=token, timeframe=timeframe)
            
            if not performance:
                return []
                
            # Get recent prediction outcomes
            recent_outcomes = self.db.get_recent_prediction_outcomes(token=token, limit=10)
            
            # Filter for the specific timeframe
            filtered_outcomes = [outcome for outcome in recent_outcomes if outcome.get('timeframe') == timeframe]
            
            # Format for Claude input
            formatted_outcomes = []
            for outcome in filtered_outcomes:
                formatted_outcomes.append({
                    "prediction_value": outcome.get("prediction_value", 0),
                    "actual_outcome": outcome.get("actual_outcome", 0),
                    "was_correct": outcome.get("was_correct", 0) == 1,
                    "accuracy_percentage": outcome.get("accuracy_percentage", 0),
                    "evaluation_time": outcome.get("evaluation_time", "")
                })
                
            return formatted_outcomes
            
        except Exception as e:
            logger.log_error(f"Get Recent Prediction Performance - {token} ({timeframe})", str(e))
            return []
    
    def _combine_predictions(self, token: str, current_price: float, technical_analysis: Dict[str, Any],
                           statistical_forecast: Dict[str, Any], ml_forecast: Dict[str, Any],
                           market_conditions: Dict[str, Any], timeframe: str = "1h") -> Dict[str, Any]:
        """
        Manually combine predictions when Claude is not available
        Adjusted for different timeframes
        """
        # Extract forecasts
        stat_forecast = statistical_forecast.get("forecast", [current_price])[0]
        ml_forecast = ml_forecast.get("forecast", [current_price])[0]
        
        # Get technical trend
        trend = technical_analysis.get("overall_trend", "neutral")
        trend_strength = technical_analysis.get("trend_strength", 50)
        
        # Determine weights based on trend strength and timeframe
        # If trend is strong, give more weight to technical analysis
        # For longer timeframes, rely more on statistical and ML models
        if timeframe == "1h":
            # For hourly, technical indicators matter more
            if trend_strength > 70:
                tech_weight = 0.5
                stat_weight = 0.25
                ml_weight = 0.25
            elif trend_strength > 50:
                tech_weight = 0.4
                stat_weight = 0.3
                ml_weight = 0.3
            else:
                tech_weight = 0.2
                stat_weight = 0.4
                ml_weight = 0.4
        elif timeframe == "24h":
            # For daily, balance between technical and models
            if trend_strength > 70:
                tech_weight = 0.4
                stat_weight = 0.3
                ml_weight = 0.3
            elif trend_strength > 50:
                tech_weight = 0.35
                stat_weight = 0.35
                ml_weight = 0.3
            else:
                tech_weight = 0.25
                stat_weight = 0.4
                ml_weight = 0.35
        else:  # 7d
            # For weekly, models matter more than short-term indicators
            if trend_strength > 70:
                tech_weight = 0.35
                stat_weight = 0.35
                ml_weight = 0.3
            elif trend_strength > 50:
                tech_weight = 0.3
                stat_weight = 0.4
                ml_weight = 0.3
            else:
                tech_weight = 0.2
                stat_weight = 0.45
                ml_weight = 0.35
            
        # Calculate weighted prediction
        tech_prediction = current_price * (1 + (0.01 * (trend_strength - 50) / 50))
        if "bearish" in trend:
            tech_prediction = current_price * (1 - (0.01 * (trend_strength - 50) / 50))
            
        weighted_prediction = (
            tech_weight * tech_prediction + 
            stat_weight * stat_forecast + 
            ml_weight * ml_forecast
        )
        
        # Calculate confidence level
        # Higher trend strength = higher confidence
        # Longer timeframe = lower confidence
        base_confidence = {
            "1h": 65,
            "24h": 60,
            "7d": 55
        }.get(timeframe, 65)
        
        confidence_boost = min(20, (trend_strength - 50) * 0.4) if trend_strength > 50 else 0
        confidence = base_confidence + confidence_boost
        
        # Calculate price range based on timeframe
        # Wider ranges for longer timeframes
        volatility = technical_analysis.get("volatility", 5.0)
        base_range_factor = {
            "1h": 0.005,
            "24h": 0.015,
            "7d": 0.025
        }.get(timeframe, 0.005)
        
        range_factor = max(base_range_factor, min(base_range_factor * 4, volatility / 200))
        
        lower_bound = weighted_prediction * (1 - range_factor)
        upper_bound = weighted_prediction * (1 + range_factor)
        
        # Calculate percentage change
        percent_change = ((weighted_prediction / current_price) - 1) * 100
        
        # Determine sentiment
        # Adjust thresholds based on timeframe
        sentiment_thresholds = {
            "1h": 1.0,    # 1% for hourly
            "24h": 2.5,   # 2.5% for daily
            "7d": 5.0     # 5% for weekly
        }.get(timeframe, 1.0)
        
        if percent_change > sentiment_thresholds:
            sentiment = "BULLISH"
        elif percent_change < -sentiment_thresholds:
            sentiment = "BEARISH"
        else:
            sentiment = "NEUTRAL"
            
        # Generate rationale based on technical analysis and market conditions
        timeframe_desc = {
            "1h": "hour",
            "24h": "24 hours",
            "7d": "week"
        }.get(timeframe, timeframe)
        
        if sentiment == "BULLISH":
            rationale = f"Strong {trend} with confluence from statistical models suggest upward momentum in the next {timeframe_desc}."
        elif sentiment == "BEARISH":
            rationale = f"{trend.capitalize()} confirmed by multiple indicators, suggesting continued downward pressure over the next {timeframe_desc}."
        else:
            rationale = f"Mixed signals with {trend} but limited directional conviction for the {timeframe_desc} ahead."
            
        # Identify key factors
        key_factors = []
        
        # Add technical factor
        tech_signals = technical_analysis.get("signals", {})
        strongest_signal = max(tech_signals.items(), key=lambda x: 1 if x[1] in ["bullish", "bearish"] else 0)
        key_factors.append(f"{strongest_signal[0].upper()} {strongest_signal[1]}")
        
        # Add statistical factor
        stat_confidence = statistical_forecast.get("confidence_intervals", [{"80": [0, 0]}])[0]["80"]
        stat_range = abs(stat_confidence[1] - stat_confidence[0])
        key_factors.append(f"Statistical forecast: ${stat_forecast:.4f} ±{stat_range:.2f}")
        
        # Add market factor
        key_factors.append(f"Market trend: {market_conditions.get('market_trend', 'neutral')}")
        
        # Add timeframe-specific factors
        if timeframe == "24h":
            # Add ADX if available
            if "adx" in tech_signals:
                key_factors.append(f"ADX: {tech_signals['adx']}")
        elif timeframe == "7d":
            # Add sector performance for weekly
            sector_perf = market_conditions.get("sector_performance", {})
            if sector_perf:
                sector = max(sector_perf.items(), key=lambda x: abs(x[1]))
                key_factors.append(f"{sector[0].capitalize()} sector: {sector[1]:.1f}%")
        
        return {
            "prediction": {
                "price": weighted_prediction,
                "confidence": confidence,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "percent_change": percent_change,
                "timeframe": timeframe
            },
            "rationale": rationale,
            "sentiment": sentiment,
            "key_factors": key_factors,
            "model_weights": {
                "technical_analysis": tech_weight,
                "statistical_models": stat_weight,
                "machine_learning": ml_weight,
                "claude_enhanced": 0.0
            }
        }
    
    def _apply_fomo_enhancement(self, prediction: Dict[str, Any], current_price: float, 
                              tech_analysis: Dict[str, Any], timeframe: str = "1h") -> Dict[str, Any]:
        """
        Apply FOMO-inducing enhancements to predictions
        Makes ranges tighter and slightly exaggerates movement while staying realistic
        Adjusted for different timeframes
        """
        sentiment = prediction.get("sentiment", "NEUTRAL")
        original_price = prediction["prediction"]["price"]
        percent_change = prediction["prediction"]["percent_change"]
        
        # Adjust based on timeframe - don't modify very extreme predictions
        max_change_threshold = {
            "1h": 5.0,
            "24h": 10.0,
            "7d": 20.0
        }.get(timeframe, 5.0)
        
        # Don't modify predictions that are already very bullish or bearish
        if abs(percent_change) > max_change_threshold:
            return prediction
            
        # Get volatility from tech analysis
        volatility = tech_analysis.get("volatility", 5.0)
        
        # Enhance prediction based on sentiment and timeframe
        if sentiment == "BULLISH":
            # Slightly boost bullish predictions to generate FOMO
            # Boost amount increases with timeframe
            if timeframe == "1h":
                fomo_boost = max(0.2, min(0.8, volatility / 10))
            elif timeframe == "24h":
                fomo_boost = max(0.5, min(1.5, volatility / 8))
            else:  # 7d
                fomo_boost = max(1.0, min(2.5, volatility / 6))
                
            enhanced_price = original_price * (1 + (fomo_boost / 100))
            enhanced_pct = ((enhanced_price / current_price) - 1) * 100
            
            # Make ranges tighter
            base_range_factor = {
                "1h": 0.004,
                "24h": 0.01,
                "7d": 0.015
            }.get(timeframe, 0.004)
            
            range_factor = max(base_range_factor, min(base_range_factor * 3, volatility / 300))
            lower_bound = enhanced_price * (1 - range_factor)
            upper_bound = enhanced_price * (1 + range_factor)
            
            # Make sure upper bound is exciting enough based on timeframe
            min_upper_gain = {
                "1h": 1.01,
                "24h": 1.025,
                "7d": 1.05
            }.get(timeframe, 1.01)
            
            if (upper_bound / current_price) < min_upper_gain:
                upper_bound = current_price * min_upper_gain
                
        elif sentiment == "BEARISH":
            # Slightly exaggerate bearish predictions
            if timeframe == "1h":
                fomo_boost = max(0.2, min(0.8, volatility / 10))
            elif timeframe == "24h":
                fomo_boost = max(0.5, min(1.5, volatility / 8))
            else:  # 7d
                fomo_boost = max(1.0, min(2.5, volatility / 6))
                
            enhanced_price = original_price * (1 - (fomo_boost / 100))
            enhanced_pct = ((enhanced_price / current_price) - 1) * 100
            
            # Make ranges tighter
            base_range_factor = {
                "1h": 0.004,
                "24h": 0.01,
                "7d": 0.015
            }.get(timeframe, 0.004)
            
            range_factor = max(base_range_factor, min(base_range_factor * 3, volatility / 300))
            lower_bound = enhanced_price * (1 - range_factor)
            upper_bound = enhanced_price * (1 + range_factor)
            
            # Make sure lower bound is concerning enough based on timeframe
            min_lower_loss = {
                "1h": 0.99,
                "24h": 0.975,
                "7d": 0.95
            }.get(timeframe, 0.99)
            
            if (lower_bound / current_price) > min_lower_loss:
                lower_bound = current_price * min_lower_loss
                
        else:  # NEUTRAL
            # For neutral, make ranges a bit wider to be more accurate
            enhanced_price = original_price
            enhanced_pct = percent_change
            
            # Make ranges slightly tighter than original but not too tight
            base_range_factor = {
                "1h": 0.006,
                "24h": 0.015,
                "7d": 0.025
            }.get(timeframe, 0.006)
            
            range_factor = max(base_range_factor, min(base_range_factor * 3, volatility / 250))
            lower_bound = enhanced_price * (1 - range_factor)
            upper_bound = enhanced_price * (1 + range_factor)
            
        # Update prediction with enhanced values
        prediction["prediction"]["price"] = enhanced_price
        prediction["prediction"]["percent_change"] = enhanced_pct
        prediction["prediction"]["lower_bound"] = lower_bound
        prediction["prediction"]["upper_bound"] = upper_bound
        
        # Slightly boost confidence for FOMO
        # For longer timeframes, apply smaller confidence boost
        confidence_boost = {
            "1h": 5,
            "24h": 3,
            "7d": 2
        }.get(timeframe, 5)
        
        original_confidence = prediction["prediction"]["confidence"]
        prediction["prediction"]["confidence"] = min(85, original_confidence + confidence_boost)
        
        return prediction
    
    def _generate_fallback_prediction(self, token: str, token_data: Dict[str, Any], timeframe: str) -> Dict[str, Any]:
        """
        Generate a simple fallback prediction when other methods fail
        Adjusted for different timeframes
        """
        current_price = token_data.get('current_price', 0)
        if current_price == 0:
            current_price = 100  # Default if no price available
            
        # Generate a very simple prediction
        # Slight bullish bias for FOMO
        # Adjust based on timeframe
        if timeframe == "1h":
            prediction_change = 0.5  # 0.5% for 1 hour
            confidence = 70
            lower_factor = 0.995
            upper_factor = 1.015
        elif timeframe == "24h":
            prediction_change = 1.2  # 1.2% for 24 hours
            confidence = 65
            lower_factor = 0.985
            upper_factor = 1.035
        else:  # 7d
            prediction_change = 2.5  # 2.5% for 7 days
            confidence = 60
            lower_factor = 0.97
            upper_factor = 1.06
            
        prediction_price = current_price * (1 + prediction_change / 100)
        percent_change = prediction_change
        
        # Confidence level and range
        lower_bound = current_price * lower_factor
        upper_bound = current_price * upper_factor
        
        # Timeframe description for rationale
        timeframe_desc = {
            "1h": "hour",
            "24h": "24 hours",
            "7d": "week"
        }.get(timeframe, timeframe)
        
        return {
            "prediction": {
                "price": prediction_price,
                "confidence": confidence,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "percent_change": percent_change,
                "timeframe": timeframe
            },
            "rationale": f"Technical indicators suggest slight upward momentum for {token} in the next {timeframe_desc}.",
            "sentiment": "NEUTRAL",
            "key_factors": ["Technical analysis", "Recent price action", "Market conditions"],
            "model_weights": {
                "technical_analysis": 0.6,
                "statistical_models": 0.2,
                "machine_learning": 0.1,
                "claude_enhanced": 0.1
            },
            "is_fallback": True
        }
    
    def _store_prediction(self, token: str, prediction: Dict[str, Any], timeframe: str) -> None:
        """Store prediction in database"""
        prediction_data = prediction["prediction"]
        
        # Set appropriate expiration time based on timeframe
        if timeframe == "1h":
            expiration_time = datetime.now() + timedelta(hours=1)
        elif timeframe == "24h":
            expiration_time = datetime.now() + timedelta(hours=24)
        elif timeframe == "7d":
            expiration_time = datetime.now() + timedelta(days=7)
        else:
            expiration_time = datetime.now() + timedelta(hours=1)  # Default to 1h
        
        # Store in database
        try:
            conn, cursor = self.db._get_connection()
            
            cursor.execute("""
                INSERT INTO price_predictions (
                    timestamp, token, timeframe, prediction_type,
                    prediction_value, confidence_level, lower_bound, upper_bound,
                    prediction_rationale, method_weights, model_inputs, technical_signals,
                    expiration_time
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(),
                token,
                timeframe,
                "price",
                prediction_data["price"],
                prediction_data["confidence"],
                prediction_data["lower_bound"],
                prediction_data["upper_bound"],
                prediction["rationale"],
                json.dumps(prediction.get("model_weights", {})),
                json.dumps(prediction.get("inputs", {})),
                json.dumps(prediction.get("key_factors", [])),
                expiration_time
            ))
            
            conn.commit()
            logger.logger.debug(f"Stored {timeframe} prediction for {token}")
            
        except Exception as e:
            logger.log_error(f"Store Prediction - {token} ({timeframe})", str(e))
            if conn:
                conn.rollback()
    
    def evaluate_predictions(self) -> None:
        """
        Evaluate expired predictions and calculate accuracy
        """
        try:
            conn, cursor = self.db._get_connection()
            
            # Get expired but unevaluated predictions
            cursor.execute("""
                SELECT p.*, o.id as outcome_id 
                FROM price_predictions p
                LEFT JOIN prediction_outcomes o ON p.id = o.prediction_id
                WHERE p.expiration_time <= datetime('now')
                AND o.id IS NULL
            """)
            
            expired_predictions = cursor.fetchall()
            
            for prediction in expired_predictions:
                token = prediction["token"]
                prediction_value = prediction["prediction_value"]
                lower_bound = prediction["lower_bound"]
                upper_bound = prediction["upper_bound"]
                timeframe = prediction["timeframe"]
                
                # Get the actual price at expiration time
                cursor.execute("""
                    SELECT price
                    FROM market_data
                    WHERE chain = ?
                    AND timestamp <= ?
                    ORDER BY timestamp DESC
                    LIMIT 1
                """, (token, prediction["expiration_time"]))
                
                actual_result = cursor.fetchone()
                
                if not actual_result:
                    logger.logger.warning(f"No actual price found for {token} at evaluation time")
                    continue
                    
                actual_price = actual_result["price"]
                
                # Calculate accuracy
                price_diff = abs(actual_price - prediction_value)
                accuracy_percentage = (1 - (price_diff / prediction_value)) * 100
                
                # Determine if prediction was correct
                was_correct = lower_bound <= actual_price <= upper_bound
                
                # Calculate deviation
                deviation = ((actual_price / prediction_value) - 1) * 100
                
                # Store evaluation result
                cursor.execute("""
                    INSERT INTO prediction_outcomes (
                        prediction_id, actual_outcome, accuracy_percentage,
                        was_correct, evaluation_time, deviation_from_prediction,
                        market_conditions
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    prediction["id"],
                    actual_price,
                    accuracy_percentage,
                    1 if was_correct else 0,
                    datetime.now(),
                    deviation,
                    json.dumps({"evaluation_time": datetime.now().isoformat()})
                ))
                
                # Update performance summary
                self._update_prediction_performance(token, timeframe, prediction["prediction_type"], was_correct, abs(deviation))
                
            conn.commit()
            logger.logger.info(f"Evaluated {len(expired_predictions)} expired predictions")
            
        except Exception as e:
            logger.log_error("Prediction Evaluation", str(e))
            if conn:
                conn.rollback()

    def _update_prediction_performance(self, token: str, timeframe: str, prediction_type: str, was_correct: bool, deviation: float) -> None:
        """Update prediction performance summary"""
        conn, cursor = self.db._get_connection()
        
        try:
            # Check if performance record exists
            cursor.execute("""
                SELECT * FROM prediction_performance
                WHERE token = ? AND timeframe = ? AND prediction_type = ?
            """, (token, timeframe, prediction_type))
            
            performance = cursor.fetchone()
            
            if performance:
                # Update existing record
                performance_dict = dict(performance)
                total_predictions = performance_dict["total_predictions"] + 1
                correct_predictions = performance_dict["correct_predictions"] + (1 if was_correct else 0)
                accuracy_rate = (correct_predictions / total_predictions) * 100
                
                # Update average deviation (weighted average)
                avg_deviation = (performance_dict["avg_deviation"] * performance_dict["total_predictions"] + deviation) / total_predictions
                
                cursor.execute("""
                    UPDATE prediction_performance
                    SET total_predictions = ?,
                        correct_predictions = ?,
                        accuracy_rate = ?,
                        avg_deviation = ?,
                        updated_at = ?
                    WHERE id = ?
                """, (
                    total_predictions,
                    correct_predictions,
                    accuracy_rate,
                    avg_deviation,
                    datetime.now(),
                    performance_dict["id"]
                ))
                
            else:
                # Create new record
                cursor.execute("""
                    INSERT INTO prediction_performance (
                        token, timeframe, prediction_type, total_predictions,
                        correct_predictions, accuracy_rate, avg_deviation, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    token,
                    timeframe,
                    prediction_type,
                    1,
                    1 if was_correct else 0,
                    100 if was_correct else 0,
                    deviation,
                    datetime.now()
                ))
                
        except Exception as e:
            logger.log_error(f"Update Prediction Performance - {token}", str(e))
            raise
            
    def get_model_accuracy_by_timeframe(self) -> Dict[str, Dict[str, Any]]:
        """
        Get model accuracy statistics for each timeframe
        """
        try:
            results = {}
            
            for timeframe in ["1h", "24h", "7d"]:
                timeframe_stats = self.db.get_prediction_accuracy_by_model(timeframe=timeframe, days=30)
                
                if timeframe_stats:
                    results[timeframe] = timeframe_stats
                    
            return results
            
        except Exception as e:
            logger.log_error("Get Model Accuracy By Timeframe", str(e))
            return {}
