#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Optional, Any, Union, List, Tuple
import sys
import os
import time
import requests
import re
import numpy as np
from datetime import datetime, timedelta
import anthropic
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.common.keys import Keys
import random
import statistics

from utils.logger import logger
from utils.browser import browser
from config import config
from coingecko_handler import CoinGeckoHandler
from mood_config import MoodIndicators, determine_advanced_mood, Mood, MemePhraseGenerator
from meme_phrases import MEME_PHRASES
from prediction_engine import PredictionEngine

class CryptoAnalysisBot:
    def __init__(self) -> None:
        self.browser = browser
        self.config = config
        self.claude_client = anthropic.Client(api_key=self.config.CLAUDE_API_KEY)
        self.past_predictions = []
        self.meme_phrases = MEME_PHRASES
        self.last_check_time = datetime.now()
        self.last_market_data = {}
        
        # Initialize prediction engine with database and Claude API key
        self.prediction_engine = PredictionEngine(
            database=self.config.db,
            claude_api_key=self.config.CLAUDE_API_KEY
        )
        
        # Initialize CoinGecko handler with 60s cache duration
        self.coingecko = CoinGeckoHandler(
            base_url=self.config.COINGECKO_BASE_URL,
            cache_duration=60
        )
        
        # Target chains to analyze
        self.target_chains = {
            'BTC': 'bitcoin',
            'ETH': 'ethereum',
            'SOL': 'solana',
            'XRP': 'ripple',
            'BNB': 'binancecoin',
            'AVAX': 'avalanche-2',
            'DOT': 'polkadot',
            'UNI': 'uniswap',
            'NEAR': 'near',
            'AAVE': 'aave',
            'FIL': 'filecoin',
            'POL': 'matic-network',
            'KAITO': 'kaito'  # Kept in the list but not given special treatment
        }

        # All tokens for reference and comparison
        self.reference_tokens = list(self.target_chains.keys())
        
        # Chain name mapping for display
        self.chain_name_mapping = self.target_chains.copy()
        
        self.CORRELATION_THRESHOLD = 0.75  
        self.VOLUME_THRESHOLD = 0.60  
        self.TIME_WINDOW = 24
        
        # Smart money thresholds
        self.SMART_MONEY_VOLUME_THRESHOLD = 1.5  # 50% above average
        self.SMART_MONEY_ZSCORE_THRESHOLD = 2.0  # 2 standard deviations
        
        logger.log_startup()
    def _get_historical_volume_data(self, chain: str, minutes: int = None) -> List[Dict[str, Any]]:
        """
        Get historical volume data for the specified window period
        """
        try:
            # Use config's window minutes if not specified
            if minutes is None:
                minutes = self.config.VOLUME_WINDOW_MINUTES
                
            window_start = datetime.now() - timedelta(minutes=minutes)
            query = """
                SELECT timestamp, volume
                FROM market_data
                WHERE chain = ? AND timestamp >= ?
                ORDER BY timestamp DESC
            """
            
            conn = self.config.db.conn
            cursor = conn.cursor()
            cursor.execute(query, (chain, window_start))
            results = cursor.fetchall()
            
            volume_data = [
                {
                    'timestamp': datetime.fromisoformat(row[0]),
                    'volume': float(row[1])
                }
                for row in results
            ]
            
            logger.logger.debug(
                f"Retrieved {len(volume_data)} volume data points for {chain} "
                f"over last {minutes} minutes"
            )
            
            return volume_data
            
        except Exception as e:
            logger.log_error(f"Historical Volume Data - {chain}", str(e))
            return []
            
    def _is_duplicate_analysis(self, new_tweet: str, last_posts: List[str]) -> bool:
        """
        Enhanced duplicate detection with time-based thresholds.
        Applies different checks based on how recently similar content was posted:
        - Very recent posts (< 15 min): Check for exact matches
        - Recent posts (15-30 min): Check for high similarity
        - Older posts (> 30 min): Allow similar content
        """
        try:
            # Log that we're using enhanced duplicate detection
            logger.logger.info("Using enhanced time-based duplicate detection")
            
            # Define time windows for different levels of duplicate checking
            VERY_RECENT_WINDOW_MINUTES = 15
            RECENT_WINDOW_MINUTES = 30
            
            # Define similarity thresholds
            HIGH_SIMILARITY_THRESHOLD = 0.85  # 85% similar for recent posts
            
            # 1. Check for exact matches in very recent database entries (last 15 minutes)
            conn = self.config.db.conn
            cursor = conn.cursor()
            
            # Very recent exact duplicates check
            cursor.execute("""
                SELECT content FROM posted_content 
                WHERE timestamp >= datetime('now', '-' || ? || ' minutes')
            """, (VERY_RECENT_WINDOW_MINUTES,))
            
            very_recent_posts = [row[0] for row in cursor.fetchall()]
            
            # Check for exact matches in very recent posts
            for post in very_recent_posts:
                if post.strip() == new_tweet.strip():
                    logger.logger.info(f"Exact duplicate detected within last {VERY_RECENT_WINDOW_MINUTES} minutes")
                    return True
            
            # 2. Check for high similarity in recent posts (15-30 minutes)
            cursor.execute("""
                SELECT content FROM posted_content 
                WHERE timestamp >= datetime('now', '-' || ? || ' minutes')
                AND timestamp < datetime('now', '-' || ? || ' minutes')
            """, (RECENT_WINDOW_MINUTES, VERY_RECENT_WINDOW_MINUTES))
            
            recent_posts = [row[0] for row in cursor.fetchall()]
            
            # Calculate similarity for recent posts
            new_content = new_tweet.lower()
            
            for post in recent_posts:
                post_content = post.lower()
                
                # Calculate a simple similarity score based on word overlap
                new_words = set(new_content.split())
                post_words = set(post_content.split())
                
                if new_words and post_words:
                    overlap = len(new_words.intersection(post_words))
                    similarity = overlap / max(len(new_words), len(post_words))
                    
                    # Apply high similarity threshold for recent posts
                    if similarity > HIGH_SIMILARITY_THRESHOLD:
                        logger.logger.info(f"High similarity ({similarity:.2f}) detected within last {RECENT_WINDOW_MINUTES} minutes")
                        return True
            
            # 3. Also check exact duplicates in last posts from Twitter
            # This prevents double-posting in case of database issues
            for post in last_posts:
                if post.strip() == new_tweet.strip():
                    logger.logger.info("Exact duplicate detected in recent Twitter posts")
                    return True
            
            # If we get here, it's not a duplicate according to our criteria
            logger.logger.info("No duplicates detected with enhanced time-based criteria")
            return False
            
        except Exception as e:
            logger.log_error("Duplicate Check", str(e))
            # If the duplicate check fails, allow the post to be safe
            logger.logger.warning("Duplicate check failed, allowing post to proceed")
            return False

    def _login_to_twitter(self) -> bool:
        """Log into Twitter with enhanced verification"""
        try:
            logger.logger.info("Starting Twitter login")
            self.browser.driver.set_page_load_timeout(45)
            self.browser.driver.get('https://twitter.com/login')
            time.sleep(5)

            username_field = WebDriverWait(self.browser.driver, 20).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "input[autocomplete='username']"))
            )
            username_field.click()
            time.sleep(1)
            username_field.send_keys(self.config.TWITTER_USERNAME)
            time.sleep(2)

            next_button = WebDriverWait(self.browser.driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//span[text()='Next']"))
            )
            next_button.click()
            time.sleep(3)

            password_field = WebDriverWait(self.browser.driver, 20).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "input[type='password']"))
            )
            password_field.click()
            time.sleep(1)
            password_field.send_keys(self.config.TWITTER_PASSWORD)
            time.sleep(2)

            login_button = WebDriverWait(self.browser.driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//span[text()='Log in']"))
            )
            login_button.click()
            time.sleep(10) 

            return self._verify_login()

        except Exception as e:
            logger.log_error("Twitter Login", str(e))
            return False

    def _verify_login(self) -> bool:
        """Verify Twitter login success"""
        try:
            verification_methods = [
                lambda: WebDriverWait(self.browser.driver, 30).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, '[data-testid="SideNav_NewTweet_Button"]'))
                ),
                lambda: WebDriverWait(self.browser.driver, 30).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, '[data-testid="AppTabBar_Profile_Link"]'))
                ),
                lambda: any(path in self.browser.driver.current_url 
                          for path in ['home', 'twitter.com/home'])
            ]
            
            for method in verification_methods:
                try:
                    if method():
                        return True
                except:
                    continue
            
            return False
            
        except Exception as e:
            logger.log_error("Login Verification", str(e))
            return False
            
    def _post_analysis(self, tweet_text: str) -> bool:
        """Post analysis to Twitter with robust button handling"""
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                self.browser.driver.get('https://twitter.com/compose/tweet')
                time.sleep(3)
                
                text_area = WebDriverWait(self.browser.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, '[data-testid="tweetTextarea_0"]'))
                )
                text_area.click()
                time.sleep(1)
                
                # Ensure tweet text only contains BMP characters
                safe_tweet_text = ''.join(char for char in tweet_text if ord(char) < 0x10000)
                
                # Simply send the tweet text directly - no handling of hashtags needed
                text_area.send_keys(safe_tweet_text)
                time.sleep(2)

                post_button = None
                button_locators = [
                    (By.CSS_SELECTOR, '[data-testid="tweetButton"]'),
                    (By.XPATH, "//div[@role='button'][contains(., 'Post')]"),
                    (By.XPATH, "//span[text()='Post']")
                ]

                for locator in button_locators:
                    try:
                        post_button = WebDriverWait(self.browser.driver, 5).until(
                            EC.element_to_be_clickable(locator)
                        )
                        if post_button:
                            break
                    except:
                        continue

                if post_button:
                    self.browser.driver.execute_script("arguments[0].scrollIntoView(true);", post_button)
                    time.sleep(1)
                    self.browser.driver.execute_script("arguments[0].click();", post_button)
                    time.sleep(5)
                    logger.logger.info("Tweet posted successfully")
                    return True
                else:
                    logger.logger.error("Could not find post button")
                    retry_count += 1
                    time.sleep(2)
                    
            except Exception as e:
                logger.logger.error(f"Tweet posting error, attempt {retry_count + 1}: {str(e)}")
                retry_count += 1
                wait_time = retry_count * 10
                logger.logger.warning(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
                continue
        
        logger.log_error("Tweet Creation", "Maximum retries reached")
        return False
        
    def _cleanup(self) -> None:
        """Cleanup resources"""
        try:
            if self.browser:
                logger.logger.info("Closing browser...")
                try:
                    self.browser.close_browser()
                    time.sleep(1)
                except Exception as e:
                    logger.logger.warning(f"Error during browser close: {str(e)}")
                    
            if self.config:
                self.config.cleanup()
                
            logger.log_shutdown()
        except Exception as e:
            logger.log_error("Cleanup", str(e))

    def _get_last_posts(self) -> List[str]:
        """Get last 10 posts to check for duplicates"""
        try:
            self.browser.driver.get(f'https://twitter.com/{self.config.TWITTER_USERNAME}')
            time.sleep(3)
            
            posts = WebDriverWait(self.browser.driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, '[data-testid="tweetText"]'))
            )
            
            return [post.text for post in posts[:10]]
        except Exception as e:
            logger.log_error("Get Last Posts", str(e))
            return []

    def _get_crypto_data(self) -> Optional[Dict[str, Any]]:
        """Fetch crypto data from CoinGecko with retries"""
        try:
            params = {
                **self.config.get_coingecko_params(),
                'ids': ','.join(self.target_chains.values()), 
                'sparkline': True 
            }
            
            data = self.coingecko.get_market_data(params)
            if not data:
                logger.logger.error("Failed to fetch market data from CoinGecko")
                return None
                
            formatted_data = {
                coin['symbol'].upper(): {
                    'current_price': coin['current_price'],
                    'volume': coin['total_volume'],
                    'price_change_percentage_24h': coin['price_change_percentage_24h'],
                    'sparkline': coin.get('sparkline_in_7d', {}).get('price', []),
                    'market_cap': coin['market_cap'],
                    'market_cap_rank': coin['market_cap_rank'],
                    'total_supply': coin.get('total_supply'),
                    'max_supply': coin.get('max_supply'),
                    'circulating_supply': coin.get('circulating_supply'),
                    'ath': coin.get('ath'),
                    'ath_change_percentage': coin.get('ath_change_percentage')
                } for coin in data
            }
            
            # Map to correct symbol if needed (particularly for POL which might return as MATIC)
            symbol_corrections = {'MATIC': 'POL'}
            for old_sym, new_sym in symbol_corrections.items():
                if old_sym in formatted_data and new_sym not in formatted_data:
                    formatted_data[new_sym] = formatted_data[old_sym]
                    logger.logger.debug(f"Mapped {old_sym} data to {new_sym}")
            
            # Log API usage statistics
            stats = self.coingecko.get_request_stats()
            logger.logger.debug(
                f"CoinGecko API stats - Daily requests: {stats['daily_requests']}, "
                f"Failed: {stats['failed_requests']}, Cache size: {stats['cache_size']}"
            )
            
            # Store market data in database
            for chain, chain_data in formatted_data.items():
                self.config.db.store_market_data(chain, chain_data)
            
            # Check if all data was retrieved
            missing_tokens = [token for token in self.reference_tokens if token not in formatted_data]
            if missing_tokens:
                logger.logger.warning(f"Missing data for tokens: {', '.join(missing_tokens)}")
                
                # Try fallback mechanism for missing tokens
                if 'POL' in missing_tokens and 'MATIC' in formatted_data:
                    formatted_data['POL'] = formatted_data['MATIC']
                    missing_tokens.remove('POL')
                    logger.logger.info("Applied fallback for POL using MATIC data")
                
            logger.logger.info(f"Successfully fetched crypto data for {', '.join(formatted_data.keys())}")
            return formatted_data
                
        except Exception as e:
            logger.log_error("CoinGecko API", str(e))
            return None

    def _analyze_volume_trend(self, current_volume: float, historical_data: List[Dict[str, Any]]) -> Tuple[float, str]:
        """
        Analyze volume trend over the window period
        Returns (percentage_change, trend_description)
        """
        if not historical_data:
            return 0.0, "insufficient_data"
            
        try:
            # Calculate average volume excluding the current volume
            historical_volumes = [entry['volume'] for entry in historical_data]
            avg_volume = statistics.mean(historical_volumes) if historical_volumes else current_volume
            
            # Calculate percentage change
            volume_change = ((current_volume - avg_volume) / avg_volume) * 100
            
            # Determine trend
            if volume_change >= self.config.VOLUME_TREND_THRESHOLD:
                trend = "significant_increase"
            elif volume_change <= -self.config.VOLUME_TREND_THRESHOLD:
                trend = "significant_decrease"
            elif volume_change >= 5:  # Smaller but notable increase
                trend = "moderate_increase"
            elif volume_change <= -5:  # Smaller but notable decrease
                trend = "moderate_decrease"
            else:
                trend = "stable"
                
            logger.logger.debug(
                f"Volume trend analysis: {volume_change:.2f}% change from average. "
                f"Current: {current_volume:,.0f}, Avg: {avg_volume:,.0f}, "
                f"Trend: {trend}"
            )
            
            return volume_change, trend
            
        except Exception as e:
            logger.log_error("Volume Trend Analysis", str(e))
            return 0.0, "error"

    def _analyze_smart_money_indicators(self, token: str, token_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze potential smart money movements in a token
        Look for unusual volume spikes, price-volume divergence, and accumulation patterns
        """
        try:
            # Get historical data over multiple timeframes
            hourly_data = self._get_historical_volume_data(token, minutes=60)
            daily_data = self._get_historical_volume_data(token, minutes=1440)
            
            current_volume = token_data['volume']
            current_price = token_data['current_price']
            
            # Volume anomaly detection
            hourly_volumes = [entry['volume'] for entry in hourly_data]
            daily_volumes = [entry['volume'] for entry in daily_data]
            
            # Calculate baselines
            avg_hourly_volume = statistics.mean(hourly_volumes) if hourly_volumes else current_volume
            avg_daily_volume = statistics.mean(daily_volumes) if daily_volumes else current_volume
            
            # Volume Z-score (how many standard deviations from mean)
            hourly_std = statistics.stdev(hourly_volumes) if len(hourly_volumes) > 1 else 1
            volume_z_score = (current_volume - avg_hourly_volume) / hourly_std if hourly_std != 0 else 0
            
            # Price-volume divergence
            # (Price going down while volume increasing suggests accumulation)
            price_direction = 1 if token_data['price_change_percentage_24h'] > 0 else -1
            volume_direction = 1 if current_volume > avg_daily_volume else -1
            
            # Divergence detected when price and volume move in opposite directions
            divergence = (price_direction != volume_direction)
            
            # Check for abnormal volume with minimal price movement (potential accumulation)
            stealth_accumulation = (abs(token_data['price_change_percentage_24h']) < 2) and (current_volume > avg_daily_volume * 1.5)
            
            # Calculate volume profile - percentage of volume in each hour
            volume_profile = {}
            if hourly_data:
                for i in range(24):
                    hour_window = datetime.now() - timedelta(hours=i+1)
                    hour_volume = sum(entry['volume'] for entry in hourly_data if hour_window <= entry['timestamp'] <= hour_window + timedelta(hours=1))
                    volume_profile[f"hour_{i+1}"] = hour_volume
            
            # Detect unusual trading hours (potential institutional activity)
            total_volume = sum(volume_profile.values()) if volume_profile else 0
            unusual_hours = []
            
            if total_volume > 0:
                for hour, vol in volume_profile.items():
                    hour_percentage = (vol / total_volume) * 100
                    if hour_percentage > 15:  # More than 15% of daily volume in a single hour
                        unusual_hours.append(hour)
            
            # Detect volume clusters (potential accumulation zones)
            volume_cluster_detected = False
            if len(hourly_volumes) >= 3:
                for i in range(len(hourly_volumes)-2):
                    if all(vol > avg_hourly_volume * 1.3 for vol in hourly_volumes[i:i+3]):
                        volume_cluster_detected = True
                        break
            
            # Results
            return {
                'volume_z_score': volume_z_score,
                'price_volume_divergence': divergence,
                'stealth_accumulation': stealth_accumulation,
                'abnormal_volume': abs(volume_z_score) > self.SMART_MONEY_ZSCORE_THRESHOLD,
                'volume_vs_hourly_avg': (current_volume / avg_hourly_volume) - 1,
                'volume_vs_daily_avg': (current_volume / avg_daily_volume) - 1,
                'unusual_trading_hours': unusual_hours,
                'volume_cluster_detected': volume_cluster_detected
            }
        except Exception as e:
            logger.log_error(f"Smart Money Analysis - {token}", str(e))
            return {}

    def _analyze_token_vs_market(self, token: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze token performance relative to the overall crypto market
        """
        try:
            token_data = market_data.get(token, {})
            if not token_data:
                return {}
                
            # Filter out the token itself from reference tokens to avoid self-comparison
            reference_tokens = [t for t in self.reference_tokens if t != token]
            
            # Compare 24h performance
            market_avg_change = statistics.mean([
                market_data.get(ref_token, {}).get('price_change_percentage_24h', 0) 
                for ref_token in reference_tokens
                if ref_token in market_data
            ])
            
            performance_diff = token_data['price_change_percentage_24h'] - market_avg_change
            
            # Compare volume growth
            market_avg_volume_change = statistics.mean([
                self._analyze_volume_trend(
                    market_data.get(ref_token, {}).get('volume', 0),
                    self._get_historical_volume_data(ref_token)
                )[0]
                for ref_token in reference_tokens
                if ref_token in market_data
            ])
            
            token_volume_change = self._analyze_volume_trend(
                token_data['volume'],
                self._get_historical_volume_data(token)
            )[0]
            
            volume_growth_diff = token_volume_change - market_avg_volume_change
            
            # Calculate correlation with each reference token
            correlations = {}
            for ref_token in reference_tokens:
                if ref_token in market_data:
                    # Simple correlation based on 24h change direction
                    token_direction = 1 if token_data['price_change_percentage_24h'] > 0 else -1
                    ref_token_direction = 1 if market_data[ref_token]['price_change_percentage_24h'] > 0 else -1
                    correlated = token_direction == ref_token_direction
                    
                    correlations[ref_token] = {
                        'correlated': correlated,
                        'token_change': token_data['price_change_percentage_24h'],
                        'ref_token_change': market_data[ref_token]['price_change_percentage_24h']
                    }
            
            # Determine if token is outperforming the market
            outperforming = performance_diff > 0
            
            # Store for any token using the generic method
            self.config.db.store_token_market_comparison(
                token,
                performance_diff,
                volume_growth_diff,
                outperforming,
                correlations
            )
            
            return {
                'vs_market_avg_change': performance_diff,
                'vs_market_volume_growth': volume_growth_diff,
                'correlations': correlations,
                'outperforming_market': outperforming
            }
            
        except Exception as e:
            logger.log_error(f"Token vs Market Analysis - {token}", str(e))
            return {}
        
    def _calculate_momentum_score(self, token: str, market_data: Dict[str, Any]) -> float:
        """Calculate a momentum score (0-100) for a token based on various metrics"""
        try:
            token_data = market_data.get(token, {})
            if not token_data:
              return 50.0  # Neutral score
            
            # Get basic metrics
            price_change = token_data.get('price_change_percentage_24h', 0)
            volume = token_data.get('volume', 0)
        
            # Get historical volume for volume change
            historical_volume = self._get_historical_volume_data(token)
            volume_change, _ = self._analyze_volume_trend(volume, historical_volume)
        
            # Get smart money indicators
            smart_money = self._analyze_smart_money_indicators(token, token_data)
        
            # Get market comparison
            vs_market = self._analyze_token_vs_market(token, market_data)
        
            # Calculate score components (0-20 points each)
            price_score = min(20, max(0, (price_change + 5) * 2))  # -5% to +5% → 0-20 points
        
            volume_score = min(20, max(0, (volume_change + 10) * 1))  # -10% to +10% → 0-20 points
        
            # Smart money score
            smart_money_score = 0
            if smart_money.get('abnormal_volume', False):
                smart_money_score += 5
            if smart_money.get('stealth_accumulation', False):
                smart_money_score += 5
            if smart_money.get('volume_cluster_detected', False):
                smart_money_score += 5
            if smart_money.get('volume_z_score', 0) > 1.0:
                smart_money_score += 5
            smart_money_score = min(20, smart_money_score)
        
            # Market comparison score
            market_score = 0
            if vs_market.get('outperforming_market', False):
                market_score += 10
            market_score += min(10, max(0, (vs_market.get('vs_market_avg_change', 0) + 5)))
            market_score = min(20, market_score)
        
            # Trend consistency score
            trend_score = 20 if all([price_score > 10, volume_score > 10, smart_money_score > 10, market_score > 10]) else 0
        
            # Calculate total score (0-100)
            total_score = price_score + volume_score + smart_money_score + market_score + trend_score
        
            return total_score
        
        except Exception as e:
            logger.log_error(f"Momentum Score - {token}", str(e))
            return 50.0  # Neutral score on error

    def _prioritize_tokens(self, available_tokens: List[str], market_data: Dict[str, Any]) -> List[str]:
        """Prioritize tokens based on momentum score and other factors"""
        try:
            token_priorities = []
        
            for token in available_tokens:
                # Calculate token-specific priority score
                momentum_score = self._calculate_momentum_score(token, market_data)
            
                # Get latest prediction time for this token
                last_prediction = self.config.db.get_active_predictions(token=token, timeframe="1h")
                hours_since_prediction = 24  # Default high value
            
                if last_prediction:
                    last_time = datetime.fromisoformat(last_prediction[0]["timestamp"])
                    hours_since_prediction = (datetime.now() - last_time).total_seconds() / 3600
            
                # Priority score combines momentum and time since last prediction
                priority_score = momentum_score + (hours_since_prediction * 2)
            
                token_priorities.append((token, priority_score))
        
            # Sort by priority score (highest first)
            sorted_tokens = [t[0] for t in sorted(token_priorities, key=lambda x: x[1], reverse=True)]
        
            return sorted_tokens
        
        except Exception as e:
            logger.log_error("Token Prioritization", str(e))
            return available_tokens  # Return original list on error        

    def _calculate_correlations(self, token: str, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate token correlations with the market"""
        try:
            token_data = market_data.get(token, {})
            if not token_data:
                return {}
                
            # Filter out the token itself from reference tokens to avoid self-comparison
            reference_tokens = [t for t in self.reference_tokens if t != token]
            
            correlations = {}
            
            # Calculate correlation with each reference token
            for ref_token in reference_tokens:
                if ref_token not in market_data:
                    continue
                    
                ref_data = market_data[ref_token]
                
                # Price correlation (simplified)
                price_correlation = abs(
                    token_data['price_change_percentage_24h'] - 
                    ref_data['price_change_percentage_24h']
                ) / max(abs(token_data['price_change_percentage_24h']), 
                       abs(ref_data['price_change_percentage_24h']))
                
                # Volume correlation (simplified)
                volume_correlation = abs(
                    (token_data['volume'] - ref_data['volume']) / 
                    max(token_data['volume'], ref_data['volume'])
                )
                
                correlations[f'price_correlation_{ref_token}'] = 1 - price_correlation
                correlations[f'volume_correlation_{ref_token}'] = 1 - volume_correlation
            
            # Calculate average correlations
            price_correlations = [v for k, v in correlations.items() if 'price_correlation_' in k]
            volume_correlations = [v for k, v in correlations.items() if 'volume_correlation_' in k]
            
            correlations['avg_price_correlation'] = statistics.mean(price_correlations) if price_correlations else 0
            correlations['avg_volume_correlation'] = statistics.mean(volume_correlations) if volume_correlations else 0
            
            # Store correlation data for any token using the generic method
            self.config.db.store_token_correlations(token, correlations)
            
            logger.logger.debug(
                f"{token} correlations calculated - Avg Price: {correlations['avg_price_correlation']:.2f}, "
                f"Avg Volume: {correlations['avg_volume_correlation']:.2f}"
            )
            
            return correlations
            
        except Exception as e:
            logger.log_error(f"Correlation Calculation - {token}", str(e))
            return {
                'avg_price_correlation': 0.0,
                'avg_volume_correlation': 0.0
            }

    def _track_prediction(self, token: str, prediction: Dict[str, Any], relevant_tokens: List[str]) -> None:
        """Track predictions for future spicy callbacks"""
        MAX_PREDICTIONS = 20  
        current_prices = {chain: prediction.get(f'{chain.upper()}_price', 0) for chain in relevant_tokens if f'{chain.upper()}_price' in prediction}
        
        self.past_predictions.append({
            'timestamp': datetime.now(),
            'token': token,
            'prediction': prediction['analysis'],
            'prices': current_prices,
            'sentiment': prediction['sentiment'],
            'outcome': None
        })
        
        # Keep only predictions from the last 24 hours, up to MAX_PREDICTIONS
        self.past_predictions = [p for p in self.past_predictions 
                               if (datetime.now() - p['timestamp']).total_seconds() < 86400]
        
        # Trim to max predictions if needed
        if len(self.past_predictions) > MAX_PREDICTIONS:
            self.past_predictions = self.past_predictions[-MAX_PREDICTIONS:]
            
    def _validate_past_prediction(self, prediction: Dict[str, Any], current_prices: Dict[str, float]) -> str:
        """Check if a past prediction was hilariously wrong"""
        sentiment_map = {
            'bullish': 1,
            'bearish': -1,
            'neutral': 0,
            'volatile': 0,
            'recovering': 0.5
        }
        
        wrong_tokens = []
        for token, old_price in prediction['prices'].items():
            if token in current_prices and old_price > 0:
                price_change = ((current_prices[token] - old_price) / old_price) * 100
                
                # Get sentiment for this token
                token_sentiment_key = token.upper() if token.upper() in prediction['sentiment'] else token
                token_sentiment_value = prediction['sentiment'].get(token_sentiment_key)
                
                # Handle nested dictionary structure
                if isinstance(token_sentiment_value, dict) and 'mood' in token_sentiment_value:
                    token_sentiment = sentiment_map.get(token_sentiment_value['mood'], 0)
                else:
                    token_sentiment = sentiment_map.get(token_sentiment_value, 0)
                
                # A prediction is wrong if:
                # 1. Bullish but price dropped more than 2%
                # 2. Bearish but price rose more than 2%
                if (token_sentiment * price_change) < -2:
                    wrong_tokens.append(token)
        
        return 'wrong' if wrong_tokens else 'right'
        
    def _get_spicy_callback(self, token: str, current_prices: Dict[str, float]) -> Optional[str]:
        """Generate witty callbacks to past terrible predictions"""
        recent_predictions = [p for p in self.past_predictions 
                            if p['timestamp'] > (datetime.now() - timedelta(hours=24))
                            and p['token'] == token]
        
        if not recent_predictions:
            return None
            
        for pred in recent_predictions:
            if pred['outcome'] is None:
                pred['outcome'] = self._validate_past_prediction(pred, current_prices)
                
        wrong_predictions = [p for p in recent_predictions if p['outcome'] == 'wrong']
        if wrong_predictions:
            worst_pred = wrong_predictions[-1]
            time_ago = int((datetime.now() - worst_pred['timestamp']).total_seconds() / 3600)
            
            # If time_ago is 0, set it to 1 to avoid awkward phrasing
            if time_ago == 0:
                time_ago = 1
            
            # Token-specific callbacks
            callbacks = [
                f"(Unlike my galaxy-brain take {time_ago}h ago about {worst_pred['prediction'].split('.')[0]}... this time I'm sure!)",
                f"(Looks like my {time_ago}h old prediction about {token} aged like milk. But trust me bro!)",
                f"(That awkward moment when your {time_ago}h old {token} analysis was completely wrong... but this one's different!)",
                f"(My {token} trading bot would be down bad after that {time_ago}h old take. Good thing I'm just an analyst!)",
                f"(Excuse the {time_ago}h old miss on {token}. Even the best crypto analysts are wrong sometimes... just not usually THIS wrong!)"
            ]
            return callbacks[hash(str(datetime.now())) % len(callbacks)]
            
        return None
        
    def _format_tweet_analysis(self, token: str, analysis: str, crypto_data: Dict[str, Any]) -> str:
        """Format analysis for Twitter with no hashtags to maximize content"""
        # Simply use the analysis text with no hashtags
        tweet = analysis
        
        # Sanitize text to remove non-BMP characters that ChromeDriver can't handle
        tweet = ''.join(char for char in tweet if ord(char) < 0x10000)
        
        # Check for minimum length
        min_length = self.config.TWEET_CONSTRAINTS['MIN_LENGTH']
        if len(tweet) < min_length:
            logger.logger.warning(f"Analysis too short ({len(tweet)} chars). Minimum: {min_length}")
            # Not much we can do here since Claude should have generated the right length
            # We'll log but not try to fix, as Claude should be instructed correctly
        
        # Check for maximum length
        max_length = self.config.TWEET_CONSTRAINTS['MAX_LENGTH']
        if len(tweet) > max_length:
            logger.logger.warning(f"Analysis too long ({len(tweet)} chars). Maximum: {max_length}")
        
        # Check for hard stop length
        hard_stop = self.config.TWEET_CONSTRAINTS['HARD_STOP_LENGTH']
        if len(tweet) > hard_stop:
            # Smart truncation - find the last sentence boundary before the limit
            # First try to end on a period, question mark, or exclamation
            last_period = tweet[:hard_stop-3].rfind('. ')
            last_question = tweet[:hard_stop-3].rfind('? ')
            last_exclamation = tweet[:hard_stop-3].rfind('! ')
            
            # Find the last sentence-ending punctuation
            last_sentence_end = max(last_period, last_question, last_exclamation)
            
            if last_sentence_end > hard_stop * 0.7:  # If we can find a good sentence break in the latter 30% of the text
                # Truncate at the end of a sentence and add no ellipsis
                tweet = tweet[:last_sentence_end+1]  # Include the punctuation
            else:
                # Fallback: find the last word boundary
                last_space = tweet[:hard_stop-3].rfind(' ')
                if last_space > 0:
                    tweet = tweet[:last_space] + "..."
                else:
                    # Last resort: hard truncation
                    tweet = tweet[:hard_stop-3] + "..."
                
            logger.logger.warning(f"Trimmed analysis to {len(tweet)} chars using smart truncation")
        
        return tweet

    def _should_post_update(self, token: str, new_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Determine if we should post an update based on market changes
        Returns (should_post, trigger_reason)
        """
        if not self.last_market_data:
            self.last_market_data = new_data
            return True, "initial_post"

        trigger_reason = None

        # Check token for significant changes
        if token in new_data and token in self.last_market_data:
            # Calculate immediate price change since last check
            price_change = abs(
                (new_data[token]['current_price'] - self.last_market_data[token]['current_price']) /
                self.last_market_data[token]['current_price'] * 100
            )
            
            # Calculate immediate volume change since last check
            immediate_volume_change = abs(
                (new_data[token]['volume'] - self.last_market_data[token]['volume']) /
                self.last_market_data[token]['volume'] * 100
            )

            logger.logger.debug(
                f"{token} immediate changes - Price: {price_change:.2f}%, Volume: {immediate_volume_change:.2f}%"
            )

            # Check immediate price change
            if price_change >= self.config.PRICE_CHANGE_THRESHOLD:
                trigger_reason = f"price_change_{token.lower()}"
                logger.logger.info(f"Significant price change detected for {token}: {price_change:.2f}%")
                
            # Check immediate volume change
            elif immediate_volume_change >= self.config.VOLUME_CHANGE_THRESHOLD:
                trigger_reason = f"volume_change_{token.lower()}"
                logger.logger.info(f"Significant immediate volume change detected for {token}: {immediate_volume_change:.2f}%")
                
            # Check rolling window volume trend
            else:
                historical_volume = self._get_historical_volume_data(token)
                if historical_volume:
                    volume_change_pct, trend = self._analyze_volume_trend(
                        new_data[token]['volume'],
                        historical_volume
                    )
                    
                    # Log the volume trend
                    logger.logger.debug(
                        f"{token} rolling window volume trend: {volume_change_pct:.2f}% ({trend})"
                    )
                    
                    # Check if trend is significant enough to trigger
                    if trend in ["significant_increase", "significant_decrease"]:
                        trigger_reason = f"volume_trend_{token.lower()}_{trend}"
                        logger.logger.info(
                            f"Significant volume trend detected for {token}: "
                            f"{volume_change_pct:.2f}% over {self.config.VOLUME_WINDOW_MINUTES} minutes"
                        )
            
            # Check for smart money indicators
            if not trigger_reason:
                smart_money = self._analyze_smart_money_indicators(token, new_data[token])
                if smart_money.get('abnormal_volume') or smart_money.get('stealth_accumulation'):
                    trigger_reason = f"smart_money_{token.lower()}"
                    logger.logger.info(f"Smart money movement detected for {token}")
            
            # Check for significant outperformance vs market
            if not trigger_reason:
                vs_market = self._analyze_token_vs_market(token, new_data)
                if vs_market.get('outperforming_market') and abs(vs_market.get('vs_market_avg_change', 0)) > 5.0:
                    trigger_reason = f"{token.lower()}_outperforming_market"
                    logger.logger.info(f"{token} significantly outperforming market")
                    
            # Check if we need to post prediction update
            # Trigger prediction post based on time since last prediction
            if not trigger_reason:
                # Check when the last prediction was posted
                last_prediction = self.config.db.get_active_predictions(token=token, timeframe="1h")
                if not last_prediction:
                    # No recent 1h predictions, should post one
                    trigger_reason = f"prediction_needed_{token.lower()}"
                    logger.logger.info(f"No recent prediction for {token}, triggering prediction post")

        # Check if regular interval has passed
        if not trigger_reason:
            time_since_last = (datetime.now() - self.last_check_time).total_seconds()
            if time_since_last >= self.config.BASE_INTERVAL:
                trigger_reason = "regular_interval"
                logger.logger.debug("Regular interval check triggered")

        should_post = trigger_reason is not None
        if should_post:
            self.last_market_data = new_data
            logger.logger.info(f"Update triggered by: {trigger_reason}")
        else:
            logger.logger.debug("No triggers activated, skipping update")

        return should_post, trigger_reason

    def _analyze_market_sentiment(self, token: str, crypto_data: Dict[str, Any], trigger_type: str) -> Tuple[Optional[str], Optional[Dict]]:
        """
        Generate token-specific market analysis with focus on volume and smart money.
        Returns the formatted tweet and data needed to store it in the database.
        """
        max_retries = 3
        retry_count = 0
        
        # Define rotating focus areas for more varied analyses
        focus_areas = [
            "Focus on volume patterns, smart money movements, and how the token is performing relative to the broader market.",
            "Emphasize technical indicators showing money flow in the market. Pay special attention to volume-to-price divergence.",
            "Analyze accumulation patterns and capital rotation. Look for subtle signs of institutional interest.",
            "Examine volume preceding price action. Note any leading indicators.",
            "Highlight the relationship between price action and significant volume changes.",
            "Investigate potential smart money positioning ahead of market moves. Note any anomalous volume signatures.",
            "Focus on recent volume clusters and their impact on price stability. Look for divergence patterns.",
            "Analyze volatility profile compared to the broader market and what this suggests about sentiment."
        ]
        
        while retry_count < max_retries:
            try:
                logger.logger.debug(f"Starting {token} market sentiment analysis (attempt {retry_count + 1})")
                
                # Get token data
                token_data = crypto_data.get(token, {})
                if not token_data:
                    logger.log_error("Market Analysis", f"Missing {token} data")
                    return None, None
                
                # Calculate correlations with market
                correlations = self._calculate_correlations(token, crypto_data)
                
                # Get smart money indicators
                smart_money = self._analyze_smart_money_indicators(token, token_data)
                
                # Get token vs market performance
                vs_market = self._analyze_token_vs_market(token, crypto_data)
                
                # Get spicy callback for previous predictions
                callback = self._get_spicy_callback(token, {sym: data['current_price'] 
                                                   for sym, data in crypto_data.items()})
                
                # Analyze mood
                indicators = MoodIndicators(
                    price_change=token_data['price_change_percentage_24h'],
                    trading_volume=token_data['volume'],
                    volatility=abs(token_data['price_change_percentage_24h']) / 100,
                    social_sentiment=None,
                    funding_rates=None,
                    liquidation_volume=None
                )
                
                mood = determine_advanced_mood(indicators)
                token_mood = {
                    'mood': mood.value,
                    'change': token_data['price_change_percentage_24h'],
                    'ath_distance': token_data['ath_change_percentage']
                }
                
                # Store mood data
                self.config.db.store_mood(token, mood.value, indicators)
                
                # Generate meme phrase - use the generic method for all tokens
                meme_context = MemePhraseGenerator.generate_meme_phrase(
                    chain=token,
                    mood=Mood(mood.value)
                )
                
                # Get volume trend for additional context
                historical_volume = self._get_historical_volume_data(token)
                if historical_volume:
                    volume_change_pct, trend = self._analyze_volume_trend(
                        token_data['volume'],
                        historical_volume
                    )
                    volume_trend = {
                        'change_pct': volume_change_pct,
                        'trend': trend
                    }
                else:
                    volume_trend = {'change_pct': 0, 'trend': 'stable'}

                # Get historical context from database
                stats = self.config.db.get_chain_stats(token, hours=24)
                if stats:
                    historical_context = f"24h Avg: ${stats['avg_price']:,.2f}, "
                    historical_context += f"High: ${stats['max_price']:,.2f}, "
                    historical_context += f"Low: ${stats['min_price']:,.2f}"
                else:
                    historical_context = "No historical data"
                
                # Check if this is a volume trend trigger
                volume_context = ""
                if "volume_trend" in trigger_type:
                    change = volume_trend['change_pct']
                    direction = "increase" if change > 0 else "decrease"
                    volume_context = f"\nVolume Analysis:\n{token} showing {abs(change):.1f}% {direction} in volume over last hour. This is a significant {volume_trend['trend']}."

                # Smart money context
                smart_money_context = ""
                if smart_money.get('abnormal_volume'):
                    smart_money_context += f"\nAbnormal volume detected: {smart_money['volume_z_score']:.1f} standard deviations from mean."
                if smart_money.get('stealth_accumulation'):
                    smart_money_context += f"\nPotential stealth accumulation detected with minimal price movement and elevated volume."
                if smart_money.get('volume_cluster_detected'):
                    smart_money_context += f"\nVolume clustering detected, suggesting possible institutional activity."
                if smart_money.get('unusual_trading_hours'):
                    smart_money_context += f"\nUnusual trading hours detected: {', '.join(smart_money['unusual_trading_hours'])}."

                # Market comparison context
                market_context = ""
                if vs_market.get('outperforming_market'):
                    market_context += f"\n{token} outperforming market average by {vs_market['vs_market_avg_change']:.1f}%"
                else:
                    market_context += f"\n{token} underperforming market average by {abs(vs_market['vs_market_avg_change']):.1f}%"
                
                # Market volume flow technical analysis
                reference_tokens = [t for t in self.reference_tokens if t != token and t in crypto_data]
                market_total_volume = sum([data['volume'] for sym, data in crypto_data.items() if sym in reference_tokens])
                market_volume_ratio = (token_data['volume'] / market_total_volume * 100) if market_total_volume > 0 else 0
                
                capital_rotation = "Yes" if vs_market.get('outperforming_market', False) and smart_money.get('volume_vs_daily_avg', 0) > 0.2 else "No"
                
                selling_pattern = "Detected" if vs_market.get('vs_market_volume_growth', 0) < 0 and volume_trend['change_pct'] > 5 else "Not detected"
                
                # Find top 2 correlated tokens
                price_correlations = {k.replace('price_correlation_', ''): v 
                                     for k, v in correlations.items() 
                                     if k.startswith('price_correlation_')}
                top_correlated = sorted(price_correlations.items(), key=lambda x: x[1], reverse=True)[:2]
                
                technical_context = f"""
Market Flow Analysis:
- {token}/Market volume ratio: {market_volume_ratio:.2f}%
- Potential capital rotation: {capital_rotation}
- Market selling {token} buying patterns: {selling_pattern}
"""
                if top_correlated:
                    technical_context += "- Highest correlations: "
                    for corr_token, corr_value in top_correlated:
                        technical_context += f"{corr_token}: {corr_value:.2f}, "
                    technical_context = technical_context.rstrip(", ")

                # Select a focus area using a deterministic but varied approach
                # Use a combination of date, hour, token and trigger type to ensure variety
                focus_seed = f"{datetime.now().date()}_{datetime.now().hour}_{token}_{trigger_type}"
                focus_index = hash(focus_seed) % len(focus_areas)
                selected_focus = focus_areas[focus_index]

                prompt = f"""Write a witty market analysis focusing on {token} token with attention to volume changes and smart money movements. Format as a single paragraph.

IMPORTANT: 
1. The analysis MUST be between 260-275 characters long. Target exactly 270 characters. This is a STRICT requirement.
2. Always use #{token} instead of {token} when referring to the token in your analysis. This is critical!
3. Do NOT use any emojis or special Unicode characters. Stick to basic ASCII and standard punctuation only!
4. End with a complete sentence and a proper punctuation mark (., !, or ?). Make sure your final sentence is complete.
5. Count your characters carefully before submitting!

Market data:
                
{token} Performance:
- Price: ${token_data['current_price']:,.4f}
- 24h Change: {token_mood['change']:.1f}% ({token_mood['mood']})
- Volume: ${token_data['volume']:,.0f}
                
Historical Context:
- {token}: {historical_context}
                
Volume Analysis:
- 24h trend: {volume_trend['change_pct']:.1f}% over last hour ({volume_trend['trend']})
- vs hourly avg: {smart_money.get('volume_vs_hourly_avg', 0)*100:.1f}%
- vs daily avg: {smart_money.get('volume_vs_daily_avg', 0)*100:.1f}%
{volume_context}
                
Smart Money Indicators:
- Volume Z-score: {smart_money.get('volume_z_score', 0):.2f}
- Price-Volume Divergence: {smart_money.get('price_volume_divergence', False)}
- Stealth Accumulation: {smart_money.get('stealth_accumulation', False)}
- Abnormal Volume: {smart_money.get('abnormal_volume', False)}
- Volume Clustering: {smart_money.get('volume_cluster_detected', False)}
{smart_money_context}
                
Market Comparison:
- vs Market avg change: {vs_market.get('vs_market_avg_change', 0):.1f}%
- vs Market volume growth: {vs_market.get('vs_market_volume_growth', 0):.1f}%
- Outperforming Market: {vs_market.get('outperforming_market', False)}
{market_context}
                
ATH Distance:
- {token}: {token_mood['ath_distance']:.1f}%
                
{technical_context}
                
Token-specific context:
- Meme: {meme_context}
                
Trigger Type: {trigger_type}
                
Past Context: {callback if callback else 'None'}
                
Note: {selected_focus} Keep the analysis fresh and varied. Avoid repetitive phrases."""
                
                logger.logger.debug("Sending analysis request to Claude")
                response = self.claude_client.messages.create(
                    model="claude-3-7-sonnet-20250219",
                    max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                analysis = response.content[0].text
                logger.logger.debug("Received analysis from Claude")
                
                # Store prediction data
                prediction_data = {
                    'analysis': analysis,
                    'sentiment': {token: token_mood['mood']},
                    **{f"{sym.upper()}_price": data['current_price'] for sym, data in crypto_data.items()}
                }
                self._track_prediction(token, prediction_data, [token])
                
                formatted_tweet = self._format_tweet_analysis(token, analysis, crypto_data)
                
                # Create the storage data to be stored later (after duplicate check)
                storage_data = {
                    'content': formatted_tweet,
                    'sentiment': {token: token_mood},
                    'trigger_type': trigger_type,
                    'price_data': {token: {'price': token_data['current_price'], 
                                         'volume': token_data['volume']}},
                    'meme_phrases': {token: meme_context}
                }
                
                return formatted_tweet, storage_data
                
            except Exception as e:
                retry_count += 1
                wait_time = retry_count * 10
                logger.logger.error(f"Analysis error details: {str(e)}", exc_info=True)
                logger.logger.warning(f"Analysis error, attempt {retry_count}, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
        
        logger.log_error("Market Analysis", "Maximum retries reached")
        return None, None

    def _evaluate_expired_predictions(self) -> None:
        """
        Find and evaluate expired predictions
        """
        try:
            # Get expired unevaluated predictions
            expired_predictions = self.config.db.get_expired_unevaluated_predictions()
            
            if not expired_predictions:
                logger.logger.debug("No expired predictions to evaluate")
                return
                
            logger.logger.info(f"Evaluating {len(expired_predictions)} expired predictions")
            
            # Get current market data
            market_data = self._get_crypto_data()
            if not market_data:
                logger.logger.error("Failed to fetch market data for prediction evaluation")
                return
                
            # Evaluate each prediction
            for prediction in expired_predictions:
                token = prediction["token"]
                prediction_id = prediction["id"]
                
                # Get current price for the token
                token_data = market_data.get(token, {})
                if not token_data:
                    logger.logger.warning(f"No current price data for {token}, skipping evaluation")
                    continue
                    
                current_price = token_data.get("current_price", 0)
                if current_price == 0:
                    logger.logger.warning(f"Zero price for {token}, skipping evaluation")
                    continue
                    
                # Record the outcome
                result = self.config.db.record_prediction_outcome(prediction_id, current_price)
                
                if result:
                    logger.logger.debug(f"Evaluated prediction {prediction_id} for {token}")
                else:
                    logger.logger.error(f"Failed to evaluate prediction {prediction_id} for {token}")
                    
            # Get updated performance stats
            performance_stats = self.config.db.get_prediction_performance()
            
            # Log overall performance
            if performance_stats:
                total_correct = sum(p["correct_predictions"] for p in performance_stats)
                total_predictions = sum(p["total_predictions"] for p in performance_stats)
                
                if total_predictions > 0:
                    overall_accuracy = (total_correct / total_predictions) * 100
                    logger.logger.info(f"Overall prediction accuracy: {overall_accuracy:.1f}% ({total_correct}/{total_predictions})")
                    
        except Exception as e:
            logger.log_error("Evaluate Expired Predictions", str(e))

    def _generate_weekly_summary(self) -> bool:
        """Generate and post a weekly summary of predictions and performance"""
        try:
            # Check if it's Sunday (weekday 6) and around midnight
            now = datetime.now()
            if now.weekday() != 6 or now.hour != 0:
                return False
                
            # Get performance stats
            performance_stats = self.config.db.get_prediction_performance()
            
            if not performance_stats:
                return False
                
            # Calculate overall stats
            total_correct = sum(p["correct_predictions"] for p in performance_stats)
            total_predictions = sum(p["total_predictions"] for p in performance_stats)
            
            if total_predictions == 0:
                return False
                
            overall_accuracy = (total_correct / total_predictions) * 100
            
            # Get token-specific stats
            token_stats = {}
            for stat in performance_stats:
                token = stat["token"]
                if stat["total_predictions"] > 0:
                    token_stats[token] = {
                        "accuracy": stat["accuracy_rate"],
                        "total": stat["total_predictions"]
                    }
            
            # Sort tokens by accuracy
            sorted_tokens = sorted(token_stats.items(), key=lambda x: x[1]["accuracy"], reverse=True)
            
            # Generate report
            report = "📊 WEEKLY PREDICTION SUMMARY 📊\n\n"
            report += f"Overall Accuracy: {overall_accuracy:.1f}% ({total_correct}/{total_predictions})\n\n"
            report += "Top Performers:\n"
            
            for token, stats in sorted_tokens[:3]:
                report += f"#{token}: {stats['accuracy']:.1f}% ({stats['total']} predictions)\n"
                
            report += "\nBottom Performers:\n"
            for token, stats in sorted_tokens[-3:]:
                report += f"#{token}: {stats['accuracy']:.1f}% ({stats['total']} predictions)\n"
            
            # Post the weekly summary
            return self._post_analysis(report)
            
        except Exception as e:
            logger.log_error("Weekly Summary", str(e))
            return False        

    def _generate_predictions(self, token: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate market predictions for a specific token
        """
        try:
            logger.logger.info(f"Generating predictions for {token}")
            
            # Generate prediction for 1-hour timeframe
            prediction = self.prediction_engine.generate_prediction(
                token=token,
                market_data=market_data,
                timeframe="1h"
            )
            
            # Store prediction in database
            prediction_id = self.config.db.store_prediction(token, prediction, timeframe="1h")
            logger.logger.info(f"Stored {token} prediction with ID {prediction_id}")
            
            return prediction
            
        except Exception as e:
            logger.log_error(f"Generate Predictions - {token}", str(e))
            return {}

    def _format_prediction_tweet(self, token: str, prediction: Dict[str, Any], crypto_data: Dict[str, Any]) -> str:
        """
        Format a prediction into a tweet with FOMO-inducing content
        """
        try:
            # Get prediction details
            pred_data = prediction.get("prediction", {})
            sentiment = prediction.get("sentiment", "NEUTRAL")
            rationale = prediction.get("rationale", "")
            
            # Format prediction values
            price = pred_data.get("price", 0)
            confidence = pred_data.get("confidence", 70)
            lower_bound = pred_data.get("lower_bound", 0)
            upper_bound = pred_data.get("upper_bound", 0)
            percent_change = pred_data.get("percent_change", 0)
            
            # Get current price
            token_data = crypto_data.get(token, {})
            current_price = token_data.get("current_price", 0)
            
            # Format the tweet
            tweet = f"#{token} 1HR PREDICTION:\n\n"
            
            # Sentiment-based emoji
            if sentiment == "BULLISH":
                tweet += "BULLISH ALERT\n"
            elif sentiment == "BEARISH":
                tweet += "BEARISH WARNING\n"
            else:
                tweet += "MARKET ANALYSIS\n"
                
            # Add prediction with confidence
            tweet += f"Target: ${price:.4f} ({percent_change:+.2f}%)\n"
            tweet += f"Range: ${lower_bound:.4f} - ${upper_bound:.4f}\n"
            tweet += f"Confidence: {confidence}%\n\n"
            
            # Add rationale
            tweet += f"{rationale}\n\n"
            
            # Add accuracy tracking if available
            performance = self.config.db.get_prediction_performance(token=token, timeframe="1h")
            if performance and performance[0]["total_predictions"] > 0:
                accuracy = performance[0]["accuracy_rate"]
                tweet += f"Accuracy: {accuracy:.1f}% on {performance[0]['total_predictions']} predictions"
                
            # Ensure tweet is within the hard stop length
            max_length = self.config.TWEET_CONSTRAINTS['HARD_STOP_LENGTH']
            if len(tweet) > max_length:
                # Smart truncate to preserve essential info
                tweet = tweet[:max_length-3] + "..."
                
            return tweet
            
        except Exception as e:
            logger.log_error(f"Format Prediction Tweet - {token}", str(e))
            return f"#{token} 1HR PREDICTION: ${price:.4f} ({percent_change:+.2f}%) - {sentiment}"
        
    def _generate_correlation_report(self, market_data: Dict[str, Any]) -> str:
        """Generate a report of correlations between top tokens"""
        try:
            tokens = ['BTC', 'ETH', 'SOL', 'BNB', 'AVAX']  # Focus on major tokens
        
            # Create correlation matrix
            correlation_matrix = {}
            for token1 in tokens:
                correlation_matrix[token1] = {}
            for token2 in tokens:
                if token1 == token2:
                    correlation_matrix[token1][token2] = 1.0
                    continue
                
                if token1 not in market_data or token2 not in market_data:
                    correlation_matrix[token1][token2] = 0.0
                    continue
                
                # Calculate price correlation (simplified)
                price_change1 = market_data[token1]['price_change_percentage_24h']
                price_change2 = market_data[token2]['price_change_percentage_24h']
                
                price_direction1 = 1 if price_change1 > 0 else -1
                price_direction2 = 1 if price_change2 > 0 else -1
                
                # Basic correlation (-1.0 to 1.0)
                correlation = 1.0 if price_direction1 == price_direction2 else -1.0
                correlation_matrix[token1][token2] = correlation
        
                # Format as tweet
                report = "24H CORRELATION MATRIX:\n\n"
        
                # Create ASCII art heatmap
            for token1 in tokens:
                report += f"{token1} "
            for token2 in tokens:
                corr = correlation_matrix[token1][token2]
                if token1 == token2:
                    report += "✦ "  # Self correlation
                elif corr > 0.5:
                    report += "➕ "  # Strong positive
                elif corr > 0:
                    report += "📈 "  # Positive
                elif corr < -0.5:
                    report += "➖ "  # Strong negative
                else:
                    report += "📉 "  # Negative
            report += "\n"
            
            report += "\nKey: ✦=Same ➕=Strong+ 📈=Weak+ ➖=Strong- 📉=Weak-"
            return report
        except Exception as e:
            logger.log_error("Correlation Report", str(e))
            return "Failed to generate correlation report."    

    def _run_analysis_cycle(self) -> None:
        """Run analysis and posting cycle for all tokens with prediction integration"""
        try:
            # First, evaluate any expired predictions
            self._evaluate_expired_predictions()
            
            # Get market data
            market_data = self._get_crypto_data()
            if not market_data:
                logger.logger.error("Failed to fetch market data")
                return
                
            # Get available tokens and shuffle them to try in random order
            available_tokens = [token for token in self.reference_tokens if token in market_data]
            if not available_tokens:
                logger.logger.error("No token data available")
                return
                
            # Shuffle the tokens to prevent always analyzing the same ones first
            import random
            available_tokens = self._prioritize_tokens(available_tokens, market_data)
            
            # Try each token until we find one that's not a duplicate
            for token_to_analyze in available_tokens:
                should_post, trigger_type = self._should_post_update(token_to_analyze, market_data)
                
                if should_post:
                    logger.logger.info(f"Starting {token_to_analyze} analysis cycle - Trigger: {trigger_type}")
                    
                    # Generate prediction for this token
                    prediction = self._generate_predictions(token_to_analyze, market_data)
                    
                    if not prediction:
                        logger.logger.error(f"Failed to generate prediction for {token_to_analyze}")
                        continue

                    # Get both standard analysis and prediction-focused content 
                    standard_analysis, storage_data = self._analyze_market_sentiment(token_to_analyze, market_data, trigger_type)
                    prediction_tweet = self._format_prediction_tweet(token_to_analyze, prediction, market_data)
                    
                    # Choose which type of content to post based on trigger and past posts
                    # For prediction-specific triggers or every third post, post prediction
                    should_post_prediction = (
                        "prediction" in trigger_type or 
                        random.random() < 0.35  # 35% chance of posting prediction instead of analysis
                    )
                    
                    if should_post_prediction:
                        analysis_to_post = prediction_tweet
                        # Add prediction data to storage
                        if storage_data:
                            storage_data['is_prediction'] = True
                            storage_data['prediction_data'] = prediction
                    else:
                        analysis_to_post = standard_analysis
                        if storage_data:
                            storage_data['is_prediction'] = False
                    
                    if not analysis_to_post:
                        logger.logger.error(f"Failed to generate content for {token_to_analyze}")
                        continue
                        
                    # Check for duplicates
                    last_posts = self._get_last_posts()
                    if not self._is_duplicate_analysis(analysis_to_post, last_posts):
                        if self._post_analysis(analysis_to_post):
                            # Only store in database after successful posting
                            if storage_data:
                                self.config.db.store_posted_content(**storage_data)
                                
                            logger.logger.info(f"Successfully posted {token_to_analyze} {'prediction' if should_post_prediction else 'analysis'} - Trigger: {trigger_type}")
                            
                            # Store additional smart money metrics
                            if token_to_analyze in market_data:
                                smart_money = self._analyze_smart_money_indicators(token_to_analyze, market_data[token_to_analyze])
                                self.config.db.store_smart_money_indicators(token_to_analyze, smart_money)
                                
                                # Store market comparison data
                                vs_market = self._analyze_token_vs_market(token_to_analyze, market_data)
                                if vs_market:
                                    self.config.db.store_token_market_comparison(
                                        token_to_analyze,
                                        vs_market.get('vs_market_avg_change', 0),
                                        vs_market.get('vs_market_volume_growth', 0),
                                        vs_market.get('outperforming_market', False),
                                        vs_market.get('correlations', {})
                                    )
                            
                            # Successfully posted, so we're done with this cycle
                            return
                        else:
                            logger.logger.error(f"Failed to post {token_to_analyze} {'prediction' if should_post_prediction else 'analysis'}")
                            continue  # Try next token
                    else:
                        logger.logger.info(f"Skipping duplicate {token_to_analyze} content - trying another token")
                        continue  # Try next token
                else:
                    logger.logger.debug(f"No significant {token_to_analyze} changes detected, trying another token")
            
                # If we couldn't find any token-specific update to post, 
                # try posting a correlation report on regular intervals
                if "regular_interval" in trigger_type:
                    correlation_report = self._generate_correlation_report(market_data)
                    if correlation_report and self._post_analysis(correlation_report):
                        logger.logger.info("Posted correlation matrix report")
                        return      

            # If we get here, we tried all tokens but couldn't post anything
            logger.logger.warning("Tried all available tokens but couldn't post any analysis")
                
        except Exception as e:
            logger.log_error("Token Analysis Cycle", str(e))

    def start(self) -> None:
        """Main bot execution loop"""
        try:
            retry_count = 0
            max_setup_retries = 3
            
            while retry_count < max_setup_retries:
                if not self.browser.initialize_driver():
                    retry_count += 1
                    logger.logger.warning(f"Browser initialization attempt {retry_count} failed, retrying...")
                    time.sleep(10)
                    continue
                    
                if not self._login_to_twitter():
                    retry_count += 1
                    logger.logger.warning(f"Twitter login attempt {retry_count} failed, retrying...")
                    time.sleep(15)
                    continue
                    
                break
            
            if retry_count >= max_setup_retries:
                raise Exception("Failed to initialize bot after maximum retries")

            logger.logger.info("Bot initialized successfully")

            while True:
                try:
                    self._run_analysis_cycle()
                    
                    # Calculate sleep time until next regular check
                    time_since_last = (datetime.now() - self.last_check_time).total_seconds()
                    sleep_time = max(0, self.config.BASE_INTERVAL - time_since_last)
                    
                    # Check if we should post a weekly summary
                    if self._generate_weekly_summary():
                        logger.logger.info("Posted weekly performance summary")   

                    logger.logger.debug(f"Sleeping for {sleep_time:.1f}s until next check")
                    time.sleep(sleep_time)
                    
                    self.last_check_time = datetime.now()
                    
                except Exception as e:
                    logger.log_error("Analysis Cycle", str(e), exc_info=True)
                    time.sleep(60)  # Shorter sleep on error
                    continue

        except KeyboardInterrupt:
            logger.logger.info("Bot stopped by user")
        except Exception as e:
            logger.log_error("Bot Execution", str(e))
        finally:
            self._cleanup()

if __name__ == "__main__":
    try:
        bot = CryptoAnalysisBot()
        bot.start()
    except Exception as e:
        logger.log_error("Bot Startup", str(e))                                        
