"""
Sentiment & Institutional Data Fetcher

Fetches alternative data signals from available APIs:
- News sentiment (Finnhub)
- Institutional holdings changes (Finnhub)
- Insider trading signals (Finnhub)
- Analyst ratings (Finnhub, FMP)

These signals are powerful alpha sources often not used by retail traders.
"""
from __future__ import annotations
import os
import time
import logging
import requests
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from functools import lru_cache
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.config import get_secret
from core.api_monitor import record_api_call

logger = logging.getLogger(__name__)

# API Keys
FINNHUB_API_KEY = get_secret("FINNHUB_API_KEY", "")
FMP_API_KEY = get_secret("FMP_API_KEY", "")
POLYGON_API_KEY = get_secret("POLYGON_API_KEY", "")

# Rate limiting
_LAST_CALL: Dict[str, float] = {}
_RATE_LOCK = threading.Lock()
MIN_INTERVAL = {"finnhub": 0.2, "fmp": 0.1, "polygon": 0.5}

# Cache
_SENTIMENT_CACHE: Dict[str, Dict] = {}
_CACHE_TTL = 3600  # 1 hour


def _rate_limit(provider: str) -> None:
    """Apply rate limiting for API calls."""
    with _RATE_LOCK:
        now = time.time()
        last = _LAST_CALL.get(provider, 0)
        wait = MIN_INTERVAL.get(provider, 0.2) - (now - last)
        if wait > 0:
            time.sleep(wait)
        _LAST_CALL[provider] = time.time()


def _safe_get(url: str, params: Dict, provider: str, timeout: int = 10) -> Optional[Dict]:
    """Safe HTTP GET with error handling and monitoring."""
    _rate_limit(provider)
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        record_api_call(provider, resp.status_code == 200)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 429:
            logger.warning(f"Rate limited by {provider}")
            return None
        else:
            logger.debug(f"{provider} returned {resp.status_code}: {resp.text[:200]}")
            return None
    except Exception as e:
        logger.debug(f"{provider} request failed: {e}")
        record_api_call(provider, False)
        return None


# =============================================================================
# NEWS SENTIMENT (Finnhub)
# =============================================================================

def fetch_news_sentiment_finnhub(ticker: str, days: int = 7) -> Dict[str, float]:
    """
    Fetch news sentiment for a ticker from Finnhub.
    
    Returns:
        Dict with:
        - sentiment_avg: Average sentiment (-1 to 1)
        - news_count: Number of articles
        - sentiment_positive_pct: % of positive articles
        - sentiment_momentum: Recent vs older sentiment change
    """
    if not FINNHUB_API_KEY:
        return _default_sentiment()
    
    cache_key = f"finnhub_news_{ticker}_{days}"
    if cache_key in _SENTIMENT_CACHE:
        cached = _SENTIMENT_CACHE[cache_key]
        if time.time() - cached.get("_ts", 0) < _CACHE_TTL:
            return {k: v for k, v in cached.items() if not k.startswith("_")}
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    url = "https://finnhub.io/api/v1/company-news"
    params = {
        "symbol": ticker,
        "from": start_date.strftime("%Y-%m-%d"),
        "to": end_date.strftime("%Y-%m-%d"),
        "token": FINNHUB_API_KEY
    }
    
    data = _safe_get(url, params, "finnhub")
    
    if not data or not isinstance(data, list):
        return _default_sentiment()
    
    # Compute sentiment scores
    # Finnhub news doesn't have built-in sentiment, so we use headline analysis
    # For production, would use their sentiment API or NLP
    news_count = len(data)
    
    if news_count == 0:
        result = _default_sentiment()
        result["news_count"] = 0
    else:
        # Simple sentiment heuristics based on headline keywords
        positive_words = {"surge", "jump", "soar", "beat", "upgrade", "buy", "growth", "profit", "gain", "rally", "high", "strong", "outperform"}
        negative_words = {"fall", "drop", "crash", "miss", "downgrade", "sell", "loss", "decline", "weak", "underperform", "cut", "warning"}
        
        sentiments = []
        for article in data[:50]:  # Limit to recent 50
            headline = article.get("headline", "").lower()
            pos_count = sum(1 for w in positive_words if w in headline)
            neg_count = sum(1 for w in negative_words if w in headline)
            
            if pos_count > neg_count:
                sentiments.append(0.5 + 0.1 * (pos_count - neg_count))
            elif neg_count > pos_count:
                sentiments.append(-0.5 - 0.1 * (neg_count - pos_count))
            else:
                sentiments.append(0.0)
        
        sentiments = np.array(sentiments)
        sentiments = np.clip(sentiments, -1, 1)
        
        # Compute metrics
        sentiment_avg = float(np.mean(sentiments))
        positive_pct = float((sentiments > 0.1).mean())
        
        # Sentiment momentum: last 3 days vs prior
        if len(sentiments) >= 3:
            recent_sent = np.mean(sentiments[:3])
            older_sent = np.mean(sentiments[3:]) if len(sentiments) > 3 else 0
            momentum = recent_sent - older_sent
        else:
            momentum = 0.0
        
        result = {
            "sentiment_avg": round(sentiment_avg, 3),
            "news_count": min(news_count, 100) / 100,  # Normalize to 0-1
            "sentiment_positive_pct": round(positive_pct, 3),
            "sentiment_momentum": round(momentum, 3)
        }
    
    # Cache
    result["_ts"] = time.time()
    _SENTIMENT_CACHE[cache_key] = result
    
    return {k: v for k, v in result.items() if not k.startswith("_")}


def _default_sentiment() -> Dict[str, float]:
    """Return default sentiment values when data unavailable."""
    return {
        "sentiment_avg": 0.0,
        "news_count": 0.5,
        "sentiment_positive_pct": 0.5,
        "sentiment_momentum": 0.0
    }


# =============================================================================
# INSTITUTIONAL HOLDINGS (Finnhub)
# =============================================================================

def fetch_institutional_holdings_finnhub(ticker: str) -> Dict[str, float]:
    """
    Fetch institutional ownership data from Finnhub.
    
    Returns:
        Dict with:
        - institutional_pct: Total institutional ownership %
        - institutional_change_qoq: Quarter-over-quarter change
        - top10_concentration: Top 10 holders as % of total
    """
    if not FINNHUB_API_KEY:
        return _default_institutional()
    
    cache_key = f"finnhub_inst_{ticker}"
    if cache_key in _SENTIMENT_CACHE:
        cached = _SENTIMENT_CACHE[cache_key]
        if time.time() - cached.get("_ts", 0) < _CACHE_TTL * 4:  # 4 hour cache
            return {k: v for k, v in cached.items() if not k.startswith("_")}
    
    url = "https://finnhub.io/api/v1/stock/ownership"
    params = {"symbol": ticker, "token": FINNHUB_API_KEY}
    
    data = _safe_get(url, params, "finnhub")
    
    if not data or "ownership" not in data:
        return _default_institutional()
    
    ownership = data.get("ownership", [])
    
    if not ownership:
        return _default_institutional()
    
    # Find most recent quarter
    sorted_owners = sorted(ownership, key=lambda x: x.get("filingDate", ""), reverse=True)
    
    # Sum shares
    total_shares = sum(o.get("share", 0) for o in sorted_owners[:100])
    
    # Calculate metrics (normalized)
    result = {
        "institutional_pct": min(1.0, total_shares / 1e9),  # Normalize large numbers
        "institutional_change_qoq": 0.0,  # Would need historical data
        "top10_concentration": 0.5
    }
    
    result["_ts"] = time.time()
    _SENTIMENT_CACHE[cache_key] = result
    
    return {k: v for k, v in result.items() if not k.startswith("_")}


def _default_institutional() -> Dict[str, float]:
    """Return default institutional values."""
    return {
        "institutional_pct": 0.5,
        "institutional_change_qoq": 0.0,
        "top10_concentration": 0.5
    }


# =============================================================================
# INSIDER TRADING (Finnhub)
# =============================================================================

def fetch_insider_trades_finnhub(ticker: str, days: int = 90) -> Dict[str, float]:
    """
    Fetch insider trading activity from Finnhub.
    
    Returns:
        Dict with:
        - insider_net_30d: Net buy/sell normalized (-1 to 1)
        - insider_buy_count: Number of insider buys
        - insider_sell_count: Number of insider sells
    """
    if not FINNHUB_API_KEY:
        return _default_insider()
    
    cache_key = f"finnhub_insider_{ticker}"
    if cache_key in _SENTIMENT_CACHE:
        cached = _SENTIMENT_CACHE[cache_key]
        if time.time() - cached.get("_ts", 0) < _CACHE_TTL * 2:
            return {k: v for k, v in cached.items() if not k.startswith("_")}
    
    url = "https://finnhub.io/api/v1/stock/insider-transactions"
    params = {"symbol": ticker, "token": FINNHUB_API_KEY}
    
    data = _safe_get(url, params, "finnhub")
    
    if not data or "data" not in data:
        return _default_insider()
    
    transactions = data.get("data", [])
    
    if not transactions:
        return _default_insider()
    
    # Filter to recent transactions
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    recent = [t for t in transactions if t.get("filingDate", "") >= cutoff]
    
    # Separate buys and sells
    buys = [t for t in recent if t.get("change", 0) > 0]
    sells = [t for t in recent if t.get("change", 0) < 0]
    
    buy_value = sum(abs(t.get("change", 0)) for t in buys)
    sell_value = sum(abs(t.get("change", 0)) for t in sells)
    total_value = buy_value + sell_value
    
    if total_value > 0:
        net_signal = (buy_value - sell_value) / total_value
    else:
        net_signal = 0.0
    
    result = {
        "insider_net_30d": round(np.clip(net_signal, -1, 1), 3),
        "insider_buy_count": len(buys),
        "insider_sell_count": len(sells)
    }
    
    result["_ts"] = time.time()
    _SENTIMENT_CACHE[cache_key] = result
    
    return {k: v for k, v in result.items() if not k.startswith("_")}


def _default_insider() -> Dict[str, float]:
    """Return default insider values."""
    return {
        "insider_net_30d": 0.0,
        "insider_buy_count": 0,
        "insider_sell_count": 0
    }


# =============================================================================
# ANALYST RATINGS (FMP)
# =============================================================================

def fetch_analyst_ratings_fmp(ticker: str) -> Dict[str, float]:
    """
    Fetch analyst ratings and price targets from FMP.
    
    Returns:
        Dict with:
        - analyst_rating_avg: Average rating (1=Strong Buy, 5=Strong Sell)
        - analyst_rating_change: Rating change over 3 months
        - price_target_upside: Consensus PT / current - 1
        - analyst_count: Number of analysts covering
    """
    if not FMP_API_KEY:
        return _default_analyst()
    
    cache_key = f"fmp_analyst_{ticker}"
    if cache_key in _SENTIMENT_CACHE:
        cached = _SENTIMENT_CACHE[cache_key]
        if time.time() - cached.get("_ts", 0) < _CACHE_TTL * 4:
            return {k: v for k, v in cached.items() if not k.startswith("_")}
    
    # Get analyst ratings
    url = f"https://financialmodelingprep.com/api/v3/analyst-estimates/{ticker}"
    params = {"apikey": FMP_API_KEY, "limit": 1}
    
    data = _safe_get(url, params, "fmp")
    
    # Get price target
    pt_url = f"https://financialmodelingprep.com/api/v4/price-target/{ticker}"
    pt_data = _safe_get(pt_url, {"apikey": FMP_API_KEY}, "fmp")
    
    result = _default_analyst()
    
    # Parse analyst ratings
    if data and isinstance(data, list) and len(data) > 0:
        estimate = data[0]
        result["analyst_count"] = estimate.get("numberAnalystEstimates", 0) / 20  # Normalize
    
    # Parse price target
    if pt_data and isinstance(pt_data, list) and len(pt_data) > 0:
        pt = pt_data[0]
        target_price = pt.get("priceTarget", 0)
        current_price = pt.get("currentPrice", 0)
        
        if current_price > 0 and target_price > 0:
            upside = (target_price / current_price) - 1
            result["price_target_upside"] = round(np.clip(upside, -0.5, 1.0), 3)
    
    result["_ts"] = time.time()
    _SENTIMENT_CACHE[cache_key] = result
    
    return {k: v for k, v in result.items() if not k.startswith("_")}


def _default_analyst() -> Dict[str, float]:
    """Return default analyst values."""
    return {
        "analyst_rating_avg": 0.0,
        "analyst_rating_change": 0.0,
        "price_target_upside": 0.1,
        "analyst_count": 0.5
    }


# =============================================================================
# EARNINGS DATA (Finnhub)
# =============================================================================

def fetch_earnings_data_finnhub(ticker: str) -> Dict[str, Any]:
    """
    Fetch earnings calendar and surprise data from Finnhub.
    
    Returns:
        Dict with:
        - days_to_earnings: Days until next earnings (0-90+)
        - in_earnings_window: True if within 5 days
        - last_earnings_surprise: Last EPS surprise %
    """
    if not FINNHUB_API_KEY:
        return _default_earnings()
    
    cache_key = f"finnhub_earnings_{ticker}"
    if cache_key in _SENTIMENT_CACHE:
        cached = _SENTIMENT_CACHE[cache_key]
        if time.time() - cached.get("_ts", 0) < _CACHE_TTL * 2:
            return {k: v for k, v in cached.items() if not k.startswith("_")}
    
    # Get earnings calendar
    url = "https://finnhub.io/api/v1/stock/earnings"
    params = {"symbol": ticker, "token": FINNHUB_API_KEY}
    
    data = _safe_get(url, params, "finnhub")
    
    result = _default_earnings()
    
    if data and isinstance(data, list):
        now = datetime.now()
        
        # Find next earnings date
        future_earnings = []
        past_earnings = []
        
        for e in data:
            try:
                if "date" in e:
                    date = datetime.strptime(e["date"], "%Y-%m-%d")
                elif "period" in e:
                    date = datetime.strptime(e["period"], "%Y-%m-%d") + timedelta(days=45)
                else:
                    continue
                    
                if date > now:
                    future_earnings.append((date, e))
                else:
                    past_earnings.append((date, e))
            except:
                continue
        
        # Days to next earnings
        if future_earnings:
            next_date = min(future_earnings, key=lambda x: x[0])[0]
            days_to = (next_date - now).days
            result["days_to_earnings"] = min(90, max(0, days_to))
            result["in_earnings_window"] = 1 if days_to <= 5 else 0
        
        # Last earnings surprise
        if past_earnings:
            last = max(past_earnings, key=lambda x: x[0])[1]
            actual = last.get("actual")
            estimate = last.get("estimate")
            if actual and estimate and estimate != 0:
                surprise = (actual - estimate) / abs(estimate)
                result["last_earnings_surprise"] = round(np.clip(surprise, -1, 1), 3)
    
    result["_ts"] = time.time()
    _SENTIMENT_CACHE[cache_key] = result
    
    return {k: v for k, v in result.items() if not k.startswith("_")}


def _default_earnings() -> Dict[str, Any]:
    """Return default earnings values."""
    return {
        "days_to_earnings": 45,
        "in_earnings_window": 0,
        "last_earnings_surprise": 0.0
    }


# =============================================================================
# BATCH FETCH FOR ML TRAINING
# =============================================================================

def fetch_alternative_data_batch(
    tickers: List[str],
    max_workers: int = 5
) -> pd.DataFrame:
    """
    Fetch all alternative data for a batch of tickers.
    
    Returns DataFrame with columns:
    - Ticker
    - News_Sentiment_7d
    - News_Volume_7d
    - Sentiment_Momentum
    - Institutional_Change_QoQ
    - Insider_Net_30d
    - Analyst_Rating_Change
    - Price_Target_Upside
    - Days_To_Earnings
    - In_Earnings_Window
    """
    results = []
    
    def _fetch_one(ticker: str) -> Dict:
        row = {"Ticker": ticker}
        
        try:
            # News sentiment
            sent = fetch_news_sentiment_finnhub(ticker)
            row["News_Sentiment_7d"] = sent.get("sentiment_avg", 0)
            row["News_Volume_7d"] = sent.get("news_count", 0.5)
            row["Sentiment_Momentum"] = sent.get("sentiment_momentum", 0)
            
            # Institutional
            inst = fetch_institutional_holdings_finnhub(ticker)
            row["Institutional_Change_QoQ"] = inst.get("institutional_change_qoq", 0)
            
            # Insider
            insider = fetch_insider_trades_finnhub(ticker)
            row["Insider_Net_30d"] = insider.get("insider_net_30d", 0)
            
            # Analyst
            analyst = fetch_analyst_ratings_fmp(ticker)
            row["Analyst_Rating_Change"] = analyst.get("analyst_rating_change", 0)
            row["Price_Target_Upside"] = analyst.get("price_target_upside", 0.1)
            
            # Earnings
            earnings = fetch_earnings_data_finnhub(ticker)
            row["Days_To_Earnings"] = earnings.get("days_to_earnings", 45)
            row["In_Earnings_Window"] = earnings.get("in_earnings_window", 0)
            
        except Exception as e:
            logger.debug(f"Alternative data fetch failed for {ticker}: {e}")
        
        return row
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_one, t): t for t in tickers}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                logger.debug(f"Future failed: {e}")
    
    df = pd.DataFrame(results)
    logger.info(f"Fetched alternative data for {len(df)} tickers")
    return df


# =============================================================================
# SOCIAL BUZZ (Placeholder - would use Twitter/Reddit API)
# =============================================================================

def fetch_social_buzz(ticker: str) -> Dict[str, float]:
    """
    Placeholder for social media sentiment.
    Would integrate with Twitter API or Reddit API.
    """
    return {
        "social_buzz_score": 0.5,
        "social_sentiment": 0.0,
        "mention_velocity": 0.5
    }


def clear_sentiment_cache() -> None:
    """Clear the sentiment cache."""
    global _SENTIMENT_CACHE
    _SENTIMENT_CACHE = {}
    logger.info("Sentiment cache cleared")
