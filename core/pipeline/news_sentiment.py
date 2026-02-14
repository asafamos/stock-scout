"""News fetching and sentiment analysis helpers for the pipeline."""

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List

import requests

from core.config import get_secret

logger = logging.getLogger(__name__)


def fetch_latest_company_news(symbol: str, count: int = 5) -> List[Dict[str, Any]]:
    """Fetch latest company news via Finnhub.

    Args:
        symbol: Ticker symbol
        count: Number of headlines to return
    """
    token = get_secret("FINNHUB_API_KEY", "")
    if not token:
        return []
    # Use last 7 days window
    to_dt = datetime.utcnow().date()
    from_dt = to_dt - timedelta(days=7)
    url = "https://finnhub.io/api/v1/company-news"
    params = {
        "symbol": symbol,
        "from": str(from_dt),
        "to": str(to_dt),
        "token": token,
    }
    try:
        r = requests.get(url, params=params, timeout=6)
        if r.status_code != 200:
            return []
        items = r.json() or []
        # Sort by datetime descending and take top count
        items.sort(key=lambda x: x.get("datetime", 0), reverse=True)
        return items[:count]
    except (requests.RequestException, ValueError, KeyError) as exc:
        logger.debug(f"Finnhub news fetch failed: {exc}")
        return []


def analyze_sentiment_openai(headlines: List[str]) -> Dict[str, Any]:
    """Call OpenAI Chat Completions to score sentiment for a set of headlines.

    Returns a dict with overall sentiment and per-headline scores.
    If OPENAI_API_KEY missing, returns a neutral placeholder.
    """
    api_key = get_secret("OPENAI_API_KEY", "")
    model = get_secret("OPENAI_API_MODEL", "gpt-4o-mini")
    if not api_key or not headlines:
        return {"overall": "NEUTRAL", "confidence": 0.0, "details": []}
    try:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        prompt = (
            "You are an equity news analyst. Given the following 5 headlines, "
            "return a JSON with fields: overall in {POSITIVE, NEGATIVE, NEUTRAL}, "
            "confidence (0-1), and details per headline with sentiment and rationale "
            "(short). Keep it concise."
        )
        messages = [
            {"role": "system", "content": "Analyze equity news sentiment succinctly."},
            {
                "role": "user",
                "content": prompt + "\n\n" + "\n".join(f"- {h}" for h in headlines),
            },
        ]
        body = {"model": model, "messages": messages, "temperature": 0.2}
        r = requests.post(url, headers=headers, json=body, timeout=15)
        if r.status_code != 200:
            return {"overall": "NEUTRAL", "confidence": 0.0, "details": []}
        js = r.json()
        content = (
            js.get("choices", [{}])[0].get("message", {}).get("content", "{}")
        )
        # Try to parse as JSON; if it isn't pure JSON, return as text blob
        try:
            return json.loads(content)
        except (ValueError, TypeError, KeyError):
            return {
                "overall": "NEUTRAL",
                "confidence": 0.0,
                "details": [{"raw": content}],
            }
    except (requests.RequestException, KeyError, TypeError) as api_exc:
        logger.debug(f"OpenAI regime analysis failed: {api_exc}")
        return {"overall": "NEUTRAL", "confidence": 0.0, "details": []}
