"""
Lightweight provider telemetry collector for pipeline runs.

Tracks which providers were used across domains and any fallback events.

Export schema:
{
  "universe_provider": <str or None>,
  "price": {provider: true, ...},
  "fundamentals": {provider: true, ...},
  "index": {symbol: provider, ...},
  "fallback_events": [ {"stage": str, "from": str, "to": str, "reason": str}, ... ]
}
"""
from __future__ import annotations
from typing import Dict, Any, Optional


class Telemetry:
    def __init__(self) -> None:
        self._universe_provider: Optional[str] = None
        self._domains: Dict[str, Dict[str, bool]] = {"price": {}, "fundamentals": {}}
        self._index: Dict[str, str] = {}
        self._fallback_events: list[Dict[str, Any]] = []
        self._domain_status: Dict[str, str] = {}
        self._provider_states: Dict[str, Any] = {}

    def _normalize_provider(self, name: str) -> str:
        try:
            n = str(name or "").strip()
            upper = n.upper()
            # Common provider normalizations
            mapping = {
                "POLYGON": "POLYGON",
                "FMP": "FMP",
                "FINNHUB": "FINNHUB",
                "TIINGO": "TIINGO",
                "ALPHAVANTAGE": "ALPHAVANTAGE",
                "EODHD": "EODHD",
                "SIMFIN": "SIMFIN",
                "MARKETSTACK": "MARKETSTACK",
                "YFINANCE": "YFINANCE",
            }
            # Handle common camelcase inputs
            alt = {
                "ALPHAVANTAGE": ["ALPHA VANTAGE", "ALPHA_VANTAGE", "ALPHAVANTAGE"],
                "YFINANCE": ["YFINANCE", "YAHOOFINANCE", "YAHOO_FINANCE"],
            }
            # Synthetic proxies
            if "SYNTHETICVIX" in upper or "SYNTHETIC_VIX" in upper:
                return "SYNTHETIC_VIX_PROXY"
            # Normalize known variations
            for canon, variants in alt.items():
                if upper in variants:
                    return canon
            # Default mapping
            return mapping.get(upper, upper)
        except Exception:
            return str(name).upper()

    def mark_used(self, domain: str, provider: str) -> None:
        try:
            d = str(domain).lower()
            p = self._normalize_provider(provider)
            if d not in self._domains:
                self._domains[d] = {}
            self._domains[d][p] = True
        except Exception:
            pass

    def mark_index(self, symbol: str, provider: str) -> None:
        try:
            self._index[str(symbol)] = self._normalize_provider(provider)
        except Exception:
            pass

    def record_fallback(self, stage: str, from_provider: str, to_provider: str, reason: str) -> None:
        try:
            self._fallback_events.append({
                "stage": str(stage),
                "from": self._normalize_provider(from_provider),
                "to": self._normalize_provider(to_provider),
                "reason": str(reason),
            })
        except Exception:
            pass

    def set_value(self, key: str, value: Any) -> None:
        try:
            if key == "universe_provider":
                self._universe_provider = str(value) if value is not None else None
            elif key == "index" and isinstance(value, dict):
                for k, v in value.items():
                    self.mark_index(k, v)
            elif key.endswith("_status") and isinstance(key, str):
                # e.g., fundamentals_status
                domain = key.replace("_status", "")
                self._domain_status[str(domain).lower()] = str(value)
            elif key in ("price", "fundamentals") and isinstance(value, dict):
                for p, flag in value.items():
                    if flag:
                        self.mark_used(key, p)
            elif key == "provider_states" and isinstance(value, dict):
                self._provider_states = dict(value)
        except Exception:
            pass

    def export(self) -> Dict[str, Any]:
        try:
            out = {
                "universe_provider": self._universe_provider,
                "price": dict(self._domains.get("price", {})),
                "fundamentals": dict(self._domains.get("fundamentals", {})),
                "index": dict(self._index),
                "fallback_events": list(self._fallback_events),
                "provider_states": dict(self._provider_states) if self._provider_states else {},
            }
            # Attach domain statuses compactly
            fund_status = self._domain_status.get("fundamentals")
            if fund_status:
                fdict = out.get("fundamentals") or {}
                fdict["_status"] = fund_status
                out["fundamentals"] = fdict
            return out
        except Exception:
            return {
                "universe_provider": self._universe_provider,
                "price": {},
                "fundamentals": {"_status": self._domain_status.get("fundamentals")} if self._domain_status.get("fundamentals") else {},
                "index": {},
                "fallback_events": [],
                "provider_states": dict(self._provider_states) if self._provider_states else {},
            }
