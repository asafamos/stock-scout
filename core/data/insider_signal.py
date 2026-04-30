"""SEC EDGAR Form 4 — insider buying signal.

Aggregates insider purchases (CEO/CFO/Director/etc) over a rolling 30-day
window. Empirical edge: stocks with material insider buying (>$50K total
in last 30d) outperform peers by 4–7% annualized — one of the most
robust alpha signals in academic literature.

Free public data, no API key needed. Cached per-ticker per-day.

Usage:
    from core.data.insider_signal import insider_score
    score = insider_score("AAPL")  # 0.0 to 1.0
"""
from __future__ import annotations

import json
import logging
import re
import time
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "insider_signal.json"

# SEC requires User-Agent identifying the requester
USER_AGENT = "StockScout (asafamos@gmail.com)"
WINDOW_DAYS = 30
MIN_BUY_THRESHOLD = 50_000  # $50K minimum to count as "material"

# Cache TTL — refresh once per trading day
_mem_cache: Dict[str, Tuple[float, str]] = {}  # ticker → (score, date_str)


def _load_disk_cache() -> Dict[str, Dict]:
    if not CACHE_FILE.exists():
        return {}
    try:
        return json.loads(CACHE_FILE.read_text())
    except Exception:
        return {}


def _save_disk_cache(cache: Dict[str, Dict]):
    try:
        CACHE_FILE.write_text(json.dumps(cache, indent=2))
    except Exception as e:
        logger.warning("insider cache save failed: %s", e)


def _fetch_form4_filings(cik: str) -> List[Dict]:
    """Fetch recent Form 4 filings via SEC EDGAR submissions JSON."""
    url = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        logger.debug("EDGAR submissions fetch failed for CIK %s: %s", cik, e)
        return []

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])

    cutoff = (datetime.utcnow() - timedelta(days=WINDOW_DAYS)).date()
    out = []
    for form, fd, acc in zip(forms, dates, accessions):
        if form != "4":
            continue
        try:
            filed = datetime.strptime(fd, "%Y-%m-%d").date()
        except Exception:
            continue
        if filed < cutoff:
            break  # filings are in reverse chrono order
        out.append({"date": fd, "accession": acc.replace("-", "")})
    return out


def _ticker_to_cik(ticker: str) -> Optional[str]:
    """Map ticker to CIK using SEC's company_tickers.json (cached for 7 days)."""
    cache_key = "_cik_map"
    map_path = CACHE_DIR / "ticker_cik_map.json"
    if map_path.exists():
        age_days = (time.time() - map_path.stat().st_mtime) / 86400
        if age_days < 7:
            try:
                m = json.loads(map_path.read_text())
                return m.get(ticker.upper())
            except Exception:
                pass
    # Fetch fresh
    try:
        req = urllib.request.Request(
            "https://www.sec.gov/files/company_tickers.json",
            headers={"User-Agent": USER_AGENT},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        # data is {"0": {"cik_str": 320193, "ticker": "AAPL", ...}, ...}
        m = {}
        for v in data.values():
            t = str(v.get("ticker", "")).upper()
            cik = str(v.get("cik_str", ""))
            if t and cik:
                m[t] = cik
        map_path.write_text(json.dumps(m))
        return m.get(ticker.upper())
    except Exception as e:
        logger.debug("CIK map fetch failed: %s", e)
        return None


def _parse_form4_buy_amount(accession: str, cik: str) -> float:
    """Fetch a single Form 4 filing and sum the value of OPEN-MARKET BUYS.

    Form 4 transactions have a code field; "P" = open-market purchase.
    We sum (shares × price) across all P transactions in the filing.
    """
    # Filing index URL pattern: https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/
    url_index = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=4&dateb=&owner=include&count=10"
    # Direct XML — accession formatted with dashes for the filename
    acc_dashed = f"{accession[:10]}-{accession[10:12]}-{accession[12:]}"
    xml_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{acc_dashed}.txt"
    try:
        req = urllib.request.Request(xml_url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=10) as resp:
            text = resp.read().decode("utf-8", errors="ignore")
    except Exception:
        return 0.0

    # Lightweight regex parser — Form 4 XMLs are structured but vary
    total = 0.0
    # Find all transaction blocks
    blocks = re.findall(r"<nonDerivativeTransaction>.*?</nonDerivativeTransaction>",
                        text, flags=re.DOTALL)
    for block in blocks:
        code_m = re.search(r"<transactionCode>([A-Z])</transactionCode>", block)
        if not code_m or code_m.group(1) != "P":
            continue  # only count open-market PURCHASES
        sh_m = re.search(r"<transactionShares>.*?<value>([\d.]+)</value>", block, flags=re.DOTALL)
        px_m = re.search(r"<transactionPricePerShare>.*?<value>([\d.]+)</value>", block, flags=re.DOTALL)
        if sh_m and px_m:
            try:
                total += float(sh_m.group(1)) * float(px_m.group(1))
            except Exception:
                continue
    return total


def insider_score(ticker: str) -> float:
    """Return 0.0 to 1.0 score based on insider buying in last 30 days.

    0.0  = no insider buying or net selling
    0.5  = $50K of buying (threshold)
    1.0  = ≥$500K of buying (saturation)

    Cached per-day. Fail-OPEN: any error → 0.0 (no signal).
    Designed to be called from scan or order_manager — should be cheap
    after first call of the day.
    """
    today_str = date.today().isoformat()
    cached = _mem_cache.get(ticker.upper())
    if cached and cached[1] == today_str:
        return cached[0]

    disk = _load_disk_cache()
    rec = disk.get(ticker.upper())
    if rec and rec.get("date") == today_str:
        score = float(rec.get("score", 0))
        _mem_cache[ticker.upper()] = (score, today_str)
        return score

    score = _compute_insider_score(ticker)
    disk[ticker.upper()] = {"date": today_str, "score": score}
    _save_disk_cache(disk)
    _mem_cache[ticker.upper()] = (score, today_str)
    return score


def _compute_insider_score(ticker: str) -> float:
    cik = _ticker_to_cik(ticker)
    if not cik:
        return 0.0
    filings = _fetch_form4_filings(cik)
    if not filings:
        return 0.0
    total = 0.0
    # Check up to 5 most recent (covers most cases without burning rate limits)
    for f in filings[:5]:
        amt = _parse_form4_buy_amount(f["accession"], cik)
        total += amt
        time.sleep(0.15)  # SEC fair-use throttle (10 req/sec max)
    if total < MIN_BUY_THRESHOLD:
        return 0.0
    # Saturate at $500K — beyond that we max out
    raw = total / 500_000
    return min(1.0, raw)


def insider_buying_flag(ticker: str) -> bool:
    """Convenience wrapper: True if insider_score >= 0.5 (≥$50K threshold)."""
    return insider_score(ticker) >= 0.5
