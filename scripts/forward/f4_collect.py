"""F4 collector / batcher / finalizer (prereg docs/research_prereg_F4_earnings_text.md). Deterministic parts only; the scoring is done by fresh subagents.

  python scripts/forward/f4_collect.py collect [--cutoff-et 2026-10-30T08:20]   # new Item 2.02 + EX-99.1 filings since the previous cut-off -> data/forward_text/batches/<entry>_events.json
  python scripts/forward/f4_collect.py prompt K --entry-date D                  # EXACT prompt for scorer batch K (all text inline, no tools needed)
  python scripts/forward/f4_collect.py finalize --entry-date D --scores s1.json [s2.json ...] [--commit]
  python scripts/forward/f4_collect.py status

Secrets: FMP_API_KEY and F4_SEC_CONTACT (sec.gov requires a contact in the User-Agent) come from the untracked .env, never printed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.request
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
ET = ZoneInfo("America/New_York")
PROMPT_FILE = "scripts/forward/f4_prompt.txt"
TEXT_CHARS = 12000
BATCH_SIZE = 10
FIRST_CUTOFF_UTC = "2026-10-29T19:05:35+00:00"   # commit time of the approved prereg: nothing earlier may count
UNIVERSE = {"min_price": 5.0, "min_adv_usd": 1_000_000.0, "min_mcap": 300e6, "max_mcap": 10e9}
# NYSE full-day closures (entry dates must skip them); extend yearly.
HOLIDAYS = {date(2026, 11, 26), date(2026, 12, 25), date(2027, 1, 1), date(2027, 1, 18), date(2027, 2, 15), date(2027, 3, 26), date(2027, 5, 31)}
SUFFIX = r"(incorporated|inc|corporation|corp|company|co|limited|ltd|holdings|holding|plc|llc|lp|n\.v|s\.a|group|the)"


def data_dir() -> Path:
    return Path(os.environ.get("F4_DATA_DIR", ROOT / "data" / "forward_text"))


def sha_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


# ---------- pure helpers (unit-tested) ----------
def entry_date(run_et: datetime) -> date:
    """Open of this session if the batch runs before 09:30 ET, else the next one; skips weekends and NYSE holidays."""
    d = run_et.date()
    if run_et.time() >= dtime(9, 30):
        d += timedelta(days=1)
    while d.weekday() >= 5 or d in HOLIDAYS:
        d += timedelta(days=1)
    return d


def entry_open_utc(d: date) -> datetime:
    return datetime.combine(d, dtime(9, 30), tzinfo=ET).astimezone(timezone.utc)


def parse_items(html: str) -> List[str]:
    return re.findall(r"Item (\d\.\d\d)", html.split("Items</div>", 1)[1].split("<!-- END FILING DIV", 1)[0]) if "Items</div>" in html else []


def parse_exhibit_href(html: str) -> Optional[str]:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "lxml")
    for tr in soup.select("table.tableFile tr"):
        tds = tr.find_all("td")
        if len(tds) >= 4 and tds[3].get_text(strip=True).upper().startswith("EX-99.1"):
            a = tds[2].find("a")
            if a and a.get("href"):
                return a["href"].replace("/ix?doc=", "")
    return None


def html_to_text(html: str) -> str:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "lxml")
    for t in soup(["script", "style"]):
        t.decompose()
    return re.sub(r"[ \t\xa0]+", " ", re.sub(r"\n\s*\n+", "\n\n", soup.get_text("\n"))).strip()


def strip_exhibit_header(text: str) -> str:
    """Drop the wrapper lines some filings put before the release ('EX-99.1', sequence number, file name, 'Document')."""
    lines = text.split("\n")
    i = 0
    while i < min(len(lines), 12) and (not lines[i].strip() or re.fullmatch(r"(EX-99\.1|\d+|\S+\.htm|Document)", lines[i].strip(), flags=re.I)):
        i += 1
    return "\n".join(lines[i:]).strip()


def core_names(company: str) -> List[str]:
    full = re.sub(r"\s+", " ", company or "").strip()
    core = re.sub(rf"[ ,.]+{SUFFIX}\.?$", "", full, flags=re.I)
    core = re.sub(rf"[ ,.]+{SUFFIX}\.?$", "", core, flags=re.I).strip(" ,.")
    return [n for n in dict.fromkeys([full, core]) if len(n) >= 3]


def mask_text(text: str, company: str, ticker: str) -> str:
    out = re.sub(r"https?://\S+|www\.\S+|\b[\w.+-]+@[\w-]+\.[\w.]+\b", "[URL]", text)
    for n in sorted(core_names(company), key=len, reverse=True):
        out = re.sub(re.escape(n), "[COMPANY]", out, flags=re.I)
    if ticker and len(ticker) >= 2:
        out = re.sub(rf"\b{re.escape(ticker)}\b", "[TICKER]", out)
    return out


def in_universe(profile: dict) -> tuple:
    price, mcap = float(profile.get("price") or 0), float(profile.get("marketCap") or 0)
    adv = float(profile.get("averageVolume") or 0) * price
    if price < UNIVERSE["min_price"]:
        return False, f"price {price:.2f} < {UNIVERSE['min_price']}"
    if adv < UNIVERSE["min_adv_usd"]:
        return False, f"ADV$ {adv:,.0f} < {UNIVERSE['min_adv_usd']:,.0f}"
    if not UNIVERSE["min_mcap"] <= mcap < UNIVERSE["max_mcap"]:
        return False, f"marketCap {mcap:,.0f} outside band"
    return True, "ok"


def make_batches(events: List[dict], seed: str, size: int = BATCH_SIZE) -> List[List[dict]]:
    ev = sorted(events, key=lambda e: e["id"])
    random.Random(seed).shuffle(ev)
    return [ev[i:i + size] for i in range(0, len(ev), size)]


def validate_scores(batch_events: List[dict], reply: dict) -> Dict[str, dict]:
    want = {e["id"] for e in batch_events}
    got = {}
    for s in reply.get("scores", []):
        i, sc = str(s.get("id")), s.get("score")
        if i not in want:
            raise ValueError(f"unknown id {i}")
        if i in got:
            raise ValueError(f"duplicate id {i}")
        if not isinstance(sc, int) or isinstance(sc, bool) or not -5 <= sc <= 5:
            raise ValueError(f"score for {i} must be an integer in -5..5")
        got[i] = {"score": sc, "reason": str(s.get("reason", ""))[:300]}
    if set(got) != want:
        raise ValueError(f"missing ids: {sorted(want - set(got))}")
    return got


def parse_reply(text: str) -> dict:
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError("no JSON object found")
    return json.loads(m.group(0))


# ---------- IO ----------
def _ua() -> str:
    c = os.environ.get("F4_SEC_CONTACT")
    if not c:
        sys.exit("F4_SEC_CONTACT missing in .env")
    return f"StockScout-research {c}"


def sec_get(url: str) -> str:
    time.sleep(0.15)                              # <= ~6 req/s, well under the SEC's 10/s limit
    req = urllib.request.Request("https://www.sec.gov" + url if url.startswith("/") else url, headers={"User-Agent": _ua()})
    with urllib.request.urlopen(req, timeout=40) as r:
        return r.read().decode("utf-8", "replace")


def _json(url: str, secrets: tuple) -> object:
    try:
        with urllib.request.urlopen(url, timeout=40) as r:
            return json.loads(r.read())
    except Exception as e:
        msg = str(e)
        for s in secrets:
            msg = msg.replace(s, "***")
        raise RuntimeError(msg[:160]) from None


def parse_atom(xml: str) -> List[dict]:
    """SEC 'getcurrent' atom feed -> [{accession, cik, url, accepted (aware datetime), items, amended}]. The feed carries the item numbers, so only Item 2.02 filings need a second request."""
    import html as _html
    out = []
    for e in re.findall(r"<entry>(.*?)</entry>", xml, flags=re.S):
        title = _html.unescape(re.search(r"<title>(.*?)</title>", e, flags=re.S).group(1))
        acc = re.search(r"AccNo:</b>\s*([\d-]+)", _html.unescape(e))
        upd = re.search(r"<updated>(.*?)</updated>", e)
        link = re.search(r'href="([^"]+-index\.html?)"', e)
        cik = re.search(r"\((\d{10})\)", title)
        if not (acc and upd and link and cik):
            continue
        out.append({"accession": acc.group(1), "cik": str(int(cik.group(1))), "url": link.group(1), "accepted": datetime.fromisoformat(upd.group(1)),
                    "items": re.findall(r"Item (\d\.\d\d)", _html.unescape(e)), "amended": title.startswith("8-K/A")})
    return out


def fetch_feed(start_utc: datetime, max_pages: int = 40) -> List[dict]:
    rows = []
    for page in range(max_pages):
        xml = sec_get(f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&owner=include&start={page * 100}&count=100&output=atom")
        got = parse_atom(xml)
        rows += got
        if not got or min(g["accepted"] for g in got) <= start_utc:
            break
    return rows


def cik_tickers() -> Dict[str, str]:
    d = json.loads(sec_get("https://www.sec.gov/files/company_tickers.json"))
    out: Dict[str, str] = {}
    for r in d.values():
        out.setdefault(str(int(r["cik_str"])), r["ticker"])      # first (primary) ticker per CIK
    return out


def state_path() -> Path:
    return data_dir() / "state.json"


def load_cutoff() -> datetime:
    p = state_path()
    return datetime.fromisoformat(json.load(open(p))["last_cutoff_utc"] if p.exists() else FIRST_CUTOFF_UTC)


def seen_accessions() -> set:
    out = set()
    for p in (data_dir() / "batches").glob("*_events.json"):
        out |= {e["accession"] for e in json.load(open(p))["events"]}
    return out


def collect(cutoff_et: datetime) -> Path:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    fmp = os.environ["FMP_API_KEY"]
    fh = ""
    start_utc, cutoff_utc = load_cutoff(), cutoff_et.astimezone(timezone.utc)
    ent = entry_date(cutoff_et)
    d0, d1 = start_utc.astimezone(ET).date(), cutoff_et.date()
    filings = fetch_feed(start_utc)
    tick = cik_tickers()
    seen = seen_accessions()
    cal = {r["symbol"]: r for r in _json(f"https://financialmodelingprep.com/stable/earnings-calendar?from={d0 - timedelta(days=3)}&to={d1 + timedelta(days=1)}&apikey={fmp}", (fh, fmp))}
    stats = {"filings": 0, "in_window": 0, "item_202": 0, "no_ticker": 0, "no_exhibit": 0, "out_of_universe": 0, "dup": 0, "kept": 0}
    events = []
    for f in filings:
        stats["filings"] += 1
        acc, sym = f["accession"], tick.get(f["cik"])
        acc_et = f["accepted"].astimezone(ET)
        if f["amended"] or "2.02" not in f["items"] or not (start_utc < acc_et.astimezone(timezone.utc) <= cutoff_utc):
            continue
        stats["in_window"] += 1
        stats["item_202"] += 1
        if not sym:
            stats["no_ticker"] += 1
            continue
        if acc in seen:
            stats["dup"] += 1
            continue
        try:
            idx = sec_get(f["url"])
            href = parse_exhibit_href(idx)
            if not href:
                stats["no_exhibit"] += 1
                continue
            raw = strip_exhibit_header(html_to_text(sec_get(href)))
            prof = _json(f"https://financialmodelingprep.com/stable/profile?symbol={sym}&apikey={fmp}", (fh, fmp))
            prof = prof[0] if prof else {}
        except Exception as e:                                   # one bad filing must not lose the batch
            print(f"skip {acc}: {str(e)[:100]}", file=sys.stderr)
            continue
        ok, why = in_universe(prof)
        if not ok:
            stats["out_of_universe"] += 1
            continue
        masked = mask_text(raw, prof.get("companyName", ""), sym)[:TEXT_CHARS]
        c = cal.get(sym, {})
        events.append({"id": acc, "accession": acc, "ticker": sym, "cik": f["cik"], "accepted_et": acc_et.strftime("%Y-%m-%d %H:%M:%S"), "accepted_utc": acc_et.astimezone(timezone.utc).isoformat(timespec="seconds"),
                       "price": prof.get("price"), "market_cap": prof.get("marketCap"), "adv_usd": round(float(prof.get("averageVolume") or 0) * float(prof.get("price") or 0)),
                       "sector": prof.get("sector"), "consensus": {k: c.get(k) for k in ("epsActual", "epsEstimated", "revenueActual", "revenueEstimated")},
                       "text_sha256": sha_text(masked), "text": masked})
        stats["kept"] += 1
    out = data_dir() / "batches" / f"{ent}_events.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        sys.exit(f"{out.name} already exists — one batch per entry date")
    out.write_text(json.dumps({"entry_date": str(ent), "window_start_utc": start_utc.isoformat(timespec="seconds"), "cutoff_utc": cutoff_utc.isoformat(timespec="seconds"),
                               "collected_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "stats": stats, "events": events}, indent=1))
    state_path().write_text(json.dumps({"last_cutoff_utc": cutoff_utc.isoformat(timespec="seconds")}))
    print(json.dumps({"entry_date": str(ent), "file": out.name, **stats}))
    return out


def load_events(ent: str) -> dict:
    return json.load(open(data_dir() / "batches" / f"{ent}_events.json"))


def render_batch(ent: str, k: int) -> str:
    doc = load_events(ent)
    items = []
    for e in make_batches(doc["events"], seed=ent)[k]:
        c = e["consensus"]
        cons = ", ".join(f"{n}={c.get(n)}" for n in ("epsActual", "epsEstimated", "revenueActual", "revenueEstimated"))
        items.append(f"=== RELEASE id={e['id']} | sector={e.get('sector')} | {cons} ===\n{e['text']}")
    return "\n\n".join(items) + "\n"


def prompt_for(ent: str, k: int) -> str:
    """Short prompt that points at a derived batch file (the 100KB of releases would otherwise have to be re-typed by the parent session). The file is a pure function of the committed events file."""
    path = data_dir() / "batches" / f"{ent}_batch{k}.txt"
    path.write_text(render_batch(ent, k))
    base = (ROOT / PROMPT_FILE).read_text().rstrip("\n")
    return f"{base}\n\nReleases file: {path}"


def finalize(ent: str, replies: List[dict], now_utc: Optional[datetime] = None) -> Path:
    now_utc = now_utc or datetime.now(timezone.utc)
    ed = date.fromisoformat(ent)
    if now_utc >= entry_open_utc(ed):
        raise RuntimeError(f"too late: scores must be committed before the entry open {entry_open_utc(ed).isoformat(timespec='minutes')}")
    out = data_dir() / "batches" / f"{ent}_scores.json"
    if out.exists():
        raise RuntimeError(f"{out.name} already exists — one scores file per entry date")
    doc = load_events(ent)
    batches = make_batches(doc["events"], seed=ent)
    if len(replies) != len(batches):
        raise ValueError(f"need {len(batches)} score files (one per batch), got {len(replies)}")
    scores: Dict[str, dict] = {}
    for b, r in zip(batches, replies):
        scores.update(validate_scores(b, r))
    rows = [{"id": e["id"], "ticker": e["ticker"], "accepted_utc": e["accepted_utc"], "text_sha256": e["text_sha256"], **scores[e["id"]]} for e in doc["events"]]
    out.write_text(json.dumps({"entry_date": ent, "prompt_file": PROMPT_FILE, "prompt_sha256": sha_text((ROOT / PROMPT_FILE).read_text()), "scored_utc": now_utc.isoformat(timespec="seconds"),
                               "n": len(rows), "scores": rows}, indent=1))
    return out


def commit_and_push(ent: str) -> None:
    files = [str(p.relative_to(ROOT)) for p in (data_dir() / "batches").glob(f"{ent}_*.json")] + [str((data_dir() / "state.json").relative_to(ROOT))]
    run = lambda *a: subprocess.run(["git", "-C", str(ROOT), *a], check=True)
    run("add", *files)
    run("commit", "-q", "-m", f"F4 batch {ent}: events + scores\n\nCo-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>")
    run("pull", "-q", "--rebase", "--autostash", "origin", "main")
    run("push", "-q", "origin", "main")


def status() -> None:
    n_ev = n_sc = days = 0
    for p in sorted((data_dir() / "batches").glob("*_events.json")):
        n_ev += len(json.load(open(p))["events"]); days += 1
    for p in (data_dir() / "batches").glob("*_scores.json"):
        n_sc += json.load(open(p))["n"]
    print(json.dumps({"batches": days, "events_collected": n_ev, "events_scored": n_sc}))


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("collect"); c.add_argument("--cutoff-et", help="ISO local New York time; default: now")
    p = sp.add_parser("prompt"); p.add_argument("k", type=int); p.add_argument("--entry-date", required=True)
    f = sp.add_parser("finalize"); f.add_argument("--entry-date", required=True); f.add_argument("--scores", nargs="*", default=[]); f.add_argument("--commit", action="store_true")
    n = sp.add_parser("nbatches"); n.add_argument("--entry-date", required=True)
    sp.add_parser("status")
    a = ap.parse_args(argv)
    if a.cmd == "collect":
        collect(datetime.fromisoformat(a.cutoff_et).replace(tzinfo=ET) if a.cutoff_et else datetime.now(ET)); return 0
    if a.cmd == "prompt":
        print(prompt_for(a.entry_date, a.k)); return 0
    if a.cmd == "nbatches":
        print(len(make_batches(load_events(a.entry_date)["events"], seed=a.entry_date))); return 0
    if a.cmd == "status":
        status(); return 0
    path = finalize(a.entry_date, [parse_reply(Path(s).read_text()) for s in a.scores])
    print("written", path)
    if a.commit:
        commit_and_push(a.entry_date); print("committed and pushed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
