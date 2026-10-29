"""F2 weekly round — deterministic helper for the automated run (docs/f2_round_runbook.md, prereg docs/research_prereg_F2_llm_picker.md A6).

Everything that needs no judgement lives here and is unit-tested; the scheduled Claude session only runs the three blind pickers in between.

  python scripts/forward/f2_round.py plan                        # JSON: round_date, asof, deadline_utc, status (run | done | too_late)
  python scripts/forward/f2_round.py build  --round-date D       # pool + packs (arm A/C numeric pack, arm B news pack)
  python scripts/forward/f2_round.py prompt A|B|C --round-date D # the EXACT prompt text for that arm's fresh picker
  python scripts/forward/f2_round.py finalize --round-date D --a a.json --b b.json --c c.json [--commit]

Rules enforced here (not left to the operator): picks must be committed BEFORE the entry open (09:30 New York time of the first trading day
after the round date; Monday is used as the conservative deadline), exactly 5 distinct tickers per arm from the logged pool, one picks file per round.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
NY = ZoneInfo("America/New_York")
PROMPTS = {"A": "scripts/forward/f2_prompt.txt", "B": "scripts/forward/f2_prompt_news.txt", "C": "scripts/forward/f2_prompt_agent.txt"}


def fd() -> Path:
    return Path(os.environ.get("F2_DATA_DIR", ROOT / "data" / "forward_llm"))


def round_date_for(now_utc: datetime) -> date:
    """The round date = the Sunday on/before 'now' (New York calendar)."""
    d = now_utc.astimezone(NY).date()
    return d - timedelta(days=(d.weekday() + 1) % 7)


def asof_for(round_date: date) -> date:
    """Data cut-off = the last completed session before the Sunday round = Friday (a holiday Friday just yields Thursday's last bar)."""
    return round_date - timedelta(days=2)


def deadline_utc(round_date: date) -> datetime:
    """Conservative deadline: 09:30 New York on the Monday after the round date."""
    monday = round_date + timedelta(days=1)
    return datetime.combine(monday, time(9, 30), tzinfo=NY).astimezone(timezone.utc)


def picks_path(round_date: date) -> Path:
    return fd() / f"round_{round_date}_picks.json"


def plan(now_utc: Optional[datetime] = None) -> Dict[str, str]:
    now_utc = now_utc or datetime.now(timezone.utc)
    rd = round_date_for(now_utc)
    dl = deadline_utc(rd)
    status = "done" if picks_path(rd).exists() else ("too_late" if now_utc >= dl else "run")
    return {"round_date": str(rd), "asof": str(asof_for(rd)), "deadline_utc": dl.isoformat(timespec="minutes"), "status": status}


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def prompt_for(arm: str, round_date: date) -> str:
    """Exactly the wording used in the manually run rounds (prompt file + one line pointing at the arm's pack file)."""
    base = (ROOT / PROMPTS[arm]).read_text().rstrip("\n")
    pack = fd() / (f"round_{round_date}_newspack.txt" if arm == "B" else f"round_{round_date}_pack.txt")
    if arm == "A":
        tail = f"The pool data is in the file {pack} (pipe-separated table; columns are named in the first line). Read ONLY that one file with the Read tool and nothing else: no other file, no search, no web."
    elif arm == "B":
        tail = f"The pool is in the file {pack}. Read ONLY that one file with the Read tool and nothing else: no other file, no search, no web."
    else:
        tail = f"Pool file: {pack}"
    return f"{base}\n\n{tail}"


def parse_picks(text: str) -> dict:
    """Accept the picker's JSON reply, tolerating a ```json fence or surrounding prose."""
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError("no JSON object found")
    return json.loads(m.group(0))


def validate_arm(name: str, obj: dict, pool: set) -> List[dict]:
    picks = obj.get("picks")
    if not isinstance(picks, list) or len(picks) != 5:
        raise ValueError(f"arm {name}: need exactly 5 picks")
    out, seen = [], set()
    for p in picks:
        t = str(p.get("ticker", "")).upper().strip()
        c = p.get("confidence")
        if t in seen:
            raise ValueError(f"arm {name}: duplicate ticker {t}")
        if t not in pool:
            raise ValueError(f"arm {name}: {t} is not in the pool")
        if not isinstance(c, int) or not 1 <= c <= 5:
            raise ValueError(f"arm {name}: confidence for {t} must be an integer 1-5")
        seen.add(t)
        out.append({"ticker": t, "confidence": c})
    return out


def finalize(round_date: date, a: dict, b: dict, c: dict, now_utc: Optional[datetime] = None) -> Path:
    now_utc = now_utc or datetime.now(timezone.utc)
    if now_utc >= deadline_utc(round_date):
        raise RuntimeError(f"too late: picks must be committed before {deadline_utc(round_date).isoformat(timespec='minutes')} (entry open)")
    out_path = picks_path(round_date)
    if out_path.exists():
        raise RuntimeError(f"{out_path.name} already exists — one picks file per round")
    pool_file = fd() / f"round_{round_date}_pool.json"
    pool = {p["ticker"] for p in json.load(open(pool_file))["pool"]}
    if len(pool) != 100:
        raise ValueError(f"pool has {len(pool)} names, expected 100")
    pa, pb, pc = validate_arm("A", a, pool), validate_arm("B", b, pool), validate_arm("C", c, pool)
    pack, news = fd() / f"round_{round_date}_pack.txt", fd() / f"round_{round_date}_newspack.txt"
    sources = [str(s) for s in (c.get("sources") or [])][:5]
    doc = {
        "round_date": str(round_date), "data_asof": str(asof_for(round_date)), "automated": True,
        "note": "automated weekly round (prereg A6): data as of Friday close; entry = Open of the first trading day after the round date",
        "exit": "open 20 trading days after entry", "pool_file": pool_file.name, "pool_sha256": sha(pool_file),
        "logged_utc": now_utc.isoformat(timespec="seconds"),
        "arms": {
            "A_numbers": {"prompt_file": PROMPTS["A"], "prompt_sha256": sha(ROOT / PROMPTS["A"]), "pack_file": f"data/forward_llm/{pack.name}", "pack_sha256": sha(pack),
                          "picker": "fresh subagent, no tools except reading the pack", "picks": pa},
            "B_news": {"prompt_file": PROMPTS["B"], "prompt_sha256": sha(ROOT / PROMPTS["B"]), "pack_file": f"data/forward_llm/{news.name}", "pack_sha256": sha(news),
                       "picker": "fresh subagent, no tools except reading the pack", "picks": pb},
            "C_web_agent": {"prompt_file": PROMPTS["C"], "prompt_sha256": sha(ROOT / PROMPTS["C"]), "pack_file": f"data/forward_llm/{pack.name}", "pack_sha256": sha(pack),
                            "picker": "fresh subagent WITH web search/fetch", "picks": pc, "sources": sources},
        },
    }
    out_path.write_text(json.dumps(doc, indent=1))
    return out_path


def build(round_date: date) -> None:
    env = dict(os.environ, F2_ASOF=str(asof_for(round_date)))
    out = fd() / f"round_{round_date}_pack.txt"
    with open(out, "w") as fh:
        subprocess.run([sys.executable, str(ROOT / "scripts/forward/f2_build_pack.py"), str(round_date)], stdout=fh, env=env, check=True, cwd=ROOT)
    subprocess.run([sys.executable, str(ROOT / "scripts/forward/f2_build_news.py"), str(round_date)], env=env, check=True, cwd=ROOT)


def commit_and_push(round_date: date) -> None:
    files = [str(p.relative_to(ROOT)) for p in fd().glob(f"round_{round_date}_*")]
    run = lambda *a: subprocess.run(["git", "-C", str(ROOT), *a], check=True)
    run("add", *files)
    run("commit", "-q", "-m", f"F2 automated round {round_date}: pool, packs, picks (arms A/B/C)\n\nCo-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>")
    run("pull", "-q", "--rebase", "--autostash", "origin", "main")
    run("push", "-q", "origin", "main")


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("plan")
    b = sp.add_parser("build"); b.add_argument("--round-date", required=True)
    p = sp.add_parser("prompt"); p.add_argument("arm", choices=["A", "B", "C"]); p.add_argument("--round-date", required=True)
    f = sp.add_parser("finalize"); f.add_argument("--round-date", required=True)
    for k in "abc":
        f.add_argument(f"--{k}", required=True, help="file with that arm's JSON reply")
    f.add_argument("--commit", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        print(json.dumps(plan())); return 0
    rd = date.fromisoformat(a.round_date)
    if a.cmd == "build":
        build(rd); return 0
    if a.cmd == "prompt":
        print(prompt_for(a.arm, rd)); return 0
    path = finalize(rd, *(parse_picks(Path(getattr(a, k)).read_text()) for k in "abc"))
    print("written", path)
    if a.commit:
        commit_and_push(rd); print("committed and pushed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
