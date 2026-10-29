"""F2 automation helper: the parts that must never depend on a human remembering them (dates, the pre-open deadline, pick validation)."""
import json
from datetime import date, datetime, timezone

import pytest

from scripts.forward import f2_round as f2

UTC = timezone.utc


def _dt(y, m, d, hh=12, mm=0):
    return datetime(y, m, d, hh, mm, tzinfo=UTC)


@pytest.fixture
def fdir(tmp_path, monkeypatch):
    monkeypatch.setenv("F2_DATA_DIR", str(tmp_path))
    return tmp_path


def _make_round(fdir, rd="2026-11-01"):
    pool = [{"ticker": f"T{i:03d}"} for i in range(100)]
    (fdir / f"round_{rd}_pool.json").write_text(json.dumps({"pool": pool}))
    (fdir / f"round_{rd}_pack.txt").write_text("ticker | x\n" + "\n".join(p["ticker"] for p in pool))
    (fdir / f"round_{rd}_newspack.txt").write_text("\n".join(p["ticker"] for p in pool))
    return {p["ticker"] for p in pool}


def _arm(*tickers, conf=3):
    return {"picks": [{"ticker": t, "confidence": conf, "reason": "x"} for t in tickers]}


def test_round_date_is_the_sunday_on_or_before_now_in_new_york():
    assert f2.round_date_for(_dt(2026, 10, 25, 12)) == date(2026, 10, 25)      # Sunday
    assert f2.round_date_for(_dt(2026, 10, 26, 8)) == date(2026, 10, 25)       # Monday morning still belongs to that Sunday's round
    assert f2.round_date_for(_dt(2026, 10, 29, 11)) == date(2026, 10, 25)
    assert f2.round_date_for(_dt(2026, 11, 2, 13)) == date(2026, 11, 1)        # across the US DST change
    assert f2.round_date_for(_dt(2026, 10, 26, 3, 59)) == date(2026, 10, 25)   # Sunday evening in New York


def test_deadline_is_monday_0930_new_york_across_dst():
    assert f2.deadline_utc(date(2026, 10, 25)) == _dt(2026, 10, 26, 13, 30)    # EDT: 13:30 UTC
    assert f2.deadline_utc(date(2026, 11, 1)) == _dt(2026, 11, 2, 14, 30)      # EST: 14:30 UTC
    assert f2.asof_for(date(2026, 10, 25)) == date(2026, 10, 23)               # Friday


def test_plan_statuses(fdir):
    assert f2.plan(_dt(2026, 10, 25, 12))["status"] == "run"
    assert f2.plan(_dt(2026, 10, 26, 8))["status"] == "run"                    # the Monday catch-up window
    assert f2.plan(_dt(2026, 10, 26, 13, 31))["status"] == "too_late"          # after the open: a round can no longer count
    (fdir / "round_2026-10-25_picks.json").write_text("{}")
    assert f2.plan(_dt(2026, 10, 26, 8))["status"] == "done"                   # idempotent: the catch-up does nothing


def test_prompt_wording_matches_the_manual_rounds(fdir):
    _make_round(fdir, "2026-11-01")
    rd = date(2026, 11, 1)
    a, b, c = (f2.prompt_for(x, rd) for x in "ABC")
    assert a.startswith("You are an equity analyst. Below is a pool of 100 US stocks with point-in-time data") and a.rstrip().endswith("no search, no web.")
    assert "round_2026-11-01_pack.txt" in a and "round_2026-11-01_newspack.txt" not in a
    assert "round_2026-11-01_newspack.txt" in b and "There is NO price or fundamental data" in b
    assert "You MAY use web search" in c and c.rstrip().endswith("round_2026-11-01_pack.txt")


def test_parse_picks_tolerates_fences_and_prose():
    assert f2.parse_picks('Here:\n```json\n{"picks":[{"ticker":"A","confidence":3}]}\n```')["picks"][0]["ticker"] == "A"
    with pytest.raises(ValueError):
        f2.parse_picks("no json here")


def test_finalize_writes_hashes_and_validates(fdir):
    names = sorted(_make_round(fdir))
    rd = date(2026, 11, 1)
    c = _arm(*names[:5]); c["sources"] = ["u1", "u2"]
    p = f2.finalize(rd, _arm(*names[5:10]), _arm(*names[10:15]), c, now_utc=_dt(2026, 11, 1, 12))
    doc = json.loads(p.read_text())
    assert doc["round_date"] == "2026-11-01" and doc["data_asof"] == "2026-10-30" and doc["automated"] is True
    assert [x["ticker"] for x in doc["arms"]["A_numbers"]["picks"]] == names[5:10]
    assert doc["arms"]["C_web_agent"]["sources"] == ["u1", "u2"]
    assert len(doc["pool_sha256"]) == 64 and len(doc["arms"]["B_news"]["pack_sha256"]) == 64
    with pytest.raises(RuntimeError, match="already exists"):
        f2.finalize(rd, _arm(*names[5:10]), _arm(*names[10:15]), c, now_utc=_dt(2026, 11, 1, 13))


@pytest.mark.parametrize("bad,msg", [
    (lambda n: _arm(*n[:4]), "exactly 5"),
    (lambda n: _arm(n[0], n[0], n[1], n[2], n[3]), "duplicate"),
    (lambda n: _arm(*n[:4], "ZZZ"), "not in the pool"),
    (lambda n: {"picks": [{"ticker": t, "confidence": 9} for t in n[:5]]}, "confidence"),
])
def test_finalize_rejects_bad_picks(fdir, bad, msg):
    names = sorted(_make_round(fdir))
    good = _arm(*names[5:10])
    with pytest.raises(ValueError, match=msg):
        f2.finalize(date(2026, 11, 1), bad(names), good, good, now_utc=_dt(2026, 11, 1, 12))


def test_finalize_refuses_after_the_open(fdir):
    names = sorted(_make_round(fdir))
    good = _arm(*names[:5])
    with pytest.raises(RuntimeError, match="too late"):
        f2.finalize(date(2026, 11, 1), good, good, good, now_utc=_dt(2026, 11, 2, 14, 31))
    assert not (fdir / "round_2026-11-01_picks.json").exists()


def test_watchdog_shell_logic_counts_any_picks_file_on_or_after_the_sunday(tmp_path):
    """The f2_watchdog workflow line that decides 'is the round logged?' (off-schedule rounds count too)."""
    import subprocess
    from pathlib import Path
    wf = (Path(__file__).resolve().parents[1] / ".github" / "workflows" / "f2_watchdog.yml").read_text()
    line = next(l.strip() for l in wf.splitlines() if l.strip().startswith("FOUND=$(ls data/forward_llm"))
    d = tmp_path / "data" / "forward_llm"; d.mkdir(parents=True)
    for n in ("2026-10-11", "2026-10-18", "2026-10-29"):
        (d / f"round_{n}_picks.json").write_text("{}")
    run = lambda sun: subprocess.run(["bash", "-c", f'cd "{tmp_path}"; SUN={sun}\n{line}\necho "[$FOUND]"'], capture_output=True, text=True).stdout.strip()
    assert run("2026-10-25") == "[2026-10-29]"      # the off-schedule round after Sunday 10-25 counts
    assert run("2026-11-01") == "[]"                 # nothing on/after 11-01 -> the watchdog would alert
    assert run("2026-10-18") == "[2026-10-18]"
