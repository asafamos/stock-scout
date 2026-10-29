import json
from datetime import datetime, date, timezone
from zoneinfo import ZoneInfo

import pytest

import scripts.forward.f4_collect as f4

ET = ZoneInfo("America/New_York")

IDX = """<div class="formGrouping"><div class="infoHead">Items</div><div class="info">Item 2.02: Results of Operations and Financial Condition<br />Item 9.01: Financial Statements and Exhibits<br /></div></div>
<!-- END FILING DIV -->
<table class="tableFile" summary="Document Format Files"><tr><th>Seq</th><th>Description</th><th>Document</th><th>Type</th><th>Size</th></tr>
<tr><td>1</td><td>FORM 8-K</td><td><a href="/ix?doc=/Archives/edgar/data/1/2/main_8k.htm">main_8k.htm</a></td><td>8-K</td><td>1</td></tr>
<tr><td>2</td><td>PRESS</td><td><a href="/Archives/edgar/data/1/2/ex991.htm">ex991.htm</a></td><td>EX-99.1</td><td>9</td></tr></table>"""


def test_parse_items_and_exhibit():
    assert f4.parse_items(IDX) == ["2.02", "9.01"]
    assert f4.parse_exhibit_href(IDX) == "/Archives/edgar/data/1/2/ex991.htm"
    assert f4.parse_exhibit_href(IDX.replace("EX-99.1", "EX-10.1")) is None


def test_entry_date_before_and_after_open_and_holidays():
    assert f4.entry_date(datetime(2026, 10, 30, 8, 20, tzinfo=ET)) == date(2026, 10, 30)      # Friday pre-open
    assert f4.entry_date(datetime(2026, 10, 30, 9, 30, tzinfo=ET)) == date(2026, 11, 2)       # at/after open -> Monday
    assert f4.entry_date(datetime(2026, 11, 25, 9, 31, tzinfo=ET)) == date(2026, 11, 27)      # Thanksgiving skipped
    assert f4.entry_date(datetime(2026, 10, 31, 8, 0, tzinfo=ET)) == date(2026, 11, 2)        # Saturday


def test_entry_open_is_0930_et_in_utc():
    assert f4.entry_open_utc(date(2026, 10, 30)) == datetime(2026, 10, 30, 13, 30, tzinfo=timezone.utc)   # EDT
    assert f4.entry_open_utc(date(2026, 11, 2)) == datetime(2026, 11, 2, 14, 30, tzinfo=timezone.utc)      # EST


def test_mask_text_hides_name_ticker_urls():
    t = "TechPrecision Corporation (NASDAQ: TPCS) reported. Visit https://www.techprecision.com or ir@techprecision.com. TechPrecision grew."
    m = f4.mask_text(t, "TechPrecision Corporation", "TPCS")
    assert "TechPrecision" not in m and "TPCS" not in m and "http" not in m and "@" not in m
    assert m.count("[COMPANY]") >= 2


def test_universe_band():
    ok = {"price": 20, "averageVolume": 200_000, "marketCap": 2e9}
    assert f4.in_universe(ok)[0]
    assert not f4.in_universe({**ok, "price": 4})[0]
    assert not f4.in_universe({**ok, "averageVolume": 10_000})[0]
    assert not f4.in_universe({**ok, "marketCap": 50e9})[0]
    assert not f4.in_universe({**ok, "marketCap": 1e8})[0]


def test_batches_are_deterministic_and_cover_all():
    ev = [{"id": str(i)} for i in range(23)]
    a, b = f4.make_batches(ev, "2026-10-30"), f4.make_batches(list(reversed(ev)), "2026-10-30")
    assert a == b and [len(x) for x in a] == [10, 10, 3]
    assert sorted(e["id"] for x in a for e in x) == sorted(e["id"] for e in ev)
    assert f4.make_batches(ev, "2026-10-31") != a


def test_validate_scores():
    be = [{"id": "a"}, {"id": "b"}]
    ok = f4.validate_scores(be, {"scores": [{"id": "a", "score": -5, "reason": "x"}, {"id": "b", "score": 3}]})
    assert ok["a"]["score"] == -5
    for bad in ({"scores": [{"id": "a", "score": 6}, {"id": "b", "score": 0}]},
                {"scores": [{"id": "a", "score": 1.5}, {"id": "b", "score": 0}]},
                {"scores": [{"id": "a", "score": 1}]},
                {"scores": [{"id": "a", "score": 1}, {"id": "a", "score": 2}, {"id": "b", "score": 0}]},
                {"scores": [{"id": "z", "score": 1}, {"id": "a", "score": 1}, {"id": "b", "score": 0}]}):
        with pytest.raises(ValueError):
            f4.validate_scores(be, bad)


def test_finalize_refuses_after_open_and_double_write(tmp_path, monkeypatch):
    monkeypatch.setenv("F4_DATA_DIR", str(tmp_path))
    (tmp_path / "batches").mkdir()
    ev = [{"id": "a", "ticker": "AA", "accepted_utc": "2026-10-29T21:00:00+00:00", "text_sha256": "h", "text": "t", "consensus": {}}]
    (tmp_path / "batches" / "2026-10-30_events.json").write_text(json.dumps({"events": ev}))
    reply = [{"scores": [{"id": "a", "score": 2, "reason": "ok"}]}]
    with pytest.raises(RuntimeError, match="too late"):
        f4.finalize("2026-10-30", reply, now_utc=datetime(2026, 10, 30, 13, 30, tzinfo=timezone.utc))
    out = f4.finalize("2026-10-30", reply, now_utc=datetime(2026, 10, 30, 12, 30, tzinfo=timezone.utc))
    assert json.loads(out.read_text())["scores"][0]["score"] == 2
    with pytest.raises(RuntimeError, match="already exists"):
        f4.finalize("2026-10-30", reply, now_utc=datetime(2026, 10, 30, 12, 31, tzinfo=timezone.utc))


def test_prompt_points_to_batch_file_with_all_releases(tmp_path, monkeypatch):
    monkeypatch.setenv("F4_DATA_DIR", str(tmp_path))
    (tmp_path / "batches").mkdir()
    ev = [{"id": f"x{i}", "ticker": "T", "text": f"body{i}", "sector": "Tech", "consensus": {"epsActual": 1}} for i in range(3)]
    (tmp_path / "batches" / "2026-10-30_events.json").write_text(json.dumps({"events": ev}))
    p = f4.prompt_for("2026-10-30", 0)
    f = tmp_path / "batches" / "2026-10-30_batch0.txt"
    body = f.read_text()
    assert str(f) in p and "NO prices" in p and "body0" not in p            # prompt is short and points at the file
    assert all(f"id=x{i}" in body and f"body{i}" in body for i in range(3))   # the file carries every release


ATOM = """<feed><entry><title>8-K - OSHKOSH CORP (0000775158) (Filer)</title>
<link rel="alternate" type="text/html" href="https://www.sec.gov/Archives/edgar/data/775158/000119312526437039/0001193125-26-437039-index.htm"/>
<summary type="html">&lt;b&gt;Filed:&lt;/b&gt; 2026-10-29 &lt;b&gt;AccNo:&lt;/b&gt; 0001193125-26-437039 &lt;b&gt;Size:&lt;/b&gt; 1 MB &lt;br&gt;Item 2.02: Results of Operations &lt;br&gt;Item 9.01: Exhibits</summary>
<updated>2026-10-29T06:30:00-04:00</updated></entry>
<entry><title>8-K/A - SOME CO (0000000123) (Filer)</title><link rel="alternate" href="https://www.sec.gov/Archives/edgar/data/123/000000000026000001/0000000000-26-000001-index.htm"/>
<summary type="html">&lt;b&gt;AccNo:&lt;/b&gt; 0000000000-26-000001 &lt;br&gt;Item 5.02: x</summary><updated>2026-10-29T07:00:00-04:00</updated></entry></feed>"""


def test_parse_atom():
    rows = f4.parse_atom(ATOM)
    assert len(rows) == 2
    r = rows[0]
    assert r["accession"] == "0001193125-26-437039" and r["cik"] == "775158" and r["items"] == ["2.02", "9.01"] and not r["amended"]
    assert r["accepted"].utcoffset().total_seconds() == -4 * 3600 and r["accepted"].astimezone(timezone.utc).hour == 10
    assert rows[1]["amended"] and rows[1]["items"] == ["5.02"]


def test_strip_exhibit_header():
    raw = "EX-99.1\n\n2\n\na9302026exhibit991.htm\n\nEX-99.1\nDocument\nExhibit 99.1\nCOMPANY REPORTS RESULTS"
    assert f4.strip_exhibit_header(raw).startswith("Exhibit 99.1")


def test_resolver_counts_only(tmp_path, monkeypatch):
    import scripts.forward.f4_resolve as r
    monkeypatch.setenv("F4_DATA_DIR", str(tmp_path))
    (tmp_path / "batches").mkdir()
    (tmp_path / "batches" / "2026-10-30_scores.json").write_text(json.dumps({"entry_date": "2026-10-30", "n": 5, "scores": []}))
    out = r.main(today=date(2026, 12, 1))
    assert out["events_scored"] == 5 and out["events_resolved_by_calendar"] == 5 and not out["stopping_rule_met"]
    assert set(out) == {"events_scored", "events_resolved_by_calendar", "resolved_entry_dates", "stopping_rule_met", "note"}   # nothing about returns
    assert r.trading_days_after(date(2026, 11, 20), 5) == date(2026, 11, 30)    # skips Thanksgiving + weekend
