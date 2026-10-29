"""2026-10-28: the healthcheck read the LAST LINE of pending_scans.jsonl as 'latest scan_date'; after the resolver rewrote the file a few
unresolved 2026-09-29 rows sat at the end, so a perfectly healthy tracker looked 29 days stale and alerted every 15 minutes."""
import re
import subprocess
from pathlib import Path

HC = Path(__file__).resolve().parents[1] / "deploy" / "healthcheck.sh"


def _last_scan(pending: Path) -> str:
    src = HC.read_text()
    line = next(l for l in src.splitlines() if l.strip().startswith("LAST_SCAN=$(grep"))
    out = subprocess.run(["bash", "-c", f'PENDING="{pending}"\n{line.strip()}\necho "$LAST_SCAN"'], capture_output=True, text=True, check=True)
    return out.stdout.strip()


def test_uses_the_max_date_not_the_last_line(tmp_path):
    f = tmp_path / "pending_scans.jsonl"
    rows = ['{"scan_date": "2026-10-27", "ticker": "AAA"}', '{"scan_date": "2026-10-28", "ticker": "BBB"}',
            '{"scan_date": "2026-09-29", "ticker": "QRVO"}', '{"scan_date": "2026-09-29", "ticker": "CCC"}']     # stuck old rows at the END
    f.write_text("\n".join(rows) + "\n")
    assert _last_scan(f) == "2026-10-28"


def test_handles_compact_json_and_empty_file(tmp_path):
    f = tmp_path / "p.jsonl"
    f.write_text('{"scan_date":"2026-10-20","ticker":"A"}\n{"scan_date":"2026-10-22","ticker":"B"}\n')
    assert _last_scan(f) == "2026-10-22"
    f.write_text("")
    assert _last_scan(f) == ""


def test_the_alert_is_deduplicated_and_cleared_when_healthy():
    src = HC.read_text()
    assert re.search(r'send_alert_dedup "outcomes_stale"', src)
    assert 'clear_alert_dedup "outcomes_stale"' in src
