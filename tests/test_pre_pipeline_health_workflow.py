"""2026-10-29: the pre-pipeline health check warned 'cash below $50' before EVERY pipeline (CoreTrend holds ~75% of the account, free cash is
permanently ~$46 and the legacy channel needs a $150 minimum notional anyway), burying real warnings. The real checks must stay."""
from pathlib import Path

WF = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "pre_pipeline_health.yml"


def test_permanent_low_cash_warning_is_gone_but_cash_is_still_reported():
    s = WF.read_text()
    assert "CASH < 50" not in s and "below \\$50 buffer" not in s
    assert "cash: \\$${CASH}" in s                        # still shown in the message when a real problem is reported


def test_real_problem_checks_are_intact():
    s = WF.read_text()
    for needle in ("state-feed ${STATE_AGE_MIN}m old", "services not all_active", "auto-trade PAUSED", "pipeline.state=$PIPELINE_STATE"):
        assert needle in s, needle


def test_no_unindented_line_inside_the_run_block():
    """The 2026-05-04 incident: a heredoc body at column 0 ended the YAML block scalar and silently disabled the workflow."""
    lines = WF.read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if "PROBLEMS=()" in l)
    for l in lines[start:start + 40]:
        assert l == "" or l.startswith(" ") or l.startswith("#"), repr(l)
