from scripts import weekly_vs_spy as w


def test_compare_nets_out_external_flows():
    # account 1000 -> 1250 with a +200 deposit = +5% ; SPY +3% ; excess +2pp
    c = w.compare(1250.0, 1000.0, 200.0, 103.0, 100.0)
    assert round(c["account_pct"], 2) == 5.0 and round(c["spy_pct"], 2) == 3.0 and round(c["excess_pp"], 2) == 2.0


def test_compare_handles_loss_vs_up_market():
    c = w.compare(821.0, 977.5, 0.0, 104.1, 100.0)
    assert c["account_pct"] < -15 and c["excess_pp"] < -19
