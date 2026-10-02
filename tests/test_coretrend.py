from datetime import date

from core.trading import coretrend as ct


def test_signal_needs_enough_history_and_flips_on_the_sma():
    assert ct.signal([100.0] * 9) is None
    rising = [100 + i for i in range(12)]                 # last close above its 10-month SMA
    s = ct.signal(rising)
    assert s["state"] == ct.RISK_ON and s["target"] == "QQQ" and s["gap_pct"] > 0
    falling = [200 - i for i in range(12)]
    s = ct.signal(falling)
    assert s["state"] == ct.RISK_OFF and s["target"] == "IEF" and s["gap_pct"] < 0


def test_exactly_at_the_sma_is_risk_off():
    assert ct.signal([100.0] * 12)["state"] == ct.RISK_OFF         # strict '>' : no edge, no risk


def test_month_end_closes_takes_the_last_close_of_each_month():
    daily = [(date(2026, 1, 29), 10.0), (date(2026, 1, 30), 11.0), (date(2026, 2, 2), 12.0), (date(2026, 2, 27), 13.0),
             (date(2026, 3, 2), 14.0)]
    assert ct.month_end_closes(daily) == [11.0, 13.0, 14.0]


def test_last_trading_day_of_month_handles_weekends_and_holidays():
    assert ct.is_last_trading_day_of_month(date(2026, 9, 30))          # Wed, next trading day is Oct 1
    assert not ct.is_last_trading_day_of_month(date(2026, 9, 29))
    assert ct.is_last_trading_day_of_month(date(2026, 10, 30))         # Fri before Saturday 31st
    assert not ct.is_last_trading_day_of_month(date(2026, 10, 31))     # Saturday: not a trading day
    assert ct.is_last_trading_day_of_month(date(2026, 11, 30))         # Mon 30th


def test_paper_track_holds_the_prior_month_end_decision_and_switches_on_signals():
    from scripts import coretrend_paper as cp
    from datetime import timedelta
    # synthetic QQQ: rises for 14 months then falls; IEF flat-ish; SPY gently up
    d0 = date(2025, 6, 2); days = []
    d = d0
    while d <= date(2026, 12, 31):
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    qqq, ief, spy = [], [], []
    for k, d in enumerate(days):
        q = 100 + k * 0.30 if d < date(2026, 8, 1) else 100 + k * 0.30 - (k - 290) * 0.9
        qqq.append((d, q)); ief.append((d, 100 + k * 0.01)); spy.append((d, 100 + k * 0.05))
    t = cp.build_track(qqq, ief, spy, inception=date(2026, 3, 2), start_nav=1000.0)
    assert t["state"] in (ct.RISK_ON, ct.RISK_OFF) and t["days"] > 100
    # in the final falling phase the rule must be RISK_OFF and the paper account must be in IEF
    assert t["state"] == ct.RISK_OFF and t["holding_now"] == "IEF"
    assert t["paper_nav"] > 0


from core.trading import coretrend_exec as cx


def test_plan_buys_the_risk_instrument_with_whole_shares_and_keeps_a_reserve():
    p = cx.plan("RISK_ON", {}, {"QQQM": 300.0, "IEF": 89.0}, cash=821.0, netliq=821.0, alloc_pct=100, reserve_usd=25,
                risk_inst="QQQM", safe_inst="IEF")
    assert p == [{"action": "BUY", "symbol": "QQQM", "qty": 2, "price": 300.0, "why": "signal RISK_ON: hold QQQM"}]
    assert 821 - 2 * 300 >= 25


def test_plan_sells_the_wrong_instrument_first_and_buys_later():
    p = cx.plan("RISK_OFF", {"QQQM": 2}, {"QQQM": 300.0, "IEF": 89.0}, 100.0, 700.0, risk_inst="QQQM", safe_inst="IEF")
    assert [a["action"] for a in p] == ["SELL_ALL"] and p[0]["symbol"] == "QQQM" and p[0]["qty"] == 2   # no BUY in the same run


def test_plan_is_idempotent_when_already_in_the_right_instrument():
    assert cx.plan("RISK_ON", {"QQQM": 2}, {"QQQM": 300.0, "IEF": 89.0}, cash=221.0, netliq=821.0,
                   risk_inst="QQQM", safe_inst="IEF") == []


def test_plan_never_spends_below_the_reserve_or_without_a_price():
    assert cx.plan("RISK_ON", {}, {"QQQM": 300.0}, cash=310.0, netliq=821.0, reserve_usd=25, risk_inst="QQQM", safe_inst="IEF") == []
    assert cx.plan("RISK_ON", {}, {"QQQM": 0.0}, cash=800.0, netliq=821.0, risk_inst="QQQM", safe_inst="IEF") == []
