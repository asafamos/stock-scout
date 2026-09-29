"""A failed target-hit sell must not leave the position unprotected (2026-09-29 audit).

_target_hit_pass cancels the position's TRAIL+LMT to make room for the sell. If the sell then
fails (sub-$2k tier: limit unfilled, no MKT fallback), force_exit_via_trail has no TRAIL to
modify -> the position stayed naked until the next monitor cycle, and the alert lied
("TRAIL still protecting downside"). It must restore protection immediately and say so truthfully.
"""
from types import SimpleNamespace

import scripts.monitor_positions as mp
from core.trading import live_quote as lq


class _Notify:
    def __init__(self):
        self.errors, self.sent = [], []

    def notify_error(self, ctx, msg):
        self.errors.append((ctx, msg))

    def _send(self, msg):
        self.sent.append(msg)


def _harness(resubmit_status="Submitted", resubmit_raises=False):
    pos = {"ticker": "PBF", "entry_price": 70.0, "target_price": 80.0, "quantity": 8,
           "trailing_stop_pct": 5.5, "order_ids": {"oca_group": "SS_PBF_1"}}
    tracker = SimpleNamespace(get_open_positions=lambda: [pos], _save_positions=lambda p: None)
    live = SimpleNamespace(order=SimpleNamespace(orderId=11))
    cancelled, resubmits = [], []
    ib = SimpleNamespace(
        portfolio=lambda: [SimpleNamespace(contract=SimpleNamespace(symbol="PBF"), position=8, marketPrice=80.5)],
        openTrades=lambda: [live], cancelOrder=lambda o: cancelled.append(o.orderId), sleep=lambda s: None)

    def resubmit(ticker, qty, trail, target, **kw):
        if resubmit_raises:
            raise RuntimeError("IB down")
        resubmits.append((ticker, qty, trail, target))
        st = SimpleNamespace(status=resubmit_status, error="")
        return {"trailing_stop": st, "limit_sell": st}

    client = SimpleNamespace(
        _ib=ib,
        _sell_market=lambda t, q: SimpleNamespace(status="Cancelled_LimitUnfilled_SubTier", filled_price=0.0, order_id=1),
        force_exit_via_trail=lambda t, aggressive=True: SimpleNamespace(status="no_active_trail", filled_price=0.0),
        resubmit_protective_orders_retry=resubmit)
    orders = [{"ticker": "PBF", "status": "Submitted", "oca_group": "SS_PBF_1", "order_type": "TRAIL", "order_id": 11}]
    return pos, tracker, client, orders, cancelled, resubmits


def test_failed_sell_restores_protection_and_alert_is_truthful(monkeypatch):
    monkeypatch.setattr(lq, "get_realtime_price", lambda t: None)  # use the IB mark (80.5 >= target)
    pos, tracker, client, orders, cancelled, resubmits = _harness()
    n = _Notify()
    mp._target_hit_pass(tracker, client, orders, n)
    assert cancelled == [11], "the OCA leg was cancelled to make room for the sell"
    assert resubmits == [("PBF", 8, 5.5, 80.0)], "protection must be restored immediately"
    msg = " ".join(m for _, m in n.errors)
    assert "TRAIL restored" in msg and "still protecting" not in msg


def test_failed_sell_and_failed_restore_says_unprotected(monkeypatch):
    monkeypatch.setattr(lq, "get_realtime_price", lambda t: None)
    pos, tracker, client, orders, *_ = _harness(resubmit_raises=True)
    n = _Notify()
    mp._target_hit_pass(tracker, client, orders, n)
    msg = " ".join(m for _, m in n.errors)
    assert "UNPROTECTED" in msg, "a failed restore must be reported loudly, not hidden"
