"""Regression tests for IBKRClient.buy_with_bracket fill handling (2026-09-29 audit).

Bugs these guard:
  * a partial fill whose order was then cancelled (status "Cancelled", filled>0) was treated
    as "unfilled" -> shares held with NO protective orders and NO tracker row;
  * cancel confirmation was a fixed 1s sleep, so a PendingCancel order could fill late;
  * an exception after the BUY was placed returned Error with no cleanup.
A tiny fake IB drives the REAL buy_with_bracket code; no network, no ib connection.
"""
from types import SimpleNamespace

import pytest

pytest.importorskip("ib_insync")
from core.trading import ibkr_client as ic  # noqa: E402


class FakeTrade:
    def __init__(self, order, status="PendingSubmit", filled=0, avg=0.0):
        self.order = order
        self.orderStatus = SimpleNamespace(status=status, filled=filled, avgFillPrice=avg, lastFillPrice=avg)
        self.log = []


class FakeIB:
    """scenario: how the BUY (first order) behaves.
    after_cancel: status the BUY reaches after cancelOrder (after `cancel_lag` sleeps).
    """

    def __init__(self, buy_status="Submitted", buy_filled=0, buy_avg=0.0,
                 after_cancel="Cancelled", cancel_lag=0, held=0, fail_on_second_place=False):
        self.buy_status, self.buy_filled, self.buy_avg = buy_status, buy_filled, buy_avg
        self.after_cancel, self.cancel_lag = after_cancel, cancel_lag
        self.held, self.fail_on_second_place = held, fail_on_second_place
        self.placed, self.cancelled, self.sleeps = [], [], 0
        self._cancel_at = None
        self._buy_trade = None

    def qualifyContracts(self, c):
        return [c]

    def placeOrder(self, contract, order):
        self.placed.append(order)
        if len(self.placed) == 1:  # the parent BUY
            self._buy_trade = FakeTrade(order, self.buy_status, self.buy_filled, self.buy_avg)
            return self._buy_trade
        if self.fail_on_second_place:
            raise RuntimeError("placeOrder blew up")
        return FakeTrade(order, "Submitted")

    def cancelOrder(self, order):
        self.cancelled.append(order)
        self._cancel_at = self.sleeps + self.cancel_lag

    def sleep(self, s):
        self.sleeps += 1
        t = self._buy_trade
        if t is not None and self._cancel_at is not None and self.sleeps >= self._cancel_at:
            t.orderStatus.status = self.after_cancel
        elif t is not None and self._cancel_at is not None:
            t.orderStatus.status = "PendingCancel"

    def positions(self):
        if self.held <= 0:
            return []
        return [SimpleNamespace(contract=SimpleNamespace(symbol="PBF"), position=float(self.held), avgCost=75.0)]


def make_client(fake):
    c = ic.IBKRClient.__new__(ic.IBKRClient)
    c._ib = fake
    c.cfg = SimpleNamespace(dry_run=False, entry_use_limit=True, entry_limit_fill_wait_sec=2)
    c.get_net_liquidation = lambda: 800.0  # sub-$2k tier
    c._next_trading_day_open_utc = lambda: ""
    return c


def _protective_qtys(fake):
    return [o.totalQuantity for o in fake.placed[1:]]


def test_partial_fill_then_cancel_is_kept_and_protected():
    fake = FakeIB(buy_status="Submitted", buy_filled=3, buy_avg=75.30, after_cancel="Cancelled")
    res = make_client(fake).buy_with_bracket("PBF", 8, 9.0, 84.0, limit_price=75.56)
    assert res["buy"].status != "Error"
    assert res["buy"].quantity == 3 and res["buy"].filled_price == pytest.approx(75.30)
    assert _protective_qtys(fake) == [3, 3]  # TRAIL + LMT sized to the 3 shares actually owned


def test_zero_fill_is_unfilled_and_places_no_protective_orders():
    fake = FakeIB(buy_status="Submitted", buy_filled=0, after_cancel="Cancelled")
    res = make_client(fake).buy_with_bracket("PBF", 8, 9.0, 84.0, limit_price=75.56)
    assert res["buy"].status == "Error"
    assert len(fake.placed) == 1  # only the BUY; nothing to protect


def test_waits_for_cancel_confirmation_instead_of_fixed_1s():
    # cancel is confirmed only after 5 more sleeps; the old code gave up after 1
    fake = FakeIB(buy_status="Submitted", buy_filled=0, after_cancel="Cancelled", cancel_lag=5)
    make_client(fake).buy_with_bracket("PBF", 8, 9.0, 84.0, limit_price=75.56)
    assert fake.cancelled, "remainder must be cancelled"
    assert fake._buy_trade.orderStatus.status == "Cancelled"


def test_unconfirmed_cancel_adopts_what_ib_actually_holds():
    # cancel never confirms (stays PendingCancel) but IB shows 5 shares -> protect those 5
    fake = FakeIB(buy_status="Submitted", buy_filled=0, after_cancel="PendingCancel", cancel_lag=999, held=5)
    res = make_client(fake).buy_with_bracket("PBF", 8, 9.0, 84.0, limit_price=75.56)
    assert res["buy"].status != "Error"
    assert res["buy"].quantity == 5
    assert _protective_qtys(fake) == [5, 5]


def test_full_fill_unchanged():
    fake = FakeIB(buy_status="Filled", buy_filled=8, buy_avg=75.4)
    res = make_client(fake).buy_with_bracket("PBF", 8, 9.0, 84.0, limit_price=75.56)
    assert res["buy"].status == "Filled" and res["buy"].quantity == 8
    assert not fake.cancelled and _protective_qtys(fake) == [8, 8]


def test_exception_after_buy_with_held_shares_attempts_protection(monkeypatch):
    fake = FakeIB(buy_status="Filled", buy_filled=8, buy_avg=75.4, held=8, fail_on_second_place=True)
    client = make_client(fake)
    calls = []
    client.resubmit_protective_orders = lambda t, q, tp, tg, same_day_guard=False: calls.append((t, q)) or {"ok": True}
    res = client.buy_with_bracket("PBF", 8, 9.0, 84.0, limit_price=75.56)
    assert res["buy"].status == "Error"  # caller still must not record a tracker row
    assert calls == [("PBF", 8)], "held shares must be auto-protected via the resubmit path"


def test_resubmit_preserves_high_water_mark(monkeypatch):
    """A resubmitted TRAIL must start from the old peak, not from the current price."""
    fake = FakeIB(buy_status="Filled", buy_filled=5, buy_avg=100.0)
    fake.openTrades = lambda: []
    fake.reqAllOpenOrders = lambda: None
    fake.portfolio = lambda: []
    client = make_client(fake)
    monkeypatch.setattr("core.trading.live_quote.get_exit_price", lambda t, f=None: (110.0, "test"))
    res = client.resubmit_protective_orders("PBF", 5, 5.5, 130.0, peak_price=115.0)
    trail = fake.placed[0]
    assert trail.orderType == "TRAIL" and trail.trailStopPrice == pytest.approx(115.0 * 0.945, abs=0.01)
    assert res["trailing_stop"].status != "Error"


def test_resubmit_skips_peak_stop_when_it_would_fire_immediately(monkeypatch):
    fake = FakeIB(buy_status="Filled", buy_filled=5, buy_avg=100.0)
    fake.openTrades = lambda: []
    fake.reqAllOpenOrders = lambda: None
    fake.portfolio = lambda: []
    client = make_client(fake)
    monkeypatch.setattr("core.trading.live_quote.get_exit_price", lambda t, f=None: (105.0, "test"))
    client.resubmit_protective_orders("PBF", 5, 5.5, 130.0, peak_price=115.0)  # init 108.7 > last
    assert fake.placed[0].trailStopPrice > 1e300, "left unset (IB UNSET_DOUBLE) -> IB starts fresh"


def test_resubmit_falls_back_to_plain_trail_if_ib_refuses_preset_stop(monkeypatch):
    fake = FakeIB(buy_status="Filled", buy_filled=5, buy_avg=100.0)
    fake.openTrades = lambda: []
    fake.reqAllOpenOrders = lambda: None
    fake.portfolio = lambda: []
    orig = fake.placeOrder

    def place(contract, order):
        t = orig(contract, order)
        if order.orderType == "TRAIL" and order.trailStopPrice < 1e300:
            t.orderStatus.status = "Inactive"      # IB refuses the preset stop price
        return t
    fake.placeOrder = place
    client = make_client(fake)
    monkeypatch.setattr("core.trading.live_quote.get_exit_price", lambda t, f=None: (110.0, "test"))
    res = client.resubmit_protective_orders("PBF", 5, 5.5, 130.0, peak_price=115.0)
    trails = [o for o in fake.placed if o.orderType == "TRAIL"]
    assert len(trails) == 2 and trails[1].trailStopPrice > 1e300
    assert res["trailing_stop"].status == "Submitted"
