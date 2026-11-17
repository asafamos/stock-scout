"""Sub-$2k target-date exit (2026-11-17): exit via the existing TRAIL, never cancel protection."""
from types import SimpleNamespace as NS

from scripts import monitor_positions as mp


class FakeClient:
    def __init__(self, status="Filled", price=12.4, boom=False):
        self.calls, self.status, self.price, self.boom = [], status, price, boom

    def force_exit_via_trail(self, ticker, aggressive=True):
        self.calls.append(("force", ticker))
        if self.boom:
            raise RuntimeError("ib down")
        return NS(status=self.status, filled_price=self.price if self.status == "Filled" else 0.0)

    # anything that would cancel/replace protection must NOT be called
    def _sell_market(self, *a, **k): raise AssertionError("standalone SELL must not be used")
    def resubmit_protective_orders_retry(self, *a, **k): raise AssertionError("no cancel/restore dance")


class FakeTracker:
    def __init__(self): self.removed = []
    def remove_position(self, t, p, r): self.removed.append((t, p, r))


class FakeNotify:
    def __init__(self): self.sells = []
    def notify_sell(self, *a): self.sells.append(a)


POS = {"entry_price": 13.16, "quantity": 1}


def test_filled_exit_is_recorded():
    c, t, n = FakeClient("Filled", 12.4), FakeTracker(), FakeNotify()
    assert mp._target_date_exit_via_trail("PATH", POS, c, t, n) is True
    assert t.removed == [("PATH", 12.4, "target_date_exit")] and len(n.sells) == 1


def test_pending_exit_keeps_position_and_does_not_alert():
    c, t, n = FakeClient("Working:Submitted"), FakeTracker(), FakeNotify()
    assert mp._target_date_exit_via_trail("PATH", POS, c, t, n) is False
    assert t.removed == [] and n.sells == []


def test_exception_leaves_protection_alone():
    c, t, n = FakeClient(boom=True), FakeTracker(), FakeNotify()
    assert mp._target_date_exit_via_trail("PATH", POS, c, t, n) is False
    assert t.removed == []
