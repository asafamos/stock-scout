"""Run the v2 sleeve once (volatility + size tilt, next-open protocol). Default OFF (TRADE_V2_SLEEVE).

    python -m scripts.run_v2_sleeve --dry-run [--parquet PATH]
    TRADE_LIVE_CONFIRMED=1 python -m scripts.run_v2_sleeve --parquet data/state/v2_scan.parquet   (systemd)

Same live-trading safety policy as run_auto_trade: real orders only with --live or systemd's
TRADE_LIVE_CONFIRMED=1; otherwise forced DRY.
"""
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("run_v2_sleeve")


def main() -> int:
    args = sys.argv[1:]
    parquet = None
    if "--parquet" in args:
        parquet = args[args.index("--parquet") + 1]
    if "--dry-run" in args:
        os.environ["TRADE_DRY_RUN"] = "1"
    elif "--live" in args:
        os.environ["TRADE_DRY_RUN"] = "0"
    elif os.getenv("TRADE_LIVE_CONFIRMED") != "1":
        os.environ["TRADE_DRY_RUN"] = "1"      # neither flag nor systemd authorisation -> DRY
    from core.trading.order_manager import OrderManager
    import pandas as pd
    df = pd.read_parquet(parquet) if parquet else None
    results = OrderManager().execute_v2_sleeve(df)
    for r in results:
        logger.info("v2 result: %s", r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
