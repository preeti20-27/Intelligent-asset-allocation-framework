"""End-to-end backtest runner.

Loads the real processed data, runs backtest.engine.BacktestEngine over
the full available date range using a chosen allocator, prints a
performance summary, saves an equity-curve plot, and confirms
monitor.monitor.check_portfolio can process every day's PortfolioState
without crashing (counting how often each action fires).

Usage:
    python scripts/run_backtest.py --allocator equal_weight
    python scripts/run_backtest.py --allocator hrp
    python scripts/run_backtest.py --allocator hrp --window 126

The only thing that changes between allocators is which allocate()
function gets passed to BacktestEngine.run() -- everything else (data,
date range, window, rebalance calendar, constraints, costs, monitor
checks) is identical, so the two runs are a fair comparison.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backtest.calendar import DEFAULT_PRICES_PATH, weekly_rebalance_dates  # noqa: E402
from backtest.engine import BacktestEngine, DEFAULT_WINDOW, load_default_returns  # noqa: E402
from eval.metrics import performance_summary  # noqa: E402
from eval.plots import plot_equity_curve  # noqa: E402
from monitor.monitor import check_portfolio  # noqa: E402
from monitor.rules import HOLD, WARN_CONCENTRATION, WARN_DRAWDOWN  # noqa: E402

ALLOCATOR_MODULES = {
    "equal_weight": "allocators.equal_weight",
    "hrp": "allocators.hrp",
}


def load_allocator(name: str):
    """Import allocate() from allocators.<name> without hardcoding which one."""
    module = importlib.import_module(ALLOCATOR_MODULES[name])
    return module.allocate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allocator",
        choices=sorted(ALLOCATOR_MODULES),
        required=True,
        help="Which allocators.<name>.allocate() to run.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=DEFAULT_WINDOW,
        help=f"Rolling lookback window fed to the allocator (default {DEFAULT_WINDOW}).",
    )
    args = parser.parse_args()

    allocator_fn = load_allocator(args.allocator)

    prices_df = pd.read_parquet(DEFAULT_PRICES_PATH)
    returns_df = load_default_returns()

    trading_days = pd.DatetimeIndex(sorted(prices_df.index.unique()))
    rebalance_dates = weekly_rebalance_dates(trading_days)

    # Warm-up: find the first date whose trailing `window`-day window
    # satisfies HRP's own >=90%-non-null-per-ticker requirement (the same
    # check allocators.hrp.allocate performs itself). This is NOT simply
    # "skip `window` rows" or "skip past the newest ticker's inception
    # date": LTGILTBEES doesn't just start partway through the panel
    # (first valid return 2016-07-25) -- for years afterward it also
    # trades sparsely, with scattered multi-week gaps of missing data
    # (illiquidity, not corporate actions), so a rolling 252-day window
    # can still be mostly-NaN for LTGILTBEES well into 2019. Checking the
    # rolling non-null fraction directly, rather than assuming inception
    # date + window is enough, is what actually finds a safe start.
    # Applied identically regardless of allocator (even though
    # equal_weight itself needs no history), so the --allocator flag is
    # the only thing that differs between runs.
    rolling_non_null_frac = returns_df.notna().rolling(args.window).mean()
    window_ok = (rolling_non_null_frac >= 0.90).all(axis=1).to_numpy()
    bad_positions = np.where(~window_ok)[0]
    # A window ending at position p is bad if p is in bad_positions (or if
    # the rolling window isn't even full yet, i.e. p < window - 1, which
    # rolling() already reports as NaN -> not >= 0.90 -> already excluded
    # by window_ok). The run's start_date at position s uses the window
    # ending at s - 1, so the first safe start_date is one position past
    # the last bad window-ending position.
    last_bad_window_end = int(bad_positions[-1]) if len(bad_positions) else args.window - 2
    start_pos = last_bad_window_end + 2
    if start_pos >= len(returns_df):
        raise ValueError(
            f"window={args.window}: no date in the panel has a fully-populated, "
            "90%-non-null trailing window for every ticker before the panel ends."
        )
    start_date = returns_df.index[start_pos]
    end_date = returns_df.index[-1]
    print(
        f"Warm-up: skipping to {start_date.date()} (of {returns_df.index[0].date()} "
        f"to {end_date.date()} available) so every ticker's trailing {args.window}-day "
        f"window is >=90% non-null -- see comments in this script for why this isn't "
        f"just 'skip the first {args.window} rows'."
    )

    engine = BacktestEngine(returns_df)
    equity_curve, states = engine.run(
        start_date=start_date,
        end_date=end_date,
        allocator_fn=allocator_fn,
        window=args.window,
        rebalance_dates=rebalance_dates,
    )

    daily_returns = equity_curve.pct_change().dropna()
    summary = performance_summary(equity_curve, daily_returns)

    print(f"=== Backtest: {args.allocator} (window={args.window}) ===")
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print(f"Trading days: {len(states)}")
    print()
    print(summary["summary"])
    print()

    # Confirm the monitor can process every day's PortfolioState without
    # crashing, and tally how often each action fires. Checked on every
    # trading day's state (not just weekly rebalance dates) -- a stricter
    # crash test, and a more complete action count, than checking weekly.
    action_counts = Counter(check_portfolio(state) for state in states)

    print(f"Monitor checked on all {len(states)} daily PortfolioStates without crashing.")
    print("Monitor action counts:")
    for action in (HOLD, WARN_DRAWDOWN, WARN_CONCENTRATION):
        print(f"  {action}: {action_counts.get(action, 0)}")

    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    plot_path = output_dir / f"equity_curve_{args.allocator}.png"
    plot_equity_curve(
        equity_curve,
        title=f"Equity Curve -- {args.allocator} (window={args.window})",
        save_path=str(plot_path),
    )
    print(f"\nEquity curve saved to {plot_path}")


if __name__ == "__main__":
    main()
