"""Regime diagnosis: why did the calibrated momentum view generator
(views/momentum_generator.py, lookback=40, tilt_scale=0.20 --
scripts/calibrate_momentum_generator.py) underperform plain HRP
out-of-sample (2023-01-02 to 2026-09-11), after beating it in-sample
(2019-04-05 to 2022-12-31)?

Two things, both read-only / analysis-only -- no production code changed:

1. Characterizes the IN-SAMPLE and OOS windows side by side on: realized
   volatility, return autocorrelation, the momentum signal's own
   horizon-40-day persistence (does a cluster's trailing 40-day return
   predict its *next* 40-day return, positively or negatively?),
   cross-sectional return dispersion, plain-HRP drawdown, and plain-HRP's
   own rebalance-to-rebalance turnover (a stability proxy for the
   HRP-implied risk allocation itself, independent of any view).
2. Splits the IN-SAMPLE window into two independent halves (2019-04-05 to
   2021-02-26 and 2021-03-01 to 2022-12-31, per the task's suggested
   split) and reports plain HRP vs. VI-HRP+momentum on each separately,
   to see whether momentum's in-sample edge is consistent across regimes
   or concentrated in one sub-period.

Both backtests (plain HRP, VI-HRP+momentum) are run ONCE each over the
full warm-up-end..end-of-history range; every segment's metrics (subA,
subB, full in-sample, OOS) are sliced out of those two continuous equity
curves -- Sharpe/return/drawdown are scale-invariant under slicing (see
scripts/calibrate_momentum_generator.py's _segment_metrics), so this
doesn't need four separate backtest runs.

Note on reproducibility: BLAS thread count is pinned to 1 below (set
before numpy/pandas/scipy import) because the hierarchical-clustering
tree (tree.cluster_tree.ClusterTree, via scipy's single-linkage) makes
strict floating-point comparisons to decide merge order; with the
correlation matrix computed by a multi-threaded BLAS, run-to-run
floating-point summation-order jitter (~1e-15) can occasionally flip
which of two near-tied distances is "smaller," changing the tree shape
at that one rebalance and cascading into materially different exact
Sharpe/return numbers across repeated runs of the *same* script on the
*same* data (observed directly: two runs of
scripts/calibrate_momentum_generator.py produced in-sample Sharpe 0.6685
vs. 0.6695 and OOS Sharpe 1.1697 vs. 1.1249 for the identical
(lookback=40, tilt_scale=0.20) cell). Pinning threads here makes this
script's own numbers reproducible; it does not retroactively fix that
jitter in already-committed calibration output, which is why this
script's diagnosis leans on directional/qualitative conclusions
(which direction, how large, consistent across sub-windows or not)
rather than treating any single decimal as exact.

Usage:
    python scripts/diagnose_regime.py
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from allocators.hrp import allocate as hrp_allocate  # noqa: E402
from allocators.vihrp import allocate as vihrp_allocate  # noqa: E402
from backtest.calendar import DEFAULT_PRICES_PATH, weekly_rebalance_dates  # noqa: E402
from backtest.engine import DEFAULT_WINDOW, BacktestEngine, load_default_returns  # noqa: E402
from eval.metrics import max_drawdown, sharpe_ratio, total_return  # noqa: E402
from tree.cluster_tree import ClusterTree  # noqa: E402
from views.momentum_generator import LOOKBACK, TILT_SCALE, generate_momentum_views  # noqa: E402

TRADING_DAYS_PER_YEAR = 252
SIGNAL_HORIZON = LOOKBACK  # the momentum generator's own calibrated lookback (40 trading days)

IN_SAMPLE_END = pd.Timestamp("2022-12-31")
SUB_SPLIT = pd.Timestamp("2021-02-28")  # subA: start..2021-02-26ish, subB: 2021-03-01ish..in_sample_end

OUTPUT_CSV_PATH = PROJECT_ROOT / "scripts" / "output" / "regime_diagnosis.csv"


def _find_warmup_start(returns_df: pd.DataFrame, window: int) -> pd.Timestamp:
    rolling_non_null_frac = returns_df.notna().rolling(window).mean()
    window_ok = (rolling_non_null_frac >= 0.90).all(axis=1).to_numpy()
    bad_positions = np.where(~window_ok)[0]
    last_bad_window_end = int(bad_positions[-1]) if len(bad_positions) else window - 2
    start_pos = last_bad_window_end + 2
    return returns_df.index[start_pos]


def _segment_metrics(equity_curve: pd.Series, seg_start: pd.Timestamp, seg_end: pd.Timestamp) -> dict:
    seg = equity_curve.loc[seg_start:seg_end]
    daily_returns = seg.pct_change().dropna()
    dd = max_drawdown(seg)
    return {
        "n_days": len(seg),
        "total_return_pct": total_return(seg) * 100,
        "annualized_sharpe": sharpe_ratio(daily_returns),
        "annualized_volatility_pct": float(daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)) * 100,
        "max_drawdown_pct": dd.max_drawdown * 100,
    }


def _segment_turnover(states: list, seg_start: pd.Timestamp, seg_end: pd.Timestamp) -> tuple[float, int]:
    """(total turnover, number of rebalances that actually changed weights) within the segment."""
    total = 0.0
    n_rebalances = 0
    prev_weights = None
    for state in states:
        date = pd.Timestamp(state.date)
        weights = pd.Series(state.weights, dtype=float)
        if prev_weights is not None and seg_start <= date <= seg_end:
            aligned_prev = prev_weights.reindex(weights.index, fill_value=0.0)
            change = float((weights - aligned_prev).abs().sum())
            total += change
            if change > 1e-9:
                n_rebalances += 1
        prev_weights = weights
    return total, n_rebalances


def _make_vihrp_allocator_fn(returns_df_full: pd.DataFrame, lookback: int, tilt_scale: float):
    def vihrp_allocator_fn(window_returns: pd.DataFrame) -> pd.Series:
        tree = ClusterTree.build(window_returns.corr())
        views = generate_momentum_views(window_returns, tree, lookback=lookback, tilt_scale=tilt_scale)
        return vihrp_allocate(window_returns, views=views)

    return vihrp_allocator_fn


def _regime_stats(returns_df: pd.DataFrame, seg_start: pd.Timestamp, seg_end: pd.Timestamp) -> dict:
    """Pure data diagnostics -- no backtest, no allocator -- for one segment."""
    seg = returns_df.loc[seg_start:seg_end]

    ann_vol_per_asset = seg.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    lag1_autocorr_per_asset = seg.apply(lambda col: col.dropna().autocorr(lag=1))

    cross_sectional_daily_dispersion = seg.std(axis=1, skipna=True).mean()

    # Signal persistence: does a cluster/asset's trailing SIGNAL_HORIZON-day
    # compounded return predict its OWN NEXT SIGNAL_HORIZON-day compounded
    # return? Pooled correlation across assets and time within the segment.
    # Uses the FULL returns_df (not just `seg`) so the rolling window can
    # see data just before seg_start -- otherwise the first SIGNAL_HORIZON
    # rows of every segment would be dropped for no real reason.
    full_compounded = (1.0 + returns_df).rolling(SIGNAL_HORIZON).apply(lambda x: np.prod(x) - 1.0, raw=True)
    current = full_compounded.loc[seg_start:seg_end]
    future = full_compounded.shift(-SIGNAL_HORIZON).loc[seg_start:seg_end]
    pooled = pd.DataFrame({"current": current.stack(), "future": future.stack()}).dropna()
    signal_persistence_corr = pooled["current"].corr(pooled["future"]) if len(pooled) >= 2 else float("nan")

    cross_sectional_signal_dispersion = current.std(axis=1, skipna=True).mean()

    return {
        "n_days": len(seg),
        "mean_annualized_vol_pct": float(ann_vol_per_asset.mean()) * 100,
        "median_annualized_vol_pct": float(ann_vol_per_asset.median()) * 100,
        "mean_lag1_autocorr": float(lag1_autocorr_per_asset.mean()),
        "cross_sectional_daily_dispersion_pct": float(cross_sectional_daily_dispersion) * 100,
        "cross_sectional_signal_dispersion_pct": float(cross_sectional_signal_dispersion) * 100,
        "signal_persistence_corr_n_obs": len(pooled),
        f"signal_persistence_corr_{SIGNAL_HORIZON}d": float(signal_persistence_corr),
    }


def main():
    returns_df = load_default_returns()
    prices_df = pd.read_parquet(DEFAULT_PRICES_PATH)
    trading_days = pd.DatetimeIndex(sorted(prices_df.index.unique()))
    rebalance_dates = weekly_rebalance_dates(trading_days)

    start_date = _find_warmup_start(returns_df, DEFAULT_WINDOW)
    end_date = returns_df.index[-1]
    in_sample_end = IN_SAMPLE_END
    oos_start = trading_days[trading_days > in_sample_end][0]
    sub_a_end = trading_days[trading_days <= SUB_SPLIT][-1]
    sub_b_start = trading_days[trading_days > SUB_SPLIT][0]

    segments = {
        "subA_2019-04_to_2021-02": (start_date, sub_a_end),
        "subB_2021-03_to_2022-12": (sub_b_start, in_sample_end),
        "full_in_sample": (start_date, in_sample_end),
        "OOS_2023-01_to_2026-09": (oos_start, end_date),
    }

    print("=" * 100)
    print("SEGMENT DATE RANGES")
    print("=" * 100)
    for name, (s, e) in segments.items():
        print(f"  {name:28s}: {s.date()} to {e.date()}")
    print()

    # --- Part 1: pure regime diagnostics (no backtest) ---
    regime_rows = []
    for name, (s, e) in segments.items():
        row = {"segment": name}
        row.update(_regime_stats(returns_df, s, e))
        regime_rows.append(row)
    regime_table = pd.DataFrame(regime_rows).set_index("segment")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")

    print("=" * 100)
    print("PART 1: REGIME DIAGNOSTICS (pure data, no allocator)")
    print("=" * 100)
    print(regime_table.to_string())
    print(
        "\nReading guide: mean_lag1_autocorr > 0 and signal_persistence_corr > 0 both mean "
        "'recent winners keep winning' (momentum-friendly); < 0 means 'recent winners reverse' "
        "(mean-reversion-friendly); near 0 means neither direction has real predictive power over "
        "that horizon (an efficient-ish, hard-to-time regime).\n"
    )

    # --- Part 2: backtests (one continuous run each, sliced per segment) ---
    hrp_engine = BacktestEngine(returns_df)
    hrp_equity, hrp_states = hrp_engine.run(
        start_date=start_date, end_date=end_date, allocator_fn=hrp_allocate,
        window=DEFAULT_WINDOW, rebalance_dates=rebalance_dates,
    )

    vihrp_fn = _make_vihrp_allocator_fn(returns_df, LOOKBACK, TILT_SCALE)
    vihrp_engine = BacktestEngine(returns_df)
    vihrp_equity, vihrp_states = vihrp_engine.run(
        start_date=start_date, end_date=end_date, allocator_fn=vihrp_fn,
        window=DEFAULT_WINDOW, rebalance_dates=rebalance_dates,
    )

    backtest_rows = []
    for name, (s, e) in segments.items():
        hrp_m = _segment_metrics(hrp_equity, s, e)
        vihrp_m = _segment_metrics(vihrp_equity, s, e)
        hrp_turnover, hrp_n_rebal = _segment_turnover(hrp_states, s, e)
        backtest_rows.append({
            "segment": name,
            "hrp_sharpe": hrp_m["annualized_sharpe"],
            "hrp_return_pct": hrp_m["total_return_pct"],
            "hrp_max_dd_pct": hrp_m["max_drawdown_pct"],
            "hrp_turnover": hrp_turnover,
            "hrp_n_active_rebalances": hrp_n_rebal,
            "vihrp_momentum_sharpe": vihrp_m["annualized_sharpe"],
            "vihrp_momentum_return_pct": vihrp_m["total_return_pct"],
            "vihrp_momentum_max_dd_pct": vihrp_m["max_drawdown_pct"],
            "momentum_beats_hrp_sharpe": vihrp_m["annualized_sharpe"] > hrp_m["annualized_sharpe"],
        })
    backtest_table = pd.DataFrame(backtest_rows).set_index("segment")

    print("=" * 100)
    print("PART 2: PLAIN HRP vs. VI-HRP+MOMENTUM (lookback=40, tilt_scale=0.20), per segment")
    print("=" * 100)
    print(backtest_table.to_string())

    n_beats = int(backtest_table["momentum_beats_hrp_sharpe"].sum())
    print(f"\nMomentum beats plain HRP on Sharpe in {n_beats} of {len(backtest_table)} segments.")

    combined = regime_table.join(backtest_table)
    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_CSV_PATH)
    print(f"\nFull combined table saved to {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
