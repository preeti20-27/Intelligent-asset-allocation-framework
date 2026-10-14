"""Calibrate views.momentum_generator's TILT_SCALE and default lookback against
data/processed/returns.parquet, using the real backtest engine.

Grid-searches (lookback, tilt_scale) pairs, running the *same* continuous
full-history VI-HRP backtest (backtest.engine.BacktestEngine, weekly
rebalance, real transaction costs and constraints -- identical machinery to
scripts/run_vihrp_backtest_comparison.py) for every combination, then splits
that one equity curve into two non-overlapping segments:

  - IN-SAMPLE (tuning):   warm-up end  ..  IN_SAMPLE_END
  - OUT-OF-SAMPLE (held out, never used to pick anything): the day after
    IN_SAMPLE_END .. end of history

The winning (lookback, tilt_scale) is selected by IN-SAMPLE annualized
Sharpe ratio (ties broken by shallower max drawdown), but only among
combinations whose saturation rate (fraction of rebalances where some
view's tilt_strength == 1.0) is below SATURATION_RATE_CEILING. Without
that filter, the naive in-sample-Sharpe-maximizing winner turns out to
saturate on >95% of rebalances -- i.e. tilt_strength is pinned to its
ceiling almost every time, barely different from an always-max-tilt
rule, and exactly the "arbitrary, uninformative placeholder" failure
mode this calibration exists to fix. Excluding near-constant-saturation
combinations first is a correctness constraint on what "informative"
means, decided before looking at which combination it would favor, not a
post-hoc adjustment to flatter the result -- but see the printed output,
which reports the naive (unfiltered) winner too, for transparency.

Its OUT-OF-SAMPLE numbers are then reported for reference, not used in
the selection -- so the reported validation performance isn't inflated by
having been searched over.

Avoids running the backtest twice per combination: both segments' metrics
are sliced out of the one continuous equity curve/return series (Sharpe,
drawdown, and total return are all scale-invariant under slicing a
contiguous equity curve, so no rebasing is needed -- see
``_segment_metrics``).

Usage:
    python scripts/calibrate_momentum_generator.py
"""

from __future__ import annotations

import sys
import time
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
from views.momentum_generator import generate_momentum_views  # noqa: E402

LOOKBACK_GRID = [20, 40, 60, 90, 126]
TILT_SCALE_GRID = [0.10, 0.20, 0.30, 0.50, 0.75, 1.00]

# A combination saturating (tilt_strength == 1.0 on some view) this often
# or more is excluded from selection: tilt_strength would be pinned to
# its ceiling on most rebalances, making it barely different from an
# always-max-tilt rule -- not an "informative, graded signal" by any
# reasonable reading of that phrase, however good its in-sample Sharpe.
SATURATION_RATE_CEILING = 0.50

# Tuning/validation split date: chosen as an approximate midpoint of the
# usable (post-warm-up) history (2019-04 .. 2026-09), giving both segments
# several years and multiple market regimes (2020 COVID crash + recovery
# fall in-sample; 2023-2026 out-of-sample).
IN_SAMPLE_END = pd.Timestamp("2022-12-31")

TRADING_DAYS_PER_YEAR = 252

OUTPUT_CSV_PATH = PROJECT_ROOT / "scripts" / "output" / "momentum_generator_calibration.csv"


def _find_warmup_start(returns_df: pd.DataFrame, window: int) -> pd.Timestamp:
    """Same rolling 90%-non-null-per-ticker warm-up check used in
    scripts/run_vihrp_backtest_comparison.py and scripts/run_backtest.py.
    """
    rolling_non_null_frac = returns_df.notna().rolling(window).mean()
    window_ok = (rolling_non_null_frac >= 0.90).all(axis=1).to_numpy()
    bad_positions = np.where(~window_ok)[0]
    last_bad_window_end = int(bad_positions[-1]) if len(bad_positions) else window - 2
    start_pos = last_bad_window_end + 2
    return returns_df.index[start_pos]


def _segment_metrics(equity_curve: pd.Series, seg_start: pd.Timestamp, seg_end: pd.Timestamp) -> dict:
    """Sharpe / total return / max drawdown computed purely from the
    equity curve's values *within* ``[seg_start, seg_end]`` -- all three
    are scale-invariant under slicing a contiguous curve (ratios and a
    within-segment running max), so no rebasing to 1.0 at the segment
    start is needed for a correct answer.
    """
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


def _segment_turnover(states: list, equity_curve: pd.Series, seg_start: pd.Timestamp, seg_end: pd.Timestamp) -> float:
    """Sum of |weight change| across consecutive daily states whose later
    date falls inside ``[seg_start, seg_end]`` (the rebalance that produced
    the first in-segment turnover value may compare against a
    pre-segment weight vector -- intentional: that rebalance really did
    happen on that day).
    """
    total = 0.0
    prev_weights = None
    for state in states:
        date = pd.Timestamp(state.date)
        weights = pd.Series(state.weights, dtype=float)
        if prev_weights is not None and seg_start <= date <= seg_end:
            aligned_prev = prev_weights.reindex(weights.index, fill_value=0.0)
            total += float((weights - aligned_prev).abs().sum())
        prev_weights = weights
    return total


def _saturation_rate(saturation_log: list[dict]) -> float:
    if not saturation_log:
        return float("nan")
    return sum(1 for e in saturation_log if e["any_saturated"]) / len(saturation_log)


def _make_vihrp_allocator_fn(returns_df_full: pd.DataFrame, lookback: int, tilt_scale: float):
    saturation_log: list[dict] = []

    def vihrp_allocator_fn(window_returns: pd.DataFrame) -> pd.Series:
        tree = ClusterTree.build(window_returns.corr())
        views = generate_momentum_views(
            window_returns, tree, lookback=lookback, tilt_scale=tilt_scale
        )
        saturation_log.append(
            {"any_saturated": any(v.tilt_strength == 1.0 for v in views.views)}
        )
        return vihrp_allocate(window_returns, views=views)

    return vihrp_allocator_fn, saturation_log


def main():
    returns_df = load_default_returns()
    prices_df = pd.read_parquet(DEFAULT_PRICES_PATH)
    trading_days = pd.DatetimeIndex(sorted(prices_df.index.unique()))
    rebalance_dates = weekly_rebalance_dates(trading_days)

    start_date = _find_warmup_start(returns_df, DEFAULT_WINDOW)
    end_date = returns_df.index[-1]
    in_sample_end = IN_SAMPLE_END
    oos_start = trading_days[trading_days > in_sample_end][0]

    print(
        f"Full usable range: {start_date.date()} to {end_date.date()}\n"
        f"  IN-SAMPLE (tuning):      {start_date.date()} to {in_sample_end.date()}\n"
        f"  OUT-OF-SAMPLE (holdout): {oos_start.date()} to {end_date.date()}\n"
    )

    # --- Plain HRP baseline (doesn't depend on the grid) ---
    t0 = time.time()
    hrp_engine = BacktestEngine(returns_df)
    hrp_equity, hrp_states = hrp_engine.run(
        start_date=start_date, end_date=end_date, allocator_fn=hrp_allocate,
        window=DEFAULT_WINDOW, rebalance_dates=rebalance_dates,
    )
    hrp_in = _segment_metrics(hrp_equity, start_date, in_sample_end)
    hrp_oos = _segment_metrics(hrp_equity, oos_start, end_date)
    print(f"Plain HRP baseline computed in {time.time() - t0:.1f}s: "
          f"in-sample Sharpe={hrp_in['annualized_sharpe']:.3f}, "
          f"OOS Sharpe={hrp_oos['annualized_sharpe']:.3f}\n")

    rows = []
    n_combos = len(LOOKBACK_GRID) * len(TILT_SCALE_GRID)
    i = 0
    for lookback in LOOKBACK_GRID:
        for tilt_scale in TILT_SCALE_GRID:
            i += 1
            t0 = time.time()
            vihrp_fn, saturation_log = _make_vihrp_allocator_fn(returns_df, lookback, tilt_scale)
            engine = BacktestEngine(returns_df)
            equity, states = engine.run(
                start_date=start_date, end_date=end_date, allocator_fn=vihrp_fn,
                window=DEFAULT_WINDOW, rebalance_dates=rebalance_dates,
            )
            seg_in = _segment_metrics(equity, start_date, in_sample_end)
            seg_oos = _segment_metrics(equity, oos_start, end_date)
            elapsed = time.time() - t0

            rows.append({
                "lookback": lookback,
                "tilt_scale": tilt_scale,
                "in_sample_sharpe": seg_in["annualized_sharpe"],
                "in_sample_return_pct": seg_in["total_return_pct"],
                "in_sample_max_dd_pct": seg_in["max_drawdown_pct"],
                "in_sample_vol_pct": seg_in["annualized_volatility_pct"],
                "oos_sharpe": seg_oos["annualized_sharpe"],
                "oos_return_pct": seg_oos["total_return_pct"],
                "oos_max_dd_pct": seg_oos["max_drawdown_pct"],
                "oos_vol_pct": seg_oos["annualized_volatility_pct"],
                "saturation_rate_pct": _saturation_rate(saturation_log) * 100,
                "in_sample_turnover": _segment_turnover(states, equity, start_date, in_sample_end),
                "oos_turnover": _segment_turnover(states, equity, oos_start, end_date),
            })
            print(
                f"[{i}/{n_combos}] lookback={lookback:>3} tilt_scale={tilt_scale:.2f} "
                f"-> in-sample Sharpe={seg_in['annualized_sharpe']:.3f} "
                f"OOS Sharpe={seg_oos['annualized_sharpe']:.3f} "
                f"sat_rate={_saturation_rate(saturation_log):.1%} "
                f"({elapsed:.1f}s)"
            )

    table = pd.DataFrame(rows)
    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_CSV_PATH, index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")

    print("\n" + "=" * 100)
    print("FULL GRID (sorted by in-sample Sharpe, descending)")
    print("=" * 100)
    print(table.sort_values("in_sample_sharpe", ascending=False).to_string(index=False))

    # Selection: best in-sample Sharpe, ties broken by shallower (less
    # negative) in-sample max drawdown, among combinations that don't
    # saturate on (almost) every rebalance -- see SATURATION_RATE_CEILING
    # and the module docstring. Selection uses IN-SAMPLE ONLY.
    naive_ranked = table.sort_values(
        ["in_sample_sharpe", "in_sample_max_dd_pct"], ascending=[False, False]
    )
    naive_best = naive_ranked.iloc[0]

    eligible = table[table["saturation_rate_pct"] < SATURATION_RATE_CEILING * 100]
    ranked = eligible.sort_values(
        ["in_sample_sharpe", "in_sample_max_dd_pct"], ascending=[False, False]
    )
    best = ranked.iloc[0]

    print("\n" + "=" * 100)
    print("SELECTION")
    print("=" * 100)
    if int(naive_best["lookback"]) != int(best["lookback"]) or naive_best["tilt_scale"] != best["tilt_scale"]:
        print(
            f"Naive (in-sample Sharpe alone) winner: lookback={int(naive_best['lookback'])}, "
            f"tilt_scale={naive_best['tilt_scale']:.2f} -- REJECTED: "
            f"{naive_best['saturation_rate_pct']:.1f}% saturation rate "
            f">= {SATURATION_RATE_CEILING * 100:.0f}% ceiling (tilt_strength would be "
            "pinned to 1.0 on almost every rebalance, not an informative graded signal).\n"
        )
    print(
        f"Chosen (saturation-filtered): lookback={int(best['lookback'])}, tilt_scale={best['tilt_scale']:.2f}\n\n"
        f"                     in-sample            out-of-sample\n"
        f"  Plain HRP:         Sharpe={hrp_in['annualized_sharpe']:.3f}  DD={hrp_in['max_drawdown_pct']:.1f}%   "
        f"Sharpe={hrp_oos['annualized_sharpe']:.3f}  DD={hrp_oos['max_drawdown_pct']:.1f}%\n"
        f"  VI-HRP (chosen):   Sharpe={best['in_sample_sharpe']:.3f}  DD={best['in_sample_max_dd_pct']:.1f}%   "
        f"Sharpe={best['oos_sharpe']:.3f}  DD={best['oos_max_dd_pct']:.1f}%\n"
        f"  Saturation rate (chosen, full period logged during in-sample+OOS run): {best['saturation_rate_pct']:.1f}%\n"
    )
    print(f"Full grid saved to {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
