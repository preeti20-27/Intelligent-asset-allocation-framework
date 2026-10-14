"""Calibrate views.standardized_momentum_reversal_generator's TILT_SCALE and
REVERSAL_FORMATION_DAYS against data/processed/returns.parquet, using the
real backtest engine -- same methodology as
scripts/calibrate_momentum_generator.py (read that script's docstring
first; this one reuses its approach line for line, just against a
different generator's parameters).

Grid-searches (reversal_formation_days, tilt_scale) pairs, running the
*same* continuous full-history VI-HRP backtest (backtest.engine.BacktestEngine,
weekly rebalance, real transaction costs and constraints -- identical
machinery to scripts/backtest_standardized_generator.py) for every
combination, then splits that one equity curve into the SAME two
non-overlapping segments scripts/calibrate_momentum_generator.py uses:

  - IN-SAMPLE (tuning):   warm-up end  ..  IN_SAMPLE_END (2022-12-31)
  - OUT-OF-SAMPLE (held out, never used to pick anything): the day after
    IN_SAMPLE_END .. end of history

``momentum_formation_days``/``momentum_skip_days`` are held fixed at the
module's defaults (231/21, the standard 12-month-minus-1-month
convention) rather than grid-searched -- that convention comes from the
momentum literature itself (Jegadeesh-Titman, reused by AMP), not a free
parameter tuned to this dataset, unlike ``reversal_formation_days``
(already documented in the generator's own module docstring as an
adaptation forced by this dataset's ~11.6-year history and LTGILTBEES's
2016-07-25 listing, not a value chosen freely from AMP's 3-5yr range --
exactly the kind of choice this calibration should actually search over).
Leg-combination weighting (currently a fixed 50/50 average of the two
z-scores) is NOT grid-searched here either -- flagged explicitly as a
remaining gap in this script's own printed output and the generator's
module docstring, not silently skipped.

Selection rule -- IDENTICAL to scripts/calibrate_momentum_generator.py's:
best IN-SAMPLE annualized Sharpe ratio (ties broken by shallower max
drawdown), but only among combinations whose saturation rate (fraction
of rebalances where some view's tilt_strength == 1.0) is below
SATURATION_RATE_CEILING (50%) -- excluding near-constant-saturation
combinations BEFORE ranking by Sharpe, not after seeing which one it
would favor, for the exact reason scripts/calibrate_momentum_generator.py
excludes them: a saturating tilt_strength is pinned to its ceiling on
most rebalances, barely different from an always-max-tilt rule, not an
informative graded signal regardless of how good its in-sample Sharpe
looks.

Its OUT-OF-SAMPLE numbers are then reported for reference, not used in
the selection -- so the reported validation performance isn't inflated by
having been searched over.

Note on reproducibility: BLAS thread count is pinned to 1 (see
scripts/diagnose_regime.py's docstring for the full explanation of why
hierarchical clustering's floating-point tie-breaking makes this
necessary for run-to-run reproducibility of exact decimals; directional
conclusions are unaffected).

Usage:
    python scripts/calibrate_standardized_generator.py
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

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
from views.standardized_momentum_reversal_generator import (  # noqa: E402
    MOMENTUM_FORMATION_DAYS,
    MOMENTUM_SKIP_DAYS,
    generate_standardized_momentum_reversal_views,
)

REVERSAL_FORMATION_DAYS_GRID = [504, 756, 1008]  # 2yr, 3yr (current default), 4yr
TILT_SCALE_GRID = [0.30, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00]

# Same ceiling, same rationale, as scripts/calibrate_momentum_generator.py.
SATURATION_RATE_CEILING = 0.50

# Same tuning/validation split as scripts/calibrate_momentum_generator.py
# and scripts/diagnose_regime.py, for direct comparability.
IN_SAMPLE_END = pd.Timestamp("2022-12-31")

TRADING_DAYS_PER_YEAR = 252
HISTORY_BUFFER_DAYS = 50  # extra rows beyond reversal_formation_days handed to the generator

OUTPUT_CSV_PATH = PROJECT_ROOT / "scripts" / "output" / "standardized_generator_calibration.csv"


def _find_warmup_start(returns_df: pd.DataFrame, window: int) -> pd.Timestamp:
    """Same rolling 90%-non-null-per-ticker warm-up check used in every
    other backtest/calibration script in this project.
    """
    rolling_non_null_frac = returns_df.notna().rolling(window).mean()
    window_ok = (rolling_non_null_frac >= 0.90).all(axis=1).to_numpy()
    bad_positions = np.where(~window_ok)[0]
    last_bad_window_end = int(bad_positions[-1]) if len(bad_positions) else window - 2
    start_pos = last_bad_window_end + 2
    return returns_df.index[start_pos]


def _segment_metrics(equity_curve: pd.Series, seg_start: pd.Timestamp, seg_end: pd.Timestamp) -> dict:
    """Scale-invariant under slicing a contiguous equity curve -- see
    scripts/calibrate_momentum_generator.py's identical helper.
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


def _saturation_rate(saturation_log: list[dict]) -> float:
    if not saturation_log:
        return float("nan")
    return sum(1 for e in saturation_log if e["any_saturated"]) / len(saturation_log)


def _make_standardized_allocator_fn(returns_df_full: pd.DataFrame, reversal_formation_days: int, tilt_scale: float):
    needed_history = reversal_formation_days + HISTORY_BUFFER_DAYS
    saturation_log: list[dict] = []

    def allocator_fn(window_returns: pd.DataFrame) -> pd.Series:
        tree = ClusterTree.build(window_returns.corr())
        last_date = window_returns.index[-1]
        long_history = returns_df_full.loc[:last_date].tail(needed_history)
        views = generate_standardized_momentum_reversal_views(
            long_history,
            tree,
            momentum_formation_days=MOMENTUM_FORMATION_DAYS,
            momentum_skip_days=MOMENTUM_SKIP_DAYS,
            reversal_formation_days=reversal_formation_days,
            tilt_scale=tilt_scale,
        )
        saturation_log.append(
            {"any_saturated": any(v.tilt_strength == 1.0 for v in views.views)}
        )
        return vihrp_allocate(window_returns, views=views)

    return allocator_fn, saturation_log


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
        f"  momentum_formation_days={MOMENTUM_FORMATION_DAYS}, momentum_skip_days={MOMENTUM_SKIP_DAYS} "
        f"(fixed, NOT grid-searched -- literature convention, see module docstring)\n"
        f"  NOTE: leg-combination weighting (50/50 average) is also NOT grid-searched here.\n"
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
    n_combos = len(REVERSAL_FORMATION_DAYS_GRID) * len(TILT_SCALE_GRID)
    i = 0
    for reversal_formation_days in REVERSAL_FORMATION_DAYS_GRID:
        for tilt_scale in TILT_SCALE_GRID:
            i += 1
            t0 = time.time()
            allocator_fn, saturation_log = _make_standardized_allocator_fn(
                returns_df, reversal_formation_days, tilt_scale
            )
            engine = BacktestEngine(returns_df)
            equity, states = engine.run(
                start_date=start_date, end_date=end_date, allocator_fn=allocator_fn,
                window=DEFAULT_WINDOW, rebalance_dates=rebalance_dates,
            )
            seg_in = _segment_metrics(equity, start_date, in_sample_end)
            seg_oos = _segment_metrics(equity, oos_start, end_date)
            elapsed = time.time() - t0

            rows.append({
                "reversal_formation_days": reversal_formation_days,
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
            })
            print(
                f"[{i}/{n_combos}] reversal_days={reversal_formation_days:>4} tilt_scale={tilt_scale:.2f} "
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

    # Selection: IDENTICAL rule to scripts/calibrate_momentum_generator.py.
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
    if (
        int(naive_best["reversal_formation_days"]) != int(best["reversal_formation_days"])
        or naive_best["tilt_scale"] != best["tilt_scale"]
    ):
        print(
            f"Naive (in-sample Sharpe alone) winner: reversal_days={int(naive_best['reversal_formation_days'])}, "
            f"tilt_scale={naive_best['tilt_scale']:.2f} -- REJECTED: "
            f"{naive_best['saturation_rate_pct']:.1f}% saturation rate "
            f">= {SATURATION_RATE_CEILING * 100:.0f}% ceiling (tilt_strength would be "
            "pinned to 1.0 on almost every rebalance, not an informative graded signal).\n"
        )
    else:
        print("Naive (in-sample Sharpe alone) winner matches the saturation-filtered winner -- no rejection needed.\n")

    print(
        f"Chosen (saturation-filtered): reversal_formation_days={int(best['reversal_formation_days'])}, "
        f"tilt_scale={best['tilt_scale']:.2f}\n\n"
        f"                     in-sample            out-of-sample\n"
        f"  Plain HRP:         Sharpe={hrp_in['annualized_sharpe']:.3f}  DD={hrp_in['max_drawdown_pct']:.1f}%   "
        f"Sharpe={hrp_oos['annualized_sharpe']:.3f}  DD={hrp_oos['max_drawdown_pct']:.1f}%\n"
        f"  Standardized (chosen): Sharpe={best['in_sample_sharpe']:.3f}  DD={best['in_sample_max_dd_pct']:.1f}%   "
        f"Sharpe={best['oos_sharpe']:.3f}  DD={best['oos_max_dd_pct']:.1f}%\n"
        f"  Saturation rate (chosen): {best['saturation_rate_pct']:.1f}%\n"
    )
    print(f"Full grid saved to {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
