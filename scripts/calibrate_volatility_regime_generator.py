"""Calibrate views.volatility_regime_generator's TILT_SCALE and
SHORT_VOL_WINDOW against data/processed/returns.parquet, using the real
backtest engine -- same methodology as
scripts/calibrate_momentum_generator.py (read that script's docstring
first; this one reuses its approach against a different generator's
parameters).

Grid-searches (short_vol_window, tilt_scale) pairs, running the *same*
continuous full-history VI-HRP backtest (backtest.engine.BacktestEngine,
weekly rebalance, real transaction costs and constraints) for every
combination, then splits that one equity curve into the SAME two
non-overlapping segments every calibration script in this project uses:

  - IN-SAMPLE (tuning):   warm-up end  ..  IN_SAMPLE_END (2022-12-31)
  - OUT-OF-SAMPLE (held out, never used to pick anything): the day after
    IN_SAMPLE_END .. end of history

``long_vol_window`` is held fixed at the module's default (180 days, ~9
months) rather than grid-searched -- it was chosen specifically so
``short_vol_window + long_vol_window`` fits inside
``backtest.engine.BacktestEngine``'s default 252-row window across the
whole ``short_vol_window`` grid tested here; grid-searching it too would
risk picking a value that no longer fits, reintroducing the
extra-history-feeding complexity ``views.volatility_regime_generator``
was deliberately designed to avoid (see its module docstring).

Selection rule -- IDENTICAL to scripts/calibrate_momentum_generator.py's:
best IN-SAMPLE annualized Sharpe ratio (ties broken by shallower max
drawdown), but only among combinations whose saturation rate (fraction
of rebalances where some view's tilt_strength == 1.0) is below
SATURATION_RATE_CEILING (50%).

Its OUT-OF-SAMPLE numbers are then reported for reference, not used in
the selection.

Special attention to subA (2019-04-05 to 2021-02-26, containing the 2020
COVID crash): this generator is specifically designed to react to that
kind of volatility regime, so its in-sample Sharpe/drawdown there is
reported separately alongside the standard IN_SAMPLE/OOS split, even
though selection itself still uses the full in-sample window (not subA
alone) to avoid overfitting to one crisis.

Note on reproducibility: BLAS thread count is pinned to 1 (see
scripts/diagnose_regime.py's docstring for the full explanation).

Usage:
    python scripts/calibrate_volatility_regime_generator.py
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
from views.volatility_regime_generator import LONG_VOL_WINDOW, generate_volatility_regime_views  # noqa: E402

SHORT_VOL_WINDOW_GRID = [10, 20, 40]
TILT_SCALE_GRID = [0.30, 0.50, 0.75, 1.00, 1.50, 2.00]

SATURATION_RATE_CEILING = 0.50

IN_SAMPLE_END = pd.Timestamp("2022-12-31")
SUB_SPLIT = pd.Timestamp("2021-02-28")

TRADING_DAYS_PER_YEAR = 252

OUTPUT_CSV_PATH = PROJECT_ROOT / "scripts" / "output" / "volatility_regime_calibration.csv"


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
        "annualized_sharpe": sharpe_ratio(daily_returns),
        "total_return_pct": total_return(seg) * 100,
        "max_drawdown_pct": dd.max_drawdown * 100,
        "annualized_volatility_pct": float(daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)) * 100,
    }


def _saturation_rate(saturation_log: list[dict]) -> float:
    if not saturation_log:
        return float("nan")
    return sum(1 for e in saturation_log if e["any_saturated"]) / len(saturation_log)


def _make_allocator_fn(short_vol_window: int, tilt_scale: float):
    saturation_log: list[dict] = []

    def allocator_fn(window_returns: pd.DataFrame) -> pd.Series:
        tree = ClusterTree.build(window_returns.corr())
        views = generate_volatility_regime_views(
            window_returns, tree,
            short_vol_window=short_vol_window, long_vol_window=LONG_VOL_WINDOW, tilt_scale=tilt_scale,
        )
        saturation_log.append({"any_saturated": any(v.tilt_strength == 1.0 for v in views.views)})
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
    sub_a_end = trading_days[trading_days <= SUB_SPLIT][-1]

    print(
        f"Full usable range: {start_date.date()} to {end_date.date()}\n"
        f"  subA (COVID, reported separately): {start_date.date()} to {sub_a_end.date()}\n"
        f"  IN-SAMPLE (tuning):      {start_date.date()} to {in_sample_end.date()}\n"
        f"  OUT-OF-SAMPLE (holdout): {oos_start.date()} to {end_date.date()}\n"
        f"  long_vol_window={LONG_VOL_WINDOW} (fixed, NOT grid-searched -- see module docstring)\n"
    )

    t0 = time.time()
    hrp_engine = BacktestEngine(returns_df)
    hrp_equity, _hrp_states = hrp_engine.run(
        start_date=start_date, end_date=end_date, allocator_fn=hrp_allocate,
        window=DEFAULT_WINDOW, rebalance_dates=rebalance_dates,
    )
    hrp_in = _segment_metrics(hrp_equity, start_date, in_sample_end)
    hrp_oos = _segment_metrics(hrp_equity, oos_start, end_date)
    hrp_suba = _segment_metrics(hrp_equity, start_date, sub_a_end)
    print(f"Plain HRP baseline computed in {time.time() - t0:.1f}s: "
          f"in-sample Sharpe={hrp_in['annualized_sharpe']:.3f}, "
          f"OOS Sharpe={hrp_oos['annualized_sharpe']:.3f}, "
          f"subA Sharpe={hrp_suba['annualized_sharpe']:.3f}\n")

    rows = []
    n_combos = len(SHORT_VOL_WINDOW_GRID) * len(TILT_SCALE_GRID)
    i = 0
    for short_vol_window in SHORT_VOL_WINDOW_GRID:
        for tilt_scale in TILT_SCALE_GRID:
            i += 1
            t0 = time.time()
            allocator_fn, saturation_log = _make_allocator_fn(short_vol_window, tilt_scale)
            engine = BacktestEngine(returns_df)
            equity, _states = engine.run(
                start_date=start_date, end_date=end_date, allocator_fn=allocator_fn,
                window=DEFAULT_WINDOW, rebalance_dates=rebalance_dates,
            )
            seg_in = _segment_metrics(equity, start_date, in_sample_end)
            seg_oos = _segment_metrics(equity, oos_start, end_date)
            seg_suba = _segment_metrics(equity, start_date, sub_a_end)
            elapsed = time.time() - t0

            rows.append({
                "short_vol_window": short_vol_window,
                "tilt_scale": tilt_scale,
                "in_sample_sharpe": seg_in["annualized_sharpe"],
                "in_sample_return_pct": seg_in["total_return_pct"],
                "in_sample_max_dd_pct": seg_in["max_drawdown_pct"],
                "oos_sharpe": seg_oos["annualized_sharpe"],
                "oos_return_pct": seg_oos["total_return_pct"],
                "oos_max_dd_pct": seg_oos["max_drawdown_pct"],
                "subA_sharpe": seg_suba["annualized_sharpe"],
                "subA_return_pct": seg_suba["total_return_pct"],
                "subA_max_dd_pct": seg_suba["max_drawdown_pct"],
                "saturation_rate_pct": _saturation_rate(saturation_log) * 100,
            })
            print(
                f"[{i}/{n_combos}] short_window={short_vol_window:>3} tilt_scale={tilt_scale:.2f} "
                f"-> in-sample Sharpe={seg_in['annualized_sharpe']:.3f} "
                f"subA Sharpe={seg_suba['annualized_sharpe']:.3f} "
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

    naive_ranked = table.sort_values(["in_sample_sharpe", "in_sample_max_dd_pct"], ascending=[False, False])
    naive_best = naive_ranked.iloc[0]

    eligible = table[table["saturation_rate_pct"] < SATURATION_RATE_CEILING * 100]
    ranked = eligible.sort_values(["in_sample_sharpe", "in_sample_max_dd_pct"], ascending=[False, False])
    best = ranked.iloc[0]

    print("\n" + "=" * 100)
    print("SELECTION")
    print("=" * 100)
    if int(naive_best["short_vol_window"]) != int(best["short_vol_window"]) or naive_best["tilt_scale"] != best["tilt_scale"]:
        print(
            f"Naive (in-sample Sharpe alone) winner: short_window={int(naive_best['short_vol_window'])}, "
            f"tilt_scale={naive_best['tilt_scale']:.2f} -- REJECTED: "
            f"{naive_best['saturation_rate_pct']:.1f}% saturation rate >= {SATURATION_RATE_CEILING*100:.0f}% ceiling.\n"
        )
    else:
        print("Naive (in-sample Sharpe alone) winner matches the saturation-filtered winner -- no rejection needed.\n")

    print(
        f"Chosen: short_vol_window={int(best['short_vol_window'])}, tilt_scale={best['tilt_scale']:.2f}\n\n"
        f"                 in-sample          subA (COVID)        out-of-sample\n"
        f"  Plain HRP:     Sharpe={hrp_in['annualized_sharpe']:.3f} DD={hrp_in['max_drawdown_pct']:.1f}%   "
        f"Sharpe={hrp_suba['annualized_sharpe']:.3f} DD={hrp_suba['max_drawdown_pct']:.1f}%   "
        f"Sharpe={hrp_oos['annualized_sharpe']:.3f} DD={hrp_oos['max_drawdown_pct']:.1f}%\n"
        f"  Chosen:        Sharpe={best['in_sample_sharpe']:.3f} DD={best['in_sample_max_dd_pct']:.1f}%   "
        f"Sharpe={best['subA_sharpe']:.3f} DD={best['subA_max_dd_pct']:.1f}%   "
        f"Sharpe={best['oos_sharpe']:.3f} DD={best['oos_max_dd_pct']:.1f}%\n"
        f"  Saturation rate (chosen): {best['saturation_rate_pct']:.1f}%\n"
    )
    print(f"Full grid saved to {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
