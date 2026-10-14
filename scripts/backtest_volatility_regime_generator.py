"""Five-way comparison: plain HRP vs. the three prior VI-HRP generators
(momentum, mean-reversion, standardized momentum+reversal) vs. the new
volatility-regime generator.

Companion to scripts/diagnose_universe_structure.py (read that script's
docstring and committed output, scripts/output/universe_structure_diagnosis.md,
first for why this generator exists: virtually all of plain HRP's edge
comes from the two asset-class-level tree splits, not equity-internal
weighting, and the prior three generators' trailing-return signals carry
no real information there). Runs the same backtest machinery
(backtest.engine.BacktestEngine, weekly rebalance, real transaction
costs/constraints) for all five allocators and reports them side by side
on the same four segments every prior comparison script in this project
uses: the two in-sample halves (subA: 2019-04-05 to 2021-02-26 --
containing the 2020 COVID crash this generator specifically targets;
subB: 2021-03-01 to 2022-12-31), the full in-sample window, and the
out-of-sample holdout (2023-01-02 to 2026-09-11).

Each of the five allocators is backtested ONCE over the full
warm-up-end..end-of-history range; every segment's metrics are sliced
out of that one continuous equity curve (scale-invariant under slicing a
contiguous curve -- see scripts/calibrate_momentum_generator.py's
``_segment_metrics``).

Note on reproducibility: BLAS thread count is pinned to 1 (see
scripts/diagnose_regime.py's docstring).

Usage:
    python scripts/backtest_volatility_regime_generator.py
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
from views.mean_reversion_generator import generate_mean_reversion_views  # noqa: E402
from views.momentum_generator import LOOKBACK, TILT_SCALE, generate_momentum_views  # noqa: E402
from views.standardized_momentum_reversal_generator import (  # noqa: E402
    MOMENTUM_FORMATION_DAYS,
    MOMENTUM_SKIP_DAYS,
    REVERSAL_FORMATION_DAYS,
    TILT_SCALE as STD_TILT_SCALE,
    generate_standardized_momentum_reversal_views,
)
from views.volatility_regime_generator import (  # noqa: E402
    LONG_VOL_WINDOW,
    SHORT_VOL_WINDOW,
    TILT_SCALE as VOL_TILT_SCALE,
    generate_volatility_regime_views,
)

TRADING_DAYS_PER_YEAR = 252
IN_SAMPLE_END = pd.Timestamp("2022-12-31")
SUB_SPLIT = pd.Timestamp("2021-02-28")

OUTPUT_CSV_PATH = PROJECT_ROOT / "scripts" / "output" / "volatility_regime_comparison.csv"


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
        "sharpe": sharpe_ratio(daily_returns),
        "return_pct": total_return(seg) * 100,
        "max_dd_pct": dd.max_drawdown * 100,
        "vol_pct": float(daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)) * 100,
    }


def _make_vihrp_allocator_fn(generator_fn, **generator_kwargs):
    def allocator_fn(window_returns: pd.DataFrame) -> pd.Series:
        tree = ClusterTree.build(window_returns.corr())
        views = generator_fn(window_returns, tree, **generator_kwargs)
        return vihrp_allocate(window_returns, views=views)

    return allocator_fn


def _make_standardized_allocator_fn(returns_df_full: pd.DataFrame):
    needed_history = REVERSAL_FORMATION_DAYS + 50

    def allocator_fn(window_returns: pd.DataFrame) -> pd.Series:
        tree = ClusterTree.build(window_returns.corr())
        last_date = window_returns.index[-1]
        long_history = returns_df_full.loc[:last_date].tail(needed_history)
        views = generate_standardized_momentum_reversal_views(
            long_history, tree,
            momentum_formation_days=MOMENTUM_FORMATION_DAYS,
            momentum_skip_days=MOMENTUM_SKIP_DAYS,
            reversal_formation_days=REVERSAL_FORMATION_DAYS,
            tilt_scale=STD_TILT_SCALE,
        )
        return vihrp_allocate(window_returns, views=views)

    return allocator_fn


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

    allocators = {
        "plain_hrp": hrp_allocate,
        "vihrp_momentum": _make_vihrp_allocator_fn(generate_momentum_views, lookback=LOOKBACK, tilt_scale=TILT_SCALE),
        "vihrp_mean_reversion": _make_vihrp_allocator_fn(generate_mean_reversion_views, lookback=LOOKBACK, tilt_scale=TILT_SCALE),
        "vihrp_standardized": _make_standardized_allocator_fn(returns_df),
        "vihrp_volatility_regime": _make_vihrp_allocator_fn(
            generate_volatility_regime_views,
            short_vol_window=SHORT_VOL_WINDOW, long_vol_window=LONG_VOL_WINDOW, tilt_scale=VOL_TILT_SCALE,
        ),
    }

    equity_curves = {}
    for name, fn in allocators.items():
        engine = BacktestEngine(returns_df)
        equity, _states = engine.run(
            start_date=start_date, end_date=end_date, allocator_fn=fn,
            window=DEFAULT_WINDOW, rebalance_dates=rebalance_dates,
        )
        equity_curves[name] = equity
        print(f"{name}: backtest complete.")

    rows = []
    for seg_name, (s, e) in segments.items():
        row = {"segment": seg_name}
        for alloc_name, equity in equity_curves.items():
            m = _segment_metrics(equity, s, e)
            row[f"{alloc_name}_sharpe"] = m["sharpe"]
            row[f"{alloc_name}_return_pct"] = m["return_pct"]
            row[f"{alloc_name}_max_dd_pct"] = m["max_dd_pct"]
        rows.append(row)
    table = pd.DataFrame(rows).set_index("segment")

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")

    print("\n" + "=" * 130)
    print("FIVE-WAY COMPARISON: plain HRP vs. momentum vs. mean-reversion vs. standardized vs. volatility-regime")
    print("=" * 130)
    sharpe_cols = [c for c in table.columns if c.endswith("_sharpe")]
    dd_cols = [c for c in table.columns if c.endswith("_max_dd_pct")]
    ret_cols = [c for c in table.columns if c.endswith("_return_pct")]
    print("\n-- Sharpe --")
    print(table[sharpe_cols].to_string())
    print("\n-- Total return (%) --")
    print(table[ret_cols].to_string())
    print("\n-- Max drawdown (%) --")
    print(table[dd_cols].to_string())

    print("\n-- Winner by Sharpe, per segment --")
    for seg_name in table.index:
        scores = {c.replace("_sharpe", ""): table.loc[seg_name, c] for c in sharpe_cols}
        winner = max(scores, key=scores.get)
        print(f"  {seg_name:28s}: {winner} ({scores[winner]:.3f})")

    print("\n-- subA (COVID) deep dive: Sharpe AND drawdown side by side --")
    suba = table.loc["subA_2019-04_to_2021-02"]
    for alloc_name in allocators:
        print(f"  {alloc_name:28s} Sharpe={suba[f'{alloc_name}_sharpe']:.3f}  DD={suba[f'{alloc_name}_max_dd_pct']:.1f}%")

    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_CSV_PATH)
    print(f"\nFull table saved to {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
