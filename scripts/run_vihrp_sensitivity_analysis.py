"""Sensitivity analysis: how much does VI-HRP's momentum tilt move weights,
across different time windows and tilt_scale settings?

Slices data/processed/returns.parquet into non-overlapping ~252-day
windows spanning the full available history (2015-2026), and for each
window x tilt_scale combination, compares allocators.hrp.allocate()
(baseline) against allocators.vihrp.allocate() with momentum views
generated at that scale. Answers: are the large weight swings seen in a
single-snapshot demo (scripts/run_vihrp_momentum_demo.py) typical, or an
outlier from one particular window?

Some early windows fail outright: LTGILTBEES trades too sparsely for
allocators.hrp's 90%-non-null check to pass until ~2019 (discovered while
building the backtester -- see backtest/README.md-adjacent history in
scripts/run_backtest.py's warm-up logic). Those windows are reported with
status="insufficient_data" and NaN metrics rather than silently skipped,
since that itself is a relevant sensitivity-analysis finding, not just an
implementation inconvenience.

Usage:
    python scripts/run_vihrp_sensitivity_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from allocators.hrp import allocate as hrp_allocate  # noqa: E402
from allocators.vihrp import allocate as vihrp_allocate  # noqa: E402
from backtest.engine import DEFAULT_RETURNS_PATH  # noqa: E402
from tree.cluster_tree import ClusterTree  # noqa: E402
from views.momentum_generator import generate_momentum_views  # noqa: E402

WINDOW_SIZE = 252
TILT_SCALES = [0.10, 0.20, 0.30]
LOOKBACK = 60

# Deliberately crude: a single "if the whole rebalance were done at
# max_abs_weight_change" cost estimate, not the full per-asset-tier
# backtest.costs.transaction_cost calculation -- good enough to gauge
# order of magnitude across many window/scale combinations, not meant to
# be a precise backtest.
ONE_WAY_COST_BPS = 20
ONE_WAY_COST_FRACTION = ONE_WAY_COST_BPS / 10_000.0

OUTPUT_CSV_PATH = PROJECT_ROOT / "scripts" / "output" / "vihrp_sensitivity.csv"


def _make_windows(returns_df: pd.DataFrame, window_size: int) -> list[pd.DataFrame]:
    """Non-overlapping, contiguous windows of exactly `window_size` rows,
    covering as much of `returns_df` as divides evenly (any remainder at
    the end, smaller than a full window, is dropped).
    """
    n_windows = len(returns_df) // window_size
    return [
        returns_df.iloc[i * window_size : (i + 1) * window_size] for i in range(n_windows)
    ]


def _analyze_window(window_idx: int, returns_window: pd.DataFrame) -> list[dict]:
    """Run the baseline + all tilt_scale variants for one window.

    Returns one result dict per tilt_scale (all sharing the same
    window/status/date-range info), or a single-row list with
    status="insufficient_data" if even the baseline allocation fails.
    """
    window_start = returns_window.index[0].date()
    window_end = returns_window.index[-1].date()

    base_row = {
        "window_idx": window_idx,
        "window_start": window_start,
        "window_end": window_end,
    }

    try:
        baseline = hrp_allocate(returns_window)
    except ValueError as exc:
        return [
            {
                **base_row,
                "tilt_scale": scale,
                "status": "insufficient_data",
                "error": str(exc),
                "num_views": np.nan,
                "max_abs_weight_change": np.nan,
                "mean_abs_weight_change": np.nan,
                "turnover_cost_20bps_estimate": np.nan,
            }
            for scale in TILT_SCALES
        ]

    tree = ClusterTree.build(returns_window.corr())

    rows = []
    for scale in TILT_SCALES:
        views = generate_momentum_views(
            returns_window, tree, lookback=LOOKBACK, tilt_scale=scale
        )
        tilted = vihrp_allocate(returns_window, views=views)

        abs_change = (tilted.reindex(baseline.index) - baseline).abs()
        max_abs_change = float(abs_change.max())
        mean_abs_change = float(abs_change.mean())
        turnover_cost_estimate = max_abs_change * ONE_WAY_COST_FRACTION

        rows.append(
            {
                **base_row,
                "tilt_scale": scale,
                "status": "ok",
                "error": "",
                "num_views": len(views.views),
                "max_abs_weight_change": max_abs_change,
                "mean_abs_weight_change": mean_abs_change,
                "turnover_cost_20bps_estimate": turnover_cost_estimate,
            }
        )

    return rows


def main():
    returns_df = pd.read_parquet(DEFAULT_RETURNS_PATH)
    windows = _make_windows(returns_df, WINDOW_SIZE)
    print(
        f"Full data range: {returns_df.index[0].date()} to {returns_df.index[-1].date()} "
        f"({len(returns_df)} rows)."
    )
    print(
        f"Sliced into {len(windows)} non-overlapping {WINDOW_SIZE}-day windows; "
        f"tilt_scale in {TILT_SCALES}; lookback={LOOKBACK} days.\n"
    )

    results = []
    for idx, window in enumerate(windows):
        results.extend(_analyze_window(idx, window))

    table = pd.DataFrame(results)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", lambda x: f"{x:.5f}")

    print("=" * 100)
    print("VI-HRP momentum-tilt sensitivity: window x tilt_scale")
    print("=" * 100)
    print(
        table[
            [
                "window_idx",
                "window_start",
                "window_end",
                "tilt_scale",
                "status",
                "num_views",
                "max_abs_weight_change",
                "mean_abs_weight_change",
                "turnover_cost_20bps_estimate",
            ]
        ].to_string(index=False)
    )

    n_ok = (table["status"] == "ok").sum()
    n_insufficient = (table["status"] == "insufficient_data").sum()
    print("-" * 100)
    print(f"{n_ok} row(s) computed successfully; {n_insufficient} row(s) hit insufficient_data.")

    ok_table = table[table["status"] == "ok"]
    if not ok_table.empty:
        print()
        print("Max single-asset weight change, by tilt_scale (across all successful windows):")
        print(
            ok_table.groupby("tilt_scale")["max_abs_weight_change"]
            .agg(["mean", "min", "max"])
            .to_string()
        )

    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\nFull table saved to {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
