"""Full-history backtest comparison: plain HRP vs. VI-HRP + momentum views.

Runs backtest.engine.BacktestEngine.run() unchanged (weekly rebalance,
transaction costs, constraints all as-is) twice over the entire available
history:
  a) plain HRP    -- allocators.hrp.allocate() at every rebalance
  b) VI-HRP       -- at every rebalance: build a ClusterTree, generate
                     momentum views (60-day lookback) at the given
                     tilt_scale, call allocators.vihrp.allocate() with
                     those views

Records cumulative return, annualized Sharpe, max drawdown, annualized
volatility, and total turnover for both, plus (VI-HRP only) whether any
view saturated (tilt_strength == 1.0) at each rebalance -- both "any node
anywhere" and "the root split specifically" -- to test whether the
tilt-saturation behavior found in the sensitivity analysis
(scripts/run_vihrp_sensitivity_analysis.py,
scripts/investigate_extreme_windows.py) helps or hurts realized returns,
and whether raising tilt_scale meaningfully reduces how often that
saturation happens.

Each run appends/updates its row (keyed by tilt_scale) in
scripts/output/vihrp_scale_sweep.csv and reprints the combined table
across every tilt_scale run so far, so running this three times at
different --tilt-scale values builds up one comparison table.

Usage:
    python scripts/run_vihrp_backtest_comparison.py --tilt-scale 0.10
    python scripts/run_vihrp_backtest_comparison.py --tilt-scale 0.30
    python scripts/run_vihrp_backtest_comparison.py --tilt-scale 0.50
"""

from __future__ import annotations

import argparse
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
from backtest.engine import (  # noqa: E402
    DEFAULT_WINDOW,
    BacktestEngine,
    load_default_returns,
)
from eval.metrics import max_drawdown, sharpe_ratio, total_return  # noqa: E402
from tree.cluster_tree import ClusterTree  # noqa: E402
from views.momentum_generator import TILT_SCALE, generate_momentum_views  # noqa: E402

MOMENTUM_LOOKBACK = 60
FORWARD_RETURN_DAYS = 20  # ~4 trading weeks
TRADING_DAYS_PER_YEAR = 252

# Below this saturation rate, a "saturated vs. non-saturated" comparison
# has a large enough non-saturated bucket to be worth showing at all.
SATURATION_RATE_THRESHOLD_FOR_TABLE2 = 0.90

SWEEP_CSV_PATH = PROJECT_ROOT / "scripts" / "output" / "vihrp_scale_sweep.csv"


def _find_warmup_start(returns_df: pd.DataFrame, window: int) -> pd.Timestamp:
    """Same rolling 90%-non-null-per-ticker check used in run_backtest.py:
    LTGILTBEES trades too sparsely for allocate()'s own validation to pass
    until well after its 2016 listing (scattered gaps into 2019), so a
    flat "skip N rows" warm-up isn't enough -- see run_backtest.py's
    comments for the full story.
    """
    rolling_non_null_frac = returns_df.notna().rolling(window).mean()
    window_ok = (rolling_non_null_frac >= 0.90).all(axis=1).to_numpy()
    bad_positions = np.where(~window_ok)[0]
    last_bad_window_end = int(bad_positions[-1]) if len(bad_positions) else window - 2
    start_pos = last_bad_window_end + 2
    return returns_df.index[start_pos]


def _compute_total_turnover(states: list) -> float:
    """Sum of |weight change| across every consecutive pair of daily
    states -- zero on non-rebalance days (weights held constant), so this
    equals the sum of turnover across every actual rebalance.
    """
    total = 0.0
    prev_weights = None
    for state in states:
        weights = pd.Series(state.weights, dtype=float)
        if prev_weights is not None:
            aligned_prev = prev_weights.reindex(weights.index, fill_value=0.0)
            total += float((weights - aligned_prev).abs().sum())
        prev_weights = weights
    return total


def _make_vihrp_allocator_fn(returns_df_full: pd.DataFrame, lookback: int, tilt_scale: float = TILT_SCALE):
    """Wraps allocators.vihrp.allocate so every call also logs, keyed by
    the actual rebalance date (the day after the fed window's last row):
      - any_saturated: whether ANY generated view had tilt_strength == 1.0
      - root_saturated: whether the ROOT split specifically had an
        emitted view with tilt_strength == 1.0 -- the more meaningful
        definition, since root-level saturation is what drove both
        extreme windows found in scripts/investigate_extreme_windows.py
        (a saturated tilt at the root funnels the entire portfolio, not
        just one sub-cluster's share).
    """
    saturation_log: list[dict] = []

    def vihrp_allocator_fn(window_returns: pd.DataFrame) -> pd.Series:
        tree = ClusterTree.build(window_returns.corr())
        views = generate_momentum_views(
            window_returns, tree, lookback=lookback, tilt_scale=tilt_scale
        )
        any_saturated = any(v.tilt_strength == 1.0 for v in views.views)

        root_saturated = False
        if not tree.is_leaf(tree.root):
            left, right = tree.children(tree.root)
            root_child_asset_sets = {
                frozenset(tree.assets_under(left)),
                frozenset(tree.assets_under(right)),
            }
            root_saturated = any(
                v.node_assets in root_child_asset_sets and v.tilt_strength == 1.0
                for v in views.views
            )

        last_date = window_returns.index[-1]
        pos = returns_df_full.index.get_loc(last_date)
        rebalance_date = (
            returns_df_full.index[pos + 1] if pos + 1 < len(returns_df_full.index) else None
        )
        saturation_log.append(
            {
                "rebalance_date": rebalance_date,
                "any_saturated": any_saturated,
                "root_saturated": root_saturated,
                "num_views": len(views.views),
            }
        )
        return vihrp_allocate(window_returns, views=views)

    return vihrp_allocator_fn, saturation_log


def _summarize_backtest(label: str, equity_curve: pd.Series, states: list) -> dict:
    daily_returns = equity_curve.pct_change().dropna()
    dd = max_drawdown(equity_curve)
    return {
        "strategy": label,
        "cumulative_return_pct": total_return(equity_curve) * 100,
        "annualized_sharpe": sharpe_ratio(daily_returns),
        "max_drawdown_pct": dd.max_drawdown * 100,
        "annualized_volatility_pct": float(daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)) * 100,
        "total_turnover": _compute_total_turnover(states),
        "n_trading_days": len(states),
    }


def _forward_return_table(
    saturation_log: list[dict], equity_curve: pd.Series, bucket_key: str
) -> tuple[pd.DataFrame, int]:
    """Builds the "forward FORWARD_RETURN_DAYS-day return by bucket" table,
    bucketing on `saturation_log`'s `bucket_key` field (e.g. "any_saturated"
    or "root_saturated").
    """
    forward_returns = {"saturated": [], "non_saturated": []}
    skipped_at_end = 0
    for entry in saturation_log:
        rebalance_date = entry["rebalance_date"]
        if rebalance_date is None or rebalance_date not in equity_curve.index:
            skipped_at_end += 1
            continue
        pos = equity_curve.index.get_loc(rebalance_date)
        if pos + FORWARD_RETURN_DAYS >= len(equity_curve):
            skipped_at_end += 1
            continue
        fwd_return = equity_curve.iloc[pos + FORWARD_RETURN_DAYS] / equity_curve.iloc[pos] - 1.0
        bucket = "saturated" if entry[bucket_key] else "non_saturated"
        forward_returns[bucket].append(fwd_return)

    rows = []
    for bucket, values in forward_returns.items():
        if values:
            rows.append(
                {
                    "rebalance_type": bucket,
                    "n_rebalances": len(values),
                    "mean_forward_4wk_return_pct": float(np.mean(values)) * 100,
                    "median_forward_4wk_return_pct": float(np.median(values)) * 100,
                    "std_forward_4wk_return_pct": float(np.std(values, ddof=1)) * 100
                    if len(values) > 1
                    else np.nan,
                }
            )
        else:
            rows.append(
                {
                    "rebalance_type": bucket,
                    "n_rebalances": 0,
                    "mean_forward_4wk_return_pct": np.nan,
                    "median_forward_4wk_return_pct": np.nan,
                    "std_forward_4wk_return_pct": np.nan,
                }
            )

    return pd.DataFrame(rows).set_index("rebalance_type"), skipped_at_end


def _update_sweep_csv(row: dict) -> pd.DataFrame:
    """Upserts `row` (keyed by tilt_scale) into SWEEP_CSV_PATH and returns
    the full, sorted combined table.
    """
    if SWEEP_CSV_PATH.exists():
        existing = pd.read_csv(SWEEP_CSV_PATH)
        existing = existing[existing["tilt_scale"] != row["tilt_scale"]]
        combined = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    else:
        combined = pd.DataFrame([row])

    combined = combined.sort_values("tilt_scale").reset_index(drop=True)
    SWEEP_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(SWEEP_CSV_PATH, index=False)
    return combined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tilt-scale",
        type=float,
        default=TILT_SCALE,
        help=f"tilt_scale passed to generate_momentum_views (default {TILT_SCALE}).",
    )
    args = parser.parse_args()
    tilt_scale = args.tilt_scale

    returns_df = load_default_returns()
    prices_df = pd.read_parquet(DEFAULT_PRICES_PATH)

    trading_days = pd.DatetimeIndex(sorted(prices_df.index.unique()))
    rebalance_dates = weekly_rebalance_dates(trading_days)

    start_date = _find_warmup_start(returns_df, DEFAULT_WINDOW)
    end_date = returns_df.index[-1]
    print(
        f"Backtest range: {start_date.date()} to {end_date.date()} "
        f"({len(returns_df.loc[start_date:end_date])} trading days), "
        f"window={DEFAULT_WINDOW}, momentum lookback={MOMENTUM_LOOKBACK}, "
        f"tilt_scale={tilt_scale}.\n"
    )

    # --- Plain HRP (doesn't depend on tilt_scale, but rerun for a fully
    # self-contained, reproducible invocation each time) ---
    hrp_engine = BacktestEngine(returns_df)
    hrp_equity_curve, hrp_states = hrp_engine.run(
        start_date=start_date,
        end_date=end_date,
        allocator_fn=hrp_allocate,
        window=DEFAULT_WINDOW,
        rebalance_dates=rebalance_dates,
    )
    print(f"Plain HRP backtest complete: {len(hrp_states)} trading days.")

    # --- VI-HRP + momentum views ---
    vihrp_allocator_fn, saturation_log = _make_vihrp_allocator_fn(
        returns_df, MOMENTUM_LOOKBACK, tilt_scale=tilt_scale
    )
    vihrp_engine = BacktestEngine(returns_df)
    vihrp_equity_curve, vihrp_states = vihrp_engine.run(
        start_date=start_date,
        end_date=end_date,
        allocator_fn=vihrp_allocator_fn,
        window=DEFAULT_WINDOW,
        rebalance_dates=rebalance_dates,
    )
    print(
        f"VI-HRP backtest complete: {len(vihrp_states)} trading days, "
        f"{len(saturation_log)} rebalances logged.\n"
    )

    # --- Table 1: full-period comparison (this run's tilt_scale) ---
    summary_hrp = _summarize_backtest("plain_hrp", hrp_equity_curve, hrp_states)
    summary_vihrp = _summarize_backtest("vihrp_momentum", vihrp_equity_curve, vihrp_states)
    comparison_table = pd.DataFrame([summary_hrp, summary_vihrp]).set_index("strategy")

    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")

    print("=" * 100)
    print(f"TABLE 1: Full-period comparison, plain HRP vs. VI-HRP + momentum (tilt_scale={tilt_scale})")
    print("=" * 100)
    print(comparison_table.to_string())
    print()

    n = len(saturation_log)
    n_any_saturated = sum(1 for e in saturation_log if e["any_saturated"])
    n_root_saturated = sum(1 for e in saturation_log if e["root_saturated"])
    any_rate = n_any_saturated / n
    root_rate = n_root_saturated / n
    print(
        f"Saturation events (any node): {n_any_saturated} of {n} rebalances ({any_rate:.1%})\n"
        f"Saturation events (root split specifically): {n_root_saturated} of {n} rebalances ({root_rate:.1%})\n"
    )

    # --- Update and print the combined cross-scale sweep table ---
    sweep_row = {
        "tilt_scale": tilt_scale,
        "cumulative_return_pct": summary_vihrp["cumulative_return_pct"],
        "annualized_sharpe": summary_vihrp["annualized_sharpe"],
        "max_drawdown_pct": summary_vihrp["max_drawdown_pct"],
        "annualized_volatility_pct": summary_vihrp["annualized_volatility_pct"],
        "total_turnover": summary_vihrp["total_turnover"],
        "saturation_rate_pct": any_rate * 100,
        "root_saturation_rate_pct": root_rate * 100,
        "plain_hrp_cumulative_return_pct": summary_hrp["cumulative_return_pct"],
        "plain_hrp_annualized_sharpe": summary_hrp["annualized_sharpe"],
    }
    combined_sweep_table = _update_sweep_csv(sweep_row)

    print("=" * 100)
    print("COMBINED SWEEP TABLE: VI-HRP metrics across all tilt_scale values run so far")
    print("=" * 100)
    print(
        combined_sweep_table[
            [
                "tilt_scale",
                "cumulative_return_pct",
                "annualized_sharpe",
                "max_drawdown_pct",
                "annualized_volatility_pct",
                "total_turnover",
                "saturation_rate_pct",
                "root_saturation_rate_pct",
            ]
        ].to_string(index=False)
    )
    print(f"\n(saved to {SWEEP_CSV_PATH}; plain HRP baseline this run: "
          f"{summary_hrp['cumulative_return_pct']:.2f}% return, "
          f"{summary_hrp['annualized_sharpe']:.3f} Sharpe)\n")

    # --- Table 2 (original definition: any node saturated) ---
    any_table, any_skipped = _forward_return_table(saturation_log, vihrp_equity_curve, "any_saturated")
    print("=" * 100)
    print(
        f"TABLE 2a: VI-HRP forward {FORWARD_RETURN_DAYS}-day (~4wk) return, "
        f"ANY-node-saturated vs. not (tilt_scale={tilt_scale})"
    )
    print("=" * 100)
    print(any_table.to_string())
    print(f"({any_skipped} rebalance(s) excluded: too close to the end of the backtest.)\n")

    # --- Table 2b (redefined: root split specifically), only when it's
    # actually informative (root saturation rate below threshold) ---
    if root_rate < SATURATION_RATE_THRESHOLD_FOR_TABLE2:
        root_table, root_skipped = _forward_return_table(
            saturation_log, vihrp_equity_curve, "root_saturated"
        )
        print("=" * 100)
        print(
            f"TABLE 2b: VI-HRP forward {FORWARD_RETURN_DAYS}-day (~4wk) return, "
            f"ROOT-split-saturated vs. not (tilt_scale={tilt_scale})"
        )
        print("=" * 100)
        print(root_table.to_string())
        print(f"({root_skipped} rebalance(s) excluded: too close to the end of the backtest.)")

        smaller_bucket_n = int(root_table["n_rebalances"].min())
        if smaller_bucket_n >= 20:
            print(f"\nSmaller bucket has n={smaller_bucket_n} -- large enough to be a meaningful comparison.")
        else:
            print(f"\nSmaller bucket has only n={smaller_bucket_n} -- still too small to draw a firm conclusion.")
    else:
        print(
            f"Root saturation rate ({root_rate:.1%}) is still >= "
            f"{SATURATION_RATE_THRESHOLD_FOR_TABLE2:.0%} at this tilt_scale -- "
            "skipping the root-based Table 2b (non-saturated bucket would be too small)."
        )


if __name__ == "__main__":
    main()
