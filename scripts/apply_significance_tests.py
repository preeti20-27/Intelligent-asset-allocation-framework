"""Statistical significance of every Sharpe comparison made so far.

Every prior comparison script in this project (e.g.
scripts/backtest_volatility_regime_generator.py) reported Sharpe ratios as
point estimates. This script backtests ALL nine strategies on the same
machinery (backtest.engine.BacktestEngine, weekly rebalance, real
transaction costs/constraints, 252-day window) and tests which Sharpe
differences survive proper inference, on the same four segments used
throughout (subA 2019-04 -> 2021-02, subB 2021-03 -> 2022-12, full
in-sample 2019-04 -> 2022-12, OOS 2023-01 -> 2026-09):

* plain_hrp, equal_weight
* vihrp_momentum, vihrp_mean_reversion, vihrp_standardized,
  vihrp_volatility_regime (the four VI-HRP signal variants, built exactly
  as in scripts/backtest_volatility_regime_generator.py)
* black_litterman (zero views, equal-weight prior), herc, hcaa

Two procedures, per segment (see eval/bootstrap.py and eval/mcs.py):

1. Pairwise paired circular-block-bootstrap tests of SR_a - SR_b = 0 for
   all 36 pairs, with Holm-adjusted p-values over two families:
   ``p_holm_all_pairs`` (all 36 pairs in the segment) and
   ``p_holm_vs_plain_hrp`` (the 8 "X vs. plain HRP" comparisons, the
   family the paper's claims are actually about; NaN for other pairs).
2. The Hansen-Lunde-Nason Model Confidence Set over all nine strategies
   (90% MCS: ``in_mcs_90``; 75% MCS: ``in_mcs_75``).

As in every prior comparison script, each strategy is backtested ONCE
over warm-up-end..end-of-history and each segment's daily returns are
sliced from that one continuous equity curve, so the five previously
reported strategies' Sharpe ratios reproduce
scripts/output/volatility_regime_comparison.csv exactly (checked at
runtime).

Caveat that no test here can fix: the in-sample segments were also the
segments the view generators' parameters were grid-searched on (and the
volatility-regime generator was the fourth generator tried), so in-sample
p-values for the VI-HRP variants are optimistic. The OOS segment is the
only honest one for them.

Output: scripts/output/significance_results.csv (long format; ``test`` is
"pairwise" or "mcs").

Usage:
    python scripts/apply_significance_tests.py
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from allocators.bl import allocate as bl_allocate  # noqa: E402
from allocators.equal_weight import allocate as equal_weight_allocate  # noqa: E402
from allocators.hcaa import allocate as hcaa_allocate  # noqa: E402
from allocators.herc import allocate as herc_allocate  # noqa: E402
from allocators.hrp import allocate as hrp_allocate  # noqa: E402
from allocators.vihrp import allocate as vihrp_allocate  # noqa: E402
from backtest.calendar import DEFAULT_PRICES_PATH, weekly_rebalance_dates  # noqa: E402
from backtest.engine import DEFAULT_WINDOW, BacktestEngine, load_default_returns  # noqa: E402
from eval.bootstrap import equity_to_returns, sharpe_difference_test  # noqa: E402
from eval.mcs import holm_adjust, model_confidence_set  # noqa: E402
from eval.metrics import sharpe_ratio  # noqa: E402
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

IN_SAMPLE_END = pd.Timestamp("2022-12-31")
SUB_SPLIT = pd.Timestamp("2021-02-28")

N_BOOT = 5000
SEED = 20261020
ALPHA = 0.05

OUTPUT_CSV_PATH = PROJECT_ROOT / "scripts" / "output" / "significance_results.csv"
PRIOR_RESULTS_PATH = PROJECT_ROOT / "scripts" / "output" / "volatility_regime_comparison.csv"


def _find_warmup_start(returns_df: pd.DataFrame, window: int) -> pd.Timestamp:
    rolling_non_null_frac = returns_df.notna().rolling(window).mean()
    window_ok = (rolling_non_null_frac >= 0.90).all(axis=1).to_numpy()
    bad_positions = np.where(~window_ok)[0]
    last_bad_window_end = int(bad_positions[-1]) if len(bad_positions) else window - 2
    start_pos = last_bad_window_end + 2
    return returns_df.index[start_pos]


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


def _check_against_prior_results(segment_returns: dict) -> None:
    """Fail loudly if the reused strategies don't reproduce the committed Sharpe table."""
    if not PRIOR_RESULTS_PATH.exists():
        print(f"(skipping reproduction check: {PRIOR_RESULTS_PATH.name} not found)")
        return
    prior = pd.read_csv(PRIOR_RESULTS_PATH, index_col="segment")
    for seg_name, frame in segment_returns.items():
        for column in prior.columns:
            if not column.endswith("_sharpe"):
                continue
            name = column[: -len("_sharpe")]
            recomputed = sharpe_ratio(frame[name].dropna())
            if not np.isclose(recomputed, prior.loc[seg_name, column], atol=1e-9):
                raise RuntimeError(
                    f"{name} {seg_name} Sharpe {recomputed:.6f} != committed "
                    f"{prior.loc[seg_name, column]:.6f}; backtests no longer match prior results."
                )
    print("Reproduction check passed: prior five strategies' Sharpe ratios match exactly.")


def main():
    returns_df = load_default_returns()
    prices_df = pd.read_parquet(DEFAULT_PRICES_PATH)
    trading_days = pd.DatetimeIndex(sorted(prices_df.index.unique()))
    rebalance_dates = weekly_rebalance_dates(trading_days)

    start_date = _find_warmup_start(returns_df, DEFAULT_WINDOW)
    end_date = returns_df.index[-1]
    oos_start = trading_days[trading_days > IN_SAMPLE_END][0]
    sub_a_end = trading_days[trading_days <= SUB_SPLIT][-1]
    sub_b_start = trading_days[trading_days > SUB_SPLIT][0]

    segments = {
        "subA_2019-04_to_2021-02": (start_date, sub_a_end),
        "subB_2021-03_to_2022-12": (sub_b_start, IN_SAMPLE_END),
        "full_in_sample": (start_date, IN_SAMPLE_END),
        "OOS_2023-01_to_2026-09": (oos_start, end_date),
    }

    allocators = {
        "plain_hrp": hrp_allocate,
        "equal_weight": equal_weight_allocate,
        "vihrp_momentum": _make_vihrp_allocator_fn(generate_momentum_views, lookback=LOOKBACK, tilt_scale=TILT_SCALE),
        "vihrp_mean_reversion": _make_vihrp_allocator_fn(generate_mean_reversion_views, lookback=LOOKBACK, tilt_scale=TILT_SCALE),
        "vihrp_standardized": _make_standardized_allocator_fn(returns_df),
        "vihrp_volatility_regime": _make_vihrp_allocator_fn(
            generate_volatility_regime_views,
            short_vol_window=SHORT_VOL_WINDOW, long_vol_window=LONG_VOL_WINDOW, tilt_scale=VOL_TILT_SCALE,
        ),
        "black_litterman": bl_allocate,
        "herc": herc_allocate,
        "hcaa": hcaa_allocate,
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

    # Same slicing as every prior script's _segment_metrics: slice the
    # equity curve to the segment, then take its daily returns.
    segment_returns = {
        seg_name: pd.DataFrame({name: equity_to_returns(eq.loc[s:e]) for name, eq in equity_curves.items()})
        for seg_name, (s, e) in segments.items()
    }
    _check_against_prior_results(segment_returns)

    rows = []
    for seg_name, frame in segment_returns.items():
        pair_rows = []
        for name_a, name_b in combinations(allocators, 2):
            r = sharpe_difference_test(frame[name_a], frame[name_b], n_boot=N_BOOT, alpha=ALPHA, seed=SEED)
            pair_rows.append({
                "segment": seg_name, "test": "pairwise",
                "strategy_a": name_a, "strategy_b": name_b,
                "sharpe_a": r.sharpe_a, "sharpe_b": r.sharpe_b, "sharpe_diff": r.difference,
                "ci95_low": r.ci_low, "ci95_high": r.ci_high, "p_value": r.p_value,
                "block_size": r.block_size, "n_obs": r.n_obs,
            })
        pairs = pd.DataFrame(pair_rows)
        pairs["p_holm_all_pairs"] = holm_adjust(pairs["p_value"])
        vs_hrp = pairs["strategy_a"].eq("plain_hrp") | pairs["strategy_b"].eq("plain_hrp")
        pairs["p_holm_vs_plain_hrp"] = np.nan
        pairs.loc[vs_hrp, "p_holm_vs_plain_hrp"] = holm_adjust(pairs.loc[vs_hrp, "p_value"])
        pairs["significant_raw"] = pairs["p_value"] < ALPHA
        pairs["significant_holm_all_pairs"] = pairs["p_holm_all_pairs"] < ALPHA
        pairs["significant_holm_vs_plain_hrp"] = pairs["p_holm_vs_plain_hrp"] < ALPHA
        rows.extend(pairs.to_dict("records"))

        mcs = model_confidence_set(frame, alpha=0.10, n_boot=N_BOOT, seed=SEED)
        for model, m in mcs.iterrows():
            rows.append({
                "segment": seg_name, "test": "mcs", "strategy_a": model,
                "sharpe_a": m["sharpe"], "mcs_p_value": m["mcs_p_value"],
                "mcs_elimination_order": int(m["elimination_order"]),
                "in_mcs_90": bool(m["mcs_p_value"] >= 0.10),
                "in_mcs_75": bool(m["mcs_p_value"] >= 0.25),
                "n_obs": len(frame.dropna()),
            })

    results = pd.DataFrame(rows)
    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_CSV_PATH, index=False)

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")

    pairwise = results[results["test"] == "pairwise"]
    for seg_name in segments:
        print("\n" + "=" * 110)
        print(seg_name)
        print("=" * 110)
        seg = pairwise[pairwise["segment"] == seg_name]
        print("-- vs. plain HRP --")
        print(seg[seg["p_holm_vs_plain_hrp"].notna()][
            ["strategy_a", "strategy_b", "sharpe_a", "sharpe_b", "sharpe_diff",
             "ci95_low", "ci95_high", "p_value", "p_holm_vs_plain_hrp"]
        ].to_string(index=False))
        sig = seg[seg["significant_holm_all_pairs"]]
        print(f"-- pairs significant after Holm over all {len(seg)} pairs: {len(sig)} --")
        if len(sig):
            print(sig[["strategy_a", "strategy_b", "sharpe_diff", "p_value", "p_holm_all_pairs"]].to_string(index=False))
        mcs = results[(results["test"] == "mcs") & (results["segment"] == seg_name)]
        print("-- Model Confidence Set (best first) --")
        print(mcs[["strategy_a", "sharpe_a", "mcs_p_value", "in_mcs_90", "in_mcs_75"]].to_string(index=False))

    print(f"\nFull results saved to {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
