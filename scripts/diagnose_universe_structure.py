"""Why does plain HRP keep beating VI-HRP + every tested signal on this
12-asset universe (10 NSE equities + GOLDBEES + LTGILTBEES)? Three
competing hypotheses, tested directly against data, not argued from
priors:

  (a) Plain HRP's edge comes almost entirely from cross-asset-class
      diversification (equity vs. gold vs. gilt) that any tilt
      necessarily sacrifices some of.
  (b) The 10-equity sub-universe is too internally correlated to have
      exploitable cross-sectional dispersion in the first place.
  (c) VI-HRP's cluster-level tilting washes out any asset-specific
      signal by averaging it with uninformative assets in the same tree
      node.

Four read-only/analysis-only sections (no production code touched):

  2a. Full 12x12 pairwise correlation matrix (data/processed/returns.parquet,
      pandas pairwise-complete, same convention allocators.hrp.compute_correlation
      uses) -- tests (b) directly: are the 10 equities too intercorrelated
      to leave independent cross-sectional variation?
  2b. The actual ClusterTree (tree.cluster_tree.ClusterTree, identical to
      what allocators.hrp/vihrp build) over this universe, several ways
      (full-sample + each of the 4 backtest segments' own correlation),
      reporting every split's two sides, sizes, and whether it's
      "equity vs. equity" (where a return-based signal operates) or
      involves GOLDBEES/LTGILTBEES -- tests (a) (where does the
      asset-class boundary actually sit in the tree?) and (c) (how many
      equities get averaged together on each side of an equity-only
      split, i.e. how much dilution is structurally possible?).
  2c. A variance/Sharpe decomposition of plain HRP's OWN realized
      backtest performance (backtest.engine.BacktestEngine, identical
      machinery to every prior script in this project) across a 4-rung
      ladder:
        EW12            -- equal weight, all 12 tickers, no class
                           awareness at all.
        NAIVE3WAY       -- equal weight across the 3 asset classes
                           (1/3 each), equal weight WITHIN the equity
                           class -- class-aware, but with an arbitrary
                           (not data-driven) class ratio.
        HRP_TOP_NAIVE_EQ -- plain HRP's OWN real, data-driven class-level
                           mass (computed by just summing plain HRP's
                           own per-rebalance weights within each class --
                           well-defined regardless of HRP's list-bisection
                           mechanics, no literal single "top node"
                           required), but equal weight WITHIN the equity
                           class (plain HRP's own equity-internal
                           risk-parity weighting replaced by equal-weight).
        PLAIN_HRP       -- plain HRP's actual weights, unmodified (both
                           the real class-level mass AND the real
                           equity-internal risk-parity weighting).
      EW12->NAIVE3WAY isolates the value of class-AWARENESS alone (an
      arbitrary 1/3-1/3-1/3 split vs. no split at all). NAIVE3WAY->
      HRP_TOP_NAIVE_EQ isolates the value of getting the class-level
      RATIO right (data-driven vs. arbitrary), holding equity-internal
      naive throughout -- this is "the top-level asset-class split"
      tested directly against (a). HRP_TOP_NAIVE_EQ->PLAIN_HRP isolates
      the value of HRP's equity-internal risk-parity weighting alone,
      holding the class-level mass fixed at HRP's own real split --
      tests (c)'s flip side (if HRP's OWN internal reweighting barely
      matters, a signal operating only within the equity block has very
      little room to add value even in principle, regardless of
      dilution).
  2d. For each of the same 4 segments used in every prior analysis in
      this project (subA 2019-04-05 to 2021-02-26, subB 2021-03-01 to
      2022-12-31, full in-sample 2019-04-05 to 2022-12-31, OOS 2023-01-02
      to 2026-09-11): realized annualized variance for all four rungs,
      and what FRACTION of the total EW12->PLAIN_HRP variance reduction
      is attributable to the class-level-ratio step
      (NAIVE3WAY->HRP_TOP_NAIVE_EQ) vs. the equity-internal step
      (HRP_TOP_NAIVE_EQ->PLAIN_HRP).

Outputs (scripts/output/):
  universe_correlation_matrix.csv   -- full 12x12, full-sample.
  universe_correlation_summary.csv  -- mean/median equity-equity corr,
                                        mean equity-GOLDBEES/LTGILTBEES
                                        corr, GOLDBEES-LTGILTBEES corr;
                                        one row per segment + full-sample.
  universe_tree_structure.csv       -- every split, every segment's tree
                                        (+ full-sample), sizes and type.
  universe_variance_decomposition.csv -- the 2c/2d ladder, per segment.
  universe_structure_diagnosis.md   -- plain-language summary + verdict.

Note on reproducibility: BLAS thread count is pinned to 1, same reason
(and same caveat) as every other calibration/diagnosis script in this
project -- see scripts/diagnose_regime.py's docstring.

Usage:
    python scripts/diagnose_universe_structure.py
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
from backtest.calendar import DEFAULT_PRICES_PATH, weekly_rebalance_dates  # noqa: E402
from backtest.engine import DEFAULT_WINDOW, BacktestEngine, load_default_returns  # noqa: E402
from data.universe import STATIC_UNIVERSE  # noqa: E402
from eval.metrics import max_drawdown, sharpe_ratio, total_return  # noqa: E402
from tree.cluster_tree import ClusterTree  # noqa: E402

EQUITY_TICKERS = [t for t in STATIC_UNIVERSE if t not in ("GOLDBEES", "LTGILTBEES")]
GOLD_TICKER = "GOLDBEES"
GILT_TICKER = "LTGILTBEES"

TRADING_DAYS_PER_YEAR = 252
IN_SAMPLE_END = pd.Timestamp("2022-12-31")
SUB_SPLIT = pd.Timestamp("2021-02-28")

OUTPUT_DIR = PROJECT_ROOT / "scripts" / "output"
CORR_MATRIX_CSV = OUTPUT_DIR / "universe_correlation_matrix.csv"
CORR_SUMMARY_CSV = OUTPUT_DIR / "universe_correlation_summary.csv"
TREE_STRUCTURE_CSV = OUTPUT_DIR / "universe_tree_structure.csv"
VARIANCE_DECOMP_CSV = OUTPUT_DIR / "universe_variance_decomposition.csv"
SUMMARY_MD = OUTPUT_DIR / "universe_structure_diagnosis.md"


def _find_warmup_start(returns_df: pd.DataFrame, window: int) -> pd.Timestamp:
    rolling_non_null_frac = returns_df.notna().rolling(window).mean()
    window_ok = (rolling_non_null_frac >= 0.90).all(axis=1).to_numpy()
    bad_positions = np.where(~window_ok)[0]
    last_bad_window_end = int(bad_positions[-1]) if len(bad_positions) else window - 2
    start_pos = last_bad_window_end + 2
    return returns_df.index[start_pos]


# ---------------------------------------------------------------------------
# 2a. Pairwise correlation
# ---------------------------------------------------------------------------

def _correlation_summary(corr: pd.DataFrame, label: str) -> dict:
    equity_equity = corr.loc[EQUITY_TICKERS, EQUITY_TICKERS].values
    n = len(EQUITY_TICKERS)
    off_diag_mask = ~np.eye(n, dtype=bool)
    equity_equity_pairs = equity_equity[off_diag_mask]

    equity_gold = corr.loc[EQUITY_TICKERS, GOLD_TICKER]
    equity_gilt = corr.loc[EQUITY_TICKERS, GILT_TICKER]
    gold_gilt = corr.loc[GOLD_TICKER, GILT_TICKER]

    return {
        "segment": label,
        "mean_equity_equity_corr": float(np.mean(equity_equity_pairs)),
        "median_equity_equity_corr": float(np.median(equity_equity_pairs)),
        "min_equity_equity_corr": float(np.min(equity_equity_pairs)),
        "max_equity_equity_corr": float(np.max(equity_equity_pairs)),
        "mean_equity_goldbees_corr": float(equity_gold.mean()),
        "mean_equity_ltgiltbees_corr": float(equity_gilt.mean()),
        "goldbees_ltgiltbees_corr": float(gold_gilt),
    }


# ---------------------------------------------------------------------------
# 2b. Tree structure
# ---------------------------------------------------------------------------

def _classify_split(left_assets: list, right_assets: list) -> str:
    left_has_nonequity = any(a in (GOLD_TICKER, GILT_TICKER) for a in left_assets)
    right_has_nonequity = any(a in (GOLD_TICKER, GILT_TICKER) for a in right_assets)
    if not left_has_nonequity and not right_has_nonequity:
        return "equity_vs_equity"
    if left_has_nonequity and right_has_nonequity:
        return "assetclass_vs_assetclass"
    return "equity_vs_assetclass"


def _tree_structure_rows(tree: ClusterTree, label: str) -> list[dict]:
    rows = []

    def walk(node):
        if tree.is_leaf(node):
            return
        left, right = tree.children(node)
        left_assets = tree.assets_under(left)
        right_assets = tree.assets_under(right)
        rows.append({
            "segment": label,
            "node": node,
            "left_size": len(left_assets),
            "right_size": len(right_assets),
            "left_assets": ",".join(sorted(left_assets)),
            "right_assets": ",".join(sorted(right_assets)),
            "split_type": _classify_split(left_assets, right_assets),
            "contains_goldbees": GOLD_TICKER in left_assets or GOLD_TICKER in right_assets,
            "contains_ltgiltbees": GILT_TICKER in left_assets or GILT_TICKER in right_assets,
        })
        walk(left)
        walk(right)

    walk(tree.root)
    return rows


# ---------------------------------------------------------------------------
# 2c/2d. Variance/Sharpe decomposition ladder
# ---------------------------------------------------------------------------

def _ew12_allocator_fn(window_returns: pd.DataFrame) -> pd.Series:
    tickers = list(window_returns.columns)
    return pd.Series(1.0 / len(tickers), index=tickers)


def _naive3way_allocator_fn(window_returns: pd.DataFrame) -> pd.Series:
    tickers = list(window_returns.columns)
    equity_tickers = [t for t in tickers if t not in (GOLD_TICKER, GILT_TICKER)]
    weights = pd.Series(0.0, index=tickers)
    weights[equity_tickers] = (1.0 / 3.0) / len(equity_tickers)
    weights[GOLD_TICKER] = 1.0 / 3.0
    weights[GILT_TICKER] = 1.0 / 3.0
    return weights


def _hrp_top_naive_equity_allocator_fn(window_returns: pd.DataFrame) -> pd.Series:
    """Plain HRP's own real per-rebalance weights, with the equity-internal
    allocation replaced by equal-weight (the class-level mass -- sum of
    HRP's weights within each class -- is left exactly as HRP computed
    it; only how that equity-class mass is split AMONG the 10 equities
    changes, from HRP's risk-parity weighting to equal shares).
    """
    hrp_weights = hrp_allocate(window_returns)
    equity_tickers = [t for t in hrp_weights.index if t not in (GOLD_TICKER, GILT_TICKER)]
    equity_mass = float(hrp_weights[equity_tickers].sum())

    weights = hrp_weights.copy()
    weights[equity_tickers] = equity_mass / len(equity_tickers)
    return weights


def _segment_metrics(equity_curve: pd.Series, seg_start: pd.Timestamp, seg_end: pd.Timestamp) -> dict:
    seg = equity_curve.loc[seg_start:seg_end]
    daily_returns = seg.pct_change().dropna()
    dd = max_drawdown(seg)
    annualized_vol = float(daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
    return {
        "sharpe": sharpe_ratio(daily_returns),
        "return_pct": total_return(seg) * 100,
        "max_dd_pct": dd.max_drawdown * 100,
        "annualized_vol_pct": annualized_vol * 100,
        "annualized_variance": annualized_vol ** 2,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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
    print("2a. PAIRWISE CORRELATION")
    print("=" * 100)

    full_sample_corr = returns_df.corr()
    full_sample_corr.to_csv(CORR_MATRIX_CSV)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print("\nFull 12x12 correlation matrix (full sample, pairwise-complete):")
    print(full_sample_corr.to_string())

    corr_summary_rows = [_correlation_summary(full_sample_corr, "full_sample")]
    for seg_name, (s, e) in segments.items():
        seg_corr = returns_df.loc[s:e].corr()
        corr_summary_rows.append(_correlation_summary(seg_corr, seg_name))
    corr_summary = pd.DataFrame(corr_summary_rows).set_index("segment")
    corr_summary.to_csv(CORR_SUMMARY_CSV)

    print("\nCorrelation summary (full sample + per segment):")
    print(corr_summary.to_string())

    mean_eq_eq_full = corr_summary.loc["full_sample", "mean_equity_equity_corr"]
    print(
        f"\n>>> Mean equity-equity correlation (full sample): {mean_eq_eq_full:.3f} "
        f"({'HIGH -- little independent cross-sectional variation' if mean_eq_eq_full > 0.5 else 'moderate/low -- meaningful independent variation likely exists'})"
    )

    print("\n" + "=" * 100)
    print("2b. CLUSTER TREE STRUCTURE")
    print("=" * 100)

    tree_rows = []
    tree_full = ClusterTree.build(full_sample_corr)
    tree_rows.extend(_tree_structure_rows(tree_full, "full_sample"))
    for seg_name, (s, e) in segments.items():
        seg_corr = returns_df.loc[s:e].corr()
        seg_tree = ClusterTree.build(seg_corr)
        tree_rows.extend(_tree_structure_rows(seg_tree, seg_name))

    tree_table = pd.DataFrame(tree_rows)
    tree_table.to_csv(TREE_STRUCTURE_CSV, index=False)

    print("\nFull-sample tree, every split:")
    full_sample_splits = tree_table[tree_table["segment"] == "full_sample"]
    print(full_sample_splits[["node", "left_size", "right_size", "split_type", "left_assets", "right_assets"]].to_string(index=False))

    split_type_counts = tree_table.groupby(["segment", "split_type"]).size().unstack(fill_value=0)
    print("\nSplit-type counts per segment (how many of each kind of split):")
    print(split_type_counts.to_string())

    # Root-level classification: what does the VERY TOP split separate?
    print("\nRoot-level split, per segment/full-sample (the single highest-level division):")
    for seg_name in ["full_sample"] + list(segments.keys()):
        seg_splits = tree_table[tree_table["segment"] == seg_name]
        root_node = seg_splits["node"].max()  # ClusterTree.root is the highest integer id
        root_row = seg_splits[seg_splits["node"] == root_node].iloc[0]
        print(
            f"  {seg_name:28s}: {root_row['split_type']:24s} "
            f"({root_row['left_size']} vs {root_row['right_size']} assets)"
        )

    print("\n" + "=" * 100)
    print("2c/2d. VARIANCE/SHARPE DECOMPOSITION LADDER")
    print("=" * 100)

    allocators = {
        "EW12": _ew12_allocator_fn,
        "NAIVE3WAY": _naive3way_allocator_fn,
        "HRP_TOP_NAIVE_EQ": _hrp_top_naive_equity_allocator_fn,
        "PLAIN_HRP": hrp_allocate,
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

    decomp_rows = []
    for seg_name, (s, e) in segments.items():
        metrics = {name: _segment_metrics(eq, s, e) for name, eq in equity_curves.items()}
        total_var_reduction = metrics["EW12"]["annualized_variance"] - metrics["PLAIN_HRP"]["annualized_variance"]
        class_awareness_reduction = metrics["EW12"]["annualized_variance"] - metrics["NAIVE3WAY"]["annualized_variance"]
        top_level_reduction = metrics["NAIVE3WAY"]["annualized_variance"] - metrics["HRP_TOP_NAIVE_EQ"]["annualized_variance"]
        equity_internal_reduction = metrics["HRP_TOP_NAIVE_EQ"]["annualized_variance"] - metrics["PLAIN_HRP"]["annualized_variance"]

        row = {"segment": seg_name}
        for name, m in metrics.items():
            row[f"{name}_sharpe"] = m["sharpe"]
            row[f"{name}_vol_pct"] = m["annualized_vol_pct"]
            row[f"{name}_variance"] = m["annualized_variance"]
        row["total_variance_reduction_ew12_to_hrp"] = total_var_reduction
        row["frac_from_class_awareness_ew12_to_naive3way"] = (
            class_awareness_reduction / total_var_reduction if total_var_reduction != 0 else float("nan")
        )
        row["frac_from_top_level_ratio_naive3way_to_hrptop"] = (
            top_level_reduction / total_var_reduction if total_var_reduction != 0 else float("nan")
        )
        row["frac_from_equity_internal_hrptop_to_fullhrp"] = (
            equity_internal_reduction / total_var_reduction if total_var_reduction != 0 else float("nan")
        )
        decomp_rows.append(row)

    decomp_table = pd.DataFrame(decomp_rows).set_index("segment")
    decomp_table.to_csv(VARIANCE_DECOMP_CSV)

    print("\nSharpe by rung, per segment:")
    sharpe_cols = [c for c in decomp_table.columns if c.endswith("_sharpe")]
    print(decomp_table[sharpe_cols].to_string())

    print("\nVariance-reduction fractions (of total EW12->PLAIN_HRP reduction), per segment:")
    frac_cols = [c for c in decomp_table.columns if c.startswith("frac_")]
    print(decomp_table[frac_cols].to_string())

    # --- Write the plain-language summary ---
    full_in_sample_frac = decomp_table.loc["full_in_sample"]
    oos_frac = decomp_table.loc["OOS_2023-01_to_2026-09"]

    md_lines = [
        "# Universe structure diagnosis\n",
        "## 2a. Pairwise correlation\n",
        f"- Mean equity-equity correlation (full sample): **{mean_eq_eq_full:.3f}**",
        f"- Mean equity-GOLDBEES correlation: **{corr_summary.loc['full_sample', 'mean_equity_goldbees_corr']:.3f}**",
        f"- Mean equity-LTGILTBEES correlation: **{corr_summary.loc['full_sample', 'mean_equity_ltgiltbees_corr']:.3f}**",
        f"- GOLDBEES-LTGILTBEES correlation: **{corr_summary.loc['full_sample', 'goldbees_ltgiltbees_corr']:.3f}**\n",
        "## 2b. Tree structure\n",
        f"- Split-type counts (full sample):\n\n{split_type_counts.loc['full_sample'].to_string()}\n",
        "## 2c/2d. Variance decomposition (full in-sample window)\n",
        f"- Fraction of variance reduction from class-awareness alone (EW12->NAIVE3WAY): "
        f"**{full_in_sample_frac['frac_from_class_awareness_ew12_to_naive3way']:.1%}**",
        f"- Fraction from the top-level asset-class RATIO (NAIVE3WAY->HRP_TOP_NAIVE_EQ): "
        f"**{full_in_sample_frac['frac_from_top_level_ratio_naive3way_to_hrptop']:.1%}**",
        f"- Fraction from equity-internal risk-parity weighting (HRP_TOP_NAIVE_EQ->PLAIN_HRP): "
        f"**{full_in_sample_frac['frac_from_equity_internal_hrptop_to_fullhrp']:.1%}**\n",
        "## 2c/2d. Variance decomposition (OOS window)\n",
        f"- Fraction from class-awareness alone: **{oos_frac['frac_from_class_awareness_ew12_to_naive3way']:.1%}**",
        f"- Fraction from the top-level asset-class RATIO: **{oos_frac['frac_from_top_level_ratio_naive3way_to_hrptop']:.1%}**",
        f"- Fraction from equity-internal risk-parity weighting: **{oos_frac['frac_from_equity_internal_hrptop_to_fullhrp']:.1%}**\n",
    ]
    SUMMARY_MD.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\nWrote {SUMMARY_MD}")
    print(f"Wrote {CORR_MATRIX_CSV}, {CORR_SUMMARY_CSV}, {TREE_STRUCTURE_CSV}, {VARIANCE_DECOMP_CSV}")


if __name__ == "__main__":
    main()
