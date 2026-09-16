"""Read-only diagnostic: what's driving the ~0.95 max weight swings found by
run_vihrp_sensitivity_analysis.py in windows 8 and 9, vs. a "normal" window?

Does not change any production code. For each window, this:
  1. Prints every View generate_momentum_views would produce (node_assets,
     tilt_strength, confidence, and the raw trailing-return difference
     behind it) by walking the same tree independently.
  2. Finds the single biggest-moving asset (hrp baseline vs vihrp+momentum),
     and reports every view on the root-to-that-asset path -- the full
     causal chain, since the final weight is a product of every ratio
     along that path, not just one split.
  3. For the most decisive view on that path, reports node size, row
     count and non-null fraction actually available, and the real
     trailing returns (in %) on both sides of that split.

Usage:
    python scripts/investigate_extreme_windows.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from allocators.hrp import allocate as hrp_allocate  # noqa: E402
from allocators.vihrp import allocate as vihrp_allocate  # noqa: E402
from backtest.engine import DEFAULT_RETURNS_PATH  # noqa: E402
from tree.cluster_tree import ClusterTree  # noqa: E402
from views.momentum_generator import (  # noqa: E402
    MIN_TILT_STRENGTH_TO_EMIT,
    _cluster_trailing_return,
    _data_availability_fraction,
)
from views.momentum_generator import generate_momentum_views  # noqa: E402

WINDOW_SIZE = 252
LOOKBACK = 60
TILT_SCALE_USED = 0.10  # matches the sensitivity script's default row


def _make_windows(returns_df: pd.DataFrame, window_size: int) -> list[pd.DataFrame]:
    n_windows = len(returns_df) // window_size
    return [returns_df.iloc[i * window_size : (i + 1) * window_size] for i in range(n_windows)]


def _walk_and_report_views(returns_df: pd.DataFrame, tree: ClusterTree, lookback: int):
    """Independently walk the tree (mirroring generate_momentum_views'
    traversal) and report every split's raw numbers -- not just the ones
    that clear MIN_TILT_STRENGTH_TO_EMIT, so we can see near-misses too.
    """
    rows = []

    def walk(node):
        if tree.is_leaf(node):
            return
        left, right = tree.children(node)
        left_assets = tree.assets_under(left)
        right_assets = tree.assets_under(right)

        ret_left = _cluster_trailing_return(returns_df, left_assets, lookback)
        ret_right = _cluster_trailing_return(returns_df, right_assets, lookback)
        diff = ret_left - ret_right
        tilt_strength = min(abs(diff) / TILT_SCALE_USED, 1.0) if diff == diff else float("nan")

        favored_assets = left_assets if diff > 0 else right_assets
        unfavored_assets = right_assets if diff > 0 else left_assets
        confidence = min(
            _data_availability_fraction(returns_df, favored_assets, lookback),
            _data_availability_fraction(returns_df, unfavored_assets, lookback),
        )

        rows.append(
            {
                "node": node,
                "left_assets": left_assets,
                "right_assets": right_assets,
                "ret_left_pct": ret_left * 100,
                "ret_right_pct": ret_right * 100,
                "diff_pct": diff * 100,
                "favored_assets": favored_assets,
                "tilt_strength": tilt_strength,
                "confidence": confidence,
                "emitted": tilt_strength >= MIN_TILT_STRENGTH_TO_EMIT,
            }
        )
        walk(left)
        walk(right)

    walk(tree.root)
    return rows


def _path_to_asset(tree: ClusterTree, asset: str):
    """List of nodes from root down to the leaf for `asset`."""
    path = []

    def walk(node):
        path.append(node)
        if tree.is_leaf(node):
            return asset in tree.assets_under(node)
        left, right = tree.children(node)
        if asset in tree.assets_under(left):
            if walk(left):
                return True
        elif asset in tree.assets_under(right):
            if walk(right):
                return True
        path.pop()
        return False

    walk(tree.root)
    return path


def investigate_window(label: str, returns_window: pd.DataFrame):
    print("=" * 100)
    print(f"WINDOW: {label}  ({returns_window.index[0].date()} to {returns_window.index[-1].date()}, "
          f"{len(returns_window)} rows)")
    print("=" * 100)

    tree = ClusterTree.build(returns_window.corr())
    baseline = hrp_allocate(returns_window)
    views = generate_momentum_views(returns_window, tree, lookback=LOOKBACK, tilt_scale=TILT_SCALE_USED)
    tilted = vihrp_allocate(returns_window, views=views)

    all_splits = _walk_and_report_views(returns_window, tree, LOOKBACK)

    print(f"\n-- Every split in the tree ({len(all_splits)} total, {len(views.views)} emitted as views) --")
    for s in sorted(all_splits, key=lambda r: -r["tilt_strength"] if r["tilt_strength"] == r["tilt_strength"] else -1):
        marker = "EMITTED" if s["emitted"] else "skipped"
        print(
            f"  [{marker}] node={s['node']}  "
            f"left({len(s['left_assets'])})={sorted(s['left_assets'])}  "
            f"right({len(s['right_assets'])})={sorted(s['right_assets'])}\n"
            f"           ret_left={s['ret_left_pct']:+.2f}%  ret_right={s['ret_right_pct']:+.2f}%  "
            f"diff={s['diff_pct']:+.2f}%  tilt_strength={s['tilt_strength']:.3f}  "
            f"confidence={s['confidence']:.3f}"
        )

    diff_series = (tilted.reindex(baseline.index) - baseline)
    biggest_mover = diff_series.abs().idxmax()
    print(f"\n-- Biggest single-asset weight change: {biggest_mover} --")
    print(f"  hrp baseline: {baseline[biggest_mover]:.4f}   vihrp+momentum: {tilted[biggest_mover]:.4f}   "
          f"change: {diff_series[biggest_mover]:+.4f}")

    path = _path_to_asset(tree, biggest_mover)
    print(f"\n-- Root-to-leaf path for {biggest_mover} ({len(path)} nodes) -- views encountered along the way:")
    path_node_assets = {frozenset(tree.assets_under(n)) for n in path}
    on_path_views = [v for v in views.views if v.node_assets in path_node_assets]
    if not on_path_views:
        print("  (no views matched any node on this path -- this asset's weight moved due to")
        print("   a view attached to its SIBLING branch at some split, not its own side)")
    for v in sorted(on_path_views, key=lambda v: -v.tilt_strength):
        print(
            f"  node_assets({len(v.node_assets)})={sorted(v.node_assets)}  "
            f"tilt_strength={v.tilt_strength:.3f}  confidence={v.confidence:.3f}  source={v.source}"
        )

    # Also check splits ALONG the path that have a view on the OTHER (sibling) side,
    # which indirectly shrinks biggest_mover's own branch.
    print(f"\n-- All splits along {biggest_mover}'s path, regardless of which side the view favors --")
    path_split_assets = set()
    for n in path:
        if not tree.is_leaf(n):
            left, right = tree.children(n)
            path_split_assets.add((frozenset(tree.assets_under(left)), frozenset(tree.assets_under(right))))

    decisive = None
    for s in all_splits:
        key = (frozenset(s["left_assets"]), frozenset(s["right_assets"]))
        if key in path_split_assets:
            print(
                f"  node={s['node']}  left({len(s['left_assets'])})={sorted(s['left_assets'])}  "
                f"right({len(s['right_assets'])})={sorted(s['right_assets'])}\n"
                f"           ret_left={s['ret_left_pct']:+.2f}%  ret_right={s['ret_right_pct']:+.2f}%  "
                f"diff={s['diff_pct']:+.2f}%  tilt_strength={s['tilt_strength']:.3f}  "
                f"confidence={s['confidence']:.3f}  emitted={s['emitted']}"
            )
            if decisive is None or s["tilt_strength"] > decisive["tilt_strength"]:
                decisive = s

    if decisive is not None:
        print(f"\n-- Decisive split (highest tilt_strength on {biggest_mover}'s path) --")
        n_favored = len(decisive["favored_assets"])
        window_for_favored = returns_window[list(decisive["favored_assets"])].tail(LOOKBACK)
        window_for_unfavored_assets = (
            decisive["right_assets"] if decisive["favored_assets"] == decisive["left_assets"] else decisive["left_assets"]
        )
        window_for_unfavored = returns_window[list(window_for_unfavored_assets)].tail(LOOKBACK)

        print(f"  favored side: {len(decisive['favored_assets'])} asset(s): {sorted(decisive['favored_assets'])}")
        print(f"  unfavored side: {len(window_for_unfavored_assets)} asset(s): {sorted(window_for_unfavored_assets)}")
        print(f"  raw diff (favored - unfavored trailing {LOOKBACK}d compounded return): {decisive['diff_pct']:+.2f}%")
        print(f"  tilt_strength = min({abs(decisive['diff_pct'])/100:.4f} / {TILT_SCALE_USED}, 1.0) = {decisive['tilt_strength']:.3f}")
        print(f"  confidence = {decisive['confidence']:.3f}")
        print(f"  data available for favored side over trailing {LOOKBACK} rows: "
              f"{len(window_for_favored)} rows, non-null fraction = {window_for_favored.notna().to_numpy().mean():.3f}")
        print(f"  data available for unfavored side over trailing {LOOKBACK} rows: "
              f"{len(window_for_unfavored)} rows, non-null fraction = {window_for_unfavored.notna().to_numpy().mean():.3f}")

    print()
    return {
        "label": label,
        "biggest_mover": biggest_mover,
        "biggest_change": float(diff_series[biggest_mover]),
        "decisive_split": decisive,
    }


def main():
    returns_df = pd.read_parquet(DEFAULT_RETURNS_PATH)
    windows = _make_windows(returns_df, WINDOW_SIZE)

    results = []
    # Extreme windows.
    results.append(investigate_window("window 8 (extreme, ~0.958)", windows[8]))
    results.append(investigate_window("window 9 (extreme, ~0.955)", windows[9]))
    # Normal window for comparison.
    results.append(investigate_window("window 6 (normal, ~0.292)", windows[6]))

    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    for r in results:
        print(f"{r['label']}: biggest mover = {r['biggest_mover']} (change {r['biggest_change']:+.4f})")


if __name__ == "__main__":
    main()
