"""Manual sanity-check demo: plain HRP vs. VI-HRP tilted by momentum views.

Loads real returns data, builds a ClusterTree, generates momentum views
via views.momentum_generator.generate_momentum_views(), runs both
allocators.hrp.allocate() (baseline) and allocators.vihrp.allocate() with
those views, and prints a side-by-side comparison of the two weight
vectors plus which assets moved the most and in which direction.

Not a formal test -- for eyeballing whether the momentum tilt does
something sensible on real data. See tests/unit/test_momentum_generator.py
and tests/unit/test_vihrp.py for the actual correctness tests.

Usage:
    python scripts/run_vihrp_momentum_demo.py
    python scripts/run_vihrp_momentum_demo.py --lookback 20 --window 252
"""

from __future__ import annotations

import argparse
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
from views.momentum_generator import generate_momentum_views  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--window",
        type=int,
        default=252,
        help="Trailing rows of returns.parquet to use for both allocation and the momentum signal (default 252).",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=60,
        help="Momentum trailing-return lookback in trading days, passed to generate_momentum_views (default 60).",
    )
    args = parser.parse_args()

    returns_df = pd.read_parquet(DEFAULT_RETURNS_PATH).tail(args.window)
    print(f"Using the trailing {len(returns_df)} rows of returns.parquet "
          f"({returns_df.index[0].date()} to {returns_df.index[-1].date()}).")

    tree = ClusterTree.build(returns_df.corr())
    views = generate_momentum_views(returns_df, tree, lookback=args.lookback)
    print(f"Generated {len(views.views)} momentum view(s) (lookback={args.lookback} days).\n")

    baseline = hrp_allocate(returns_df).sort_index()
    tilted = vihrp_allocate(returns_df, views=views).sort_index()

    comparison = pd.DataFrame({"hrp_baseline": baseline, "vihrp_momentum": tilted})
    comparison["abs_diff"] = (comparison["vihrp_momentum"] - comparison["hrp_baseline"]).abs()
    comparison["signed_diff"] = comparison["vihrp_momentum"] - comparison["hrp_baseline"]

    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print("=" * 68)
    print("Weight comparison: plain HRP (baseline) vs. VI-HRP + momentum views")
    print("=" * 68)
    print(comparison.sort_values("abs_diff", ascending=False).to_string())
    print("-" * 68)
    print(f"Sum - baseline: {baseline.sum():.6f}   Sum - tilted: {tilted.sum():.6f}")
    print()

    movers = comparison.sort_values("abs_diff", ascending=False)
    print("Biggest movers (VI-HRP vs. plain HRP):")
    for ticker, row in movers.head(5).iterrows():
        direction = "UP" if row["signed_diff"] > 0 else "DOWN"
        print(
            f"  {ticker:<12} {direction:<5} "
            f"{row['hrp_baseline']:.4f} -> {row['vihrp_momentum']:.4f} "
            f"(diff {row['signed_diff']:+.4f})"
        )

    print("\nActive momentum views, by tilt strength:")
    for view in sorted(views.views, key=lambda v: -v.tilt_strength):
        assets_str = ", ".join(sorted(view.node_assets))
        print(
            f"  tilt={view.tilt_strength:.3f} conf={view.confidence:.3f} "
            f"-> favors [{assets_str}]"
        )


if __name__ == "__main__":
    main()
