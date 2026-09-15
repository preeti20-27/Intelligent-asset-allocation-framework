"""Sweep HRP allocation window sizes to find the minimum history needed for stable weights.

For a range of candidate window sizes, this slides a window across a long
synthetic returns series in fixed-size steps, calls ``allocate()`` on each
windowed slice, and measures how much consecutive windows' weights differ.
Small windows tend to produce noisy, unstable correlation/covariance
estimates -- and therefore unstable weights -- as the window slides one
step forward. Larger windows average over more data and stabilize, but
with diminishing returns past some point. This script identifies that
"knee point" and recommends it as a practical minimum lookback window.

Usage:
    python scripts/window_stability_sweep.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: never try to open a GUI window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from allocator.hrp import allocate  # noqa: E402

# Deliberately not imported from validate_against_reference.py: that module
# imports pypfopt at the top level, which pulls in cvxpy and has its own
# separate (and sometimes conflicting) numpy/scipy version requirements.
# This generator is kept logically identical to the one there so results
# stay comparable, without coupling this script's environment to pypfopt's.
def generate_sample_returns(n_assets=10, n_days=750, seed=42) -> pd.DataFrame:
    """Synthetic multi-asset daily returns with a factor-model correlation structure."""
    rng = np.random.default_rng(seed)
    tickers = [f"ASSET_{i}" for i in range(n_assets)]

    n_factors = 4
    factor_loadings = rng.normal(0, 1, size=(n_assets, n_factors))
    factors = rng.normal(0, 0.01, size=(n_days, n_factors))
    idio_std = rng.uniform(0.003, 0.03, size=n_assets)  # varied volatilities
    idio = rng.normal(0, 1, size=(n_days, n_assets)) * idio_std

    returns = factors @ factor_loadings.T + idio
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    return pd.DataFrame(returns, index=dates, columns=tickers)


WINDOW_SIZES = [20, 30, 40, 60, 80, 100, 120, 150, 200, 252]
STEP = 10
KNEE_THRESHOLD = 0.05  # a window size "flattens out" once further increases
# improve the stability metric by less than this fraction, and keep doing so.


def sliding_window_weights(returns_df: pd.DataFrame, window_size: int, step: int) -> list[pd.Series]:
    """Run allocate() on every ``window_size``-day slice, stepping by ``step`` days."""
    n_days = len(returns_df)
    weights_by_window = []
    start = 0
    while start + window_size <= n_days:
        window_slice = returns_df.iloc[start : start + window_size]
        weights_by_window.append(allocate(window_slice))
        start += step
    return weights_by_window


def stability_metrics(weights_by_window: list[pd.Series]) -> tuple[float, float]:
    """Mean and max, over consecutive window pairs, of the summed absolute weight change.

    For each pair of consecutive windows (window i and window i+1, which
    overlap by ``window_size - step`` days), sum |w_i - w_{i+1}| across all
    assets. A larger value means weights swing around more as the window
    slides forward one step -- i.e. less stable.

    Both the mean (typical-case stability) and the max (worst-case
    stability) are returned: HRP's recursive bisection depends on a
    discrete clustering tree, so a handful of steps can flip a cluster
    split and swing much harder than the average step. The mean alone can
    look deceptively flat while the worst case is still large -- the max
    surfaces that.
    """
    diffs = [
        (prev_w - next_w).abs().sum()
        for prev_w, next_w in zip(weights_by_window[:-1], weights_by_window[1:])
    ]
    return float(np.mean(diffs)), float(np.max(diffs))


def find_knee_point(window_sizes: list[int], metrics: list[float], threshold: float) -> int:
    """Smallest window size beyond which no larger window offers >threshold more gain.

    HRP's weights are a discrete function of the clustering tree, so the
    stability metric vs. window size curve is noisy, not strictly
    decreasing (a slightly larger window can occasionally flip a cluster
    split and become *less* stable than a smaller one). A plain step-to-step
    comparison is fooled by this: a noisy uptick between two adjacent window
    sizes reads as "no improvement," which trivially looks "flat" even
    though the curve isn't actually done improving.

    Instead, for each candidate window size, compare its metric to the best
    (lowest) metric achieved by *any* larger window size still in the
    sweep. That "remaining gain available" shrinks (not necessarily
    monotonically, but reliably) as window size grows, and a single later
    uptick can't masquerade as flattening because it's never the minimum of
    the remaining suffix. The knee is the smallest window size whose
    remaining gain available is below ``threshold``.
    """
    n = len(window_sizes)
    for i in range(n):
        future_metrics = metrics[i + 1 :]
        if not future_metrics:
            return window_sizes[i]  # last window in the sweep
        best_future = min(future_metrics)
        remaining_gain = (metrics[i] - best_future) / metrics[i] if metrics[i] != 0 else 0.0
        if remaining_gain < threshold:
            return window_sizes[i]

    return window_sizes[-1]  # never flattens within the sweep range; be conservative


def main():
    returns_df = generate_sample_returns(n_assets=10, n_days=1000, seed=42)
    print(f"Synthetic returns series: {returns_df.shape[0]} days, {returns_df.shape[1]} assets\n")

    metrics = []
    max_metrics = []
    for window_size in WINDOW_SIZES:
        weights_by_window = sliding_window_weights(returns_df, window_size, STEP)
        mean_metric, max_metric = stability_metrics(weights_by_window)
        metrics.append(mean_metric)
        max_metrics.append(max_metric)
        print(
            f"window={window_size:>4}  windows_evaluated={len(weights_by_window):>3}  "
            f"mean_stability_metric={mean_metric:.6f}  max_stability_metric={max_metric:.6f}"
        )

    print()
    print(
        f"{'window_size':>12}  {'mean_metric':>12}  {'max_metric':>12}  "
        f"{'step_over_step_change':>22}"
    )
    print("-" * 65)
    print(f"{WINDOW_SIZES[0]:>12}  {metrics[0]:>12.6f}  {max_metrics[0]:>12.6f}  {'--':>22}")
    for i in range(1, len(WINDOW_SIZES)):
        step_change = (metrics[i - 1] - metrics[i]) / metrics[i - 1] if metrics[i - 1] != 0 else 0.0
        print(
            f"{WINDOW_SIZES[i]:>12}  {metrics[i]:>12.6f}  {max_metrics[i]:>12.6f}  "
            f"{step_change:>21.1%}"
        )
    print(
        "(step_over_step_change is computed on the mean metric, shown for\n"
        " visibility only; because HRP's tree structure can shift discretely,\n"
        " this curve is noisy rather than strictly decreasing, so the\n"
        " recommendation below uses a more robust criterion -- see\n"
        " find_knee_point(). max_metric is the worst single consecutive-window\n"
        " swing seen at that window size -- it stays large even at bigger\n"
        " windows because a discrete cluster-split flip can happen at any size;\n"
        " use it to judge worst-case, not just typical-case, weight jitter.)"
    )

    knee = find_knee_point(WINDOW_SIZES, metrics, KNEE_THRESHOLD)

    print()
    print("=" * 60)
    print(f"Recommended minimum window size: {knee} trading days")
    print("=" * 60)
    knee_idx = WINDOW_SIZES.index(knee)
    print(
        f"Rationale: this is the smallest window size in the sweep for which no\n"
        f"larger window tested achieves a MEAN stability metric more than\n"
        f"{KNEE_THRESHOLD:.0%} lower. In other words, trying a bigger window than this\n"
        f"from here on offers no meaningful further improvement to *typical-case*\n"
        f"stability. Below this size, weights are still meaningfully sensitive to\n"
        f"exactly which {STEP}-day slice of history is used.\n"
        f"\n"
        f"Caveat -- worst case does not improve the same way: at window={knee}, the\n"
        f"max (worst-case) metric is {max_metrics[knee_idx]:.3f}, and it never drops much\n"
        f"below ~{min(max_metrics):.2f} anywhere in the sweep, even at window=252. This is\n"
        f"expected: HRP's recursive bisection depends on a discrete clustering\n"
        f"tree, so an occasional 10-day step can flip a cluster split and cause a\n"
        f"large one-off weight swing regardless of window size. {knee} trading days\n"
        f"is a sound minimum for typical-case stability, but does not bound\n"
        f"worst-case weight jitter -- callers sensitive to that should smooth or\n"
        f"rate-limit rebalancing trades downstream of allocate(), not rely on a\n"
        f"larger window alone."
    )

    output_dir = SCRIPT_DIR / "output"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "window_stability.png"

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(WINDOW_SIZES, metrics, marker="o", color="#2b6cb0", label="mean")
    ax.plot(WINDOW_SIZES, max_metrics, marker="s", color="#dd6b20", label="max (worst-case)")
    ax.axvline(knee, color="#c53030", linestyle="--", label=f"recommended min = {knee}")
    ax.set_xlabel("Window size (trading days)")
    ax.set_ylabel("Stability metric\n(summed |Δweight| between consecutive windows)")
    ax.set_title("HRP weight stability vs. lookback window size")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"\nPlot saved to {output_path}")


if __name__ == "__main__":
    main()
