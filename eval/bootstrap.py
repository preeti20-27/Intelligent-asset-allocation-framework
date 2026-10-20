"""Block-bootstrap inference on the difference between two Sharpe ratios.

Why a block bootstrap, and why paired
-------------------------------------
Daily strategy returns are neither independent over time (volatility
clustering, and here also weekly rebalancing that holds weights fixed for
a week) nor independent of each other (every strategy in this project
holds the same 12 assets, so their daily returns are typically
correlated 0.8-0.99). An iid bootstrap would destroy the first
dependence and understate the Sharpe ratio's sampling variance; resampling
the two series independently would destroy the second and badly OVERstate
the variance of their difference (the shared market noise cancels in the
difference only if it is kept paired).

So this module uses the circular block bootstrap (Politis & Romano, 1992):
resample whole blocks of ``block_size`` consecutive days, wrapping around
the end of the sample, and apply the SAME block indices to both series.
This is the resampling scheme Ledoit & Wolf (2008), "Robust performance
hypothesis testing with the Sharpe ratio", recommend for exactly this
problem.

Default block size: ``ceil(n ** (1/3))`` -- the standard asymptotically
optimal rate for block-bootstrap variance estimation (Hall, Horowitz &
Jing, 1995); ~10 days for the ~1-4-year segments used in this project,
i.e. about two rebalance periods.

Inference
---------
``sharpe_difference_test`` reports the point estimate
``delta = SR_a - SR_b``, a percentile confidence interval from the
bootstrap distribution of ``delta*``, and a two-sided p-value for
``H0: delta = 0`` obtained by centering that distribution at zero (the
"basic"/null-shift bootstrap test):
``p = mean(|delta* - delta| >= |delta|)``.
The difference is called significant at level ``alpha`` when
``p < alpha``. Sharpe ratios are computed exactly as
``eval.metrics.sharpe_ratio`` does (zero risk-free rate, ``ddof=1``,
annualized by ``sqrt(252)``).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252
DEFAULT_N_BOOT = 5000
DEFAULT_ALPHA = 0.05
DEFAULT_SEED = 12345


class SharpeDifferenceResult(NamedTuple):
    sharpe_a: float
    sharpe_b: float
    difference: float
    ci_low: float
    ci_high: float
    p_value: float
    significant: bool
    alpha: float
    block_size: int
    n_boot: int
    n_obs: int


def default_block_size(n_obs: int) -> int:
    """``ceil(n ** (1/3))``, at least 1. See the module docstring."""
    return max(1, int(np.ceil(n_obs ** (1.0 / 3.0))))


def circular_block_indices(
    n_obs: int, block_size: int, n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    """``(n_boot, n_obs)`` row indices for the circular block bootstrap.

    Each bootstrap sample is built from ``ceil(n_obs / block_size)`` blocks
    with uniformly random start positions, each block running
    ``block_size`` consecutive days (wrapping past the end back to the
    start), truncated to ``n_obs`` rows.
    """
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}.")
    n_blocks = int(np.ceil(n_obs / block_size))
    starts = rng.integers(0, n_obs, size=(n_boot, n_blocks))
    offsets = np.arange(block_size)
    indices = (starts[:, :, None] + offsets[None, None, :]) % n_obs
    return indices.reshape(n_boot, n_blocks * block_size)[:, :n_obs]


def annualized_sharpe(returns: np.ndarray, axis: int = -1) -> np.ndarray:
    """Annualized Sharpe along ``axis`` (zero risk-free rate, ``ddof=1``)."""
    mean = returns.mean(axis=axis)
    std = returns.std(axis=axis, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe = np.where(std > 1e-10, mean / std * np.sqrt(TRADING_DAYS_PER_YEAR), 0.0)
    return sharpe


def align_returns(*series: pd.Series) -> np.ndarray:
    """Inner-join return series on their index and drop rows with any NaN.

    Returns a ``(n_obs, n_series)`` array.
    """
    frame = pd.concat(series, axis=1, join="inner").dropna()
    return frame.to_numpy(dtype=float)


def equity_to_returns(equity_curve: pd.Series) -> pd.Series:
    """Daily simple returns of an equity curve (first day dropped)."""
    return equity_curve.pct_change().dropna()


def sharpe_difference_test(
    returns_a: pd.Series,
    returns_b: pd.Series,
    block_size: int | None = None,
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = DEFAULT_ALPHA,
    seed: int = DEFAULT_SEED,
) -> SharpeDifferenceResult:
    """Paired circular-block-bootstrap test of ``SR(a) - SR(b) = 0``.

    Parameters
    ----------
    returns_a, returns_b : pd.Series
        Daily simple returns (use ``equity_to_returns`` for equity
        curves). Aligned on their common index; rows where either is NaN
        are dropped.
    block_size : int, optional
        Block length in days; defaults to ``default_block_size(n_obs)``.
    n_boot : int, optional
        Number of bootstrap resamples.
    alpha : float, optional
        Two-sided level for the CI (``1 - alpha`` coverage) and the
        significance flag.
    seed : int, optional
        RNG seed -- results are deterministic for a given seed.
    """
    data = align_returns(returns_a, returns_b)
    n_obs = data.shape[0]
    if n_obs < 3:
        raise ValueError(f"Need at least 3 aligned observations, got {n_obs}.")
    if block_size is None:
        block_size = default_block_size(n_obs)

    sharpe_a, sharpe_b = annualized_sharpe(data, axis=0)
    difference = float(sharpe_a - sharpe_b)

    rng = np.random.default_rng(seed)
    indices = circular_block_indices(n_obs, block_size, n_boot, rng)
    boot_diff = annualized_sharpe(data[indices, 0], axis=1) - annualized_sharpe(data[indices, 1], axis=1)

    ci_low, ci_high = np.quantile(boot_diff, [alpha / 2.0, 1.0 - alpha / 2.0])
    p_value = float(np.mean(np.abs(boot_diff - difference) >= abs(difference)))

    return SharpeDifferenceResult(
        sharpe_a=float(sharpe_a),
        sharpe_b=float(sharpe_b),
        difference=difference,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value=p_value,
        significant=p_value < alpha,
        alpha=alpha,
        block_size=int(block_size),
        n_boot=int(n_boot),
        n_obs=int(n_obs),
    )
