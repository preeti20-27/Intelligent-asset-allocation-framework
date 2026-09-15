"""Tests for allocator.hrp -- plain Hierarchical Risk Parity."""

import numpy as np
import pandas as pd
import pytest

from allocator.hrp import allocate, compute_returns


def _make_synthetic_returns(n_assets=8, n_days=500, seed=42) -> pd.DataFrame:
    """Generate correlated synthetic daily returns for n_assets tickers."""
    rng = np.random.default_rng(seed)
    tickers = [f"ASSET_{i}" for i in range(n_assets)]

    # Build a random covariance structure via a factor model so assets
    # are genuinely correlated (not i.i.d.), which is a more realistic
    # stress test for the clustering step than pure noise.
    n_factors = 3
    factor_loadings = rng.normal(0, 1, size=(n_assets, n_factors))
    factors = rng.normal(0, 0.01, size=(n_days, n_factors))
    idio = rng.normal(0, 0.005, size=(n_days, n_assets))
    returns = factors @ factor_loadings.T + idio

    return pd.DataFrame(returns, columns=tickers)


def test_weights_sum_to_one():
    returns_df = _make_synthetic_returns()
    weights = allocate(returns_df)
    assert weights.sum() == pytest.approx(1.0, abs=1e-9)


def test_weights_are_nonnegative_and_indexed_by_ticker():
    returns_df = _make_synthetic_returns()
    weights = allocate(returns_df)
    assert set(weights.index) == set(returns_df.columns)
    assert (weights >= 0).all()


def test_allocate_is_deterministic():
    returns_df = _make_synthetic_returns()
    weights_1 = allocate(returns_df)
    weights_2 = allocate(returns_df)
    pd.testing.assert_series_equal(weights_1.sort_index(), weights_2.sort_index())


def test_allocate_invariant_to_column_order():
    returns_df = _make_synthetic_returns()
    shuffled_columns = list(returns_df.columns)
    rng = np.random.default_rng(7)
    rng.shuffle(shuffled_columns)
    shuffled_returns_df = returns_df[shuffled_columns]

    weights_original = allocate(returns_df).sort_index()
    weights_shuffled = allocate(shuffled_returns_df).sort_index()

    pd.testing.assert_series_equal(weights_original, weights_shuffled, atol=1e-10)


def test_lower_volatility_asset_gets_higher_weight():
    rng = np.random.default_rng(123)
    n_days = 500

    low_vol = rng.normal(0, 0.002, size=n_days)
    high_vol = rng.normal(0, 0.05, size=n_days)

    returns_df = pd.DataFrame({"LOW_VOL": low_vol, "HIGH_VOL": high_vol})
    weights = allocate(returns_df)

    assert weights["LOW_VOL"] > weights["HIGH_VOL"]


def test_lower_volatility_asset_gets_higher_weight_within_larger_universe():
    # Same volatility asymmetry, but embedded among other, uncorrelated
    # assets so the recursive bisection has to route through several
    # cluster splits before reaching the low/high vol pair.
    rng = np.random.default_rng(99)
    n_days = 500

    data = {
        "LOW_VOL": rng.normal(0, 0.002, size=n_days),
        "HIGH_VOL": rng.normal(0, 0.05, size=n_days),
        "MID_A": rng.normal(0, 0.015, size=n_days),
        "MID_B": rng.normal(0, 0.015, size=n_days),
    }
    returns_df = pd.DataFrame(data)
    weights = allocate(returns_df)

    assert weights["LOW_VOL"] > weights["HIGH_VOL"]


def test_compute_returns_is_simple_not_log():
    prices = pd.DataFrame({"A": [100.0, 110.0, 121.0]})
    returns = compute_returns(prices)
    # Simple return: 110/100 - 1 = 0.10, 121/110 - 1 = 0.10
    assert returns["A"].iloc[0] == pytest.approx(0.10)
    assert returns["A"].iloc[1] == pytest.approx(0.10)


def _make_sweep_style_returns(n_assets=10, n_days=1000, seed=42) -> pd.DataFrame:
    """Matches generate_sample_returns() in scripts/window_stability_sweep.py.

    Duplicated (not imported) so this test doesn't couple to the scripts/
    directory; kept parameter-for-parameter identical so the empirical
    tolerance below (derived from actually running that sweep) stays valid.
    """
    rng = np.random.default_rng(seed)
    tickers = [f"ASSET_{i}" for i in range(n_assets)]

    n_factors = 4
    factor_loadings = rng.normal(0, 1, size=(n_assets, n_factors))
    factors = rng.normal(0, 0.01, size=(n_days, n_factors))
    idio_std = rng.uniform(0.003, 0.03, size=n_assets)
    idio = rng.normal(0, 1, size=(n_days, n_assets)) * idio_std

    returns = factors @ factor_loadings.T + idio
    return pd.DataFrame(returns, columns=tickers)


def test_weights_stable_across_overlapping_windows_at_recommended_min_size():
    # Regression test pinned to scripts/window_stability_sweep.py's findings:
    # sweeping window sizes [20..252] with a 10-day step on this same
    # generator (seed=42, 10 assets, 1000 days), the stability metric (mean
    # summed |delta weight| between consecutive 10-day-shifted windows)
    # flattens out starting at window_size=200 -- no larger window tested
    # improved on it by more than 5%. That makes 200 trading days the
    # recommended minimum lookback for allocate().
    #
    # Tolerance rationale: across the whole 1000-day sweep at window=200,
    # per-pair diffs ranged ~0.016 to ~0.52 (mean ~0.177) -- HRP's recursive
    # bisection depends on a discrete clustering tree, so a handful of
    # 10-day steps happen to flip a cluster split and swing much harder than
    # the average step. The specific overlapping pair tested here (days
    # 0-199 vs. days 10-209) is a "typical" step, not one of those outliers,
    # and empirically differs by ~0.075. The tolerance below (0.15, ~2x that
    # observed value) is meant to catch a real behavioral regression in
    # allocate() -- not to bound HRP's inherent window-to-window jitter,
    # which this single deterministic pair does not exercise.
    returns_df = _make_sweep_style_returns()
    window_size = 200
    step = 10

    weights_a = allocate(returns_df.iloc[0:window_size])
    weights_b = allocate(returns_df.iloc[step : step + window_size])

    diff = (weights_a - weights_b).abs().sum()
    assert diff < 0.15
