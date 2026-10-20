"""Tests for allocators.hcaa -- Hierarchical Clustering-based Asset Allocation.

The property that defines HCAA (and that these tests pin down) is that its
splits are naive -- 50/50 at every node of the tree, independent of any
variance estimate -- unlike HRP/VI-HRP's inverse-variance splits.
"""

import numpy as np
import pandas as pd
import pytest

from allocators.hcaa import allocate
from allocators.herc import cut_tree
from allocators.hrp import allocate as hrp_allocate
from allocators.vihrp import allocate as vihrp_allocate
from tree.cluster_tree import ClusterTree


def _make_synthetic_returns(n_assets=8, n_days=500, seed=42) -> pd.DataFrame:
    """Same factor-model generator used in tests/unit/test_hrp.py, for consistency."""
    rng = np.random.default_rng(seed)
    tickers = [f"ASSET_{i}" for i in range(n_assets)]

    n_factors = 3
    factor_loadings = rng.normal(0, 1, size=(n_assets, n_factors))
    factors = rng.normal(0, 0.01, size=(n_days, n_factors))
    idio = rng.normal(0, 0.005, size=(n_days, n_assets))
    returns = factors @ factor_loadings.T + idio

    return pd.DataFrame(returns, columns=tickers)


def _make_low_vs_high_vol_blocks(n_days=1000, seed=1) -> pd.DataFrame:
    """Two uncorrelated 2-asset blocks with a 10x volatility gap between them."""
    rng = np.random.default_rng(seed)
    f_low = rng.normal(0, 0.002, n_days)
    f_high = rng.normal(0, 0.02, n_days)
    return pd.DataFrame(
        {
            "LOW_0": f_low + rng.normal(0, 0.0005, n_days),
            "LOW_1": f_low + rng.normal(0, 0.0005, n_days),
            "HIGH_0": f_high + rng.normal(0, 0.005, n_days),
            "HIGH_1": f_high + rng.normal(0, 0.005, n_days),
        }
    )


def test_weights_sum_to_one_nonnegative_and_indexed_by_ticker():
    returns_df = _make_synthetic_returns()
    weights = allocate(returns_df)
    assert weights.sum() == pytest.approx(1.0, abs=1e-12)
    assert (weights > 0).all()
    assert list(weights.index) == list(returns_df.columns)


def test_allocate_is_deterministic():
    returns_df = _make_synthetic_returns()
    pd.testing.assert_series_equal(allocate(returns_df), allocate(returns_df))


def test_rejects_asset_with_too_much_missing_data():
    returns_df = _make_synthetic_returns()
    returns_df.iloc[:100, 3] = np.nan
    with pytest.raises(ValueError, match="ASSET_3"):
        allocate(returns_df)


def test_every_split_is_fifty_fifty_down_to_leaves():
    # With every asset its own cluster, each asset's weight must be exactly
    # 0.5 ** (its depth in the tree).
    returns_df = _make_synthetic_returns()
    weights = allocate(returns_df, n_clusters=len(returns_df.columns))
    tree = ClusterTree.build(returns_df.corr())

    def check(node, mass):
        if tree.is_leaf(node):
            assert weights[node] == pytest.approx(mass, abs=1e-15)
            return
        for child in tree.children(node):
            check(child, mass / 2)

    check(tree.root, 1.0)


def test_equal_weight_within_each_cluster_and_halving_above():
    returns_df = _make_synthetic_returns()
    weights = allocate(returns_df, n_clusters=3)
    tree = ClusterTree.build(returns_df.corr())
    for node in cut_tree(tree, 3):
        assets = tree.assets_under(node)
        assert np.allclose(weights[assets].values, weights[assets].iloc[0])
        # Each cluster's total mass is a power of 1/2 (it sits below 1+ halvings).
        exponent = -np.log2(weights[assets].sum())
        assert exponent == pytest.approx(round(exponent), abs=1e-12)


def test_weights_ignore_volatility_entirely():
    # Scaling one asset's returns by 10x changes every variance-based split
    # but not the correlation tree -- so HCAA's weights must not move.
    returns_df = _make_synthetic_returns()
    scaled = returns_df.copy()
    scaled["ASSET_0"] *= 10.0
    pd.testing.assert_series_equal(allocate(returns_df), allocate(scaled))
    assert (hrp_allocate(returns_df) - hrp_allocate(scaled)).abs().max() > 0.01


def test_naive_split_does_not_match_inverse_variance_splits():
    # Low-vol block vs. high-vol block: inverse-variance (HRP and zero-view
    # VI-HRP) puts nearly everything in the low-vol block; HCAA's root
    # split puts exactly half in each, whatever their variances.
    returns_df = _make_low_vs_high_vol_blocks()
    hcaa_w = allocate(returns_df, n_clusters=len(returns_df.columns))
    low = ["LOW_0", "LOW_1"]

    assert hcaa_w[low].sum() == pytest.approx(0.5, abs=1e-12)
    assert hrp_allocate(returns_df)[low].sum() > 0.95
    assert vihrp_allocate(returns_df)[low].sum() > 0.95


def test_real_data_produces_valid_weights():
    returns_df = pd.read_parquet("data/processed/returns.parquet").tail(252)
    weights = allocate(returns_df)
    assert weights.sum() == pytest.approx(1.0, abs=1e-12)
    assert (weights > 0).all()
    assert set(weights.index) == set(returns_df.columns)
    pd.testing.assert_series_equal(weights, allocate(returns_df))


def test_real_data_differs_from_hrp():
    returns_df = pd.read_parquet("data/processed/returns.parquet").tail(252)
    assert (allocate(returns_df) - hrp_allocate(returns_df)).abs().max() > 0.05
