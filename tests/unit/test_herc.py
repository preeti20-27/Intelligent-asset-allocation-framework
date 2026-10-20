"""Tests for allocators.herc -- Hierarchical Equal Risk Contribution."""

import numpy as np
import pandas as pd
import pytest

from allocators.herc import (
    ERC_TOLERANCE,
    allocate,
    cut_tree,
    equal_risk_contribution_weights,
    risk_contributions,
)
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


def _make_three_block_returns(n_days=1000, seed=0) -> pd.DataFrame:
    """Three clearly separated blocks of correlated assets with different vols.

    Block A (3 assets) and block B (3 assets) are each driven by their own
    factor; C is a lone, independent asset. Within block A, A0/A1 are
    near-duplicates and A2 is less correlated with them, so ERC and
    inverse-variance weighting disagree inside that block.
    """
    rng = np.random.default_rng(seed)
    fa = rng.normal(0, 0.01, n_days)
    fb = rng.normal(0, 0.02, n_days)
    data = {
        "A0": fa + rng.normal(0, 0.002, n_days),
        "A1": fa + rng.normal(0, 0.002, n_days),
        "A2": 0.4 * fa + rng.normal(0, 0.009, n_days),
        "B0": fb + rng.normal(0, 0.005, n_days),
        "B1": fb + rng.normal(0, 0.005, n_days),
        "B2": fb + rng.normal(0, 0.005, n_days),
        "C0": rng.normal(0, 0.015, n_days),
    }
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Basic contract
# ---------------------------------------------------------------------------

def test_weights_sum_to_one_nonnegative_and_indexed_by_ticker():
    returns_df = _make_synthetic_returns()
    weights = allocate(returns_df)
    assert weights.sum() == pytest.approx(1.0, abs=1e-9)
    assert (weights >= 0).all()
    assert list(weights.index) == list(returns_df.columns)


def test_allocate_is_deterministic():
    returns_df = _make_synthetic_returns()
    pd.testing.assert_series_equal(allocate(returns_df), allocate(returns_df))


def test_allocate_invariant_to_column_order():
    returns_df = _make_synthetic_returns()
    shuffled = list(returns_df.columns)
    np.random.default_rng(7).shuffle(shuffled)
    pd.testing.assert_series_equal(
        allocate(returns_df).sort_index(), allocate(returns_df[shuffled]).sort_index(), atol=1e-8
    )


def test_rejects_asset_with_too_much_missing_data():
    returns_df = _make_synthetic_returns()
    returns_df.iloc[:100, 0] = np.nan  # 80% non-null < 90% threshold
    with pytest.raises(ValueError, match="ASSET_0"):
        allocate(returns_df)


# ---------------------------------------------------------------------------
# ERC: risk contributions are actually equalized
# ---------------------------------------------------------------------------

def test_erc_solver_equalizes_risk_contributions():
    rng = np.random.default_rng(3)
    for n in (2, 3, 5, 10):
        a = rng.normal(size=(n, n))
        cov = a @ a.T + 0.1 * np.eye(n)
        w = equal_risk_contribution_weights(cov)
        rc = risk_contributions(w, cov)
        assert w.sum() == pytest.approx(1.0, abs=1e-12)
        assert (w > 0).all()
        np.testing.assert_allclose(rc, 1.0 / n, atol=ERC_TOLERANCE)


def test_erc_differs_from_inverse_variance_when_correlations_differ():
    # Equal vols, but assets 0 and 1 are highly correlated with each other
    # while 2 is independent: inverse-variance gives 1/3 each, ERC must
    # down-weight the correlated pair.
    cov = np.array([[1.0, 0.9, 0.0], [0.9, 1.0, 0.0], [0.0, 0.0, 1.0]])
    w = equal_risk_contribution_weights(cov)
    assert w[2] > w[0] + 0.1
    assert w[0] == pytest.approx(w[1], abs=1e-9)


def test_erc_matches_inverse_volatility_for_uncorrelated_assets():
    vols = np.array([0.1, 0.2, 0.4])
    w = equal_risk_contribution_weights(np.diag(vols**2))
    expected = (1 / vols) / (1 / vols).sum()
    np.testing.assert_allclose(w, expected, atol=1e-8)


def test_within_cluster_weights_have_equal_risk_contributions():
    # Pull HERC's final weights apart cluster by cluster and confirm that,
    # renormalized within each cluster, they are an ERC portfolio of that
    # cluster's own covariance submatrix.
    returns_df = _make_three_block_returns()
    weights = allocate(returns_df, n_clusters=3)
    tree = ClusterTree.build(returns_df.corr())
    cov = returns_df.cov()

    clusters = [tree.assets_under(node) for node in cut_tree(tree, 3)]
    assert sorted(sorted(c) for c in clusters) == [["A0", "A1", "A2"], ["B0", "B1", "B2"], ["C0"]]

    for assets in clusters:
        w = weights[assets].values / weights[assets].sum()
        rc = risk_contributions(w, cov.loc[assets, assets].values)
        np.testing.assert_allclose(rc, 1.0 / len(assets), atol=ERC_TOLERANCE)


def test_within_cluster_split_is_not_inverse_variance():
    returns_df = _make_three_block_returns()
    weights = allocate(returns_df, n_clusters=3)
    block = ["A0", "A1", "A2"]
    w = weights[block] / weights[block].sum()

    ivp = 1.0 / returns_df[block].var()
    ivp /= ivp.sum()
    block_cov = returns_df[block].cov().values

    # A2 is less correlated with the near-duplicate pair A0/A1, so
    # inverse-variance weighting leaves its risk contribution well below
    # 1/3; HERC's ERC step corrects that by giving A2 more weight.
    ivp_rc = risk_contributions(ivp.values, block_cov)
    herc_rc = risk_contributions(w.values, block_cov)
    assert ivp_rc[2] < 1 / 3 - 0.03
    np.testing.assert_allclose(herc_rc, 1 / 3, atol=ERC_TOLERANCE)
    assert w["A2"] > ivp["A2"] + 0.02


# ---------------------------------------------------------------------------
# Tree cut and relationship to the other tree allocators
# ---------------------------------------------------------------------------

def test_cut_tree_yields_requested_number_of_disjoint_covering_clusters():
    returns_df = _make_synthetic_returns()
    tree = ClusterTree.build(returns_df.corr())
    for k in range(1, len(returns_df.columns) + 1):
        clusters = [tree.assets_under(node) for node in cut_tree(tree, k)]
        assert len(clusters) == k
        flat = [a for c in clusters for a in c]
        assert sorted(flat) == sorted(returns_df.columns)


def test_one_cluster_is_pure_erc_on_whole_universe():
    returns_df = _make_synthetic_returns()
    weights = allocate(returns_df, n_clusters=1)
    rc = risk_contributions(weights.values, returns_df.cov().values)
    np.testing.assert_allclose(rc, 1.0 / len(weights), atol=ERC_TOLERANCE)


def test_singleton_clusters_reproduce_zero_view_vihrp_exactly():
    # With every asset its own cluster, ERC is trivially 1.0 and the only
    # thing left is the inverse-variance tree walk -- exactly VI-HRP's.
    returns_df = _make_synthetic_returns()
    herc_w = allocate(returns_df, n_clusters=len(returns_df.columns))
    vihrp_w = vihrp_allocate(returns_df)
    pd.testing.assert_series_equal(herc_w, vihrp_w, atol=1e-12)


def test_default_cut_differs_from_vihrp():
    returns_df = _make_three_block_returns()
    assert (allocate(returns_df) - vihrp_allocate(returns_df)).abs().max() > 0.01


# ---------------------------------------------------------------------------
# Real data
# ---------------------------------------------------------------------------

def test_real_data_produces_valid_weights():
    returns_df = pd.read_parquet("data/processed/returns.parquet").tail(252)
    weights = allocate(returns_df)
    assert weights.sum() == pytest.approx(1.0, abs=1e-9)
    assert (weights >= 0).all()
    assert set(weights.index) == set(returns_df.columns)
    pd.testing.assert_series_equal(weights, allocate(returns_df))


def test_real_data_within_cluster_erc_holds():
    returns_df = pd.read_parquet("data/processed/returns.parquet").tail(252)
    weights = allocate(returns_df)
    tree = ClusterTree.build(returns_df.corr())
    cov = returns_df.cov()
    for node in cut_tree(tree, 3):
        assets = tree.assets_under(node)
        w = weights[assets].values / weights[assets].sum()
        rc = risk_contributions(w, cov.loc[assets, assets].values)
        np.testing.assert_allclose(rc, 1.0 / len(assets), atol=ERC_TOLERANCE)
