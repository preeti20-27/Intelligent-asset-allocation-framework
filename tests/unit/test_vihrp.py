"""Tests for allocators.vihrp -- View-Informed HRP.

Design note (see allocators/vihrp.py's module docstring for the full
rationale): VI-HRP walks the ACTUAL clustering tree, while
allocators.hrp.allocate bisects the flat quasi-diagonalized list instead.
These are different split-generating procedures, so VI-HRP with zero
views is NOT expected to match allocators.hrp.allocate bit-for-bit, even
though both are legitimate HRP variants over the same tree/covariance.
What *is* asserted for the zero-view case is that VI-HRP still produces
valid, HRP-like weights (sum to 1, deterministic, lower-volatility assets
favored) -- this is the achievable "core correctness anchor" given that
constraint, agreed on explicitly before implementing this.
"""

import numpy as np
import pandas as pd
import pytest

from allocators.vihrp import allocate
from tree.cluster_tree import ClusterTree
from views.view import View, ViewSet


def _make_synthetic_returns(n_assets=8, n_days=500, seed=42) -> pd.DataFrame:
    """Same factor-model generator used in tests/test_hrp.py, for consistency."""
    rng = np.random.default_rng(seed)
    tickers = [f"ASSET_{i}" for i in range(n_assets)]

    n_factors = 3
    factor_loadings = rng.normal(0, 1, size=(n_assets, n_factors))
    factors = rng.normal(0, 0.01, size=(n_days, n_factors))
    idio = rng.normal(0, 0.005, size=(n_days, n_assets))
    returns = factors @ factor_loadings.T + idio

    return pd.DataFrame(returns, columns=tickers)


def _root_children_assets(returns_df: pd.DataFrame):
    """The asset sets of the root split's two children, for use as view targets."""
    tree = ClusterTree.build(returns_df.corr())
    left, right = tree.children(tree.root)
    return frozenset(tree.assets_under(left)), frozenset(tree.assets_under(right))


# ---------------------------------------------------------------------------
# Core correctness anchor: zero views produce valid, HRP-like weights.
# ---------------------------------------------------------------------------

def test_zero_views_weights_sum_to_one_and_are_nonnegative():
    returns_df = _make_synthetic_returns()
    weights = allocate(returns_df, views=None)

    assert weights.sum() == pytest.approx(1.0, abs=1e-9)
    assert (weights >= 0).all()
    assert set(weights.index) == set(returns_df.columns)


def test_none_and_empty_viewset_are_equivalent():
    returns_df = _make_synthetic_returns()
    weights_none = allocate(returns_df, views=None)
    weights_empty = allocate(returns_df, views=ViewSet([]))

    pd.testing.assert_series_equal(weights_none.sort_index(), weights_empty.sort_index())


def test_zero_views_is_deterministic():
    returns_df = _make_synthetic_returns()
    weights_1 = allocate(returns_df)
    weights_2 = allocate(returns_df)
    pd.testing.assert_series_equal(weights_1.sort_index(), weights_2.sort_index())


def test_zero_views_lower_volatility_asset_gets_higher_weight():
    # Same style of check as test_hrp.py's plain-HRP equivalent: VI-HRP's
    # zero-view path should still be recognizably HRP-like, even though
    # its exact numbers needn't match allocators.hrp.allocate.
    rng = np.random.default_rng(123)
    n_days = 500
    low_vol = rng.normal(0, 0.002, size=n_days)
    high_vol = rng.normal(0, 0.05, size=n_days)
    returns_df = pd.DataFrame({"LOW_VOL": low_vol, "HIGH_VOL": high_vol})

    weights = allocate(returns_df)
    assert weights["LOW_VOL"] > weights["HIGH_VOL"]


# ---------------------------------------------------------------------------
# View direction and confidence behavior
# ---------------------------------------------------------------------------

def test_positive_tilt_strictly_increases_targeted_side_weight():
    returns_df = _make_synthetic_returns()
    left_assets, _ = _root_children_assets(returns_df)

    baseline = allocate(returns_df)
    baseline_mass = baseline[list(left_assets)].sum()

    view = View(node_assets=left_assets, tilt_strength=0.8, confidence=1.0, source="test")
    tilted = allocate(returns_df, views=ViewSet([view]))
    tilted_mass = tilted[list(left_assets)].sum()

    assert tilted_mass > baseline_mass


def test_negative_tilt_strictly_decreases_targeted_side_weight():
    returns_df = _make_synthetic_returns()
    left_assets, _ = _root_children_assets(returns_df)

    baseline = allocate(returns_df)
    baseline_mass = baseline[list(left_assets)].sum()

    view = View(node_assets=left_assets, tilt_strength=-0.8, confidence=1.0, source="test")
    tilted = allocate(returns_df, views=ViewSet([view]))
    tilted_mass = tilted[list(left_assets)].sum()

    assert tilted_mass < baseline_mass


def test_weights_sum_to_one_regardless_of_views():
    returns_df = _make_synthetic_returns()
    left_assets, right_assets = _root_children_assets(returns_df)

    views = ViewSet(
        [
            View(node_assets=left_assets, tilt_strength=0.5, confidence=0.7, source="a"),
        ]
    )
    weights = allocate(returns_df, views=views)
    assert weights.sum() == pytest.approx(1.0, abs=1e-9)


def test_confidence_sweep_is_monotonic_with_no_discontinuous_jumps():
    returns_df = _make_synthetic_returns()
    left_assets, _ = _root_children_assets(returns_df)

    confidences = [0.0, 0.25, 0.5, 0.75, 1.0]
    masses = []
    for confidence in confidences:
        view = View(
            node_assets=left_assets, tilt_strength=0.6, confidence=confidence, source="test"
        )
        weights = allocate(returns_df, views=ViewSet([view]))
        masses.append(weights[list(left_assets)].sum())

    # Monotonically non-decreasing (positive tilt_strength).
    for earlier, later in zip(masses, masses[1:]):
        assert later > earlier - 1e-12

    # No discontinuous jumps: consecutive steps of 0.25 confidence should
    # produce comparably-sized mass changes, not one huge jump and the
    # rest flat (a rough smoothness check, not requiring exact linearity).
    steps = [later - earlier for earlier, later in zip(masses, masses[1:])]
    assert max(steps) < 3 * min(steps)

    # confidence=0 must reproduce the zero-view baseline exactly.
    baseline = allocate(returns_df)
    assert masses[0] == pytest.approx(baseline[list(left_assets)].sum(), abs=1e-12)


# ---------------------------------------------------------------------------
# Invalid / non-matching views are silently inert
# ---------------------------------------------------------------------------

def test_view_matching_no_real_node_is_silently_ignored():
    returns_df = _make_synthetic_returns()
    baseline = allocate(returns_df)

    # An arbitrary 3-asset subset essentially never exactly matches a real
    # tree node's assets_under() for an 8-asset random clustering.
    tickers = list(returns_df.columns)
    bogus_assets = frozenset(tickers[:3])
    tree = ClusterTree.build(returns_df.corr())

    def matches_any_node(node, target):
        if target == frozenset(tree.assets_under(node)):
            return True
        if tree.is_leaf(node):
            return False
        left, right = tree.children(node)
        return matches_any_node(left, target) or matches_any_node(right, target)

    assert not matches_any_node(tree.root, bogus_assets), (
        "test setup assumption violated: bogus_assets unexpectedly matches a real node"
    )

    view = View(node_assets=bogus_assets, tilt_strength=1.0, confidence=1.0, source="bogus")
    with_bogus_view = allocate(returns_df, views=ViewSet([view]))

    pd.testing.assert_series_equal(baseline.sort_index(), with_bogus_view.sort_index())


def test_view_with_zero_tilt_strength_has_no_effect():
    returns_df = _make_synthetic_returns()
    left_assets, _ = _root_children_assets(returns_df)

    baseline = allocate(returns_df)
    view = View(node_assets=left_assets, tilt_strength=0.0, confidence=1.0, source="test")
    with_view = allocate(returns_df, views=ViewSet([view]))

    pd.testing.assert_series_equal(baseline.sort_index(), with_view.sort_index())


# ---------------------------------------------------------------------------
# Real data
# ---------------------------------------------------------------------------

def test_real_data_zero_views_produces_valid_weights():
    returns_df = pd.read_parquet("data/processed/returns.parquet").tail(252)
    weights = allocate(returns_df, views=None)

    assert weights.sum() == pytest.approx(1.0, abs=1e-9)
    assert (weights >= 0).all()
    assert set(weights.index) == set(returns_df.columns)


def test_real_data_view_tilts_targeted_side_in_expected_direction():
    returns_df = pd.read_parquet("data/processed/returns.parquet").tail(252)
    left_assets, _ = _root_children_assets(returns_df)

    baseline = allocate(returns_df)
    baseline_mass = baseline[list(left_assets)].sum()

    view = View(node_assets=left_assets, tilt_strength=0.7, confidence=1.0, source="real-data-test")
    tilted = allocate(returns_df, views=ViewSet([view]))
    tilted_mass = tilted[list(left_assets)].sum()

    assert tilted_mass > baseline_mass
    assert tilted.sum() == pytest.approx(1.0, abs=1e-9)
