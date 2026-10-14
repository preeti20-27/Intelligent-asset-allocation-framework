"""Tests for views.mean_reversion_generator.generate_mean_reversion_views.

Mirrors tests/unit/test_momentum_generator.py's structure and style:
structural correctness against an independently-walked tree, not just
"it runs." Many tests here specifically check the *relationship* to
views.momentum_generator.generate_momentum_views (opposite favored side,
identical tilt_strength/confidence magnitudes), since that relationship
-- not the formula in isolation -- is the whole point of this generator.
"""

import numpy as np
import pandas as pd
import pytest

from allocators.vihrp import allocate as vihrp_allocate
from tree.cluster_tree import ClusterTree
from views.generator import ViewGenerator
from views.mean_reversion_generator import generate_mean_reversion_views
from views.momentum_generator import LOOKBACK, TILT_SCALE, generate_momentum_views

REAL_RETURNS_PATH = "data/processed/returns.parquet"


def _load_real_returns_slice(n_days: int = 252, end: str | None = None) -> pd.DataFrame:
    returns_df = pd.read_parquet(REAL_RETURNS_PATH)
    if end is not None:
        returns_df = returns_df.loc[:end]
    return returns_df.tail(n_days)


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


def _find_parent_children(tree, node, target_assets):
    if tree.is_leaf(node):
        return None
    left, right = tree.children(node)
    left_assets = frozenset(tree.assets_under(left))
    right_assets = frozenset(tree.assets_under(right))
    if left_assets == target_assets or right_assets == target_assets:
        return left_assets, right_assets
    return _find_parent_children(tree, left, target_assets) or _find_parent_children(
        tree, right, target_assets
    )


def test_conforms_to_view_generator_protocol():
    assert isinstance(generate_mean_reversion_views, ViewGenerator)


def test_default_lookback_and_tilt_scale_reuse_momentum_generators_calibrated_constants():
    """Deliberate design choice (see module docstring): this generator
    reuses views.momentum_generator's already-calibrated LOOKBACK/TILT_SCALE
    rather than being independently tuned, for a controlled comparison.
    This is a regression check that the reuse is real, not just claimed
    in prose.
    """
    returns_df = _load_real_returns_slice()
    tree = ClusterTree.build(returns_df.corr())

    views_default = generate_mean_reversion_views(returns_df, tree)
    views_explicit = generate_mean_reversion_views(returns_df, tree, lookback=LOOKBACK, tilt_scale=TILT_SCALE)

    assert len(views_default.views) == len(views_explicit.views)
    key = lambda v: sorted(v.node_assets)  # noqa: E731
    for v1, v2 in zip(sorted(views_default.views, key=key), sorted(views_explicit.views, key=key)):
        assert v1.node_assets == v2.node_assets
        assert v1.tilt_strength == v2.tilt_strength
        assert v1.confidence == v2.confidence


def test_favors_the_lower_return_side():
    # Direct check of the sign convention (mirror of
    # test_momentum_generator.py's test_favored_side_is_the_higher_momentum_side):
    # the view attached to a split should always be the side with the
    # LOWER trailing compounded return.
    returns_df = _load_real_returns_slice()
    tree = ClusterTree.build(returns_df.corr())
    views = generate_mean_reversion_views(returns_df, tree, lookback=60)

    def trailing_return(assets):
        window = returns_df[list(assets)].tail(60)
        return float(((1.0 + window).prod() - 1.0).mean())

    checked_any = False
    for view in views.views:
        siblings = _find_parent_children(tree, tree.root, view.node_assets)
        assert siblings is not None
        left_assets, right_assets = siblings
        other_assets = right_assets if left_assets == view.node_assets else left_assets

        favored_return = trailing_return(view.node_assets)
        other_return = trailing_return(other_assets)
        assert favored_return <= other_return
        checked_any = True

    assert checked_any


def test_favors_exactly_the_opposite_side_from_momentum_generator():
    # The core structural property this generator exists for: for every
    # split where generate_momentum_views emits a view,
    # generate_mean_reversion_views must emit a view on the SIBLING side,
    # with IDENTICAL tilt_strength and confidence (same formulas, same
    # |gap| -- only the favored side differs).
    returns_df = _load_real_returns_slice()
    tree = ClusterTree.build(returns_df.corr())
    momentum_views = generate_momentum_views(returns_df, tree)
    reversion_views = generate_mean_reversion_views(returns_df, tree)

    assert len(momentum_views.views) >= 1
    assert len(momentum_views.views) == len(reversion_views.views)

    for mv in momentum_views.views:
        siblings = _find_parent_children(tree, tree.root, mv.node_assets)
        assert siblings is not None
        other_assets = siblings[1] if siblings[0] == mv.node_assets else siblings[0]

        matches = [rv for rv in reversion_views.views if rv.node_assets == other_assets]
        assert len(matches) == 1, (
            f"expected exactly one mean-reversion view on the sibling side {other_assets}"
        )
        rv = matches[0]
        assert rv.tilt_strength == pytest.approx(mv.tilt_strength, abs=1e-12)
        assert rv.confidence == pytest.approx(mv.confidence, abs=1e-12)


def test_default_tilt_scale_matches_hand_computed_reference_value():
    """Independent, hand-computed check of the exact formula
    (tilt_strength = min(|diff| / TILT_SCALE, 1.0), favoring the LOWER
    side) for one specific split -- an actual recomputation from raw
    returns using a separate code path, not just internal consistency.
    """
    returns_df = _make_synthetic_returns()
    tree = ClusterTree.build(returns_df.corr())
    left, right = tree.children(tree.root)
    left_assets = tree.assets_under(left)
    right_assets = tree.assets_under(right)

    lookback = 60
    left_window = returns_df[left_assets].tail(lookback)
    right_window = returns_df[right_assets].tail(lookback)
    ret_left = float(((1.0 + left_window).prod() - 1.0).mean())
    ret_right = float(((1.0 + right_window).prod() - 1.0).mean())
    expected_diff = ret_left - ret_right
    expected_tilt_strength = min(abs(expected_diff) / TILT_SCALE, 1.0)
    # Mirror of the momentum generator: favor the LOWER side.
    expected_favored_assets = frozenset(right_assets if expected_diff > 0 else left_assets)

    views = generate_mean_reversion_views(returns_df, tree, lookback=lookback)
    matching = [v for v in views.views if v.node_assets == expected_favored_assets]

    if expected_tilt_strength < 0.01:
        assert matching == []
    else:
        assert len(matching) == 1
        assert matching[0].tilt_strength == pytest.approx(expected_tilt_strength, rel=1e-9)


def test_every_view_node_assets_matches_a_real_tree_node_directly():
    returns_df = _load_real_returns_slice()
    tree = ClusterTree.build(returns_df.corr())
    views = generate_mean_reversion_views(returns_df, tree, lookback=60)

    all_node_asset_sets = set()

    def collect(node):
        all_node_asset_sets.add(frozenset(tree.assets_under(node)))
        if not tree.is_leaf(node):
            left, right = tree.children(node)
            collect(left)
            collect(right)

    collect(tree.root)

    for view in views.views:
        assert view.node_assets in all_node_asset_sets


@pytest.mark.parametrize("lookback", [20, 40, 60, 126])
def test_tilt_strength_and_confidence_always_within_valid_ranges(lookback):
    returns_df = _load_real_returns_slice(n_days=400)
    tree = ClusterTree.build(returns_df.corr())

    views = generate_mean_reversion_views(returns_df, tree, lookback=lookback)

    for view in views.views:
        assert -1.0 <= view.tilt_strength <= 1.0
        assert 0.0 <= view.confidence <= 1.0


def test_generated_views_are_consumable_by_vihrp_and_shift_weights():
    returns_df = _load_real_returns_slice()
    tree = ClusterTree.build(returns_df.corr())
    views = generate_mean_reversion_views(returns_df, tree)

    assert len(views.views) >= 1

    baseline = vihrp_allocate(returns_df)
    tilted = vihrp_allocate(returns_df, views=views)

    assert tilted.sum() == pytest.approx(1.0, abs=1e-9)
    assert (baseline - tilted).abs().max() > 1e-6


def test_reproducibility_same_input_gives_identical_viewset():
    returns_df = _load_real_returns_slice()
    tree = ClusterTree.build(returns_df.corr())

    views_1 = generate_mean_reversion_views(returns_df, tree)
    views_2 = generate_mean_reversion_views(returns_df, tree)

    assert len(views_1.views) == len(views_2.views)
    key = lambda v: sorted(v.node_assets)  # noqa: E731
    for v1, v2 in zip(sorted(views_1.views, key=key), sorted(views_2.views, key=key)):
        assert v1.node_assets == v2.node_assets
        assert v1.tilt_strength == pytest.approx(v2.tilt_strength, abs=1e-15)
        assert v1.confidence == pytest.approx(v2.confidence, abs=1e-15)
