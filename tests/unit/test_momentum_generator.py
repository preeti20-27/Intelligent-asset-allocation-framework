"""Tests for views.momentum_generator.generate_momentum_views."""

import numpy as np
import pandas as pd
import pytest

from allocators.vihrp import allocate as vihrp_allocate
from tree.cluster_tree import ClusterTree
from views.momentum_generator import TILT_SCALE, generate_momentum_views

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


def test_default_tilt_scale_is_010_and_matches_explicit_call():
    """Regression check for the tilt_scale refactor: the default value
    must still be 0.10, and omitting tilt_scale must produce output
    identical to passing tilt_scale=0.10 explicitly -- i.e. exposing it as
    a parameter didn't change default behavior.
    """
    assert TILT_SCALE == 0.10

    returns_df = _load_real_returns_slice()
    tree = ClusterTree.build(returns_df.corr())

    views_default = generate_momentum_views(returns_df, tree, lookback=60)
    views_explicit = generate_momentum_views(returns_df, tree, lookback=60, tilt_scale=0.10)

    assert len(views_default.views) == len(views_explicit.views)
    key = lambda v: sorted(v.node_assets)  # noqa: E731
    for v1, v2 in zip(sorted(views_default.views, key=key), sorted(views_explicit.views, key=key)):
        assert v1.node_assets == v2.node_assets
        assert v1.tilt_strength == v2.tilt_strength
        assert v1.confidence == v2.confidence
        assert v1.source == v2.source


def test_default_tilt_scale_matches_hand_computed_reference_value():
    """Independent, hand-computed check of the exact default-scale
    formula (tilt_strength = min(|diff| / 0.10, 1.0)) for one specific
    split -- not just internal consistency between two calls, an actual
    recomputation from raw returns using a separate code path.
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
    expected_tilt_strength = min(abs(expected_diff) / 0.10, 1.0)
    expected_favored_assets = frozenset(left_assets if expected_diff > 0 else right_assets)

    views = generate_momentum_views(returns_df, tree, lookback=lookback)
    matching = [v for v in views.views if v.node_assets == expected_favored_assets]

    if expected_tilt_strength < 0.01:
        assert matching == []  # below MIN_TILT_STRENGTH_TO_EMIT: correctly skipped
    else:
        assert len(matching) == 1
        assert matching[0].tilt_strength == pytest.approx(expected_tilt_strength, rel=1e-9)


def test_generate_momentum_views_runs_and_returns_nonempty_viewset_on_real_data():
    returns_df = _load_real_returns_slice()
    tree = ClusterTree.build(returns_df.corr())

    views = generate_momentum_views(returns_df, tree, lookback=60)

    assert len(views.views) >= 1


def test_generated_views_are_all_consumable_by_vihrp_and_shift_weights():
    # Every generated view's node_assets must correspond to an actual tree
    # node -- i.e. vihrp.allocate() must not silently ignore all of them.
    # Proven behaviorally: applying the generated ViewSet must change the
    # resulting weights relative to the zero-view baseline.
    returns_df = _load_real_returns_slice()
    tree = ClusterTree.build(returns_df.corr())
    views = generate_momentum_views(returns_df, tree, lookback=60)

    assert len(views.views) >= 1  # otherwise the "shifts weights" check below is vacuous

    baseline = vihrp_allocate(returns_df)
    tilted = vihrp_allocate(returns_df, views=views)

    assert tilted.sum() == pytest.approx(1.0, abs=1e-9)
    assert (baseline - tilted).abs().max() > 1e-6


def test_every_view_node_assets_matches_a_real_tree_node_directly():
    # A stronger, structural version of the behavioral check above: walk
    # the same tree independently and confirm each generated view's
    # node_assets exactly equals some real node's assets_under().
    returns_df = _load_real_returns_slice()
    tree = ClusterTree.build(returns_df.corr())
    views = generate_momentum_views(returns_df, tree, lookback=60)

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


@pytest.mark.parametrize("lookback", [20, 60, 126])
def test_tilt_strength_and_confidence_always_within_valid_ranges(lookback):
    returns_df = _load_real_returns_slice(n_days=400)
    tree = ClusterTree.build(returns_df.corr())

    views = generate_momentum_views(returns_df, tree, lookback=lookback)

    for view in views.views:
        assert -1.0 <= view.tilt_strength <= 1.0
        assert 0.0 <= view.confidence <= 1.0


@pytest.mark.parametrize("end_date", ["2018-06-01", "2020-12-31", "2023-03-15", None])
def test_tilt_strength_and_confidence_valid_across_time_slices(end_date):
    returns_df = _load_real_returns_slice(n_days=300, end=end_date)
    tree = ClusterTree.build(returns_df.corr())

    views = generate_momentum_views(returns_df, tree, lookback=60)

    for view in views.views:
        assert -1.0 <= view.tilt_strength <= 1.0
        assert 0.0 <= view.confidence <= 1.0


def test_confidence_reflects_data_availability_not_just_signal_magnitude():
    # LTGILTBEES trades sparsely for years after its 2016-07-25 listing
    # (discovered while building the backtester) -- a window drawn from
    # that period should produce a below-1.0 confidence somewhere,
    # confirming confidence tracks real data gaps, not just momentum size.
    returns_df = pd.read_parquet(REAL_RETURNS_PATH).loc["2017-06-01":"2017-09-01"]
    tree = ClusterTree.build(returns_df.corr())

    views = generate_momentum_views(returns_df, tree, lookback=60)

    assert any(view.confidence < 1.0 for view in views.views)


def test_reproducibility_same_input_same_lookback_gives_identical_viewset():
    returns_df = _load_real_returns_slice()
    tree = ClusterTree.build(returns_df.corr())

    views_1 = generate_momentum_views(returns_df, tree, lookback=60)
    views_2 = generate_momentum_views(returns_df, tree, lookback=60)

    assert len(views_1.views) == len(views_2.views)

    key = lambda v: sorted(v.node_assets)  # noqa: E731
    for v1, v2 in zip(sorted(views_1.views, key=key), sorted(views_2.views, key=key)):
        assert v1.node_assets == v2.node_assets
        assert v1.tilt_strength == pytest.approx(v2.tilt_strength, abs=1e-15)
        assert v1.confidence == pytest.approx(v2.confidence, abs=1e-15)
        assert v1.source == v2.source


def test_favored_side_is_the_higher_momentum_side():
    # Direct check of the sign convention: the view attached to a split
    # should always be the side with the higher trailing compounded
    # return, per generate_momentum_views' documented contract.
    returns_df = _load_real_returns_slice()
    tree = ClusterTree.build(returns_df.corr())
    views = generate_momentum_views(returns_df, tree, lookback=60)

    def trailing_return(assets):
        window = returns_df[list(assets)].tail(60)
        return float(((1.0 + window).prod() - 1.0).mean())

    def find_parent_children(node, target_assets):
        if tree.is_leaf(node):
            return None
        left, right = tree.children(node)
        left_assets = frozenset(tree.assets_under(left))
        right_assets = frozenset(tree.assets_under(right))
        if left_assets == target_assets or right_assets == target_assets:
            return left_assets, right_assets
        return find_parent_children(left, target_assets) or find_parent_children(
            right, target_assets
        )

    checked_any = False
    for view in views.views:
        siblings = find_parent_children(tree.root, view.node_assets)
        assert siblings is not None
        left_assets, right_assets = siblings
        other_assets = right_assets if left_assets == view.node_assets else left_assets

        favored_return = trailing_return(view.node_assets)
        other_return = trailing_return(other_assets)
        assert favored_return >= other_return
        checked_any = True

    assert checked_any
