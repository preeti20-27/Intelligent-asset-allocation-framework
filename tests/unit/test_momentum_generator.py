"""Tests for views.momentum_generator.generate_momentum_views."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from allocators.vihrp import allocate as vihrp_allocate
from tree.cluster_tree import ClusterTree
from views.momentum_generator import (
    LOOKBACK,
    TILT_SCALE,
    _momentum_gap_significance,
    generate_momentum_views,
)

REAL_RETURNS_PATH = "data/processed/returns.parquet"
CALIBRATION_CSV_PATH = Path(__file__).resolve().parent.parent.parent / "scripts" / "output" / "momentum_generator_calibration.csv"


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


def test_default_tilt_scale_and_lookback_match_calibrated_constants_and_explicit_call():
    """Regression check for the tilt_scale/lookback refactor: omitting
    either parameter must produce output identical to passing the
    module's own calibrated ``TILT_SCALE``/``LOOKBACK`` explicitly -- i.e.
    exposing them as parameters didn't change default behavior.
    """
    returns_df = _load_real_returns_slice()
    tree = ClusterTree.build(returns_df.corr())

    views_default = generate_momentum_views(returns_df, tree)
    views_explicit = generate_momentum_views(returns_df, tree, lookback=LOOKBACK, tilt_scale=TILT_SCALE)

    assert len(views_default.views) == len(views_explicit.views)
    key = lambda v: sorted(v.node_assets)  # noqa: E731
    for v1, v2 in zip(sorted(views_default.views, key=key), sorted(views_explicit.views, key=key)):
        assert v1.node_assets == v2.node_assets
        assert v1.tilt_strength == v2.tilt_strength
        assert v1.confidence == v2.confidence
        assert v1.source == v2.source


def test_default_tilt_scale_matches_hand_computed_reference_value():
    """Independent, hand-computed check of the exact default-scale
    formula (tilt_strength = min(|diff| / TILT_SCALE, 1.0)) for one
    specific split -- not just internal consistency between two calls, an
    actual recomputation from raw returns using a separate code path.
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
    expected_favored_assets = frozenset(left_assets if expected_diff > 0 else right_assets)

    views = generate_momentum_views(returns_df, tree, lookback=lookback)
    matching = [v for v in views.views if v.node_assets == expected_favored_assets]

    if expected_tilt_strength < 0.01:
        assert matching == []  # below MIN_TILT_STRENGTH_TO_EMIT: correctly skipped
    else:
        assert len(matching) == 1
        assert matching[0].tilt_strength == pytest.approx(expected_tilt_strength, rel=1e-9)


def test_calibrated_constants_match_committed_grid_search_winner():
    """Ties TILT_SCALE/LOOKBACK directly to the evidence that justifies
    them: scripts/calibrate_momentum_generator.py's committed grid-search
    output must exist, and the module's constants must equal the
    (lookback, tilt_scale) row with the best IN-SAMPLE Sharpe among
    combinations with a saturation rate below 50% (ties broken by
    shallower in-sample max drawdown) -- the same selection rule the
    calibration script itself uses (see its SATURATION_RATE_CEILING and
    module docstring: the *unfiltered* in-sample-Sharpe winner saturates
    on 95.9% of rebalances, which is exactly the "pinned to the ceiling,
    not an informative signal" failure mode being fixed). Guards against
    the code's constants silently drifting away from the committed
    evidence, or being hand-edited without rerunning/recommitting the
    grid search.
    """
    assert CALIBRATION_CSV_PATH.exists(), (
        f"{CALIBRATION_CSV_PATH} is missing -- re-run "
        "scripts/calibrate_momentum_generator.py and commit its output."
    )
    grid = pd.read_csv(CALIBRATION_CSV_PATH)
    eligible = grid[grid["saturation_rate_pct"] < 50]
    ranked = eligible.sort_values(
        ["in_sample_sharpe", "in_sample_max_dd_pct"], ascending=[False, False]
    )
    best = ranked.iloc[0]

    assert int(best["lookback"]) == LOOKBACK
    assert float(best["tilt_scale"]) == pytest.approx(TILT_SCALE)


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


# ---------------------------------------------------------------------------
# _momentum_gap_significance: the new significance component of confidence
# ---------------------------------------------------------------------------

def _two_cluster_returns(n_days: int, daily_gap: float, daily_noise_std: float, seed: int) -> pd.DataFrame:
    """Synthetic two single-asset "clusters" (LEFT, RIGHT) whose daily
    return differential has mean ``daily_gap`` and std ``daily_noise_std``
    -- enough to directly control the significance test's inputs.
    """
    rng = np.random.default_rng(seed)
    right = rng.normal(0, 0.01, size=n_days)
    left = right + rng.normal(daily_gap, daily_noise_std, size=n_days)
    return pd.DataFrame({"LEFT": left, "RIGHT": right})


def test_significance_is_zero_for_an_exactly_zero_mean_differential():
    # Deterministic construction (not a random seed that could land near
    # significant by chance): an alternating +eps/-eps daily differential
    # has sample mean exactly 0 regardless of its variance, so the
    # t-statistic is exactly 0 and the two-tailed p-value is exactly 1 --
    # significance must be exactly 0.0, not just "low."
    n_days = 120
    noise = np.tile([0.01, -0.01], n_days // 2)
    right = np.zeros(n_days)
    left = right + noise
    returns_df = pd.DataFrame({"LEFT": left, "RIGHT": right})

    significance = _momentum_gap_significance(returns_df, ["LEFT"], ["RIGHT"], lookback=n_days)
    assert significance == pytest.approx(0.0, abs=1e-9)


def test_significance_is_high_for_a_large_consistent_gap():
    # A gap several times the noise std, sustained over many days, should
    # be clearly distinguishable from noise.
    returns_df = _two_cluster_returns(n_days=120, daily_gap=0.01, daily_noise_std=0.005, seed=2)
    significance = _momentum_gap_significance(returns_df, ["LEFT"], ["RIGHT"], lookback=120)
    assert significance > 0.9


def test_significance_increases_with_more_observations_of_the_same_process():
    # Same underlying daily process (same gap, same noise), just more
    # days of it: more observations should only make the estimate more
    # confident (lower standard error), not less -- so a short window
    # drawn from the same process should never be MORE significant than
    # the long window it's a prefix of.
    returns_df = _two_cluster_returns(n_days=120, daily_gap=0.003, daily_noise_std=0.01, seed=3)

    short_significance = _momentum_gap_significance(returns_df, ["LEFT"], ["RIGHT"], lookback=10)
    long_significance = _momentum_gap_significance(returns_df, ["LEFT"], ["RIGHT"], lookback=120)

    assert long_significance >= short_significance


def test_significance_matches_scipy_one_sample_ttest_directly():
    # Independent recomputation via scipy.stats.ttest_1samp (a different
    # code path than the module's manual t-stat/p-value calculation) on
    # the exact same differential series.
    returns_df = _two_cluster_returns(n_days=80, daily_gap=0.004, daily_noise_std=0.012, seed=4)
    diff = returns_df["LEFT"] - returns_df["RIGHT"]

    expected_p_value = stats.ttest_1samp(diff, popmean=0.0).pvalue
    expected_significance = max(0.0, min(1.0, 1.0 - expected_p_value))

    actual = _momentum_gap_significance(returns_df, ["LEFT"], ["RIGHT"], lookback=80)
    assert actual == pytest.approx(expected_significance, abs=1e-9)


def test_significance_handles_too_few_observations():
    returns_df = _two_cluster_returns(n_days=120, daily_gap=0.02, daily_noise_std=0.001, seed=5)
    # lookback=1 leaves a single-day differential: no defined standard error.
    assert _momentum_gap_significance(returns_df, ["LEFT"], ["RIGHT"], lookback=1) == 0.0


def test_significance_is_bounded_in_unit_interval_on_real_data():
    returns_df = _load_real_returns_slice(n_days=400)
    tree = ClusterTree.build(returns_df.corr())

    def walk(node):
        if tree.is_leaf(node):
            return
        left, right = tree.children(node)
        left_assets, right_assets = tree.assets_under(left), tree.assets_under(right)
        significance = _momentum_gap_significance(returns_df, left_assets, right_assets, 60)
        assert 0.0 <= significance <= 1.0
        walk(left)
        walk(right)

    walk(tree.root)


def test_confidence_is_the_min_of_data_availability_and_significance():
    # Structural check that generate_momentum_views' confidence is exactly
    # min(data_availability_favored, data_availability_unfavored,
    # significance) -- not, e.g., their product -- reproduced
    # independently by walking the same tree.
    from views.momentum_generator import _cluster_trailing_return, _data_availability_fraction

    returns_df = _load_real_returns_slice(n_days=300)
    tree = ClusterTree.build(returns_df.corr())
    views = generate_momentum_views(returns_df, tree, lookback=60)

    assert len(views.views) >= 1

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

    for view in views.views:
        left_assets, right_assets = find_parent_children(tree.root, view.node_assets)
        other_assets = right_assets if left_assets == view.node_assets else left_assets

        expected_confidence = min(
            _data_availability_fraction(returns_df, view.node_assets, 60),
            _data_availability_fraction(returns_df, other_assets, 60),
            _momentum_gap_significance(returns_df, left_assets, right_assets, 60),
        )
        assert view.confidence == pytest.approx(expected_confidence, abs=1e-12)
