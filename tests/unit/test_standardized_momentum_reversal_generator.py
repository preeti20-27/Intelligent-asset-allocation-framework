"""Tests for views.standardized_momentum_reversal_generator.

Unlike tests/unit/test_momentum_generator.py and
test_mean_reversion_generator.py (which test a single raw-return-gap
formula), this module's whole point is a multi-step pipeline -- per-asset
vol standardization, cross-sectional (not cluster-local) z-scoring, a
two-leg combination, THEN cluster aggregation -- so most tests here
target one pipeline step in isolation with a hand-computable synthetic
case, plus the same structural "does it behave like a ViewGenerator at
all" contract checks the other two generators have.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from allocators.vihrp import allocate as vihrp_allocate
from tree.cluster_tree import ClusterTree
from views.generator import ViewGenerator
from views.standardized_momentum_reversal_generator import (
    MOMENTUM_FORMATION_DAYS,
    MOMENTUM_SKIP_DAYS,
    REVERSAL_FORMATION_DAYS,
    TILT_SCALE,
    _combined_cross_sectional_z,
    _cross_sectional_zscore,
    _leg_raw_value,
    _momentum_raw_per_asset,
    _reversal_raw_per_asset,
    generate_standardized_momentum_reversal_views,
)

REAL_RETURNS_PATH = "data/processed/returns.parquet"
CALIBRATION_CSV_PATH = (
    Path(__file__).resolve().parent.parent.parent / "scripts" / "output" / "standardized_generator_calibration.csv"
)


def _load_real_returns_slice(n_days: int = 900, end: str | None = None) -> pd.DataFrame:
    returns_df = pd.read_parquet(REAL_RETURNS_PATH)
    if end is not None:
        returns_df = returns_df.loc[:end]
    return returns_df.tail(n_days)


def _make_synthetic_returns(n_assets=8, n_days=900, seed=42) -> pd.DataFrame:
    """Same factor-model generator used in the other generators' tests, for consistency."""
    rng = np.random.default_rng(seed)
    tickers = [f"ASSET_{i}" for i in range(n_assets)]
    n_factors = 3
    factor_loadings = rng.normal(0, 1, size=(n_assets, n_factors))
    factors = rng.normal(0, 0.01, size=(n_days, n_factors))
    idio = rng.normal(0, 0.005, size=(n_days, n_assets))
    returns = factors @ factor_loadings.T + idio
    return pd.DataFrame(returns, columns=tickers)


# ---------------------------------------------------------------------------
# Step 1: per-asset volatility standardization
# ---------------------------------------------------------------------------

def test_leg_raw_value_matches_hand_computed_compounded_over_vol():
    rng = np.random.default_rng(1)
    daily = pd.Series(rng.normal(0.0005, 0.02, size=100))

    expected_compounded = float((1.0 + daily).prod() - 1.0)
    expected_vol = float(daily.std(ddof=1) * np.sqrt(252))
    expected = expected_compounded / expected_vol

    assert _leg_raw_value(daily) == pytest.approx(expected, rel=1e-12)


def test_leg_raw_value_is_nan_with_too_few_observations():
    assert np.isnan(_leg_raw_value(pd.Series([0.01, -0.01, 0.005])))  # far below MIN_VALID_DAYS_FOR_LEG


def test_higher_volatility_shrinks_the_standardized_value_for_the_same_raw_return():
    # Two assets with the EXACTLY IDENTICAL compounded return but
    # different volatility: the standardized (vol-divided) value must be
    # smaller in magnitude for the higher-vol asset -- this is the entire
    # point of standardizing at all (a raw-return-gap generator would
    # treat them as identical).
    #
    # Exact-compounding construction: pair up daily factors
    # (1+base)*(1+d) and (1+base)/(1+d) -- each pair's product is exactly
    # (1+base)**2 regardless of d, so alternating them for n/2 pairs
    # compounds to exactly (1+base)**n no matter how large d (and hence
    # realized volatility) is. This avoids the "same compounded return"
    # premise being only approximate (arithmetic-mean anchoring plus
    # volatility drag means an additive-noise construction does NOT
    # actually preserve the compounded return across different noise
    # levels).
    n = 250
    target_total = 0.10
    base = (1.0 + target_total) ** (1.0 / n) - 1.0

    def build_series(d: float) -> pd.Series:
        up = (1.0 + base) * (1.0 + d) - 1.0
        down = (1.0 + base) / (1.0 + d) - 1.0
        return pd.Series(np.tile([up, down], n // 2))

    low_vol_daily = build_series(0.0005)
    high_vol_daily = build_series(0.05)

    # Confirms the construction: both compound to exactly target_total.
    assert float((1.0 + low_vol_daily).prod() - 1.0) == pytest.approx(target_total, rel=1e-9)
    assert float((1.0 + high_vol_daily).prod() - 1.0) == pytest.approx(target_total, rel=1e-9)

    low_value = _leg_raw_value(low_vol_daily)
    high_value = _leg_raw_value(high_vol_daily)

    assert low_value > 0 and high_value > 0
    assert low_value > high_value  # identical return, but lower vol -> bigger standardized value


# ---------------------------------------------------------------------------
# Step 2: momentum leg uses a skip-month formation window
# ---------------------------------------------------------------------------

def test_momentum_leg_ignores_the_most_recent_skip_window():
    # Two versions of the same baseline series, differing ONLY in the
    # most recent MOMENTUM_SKIP_DAYS rows (one has a huge jump planted
    # there, the other doesn't): the skip-month momentum leg must be
    # IDENTICAL for both, since that data is never read at all -- a much
    # more robust check than asserting any particular numeric value (the
    # baseline jitter's own cumulative sum, amplified by its own tiny
    # volatility denominator, isn't necessarily close to 0).
    rng = np.random.default_rng(6)
    n = MOMENTUM_FORMATION_DAYS + MOMENTUM_SKIP_DAYS
    baseline = rng.normal(0, 0.0005, size=n)
    other = rng.normal(0, 0.0005, size=n)

    with_jump = baseline.copy()
    with_jump[-MOMENTUM_SKIP_DAYS:] += 0.05  # huge jump confined to the skipped window

    raw_without_jump = _momentum_raw_per_asset(
        pd.DataFrame({"A": baseline, "B": other}), ["A", "B"], MOMENTUM_FORMATION_DAYS, MOMENTUM_SKIP_DAYS
    )
    raw_with_jump = _momentum_raw_per_asset(
        pd.DataFrame({"A": with_jump, "B": other}), ["A", "B"], MOMENTUM_FORMATION_DAYS, MOMENTUM_SKIP_DAYS
    )

    assert raw_with_jump["A"] == pytest.approx(raw_without_jump["A"], abs=1e-15)


def test_momentum_leg_sees_a_jump_inside_the_formation_window():
    rng = np.random.default_rng(7)
    n = MOMENTUM_FORMATION_DAYS + MOMENTUM_SKIP_DAYS
    series = rng.normal(0, 0.0005, size=n)
    series[10] += 0.05  # inside the formation window, well before the skip zone
    returns_df = pd.DataFrame({"A": series, "B": rng.normal(0, 0.0005, size=n)})

    raw = _momentum_raw_per_asset(returns_df, ["A", "B"], MOMENTUM_FORMATION_DAYS, MOMENTUM_SKIP_DAYS)
    assert raw["A"] > 0.0


# ---------------------------------------------------------------------------
# Step 3: reversal leg is NEGATED
# ---------------------------------------------------------------------------

def test_reversal_leg_is_negated_relative_to_the_raw_long_horizon_return():
    rng = np.random.default_rng(3)
    daily = pd.Series(rng.normal(0.001, 0.015, size=REVERSAL_FORMATION_DAYS))  # strong positive drift
    returns_df = pd.DataFrame({"WINNER": daily})

    raw_unsigned = _leg_raw_value(daily)
    raw_reversal = _reversal_raw_per_asset(returns_df, ["WINNER"], REVERSAL_FORMATION_DAYS)["WINNER"]

    assert raw_unsigned > 0  # confirms the setup: this is indeed a long-run winner
    assert raw_reversal == pytest.approx(-raw_unsigned, rel=1e-12)


def test_reversal_leg_gracefully_shrinks_the_window_when_history_is_shorter():
    # Fewer rows than REVERSAL_FORMATION_DAYS available at all: should use
    # whatever's there (min(formation_days, len(series))), not blow up or
    # silently return nan just because the ideal window doesn't fit.
    rng = np.random.default_rng(4)
    daily = pd.Series(rng.normal(0.0005, 0.01, size=100))
    returns_df = pd.DataFrame({"SHORT_HISTORY": daily})

    raw = _reversal_raw_per_asset(returns_df, ["SHORT_HISTORY"], REVERSAL_FORMATION_DAYS)["SHORT_HISTORY"]
    assert np.isfinite(raw)


# ---------------------------------------------------------------------------
# Step 4: cross-sectional z-scoring is over the FULL universe, not cluster-local
# ---------------------------------------------------------------------------

def test_cross_sectional_zscore_matches_hand_computation():
    raw = {"A": 0.5, "B": -0.2, "C": 1.1, "D": 0.0}
    z = _cross_sectional_zscore(raw)

    values = pd.Series(raw)
    expected_mean, expected_std = values.mean(), values.std(ddof=1)
    for k, v in raw.items():
        assert z[k] == pytest.approx((v - expected_mean) / expected_std, rel=1e-12)


def test_cross_sectional_zscore_excludes_nan_assets_from_the_population():
    raw = {"A": 1.0, "B": 2.0, "C": 3.0, "D": float("nan")}
    z = _cross_sectional_zscore(raw)

    # D must not pollute the population stats used for A/B/C.
    expected_population = pd.Series({"A": 1.0, "B": 2.0, "C": 3.0})
    expected_mean, expected_std = expected_population.mean(), expected_population.std(ddof=1)
    for k in ("A", "B", "C"):
        assert z[k] == pytest.approx((raw[k] - expected_mean) / expected_std, rel=1e-12)
    assert np.isnan(z["D"])


def test_combined_z_is_identical_regardless_of_how_tickers_are_grouped():
    # The whole point of "cross-sectional, not cluster-local": splitting
    # the same universe into two different orderings/subsets passed to
    # _combined_cross_sectional_z as the SAME full ticker list must give
    # byte-identical per-asset results -- there is no tree/cluster
    # parameter for this function to depend on in the first place, and
    # this pins that down as a behavioral guarantee, not just an
    # implementation accident.
    returns_df = _make_synthetic_returns()
    tickers = list(returns_df.columns)

    z_normal_order = _combined_cross_sectional_z(
        returns_df, tickers, MOMENTUM_FORMATION_DAYS, MOMENTUM_SKIP_DAYS, REVERSAL_FORMATION_DAYS
    )
    z_reversed_order = _combined_cross_sectional_z(
        returns_df, list(reversed(tickers)), MOMENTUM_FORMATION_DAYS, MOMENTUM_SKIP_DAYS, REVERSAL_FORMATION_DAYS
    )

    for ticker in tickers:
        assert z_normal_order[ticker] == pytest.approx(z_reversed_order[ticker], abs=1e-12)


def test_combined_z_changes_for_every_asset_when_the_universe_changes():
    # Cross-sectional z-scoring means EVERY asset's score depends on the
    # whole population -- removing one asset from the universe should
    # shift the others' z-scores (a cluster-local computation would be
    # unaffected for assets outside the removed one's cluster).
    returns_df = _make_synthetic_returns(n_assets=8)
    tickers = list(returns_df.columns)

    z_full = _combined_cross_sectional_z(
        returns_df, tickers, MOMENTUM_FORMATION_DAYS, MOMENTUM_SKIP_DAYS, REVERSAL_FORMATION_DAYS
    )
    z_without_one = _combined_cross_sectional_z(
        returns_df, tickers[:-1], MOMENTUM_FORMATION_DAYS, MOMENTUM_SKIP_DAYS, REVERSAL_FORMATION_DAYS
    )

    remaining = tickers[:-1]
    any_changed = any(
        abs(z_full[t] - z_without_one[t]) > 1e-9 for t in remaining if np.isfinite(z_full[t]) and np.isfinite(z_without_one[t])
    )
    assert any_changed


# ---------------------------------------------------------------------------
# Step 5: combining the two legs drops nan legs rather than treating them as 0
# ---------------------------------------------------------------------------

def test_combined_z_uses_only_the_available_leg_when_the_other_is_nan():
    # Build a universe where one asset ("RECENT") has valid data ONLY in
    # its most recent ~25 rows (just listed): the reversal leg's tail
    # window still sees those 25 rows (enough to clear
    # MIN_VALID_DAYS_FOR_LEG) and is well-defined, but the momentum leg's
    # formation window -- which explicitly SKIPS the most recent
    # MOMENTUM_SKIP_DAYS rows before looking back -- ends up seeing almost
    # none of RECENT's few valid rows, and must come out nan. RECENT's
    # combined score must then equal its reversal z-score alone, not an
    # average that treats the missing momentum leg as 0.
    n_total = REVERSAL_FORMATION_DAYS + 50
    n_valid_recent = 25
    rng = np.random.default_rng(5)
    long_history = {f"LONG_{i}": rng.normal(0.0003, 0.01, size=n_total) for i in range(4)}
    recent = np.concatenate(
        [[np.nan] * (n_total - n_valid_recent), rng.normal(0.0003, 0.01, size=n_valid_recent)]
    )
    data = {**long_history, "RECENT": recent}
    returns_df = pd.DataFrame(data)

    combined = _combined_cross_sectional_z(
        returns_df, list(returns_df.columns), MOMENTUM_FORMATION_DAYS, MOMENTUM_SKIP_DAYS, REVERSAL_FORMATION_DAYS
    )
    momentum_only = _cross_sectional_zscore(
        _momentum_raw_per_asset(returns_df, list(returns_df.columns), MOMENTUM_FORMATION_DAYS, MOMENTUM_SKIP_DAYS)
    )
    reversal_only = _cross_sectional_zscore(
        _reversal_raw_per_asset(returns_df, list(returns_df.columns), REVERSAL_FORMATION_DAYS)
    )

    assert np.isnan(momentum_only["RECENT"])  # confirms the setup: RECENT's momentum leg is indeed undefined
    assert np.isfinite(reversal_only["RECENT"])
    assert combined["RECENT"] == pytest.approx(reversal_only["RECENT"], abs=1e-12)


# ---------------------------------------------------------------------------
# Structural contract: same checks the other two generators have
# ---------------------------------------------------------------------------

def test_conforms_to_view_generator_protocol():
    assert isinstance(generate_standardized_momentum_reversal_views, ViewGenerator)


def test_calibrated_constants_match_committed_grid_search_winner():
    """Ties REVERSAL_FORMATION_DAYS/TILT_SCALE directly to the evidence
    that justifies them: scripts/calibrate_standardized_generator.py's
    committed grid-search output must exist, and the module's constants
    must equal the (reversal_formation_days, tilt_scale) row with the
    best IN-SAMPLE Sharpe among combinations with a saturation rate below
    50% (ties broken by shallower in-sample max drawdown) -- the exact
    same selection rule scripts/calibrate_momentum_generator.py uses, and
    that calibration script's own selection rule. Guards against the
    code's constants silently drifting away from the committed evidence,
    or being hand-edited without rerunning/recommitting the grid search.

    momentum_formation_days/momentum_skip_days are NOT checked here --
    they're fixed at the 12-1 literature convention (231/21), not
    grid-searched (see that calibration script's module docstring for
    why), so there's no committed grid row to tie them to.
    """
    assert CALIBRATION_CSV_PATH.exists(), (
        f"{CALIBRATION_CSV_PATH} is missing -- re-run "
        "scripts/calibrate_standardized_generator.py and commit its output."
    )
    grid = pd.read_csv(CALIBRATION_CSV_PATH)
    eligible = grid[grid["saturation_rate_pct"] < 50]
    ranked = eligible.sort_values(
        ["in_sample_sharpe", "in_sample_max_dd_pct"], ascending=[False, False]
    )
    best = ranked.iloc[0]

    assert int(best["reversal_formation_days"]) == REVERSAL_FORMATION_DAYS
    assert float(best["tilt_scale"]) == pytest.approx(TILT_SCALE)


def test_every_view_node_assets_matches_a_real_tree_node_directly():
    returns_df = _load_real_returns_slice()
    tree = ClusterTree.build(returns_df.tail(252).corr())
    views = generate_standardized_momentum_reversal_views(returns_df, tree)

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


def test_tilt_strength_and_confidence_always_within_valid_ranges():
    returns_df = _load_real_returns_slice()
    tree = ClusterTree.build(returns_df.tail(252).corr())
    views = generate_standardized_momentum_reversal_views(returns_df, tree)

    assert len(views.views) >= 1
    for view in views.views:
        assert -1.0 <= view.tilt_strength <= 1.0
        assert 0.0 <= view.confidence <= 1.0


def test_reproducibility_same_input_gives_identical_viewset():
    returns_df = _load_real_returns_slice()
    tree = ClusterTree.build(returns_df.tail(252).corr())

    views_1 = generate_standardized_momentum_reversal_views(returns_df, tree)
    views_2 = generate_standardized_momentum_reversal_views(returns_df, tree)

    assert len(views_1.views) == len(views_2.views)
    key = lambda v: sorted(v.node_assets)  # noqa: E731
    for v1, v2 in zip(sorted(views_1.views, key=key), sorted(views_2.views, key=key)):
        assert v1.node_assets == v2.node_assets
        assert v1.tilt_strength == pytest.approx(v2.tilt_strength, abs=1e-15)
        assert v1.confidence == pytest.approx(v2.confidence, abs=1e-15)


def test_generated_views_are_consumable_by_vihrp_and_shift_weights():
    returns_df = _load_real_returns_slice()
    tree = ClusterTree.build(returns_df.tail(252).corr())
    views = generate_standardized_momentum_reversal_views(returns_df, tree)

    assert len(views.views) >= 1

    tree_window = returns_df.tail(252)
    baseline = vihrp_allocate(tree_window)
    tilted = vihrp_allocate(tree_window, views=views)

    assert tilted.sum() == pytest.approx(1.0, abs=1e-9)
    assert (baseline - tilted).abs().max() > 1e-6


def test_confidence_reflects_sparse_ltgiltbees_data():
    # Same sparsity window the other two generators' tests use: LTGILTBEES
    # trades too sparsely in 2017 for a full-confidence read, and this
    # generator's reversal-window data-availability check should catch it
    # too (REVERSAL_FORMATION_DAYS is far longer than the 60d the other
    # generators check, so this should be at least as sensitive).
    returns_df = pd.read_parquet(REAL_RETURNS_PATH).loc[:"2017-09-01"].tail(900)
    tree = ClusterTree.build(returns_df.tail(252).corr())
    views = generate_standardized_momentum_reversal_views(returns_df, tree)

    assert any(view.confidence < 1.0 for view in views.views)


def test_hand_computed_tilt_strength_for_one_split_on_synthetic_data():
    # Independent recomputation: build combined z-scores directly, pick
    # the root split, hand-compute the expected diff/tilt_strength/favored
    # side, and compare against the generator's own output for that node.
    returns_df = _make_synthetic_returns()
    tree = ClusterTree.build(returns_df.tail(252).corr())
    left, right = tree.children(tree.root)
    left_assets = tree.assets_under(left)
    right_assets = tree.assets_under(right)

    combined_z = _combined_cross_sectional_z(
        returns_df, list(returns_df.columns), MOMENTUM_FORMATION_DAYS, MOMENTUM_SKIP_DAYS, REVERSAL_FORMATION_DAYS
    )
    z_left = float(pd.Series({a: combined_z[a] for a in left_assets}).mean(skipna=True))
    z_right = float(pd.Series({a: combined_z[a] for a in right_assets}).mean(skipna=True))
    expected_diff = z_left - z_right
    expected_tilt_strength = min(abs(expected_diff) / TILT_SCALE, 1.0)
    expected_favored = frozenset(left_assets if expected_diff > 0 else right_assets)

    views = generate_standardized_momentum_reversal_views(returns_df, tree)
    matching = [v for v in views.views if v.node_assets == expected_favored]

    if expected_tilt_strength < 0.01:
        assert matching == []
    else:
        assert len(matching) == 1
        assert matching[0].tilt_strength == pytest.approx(expected_tilt_strength, rel=1e-9)
