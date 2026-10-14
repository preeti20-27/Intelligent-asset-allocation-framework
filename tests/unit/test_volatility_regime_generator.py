"""Tests for views.volatility_regime_generator.generate_volatility_regime_views.

Follows the style of the other generators' test modules: structural
correctness against hand-computable synthetic cases, plus the standard
ViewGenerator contract checks. The two properties unique to this
generator -- regime classification (calm vs. stressed) and the hard
guarantee that it NEVER emits a view for anything but the two
asset-class nodes -- get dedicated sections.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from allocators.vihrp import allocate as vihrp_allocate
from tree.cluster_tree import ClusterTree
from views.generator import ViewGenerator
from views.volatility_regime_generator import (
    GILT_TICKER,
    GOLD_TICKER,
    LONG_VOL_WINDOW,
    SHORT_VOL_WINDOW,
    TILT_SCALE,
    _find_singleton_vs_rest_node,
    _vol_regime_signal,
    generate_volatility_regime_views,
)

REAL_RETURNS_PATH = "data/processed/returns.parquet"
CALIBRATION_CSV_PATH = (
    Path(__file__).resolve().parent.parent.parent / "scripts" / "output" / "volatility_regime_calibration.csv"
)


def _load_real_returns_slice(n_days: int = 252, end: str | None = None) -> pd.DataFrame:
    returns_df = pd.read_parquet(REAL_RETURNS_PATH)
    if end is not None:
        returns_df = returns_df.loc[:end]
    return returns_df.tail(n_days)


def _make_synthetic_universe(n_days: int, equity_daily_std: float, seed: int) -> pd.DataFrame:
    """12-ticker synthetic universe (10 equities + GOLDBEES + LTGILTBEES)
    with equities all sharing the same constant daily-return std (so the
    regime signal's input, the equity blend, has an easily controlled
    volatility), and GOLDBEES/LTGILTBEES as independent, unrelated noise.
    """
    rng = np.random.default_rng(seed)
    equities = {f"EQUITY_{i}": rng.normal(0, equity_daily_std, size=n_days) for i in range(10)}
    data = {
        **equities,
        GOLD_TICKER: rng.normal(0, 0.01, size=n_days),
        GILT_TICKER: rng.normal(0, 0.005, size=n_days),
    }
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------------------

def test_regime_signal_is_near_zero_when_short_and_long_vol_match():
    # Same std throughout (no genuine regime shift): regime_signal should
    # be small (not exactly 0 -- finite-sample noise -- but nowhere near
    # a "doubled volatility" reading).
    returns_df = _make_synthetic_universe(n_days=SHORT_VOL_WINDOW + LONG_VOL_WINDOW, equity_daily_std=0.01, seed=1)
    regime_signal, _ = _vol_regime_signal(returns_df, SHORT_VOL_WINDOW, LONG_VOL_WINDOW)
    assert abs(regime_signal) < 0.5


def test_regime_signal_is_strongly_positive_when_recent_vol_spikes():
    # Long window: calm. Short window (most recent SHORT_VOL_WINDOW rows):
    # much higher std -- a clean, large regime_signal is expected.
    rng = np.random.default_rng(2)
    long_part = rng.normal(0, 0.005, size=LONG_VOL_WINDOW)
    short_part = rng.normal(0, 0.03, size=SHORT_VOL_WINDOW)  # 6x the daily std
    equities = {f"EQUITY_{i}": np.concatenate([long_part, short_part]) for i in range(10)}
    returns_df = pd.DataFrame({
        **equities,
        GOLD_TICKER: rng.normal(0, 0.01, size=LONG_VOL_WINDOW + SHORT_VOL_WINDOW),
        GILT_TICKER: rng.normal(0, 0.005, size=LONG_VOL_WINDOW + SHORT_VOL_WINDOW),
    })

    regime_signal, significance = _vol_regime_signal(returns_df, SHORT_VOL_WINDOW, LONG_VOL_WINDOW)
    assert regime_signal > 2.0  # vol roughly 6x -> signal should be well above a small/noise-level reading
    assert significance > 0.9  # a 6x vol difference over these sample sizes should be highly significant


def test_regime_signal_is_nan_with_insufficient_history():
    returns_df = _make_synthetic_universe(n_days=5, equity_daily_std=0.01, seed=3)
    regime_signal, significance = _vol_regime_signal(returns_df, SHORT_VOL_WINDOW, LONG_VOL_WINDOW)
    assert np.isnan(regime_signal)
    assert significance == 0.0


def test_no_views_emitted_in_a_calm_regime():
    # Deterministic construction: the most recent (short) window has a
    # CLEARLY SMALLER amplitude than the preceding (long) window -- a
    # below-baseline-vol ("extra calm") regime, not merely an
    # equal-vol one. tilt_strength only reacts to POSITIVE regime_signal
    # (see generate_volatility_regime_views: max(regime_signal, 0)), so
    # this must give exactly zero tilt regardless of small-sample ddof=1
    # variance-estimator bias between the differently-sized windows
    # (which, empirically, an exactly-equal-amplitude construction does
    # NOT fully cancel out -- a real, expected statistical detail, not a
    # generator bug).
    n = SHORT_VOL_WINDOW + LONG_VOL_WINDOW
    long_pattern = np.tile([0.01, -0.01], LONG_VOL_WINDOW // 2 + 1)[:LONG_VOL_WINDOW]
    short_pattern = np.tile([0.002, -0.002], SHORT_VOL_WINDOW // 2 + 1)[:SHORT_VOL_WINDOW]  # 5x calmer
    pattern = np.concatenate([long_pattern, short_pattern])
    equities = {f"EQUITY_{i}": pattern for i in range(10)}
    returns_df = pd.DataFrame({
        **equities,
        GOLD_TICKER: np.tile([0.005, -0.005], n // 2 + 1)[:n],
        GILT_TICKER: np.tile([0.002, -0.002], n // 2 + 1)[:n],
    })
    tree = ClusterTree.build(returns_df.corr())
    views = generate_volatility_regime_views(returns_df, tree)
    assert len(views.views) == 0


def test_views_emitted_in_a_stressed_regime_on_real_covid_window():
    # The actual discovery case from scripts/diagnose_universe_structure.py:
    # a 252-day window ending at the COVID market bottom.
    returns_df = pd.read_parquet(REAL_RETURNS_PATH).loc[:"2020-03-23"].tail(252)
    tree = ClusterTree.build(returns_df.corr())
    views = generate_volatility_regime_views(returns_df, tree)

    assert len(views.views) >= 1
    assert any(v.node_assets == frozenset({GOLD_TICKER}) for v in views.views)
    for v in views.views:
        assert v.tilt_strength > 0.0


# ---------------------------------------------------------------------------
# Hard guarantee: only ever the two asset-class nodes
# ---------------------------------------------------------------------------

def test_find_singleton_vs_rest_node_matches_the_real_root_split():
    returns_df = _load_real_returns_slice()
    tree = ClusterTree.build(returns_df.corr())
    all_tickers = frozenset(returns_df.columns)

    match = _find_singleton_vs_rest_node(tree, GOLD_TICKER, all_tickers - {GOLD_TICKER})
    assert match is not None
    favored, unfavored = match
    assert favored == frozenset({GOLD_TICKER})
    assert GOLD_TICKER not in unfavored


def test_find_singleton_vs_rest_node_returns_none_when_shape_absent():
    # A universe where the "singleton" doesn't exist at all as a direct
    # child of any node matching the allowed-other-assets constraint --
    # e.g. asking for a node splitting off a ticker against ONLY a
    # single other specific ticker in an 8-asset synthetic universe
    # should not find a false match.
    rng = np.random.default_rng(5)
    returns_df = pd.DataFrame({f"A{i}": rng.normal(0, 0.01, size=300) for i in range(8)})
    tree = ClusterTree.build(returns_df.corr())

    # No ticker named GOLD_TICKER exists in this universe at all.
    match = _find_singleton_vs_rest_node(tree, GOLD_TICKER, frozenset(returns_df.columns))
    assert match is None


def test_never_emits_a_view_for_an_equity_internal_node():
    # Across several real-data windows (including the stressed COVID one,
    # where tilt_strength is large and views ARE emitted), every emitted
    # view's node_assets must be exactly {GOLDBEES} or exactly
    # {LTGILTBEES} -- never any other node, e.g. never a multi-equity
    # cluster or a single non-special equity.
    windows = [
        pd.read_parquet(REAL_RETURNS_PATH).loc[:"2020-03-23"].tail(252),  # stressed
        pd.read_parquet(REAL_RETURNS_PATH).tail(252),  # whatever the most recent window is
        pd.read_parquet(REAL_RETURNS_PATH).loc[:"2019-11-01"].tail(252),  # calm
    ]
    checked_any_emission = False
    for returns_df in windows:
        tree = ClusterTree.build(returns_df.corr())
        views = generate_volatility_regime_views(returns_df, tree)
        for v in views.views:
            checked_any_emission = True
            assert v.node_assets in (frozenset({GOLD_TICKER}), frozenset({GILT_TICKER}))

    assert checked_any_emission  # otherwise this test is vacuous


# ---------------------------------------------------------------------------
# Structural contract: same checks the other generators have
# ---------------------------------------------------------------------------

def test_conforms_to_view_generator_protocol():
    assert isinstance(generate_volatility_regime_views, ViewGenerator)


def test_tilt_strength_and_confidence_always_within_valid_ranges():
    returns_df = pd.read_parquet(REAL_RETURNS_PATH).loc[:"2020-03-23"].tail(252)
    tree = ClusterTree.build(returns_df.corr())
    views = generate_volatility_regime_views(returns_df, tree)

    assert len(views.views) >= 1
    for v in views.views:
        assert -1.0 <= v.tilt_strength <= 1.0
        assert 0.0 <= v.confidence <= 1.0


def test_reproducibility_same_input_gives_identical_viewset():
    returns_df = pd.read_parquet(REAL_RETURNS_PATH).loc[:"2020-03-23"].tail(252)
    tree = ClusterTree.build(returns_df.corr())

    views_1 = generate_volatility_regime_views(returns_df, tree)
    views_2 = generate_volatility_regime_views(returns_df, tree)

    assert len(views_1.views) == len(views_2.views)
    key = lambda v: sorted(v.node_assets)  # noqa: E731
    for v1, v2 in zip(sorted(views_1.views, key=key), sorted(views_2.views, key=key)):
        assert v1.node_assets == v2.node_assets
        assert v1.tilt_strength == pytest.approx(v2.tilt_strength, abs=1e-15)
        assert v1.confidence == pytest.approx(v2.confidence, abs=1e-15)


def test_generated_views_are_consumable_by_vihrp_and_shift_weights():
    returns_df = pd.read_parquet(REAL_RETURNS_PATH).loc[:"2020-03-23"].tail(252)
    tree = ClusterTree.build(returns_df.corr())
    views = generate_volatility_regime_views(returns_df, tree)

    assert len(views.views) >= 1

    baseline = vihrp_allocate(returns_df)
    tilted = vihrp_allocate(returns_df, views=views)

    assert tilted.sum() == pytest.approx(1.0, abs=1e-9)
    assert (baseline - tilted).abs().max() > 1e-6
    # The point of this generator: GOLDBEES's weight should go UP under stress.
    assert tilted[GOLD_TICKER] > baseline[GOLD_TICKER]


def test_empty_viewset_is_inert_on_vihrp():
    returns_df = _load_real_returns_slice()
    baseline = vihrp_allocate(returns_df)
    from views.view import ViewSet

    with_empty = vihrp_allocate(returns_df, views=ViewSet())
    pd.testing.assert_series_equal(baseline.sort_index(), with_empty.sort_index())


def test_default_tilt_scale_and_windows_match_module_constants():
    returns_df = pd.read_parquet(REAL_RETURNS_PATH).loc[:"2020-03-23"].tail(252)
    tree = ClusterTree.build(returns_df.corr())

    views_default = generate_volatility_regime_views(returns_df, tree)
    views_explicit = generate_volatility_regime_views(
        returns_df, tree,
        short_vol_window=SHORT_VOL_WINDOW, long_vol_window=LONG_VOL_WINDOW, tilt_scale=TILT_SCALE,
    )

    assert len(views_default.views) == len(views_explicit.views)
    key = lambda v: sorted(v.node_assets)  # noqa: E731
    for v1, v2 in zip(sorted(views_default.views, key=key), sorted(views_explicit.views, key=key)):
        assert v1.node_assets == v2.node_assets
        assert v1.tilt_strength == v2.tilt_strength
        assert v1.confidence == v2.confidence


def test_calibrated_constants_match_committed_grid_search_winner():
    """Ties SHORT_VOL_WINDOW/TILT_SCALE directly to the evidence that
    justifies them: scripts/calibrate_volatility_regime_generator.py's
    committed grid-search output must exist, and the module's constants
    must equal the (short_vol_window, tilt_scale) row with the best
    IN-SAMPLE Sharpe among combinations with a saturation rate below 50%
    (ties broken by shallower in-sample max drawdown) -- the same
    selection rule every other calibration script in this project uses.
    Guards against the code's constants silently drifting away from the
    committed evidence, or being hand-edited without rerunning/
    recommitting the grid search.

    LONG_VOL_WINDOW is NOT checked here -- it's fixed, not grid-searched
    (see the calibration script's module docstring for why), so there's
    no committed grid row to tie it to.
    """
    assert CALIBRATION_CSV_PATH.exists(), (
        f"{CALIBRATION_CSV_PATH} is missing -- re-run "
        "scripts/calibrate_volatility_regime_generator.py and commit its output."
    )
    grid = pd.read_csv(CALIBRATION_CSV_PATH)
    eligible = grid[grid["saturation_rate_pct"] < 50]
    ranked = eligible.sort_values(
        ["in_sample_sharpe", "in_sample_max_dd_pct"], ascending=[False, False]
    )
    best = ranked.iloc[0]

    assert int(best["short_vol_window"]) == SHORT_VOL_WINDOW
    assert float(best["tilt_scale"]) == pytest.approx(TILT_SCALE)
