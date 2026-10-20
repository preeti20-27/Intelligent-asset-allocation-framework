"""Tests for interface/core.py -- the logic behind the Streamlit app.

Streamlit rendering (interface/app.py) is deliberately not tested here:
the app only lays out what these functions return. What is tested is the
glue -- that a UI selection calls the right allocator / ViewGenerator with
the right arguments, that date ranges are validated against the real data,
that a selection is matched to a committed significance segment or falls
back to a live bootstrap (with the "computing..." UI hook entered only
then), and that the headline finding is derived from the CSV rather than
hardcoded. UI calls are replaced by stubs throughout.
"""

from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from backtest.calendar import load_trading_days, weekly_rebalance_dates
from backtest.engine import DEFAULT_WINDOW, load_default_returns
from eval.metrics import max_drawdown, sharpe_ratio, total_return
from interface import core
from tree.cluster_tree import ClusterTree
from views.view import ViewSet
from views import volatility_regime_generator as vol_mod
from views import momentum_generator as mom_mod


@pytest.fixture(scope="module")
def real_returns():
    return load_default_returns()


@pytest.fixture(scope="module")
def committed():
    return core.load_significance_results()


def _window(n_days=300, n_assets=6, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2020-01-01", periods=n_days)
    return pd.DataFrame(rng.normal(0, 0.01, (n_days, n_assets)), index=index, columns=[f"A{i}" for i in range(n_assets)])


class Recorder:
    """A stub callable that records its calls and returns a fixed value."""

    def __init__(self, returns):
        self.returns = returns
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.returns


class SpinnerStub:
    """Stands in for st.spinner: counts how many times the app's progress hook is entered."""

    def __init__(self):
        self.entered = 0

    @contextmanager
    def __call__(self):
        self.entered += 1
        yield


# ---------------------------------------------------------------------------
# Selection -> strategy -> the right underlying function
# ---------------------------------------------------------------------------

def test_strategy_keys_are_exactly_the_strategies_in_the_committed_csv(committed):
    in_csv = set(committed["strategy_a"]) | set(committed["strategy_b"].dropna())
    assert set(core.ALL_STRATEGIES) == in_csv
    assert len(core.ALL_STRATEGIES) == 9


def test_strategy_key_round_trips_and_rejects_invalid_combinations():
    for key in core.ALL_STRATEGIES:
        assert core.strategy_key(*core.parse_strategy_key(key)) == key
    assert core.strategy_key("vihrp", "volatility_regime") == "vihrp_volatility_regime"
    with pytest.raises(ValueError, match="needs a view signal"):
        core.strategy_key("vihrp")
    with pytest.raises(ValueError, match="only applies to VI-HRP"):
        core.strategy_key("herc", "momentum")
    with pytest.raises(ValueError, match="Unknown allocator"):
        core.strategy_key("not_an_allocator")


@pytest.mark.parametrize("allocator", list(core.BASE_ALLOCATORS))
def test_base_allocator_selection_calls_that_allocate_with_the_engine_window(monkeypatch, allocator):
    expected = pd.Series([1.0], index=["A0"])
    stub = Recorder(expected)
    monkeypatch.setitem(core.BASE_ALLOCATORS, allocator, stub)

    window = _window()
    result = core.build_allocator_fn(core.strategy_key(allocator))(window)

    assert result is expected
    assert len(stub.calls) == 1
    (args, kwargs) = stub.calls[0]
    assert args[0] is window and kwargs == {}


def test_base_allocator_registry_holds_the_real_allocate_functions():
    from allocators import bl, equal_weight, hcaa, herc, hrp

    assert core.BASE_ALLOCATORS == {
        "plain_hrp": hrp.allocate, "equal_weight": equal_weight.allocate,
        "black_litterman": bl.allocate, "herc": herc.allocate, "hcaa": hcaa.allocate,
    }


def test_vihrp_selection_calls_chosen_generator_then_vihrp_allocate(monkeypatch):
    views = ViewSet()
    generator = Recorder(views)
    vihrp = Recorder(pd.Series([1.0], index=["A0"]))
    monkeypatch.setitem(core.SIGNALS, "momentum", core.SignalSpec("stub", generator, {"lookback": 7, "tilt_scale": 0.5}))
    monkeypatch.setattr(core, "vihrp_allocate", vihrp)

    window = _window()
    core.build_allocator_fn(core.strategy_key("vihrp", "momentum"))(window)

    (gen_args, gen_kwargs), = generator.calls
    assert gen_args[0] is window
    assert isinstance(gen_args[1], ClusterTree)
    assert sorted(gen_args[1].leaf_order) == sorted(window.columns)
    assert gen_kwargs == {"lookback": 7, "tilt_scale": 0.5}

    (alloc_args, alloc_kwargs), = vihrp.calls
    assert alloc_args[0] is window and alloc_kwargs == {"views": views}


def test_only_the_selected_signal_is_called(monkeypatch):
    stubs = {name: Recorder(ViewSet()) for name in core.SIGNALS}
    for name, stub in stubs.items():
        monkeypatch.setitem(core.SIGNALS, name, core.SignalSpec(name, stub))
    monkeypatch.setattr(core, "vihrp_allocate", Recorder(pd.Series(dtype=float)))

    core.build_allocator_fn("vihrp_volatility_regime")(_window())
    assert [name for name, s in stubs.items() if s.calls] == ["volatility_regime"]


def test_signal_registry_uses_the_calibrated_generators_and_parameters():
    # Same functions and hyperparameters scripts/apply_significance_tests.py
    # used -- otherwise committed results wouldn't describe what the app runs.
    from views import mean_reversion_generator as mr_mod
    from views import standardized_momentum_reversal_generator as std_mod

    s = core.SIGNALS
    assert s["momentum"].generator is mom_mod.generate_momentum_views
    assert s["momentum"].kwargs == {"lookback": mom_mod.LOOKBACK, "tilt_scale": mom_mod.TILT_SCALE}
    assert s["mean_reversion"].generator is mr_mod.generate_mean_reversion_views
    assert s["mean_reversion"].kwargs == {"lookback": mom_mod.LOOKBACK, "tilt_scale": mom_mod.TILT_SCALE}
    assert s["standardized"].generator is std_mod.generate_standardized_momentum_reversal_views
    assert s["standardized"].kwargs == {
        "momentum_formation_days": std_mod.MOMENTUM_FORMATION_DAYS,
        "momentum_skip_days": std_mod.MOMENTUM_SKIP_DAYS,
        "reversal_formation_days": std_mod.REVERSAL_FORMATION_DAYS,
        "tilt_scale": std_mod.TILT_SCALE,
    }
    assert s["standardized"].history_days == std_mod.REVERSAL_FORMATION_DAYS + 50
    assert s["volatility_regime"].generator is vol_mod.generate_volatility_regime_views
    assert s["volatility_regime"].kwargs == {
        "short_vol_window": vol_mod.SHORT_VOL_WINDOW,
        "long_vol_window": vol_mod.LONG_VOL_WINDOW,
        "tilt_scale": vol_mod.TILT_SCALE,
    }


def test_long_history_signal_receives_trailing_slice_of_full_panel(monkeypatch):
    generator = Recorder(ViewSet())
    monkeypatch.setitem(core.SIGNALS, "standardized", core.SignalSpec("stub", generator, history_days=100))
    monkeypatch.setattr(core, "vihrp_allocate", Recorder(pd.Series(dtype=float)))

    full = _window(n_days=500)
    window = full.iloc[200:300]  # what the engine would hand over
    core.build_allocator_fn("vihrp_standardized", returns_full=full)(window)

    (args, _), = generator.calls
    pd.testing.assert_frame_equal(args[0], full.loc[: window.index[-1]].tail(100))

    with pytest.raises(ValueError, match="full returns panel"):
        core.build_allocator_fn("vihrp_standardized")


def test_run_full_backtest_passes_selection_to_engine(monkeypatch, real_returns):
    trading_days = load_trading_days()
    seen = {}

    class FakeEngine:
        def __init__(self, returns_df):
            seen["returns_df"] = returns_df

        def run(self, start_date, end_date, allocator_fn, window, rebalance_dates):
            seen.update(start=start_date, end=end_date, fn=allocator_fn, window=window, dates=rebalance_dates)
            index = pd.DatetimeIndex([start_date, end_date])
            states = [SimpleNamespace(weights={"A": 1.0}), SimpleNamespace(weights={"A": 1.0})]
            return pd.Series([1.0, 1.1], index=index), states

    herc_stub = Recorder(pd.Series([1.0], index=["A"]))
    monkeypatch.setitem(core.BASE_ALLOCATORS, "herc", herc_stub)

    result = core.run_full_backtest("herc", "weekly", real_returns, trading_days, engine_cls=FakeEngine)

    assert seen["returns_df"] is real_returns
    assert (seen["start"], seen["end"]) == core.date_bounds(real_returns)
    assert seen["window"] == DEFAULT_WINDOW
    pd.testing.assert_index_equal(seen["dates"], weekly_rebalance_dates(trading_days))
    window = real_returns.iloc[:10]
    seen["fn"](window)
    assert herc_stub.calls[0][0][0] is window
    assert list(result.weights.columns) == ["A"]


# ---------------------------------------------------------------------------
# Rebalance frequency
# ---------------------------------------------------------------------------

def test_weekly_frequency_is_the_engine_calendar_and_monthly_is_first_trading_day():
    days = load_trading_days()
    pd.testing.assert_index_equal(core.rebalance_dates("weekly", days), weekly_rebalance_dates(days))

    monthly = core.rebalance_dates("monthly", days)
    by_month = pd.Series(days, index=days).groupby(days.to_period("M")).min()
    assert list(monthly) == list(by_month)
    assert monthly.to_period("M").is_unique
    assert pd.Timestamp("2023-01-02") in monthly  # Jan 1 2023 was a Sunday
    with pytest.raises(ValueError, match="frequency"):
        core.rebalance_dates("daily", days)


# ---------------------------------------------------------------------------
# Date range bounds and validation against the real data
# ---------------------------------------------------------------------------

def test_date_bounds_match_the_data(real_returns):
    lo, hi = core.date_bounds(real_returns)
    assert hi == real_returns.index[-1]
    # Every allocator's 90% non-null check passes on the window ending the day before lo...
    window = real_returns.loc[:lo].iloc[:-1].tail(DEFAULT_WINDOW)
    assert (window.notna().mean() >= 0.9).all()
    # ...and fails one trading day earlier (lo is the earliest feasible start).
    earlier = real_returns.loc[:lo].iloc[:-2].tail(DEFAULT_WINDOW)
    assert not (earlier.notna().mean() >= 0.9).all()


def test_valid_range_is_snapped_to_trading_days(real_returns):
    start, end = core.validate_date_range("2023-01-01", "2024-06-30", real_returns)  # Sun, Sun
    assert start == pd.Timestamp("2023-01-02") and end == pd.Timestamp("2024-06-28")
    assert start in real_returns.index and end in real_returns.index


@pytest.mark.parametrize(
    "start, end, message",
    [
        ("2015-01-05", "2020-01-01", "before"),
        ("2024-01-01", "2027-01-01", "after the last date"),
        ("2024-06-01", "2024-01-01", "is after end date"),
        ("2024-01-01", "2024-02-15", "trading days"),
    ],
)
def test_invalid_ranges_are_rejected_with_a_clear_message(real_returns, start, end, message):
    with pytest.raises(ValueError, match=message):
        core.validate_date_range(start, end, real_returns)


# ---------------------------------------------------------------------------
# Segment matching
# ---------------------------------------------------------------------------

def test_tested_segments_reproduce_the_csv_observation_counts(real_returns, committed):
    segments = core.tested_segments(real_returns)
    n_obs = committed.query("test == 'pairwise'").groupby("segment")["n_obs"].first()
    assert set(segments) == set(n_obs.index)
    for name, (s, e) in segments.items():
        trading = real_returns.loc[s:e]
        assert len(trading) - 1 == n_obs[name]  # returns drop the first day


def test_selection_matches_committed_segment_only_when_exact(real_returns):
    segments = core.tested_segments(real_returns)
    # A natural calendar selection snaps onto the OOS segment.
    start, end = core.validate_date_range("2023-01-01", "2026-09-11", real_returns)
    assert core.match_segment(start, end, segments) == core.OOS_SEGMENT
    for name, (s, e) in segments.items():
        assert core.match_segment(s, e, segments) == name
    # One trading day different -> not the tested configuration.
    start, end = core.validate_date_range("2023-01-03", "2026-09-11", real_returns)
    assert core.match_segment(start, end, segments) is None


# ---------------------------------------------------------------------------
# Significance: committed lookup vs. live bootstrap fallback
# ---------------------------------------------------------------------------

def _fake_test_result(diff=0.1):
    return SimpleNamespace(sharpe_a=1.0, sharpe_b=1.0 - diff, difference=diff, ci_low=-0.5, ci_high=0.7, p_value=0.4)


def _returns(keys):
    idx = pd.bdate_range("2023-01-02", periods=100)
    return {k: pd.Series(np.linspace(0, 0.001, 100), index=idx) for k in keys}


def test_committed_lookup_is_oriented_to_the_requested_order(committed):
    forward = core.lookup_committed_pair(committed, core.OOS_SEGMENT, "plain_hrp", "equal_weight")
    backward = core.lookup_committed_pair(committed, core.OOS_SEGMENT, "equal_weight", "plain_hrp")
    assert forward["sharpe_diff"] == pytest.approx(-backward["sharpe_diff"])
    assert forward["ci95_low"] == pytest.approx(-backward["ci95_high"])
    assert forward["sharpe_a"] == backward["sharpe_b"]
    assert forward["p_value"] == backward["p_value"]
    assert core.lookup_committed_pair(committed, "no_such_segment", "plain_hrp", "equal_weight") is None


def test_matching_weekly_selection_uses_committed_results_without_computing(committed):
    spinner, test_fn = SpinnerStub(), Recorder(_fake_test_result())
    keys = ["plain_hrp", "equal_weight", "vihrp_volatility_regime"]
    table = core.pairwise_significance(_returns(keys), core.OOS_SEGMENT, "weekly", committed, spinner, test_fn)

    assert spinner.entered == 0 and test_fn.calls == []
    assert (table["source"] == "committed").all() and len(table) == 3
    row = table[(table.strategy_a == "plain_hrp") & (table.strategy_b == "equal_weight")].iloc[0]
    expected = core.lookup_committed_pair(committed, core.OOS_SEGMENT, "plain_hrp", "equal_weight")
    assert row["p_value"] == expected["p_value"] and row["sharpe_diff"] == expected["sharpe_diff"]
    np.testing.assert_allclose(table["p_holm"], core.holm_adjust(table["p_value"]))


@pytest.mark.parametrize("segment, frequency", [(None, "weekly"), (core.OOS_SEGMENT, "monthly")])
def test_unmatched_selection_falls_back_to_live_bootstrap(committed, segment, frequency):
    spinner, test_fn = SpinnerStub(), Recorder(_fake_test_result())
    returns = _returns(["plain_hrp", "herc", "hcaa"])
    table = core.pairwise_significance(returns, segment, frequency, committed, spinner, test_fn)

    assert spinner.entered == 1  # one visible "computing..." state around all live tests
    assert len(test_fn.calls) == 3
    assert (table["source"] == "live bootstrap").all()
    args, kwargs = test_fn.calls[0]
    assert args[0] is returns["plain_hrp"] and args[1] is returns["herc"]
    assert kwargs == {"n_boot": core.N_BOOT, "alpha": core.ALPHA, "seed": core.BOOTSTRAP_SEED}


def test_pair_missing_from_csv_is_computed_live_and_others_looked_up(committed):
    trimmed = committed[~((committed.strategy_a == "plain_hrp") & (committed.strategy_b == "herc"))]
    spinner, test_fn = SpinnerStub(), Recorder(_fake_test_result())
    table = core.pairwise_significance(
        _returns(["plain_hrp", "equal_weight", "herc"]), core.OOS_SEGMENT, "weekly", trimmed, spinner, test_fn
    )
    sources = dict(zip(zip(table.strategy_a, table.strategy_b), table.source))
    assert sources[("plain_hrp", "herc")] == "live bootstrap"
    assert sources[("plain_hrp", "equal_weight")] == "committed"
    assert spinner.entered == 1 and len(test_fn.calls) == 1


def test_live_bootstrap_on_a_tested_segment_reproduces_the_committed_csv(real_returns, committed):
    # End-to-end: real backtests through the interface's own wiring, then a
    # live test on the OOS slice must equal the committed row exactly
    # (same curves, same bootstrap settings and seed). ~1 minute.
    trading_days = load_trading_days()
    curves = {k: core.run_full_backtest(k, "weekly", real_returns, trading_days).equity
              for k in ("plain_hrp", "equal_weight")}
    s, e = core.tested_segments(real_returns)[core.OOS_SEGMENT]
    returns = {k: core.range_returns(eq, s, e) for k, eq in curves.items()}

    live = core.pairwise_significance(returns, None, "weekly", committed).iloc[0]
    saved = core.lookup_committed_pair(committed, core.OOS_SEGMENT, "plain_hrp", "equal_weight")
    assert live["source"] == "live bootstrap"
    for field in ("sharpe_a", "sharpe_b", "sharpe_diff", "ci95_low", "ci95_high", "p_value"):
        assert live[field] == pytest.approx(saved[field], abs=1e-12), field


# ---------------------------------------------------------------------------
# Metrics and weights
# ---------------------------------------------------------------------------

def test_range_metrics_use_eval_metrics_on_the_sliced_curve():
    idx = pd.bdate_range("2021-01-01", periods=300)
    equity = pd.Series(100 * np.cumprod(1 + np.random.default_rng(1).normal(0.0005, 0.01, 300)), index=idx)
    s, e = idx[50], idx[250]
    m = core.range_metrics(equity, s, e)
    daily = equity.loc[s:e].pct_change().dropna()
    assert m["sharpe"] == sharpe_ratio(daily)
    assert m["max_drawdown"] == max_drawdown(equity.loc[s:e]).max_drawdown
    assert m["total_return"] == total_return(equity.loc[s:e])
    assert m["volatility"] == pytest.approx(daily.std() * np.sqrt(252))


def test_weights_on_returns_last_held_weights():
    idx = pd.bdate_range("2024-01-01", periods=3)
    weights = pd.DataFrame({"A": [0.2, 0.3, 0.4], "B": [0.8, 0.7, 0.6]}, index=idx)
    pd.testing.assert_series_equal(core.weights_on(weights, idx[1]), weights.iloc[1])
    pd.testing.assert_series_equal(core.weights_on(weights, "2024-01-06"), weights.iloc[2])  # a Saturday
    with pytest.raises(ValueError):
        core.weights_on(weights, "2023-12-01")


# ---------------------------------------------------------------------------
# The honest-finding banner is driven by the CSV
# ---------------------------------------------------------------------------

def test_findings_from_committed_csv(committed):
    f = core.honest_findings(committed)
    beaten = {r["other"] for r in f.hrp_significantly_beats}
    assert {"equal_weight", "black_litterman"} <= beaten
    for r in f.hrp_significantly_beats:
        assert r["p_holm"] < core.ALPHA and r["hrp_minus_other"] > 0
    assert f.vihrp_significantly_beats_hrp == []

    text = core.findings_markdown(f)
    assert "Equal weight" in text and "Black-Litterman" in text
    assert "No tested VI-HRP signal beats plain HRP" in text


def test_vs_plain_hrp_orientation_does_not_depend_on_csv_column_order(committed):
    swapped = committed.copy()
    rows = swapped["test"] == "pairwise"
    swapped.loc[rows, ["strategy_a", "strategy_b"]] = swapped.loc[rows, ["strategy_b", "strategy_a"]].values
    swapped.loc[rows, ["sharpe_a", "sharpe_b"]] = swapped.loc[rows, ["sharpe_b", "sharpe_a"]].values
    swapped.loc[rows, "sharpe_diff"] *= -1

    original = core.vs_plain_hrp(committed).sort_values(["segment", "other"]).reset_index(drop=True)
    flipped = core.vs_plain_hrp(swapped).sort_values(["segment", "other"]).reset_index(drop=True)
    assert len(original) == 4 * 8
    pd.testing.assert_frame_equal(original, flipped)
    assert core.findings_markdown(core.honest_findings(swapped)) == core.findings_markdown(
        core.honest_findings(committed)
    )


def test_findings_text_changes_when_the_csv_changes(committed):
    # Flip one VI-HRP-vs-HRP row to a significant VI-HRP win: the banner
    # must stop saying no signal beats plain HRP.
    edited = committed.copy()
    row = (
        (edited.test == "pairwise") & (edited.segment == core.OOS_SEGMENT)
        & (edited.strategy_a == "plain_hrp") & (edited.strategy_b == "vihrp_momentum")
    )
    assert row.sum() == 1
    edited.loc[row, ["sharpe_diff", "p_value", "p_holm_vs_plain_hrp"]] = [-0.9, 0.001, 0.004]
    edited.loc[row, "sharpe_b"] = edited.loc[row, "sharpe_a"] + 0.9

    text = core.findings_markdown(core.honest_findings(edited))
    assert "No tested VI-HRP signal beats plain HRP" not in text
    assert "VI-HRP: Momentum" in text

    # And removing every significant HRP win removes that claim too.
    none_sig = committed.copy()
    none_sig["p_holm_vs_plain_hrp"] = none_sig["p_holm_vs_plain_hrp"].where(
        none_sig["p_holm_vs_plain_hrp"].isna(), 1.0
    )
    assert "does not significantly beat" in core.findings_markdown(core.honest_findings(none_sig))
