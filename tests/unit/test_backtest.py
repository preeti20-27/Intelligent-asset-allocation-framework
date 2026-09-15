"""Tests for backtest.engine.BacktestEngine.

Two kinds of tests here:

1. Direct ``step()`` tests, using fixed, hand-picked weights (not the real
   allocator) over known synthetic daily returns, so the resulting
   portfolio-value chain -- and therefore weekly_return/monthly_return --
   can be independently hand-computed and compared exactly.

2. A ``run()`` integration test using the real ``allocators.hrp.allocate``
   (per the requirement to import the real allocator, not a stub) over a
   short synthetic date range, checking the constraint/normalization
   contract end to end.
"""

import numpy as np
import pandas as pd
import pytest

from allocators.hrp import allocate
from backtest.calendar import weekly_rebalance_dates
from backtest.costs import transaction_cost
from backtest.engine import (
    MAX_EQUITY_WEIGHT,
    MIN_ETF_WEIGHT,
    BacktestEngine,
    apply_constraints,
)
from monitor.state import PortfolioState


# ---------------------------------------------------------------------------
# Direct step() tests, with fixed weights and hand-computable expectations.
# ---------------------------------------------------------------------------

N_DAYS = 30
FIXED_WEIGHTS = {"A": 0.5, "B": 0.5}


def _make_two_asset_returns(n_days=N_DAYS) -> pd.DataFrame:
    """Small, fixed (non-random) daily returns for two ordinary (non-ETF) assets."""
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    # Deterministic, mildly varying daily returns -- not all identical, so
    # the trailing-window arithmetic isn't degenerate, but still trivial to
    # recompute independently via numpy.
    returns_a = 0.001 * np.sin(np.arange(n_days) / 3.0) + 0.0005
    returns_b = -0.0003 * np.cos(np.arange(n_days) / 5.0)
    return pd.DataFrame({"A": returns_a, "B": returns_b}, index=dates)


def _expected_net_daily_returns(returns_df: pd.DataFrame, weights: dict) -> np.ndarray:
    """Independent reference calculation of each day's net (post-cost) portfolio return.

    Fixed weights every day means a rebalance (and its cost) only happens
    on day 0, when the engine goes from an empty portfolio to
    `weights` for the first time.
    """
    gross = (returns_df["A"] * weights["A"] + returns_df["B"] * weights["B"]).to_numpy()

    empty = pd.Series(dtype=float)
    fixed = pd.Series(weights)
    inception_cost = transaction_cost(empty, fixed)  # 20bps * 1.0 total notional = 0.0020

    net = gross.copy()
    net[0] -= inception_cost
    return net


def test_step_produces_valid_portfolio_state_with_correctly_typed_fields():
    returns_df = _make_two_asset_returns()
    engine = BacktestEngine(returns_df, initial_capital=10_000_000.0)

    state = engine.step(returns_df.index[0], FIXED_WEIGHTS)

    assert isinstance(state, PortfolioState)
    assert isinstance(state.date, str)
    assert isinstance(state.weights, dict)
    assert all(isinstance(v, float) for v in state.weights.values())
    assert isinstance(state.weekly_return, float)
    assert isinstance(state.monthly_return, float)
    assert isinstance(state.drawdown_from_peak, float)
    assert isinstance(state.volatility, float)
    assert isinstance(state.max_weight, float)
    assert isinstance(state.portfolio_value, float)


def test_step_weights_sum_to_one_even_from_unnormalized_input():
    returns_df = _make_two_asset_returns()
    engine = BacktestEngine(returns_df)

    # Deliberately unnormalized (sums to 1.6, not 1.0) -- step() must fix this.
    state = engine.step(returns_df.index[0], {"A": 0.8, "B": 0.8})

    assert sum(state.weights.values()) == pytest.approx(1.0, abs=1e-12)


def test_step_chain_matches_hand_computed_value_and_trailing_returns():
    returns_df = _make_two_asset_returns()
    engine = BacktestEngine(returns_df, initial_capital=10_000_000.0)

    states = [engine.step(date, FIXED_WEIGHTS) for date in returns_df.index]

    expected_net = _expected_net_daily_returns(returns_df, FIXED_WEIGHTS)
    expected_values = 10_000_000.0 * np.cumprod(1.0 + expected_net)

    # Final portfolio value matches the independently computed compounding chain.
    assert states[-1].portfolio_value == pytest.approx(expected_values[-1], rel=1e-9)

    # Weekly (5-trading-day) trailing return at the last day.
    expected_weekly = expected_values[-1] / expected_values[-1 - 5] - 1.0
    assert states[-1].weekly_return == pytest.approx(expected_weekly, rel=1e-9)

    # Monthly (21-trading-day) trailing return at the last day.
    expected_monthly = expected_values[-1] / expected_values[-1 - 21] - 1.0
    assert states[-1].monthly_return == pytest.approx(expected_monthly, rel=1e-9)

    # Trailing returns are genuinely *trailing*: the return as of an
    # earlier day must not depend on any later day's data. Recomputing the
    # weekly return as of day index 10 using only the first 11 values
    # should match what state[10] reported, even though many more days
    # were later appended to the chain.
    expected_weekly_at_10 = expected_values[10] / expected_values[10 - 5] - 1.0
    assert states[10].weekly_return == pytest.approx(expected_weekly_at_10, rel=1e-9)


def test_step_no_cost_when_weights_unchanged_between_calls():
    returns_df = _make_two_asset_returns()
    engine = BacktestEngine(returns_df)

    engine.step(returns_df.index[0], FIXED_WEIGHTS)
    state_day2 = engine.step(returns_df.index[1], FIXED_WEIGHTS)  # same weights again

    expected_day2_return = float(
        returns_df["A"].iloc[1] * 0.5 + returns_df["B"].iloc[1] * 0.5
    )
    prev_value = engine._value_history.iloc[-2]
    expected_value = prev_value * (1.0 + expected_day2_return)
    assert state_day2.portfolio_value == pytest.approx(expected_value, rel=1e-9)


def test_step_drawdown_is_nonpositive_and_zero_at_new_peak():
    returns_df = _make_two_asset_returns()
    engine = BacktestEngine(returns_df)

    states = [engine.step(date, FIXED_WEIGHTS) for date in returns_df.index]

    for state in states:
        assert state.drawdown_from_peak <= 1e-12

    # "Peak" includes the starting capital itself, before any trading --
    # the engine initializes _peak_value to initial_capital, not to the
    # first traded value, so a portfolio that's never beaten its starting
    # capital correctly shows a nonzero drawdown from day 1 (e.g. here,
    # due to the day-0 inception transaction cost).
    peak_so_far = engine.initial_capital
    for state in states:
        if state.portfolio_value >= peak_so_far:
            assert state.drawdown_from_peak == pytest.approx(0.0, abs=1e-9)
            peak_so_far = state.portfolio_value


# ---------------------------------------------------------------------------
# apply_constraints() unit tests
# ---------------------------------------------------------------------------

def test_apply_constraints_sums_to_one():
    raw = pd.Series({"RELIANCE": 0.5, "TCS": 0.3, "GOLDBEES": 0.1, "LTGILTBEES": 0.1})
    constrained = apply_constraints(raw)
    assert constrained.sum() == pytest.approx(1.0, abs=1e-12)


def test_apply_constraints_mild_violation_satisfies_bounds_exactly():
    # Only GOLDBEES needs a floor adjustment here (0.02 -> 0.05); nothing
    # else is clipped initially. With iterative projection, GOLDBEES is
    # frozen at exactly 0.05 and the remaining 0.95 is redistributed only
    # across the still-free assets (RELIANCE, TCS, INFY, LTGILTBEES),
    # preserving their relative proportions -- GOLDBEES is never touched
    # again, so its floor holds exactly rather than approximately.
    raw = pd.Series({"RELIANCE": 0.22, "TCS": 0.20, "INFY": 0.20, "GOLDBEES": 0.02, "LTGILTBEES": 0.36})
    constrained = apply_constraints(raw)

    assert constrained.sum() == pytest.approx(1.0, abs=1e-9)
    assert constrained["GOLDBEES"] == pytest.approx(MIN_ETF_WEIGHT, abs=1e-9)
    for ticker, weight in constrained.items():
        assert weight >= 0.0
        if ticker in ("GOLDBEES", "LTGILTBEES"):
            assert weight >= MIN_ETF_WEIGHT - 1e-9
        else:
            assert weight <= MAX_EQUITY_WEIGHT + 1e-9

    # The three still-free equities keep their original relative
    # proportions (22:20:20) among themselves.
    assert constrained["RELIANCE"] / constrained["TCS"] == pytest.approx(0.22 / 0.20, rel=1e-9)
    assert constrained["TCS"] / constrained["INFY"] == pytest.approx(1.0, rel=1e-9)


def test_apply_constraints_severe_violation_now_satisfies_bounds_exactly():
    # Same adversarial input that broke the old single clip-then-renormalize
    # pass (RELIANCE/TCS both wanted to exceed the 25% cap, and GOLDBEES
    # wanted to go below its 5% floor). Iterative projection freezes each
    # bound-hitting weight in turn and only ever redistributes mass across
    # what's left free, so no bound is re-violated by the redistribution
    # itself -- unlike the old approach, which pushed RELIANCE from a
    # would-be 0.25 cap up to ~0.40 purely from renormalizing.
    raw = pd.Series({"RELIANCE": 0.60, "TCS": 0.30, "GOLDBEES": 0.02, "LTGILTBEES": 0.08})
    constrained = apply_constraints(raw)

    assert constrained.sum() == pytest.approx(1.0, abs=1e-9)
    assert constrained["RELIANCE"] == pytest.approx(MAX_EQUITY_WEIGHT, abs=1e-9)
    assert constrained["TCS"] == pytest.approx(MAX_EQUITY_WEIGHT, abs=1e-9)
    assert constrained["GOLDBEES"] == pytest.approx(MIN_ETF_WEIGHT, abs=1e-9)
    # LTGILTBEES is the only free asset and has no upper cap, so it
    # absorbs all remaining mass: 1.0 - 0.25 - 0.25 - 0.05 = 0.45.
    assert constrained["LTGILTBEES"] == pytest.approx(0.45, abs=1e-9)

    for ticker, weight in constrained.items():
        assert weight >= 0.0
        if ticker in ("GOLDBEES", "LTGILTBEES"):
            assert weight >= MIN_ETF_WEIGHT - 1e-9
        else:
            assert weight <= MAX_EQUITY_WEIGHT + 1e-9


def test_apply_constraints_raises_on_infeasible_input():
    # Four equities each individually capped at 0.25 already sum to 1.0
    # on their own, leaving no room for GOLDBEES/LTGILTBEES's 5% floors --
    # no weight vector can satisfy every bound simultaneously.
    raw = pd.Series(
        {
            "RELIANCE": 0.30,
            "TCS": 0.30,
            "INFY": 0.30,
            "HDFCBANK": 0.30,
            "GOLDBEES": 0.05,
            "LTGILTBEES": 0.05,
        }
    )
    with pytest.raises(ValueError, match="Infeasible"):
        apply_constraints(raw)


def test_apply_constraints_rejects_negative_weights_as_long_only():
    raw = pd.Series({"RELIANCE": 0.9, "TCS": -0.1, "GOLDBEES": 0.1, "LTGILTBEES": 0.1})
    constrained = apply_constraints(raw)
    assert (constrained >= 0).all()
    assert constrained.sum() == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# run() integration test, using the real HRP allocator on a short synthetic range.
# ---------------------------------------------------------------------------

def _make_universe_returns(n_days=40, seed=7) -> pd.DataFrame:
    """Synthetic returns for the full 12-ticker universe (10 equities +
    both ETFs), over a short date range -- exercising the real allocator
    without needing the full 2015-2026 panel.

    Using the full-size universe (not just 2-3 equities) here matters:
    apply_constraints' single clip-then-renormalize pass can reintroduce a
    cap violation after renormalizing (see
    test_apply_constraints_severe_violation_still_sums_to_one_but_can_exceed_cap),
    and that effect is much more pronounced the fewer assets you have --
    capping 1 of 3 equities removes a large share of the total weight,
    capping 1 of 10 removes very little. A realistically-sized universe is
    what makes the bound assertions below hold in practice.
    """
    rng = np.random.default_rng(seed)
    tickers = [
        "RELIANCE", "HDFCBANK", "INFY", "TCS", "ITC", "SUNPHARMA",
        "LT", "MARUTI", "TATASTEEL", "BHARTIARTL", "GOLDBEES", "LTGILTBEES",
    ]
    n_factors = 4
    factor_loadings = rng.normal(0, 1, size=(len(tickers), n_factors))
    factors = rng.normal(0, 0.01, size=(n_days, n_factors))
    idio = rng.normal(0, 0.008, size=(n_days, len(tickers)))
    returns = factors @ factor_loadings.T + idio
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    return pd.DataFrame(returns, index=dates, columns=tickers)


def test_run_with_real_hrp_allocator_respects_constraints_and_normalization():
    returns_df = _make_universe_returns()
    engine = BacktestEngine(returns_df, initial_capital=10_000_000.0)

    start_date = returns_df.index[15]
    end_date = returns_df.index[-1]
    rebalance_dates = weekly_rebalance_dates(returns_df.index)

    equity_curve, states = engine.run(
        start_date=start_date,
        end_date=end_date,
        allocator_fn=allocate,
        window=10,
        rebalance_dates=rebalance_dates,
    )

    assert len(states) == len(returns_df.loc[start_date:end_date])
    assert len(equity_curve) == len(states)

    # apply_constraints now uses iterative projection, so the 25%/5%
    # bounds are guaranteed to hold exactly in the final output -- not
    # just approximately -- even on this realistically-sized 12-asset
    # universe with the real HRP allocator.
    for state in states:
        assert isinstance(state, PortfolioState)
        total = sum(state.weights.values())
        assert total == pytest.approx(1.0, abs=1e-9)

        for ticker, weight in state.weights.items():
            assert weight >= -1e-9  # long-only
            if ticker in ("GOLDBEES", "LTGILTBEES"):
                assert weight >= MIN_ETF_WEIGHT - 1e-9
            else:
                assert weight <= MAX_EQUITY_WEIGHT + 1e-9
