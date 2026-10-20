"""Tests for allocators.bl -- Black-Litterman."""

import numpy as np
import pandas as pd
import pytest

from allocators.bl import (
    DEFAULT_TAU,
    BlackLittermanViews,
    allocate,
    implied_equilibrium_returns,
    posterior,
)


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


def _absolute_view(tickers, ticker, q, omega=None) -> BlackLittermanViews:
    P = pd.DataFrame([[1.0 if t == ticker else 0.0 for t in tickers]], columns=tickers)
    return BlackLittermanViews(P=P, Q=[q], omega=omega)


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


def test_rejects_asset_with_too_much_missing_data():
    returns_df = _make_synthetic_returns()
    returns_df.iloc[:100, 2] = np.nan
    with pytest.raises(ValueError, match="ASSET_2"):
        allocate(returns_df)


def test_rejects_view_on_unknown_ticker():
    returns_df = _make_synthetic_returns()
    with pytest.raises(ValueError, match="NOT_A_TICKER"):
        allocate(returns_df, views=_absolute_view(["NOT_A_TICKER"], "NOT_A_TICKER", 0.001))


# ---------------------------------------------------------------------------
# Zero views: reduces to the equal-weight prior
# ---------------------------------------------------------------------------

def test_zero_view_posterior_is_prior():
    returns_df = _make_synthetic_returns()
    cov = returns_df.cov().values
    pi = implied_equilibrium_returns(cov, np.full(len(cov), 1 / len(cov)))
    mu, cov_bl = posterior(cov, pi, tau=DEFAULT_TAU)
    np.testing.assert_allclose(mu, pi)
    np.testing.assert_allclose(cov_bl, (1 + DEFAULT_TAU) * cov)


def test_zero_view_weights_match_closed_form_prior_plus_tau_minvar_tilt():
    # Derived in allocators/bl.py's module docstring:
    # w* = w_mkt / (1 + tau) + tau / (1 + tau) * w_minvar.
    returns_df = _make_synthetic_returns()
    cov = returns_df.cov().values
    n = len(cov)
    w_mkt = np.full(n, 1.0 / n)
    inv_ones = np.linalg.solve(cov, np.ones(n))
    w_minvar = inv_ones / inv_ones.sum()
    expected = w_mkt / (1 + DEFAULT_TAU) + DEFAULT_TAU / (1 + DEFAULT_TAU) * w_minvar

    weights = allocate(returns_df, long_only=False)
    np.testing.assert_allclose(weights.values, expected, atol=1e-10)


def test_zero_view_weights_are_close_to_equal_weight():
    returns_df = _make_synthetic_returns()
    weights = allocate(returns_df)
    n = len(weights)
    # Within the tau/(1+tau) budget of the min-variance tilt.
    assert (weights - 1.0 / n).abs().sum() <= 2 * DEFAULT_TAU / (1 + DEFAULT_TAU) + 1e-9
    # And as tau -> 0 the tilt vanishes: exactly the prior.
    np.testing.assert_allclose(allocate(returns_df, tau=1e-12).values, 1.0 / n, atol=1e-9)


def test_zero_view_weights_do_not_depend_on_risk_aversion():
    returns_df = _make_synthetic_returns()
    pd.testing.assert_series_equal(
        allocate(returns_df, risk_aversion=1.0), allocate(returns_df, risk_aversion=10.0), atol=1e-9
    )


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------

def test_bullish_view_increases_target_weight_and_bearish_decreases_it():
    returns_df = _make_synthetic_returns()
    tickers = list(returns_df.columns)
    base = allocate(returns_df)
    bull = allocate(returns_df, views=_absolute_view(tickers, "ASSET_3", 0.002))
    bear = allocate(returns_df, views=_absolute_view(tickers, "ASSET_3", -0.002))
    assert bull["ASSET_3"] > base["ASSET_3"] + 0.01
    assert bear["ASSET_3"] < base["ASSET_3"]
    assert bull.sum() == pytest.approx(1.0, abs=1e-9)


def test_view_matching_the_prior_changes_nothing():
    returns_df = _make_synthetic_returns()
    tickers = list(returns_df.columns)
    cov = returns_df.cov().values
    pi = implied_equilibrium_returns(cov, np.full(len(cov), 1 / len(cov)))
    view = _absolute_view(tickers, "ASSET_5", float(pi[5]))
    mu, _ = posterior(cov, pi, view.P.values, view.Q, tau=DEFAULT_TAU)
    np.testing.assert_allclose(mu, pi, atol=1e-15)


def test_very_uncertain_view_is_ignored_and_certain_view_is_honored():
    returns_df = _make_synthetic_returns()
    tickers = list(returns_df.columns)
    cov = returns_df.cov().values
    pi = implied_equilibrium_returns(cov, np.full(len(cov), 1 / len(cov)))
    P = _absolute_view(tickers, "ASSET_1", 0.0).P.values

    mu_vague, _ = posterior(cov, pi, P, [0.01], omega=[[1e6]])
    np.testing.assert_allclose(mu_vague, pi, atol=1e-10)

    mu_sure, _ = posterior(cov, pi, P, [0.01], omega=[[1e-14]])
    assert mu_sure[1] == pytest.approx(0.01, abs=1e-8)


def test_inconsistent_view_shapes_raise():
    returns_df = _make_synthetic_returns()
    tickers = list(returns_df.columns)
    P = _absolute_view(tickers, "ASSET_1", 0.0).P
    with pytest.raises(ValueError, match="Inconsistent"):
        allocate(returns_df, views=BlackLittermanViews(P=P, Q=[0.001, 0.002]))


# ---------------------------------------------------------------------------
# Real data
# ---------------------------------------------------------------------------

def test_real_data_zero_views_produces_valid_near_equal_weights():
    returns_df = pd.read_parquet("data/processed/returns.parquet").tail(252)
    weights = allocate(returns_df)
    assert weights.sum() == pytest.approx(1.0, abs=1e-9)
    assert (weights >= 0).all()
    assert set(weights.index) == set(returns_df.columns)
    assert (weights - 1.0 / len(weights)).abs().sum() <= 2 * DEFAULT_TAU / (1 + DEFAULT_TAU) + 1e-9


def test_real_data_view_moves_weight_in_expected_direction():
    returns_df = pd.read_parquet("data/processed/returns.parquet").tail(252)
    tickers = list(returns_df.columns)
    base = allocate(returns_df)
    tilted = allocate(returns_df, views=_absolute_view(tickers, "INFY", 0.002))
    assert tilted["INFY"] > base["INFY"]
    assert tilted.sum() == pytest.approx(1.0, abs=1e-9)
