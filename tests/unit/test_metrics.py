"""Tests for eval.metrics."""

import numpy as np
import pandas as pd
import pytest

from eval.metrics import max_drawdown, performance_summary, sharpe_ratio, total_return


def _make_up_down_recover_curve() -> pd.Series:
    """Equity curve: rises to a peak of 150, drops exactly 20% to a trough
    of 120, then partially recovers to 140 -- a small, fully
    hand-verifiable synthetic example.
    """
    values = [100.0, 110.0, 130.0, 150.0, 140.0, 120.0, 130.0, 140.0]
    dates = pd.bdate_range("2024-01-01", periods=len(values))
    return pd.Series(values, index=dates)


def test_total_return_matches_hand_computed_value():
    equity_curve = _make_up_down_recover_curve()
    # 140 / 100 - 1 = 0.40
    assert total_return(equity_curve) == pytest.approx(0.40, abs=1e-12)


def test_max_drawdown_catches_exactly_the_20_percent_drop():
    equity_curve = _make_up_down_recover_curve()
    result = max_drawdown(equity_curve)

    assert result.max_drawdown == pytest.approx(-0.20, abs=1e-12)
    # Peak (150) is index 3; trough (120) is index 5.
    dates = equity_curve.index
    assert result.peak_date == str(dates[3].date())
    assert result.trough_date == str(dates[5].date())


def test_max_drawdown_peak_is_the_peak_this_drawdown_fell_from():
    # The curve's global maximum-so-far only ever reaches 150 (at index 3);
    # the partial recovery to 140 at the end never creates a new peak.
    # This confirms peak_date identifies the peak the *reported* drawdown
    # fell from, not some unrelated peak elsewhere in the series.
    equity_curve = _make_up_down_recover_curve()
    result = max_drawdown(equity_curve)
    dates = equity_curve.index
    assert result.peak_date != str(dates[-1].date())


def test_max_drawdown_on_monotonically_increasing_curve_is_zero():
    equity_curve = pd.Series(
        [100.0, 110.0, 120.0, 130.0], index=pd.bdate_range("2024-01-01", periods=4)
    )
    result = max_drawdown(equity_curve)
    assert result.max_drawdown == pytest.approx(0.0, abs=1e-12)


def test_sharpe_ratio_matches_independent_reference_calculation():
    equity_curve = _make_up_down_recover_curve()
    daily_returns = equity_curve.pct_change().dropna()

    expected = daily_returns.mean() / daily_returns.std() * np.sqrt(252)

    assert sharpe_ratio(daily_returns) == pytest.approx(expected, rel=1e-9)


def test_sharpe_ratio_accounts_for_nonzero_risk_free_rate():
    equity_curve = _make_up_down_recover_curve()
    daily_returns = equity_curve.pct_change().dropna()

    risk_free_rate = 0.05  # 5% annual
    daily_rf = risk_free_rate / 252
    excess = daily_returns - daily_rf
    expected = excess.mean() / excess.std() * np.sqrt(252)

    assert sharpe_ratio(daily_returns, risk_free_rate=risk_free_rate) == pytest.approx(
        expected, rel=1e-9
    )


def test_sharpe_ratio_zero_for_constant_returns_not_nan_or_inf():
    # Zero standard deviation would otherwise divide by zero.
    daily_returns = pd.Series([0.001] * 10)
    assert sharpe_ratio(daily_returns) == 0.0


def test_sharpe_ratio_zero_for_too_few_returns():
    assert sharpe_ratio(pd.Series([0.01])) == 0.0
    assert sharpe_ratio(pd.Series(dtype=float)) == 0.0


def test_performance_summary_combines_all_three_and_formats_for_printing():
    equity_curve = _make_up_down_recover_curve()
    daily_returns = equity_curve.pct_change().dropna()

    summary = performance_summary(equity_curve, daily_returns)

    assert summary["total_return"] == pytest.approx(0.40, abs=1e-12)
    assert summary["max_drawdown"] == pytest.approx(-0.20, abs=1e-12)
    assert summary["sharpe_ratio"] == pytest.approx(
        daily_returns.mean() / daily_returns.std() * np.sqrt(252), rel=1e-9
    )

    assert "Total Return: 40.0%" in summary["summary"]
    assert "Max Drawdown: -20.0%" in summary["summary"]
    assert "Sharpe:" in summary["summary"]
    assert summary["peak_date"] in summary["summary"]
    assert summary["trough_date"] in summary["summary"]
