from state import PortfolioState
from monitor import check_portfolio
from rules import HOLD, WARN_DRAWDOWN, WARN_CONCENTRATION
import pytest


def make_state(drawdown, max_weight):
    return PortfolioState(
        date="2026-09-15",
        weights={
            "RELIANCE": 0.20,
            "TCS": 0.20,
            "INFY": 0.20,
            "GOLDBEES": 0.20,
            "GILT": 0.20
        },
        weekly_return=0.01,
        monthly_return=0.03,
        drawdown_from_peak=drawdown,
        volatility=0.15,
        max_weight=max_weight,
        portfolio_value=100000
    )


def test_normal_portfolio():
    state = make_state(-0.05, 0.20)
    assert check_portfolio(state) == HOLD


def test_drawdown_warning():
    state = make_state(-0.16, 0.20)
    assert check_portfolio(state) == WARN_DRAWDOWN


def test_concentration_warning():
    state = make_state(-0.05, 0.45)
    assert check_portfolio(state) == WARN_CONCENTRATION


def test_zero_drawdown():
    state = make_state(0.0, 0.20)
    assert check_portfolio(state) == HOLD


def test_exact_drawdown_threshold():
    state = make_state(-0.15, 0.20)
    assert check_portfolio(state) == WARN_DRAWDOWN


def test_empty_weights_rejected():
    with pytest.raises(ValueError):
        PortfolioState(
            date="2026-09-15",
            weights={},
            weekly_return=0.01,
            monthly_return=0.03,
            drawdown_from_peak=-0.05,
            volatility=0.15,
            max_weight=0.20,
            portfolio_value=100000
        )


def test_negative_weight_rejected():
    with pytest.raises(ValueError):
        PortfolioState(
            date="2026-09-15",
            weights={"RELIANCE": -0.20},
            weekly_return=0.01,
            monthly_return=0.03,
            drawdown_from_peak=-0.05,
            volatility=0.15,
            max_weight=0.20,
            portfolio_value=100000
        )


def test_negative_volatility_rejected():
    with pytest.raises(ValueError):
        PortfolioState(
            date="2026-09-15",
            weights={"RELIANCE": 1.0},
            weekly_return=0.01,
            monthly_return=0.03,
            drawdown_from_peak=-0.05,
            volatility=-0.15,
            max_weight=1.0,
            portfolio_value=100000
        )
