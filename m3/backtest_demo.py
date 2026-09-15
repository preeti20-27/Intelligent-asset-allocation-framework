from state import PortfolioState
from monitor import check_portfolio


portfolio_states = [
    PortfolioState(
        date="2026-01-05",
        weights={"RELIANCE": 0.20, "TCS": 0.20, "INFY": 0.20,
                 "GOLDBEES": 0.20, "GILT": 0.20},
        weekly_return=0.01,
        monthly_return=0.02,
        drawdown_from_peak=-0.03,
        volatility=0.12,
        max_weight=0.20,
        portfolio_value=100000
    ),

    PortfolioState(
        date="2026-01-12",
        weights={"RELIANCE": 0.20, "TCS": 0.20, "INFY": 0.20,
                 "GOLDBEES": 0.20, "GILT": 0.20},
        weekly_return=-0.02,
        monthly_return=-0.01,
        drawdown_from_peak=-0.08,
        volatility=0.18,
        max_weight=0.20,
        portfolio_value=98000
    ),

    PortfolioState(
        date="2026-01-19",
        weights={"RELIANCE": 0.50, "TCS": 0.10, "INFY": 0.10,
                 "GOLDBEES": 0.10, "GILT": 0.20},
        weekly_return=-0.06,
        monthly_return=-0.12,
        drawdown_from_peak=-0.18,
        volatility=0.30,
        max_weight=0.50,
        portfolio_value=82000
    )
]


for state in portfolio_states:

    action = check_portfolio(state)

    print(
        f"{state.date} | "
        f"Portfolio Value: ₹{state.portfolio_value:,.0f} | "
        f"Action: {action}"
    )