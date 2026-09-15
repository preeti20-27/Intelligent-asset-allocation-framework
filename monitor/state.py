from dataclasses import dataclass
from typing import Dict


@dataclass
class PortfolioState:
    date: str
    weights: Dict[str, float]
    weekly_return: float
    monthly_return: float
    drawdown_from_peak: float
    volatility: float
    max_weight: float
    portfolio_value: float

    def __post_init__(self):

        if not self.weights:
            raise ValueError("Portfolio weights cannot be empty.")

        if any(weight < 0 for weight in self.weights.values()):
            raise ValueError("Portfolio weights cannot be negative.")

        if self.portfolio_value < 0:
            raise ValueError("Portfolio value cannot be negative.")

        if self.volatility < 0:
            raise ValueError("Volatility cannot be negative.")
