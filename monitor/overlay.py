"""Risk-monitoring overlay for dynamic portfolio de-risking.

This module provides the execution layer of the RL risk-monitoring system.
It does NOT replace the underlying asset allocator (e.g. HRP or VI-HRP) or
attempt to predict unconstrained asset returns. Instead, it takes the base
weights selected by the allocator and applies defensive tilts (shifting mass
from volatile equities to safe-haven ETFs) when the RL monitor detects
elevated risk (such as worsening drawdown, volatility spikes, or concentration).

All adjusted weight vectors are strictly projected to satisfy portfolio constraints:
- Long-only: weights >= 0
- Maximum 25% per individual equity
- Minimum 5% each in GOLDBEES and LTGILTBEES
- Sum to 1.0
"""

from __future__ import annotations

from typing import Callable, Iterable, Set

import numpy as np
import pandas as pd

from monitor.base import BaseOverlay
from monitor.state import PortfolioState

# Standard project constraints
DEFAULT_MAX_EQUITY_WEIGHT = 0.25
DEFAULT_MIN_ETF_WEIGHT = 0.05
DEFAULT_ETF_TICKERS: Set[str] = {"GOLDBEES", "LTGILTBEES"}

# Discrete overlay actions
ACTION_HOLD = 0           # Keep allocator's target weights as-is
ACTION_DERISK_MILD = 1    # Shift 15% of equity exposure to defensive ETFs
ACTION_DERISK_STRONG = 2  # Shift 35% of equity exposure to defensive ETFs

ACTION_NAMES = {
    ACTION_HOLD: "HOLD",
    ACTION_DERISK_MILD: "DERISK_MILD",
    ACTION_DERISK_STRONG: "DERISK_STRONG",
}

# Shift fractions corresponding to each action
ACTION_SHIFT_FRACTIONS = {
    ACTION_HOLD: 0.0,
    ACTION_DERISK_MILD: 0.15,
    ACTION_DERISK_STRONG: 0.35,
}


def project_constraints(
    weights: pd.Series,
    max_equity_weight: float = DEFAULT_MAX_EQUITY_WEIGHT,
    min_etf_weight: float = DEFAULT_MIN_ETF_WEIGHT,
    etf_tickers: Iterable[str] = DEFAULT_ETF_TICKERS,
    max_iterations: int = 50,
) -> pd.Series:
    """Project portfolio weights onto constraints using iterative water-filling.

    Guarantees:
    1. Long-only (w_i >= 0).
    2. Equities capped at max_equity_weight (default 0.25).
    3. ETFs floored at min_etf_weight (default 0.05).
    4. Weights sum exactly to 1.0.

    Uses iterative constraint projection (clip -> freeze -> redistribute remaining
    mass across free weights) so bounded weights are never re-scaled past bounds.
    """
    original = weights.astype(float)
    if (original < 0).any():
        raise ValueError("Weights cannot be negative prior to projection.")
    if original.sum() <= 0:
        raise ValueError("Sum of weights must be positive.")

    # Normalize initial input defensively
    w = original / original.sum()

    etf_set = set(etf_tickers)
    lower_bounds = pd.Series(0.0, index=w.index)
    upper_bounds = pd.Series(np.inf, index=w.index)

    for ticker in w.index:
        if ticker in etf_set:
            lower_bounds[ticker] = min_etf_weight
        else:
            upper_bounds[ticker] = max_equity_weight

    # Initial clip
    clipped = w.clip(lower=lower_bounds, upper=upper_bounds)
    frozen = (clipped != w) | np.isclose(clipped, lower_bounds) | np.isclose(clipped, upper_bounds)
    w = clipped

    for _ in range(max_iterations):
        frozen_sum = float(w[frozen].sum())
        if frozen_sum > 1.0 + 1e-9:
            raise ValueError(
                f"Infeasible constraints: bound-pinned weights sum to {frozen_sum:.6f} > 1.0"
            )

        free_mask = ~frozen
        if not free_mask.any():
            break  # All weights are pinned

        remaining_mass = 1.0 - frozen_sum
        free_sum = float(w[free_mask].sum())

        if free_sum <= 0:
            # Distribute remaining mass equally if all free weights are zero
            candidate = pd.Series(remaining_mass / free_mask.sum(), index=w.index[free_mask])
        else:
            candidate = w[free_mask] * (remaining_mass / free_sum)

        newly_violated = (candidate < lower_bounds[free_mask] - 1e-12) | (
            candidate > upper_bounds[free_mask] + 1e-12
        )

        if not newly_violated.any():
            w.loc[free_mask] = candidate
            break  # Converged

        violating_tickers = candidate.index[newly_violated]
        w.loc[violating_tickers] = candidate[newly_violated].clip(
            lower=lower_bounds[violating_tickers], upper=upper_bounds[violating_tickers]
        )
        frozen.loc[violating_tickers] = True
    else:
        raise RuntimeError(f"Constraint projection did not converge within {max_iterations} iterations.")

    # Final floating-point drift cleanup
    return w / w.sum()


class RiskOverlay(BaseOverlay):
    """Reinforcement-learning driven risk overlay for portfolio weight adjustments.

    Parameters
    ----------
    max_equity_weight : float, default=0.25
        Maximum permitted weight for any non-ETF equity.
    min_etf_weight : float, default=0.05
        Minimum required weight for each defensive ETF ticker.
    etf_tickers : set of str, default={'GOLDBEES', 'LTGILTBEES'}
        Set of defensive tickers eligible to absorb de-risked capital.
    mild_shift : float, default=0.15
        Fraction of equity exposure shifted to defensive ETFs under ACTION_DERISK_MILD.
    strong_shift : float, default=0.35
        Fraction of equity exposure shifted to defensive ETFs under ACTION_DERISK_STRONG.
    constraint_fn : Callable[[pd.Series], pd.Series], optional
        Optional custom constraint projection function. If None, uses project_constraints.
    """

    def __init__(
        self,
        max_equity_weight: float = DEFAULT_MAX_EQUITY_WEIGHT,
        min_etf_weight: float = DEFAULT_MIN_ETF_WEIGHT,
        etf_tickers: Iterable[str] = DEFAULT_ETF_TICKERS,
        mild_shift: float = 0.15,
        strong_shift: float = 0.35,
        constraint_fn: Callable[[pd.Series], pd.Series] | None = None,
    ):
        self.max_equity_weight = max_equity_weight
        self.min_etf_weight = min_etf_weight
        self.etf_tickers = set(etf_tickers)
        self.mild_shift = mild_shift
        self.strong_shift = strong_shift
        self.constraint_fn = constraint_fn or project_constraints

    def apply(
        self,
        base_weights: pd.Series,
        action: int,
        state: PortfolioState | None = None,
    ) -> pd.Series:
        """Apply risk decision to base weights and project onto portfolio constraints.

        Parameters
        ----------
        base_weights : pd.Series
            Raw weights output by the allocator (e.g. HRP, VI-HRP).
        action : int
            Discrete risk action:
            0 (HOLD) -> maintain allocator weights
            1 (DERISK_MILD) -> shift mild_shift fraction from equities to ETFs
            2 (DERISK_STRONG) -> shift strong_shift fraction from equities to ETFs
        state : PortfolioState, optional
            Current portfolio state (for logging or context).

        Returns
        -------
        pd.Series
            Adjusted, constraint-compliant target weights summing to 1.0.
        """
        if not isinstance(base_weights, pd.Series):
            base_weights = pd.Series(base_weights, dtype=float)

        w = base_weights.astype(float).copy()
        if w.sum() <= 0:
            raise ValueError(f"Base weights sum must be > 0, got {w.sum()}.")

        # Normalize to sum 1.0 first
        w = w / w.sum()

        if action not in ACTION_NAMES:
            raise ValueError(f"Unknown overlay action {action}. Valid actions: {list(ACTION_NAMES.keys())}")

        shift_fraction = 0.0
        if action == ACTION_DERISK_MILD:
            shift_fraction = self.mild_shift
        elif action == ACTION_DERISK_STRONG:
            shift_fraction = self.strong_shift

        # If HOLD or no shift needed, directly project base weights onto bounds
        if shift_fraction > 0.0:
            equity_mask = ~w.index.isin(self.etf_tickers)
            etf_mask = w.index.isin(self.etf_tickers)

            equity_mass = float(w[equity_mask].sum())
            if equity_mass > 0 and etf_mask.any():
                mass_to_shift = equity_mass * shift_fraction
                # Proportionally scale down all equities
                w.loc[equity_mask] = w.loc[equity_mask] * (1.0 - shift_fraction)

                # Distribute the shifted mass across eligible ETFs
                # If ETFs currently hold mass, distribute proportionally; otherwise equally
                current_etf_mass = float(w[etf_mask].sum())
                if current_etf_mass > 0:
                    w.loc[etf_mask] += mass_to_shift * (w.loc[etf_mask] / current_etf_mass)
                else:
                    n_etfs = int(etf_mask.sum())
                    w.loc[etf_mask] += mass_to_shift / n_etfs

        # Always enforce portfolio constraints through projection
        return self.constraint_fn(w)
