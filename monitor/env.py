"""Reinforcement Learning Risk-Monitoring Environment.

This environment models the portfolio risk monitoring process as a Markov
Decision Process (MDP). At each daily trading step:
1. The agent observes the current PortfolioState (drawdown, volatility, trailing
   returns, concentration, ETF/equity exposure).
2. The agent selects a risk action (HOLD, DERISK_MILD, DERISK_STRONG).
3. The RiskOverlay adjusts base allocator weights (e.g. from HRP) accordingly.
4. The environment steps forward one trading day, applies transaction costs,
   updates the portfolio value, and emits a risk-adjusted reward.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from backtest.costs import transaction_cost
from monitor.base import BaseRiskEnv
from monitor.overlay import (
    ACTION_DERISK_MILD,
    ACTION_DERISK_STRONG,
    ACTION_HOLD,
    ACTION_NAMES,
    DEFAULT_ETF_TICKERS,
    RiskOverlay,
)
from monitor.monitor import DRAWDOWN_THRESHOLD
from monitor.state import PortfolioState

OBSERVATION_DIM = 8
OBSERVATION_FEATURE_NAMES = [
    "drawdown_from_peak",      # Portfolio drawdown as fraction in [-1.0, 0.0]
    "trailing_annualized_vol",  # 20-day annualized volatility
    "weekly_return",           # Trailing 5-day portfolio return
    "monthly_return",          # Trailing 21-day portfolio return
    "max_weight",              # Largest individual asset weight
    "equity_exposure",         # Total weight allocated to equities
    "etf_exposure",            # Total weight allocated to defensive ETFs
    "drawdown_breach_gap",     # Severity of breach past Phase 1 -15% threshold
]


def state_to_observation(state: PortfolioState, etf_tickers=DEFAULT_ETF_TICKERS) -> np.ndarray:
    """Convert a PortfolioState into a fixed-size normalized observation vector.

    Parameters
    ----------
    state : PortfolioState
        Current portfolio state.
    etf_tickers : set of str
        Tickers considered defensive ETFs.

    Returns
    -------
    np.ndarray, shape (8,)
        Normalized feature vector for the RL agent.
    """
    etf_set = set(etf_tickers)
    weights = pd.Series(state.weights)

    etf_exposure = float(weights[weights.index.isin(etf_set)].sum())
    equity_exposure = float(weights[~weights.index.isin(etf_set)].sum())

    # Breach gap: negative if drawdown is worse than DRAWDOWN_THRESHOLD (-0.15)
    breach_gap = min(0.0, state.drawdown_from_peak - DRAWDOWN_THRESHOLD)

    obs = np.array(
        [
            np.clip(state.drawdown_from_peak, -1.0, 0.0),
            np.clip(state.volatility, 0.0, 2.0),
            np.clip(state.weekly_return, -1.0, 1.0),
            np.clip(state.monthly_return, -1.0, 1.0),
            np.clip(state.max_weight, 0.0, 1.0),
            np.clip(equity_exposure, 0.0, 1.0),
            np.clip(etf_exposure, 0.0, 1.0),
            np.clip(breach_gap, -1.0, 0.0),
        ],
        dtype=np.float32,
    )
    return obs


def _trailing_return(value_history: pd.Series, lookback_days: int) -> float:
    """Compute trailing return over lookback_days."""
    if len(value_history) < 2:
        return 0.0
    lookback = min(lookback_days, len(value_history) - 1)
    return float(value_history.iloc[-1] / value_history.iloc[-1 - lookback] - 1.0)


def _trailing_annualized_vol(value_history: pd.Series, lookback_days: int) -> float:
    """Compute 20-day annualized volatility of daily returns."""
    if len(value_history) < 3:
        return 0.0
    daily_returns = value_history.pct_change().dropna()
    lookback = min(lookback_days, len(daily_returns))
    window = daily_returns.iloc[-lookback:]
    return float(window.std() * np.sqrt(252))


class RiskMonitorEnv(BaseRiskEnv):
    """Gym-style trading environment for the RL Risk-Monitoring Overlay.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Daily returns panel for universe tickers (dates as index).
    base_weights_df : pd.DataFrame
        Daily base allocator weights (dates as index, tickers as columns).
    overlay : RiskOverlay, optional
        Overlay instance that applies discrete actions.
    initial_capital : float, default=10,000,000.0
        Starting portfolio capital.
    drawdown_penalty_coef : float, default=2.0
        Penalty weight for drawdowns, especially past -15%.
    vol_penalty_coef : float, default=0.5
        Penalty weight for annualized volatility above target.
    vol_target : float, default=0.20
        Target annualized volatility threshold.
    action_switch_penalty : float, default=0.0005
        Small friction penalty to discourage unnecessary daily action thrashing.
    return_scale : float, default=100.0
        Scaling factor applied to net daily returns in the reward.
    """

    def __init__(
        self,
        returns_df: pd.DataFrame,
        base_weights_df: pd.DataFrame,
        overlay: RiskOverlay | None = None,
        initial_capital: float = 10_000_000.0,
        drawdown_penalty_coef: float = 2.0,
        vol_penalty_coef: float = 0.5,
        vol_target: float = 0.20,
        action_switch_penalty: float = 0.0005,
        return_scale: float = 100.0,
    ):
        self.returns_df = returns_df
        self.base_weights_df = base_weights_df
        self.overlay = overlay or RiskOverlay()
        self.initial_capital = float(initial_capital)

        # Ensure aligned dates
        common_dates = self.returns_df.index.intersection(self.base_weights_df.index)
        if len(common_dates) < 5:
            raise ValueError(f"Insufficient overlapping dates between returns and base weights: {len(common_dates)}")

        self.dates = pd.DatetimeIndex(sorted(common_dates))
        self.drawdown_penalty_coef = drawdown_penalty_coef
        self.vol_penalty_coef = vol_penalty_coef
        self.vol_target = vol_target
        self.action_switch_penalty = action_switch_penalty
        self.return_scale = return_scale

        # Observation & action space specifications
        self.observation_dim = OBSERVATION_DIM
        self.n_actions = len(ACTION_NAMES)

        # State tracking variables
        self.current_idx = 0
        self.current_value = self.initial_capital
        self.peak_value = self.initial_capital
        self.value_history = pd.Series(dtype=float)
        self.current_weights = pd.Series(dtype=float)
        self.prev_action = ACTION_HOLD
        self.current_state: PortfolioState | None = None

    def reset(self, seed: int | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to the beginning of the episode."""
        if seed is not None:
            np.random.seed(seed)

        self.current_idx = 0
        self.current_value = self.initial_capital
        self.peak_value = self.initial_capital
        self.value_history = pd.Series(dtype=float)
        self.prev_action = ACTION_HOLD

        # Initialize day 0 weights from base weights projected through constraints
        first_date = self.dates[0]
        initial_base_w = self.base_weights_df.loc[first_date]
        self.current_weights = self.overlay.apply(initial_base_w, action=ACTION_HOLD)
        self.value_history[first_date] = self.current_value

        self.current_state = PortfolioState(
            date=str(pd.Timestamp(first_date).date()),
            weights=self.current_weights.to_dict(),
            weekly_return=0.0,
            monthly_return=0.0,
            drawdown_from_peak=0.0,
            volatility=0.0,
            max_weight=float(self.current_weights.max()),
            portfolio_value=self.current_value,
        )

        obs = state_to_observation(self.current_state)
        info = {
            "date": first_date,
            "portfolio_value": self.current_value,
            "drawdown": 0.0,
            "action": ACTION_HOLD,
        }
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Advance simulation by one trading day with the chosen risk action.

        Parameters
        ----------
        action : int
            Discrete risk action: 0 (HOLD), 1 (DERISK_MILD), 2 (DERISK_STRONG).

        Returns
        -------
        Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]
            (observation, reward, terminated, truncated, info)
        """
        if action not in ACTION_NAMES:
            raise ValueError(f"Invalid action {action}. Valid: {list(ACTION_NAMES.keys())}")

        if self.current_idx >= len(self.dates) - 1:
            # End of dataset reached
            obs = state_to_observation(self.current_state)
            return obs, 0.0, True, False, {"date": self.dates[self.current_idx]}

        date = self.dates[self.current_idx]
        next_date = self.dates[self.current_idx + 1]

        # 1. Overlay adjusts base weights according to action
        base_w = self.base_weights_df.loc[date]
        target_weights = self.overlay.apply(base_w, action=action, state=self.current_state)

        # 2. Transaction cost charged on turnover
        cost_fraction = transaction_cost(self.current_weights, target_weights)

        # 3. Portfolio daily return on next_date
        next_day_returns = self.returns_df.loc[next_date]
        aligned_w = target_weights.reindex(next_day_returns.index, fill_value=0.0)
        gross_return = float((next_day_returns * aligned_w).sum())
        net_return = gross_return - cost_fraction

        # 4. Update portfolio value and drawdown
        new_value = self.current_value * (1.0 + net_return)
        self.current_value = new_value
        self.value_history[next_date] = new_value
        self.peak_value = max(self.peak_value, new_value)
        self.current_weights = target_weights

        drawdown = new_value / self.peak_value - 1.0
        weekly_ret = _trailing_return(self.value_history, 5)
        monthly_ret = _trailing_return(self.value_history, 21)
        vol = _trailing_annualized_vol(self.value_history, 20)

        # 5. Formulate new PortfolioState
        self.current_state = PortfolioState(
            date=str(pd.Timestamp(next_date).date()),
            weights=target_weights.to_dict(),
            weekly_return=weekly_ret,
            monthly_return=monthly_ret,
            drawdown_from_peak=drawdown,
            volatility=vol,
            max_weight=float(target_weights.max()),
            portfolio_value=new_value,
        )

        # 6. Calculate risk-adjusted reward:
        #    Reward = Scaled Net Return
        #           - Drawdown penalty (progressive; severe when breaching Phase 1 -15% rule)
        #           - Volatility excess penalty (if vol > target)
        #           - Action-switching friction penalty
        scaled_return = net_return * self.return_scale

        # Progressive drawdown penalty
        # Base linear penalty for any drawdown:
        dd_base_penalty = self.drawdown_penalty_coef * abs(drawdown)
        # Severe penalty if breaching DRAWDOWN_THRESHOLD (-0.15):
        dd_breach_penalty = 0.0
        if drawdown < DRAWDOWN_THRESHOLD:
            breach_depth = abs(drawdown - DRAWDOWN_THRESHOLD)
            dd_breach_penalty = 10.0 * self.drawdown_penalty_coef * (breach_depth ** 2)

        # Volatility penalty if exceeding target
        vol_penalty = 0.0
        if vol > self.vol_target:
            vol_penalty = self.vol_penalty_coef * (vol - self.vol_target)

        # Action stability penalty
        switch_penalty = self.action_switch_penalty if action != self.prev_action else 0.0

        reward = float(scaled_return - dd_base_penalty - dd_breach_penalty - vol_penalty - switch_penalty)

        self.prev_action = action
        self.current_idx += 1
        terminated = self.current_idx >= len(self.dates) - 1

        obs = state_to_observation(self.current_state)
        info = {
            "date": next_date,
            "portfolio_value": new_value,
            "drawdown": drawdown,
            "volatility": vol,
            "net_return": net_return,
            "cost_fraction": cost_fraction,
            "action": action,
            "action_name": ACTION_NAMES[action],
        }

        return obs, reward, terminated, False, info
