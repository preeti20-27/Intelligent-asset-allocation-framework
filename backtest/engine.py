"""Portfolio backtesting engine.

Ties together an injected allocator function (e.g. ``allocators.hrp.allocate``),
the weekly rebalance calendar (``backtest.calendar``), the transaction cost
model (``backtest.costs``), and portfolio constraints, to produce a daily
equity curve and one ``monitor.state.PortfolioState`` per trading day.

The engine never imports a concrete allocator itself -- ``run()`` takes
``allocator_fn`` as a parameter, so the caller decides which allocator to
use (plain HRP, VI-HRP, equal-weight, ...) and this module has no import
dependency on ``allocators.*`` at all.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from backtest.calendar import DEFAULT_PRICES_PATH, load_trading_days, weekly_rebalance_dates
from backtest.costs import ETF_TICKERS, transaction_cost
from monitor.state import PortfolioState

DEFAULT_RETURNS_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "processed" / "returns.parquet"
)

DEFAULT_INITIAL_CAPITAL = 10_000_000.0  # Rs 1 crore
DEFAULT_WINDOW = 252  # trading days; also used with 126, 504 for robustness checks

MAX_EQUITY_WEIGHT = 0.25
MIN_ETF_WEIGHT = 0.05
MAX_CONSTRAINT_ITERATIONS = 50

TRADING_DAYS_PER_YEAR = 252
WEEKLY_LOOKBACK_DAYS = 5
MONTHLY_LOOKBACK_DAYS = 21
VOLATILITY_LOOKBACK_DAYS = 20


def load_default_returns(returns_path: str | Path = DEFAULT_RETURNS_PATH) -> pd.DataFrame:
    """Load the processed daily-returns panel used for day-to-day portfolio tracking."""
    return pd.read_parquet(returns_path)


def _normalize_to_sum_one(weights: pd.Series) -> pd.Series:
    total = weights.sum()
    if total <= 0:
        raise ValueError(f"Cannot normalize weights: sum is {total!r} (must be > 0).")
    return weights / total


def apply_constraints(weights: pd.Series, max_iterations: int = MAX_CONSTRAINT_ITERATIONS) -> pd.Series:
    """Long-only, max 25% per equity, min 5% each in GOLDBEES/LTGILTBEES.

    Uses iterative constraint projection ("water-filling"), not a single
    clip-then-renormalize pass. A single clip-then-renormalize pass is
    *wrong*: renormalizing to force the sum back to 1.0 scales every
    weight uniformly, including the ones that were just clipped to their
    bound -- so a weight that was capped at 25% can get scaled back above
    25%, or one that was floored at 5% can get scaled back below 5%, by
    exactly the renormalization factor. This isn't a rare edge case: it
    happens whenever clipping moves the pre-renormalization sum away from
    1.0 at all, with the size of the re-violation roughly proportional to
    how much total clipping was needed (confirmed empirically in this
    project: a 2-equity case went from a 25% cap to ~40%, and even a
    single-asset floor adjustment on a realistic 12-asset universe
    slightly re-violated its own floor after renormalizing).

    The fix is to never again touch a weight once it's been pinned to a
    bound. Algorithm:

    1. Clip every weight to its bound (equities to ``[0, MAX_EQUITY_WEIGHT]``,
       GOLDBEES/LTGILTBEES to ``[MIN_ETF_WEIGHT, inf)``). Anything moved by
       this clip -- or already sitting exactly on a bound -- is *frozen*
       from here on.
    2. Redistribute the remaining mass (``1.0 - sum(frozen weights)``)
       proportionally across only the *unfrozen* weights, scaling them to
       preserve their relative proportions (this is what actually
       guarantees frozen weights are never touched again: unlike a global
       renormalization, the scale factor is only ever applied to the free
       set).
    3. If that redistribution would itself push a previously-free weight
       past its bound, clip and freeze that weight too, and go back to
       step 2 with a smaller free set (the mass it would have received is
       returned to the redistribution pool for the remaining free
       weights).
    4. Repeat until a pass freezes nothing new (converged) or
       ``max_iterations`` is reached.

    Each iteration either converges or freezes at least one more weight,
    so this always terminates within ``len(weights)`` iterations in
    practice; ``max_iterations`` (default 50) is a defensive cap. If
    convergence isn't reached within it, this raises rather than silently
    returning weights that may violate a bound -- as does a genuinely
    infeasible input (e.g. so many equities pinned at 25% that their sum
    alone already exceeds 1.0, leaving no room for the ETF floors).
    """
    original = weights.astype(float)

    lower_bounds = pd.Series(0.0, index=original.index)
    upper_bounds = pd.Series(np.inf, index=original.index)
    for ticker in original.index:
        if ticker in ETF_TICKERS:
            lower_bounds[ticker] = MIN_ETF_WEIGHT
        else:
            upper_bounds[ticker] = MAX_EQUITY_WEIGHT

    w = original.clip(lower=lower_bounds, upper=upper_bounds)

    # Freeze anything the initial clip moved, plus anything already
    # sitting exactly on its bound (e.g. an allocator that itself output
    # exactly 0.25 for some equity).
    frozen = (w != original) | np.isclose(w, lower_bounds) | np.isclose(w, upper_bounds)

    for _ in range(max_iterations):
        frozen_sum = float(w[frozen].sum())
        if frozen_sum > 1.0 + 1e-9:
            raise ValueError(
                f"Infeasible constraints: bound-pinned weights already sum to "
                f"{frozen_sum:.6f} > 1.0 ({list(w.index[frozen])}). No weight "
                "vector can satisfy every bound simultaneously."
            )

        free_mask = ~frozen
        if not free_mask.any():
            break  # everything is pinned; frozen_sum must be ~1.0 or we'd have raised above

        remaining_mass = 1.0 - frozen_sum
        free_sum = float(w[free_mask].sum())
        if free_sum <= 0:
            raise ValueError(
                "Cannot redistribute remaining weight: all free (unfrozen) "
                "weights are non-positive."
            )

        candidate = w[free_mask] * (remaining_mass / free_sum)

        newly_violated = (candidate < lower_bounds[free_mask] - 1e-12) | (
            candidate > upper_bounds[free_mask] + 1e-12
        )

        if not newly_violated.any():
            w.loc[free_mask] = candidate
            break  # converged

        violating_tickers = candidate.index[newly_violated]
        w.loc[violating_tickers] = candidate[newly_violated].clip(
            lower=lower_bounds[violating_tickers], upper=upper_bounds[violating_tickers]
        )
        frozen.loc[violating_tickers] = True
    else:
        raise RuntimeError(
            f"apply_constraints did not converge within {max_iterations} iterations."
        )

    # Final defensive renormalization for floating-point drift only -- by
    # construction the free set's redistributed mass already sums with the
    # frozen set to 1.0 up to rounding, so this is a negligible correction,
    # not a repeat of the large-scale renormalization this algorithm avoids.
    return w / w.sum()


def _trailing_return(value_history: pd.Series, lookback_days: int) -> float:
    """Trailing return over up to ``lookback_days`` trading days ending at the latest tracked date.

    Uses fewer days than ``lookback_days`` if the tracked history is
    shorter (e.g. near the start of a backtest), rather than raising or
    returning a placeholder for that warm-up period.
    """
    if len(value_history) < 2:
        return 0.0
    lookback = min(lookback_days, len(value_history) - 1)
    return float(value_history.iloc[-1] / value_history.iloc[-1 - lookback] - 1.0)


def _trailing_annualized_volatility(value_history: pd.Series, lookback_days: int) -> float:
    """Annualized std of daily portfolio returns over the trailing ``lookback_days``.

    Annualized via ``* sqrt(252)`` on the sample (``ddof=1``) daily std,
    using fewer days than ``lookback_days`` if not enough history has
    accumulated yet.
    """
    if len(value_history) < 3:
        return 0.0
    daily_returns = value_history.pct_change().dropna()
    lookback = min(lookback_days, len(daily_returns))
    window = daily_returns.iloc[-lookback:]
    return float(window.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


class BacktestEngine:
    """Tracks a single portfolio's daily value, applying transaction costs at rebalances.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Daily simple returns, dates as rows, tickers as columns -- the
        full universe panel (e.g. ``data/processed/returns.parquet``),
        used both to feed the allocator its rolling window and to mark
        the portfolio to market day by day.
    initial_capital : float, optional
        Starting portfolio value. Defaults to Rs 1 crore (10,000,000).
    """

    def __init__(self, returns_df: pd.DataFrame, initial_capital: float = DEFAULT_INITIAL_CAPITAL):
        self.returns_df = returns_df
        self.initial_capital = float(initial_capital)

        self._value_history = pd.Series(dtype=float)
        self._peak_value = self.initial_capital
        self._current_weights = pd.Series(dtype=float)

    def step(self, date, weights) -> PortfolioState:
        """Advance the tracked portfolio by one trading day, ending at ``date``.

        Must be called once per trading day, in chronological order,
        starting from the first day being tracked. If ``weights`` differs
        from the weights held the previous time ``step`` was called, a
        rebalance is assumed to have happened on ``date``, and the
        transaction cost of that turnover (``backtest.costs``) is charged
        against that day's return.

        ``weights`` (a ``pd.Series`` or plain ``dict``) is defensively
        renormalized to sum to 1.0 here even if the caller already applied
        ``apply_constraints`` -- this method does not rely on that having
        happened, per the requirement that ``PortfolioState.weights``
        always sums to 1.0.

        Returns
        -------
        monitor.state.PortfolioState
        """
        weights = _normalize_to_sum_one(pd.Series(weights, dtype=float))

        day_returns = self.returns_df.loc[date]
        aligned_weights = weights.reindex(day_returns.index, fill_value=0.0)
        gross_return = float((day_returns * aligned_weights).sum())

        cost_fraction = transaction_cost(self._current_weights, weights)
        net_return = gross_return - cost_fraction

        prev_value = self._value_history.iloc[-1] if len(self._value_history) else self.initial_capital
        new_value = prev_value * (1.0 + net_return)

        self._value_history.loc[date] = new_value
        self._peak_value = max(self._peak_value, new_value)
        self._current_weights = weights

        drawdown_from_peak = new_value / self._peak_value - 1.0

        return PortfolioState(
            date=str(pd.Timestamp(date).date()),
            weights=weights.to_dict(),
            weekly_return=_trailing_return(self._value_history, WEEKLY_LOOKBACK_DAYS),
            monthly_return=_trailing_return(self._value_history, MONTHLY_LOOKBACK_DAYS),
            drawdown_from_peak=drawdown_from_peak,
            volatility=_trailing_annualized_volatility(self._value_history, VOLATILITY_LOOKBACK_DAYS),
            max_weight=float(weights.max()),
            portfolio_value=new_value,
        )

    def run(
        self,
        start_date,
        end_date,
        allocator_fn: Callable[[pd.DataFrame], pd.Series],
        window: int = DEFAULT_WINDOW,
        rebalance_dates=None,
    ):
        """Run the backtest over ``[start_date, end_date]``, rebalancing weekly.

        At each rebalance date, ``allocator_fn`` is called with the
        trailing ``window``-day slice of ``self.returns_df`` ending the
        trading day *before* the rebalance date (the rebalance date's own
        return is excluded, to avoid look-ahead), and its raw output is
        passed through ``apply_constraints`` before being held constant
        until the next rebalance date.

        Parameters
        ----------
        allocator_fn : callable
            ``allocator_fn(returns_window_df) -> pd.Series`` of weights
            indexed by ticker (e.g. ``allocators.hrp.allocate``).
        window : int, optional
            Rolling lookback window fed to ``allocator_fn``, in trading
            days. Defaults to 252; 126 and 504 are also supported for
            robustness checks.
        rebalance_dates : optional
            Pre-computed rebalance dates (e.g. from
            ``backtest.calendar.weekly_rebalance_dates``). If not given,
            computed from the processed price panel's own trading
            calendar (``backtest.calendar.load_trading_days`` /
            ``DEFAULT_PRICES_PATH``) -- pass this explicitly when
            ``self.returns_df`` doesn't share that calendar (e.g. in
            tests using a synthetic date range).

        Returns
        -------
        (equity_curve: pd.Series, states: list[PortfolioState])
        """
        trading_days = self.returns_df.index
        trading_days = trading_days[
            (trading_days >= pd.Timestamp(start_date)) & (trading_days <= pd.Timestamp(end_date))
        ]

        if rebalance_dates is None:
            all_trading_days = load_trading_days(DEFAULT_PRICES_PATH)
            rebalance_dates = weekly_rebalance_dates(all_trading_days)
        rebalance_dates = set(pd.DatetimeIndex(rebalance_dates))

        current_weights = None
        states = []

        for date in trading_days:
            if current_weights is None or date in rebalance_dates:
                history = self.returns_df.loc[:date]
                window_returns = history.iloc[:-1].tail(window)
                raw_weights = allocator_fn(window_returns)
                current_weights = apply_constraints(raw_weights)

            state = self.step(date, current_weights)
            states.append(state)

        return self._value_history.copy(), states
