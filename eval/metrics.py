"""Performance metrics for a backtested equity curve.

All fractional metrics (``total_return``, ``sharpe_ratio``'s inputs,
``max_drawdown``) are plain fractions, not already-multiplied-by-100
percentages -- e.g. ``total_return`` returns ``0.423`` for a 42.3% gain,
and ``max_drawdown`` returns ``-0.182`` for an 18.2% drawdown.
``performance_summary`` is the one place percentage-formatted strings get
built, for printing.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd


class MaxDrawdownResult(NamedTuple):
    """Result of ``max_drawdown``: the drawdown value plus which dates it spans."""

    max_drawdown: float
    peak_date: str
    trough_date: str


def total_return(equity_curve: pd.Series) -> float:
    """Total return over the full equity curve, as a fraction (0.423 = 42.3%)."""
    if len(equity_curve) < 2:
        return 0.0
    return float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0)


def sharpe_ratio(
    daily_returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualized Sharpe ratio of ``daily_returns``.

    ``risk_free_rate`` is annual; it's converted to a per-period rate
    (``risk_free_rate / periods_per_year``) before being subtracted from
    each daily return. Uses the sample standard deviation (``ddof=1``,
    pandas' default). Returns 0.0 if there are fewer than 2 returns, or if
    the standard deviation is (numerically) zero, rather than dividing by
    zero.
    """
    if len(daily_returns) < 2:
        return 0.0

    daily_risk_free = risk_free_rate / periods_per_year
    excess_returns = daily_returns - daily_risk_free

    std = excess_returns.std()
    # Use a small absolute tolerance, not exact equality: floating-point
    # subtraction of nominally-identical returns rarely yields an exact
    # 0.0 standard deviation, but can leave a residue many orders of
    # magnitude smaller than any real return series' variation.
    if not np.isfinite(std) or np.isclose(std, 0.0, atol=1e-10):
        return 0.0

    return float(excess_returns.mean() / std * np.sqrt(periods_per_year))


def max_drawdown(equity_curve: pd.Series) -> MaxDrawdownResult:
    """Maximum peak-to-trough drawdown, as a negative fraction, with its dates.

    The trough is the date of the single worst drawdown; the peak is the
    date of the highest equity value *up to and including that trough*
    (not necessarily the global maximum of the whole curve, which could
    occur after the trough and would misidentify which peak this specific
    drawdown fell from).
    """
    if len(equity_curve) < 2:
        return MaxDrawdownResult(0.0, None, None)

    running_max = equity_curve.cummax()
    drawdown_series = equity_curve / running_max - 1.0

    trough_date = drawdown_series.idxmin()
    max_dd_value = float(drawdown_series.loc[trough_date])
    peak_date = equity_curve.loc[:trough_date].idxmax()

    return MaxDrawdownResult(
        max_dd_value,
        str(pd.Timestamp(peak_date).date()),
        str(pd.Timestamp(trough_date).date()),
    )


def performance_summary(equity_curve: pd.Series, daily_returns: pd.Series) -> dict:
    """Combine total return, Sharpe ratio, and max drawdown into one dict.

    Includes both the raw numeric values (for programmatic use) and a
    pre-formatted, human-readable multi-line string under ``"summary"``
    (for printing), e.g.::

        Total Return: 42.3%
        Sharpe: 1.15
        Max Drawdown: -18.2% (peak 2020-01-15, trough 2020-03-23)
    """
    ret = total_return(equity_curve)
    sharpe = sharpe_ratio(daily_returns)
    dd = max_drawdown(equity_curve)

    if dd.peak_date is None:
        drawdown_line = "Max Drawdown: n/a (not enough data)"
    else:
        drawdown_line = (
            f"Max Drawdown: {dd.max_drawdown * 100:.1f}% "
            f"(peak {dd.peak_date}, trough {dd.trough_date})"
        )

    summary_lines = [
        f"Total Return: {ret * 100:.1f}%",
        f"Sharpe: {sharpe:.2f}",
        drawdown_line,
    ]

    return {
        "total_return": ret,
        "sharpe_ratio": sharpe,
        "max_drawdown": dd.max_drawdown,
        "peak_date": dd.peak_date,
        "trough_date": dd.trough_date,
        "summary": "\n".join(summary_lines),
    }
