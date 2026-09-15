"""Trading-day calendar utilities derived from the processed price panel.

Provides the actual NSE trading calendar -- as reflected in
``data/processed/prices.parquet``'s date index, which already has weekends
and market holidays removed -- and a weekly-rebalance-date schedule derived
from it, rather than a generic fixed-frequency calendar like pandas'
``"W-MON"`` (which doesn't know about market holidays: if Monday is a
holiday, ``"W-MON"`` either produces a date the market was never open on,
or silently drifts to the following Monday instead of that week's actual
first trading day).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_PRICES_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "processed" / "prices.parquet"
)


def load_trading_days(prices_path: str | Path = DEFAULT_PRICES_PATH) -> pd.DatetimeIndex:
    """The actual trading days present in the processed price panel."""
    prices = pd.read_parquet(prices_path)
    return pd.DatetimeIndex(sorted(prices.index.unique()))


def weekly_rebalance_dates(trading_days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The first trading day of each (ISO year, ISO week) present in ``trading_days``.

    Grouping by ISO (year, week) rather than calendar week correctly
    handles the ISO year boundary (the last days of December can belong to
    week 1 of the following ISO year, and vice versa), and grouping by the
    calendar's own trading days -- not a fixed frequency -- means a week
    where the market is closed on Monday rebalances on Tuesday instead.
    """
    trading_days = pd.DatetimeIndex(trading_days)
    iso = trading_days.isocalendar()
    grouped = pd.DataFrame(
        {
            "date": trading_days,
            "iso_year": iso["year"].values,
            "iso_week": iso["week"].values,
        }
    )
    first_per_week = grouped.groupby(["iso_year", "iso_week"])["date"].min()
    return pd.DatetimeIndex(sorted(first_per_week.values))
