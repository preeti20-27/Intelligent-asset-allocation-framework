"""Equal-weight portfolio allocation -- the simplest possible baseline.

No covariance, correlation, or clustering is involved, so unlike
``allocators.hrp.allocate`` this has no minimum-history requirement: it
only looks at ``returns_df``'s columns (the tickers), never its rows, so
it works even on a single-row (or zero-row) input.
"""

from __future__ import annotations

import pandas as pd


def allocate(returns_df: pd.DataFrame) -> pd.Series:
    """Split weight equally across every ticker (column) in ``returns_df``.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Daily returns, dates as rows, tickers as columns. Only
        ``returns_df.columns`` is used.

    Returns
    -------
    pd.Series
        Weights indexed by ticker, each equal to ``1 / n_tickers``,
        summing to exactly 1.0.
    """
    tickers = returns_df.columns
    n_tickers = len(tickers)
    if n_tickers == 0:
        raise ValueError("Cannot allocate: returns_df has no columns (tickers).")

    return pd.Series(1.0 / n_tickers, index=tickers)
