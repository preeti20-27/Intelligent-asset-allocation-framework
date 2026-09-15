"""Transaction cost model applied to portfolio turnover between rebalances."""

from __future__ import annotations

import pandas as pd

EQUITY_COST_BPS = 20.0  # per side, on the 10 equities
ETF_COST_BPS = 10.0  # per side, on GOLDBEES / LTGILTBEES

ETF_TICKERS = {"GOLDBEES", "LTGILTBEES"}


def cost_rate_for(ticker: str) -> float:
    """Per-side transaction cost rate (as a fraction, not bps) for ``ticker``."""
    bps = ETF_COST_BPS if ticker in ETF_TICKERS else EQUITY_COST_BPS
    return bps / 10_000.0


def transaction_cost(old_weights: pd.Series, new_weights: pd.Series) -> float:
    """Total transaction cost, as a fraction of portfolio value, of rebalancing
    from ``old_weights`` to ``new_weights``.

    Cost = sum over assets of ``|new_weight - old_weight| * per-side cost
    rate for that asset``. ``old_weights``/``new_weights`` are aligned on
    the union of their tickers; a ticker missing from one side is treated
    as 0 weight there (e.g. the very first rebalance, effectively buying
    in from cash).
    """
    all_tickers = old_weights.index.union(new_weights.index)
    old_aligned = old_weights.reindex(all_tickers, fill_value=0.0)
    new_aligned = new_weights.reindex(all_tickers, fill_value=0.0)

    turnover = (new_aligned - old_aligned).abs()
    rates = pd.Series([cost_rate_for(t) for t in all_tickers], index=all_tickers)

    return float((turnover * rates).sum())
