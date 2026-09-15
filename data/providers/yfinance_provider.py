from __future__ import annotations

from pathlib import Path

import pandas as pd


class YFinanceProvider:
    """Cross-validation provider backed by Yahoo Finance."""

    def __init__(self, raw_dir: str | Path = "data/raw") -> None:
        self.raw_dir = Path(raw_dir)

    def fetch(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        """Return daily OHLCV rows indexed by date for NSE-suffixed tickers."""
        try:
            import yfinance as yf
        except ImportError as exc:
            raise RuntimeError(
                "YFinanceProvider requires the yfinance package."
            ) from exc

        yahoo_tickers = [ticker if ticker.endswith(".NS") else f"{ticker}.NS" for ticker in tickers]
        rows: list[pd.DataFrame] = []
        for yahoo_ticker in yahoo_tickers:
            frame = yf.download(
                yahoo_ticker,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
                group_by="ticker",
            )
            if frame.empty:
                raise RuntimeError(f"yfinance returned no data for {yahoo_ticker}")
            ticker_frame = _ticker_frame(frame, yahoo_ticker)
            ticker_frame = ticker_frame.loc[~ticker_frame.index.duplicated(keep="last")]
            ticker_frame = ticker_frame.rename(columns=str.lower)
            ticker_frame = ticker_frame.rename(columns={"adj close": "adjusted_close"})
            ticker_frame["ticker"] = yahoo_ticker.removesuffix(".NS")
            rows.append(ticker_frame.reset_index().rename(columns={"Date": "date"}))
        result = pd.concat(rows)
        result["date"] = pd.to_datetime(result["date"]).dt.normalize()
        return result.set_index(["date", "ticker"]).sort_index()


def _ticker_frame(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if isinstance(frame.columns, pd.MultiIndex):
        if ticker in frame.columns.get_level_values(0):
            return frame[ticker]
        if ticker in frame.columns.get_level_values(1):
            return frame.xs(ticker, axis=1, level=1)
    return frame