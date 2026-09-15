from __future__ import annotations

import argparse
import time

import pandas as pd
import yfinance as yf

from data.corporate_actions import adjust_ohlcv, parse_corporate_actions
from data.providers.bhavcopy import BhavcopyProvider
from data.providers.yfinance_provider import YFinanceProvider
from data.universe import members


def validate(start: str, end: str) -> pd.DataFrame:
    tickers = members(start)
    bhavcopy = BhavcopyProvider().fetch(tickers, start, end)
    yfinance = YFinanceProvider().fetch(tickers, start, end)
    actions = _download_actions(tickers, end)
    manual_path = "data/manual_corporate_actions.csv"
    if pd.io.common.file_exists(manual_path):
        actions = pd.concat(
            [actions, parse_corporate_actions(manual_path)],
            ignore_index=True,
            sort=False,
        )
    left = _adjust_bhavcopy(bhavcopy, actions, yfinance).rename("bhavcopy")
    right = _adjusted_close(yfinance).rename("yfinance")
    comparison = pd.concat([left, right], axis=1).dropna()
    comparison["absolute_diff"] = (comparison["bhavcopy"] - comparison["yfinance"]).abs()
    comparison["relative_diff_pct"] = (
        comparison["absolute_diff"] / comparison["yfinance"].abs() * 100
    )
    return comparison


def _adjusted_close(frame: pd.DataFrame) -> pd.Series:
    column = "adjusted_close" if "adjusted_close" in frame else "close"
    return frame[column]


def _download_actions(tickers: list[str], end: str) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for ticker in tickers:
        history = pd.DataFrame()
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                history = yf.Ticker(f"{ticker}.NS").actions
                if not history.empty:
                    break
            except Exception as exc:
                last_error = exc
            if attempt < 2:
                time.sleep(1.0 * (2**attempt))
        if history.empty:
            if last_error is not None:
                raise RuntimeError(f"Unable to fetch Yahoo actions for {ticker}") from last_error
            print(f"WARNING: Yahoo returned no corporate-action rows for {ticker}")
            continue
        for event_date, row in history.iterrows():
            if pd.Timestamp(event_date).tz is not None:
                event_date = pd.Timestamp(event_date).tz_localize(None)
            if event_date > pd.Timestamp(end):
                continue
            split = float(row.get("Stock Splits", 0) or 0)
            dividend = float(row.get("Dividends", 0) or 0)
            if split:
                records.append(
                    {"ticker": ticker, "date": event_date, "action": "split", "ratio": f"{split}:1"}
                )
            if dividend:
                records.append(
                    {"ticker": ticker, "date": event_date, "action": "dividend", "amount": dividend}
                )
    return pd.DataFrame(records, columns=["ticker", "date", "action", "ratio", "amount"])


def _adjust_bhavcopy(
    frame: pd.DataFrame,
    actions: pd.DataFrame,
    reference: pd.DataFrame,
) -> pd.Series:
    adjusted: list[pd.Series] = []
    for ticker, group in frame.groupby(level="ticker"):
        prices = group.droplevel("ticker").copy()
        if ticker == "GOLDBEES":
            # NSE changed the historical quote basis; infer 100x versus 1x by date.
            reference_prices = reference.loc[(slice(None), ticker), "close"].droplevel("ticker")
            ratio = prices["close"] / reference_prices.reindex(prices.index)
            prices.loc[ratio > 10, "close"] /= 100
        prices["adjusted_close"] = adjust_ohlcv(prices, actions, ticker=ticker)
        adjusted.append(prices["adjusted_close"].rename(ticker))
    return pd.concat(adjusted, keys=[series.name for series in adjusted], names=["ticker", "date"]).reorder_levels(["date", "ticker"]).sort_index()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("start")
    parser.add_argument("end")
    args = parser.parse_args()
    comparison = validate(args.start, args.end)
    comparison.to_csv("data/raw/bhavcopy_vs_yfinance.csv")
    discrepancies = comparison[comparison["relative_diff_pct"] > 1.0]
    print(f"Compared rows: {len(comparison)}")
    print(f"Rows with relative discrepancy >1%: {len(discrepancies)}")
    if not discrepancies.empty:
        print(discrepancies.to_string())