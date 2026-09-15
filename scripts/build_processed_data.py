from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from data.corporate_actions import parse_corporate_actions
from data.providers.bhavcopy import BhavcopyProvider
from data.universe import members
from validate_bhavcopy_vs_yfinance import (
    _adjust_bhavcopy,
    _download_actions,
    validate,
)


START = "2015-01-01"
END = "2026-09-15"
PROCESSED_DIR = Path("data/processed")


def main() -> None:
    tickers = members(START)
    raw = BhavcopyProvider().fetch(tickers, START, END)
    actions = _download_actions(tickers, END)
    manual = parse_corporate_actions("data/manual_corporate_actions.csv")
    actions = pd.concat([actions, manual], ignore_index=True, sort=False)
    prices = _adjust_bhavcopy(raw, actions, raw).unstack("ticker")
    prices.index.name = "date"
    prices.columns.name = "ticker"
    returns = prices.pct_change().dropna(how="all")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(PROCESSED_DIR / "prices.parquet")
    returns.to_parquet(PROCESSED_DIR / "returns.parquet")

    comparison = validate(START, END)
    comparison.to_csv(PROCESSED_DIR / "bhavcopy_vs_yfinance.csv")
    shutil.copy2(PROCESSED_DIR / "bhavcopy_vs_yfinance.csv", "data/raw/bhavcopy_vs_yfinance.csv")

    print(f"prices.parquet rows={len(prices)} columns={len(prices.columns)}")
    print(f"returns.parquet rows={len(returns)} columns={len(returns.columns)}")
    print(f"bhavcopy_vs_yfinance.csv rows={len(comparison)}")


if __name__ == "__main__":
    main()