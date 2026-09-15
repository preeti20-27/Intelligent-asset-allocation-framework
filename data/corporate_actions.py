from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


MANUAL_CORPORATE_ACTIONS = [
    {
        "ticker": "BHARTIARTL",
        "action": "rights",
        "status": "required",
        "source": "NSE/BSE corporate-actions CSV",
    },
    {
        "ticker": "RELIANCE",
        "action": "rights",
        "status": "required",
        "source": "NSE/BSE corporate-actions CSV",
    },
]


def parse_corporate_actions(path: str | Path) -> pd.DataFrame:
    """Parse a manually downloaded NSE/BSE corporate-actions CSV."""
    actions = pd.read_csv(path)
    actions.columns = [
        str(column).strip().lower().replace(" ", "_").replace("-", "_")
        for column in actions
    ]
    aliases = {
        "symbol": "ticker",
        "security_code": "ticker",
        "ex_date": "date",
        "purpose": "action",
        "corporate_action": "action",
        "face_value": "face_value",
        "ratio": "ratio",
        "dividend_per_share": "amount",
        "dividend": "amount",
        "issue_price": "issue_price",
    }
    actions = actions.rename(columns=aliases)
    required = {"ticker", "date", "action"}
    missing = required - set(actions.columns)
    if missing:
        raise ValueError(f"Corporate-actions data is missing columns: {sorted(missing)}")
    actions["ticker"] = actions["ticker"].astype(str).str.upper().str.replace(".NS", "", regex=False)
    actions["date"] = pd.to_datetime(actions["date"])
    if "record_date" in actions:
        actions["record_date"] = pd.to_datetime(actions["record_date"], errors="coerce")
    actions["action"] = actions["action"].astype(str).str.lower().str.strip()
    if "amount" in actions:
        actions["amount"] = pd.to_numeric(actions["amount"], errors="coerce")
    if "issue_price" in actions:
        actions["issue_price"] = pd.to_numeric(actions["issue_price"], errors="coerce")
    return actions.sort_values(["ticker", "date"]).reset_index(drop=True)


def adjust_ohlcv(
    ohlcv: pd.DataFrame,
    actions: pd.DataFrame,
    ticker: str | None = None,
    price_column: str = "close",
) -> pd.Series:
    """Return a backward-adjusted close series for splits, bonuses, and dividends."""
    if price_column not in ohlcv:
        raise ValueError(f"OHLCV data is missing {price_column!r}")
    prices = ohlcv[price_column].astype(float).copy()
    prices.index = pd.to_datetime(prices.index)
    relevant = actions.copy()
    if ticker is not None and "ticker" in relevant:
        relevant = relevant[relevant["ticker"].str.upper() == ticker.upper()]
    factors = pd.Series(1.0, index=prices.index)
    split_actions = relevant[relevant["action"].isin({"split", "bonus"})]
    for _, action in split_actions.sort_values("date", ascending=False).iterrows():
        event_date = pd.Timestamp(action["date"])
        mask = factors.index < event_date
        factors.loc[mask] = factors.loc[mask] * _ratio(action)

    split_adjusted_prices = prices * factors
    rights_actions = relevant[
        (relevant["action"] == "rights")
        & (relevant["date"] >= prices.index.min())
    ]
    for _, action in rights_actions.sort_values("date", ascending=False).iterrows():
        event_date = pd.Timestamp(action["date"])
        # Rights pricing uses the raw bhavcopy close immediately before ex-date.
        prior_prices = prices.loc[prices.index < event_date]
        if prior_prices.empty:
            continue
        prior_close = prior_prices.iloc[-1]
        new_shares, held_shares = _rights_ratio(action)
        issue_price = float(action["issue_price"])
        factor = (
            held_shares * prior_close + new_shares * issue_price
        ) / ((held_shares + new_shares) * prior_close)
        mask = factors.index < event_date
        factors.loc[mask] = factors.loc[mask] * factor
        split_adjusted_prices = prices * factors

    dividend_actions = relevant[
        (relevant["action"].isin({"dividend", "cash dividend"}))
        & (relevant["date"] >= prices.index.min())
    ]
    for _, action in dividend_actions.sort_values("date", ascending=False).iterrows():
        event_date = pd.Timestamp(action["date"])
        prior_prices = split_adjusted_prices.loc[split_adjusted_prices.index < event_date]
        if prior_prices.empty:
            continue
        prior_close = prior_prices.iloc[-1]
        amount = float(action.get("amount", 0) or 0)
        factor = 1.0 - (amount / prior_close) if prior_close else 1.0
        mask = factors.index < event_date
        factors.loc[mask] = factors.loc[mask] * factor

    return prices * factors


def _rights_ratio(action: pd.Series) -> tuple[float, float]:
    value = str(action["ratio"]).replace(" ", "")
    if ":" not in value:
        raise ValueError("Rights actions require a new-shares:held-shares ratio")
    new_shares, held_shares = value.split(":", 1)
    return float(new_shares), float(held_shares)


def _ratio(action: pd.Series) -> float:
    if pd.notna(action.get("ratio")):
        value = str(action["ratio"]).replace(" ", "")
        if ":" in value:
            new, old = value.split(":", 1)
            return float(old) / float(new)
        return float(value)
    new = float(action.get("new_shares", 1))
    old = float(action.get("old_shares", 1))
    return old / new