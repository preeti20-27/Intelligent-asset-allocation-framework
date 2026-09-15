from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


STATIC_UNIVERSE = [
	"RELIANCE",
	"HDFCBANK",
	"INFY",
	"TCS",
	"ITC",
	"SUNPHARMA",
	"LT",
	"MARUTI",
	"TATASTEEL",
	"BHARTIARTL",
	"GOLDBEES",
	"LTGILTBEES",
]
LIQUIDITY_FALLBACK = "GILT5YBEES"


@dataclass
class Universe:
	history: pd.DataFrame | None = None

	def members(self, date: str | pd.Timestamp) -> list[str]:
		return STATIC_UNIVERSE.copy()

	def liquidity_check(self, ticker: str, min_avg_daily_value: float = 1e7) -> bool:
		if self.history is None:
			raise ValueError("Universe liquidity_check requires available price history")
		frame = _ticker_history(self.history, ticker)
		if not {"close", "volume"}.issubset(frame.columns):
			raise ValueError("History must contain close and volume columns")
		average_daily_value = (frame["close"] * frame["volume"]).mean()
		return bool(average_daily_value >= min_avg_daily_value)


_default_universe = Universe()


def members(date: str | pd.Timestamp) -> list[str]:
	return _default_universe.members(date)


def liquidity_check(
	ticker: str,
	min_avg_daily_value: float = 1e7,
	history: pd.DataFrame | None = None,
) -> bool:
	"""Flag liquidity; use ``GILT5YBEES`` if ``LTGILTBEES`` fails."""
	return Universe(history).liquidity_check(ticker, min_avg_daily_value)


def _ticker_history(history: pd.DataFrame, ticker: str) -> pd.DataFrame:
	if isinstance(history.index, pd.MultiIndex) and "ticker" in history.index.names:
		return history.xs(ticker, level="ticker")
	if isinstance(history.columns, pd.MultiIndex) and ticker in history.columns.get_level_values(0):
		return history[ticker]
	return history
