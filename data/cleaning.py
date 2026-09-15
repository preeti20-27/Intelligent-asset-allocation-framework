from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class CleaningLog:
	records: list[dict[str, Any]] = field(default_factory=list)

	def add(self, action: str, **details: Any) -> None:
		self.records.append({"action": action, **details})


@dataclass
class CleaningResult:
	data: pd.DataFrame
	log: CleaningLog


class CleaningPipeline:
	def clean(self, prices: pd.DataFrame) -> CleaningResult:
		result = prices.copy().sort_index()
		log = CleaningLog()
		before = result.copy()
		result = result.ffill(limit=2)
		for date in result.index:
			for ticker in result.columns:
				old_value = before.at[date, ticker]
				new_value = result.at[date, ticker]
				if pd.isna(old_value) and pd.notna(new_value):
					log.add("forward_fill", date=date.isoformat(), ticker=ticker, value=float(new_value))
		missing_fraction = result.isna().mean(axis=1)
		dropped = result.index[missing_fraction > 0.20]
		for date in dropped:
			log.add("date_dropped", date=date.isoformat(), missing_fraction=float(missing_fraction.loc[date]))
		result = result.loc[missing_fraction <= 0.20]
		returns = result.pct_change()
		clipped = returns.clip(-0.25, 0.25)
		for date in clipped.index:
			for ticker in clipped.columns:
				old_return = returns.at[date, ticker]
				new_return = clipped.at[date, ticker]
				if pd.notna(old_return) and old_return != new_return:
					log.add(
						"return_winsorized",
						date=date.isoformat(),
						ticker=ticker,
						old_value=float(old_return),
						new_value=float(new_return),
					)
		result = _rebuild_prices(result, clipped)
		return CleaningResult(result, log)


def clean(prices: pd.DataFrame) -> CleaningResult:
	return CleaningPipeline().clean(prices)


def _rebuild_prices(prices: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
	rebuilt = prices.copy()
	for column in prices:
		first_valid = prices[column].first_valid_index()
		if first_valid is None:
			continue
		rebuilt.loc[first_valid:, column] = prices.loc[first_valid, column] * (
			1 + returns.loc[first_valid:, column].fillna(0)
		).cumprod()
	return rebuilt
