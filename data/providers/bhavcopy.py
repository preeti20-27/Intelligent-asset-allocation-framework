from __future__ import annotations

import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd


class BhavcopyProvider:
	"""Fetch and cache NSE daily OHLCV data through jugaad-data."""

	def __init__(
		self,
		raw_dir: str | Path = "data/raw",
		retries: int = 3,
		backoff_seconds: float = 1.0,
	) -> None:
		self.raw_dir = Path(raw_dir)
		self.retries = retries
		self.backoff_seconds = backoff_seconds

	def fetch(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
		"""Return daily OHLCV rows indexed by date for the requested tickers."""
		frames = [
			self._fetch_ticker(ticker, start, end).assign(ticker=ticker)
			for ticker in tickers
		]
		if not frames:
			return pd.DataFrame(index=pd.DatetimeIndex([], name="date"))
		return pd.concat(frames).set_index(["date", "ticker"]).sort_index()

	def _fetch_ticker(self, ticker: str, start: str, end: str) -> pd.DataFrame:
		cache_path = self.raw_dir / f"bhavcopy_ist_v2_{ticker}_{start}_{end}.csv"
		if cache_path.exists():
			cached = self._normalise(pd.read_csv(cache_path))
			if not cached["date"].duplicated().any():
				return cached

		try:
			from jugaad_data.nse import stock_df
		except ImportError as exc:
			raise RuntimeError(
				"BhavcopyProvider requires the jugaad-data package."
			) from exc

		frame = self._with_retry(stock_df, ticker, start, end)
		frame = self._normalise(frame)
		self.raw_dir.mkdir(parents=True, exist_ok=True)
		frame.to_csv(cache_path, index=False)
		return frame

	def _with_retry(self, fetcher: Any, ticker: str, start: str, end: str) -> Any:
		last_error: Exception | None = None
		for attempt in range(self.retries + 1):
			try:
				return fetcher(
					ticker,
					from_date=_as_date(start),
					to_date=_as_date(end),
				)
			except Exception as exc:  # provider errors include rate-limit responses
				last_error = exc
				if attempt == self.retries:
					break
				time.sleep(self.backoff_seconds * (2**attempt))
		raise RuntimeError(f"Unable to fetch bhavcopy data for {ticker}") from last_error

	@staticmethod
	def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
		result = frame.copy()
		if isinstance(result.index, pd.DatetimeIndex) and "date" not in result.columns:
			result = result.rename_axis("date").reset_index()
		result.columns = [str(column).strip().lower().replace(" ", "_") for column in result]
		aliases = {
			"date": "date",
			"symbol": "ticker",
			"series": "series",
			"open": "open",
			"high": "high",
			"low": "low",
			"close": "close",
			"last": "close",
			"prev_close": "previous_close",
			"tottrdqty": "volume",
			"total_traded_quantity": "volume",
			"volume": "volume",
			"tottrdval": "traded_value",
			"total_traded_value": "traded_value",
		}
		result = result.rename(columns=aliases)
		if "date" not in result:
			raise ValueError("Bhavcopy data must contain a date column")
		if "series" in result:
			result = result[result["series"].astype(str).str.upper() == "EQ"]
		# jugaad-data exposes NSE midnight IST as a naive UTC-like 18:30 value.
		result["date"] = (
			pd.to_datetime(result["date"])
			.dt.tz_localize("UTC")
			.dt.tz_convert("Asia/Kolkata")
			.dt.tz_localize(None)
			.dt.normalize()
		)
		result = result.set_index("date").sort_index().reset_index()
		required = ["date", "open", "high", "low", "close", "volume"]
		missing = [column for column in required if column not in result]
		if missing:
			raise ValueError(f"Bhavcopy data is missing columns: {missing}")
		return result[required + [column for column in ("traded_value",) if column in result]]


def _as_date(value: str) -> date:
	return datetime.fromisoformat(value).date()
