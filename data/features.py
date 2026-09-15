from __future__ import annotations

import pandas as pd


class FeatureStore:
	def build(self, data: pd.DataFrame) -> pd.DataFrame:
		"""Placeholder for features consumed later by the ViewGenerator."""
		raise NotImplementedError
