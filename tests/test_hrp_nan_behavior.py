"""Pins down allocate()'s actual behavior when returns_dataframe contains NaNs.

allocate() checks each asset's non-null fraction over the window it
actually uses (the full sample, or the most recent ``window`` rows) before
doing anything else, via ``_validate_min_non_null_fraction`` in
allocator/hrp.py. That check exists because ``compute_correlation``/
``compute_covariance`` compute pandas correlation/covariance, which default
to *pairwise-complete-observations*: for each pair of columns, rows where
either is NaN are dropped just for that pair. When different assets have
different missing patterns, different pairs get estimated from different
row subsets, and the resulting correlation matrix is not guaranteed to be
positive semi-definite -- it can silently produce a nonsensical "distance"
matrix for clustering.

Two outcomes, verified empirically below:

1. Gaps that keep every asset at or above ``MIN_NON_NULL_FRACTION`` (90%
   non-null, i.e. up to ~10% missing) -- pairwise-complete correlation/
   covariance can still be computed for every asset pair, so ``allocate``
   runs to completion silently: no exception, no NaN in the output
   weights, no asset dropped. The affected asset's correlation/covariance
   estimates are quietly based on fewer sample points than other assets' --
   nothing signals this to the caller. Gap *length* alone doesn't matter
   here, only the resulting non-null fraction.

2. Any asset below that 90% threshold -- ``allocate`` raises ``ValueError``
   itself, up front, naming the offending asset(s), their non-null
   percentage, and how many rows are missing. (Before this check existed,
   an entirely-missing column instead fell through to
   ``scipy.spatial.distance.squareform`` and raised an opaque, third-party
   ``ValueError`` with no mention of NaNs; the explicit check now catches
   that case, and every less-severe-but-still-over-threshold case, earlier
   and more clearly.)

Net effect: allocate() never silently returns NaN weights, and never
silently drops an asset. See the docstring on allocate() in
allocator/hrp.py for the summarized contract.
"""

import numpy as np
import pandas as pd
import pytest

from allocator.hrp import MIN_NON_NULL_FRACTION, allocate


def _make_base_returns(n_assets=6, n_days=300, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tickers = [f"A{i}" for i in range(n_assets)]
    factor_loadings = rng.normal(0, 1, size=(n_assets, 3))
    factors = rng.normal(0, 0.01, size=(n_days, 3))
    idio = rng.normal(0, 0.01, size=(n_days, n_assets))
    returns = factors @ factor_loadings.T + idio
    return pd.DataFrame(returns, columns=tickers)


def test_short_nan_gap_is_silently_tolerated():
    # 3-day gap in one column, out of a 300-day window: 1% missing, well
    # under the 90%-non-null threshold.
    returns_df = _make_base_returns()
    returns_df.loc[10:12, "A2"] = np.nan

    weights = allocate(returns_df)

    assert set(weights.index) == set(returns_df.columns)  # no asset silently dropped
    assert not weights.isnull().any()
    assert weights.sum() == pytest.approx(1.0, abs=1e-9)


def test_long_gap_under_threshold_is_still_silently_tolerated():
    # 25-day gap in one column out of 300 days = 8.3% missing -- still
    # under the 10%-missing / 90%-non-null threshold, so still silently
    # tolerated. Confirms gap *length* alone isn't the trigger, only the
    # resulting non-null fraction is.
    returns_df = _make_base_returns()
    returns_df.loc[10:34, "A2"] = np.nan

    weights = allocate(returns_df)

    assert set(weights.index) == set(returns_df.columns)
    assert not weights.isnull().any()
    assert weights.sum() == pytest.approx(1.0, abs=1e-9)


def test_gap_just_over_threshold_raises_valueerror_naming_asset():
    # 40-day gap out of 300 days = 13.3% missing (86.7% non-null), just
    # over the 10%-missing threshold. Confirms the new check fires based on
    # the non-null fraction itself, not just the pathological all-NaN case.
    returns_df = _make_base_returns()
    returns_df.loc[10:49, "A2"] = np.nan

    with pytest.raises(ValueError) as exc_info:
        allocate(returns_df)

    message = str(exc_info.value)
    assert "A2" in message
    assert "90%" in message
    assert "40/300" in message  # rows missing / total rows


def test_entirely_missing_asset_raises_valueerror_naming_asset():
    # Pathological case: one column has no usable data at all. Caught by
    # the same up-front check, not by the scipy squareform error it used
    # to fall through to.
    returns_df = _make_base_returns()
    returns_df["A2"] = np.nan

    with pytest.raises(ValueError) as exc_info:
        allocate(returns_df)

    message = str(exc_info.value)
    assert "A2" in message
    assert "300/300" in message


def test_multiple_offending_assets_are_all_named():
    returns_df = _make_base_returns()
    returns_df.loc[0:59, "A1"] = np.nan  # 20% missing
    returns_df.loc[0:89, "A4"] = np.nan  # 30% missing

    with pytest.raises(ValueError) as exc_info:
        allocate(returns_df)

    message = str(exc_info.value)
    assert "A1" in message
    assert "A4" in message


def test_threshold_matches_module_constant():
    # Sanity check that the 90% figure quoted throughout these tests and
    # allocate()'s docstring actually matches the constant the code uses,
    # so this file can't silently drift out of sync with allocator/hrp.py.
    assert MIN_NON_NULL_FRACTION == 0.9
