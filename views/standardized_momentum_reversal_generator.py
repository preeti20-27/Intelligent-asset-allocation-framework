"""Standardized cross-sectional momentum+reversal view generator for VI-HRP
-- a third ``views.generator.ViewGenerator``.

Why this exists
----------------
Both prior generators (``views.momentum_generator``,
``views.mean_reversion_generator``) built their tilt directly from a
RAW, non-standardized cluster-average compounded return gap: average
each side's per-asset compounded return over a lookback, diff the two
sides, scale by a constant. Two problems with that, per Asness,
Moskowitz & Pedersen, "Value and Momentum Everywhere" (2013) ("AMP") and
Baltas, "Multi-Asset Multi-Factor Strategies" (2019):

1. **No volatility standardization.** A raw return gap treats a 5%
   trailing move in GOLDBEES (annualized vol roughly 12-15% on this
   data) the same as a 5% move in a high-beta equity (annualized vol
   30%+). AMP's cross-asset signals are risk-adjusted *per asset* before
   anything else, specifically so a signal means the same thing
   ("how many of this asset's own standard deviations did it move")
   regardless of which asset it's attached to.
2. **Cluster-averaging before standardizing, not after.** The old
   generators average raw per-asset returns across a cluster first, then
   compare clusters -- diluting a strong individual-asset signal inside
   a cluster of otherwise-quiet assets, and conflating "this asset moved
   a lot" with "this asset moved a lot *relative to its own volatility*."
   AMP's approach -- and this module's -- is to compute a per-asset
   signal, standardize it, cross-sectionally rank/z-score it against the
   *entire universe* (not just the two sides of one split), and only
   *then* aggregate to whichever side of a tree split it falls on.

This generator combines two legs, both per-asset, both cross-sectionally
z-scored across the full universe before any cluster aggregation:

- **Momentum leg**: classic 12-month-minus-1-month ("12-1") momentum,
  adapted to this daily data: the compounded return over the 252-to-21
  trading-day-ago window (~12 months, skipping the most recent ~1 month
  to avoid short-term reversal contaminating the momentum signal -- the
  standard Jegadeesh-Titman adaptation AMP also uses), divided by that
  asset's own realized volatility over the same formation window.
- **Reversal leg**: AMP's long-horizon reversal, formation period
  nominally 3-5 years; adapted here to 3 years (756 trading days) -- see
  "Data history" below for why, and for the one asset (LTGILTBEES) where
  this window isn't fully populated even at this length. The compounded
  return over that window, divided by the same-window realized
  volatility, *negated* (betting the long-run winner reverts, not
  continues) -- also cross-sectionally z-scored.

The two z-scores are averaged per asset into one combined score; cluster
means of that combined score (not raw returns) are what get diffed at
each tree split to produce ``tilt_strength``, exactly mirroring how
``views.momentum_generator.generate_momentum_views`` turns a diffed
quantity into a View -- only the quantity being diffed has changed, from
a raw return gap to a standardized, cross-sectionally-ranked combined
z-score gap.

Data history (adapts the task's "use your judgment, document it")
--------------------------------------------------------------------
``data/processed/returns.parquet`` spans 2015-01-02 to 2026-09-11 (2898
rows, ~11.6 years). 11 of the 12 tickers have data for the full range.
LTGILTBEES listed 2016-07-25 and trades sparsely for a while after. At
the very first rebalance this generator is ever asked to score
(2019-04-05, the backtest's own warm-up-end date), a 756-day (3-year)
reversal window reaches back to 2016-03-17 -- fully covered for 11 of 12
tickers, but only 49.3% non-null for LTGILTBEES specifically (the window
starts 4 months before it existed, plus its own post-listing sparsity).
A true 5-year (1260-day) window was tried and rejected: it would reach
back to 2015 for every rebalance in 2019-2020, well before LTGILTBEES
existed at all, making its reversal signal undefined (not just
low-confidence) for a meaningful stretch of the backtest's early
in-sample period. 3 years (756 days) was the next candidate, itself an
adaptation, not the paper's 3-5yr range chosen freely; calibration (see
"Calibration" below) subsequently selected 2 years (504 days,
``REVERSAL_FORMATION_DAYS``'s actual value) from a grid that also
included 756 and 1008 -- which, as a side effect, sidesteps the
LTGILTBEES partial-coverage issue even more than 756 days did (its
window at the first rebalance starts in early 2017, well after
LTGILTBEES's mid-2016 listing). Whichever window is used, any asset
without enough valid history in it
gets ``nan`` for that leg (excluded from that leg's cross-sectional
z-scoring, not given a fabricated value), and the resulting lower data
availability on the affected side already flows into ``confidence`` --
exactly the mechanism ``views.momentum_generator`` uses for the same
LTGILTBEES sparsity problem, reused here via the same private helper
(``_data_availability_fraction``).

Feeding this generator enough history
--------------------------------------
``allocators.vihrp.allocate`` and this generator are both handed
whatever ``returns_df`` the caller passes in -- there is no hidden extra
data source. A ``reversal_formation_days``-length reversal window (504
by default) therefore needs the caller to pass at least that many rows
of trailing history, which is MORE than
``backtest.engine.BacktestEngine``'s default 252-row rebalance window.
``scripts/backtest_standardized_generator.py`` handles this by giving
this generator a longer trailing slice of the full returns panel than
what's used to build the tree/covariance (which stays at the standard
252 rows, identical to every prior generator's setup, so the plain-HRP
baseline and tree structure are unchanged) -- see that script's
docstring for exactly how.

Calibration (``REVERSAL_FORMATION_DAYS=504``, ``TILT_SCALE=2.0``)
--------------------------------------------------------------------
Grid-searched (3 ``reversal_formation_days`` values [504, 756, 1008] x 7
``tilt_scale`` values [0.30, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00] = 21
combinations) against ``data/processed/returns.parquet`` using the real
backtest engine, with the IDENTICAL in-sample tuning window
(2019-04-05 to 2022-12-31) / out-of-sample holdout (2023-01-02 to
2026-09-11) split and the IDENTICAL saturation-rate-<50%-before-ranking
selection rule ``scripts/calibrate_momentum_generator.py`` established.
``momentum_formation_days``/``momentum_skip_days`` (231/21) were held
fixed, not grid-searched -- that's the literature's own 12-1 convention,
not a value tuned to this dataset. Leg-combination weighting (currently
a fixed 50/50 average) was NOT grid-searched either -- a known remaining
gap, not silently dropped. Full grid: ``scripts/calibrate_standardized_generator.py``;
full results table: ``scripts/output/standardized_generator_calibration.csv``.

``TILT_SCALE`` here is in *z-score units* (a cluster-mean combined
z-score gap, typically much smaller in magnitude than an individual
asset's own z-score once a cluster has more than one member -- averaging
shrinks variance), not the raw-return-gap units
``views.momentum_generator.TILT_SCALE`` uses, so that module's
calibrated constant was never expected to carry over numerically, and
doesn't.

**What the grid actually showed, and why the result is a warning, not a
win.** Every row's saturation rate and in-sample Sharpe moved together,
monotonically, within each ``reversal_formation_days``: a HIGHER
``tilt_scale`` (a WEAKER tilt) always gave BOTH a lower saturation rate
AND a higher in-sample Sharpe, all the way to the grid's own upper
boundary (``tilt_scale=2.0``, 0% saturation) -- the naive
(unfiltered) in-sample-Sharpe winner and the saturation-filtered winner
were therefore the same row; no naive-winner-rejection was needed this
time, unlike ``views.momentum_generator``'s original calibration. On its
own, that could look like "larger tilt_scale is better, pick the
biggest" -- but a monotonic trend that keeps improving all the way to
"tilt as little as possible" is actually evidence the grid never found a
tilt strength genuinely worth using, only ever-smaller doses of
something that hurts. Directly confirming this: ``allocators.vihrp.allocate``
called with ZERO views at all (no signal, no tilt, just VI-HRP's own
tree-walk inverse-variance split) scores in-sample Sharpe 0.512 / OOS
Sharpe 2.580 on these same two windows -- BOTH HIGHER than this
calibrated generator's own 0.502 / 2.429. The calibrated "optimum" is
weaker than simply turning the signal off. Calibration here did its job
correctly (it is NOT overfit to a lucky in-sample result, and it
correctly rejects a would-be naive winner's saturation problem were
there one) -- it just correctly reports that, across every strength
tested, standardizing the inputs didn't turn this into a signal worth
using; it only made clear how much better "approximately no signal" is
than the raw version's uncalibrated-but-stronger tilt.

Before vs. after calibration (``scripts/backtest_standardized_generator.py``,
current committed output ``scripts/output/standardized_generator_comparison.csv``)
--------------------------------------------------------------------------------------
::

                          UNCALIBRATED (lookback 756, scale 1.0)   CALIBRATED (504, 2.0)
                          in-sample   OOS                          in-sample   OOS
    Plain HRP:              0.768    2.601                          0.768    2.601
    This generator:         0.303    2.065                          0.502    2.429
    Zero-view VI-HRP:         --       --                           0.512    2.580

                          subA        subB                         subA        subB
    Plain HRP:              0.774     1.293                         0.774     1.293
    This generator:         0.137     1.367 (BEAT plain HRP)        0.432     1.153 (no longer beats it)

Calibration substantially narrows the full-period gap to plain HRP (in
-sample gap 0.465 -> 0.266, a 43% reduction; OOS gap 0.536 -> 0.172, a
68% reduction) -- genuine progress, not noise, and not limited to one
sub-window. But it does NOT close the gap, and the uncalibrated
version's one apparent "win" (subB) was a mirage: it came from a
stronger, uncalibrated tilt that happened to do well in that one
sub-period specifically, and evaporates (1.367 -> 1.153, now below plain
HRP's 1.293) once the tilt strength is chosen properly, from the WHOLE
in-sample window, instead of being whatever an earlier, unjustified
guess happened to produce. Combined with the zero-view-VI-HRP comparison
above, the honest reading is: this calibration closed most of the
AVOIDABLE gap (using a tilt strength that was simply too strong for no
good reason), not the REMAINING gap (no tested or extrapolated tilt
strength, down to approximately zero, beats plain HRP on this
universe/period). Treat this generator as a calibration-verified
negative result, not a tuned win: well-constructed, properly validated,
and still not a reason to prefer it over plain HRP here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tree.cluster_tree import ClusterTree
from views.momentum_generator import (
    MIN_TILT_STRENGTH_TO_EMIT,
    _data_availability_fraction,
    _momentum_gap_significance,
)
from views.view import View, ViewSet

TRADING_DAYS_PER_YEAR = 252

# 12-month-minus-1-month momentum, in trading days: ~12 months of
# formation (MOMENTUM_FORMATION_DAYS), skipping the most recent ~1 month
# (MOMENTUM_SKIP_DAYS) to avoid short-term reversal contamination.
MOMENTUM_FORMATION_DAYS = 231  # 252 - 21
MOMENTUM_SKIP_DAYS = 21

# Long-horizon reversal formation window. Calibrated (see module
# docstring's "Calibration" section) against data/processed/returns.parquet
# via scripts/calibrate_standardized_generator.py; 504 (2yr) rather than
# the originally-adapted 756 (3yr) placeholder.
REVERSAL_FORMATION_DAYS = 504

# A leg is undefined (nan) for an asset with fewer than this many valid
# (non-null) daily returns in its formation window -- not enough to
# estimate a compounded return or a volatility at all meaningfully.
MIN_VALID_DAYS_FOR_LEG = 20

# Need at least this many assets with a valid (non-nan) leg value on a
# given day to cross-sectionally z-score that leg at all.
MIN_OBSERVATIONS_FOR_CROSS_SECTION = 3

# Saturation scale, in combined cross-sectional z-score units (NOT the
# same units as views.momentum_generator.TILT_SCALE's raw return-gap
# scale). Calibrated -- see module docstring's "Calibration" section.
TILT_SCALE = 2.0


def _leg_raw_value(formation_window: pd.Series) -> float:
    """Volatility-standardized compounded return over ``formation_window``,
    or ``nan`` if too few valid observations to trust a compounded-
    return/volatility estimate at all.
    """
    valid = formation_window.dropna()
    if len(valid) < MIN_VALID_DAYS_FOR_LEG:
        return float("nan")

    compounded = float((1.0 + valid).prod() - 1.0)
    annualized_vol = float(valid.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
    if not np.isfinite(annualized_vol) or np.isclose(annualized_vol, 0.0):
        return float("nan")

    return compounded / annualized_vol


def _momentum_raw_per_asset(
    returns_df: pd.DataFrame, tickers, formation_days: int, skip_days: int
) -> dict:
    """{ticker: vol-standardized 12-1-style momentum return}, nan where undefined."""
    raw = {}
    for ticker in tickers:
        series = returns_df[ticker]
        trimmed = series.iloc[: len(series) - skip_days] if skip_days > 0 else series
        formation_window = trimmed.tail(formation_days)
        raw[ticker] = _leg_raw_value(formation_window)
    return raw


def _reversal_raw_per_asset(returns_df: pd.DataFrame, tickers, formation_days: int) -> dict:
    """{ticker: NEGATED vol-standardized long-horizon return}, nan where undefined.

    Negated so that, like the momentum leg, a HIGHER raw value always
    means "more favorable for this asset" -- here, a long-run
    *underperformer* (low trailing return over ``formation_days``) gets
    a high raw value, betting its gap narrows.
    """
    raw = {}
    for ticker in tickers:
        series = returns_df[ticker]
        window_len = min(formation_days, len(series))
        formation_window = series.tail(window_len)
        value = _leg_raw_value(formation_window)
        raw[ticker] = -value if np.isfinite(value) else value
    return raw


def _cross_sectional_zscore(raw: dict) -> dict:
    """z-score ``raw``'s values across whichever keys have a finite value;
    nan values pass through as nan (excluded from the cross-section, not
    coerced to 0 -- a missing signal is not a neutral one).
    """
    values = pd.Series(raw, dtype=float)
    valid = values.dropna()
    if len(valid) < MIN_OBSERVATIONS_FOR_CROSS_SECTION:
        return {k: float("nan") for k in raw}

    mean = float(valid.mean())
    std = float(valid.std(ddof=1))
    if not np.isfinite(std) or np.isclose(std, 0.0):
        return {k: float("nan") for k in raw}

    z = (values - mean) / std
    return z.to_dict()


def _combined_cross_sectional_z(
    returns_df: pd.DataFrame,
    tickers,
    momentum_formation_days: int,
    momentum_skip_days: int,
    reversal_formation_days: int,
) -> dict:
    """Per-asset: average of the momentum z-score and the reversal
    z-score, each independently cross-sectionally z-scored across the
    full universe first. A leg that's nan for a given asset is dropped
    from that asset's average (not treated as 0), so an asset with only
    one well-defined leg still gets a signal from that leg alone, and an
    asset with neither leg defined gets nan (contributes nothing when a
    cluster mean is later taken, via pandas' skipna).
    """
    momentum_z = _cross_sectional_zscore(
        _momentum_raw_per_asset(returns_df, tickers, momentum_formation_days, momentum_skip_days)
    )
    reversal_z = _cross_sectional_zscore(
        _reversal_raw_per_asset(returns_df, tickers, reversal_formation_days)
    )

    combined = {}
    for ticker in tickers:
        parts = [v for v in (momentum_z[ticker], reversal_z[ticker]) if np.isfinite(v)]
        combined[ticker] = float(np.mean(parts)) if parts else float("nan")
    return combined


def _cluster_mean_z(combined_z: dict, assets) -> float:
    values = pd.Series({a: combined_z[a] for a in assets})
    return float(values.mean(skipna=True))  # nan only if every asset in `assets` is nan


def generate_standardized_momentum_reversal_views(
    returns_df: pd.DataFrame,
    tree: ClusterTree,
    momentum_formation_days: int = MOMENTUM_FORMATION_DAYS,
    momentum_skip_days: int = MOMENTUM_SKIP_DAYS,
    reversal_formation_days: int = REVERSAL_FORMATION_DAYS,
    tilt_scale: float = TILT_SCALE,
) -> ViewSet:
    """Generate one View per tree split from a standardized, cross-sectionally
    z-scored momentum+reversal combined signal.

    Unlike ``views.momentum_generator.generate_momentum_views`` (which
    diffs raw cluster-average compounded returns), this:

    1. Computes a vol-standardized 12-1 momentum return and a
       vol-standardized, negated long-horizon reversal return for EVERY
       individual asset in ``returns_df`` (not per cluster).
    2. Cross-sectionally z-scores each leg across the FULL universe
       (every ticker in ``returns_df.columns``), independently of the
       tree structure.
    3. Averages the two z-scores per asset into one combined score.
    4. Only then walks the tree and, at each split, diffs the two
       children's cluster-MEAN combined z-score (not raw returns) --
       everything from here on (favored-side selection, ``tilt_strength
       = min(|diff| / tilt_scale, 1.0)``, skip-if-below-threshold) is the
       same mechanism ``generate_momentum_views`` uses, just applied to
       a different (standardized) quantity.

    ``confidence`` is the worst of three factors, same philosophy as the
    other two generators: both sides' data availability over the
    (longer) reversal window, and the statistical significance of the
    momentum leg's raw-return differential over the momentum formation
    window (reusing ``views.momentum_generator._momentum_gap_significance``
    unchanged -- a two-tailed test on the underlying raw returns, not
    specific to how the momentum generator used it).

    Parameters
    ----------
    returns_df : pd.DataFrame
        Daily simple returns, dates as rows, tickers as columns. Needs
        enough trailing history for ``reversal_formation_days`` to be
        meaningful (see module docstring); gracefully uses less if not
        available (see ``_reversal_raw_per_asset``), at the cost of
        lower confidence for the affected asset(s).
    tree : ClusterTree
        Already-built tree over the same (or a compatible) universe.
    momentum_formation_days, momentum_skip_days : int, optional
        12-1-style momentum formation window and skip length, in trading
        days. Defaults to ``MOMENTUM_FORMATION_DAYS``/``MOMENTUM_SKIP_DAYS``
        (231/21, i.e. ~12 months skipping the most recent ~1 month).
    reversal_formation_days : int, optional
        Long-horizon reversal formation window, in trading days.
        Defaults to ``REVERSAL_FORMATION_DAYS`` (504, ~2 years --
        calibrated; see module docstring's "Calibration" section for why
        not AMP's 3-5yr, or this module's own originally-adapted 3yr).
    tilt_scale : float, optional
        Combined cross-sectional z-score gap (NOT a raw return gap --
        different units from ``views.momentum_generator.TILT_SCALE``)
        that saturates tilt_strength at 1.0. Defaults to ``TILT_SCALE``
        (2.0; calibrated -- see module docstring's "Calibration" section,
        including why this being the calibration's own selected
        "optimum" is a warning sign, not a success story).

    Returns
    -------
    ViewSet
    """
    tickers = list(returns_df.columns)
    combined_z = _combined_cross_sectional_z(
        returns_df, tickers, momentum_formation_days, momentum_skip_days, reversal_formation_days
    )

    views: list[View] = []

    def walk(node) -> None:
        if tree.is_leaf(node):
            return

        left, right = tree.children(node)
        left_assets = tree.assets_under(left)
        right_assets = tree.assets_under(right)

        z_left = _cluster_mean_z(combined_z, left_assets)
        z_right = _cluster_mean_z(combined_z, right_assets)
        diff = z_left - z_right

        if np.isfinite(diff) and diff != 0:
            if diff > 0:
                favored_assets, unfavored_assets = left_assets, right_assets
            else:
                favored_assets, unfavored_assets = right_assets, left_assets

            tilt_strength = min(abs(diff) / tilt_scale, 1.0)

            if tilt_strength >= MIN_TILT_STRENGTH_TO_EMIT:
                confidence = min(
                    _data_availability_fraction(returns_df, favored_assets, reversal_formation_days),
                    _data_availability_fraction(returns_df, unfavored_assets, reversal_formation_days),
                    _momentum_gap_significance(returns_df, left_assets, right_assets, momentum_formation_days),
                )
                views.append(
                    View(
                        node_assets=frozenset(favored_assets),
                        tilt_strength=tilt_strength,
                        confidence=confidence,
                        source=f"standardized_mom_rev_{momentum_formation_days}_{reversal_formation_days}d",
                    )
                )

        walk(left)
        walk(right)

    walk(tree.root)
    return ViewSet(views)
