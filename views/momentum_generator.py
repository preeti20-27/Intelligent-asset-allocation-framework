"""Momentum-based view generator for VI-HRP -- the first ``views.generator.ViewGenerator``.

A simple trailing-return momentum heuristic, not a sophisticated signal
model, but calibrated against real data rather than left as a round-number
placeholder.

Calibration (``TILT_SCALE=0.20``, ``LOOKBACK=40``)
----------------------------------------------------
Grid-searched (5 lookbacks x 6 tilt_scales = 30 combinations) against
``data/processed/returns.parquet`` using the real backtest engine
(``backtest.engine.BacktestEngine``, weekly rebalance, real transaction
costs/constraints), scored on realized VI-HRP Sharpe ratio vs. plain HRP.
To guard against overfitting the ~7.5 years of usable history, the search
was split into an IN-SAMPLE tuning window (2019-04-05 to 2022-12-31) used
to pick the winner, and a held-out OUT-OF-SAMPLE window (2023-01-02 to
2026-09-11) used only to report how that choice generalizes -- never to
select it. Full grid, exact dates, and selection code:
``scripts/calibrate_momentum_generator.py``; full results table (every
combination x both windows):
``scripts/output/momentum_generator_calibration.csv``.

Two findings shaped the final choice:

1. **The naive "max in-sample Sharpe" winner was pathological.** Picking
   strictly by in-sample Sharpe selects ``(lookback=40, tilt_scale=0.10)``
   -- but that combination saturates (``tilt_strength == 1.0``) on 95.9%
   of rebalances, i.e. it's barely different from an always-max-tilt rule
   and makes ``tilt_strength`` almost entirely uninformative, which is
   the exact failure mode this calibration exists to fix. The selection
   rule therefore first excludes any combination with a saturation rate
   >= 50% (tilt_strength must actually vary, not pin to its ceiling most
   of the time), *then* picks the max in-sample Sharpe (ties broken by
   shallower in-sample max drawdown) among what's left. That is
   ``(lookback=40, tilt_scale=0.20)`` -- saturates only 35.5% of the time.
2. **Even calibrated, this momentum signal did not beat plain HRP
   out-of-sample.** Plain HRP's OOS Sharpe over 2023-2026 was 2.601 (an
   unusually strong, low-volatility period for the baseline); VI-HRP with
   the chosen (40, 0.20) scored 1.170 OOS, in-sample Sharpe 0.669 vs.
   plain HRP's in-sample 0.768. No grid cell's OOS Sharpe reached plain
   HRP's. This calibration makes the signal internally well-behaved
   (graded, not saturating, and its confidence now reflects statistical
   significance -- see below) rather than an arbitrary placeholder; it
   does not establish that this momentum heuristic adds risk-adjusted
   value over plain HRP on this universe/period. Treat VI-HRP + momentum
   views as a documented, calibrated research signal, not a demonstrated
   improvement -- see this module's and the calibration script's
   docstrings for the numbers, and decide accordingly before using it
   live.

Conforms to ``views.generator.ViewGenerator``: callable as
``generate_momentum_views(returns_df, tree)`` (``lookback`` and
``tilt_scale`` are keyword arguments with calibrated defaults, not part
of that contract) -- see ``views/generator.py`` for how a future
alternative signal plugs into VI-HRP the same way.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from tree.cluster_tree import ClusterTree
from views.view import View, ViewSet

# Calibrated against data/processed/returns.parquet (see module docstring
# and scripts/calibrate_momentum_generator.py): the trailing compounded
# return gap (as a fraction, e.g. 0.20 = 20 percentage points) between a
# split's two children that saturates tilt_strength at 1.0. Exposed as
# generate_momentum_views' tilt_scale parameter (this is just its default).
TILT_SCALE = 0.20

# Calibrated default trailing window (trading days) for both the momentum
# comparison and the confidence calculation -- see module docstring.
LOOKBACK = 40

# Below this tilt_strength, the momentum gap is treated as noise and no
# view is emitted for that node at all.
MIN_TILT_STRENGTH_TO_EMIT = 0.01

# Below this many valid (day, side) observations, a momentum gap's
# significance can't be assessed at all (a t-test needs at least 2 points
# to have a defined standard error) -- treated as totally insignificant.
MIN_OBSERVATIONS_FOR_SIGNIFICANCE = 2


def _cluster_trailing_return(returns_df: pd.DataFrame, assets, lookback: int) -> float:
    """Mean, across ``assets``, of each asset's compounded return over the
    most recent ``lookback`` rows of ``returns_df``.
    """
    window = returns_df[list(assets)].tail(lookback)
    compounded = (1.0 + window).prod() - 1.0
    return float(compounded.mean())


def _data_availability_fraction(returns_df: pd.DataFrame, assets, lookback: int) -> float:
    """Fraction of (day, asset) cells that are non-null in the trailing
    ``lookback``-row window for ``assets``.

    Used as one component of confidence: "how much real data actually
    backs this signal," independent of the signal's raw magnitude -- a
    large momentum gap computed from a mostly-missing window is less
    trustworthy than the same gap computed from complete data.
    """
    window = returns_df[list(assets)].tail(lookback)
    return float(window.notna().to_numpy().mean())


def _momentum_gap_significance(
    returns_df: pd.DataFrame, left_assets, right_assets, lookback: int
) -> float:
    """How statistically distinguishable-from-noise the left/right momentum
    gap is, as ``1 - two_tailed_p_value`` of a one-sample t-test that the
    mean daily cluster-return differential is zero, in ``[0, 1]``.

    Complements ``_data_availability_fraction``: data availability alone
    can't tell a real trend from a coin flip -- a 3-day window with full
    data and a 60-day window with full data get the same availability
    score, even though a return gap measured over just 3 days is far more
    likely to be noise. This takes each side's trailing per-day mean
    return across its assets (``.mean(axis=1)``, matching how
    ``_cluster_trailing_return`` averages across a cluster), differences
    them day by day, and asks a standard one-sample t-test whether that
    daily differential series' mean is distinguishable from zero --
    folding in both the number of usable days (``n``, via the t
    distribution's degrees of freedom) and how noisy the differential is
    day to day (its sample standard deviation), not just its raw size.

    Degenerate cases: fewer than ``MIN_OBSERVATIONS_FOR_SIGNIFICANCE``
    valid days (no defined standard error) returns 0.0 -- unassessable,
    treated as insignificant. A (numerically) zero-variance differential
    returns 1.0 if its mean is nonzero (a perfectly consistent gap on
    every valid day, the limit of ``t -> infinity``) or 0.0 if the mean is
    also zero (no gap, no noise -- nothing to detect).
    """
    left_daily = returns_df[list(left_assets)].tail(lookback).mean(axis=1)
    right_daily = returns_df[list(right_assets)].tail(lookback).mean(axis=1)
    diff = (left_daily - right_daily).dropna()

    n = len(diff)
    if n < MIN_OBSERVATIONS_FOR_SIGNIFICANCE:
        return 0.0

    mean_diff = float(diff.mean())
    std_diff = float(diff.std(ddof=1))

    if not np.isfinite(std_diff) or np.isclose(std_diff, 0.0):
        return 1.0 if mean_diff != 0.0 else 0.0

    t_stat = mean_diff / (std_diff / np.sqrt(n))
    p_value = 2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=n - 1))
    return float(np.clip(1.0 - p_value, 0.0, 1.0))


def generate_momentum_views(
    returns_df: pd.DataFrame,
    tree: ClusterTree,
    lookback: int = LOOKBACK,
    tilt_scale: float = TILT_SCALE,
) -> ViewSet:
    """Generate one View per tree split with a non-trivial momentum signal.

    For every internal (non-leaf) node in ``tree``, compares its two
    children's trailing ``lookback``-day cluster returns
    (``_cluster_trailing_return``: the mean of each side's per-asset
    compounded return over that window). Whichever side has the higher
    trailing return is "favored", and a View is emitted for it
    (``node_assets = assets_under(favored_child)``), always with a
    *positive* ``tilt_strength`` -- matching ``allocators.vihrp``'s
    blending convention, where a View's positive ``tilt_strength``
    increases weight on the side it's attached to (see
    ``allocators.vihrp._apply_tilt``).

    ``tilt_strength`` formula::

        diff = favored_return - unfavored_return   (>= 0 by construction)
        tilt_strength = min(diff / tilt_scale, 1.0)

    a linearly-scaled, capped magnitude of the return gap. ``tilt_scale``
    defaults to the module's ``TILT_SCALE``, calibrated (see module
    docstring) rather than a round-number guess; passing a larger value
    makes the tilt saturate at a bigger return gap (so, for a given gap,
    a *smaller* resulting tilt_strength) -- useful for sensitivity
    analysis (see ``scripts/run_vihrp_sensitivity_analysis.py``).

    ``confidence`` formula: the *worse* (``min``) of two independent
    factors, each in ``[0, 1]`` --

    1. ``_data_availability_fraction``, the worse of the two children's
       trailing non-null data fractions -- "is there enough real data to
       trust this comparison at all." This project's real data has
       exactly the failure mode this guards against: LTGILTBEES trades
       sparsely for years after its listing (scattered multi-week gaps,
       discovered while building the backtester), so a momentum
       comparison involving it during that period gets a correspondingly
       lower-confidence view even if the raw gap looks large.
    2. ``_momentum_gap_significance``, a one-sample t-test's
       ``1 - two_tailed_p_value`` on the daily left/right return
       differential -- "is the gap itself distinguishable from noise."
       Data availability alone can't tell a real multi-week trend from a
       huge gap that happens to be measured over only 2-3 days; this
       factor folds in both the sample size and the differential's own
       day-to-day noisiness, via a standard significance test, rather
       than trusting a raw return-gap magnitude at face value.

    Taking the ``min`` (not the product) of the two keeps confidence
    bounded by whichever factor is weaker, without compounding two
    separate penalties into an artificially small number -- the same
    "worst of the two sides" idiom ``_data_availability_fraction``
    already uses internally for left vs. right.

    Nodes whose resulting ``tilt_strength`` would round to below
    ``MIN_TILT_STRENGTH_TO_EMIT`` (0.01), or whose return difference isn't
    finite (e.g. a window with no usable data at all), are skipped
    entirely -- treated as noise or unusable, not worth a view.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Daily simple returns, dates as rows, tickers as columns.
    tree : ClusterTree
        Already-built tree over the same (or a compatible) universe.
    lookback : int, optional
        Trailing window, in trading days, used for the momentum
        comparison, the data-availability check, and the significance
        test. Defaults to the module's ``LOOKBACK`` (calibrated; see
        module docstring).
    tilt_scale : float, optional
        The return-gap magnitude (as a fraction, e.g. 0.20 = 20
        percentage points) that saturates tilt_strength at 1.0. Defaults
        to ``TILT_SCALE`` (calibrated; see module docstring).

    Returns
    -------
    ViewSet
    """
    views: list[View] = []

    def walk(node) -> None:
        if tree.is_leaf(node):
            return

        left, right = tree.children(node)
        left_assets = tree.assets_under(left)
        right_assets = tree.assets_under(right)

        ret_left = _cluster_trailing_return(returns_df, left_assets, lookback)
        ret_right = _cluster_trailing_return(returns_df, right_assets, lookback)
        diff = ret_left - ret_right

        if np.isfinite(diff) and diff != 0:
            if diff > 0:
                favored_assets, unfavored_assets = left_assets, right_assets
            else:
                favored_assets, unfavored_assets = right_assets, left_assets

            tilt_strength = min(abs(diff) / tilt_scale, 1.0)

            if tilt_strength >= MIN_TILT_STRENGTH_TO_EMIT:
                confidence = min(
                    _data_availability_fraction(returns_df, favored_assets, lookback),
                    _data_availability_fraction(returns_df, unfavored_assets, lookback),
                    _momentum_gap_significance(returns_df, left_assets, right_assets, lookback),
                )
                views.append(
                    View(
                        node_assets=frozenset(favored_assets),
                        tilt_strength=tilt_strength,
                        confidence=confidence,
                        source=f"momentum_{lookback}d",
                    )
                )

        walk(left)
        walk(right)

    walk(tree.root)
    return ViewSet(views)
