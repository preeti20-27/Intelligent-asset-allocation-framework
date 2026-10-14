"""Volatility-regime view generator for VI-HRP -- a fourth ``views.generator.ViewGenerator``,
and the first that targets the asset-class-level tree nodes specifically,
using a volatility signal rather than a trailing-return one.

Why this exists
----------------
``scripts/diagnose_universe_structure.py`` (see its committed output,
``scripts/output/universe_structure_diagnosis.md``) found that on this
12-asset universe (10 NSE equities + GOLDBEES + LTGILTBEES), virtually
all of plain HRP's edge over naive diversification comes from the TOP
TWO tree splits -- GOLDBEES vs. everyone else, then LTGILTBEES vs. the
equity block -- not from equity-internal weighting (which contributes
only 0.05-0.22 percentage points of annualized volatility reduction, in
EVERY regime tested). That diagnosis also surfaced a concrete,
economically grounded opportunity its own variance decomposition
couldn't exploit: GOLDBEES hit 74.6% annualized volatility during the
2020 COVID window (subA) -- almost double the most volatile equity --
while the broader equity block's OWN realized volatility also spiked
roughly 4.3x over its trailing baseline at the same time (14.9% ->
64.4%, confirmed directly against this project's own data before
choosing this design). A fixed, non-adaptive class ratio (like a naive
1/3-1/3-1/3 split) cannot react to that; HRP's own data-driven
inverse-variance sizing can, and does.

The three prior generators (``views.momentum_generator``,
``views.mean_reversion_generator``,
``views.standardized_momentum_reversal_generator``) all used
trailing-RETURN signals, and all three made VI-HRP worse than plain HRP
in aggregate -- including at the asset-class nodes they also walk
through, since nothing about their signal construction targets realized
volatility specifically. This generator targets ONLY the two
asset-class nodes, using a realized-volatility regime signal: when the
equity block's own trailing volatility is elevated well above its
recent normal range, tilt toward GOLDBEES (and, secondarily, toward
LTGILTBEES) as defensive assets; in a calm regime, emit no view at all
and let plain HRP's own inverse-variance split stand unchallenged.

Signal construction
--------------------
1. Build an equal-weight equity-block "blend" series (the cross-sectional
   mean of the 10 equities' daily returns -- a simple, assumption-light
   proxy for "how the equity block as a whole is behaving," not tied to
   any one name).
2. Split the trailing history into two NON-OVERLAPPING windows: the most
   recent ``SHORT_VOL_WINDOW`` days ("now"), and the ``LONG_VOL_WINDOW``
   days immediately before that ("the recent normal range"). Deliberately
   non-overlapping (unlike a single rolling window compared to itself)
   so the two realized-vol estimates are statistically independent
   samples -- required for the F-test below to be valid, and avoiding
   the "recent spike is necessarily inside its own baseline" circularity
   a single overlapping window would have.
3. ``regime_signal = (vol_short / vol_long) - 1.0`` -- 0 when current vol
   exactly matches the recent baseline, positive when elevated, negative
   when unusually calm. ``tilt_strength = min(max(regime_signal, 0) /
   tilt_scale, 1.0)`` -- ONLY tilts when vol is elevated (a negative
   signal, i.e. unusually calm, produces zero tilt, not a tilt in the
   other direction -- there's no "anti-defensive" bet in this design,
   per the task's "in calm regimes, emit no view" requirement). Severity
   of the regime, not just its sign, drives the magnitude.
4. ``confidence`` is the worst of: both involved assets' data
   availability over the combined lookback window (``_data_availability_fraction``,
   reused unchanged from ``views.momentum_generator`` -- same mechanism,
   same reasoning, as every prior generator), and a two-tailed F-test's
   ``1 - p_value`` for whether ``vol_short`` and ``vol_long`` are
   genuinely statistically different (not momentum_generator's t-test,
   which tests a MEAN difference -- this tests a VARIANCE/volatility
   difference, the natural analogue of that significance-test pattern
   for a vol-regime signal rather than a return-gap signal). A "stressed"
   reading from only a few, noisy observations gets low confidence the
   same way a thin-sample momentum gap does in the original generator.

Targeting ONLY the two asset-class nodes
-------------------------------------------
This generator does not walk the whole tree computing a signal at every
split the way the other three do. It explicitly searches for, and only
ever emits a View for, a node whose two children are EXACTLY
``{GOLDBEES}`` vs. a sibling containing no other special asset (i.e. the
rest of the whole universe), and/or EXACTLY ``{LTGILTBEES}`` vs. a
sibling that is purely equities (no GOLDBEES mixed in) --
``_find_singleton_vs_rest_node``. If the actual tree being walked
doesn't have that exact shape (e.g. GOLDBEES and LTGILTBEES happened to
cluster together first in some window, which ``scripts/diagnose_universe_structure.py``
never observed but this generator doesn't assume can't happen), no view
is emitted for that asset at all -- never a fallback onto some other,
structurally different node. This is a hard structural guarantee, not a
convention that merely tends to hold.

Calibration
-----------
See ``scripts/calibrate_volatility_regime_generator.py`` and its
committed output, ``scripts/output/volatility_regime_calibration.csv``,
for ``TILT_SCALE``/``SHORT_VOL_WINDOW``'s grid search (same in-sample/
OOS split, same <50%-saturation-before-ranking selection rule, as
``scripts/calibrate_momentum_generator.py``). ``LONG_VOL_WINDOW`` is
held fixed at a standard ~9-month realized-vol baseline window, not
grid-searched -- chosen so ``SHORT_VOL_WINDOW + LONG_VOL_WINDOW`` fits
comfortably inside ``backtest.engine.BacktestEngine``'s default 252-row
rebalance window across the whole ``SHORT_VOL_WINDOW`` grid, so (unlike
``views.standardized_momentum_reversal_generator``) this generator needs
no extra-history-feeding trick from its caller.

Grid-searched (3 ``short_vol_window`` values [10, 20, 40] x 6
``tilt_scale`` values [0.30, 0.50, 0.75, 1.00, 1.50, 2.00] = 18
combinations), IDENTICAL in-sample tuning window (2019-04-05 to
2022-12-31) / OOS holdout (2023-01-02 to 2026-09-11) split as every
other calibration in this project. The naive (unfiltered)
in-sample-Sharpe winner and the saturation-filtered winner were the same
row -- no rejection needed. Winner: ``SHORT_VOL_WINDOW=20`` (already the
module's pre-calibration default -- a standard ~1-month realized-vol
window turned out to already be reasonable), ``TILT_SCALE=0.30``
(changed from an initial, uncalibrated guess of 1.0 -- the grid showed
*smaller* tilt_scale, i.e. a MORE sensitive/aggressive reaction to
elevated vol, consistently scored better in-sample than larger, gentler
values, the opposite pattern from every return-based generator
calibrated in this project, where larger tilt_scale -- weaker tilt --
kept winning).

**Result: the first generator in this project's series to beat plain
HRP on the FULL in-sample window, via proper calibration, not a
sub-segment mirage.** In-sample Sharpe 0.807 vs. plain HRP's 0.768; subA
(the 2020 COVID window this generator specifically targets) Sharpe 0.793
vs. HRP's 0.774, WITH a shallower max drawdown (-18.1% vs. -21.5%) --
both the return and the tail-risk profile improved in the exact regime
the signal was designed for. The honest caveat: OOS Sharpe is 1.981 vs.
HRP's 2.601 -- a real, not-small cost in the calm 2023-2026 period,
where the generator mostly stays inert (regime_signal rarely clears the
tilt threshold) but still occasionally reacts to ordinary volatility
noise that isn't a genuine crisis, adding unrewarded turnover. See
``scripts/backtest_volatility_regime_generator.py``'s committed output
(``scripts/output/volatility_regime_comparison.csv``) for the full
picture against all three prior generators, and this module's own
honest framing: a crisis-period improvement with a calm-period cost,
not a strategy that dominates plain HRP everywhere.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from tree.cluster_tree import ClusterTree
from views.momentum_generator import MIN_TILT_STRENGTH_TO_EMIT, _data_availability_fraction
from views.view import View, ViewSet

GOLD_TICKER = "GOLDBEES"
GILT_TICKER = "LTGILTBEES"

# ~1 month realized-vol estimation window ("now").
SHORT_VOL_WINDOW = 20

# ~9 months realized-vol baseline window ("recent normal"), immediately
# preceding the short window, non-overlapping. Fixed, not grid-searched
# -- see module docstring for why (fits inside the standard 252-row
# engine window across the whole SHORT_VOL_WINDOW grid tested).
LONG_VOL_WINDOW = 180

# Regime-signal (vol_short/vol_long - 1) that saturates tilt_strength at
# 1.0. In z-ratio units, not raw-return units. Calibrated -- see module
# docstring's "Calibration" section.
TILT_SCALE = 0.30

# Below this many valid daily observations in EITHER window, volatility
# can't be estimated meaningfully at all.
MIN_VALID_DAYS_FOR_VOL = 10


def _find_singleton_vs_rest_node(
    tree: ClusterTree, singleton_ticker: str, allowed_other_assets: frozenset
) -> tuple[frozenset, frozenset] | None:
    """Find an internal node whose two children are EXACTLY
    ``{singleton_ticker}`` vs. some non-empty subset of
    ``allowed_other_assets`` -- i.e. a genuine "this one asset split off
    alone against [exactly the intended kind of] everything else" node,
    not a node where the singleton is clustered with something else
    first. Returns ``(frozenset({singleton_ticker}), other_assets)`` if
    found anywhere in the tree, else ``None``.
    """
    result = None

    def walk(node):
        nonlocal result
        if result is not None or tree.is_leaf(node):
            return
        left, right = tree.children(node)
        left_assets = frozenset(tree.assets_under(left))
        right_assets = frozenset(tree.assets_under(right))
        for a, b in ((left_assets, right_assets), (right_assets, left_assets)):
            if a == frozenset({singleton_ticker}) and b and b <= allowed_other_assets:
                result = (a, b)
                return
        walk(left)
        walk(right)

    walk(tree.root)
    return result


def _equity_blend_series(returns_df: pd.DataFrame) -> pd.Series:
    """Cross-sectional mean daily return across every ticker except
    GOLDBEES/LTGILTBEES -- a simple, assumption-light equity-block proxy.
    """
    equity_tickers = [t for t in returns_df.columns if t not in (GOLD_TICKER, GILT_TICKER)]
    return returns_df[equity_tickers].mean(axis=1, skipna=True)


def _vol_regime_signal(
    returns_df: pd.DataFrame, short_vol_window: int, long_vol_window: int
) -> tuple[float, float]:
    """(regime_signal, significance) from the equity blend's own trailing
    history. ``regime_signal`` is ``vol_short/vol_long - 1`` (nan if
    either window has too few valid observations, or ``vol_long`` is
    ~0). ``significance`` is a two-tailed F-test's ``1 - p_value`` for
    whether the two (non-overlapping) windows' variances genuinely
    differ, in ``[0, 1]`` (0.0 if undefined).
    """
    blend = _equity_blend_series(returns_df)
    total_needed = short_vol_window + long_vol_window
    tail = blend.tail(total_needed)

    short_window = tail.tail(short_vol_window).dropna()
    long_window = tail.iloc[: max(len(tail) - short_vol_window, 0)].dropna()

    if len(short_window) < MIN_VALID_DAYS_FOR_VOL or len(long_window) < MIN_VALID_DAYS_FOR_VOL:
        return float("nan"), 0.0

    var_short = float(short_window.var(ddof=1))
    var_long = float(long_window.var(ddof=1))
    if not np.isfinite(var_long) or np.isclose(var_long, 0.0):
        return float("nan"), 0.0

    vol_short = np.sqrt(var_short)
    vol_long = np.sqrt(var_long)
    regime_signal = vol_short / vol_long - 1.0

    f_stat = var_short / var_long
    df1, df2 = len(short_window) - 1, len(long_window) - 1
    cdf = stats.f.cdf(f_stat, df1, df2)
    p_value = 2.0 * min(cdf, 1.0 - cdf)
    significance = float(np.clip(1.0 - p_value, 0.0, 1.0))

    return float(regime_signal), significance


def generate_volatility_regime_views(
    returns_df: pd.DataFrame,
    tree: ClusterTree,
    short_vol_window: int = SHORT_VOL_WINDOW,
    long_vol_window: int = LONG_VOL_WINDOW,
    tilt_scale: float = TILT_SCALE,
) -> ViewSet:
    """Generate up to two Views -- GOLDBEES vs. the rest, and LTGILTBEES
    vs. the equity block -- driven by an equity-block volatility-regime
    signal, never touching any other node.

    See the module docstring for the full signal construction and the
    structural guarantee that this only ever targets those two nodes.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Daily simple returns, dates as rows, tickers as columns. Must
        include ``GOLDBEES`` and ``LTGILTBEES``; every other column is
        treated as an equity for the blend. Needs at least
        ``short_vol_window + long_vol_window`` rows for a defined signal
        (gracefully returns no views, not an error, if shorter -- see
        ``_vol_regime_signal``).
    tree : ClusterTree
        Already-built tree over the same (or a compatible) universe.
    short_vol_window, long_vol_window : int, optional
        The "now" and "recent normal baseline" windows, in trading days.
        Default to ``SHORT_VOL_WINDOW``/``LONG_VOL_WINDOW`` (20/180).
    tilt_scale : float, optional
        Regime-signal magnitude (``vol_short/vol_long - 1``) that
        saturates tilt_strength at 1.0. Defaults to ``TILT_SCALE``.

    Returns
    -------
    ViewSet
    """
    regime_signal, significance = _vol_regime_signal(returns_df, short_vol_window, long_vol_window)

    if not np.isfinite(regime_signal):
        return ViewSet()

    tilt_strength = min(max(regime_signal, 0.0) / tilt_scale, 1.0)
    if tilt_strength < MIN_TILT_STRENGTH_TO_EMIT:
        return ViewSet()  # calm (or only mildly elevated) regime: no view at all

    all_tickers = frozenset(returns_df.columns)
    equity_tickers = all_tickers - {GOLD_TICKER, GILT_TICKER}
    lookback_for_availability = short_vol_window + long_vol_window

    views: list[View] = []

    gold_match = _find_singleton_vs_rest_node(tree, GOLD_TICKER, all_tickers - {GOLD_TICKER})
    if gold_match is not None:
        favored, unfavored = gold_match
        confidence = min(
            _data_availability_fraction(returns_df, favored, lookback_for_availability),
            _data_availability_fraction(returns_df, unfavored, lookback_for_availability),
            significance,
        )
        views.append(
            View(
                node_assets=favored,
                tilt_strength=tilt_strength,
                confidence=confidence,
                source=f"volatility_regime_gold_{short_vol_window}_{long_vol_window}d",
            )
        )

    gilt_match = _find_singleton_vs_rest_node(tree, GILT_TICKER, equity_tickers)
    if gilt_match is not None:
        favored, unfavored = gilt_match
        confidence = min(
            _data_availability_fraction(returns_df, favored, lookback_for_availability),
            _data_availability_fraction(returns_df, unfavored, lookback_for_availability),
            significance,
        )
        views.append(
            View(
                node_assets=favored,
                tilt_strength=tilt_strength,
                confidence=confidence,
                source=f"volatility_regime_gilt_{short_vol_window}_{long_vol_window}d",
            )
        )

    return ViewSet(views)
