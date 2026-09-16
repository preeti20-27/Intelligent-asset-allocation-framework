"""Minimal momentum-based view generator for VI-HRP.

Deliberately a simple placeholder heuristic -- not a final signal model --
just enough to produce real, non-trivial ``View`` objects for
backtesting VI-HRP against actual market conditions. The exact formulas
below (``TILT_SCALE``, the confidence proxy) are round-number choices
documented for transparency, not calibrated against any data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tree.cluster_tree import ClusterTree
from views.view import View, ViewSet

# Default placeholder scale: a 10-percentage-point gap in trailing
# compounded return between a split's two children saturates the tilt at
# 1.0. A round number picked as a plausible order of magnitude for real
# 60-day equity/ETF returns, not fit to any data. Exposed as
# generate_momentum_views' tilt_scale parameter (this is just its default).
TILT_SCALE = 0.10

# Below this tilt_strength, the momentum gap is treated as noise and no
# view is emitted for that node at all.
MIN_TILT_STRENGTH_TO_EMIT = 0.01


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

    Used as confidence's proxy for "how much real data actually backs
    this signal," independent of the signal's raw magnitude -- a large
    momentum gap computed from a mostly-missing window is less
    trustworthy than the same gap computed from complete data.
    """
    window = returns_df[list(assets)].tail(lookback)
    return float(window.notna().to_numpy().mean())


def generate_momentum_views(
    returns_df: pd.DataFrame,
    tree: ClusterTree,
    lookback: int = 60,
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

    ``tilt_strength`` formula (intentionally simple, a placeholder
    heuristic -- not a calibrated model)::

        diff = favored_return - unfavored_return   (>= 0 by construction)
        tilt_strength = min(diff / tilt_scale, 1.0)

    a linearly-scaled, capped magnitude of the return gap. ``tilt_scale``
    defaults to the module's ``TILT_SCALE`` (0.10); passing a larger value
    makes the tilt saturate at a bigger return gap (so, for a given gap,
    a *smaller* resulting tilt_strength) -- useful for sensitivity
    analysis (see ``scripts/run_vihrp_sensitivity_analysis.py``).

    ``confidence`` formula (also a placeholder): the *worse* of the two
    children's data-availability fractions (``_data_availability_fraction``)
    over the trailing window -- i.e. confidence reflects how much real
    data backs the comparison, not how large the return gap looks. This
    project's real data has exactly the failure mode this guards against:
    LTGILTBEES trades sparsely for years after its listing (scattered
    multi-week gaps, discovered while building the backtester), so a
    momentum comparison involving it during that period gets a
    correspondingly lower-confidence view even if the raw gap looks large.

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
        Trailing window, in trading days, used for both the momentum
        comparison and the confidence calculation. Defaults to 60.
    tilt_scale : float, optional
        The return-gap magnitude (as a fraction, e.g. 0.10 = 10
        percentage points) that saturates tilt_strength at 1.0. Defaults
        to ``TILT_SCALE`` (0.10).

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
