"""View-Informed HRP (VI-HRP): plain HRP's inverse-variance split, tilted
by user-supplied ``views.view.View`` objects attached to specific
ClusterTree nodes.

This deliberately walks the ACTUAL clustering tree via
``tree.cluster_tree.ClusterTree``'s node API (``.root``, ``.children()``,
``.is_leaf()``, ``.assets_under()``), unlike ``allocators.hrp.allocate``,
which bisects the flat quasi-diagonalized *list* instead (see
``allocators.hrp.recursive_bisection``'s docstring for why). That
difference is intentional here, not an oversight: a View names the exact
set of assets a tree node covers, and that only corresponds to a real
split the algorithm makes if the algorithm actually walks the tree's real
(generally unbalanced) merge structure. Under flat-list bisection, "the
node containing assets X, Y, Z" usually isn't a boundary the algorithm
ever draws, so there'd be nothing consistent for a View to attach to.

One direct consequence: VI-HRP with zero views is NOT guaranteed to
produce identical weights to ``allocators.hrp.allocate`` on the same
input, even though both are legitimate HRP variants built from the same
correlation/covariance matrices and the same clustering tree -- they
differ in which split-generating procedure they use (tree-walk vs.
list-bisection), not just in whether a view happens to be attached at a
particular point. Everything else -- correlation/covariance estimation,
the missing-data contract, tree construction itself -- is reused directly
from ``allocators.hrp``/``tree.cluster_tree``, not duplicated.
"""

from __future__ import annotations

import pandas as pd

from allocators.hrp import (
    _cluster_variance,
    _validate_min_non_null_fraction,
    compute_correlation,
    compute_covariance,
)
from tree.cluster_tree import ClusterTree
from views.view import ViewSet

# A tilt is never allowed to push a split's ratio all the way to a bound:
# no branch of a split should end up with literally zero (or all) of the
# weight just because of a view, however extreme its tilt_strength/
# confidence.
MIN_SPLIT_RATIO = 0.01
MAX_SPLIT_RATIO = 0.99


def _apply_tilt(preferred_side_ratio: float, tilt_strength: float, confidence: float) -> float:
    """Blend ``preferred_side_ratio`` toward a tilt_strength-scaled target.

    ``preferred_side_ratio`` is the plain (untilted) fraction of weight
    already going to the side the view prefers. The fully-tilted target
    moves a ``tilt_strength``-fraction of the remaining distance to
    whichever bound ``tilt_strength``'s sign points toward (+1 -> 1.0,
    -1 -> 0.0, 0 -> unchanged), clamped to
    ``[MIN_SPLIT_RATIO, MAX_SPLIT_RATIO]`` before blending by
    ``confidence`` -- so at ``confidence=0`` this returns
    ``preferred_side_ratio`` unchanged (matching plain HRP exactly), and
    at ``confidence=1`` it returns the (clamped) fully-tilted target.
    """
    if tilt_strength >= 0:
        tilted = preferred_side_ratio + tilt_strength * (1.0 - preferred_side_ratio)
    else:
        tilted = preferred_side_ratio + tilt_strength * preferred_side_ratio

    tilted = min(max(tilted, MIN_SPLIT_RATIO), MAX_SPLIT_RATIO)
    return preferred_side_ratio * (1.0 - confidence) + tilted * confidence


def _split_ratio(cov_df: pd.DataFrame, left_assets: list, right_assets: list, views: ViewSet, tree, left, right) -> float:
    """The (possibly view-tilted) fraction of a split's weight going to ``left``."""
    var_left = _cluster_variance(cov_df, left_assets)
    var_right = _cluster_variance(cov_df, right_assets)
    left_ratio = 1.0 - var_left / (var_left + var_right)  # plain inverse-variance split

    view_left = views.find_view_for_node(tree, left)
    view_right = views.find_view_for_node(tree, right)

    if view_left is not None:
        left_ratio = _apply_tilt(left_ratio, view_left.tilt_strength, view_left.confidence)

    if view_right is not None:
        # Apply on top of any left-view adjustment already made, in terms
        # of the right side's own preferred ratio, then convert back.
        right_ratio = 1.0 - left_ratio
        right_ratio = _apply_tilt(right_ratio, view_right.tilt_strength, view_right.confidence)
        left_ratio = 1.0 - right_ratio

    return left_ratio


def _distribute_weights(
    tree: ClusterTree,
    node,
    mass: float,
    cov_df: pd.DataFrame,
    views: ViewSet,
    weights: dict,
) -> None:
    """Recursively split ``mass`` down the real tree from ``node``, filling in ``weights``."""
    if tree.is_leaf(node):
        ticker = node if isinstance(node, str) else tree.assets_under(node)[0]
        weights[ticker] = weights.get(ticker, 0.0) + mass
        return

    left, right = tree.children(node)
    left_assets = tree.assets_under(left)
    right_assets = tree.assets_under(right)

    left_ratio = _split_ratio(cov_df, left_assets, right_assets, views, tree, left, right)

    _distribute_weights(tree, left, mass * left_ratio, cov_df, views, weights)
    _distribute_weights(tree, right, mass * (1.0 - left_ratio), cov_df, views, weights)


def allocate(returns_dataframe: pd.DataFrame, views: "ViewSet | None" = None) -> pd.Series:
    """Compute View-Informed Hierarchical Risk Parity portfolio weights.

    Parameters
    ----------
    returns_dataframe : pd.DataFrame
        Daily simple returns, dates as rows, tickers as columns.
    views : ViewSet, optional
        Directional views to tilt specific tree-node splits. Defaults to
        no views (``None``, treated identically to an empty ``ViewSet``),
        in which case every split uses the plain inverse-variance ratio
        unchanged (see the module docstring for why that's still not
        guaranteed to match ``allocators.hrp.allocate`` bit-for-bit).

    Returns
    -------
    pd.Series
        Portfolio weights indexed by ticker, summing to 1.0.
    """
    if views is None:
        views = ViewSet()

    tickers = list(returns_dataframe.columns)

    _validate_min_non_null_fraction(returns_dataframe)

    corr = compute_correlation(returns_dataframe)
    cov = compute_covariance(returns_dataframe)

    tree = ClusterTree.build(corr)

    weights: dict = {}
    _distribute_weights(tree, tree.root, 1.0, cov, views, weights)

    weights_series = pd.Series(weights).reindex(tickers)
    weights_series /= weights_series.sum()
    return weights_series
