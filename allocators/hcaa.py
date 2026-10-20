"""Hierarchical Clustering-based Asset Allocation (HCAA).

Follows Raffinot (2017), "Hierarchical Clustering-Based Asset Allocation",
Journal of Portfolio Management:

1. Tree clustering -- the same ``tree.cluster_tree.ClusterTree`` that
   ``allocators.hrp``, ``allocators.vihrp`` and ``allocators.herc`` build.
2. Tree cut into ``n_clusters`` final clusters, using the exact same cut as
   HERC (``allocators.herc.cut_tree``, default
   ``allocators.herc.DEFAULT_N_CLUSTERS = 3``; see that module's docstring
   for why 3 for this universe).
3. Top-down down to that cut, every split gives exactly 50% of the parent's
   weight to each child. Within each final cluster, every asset gets an
   equal share.

The defining simplification versus HRP
--------------------------------------
HCAA's split ratio at every node is NAIVE: it depends only on the tree's
SHAPE (which assets sit under which branch), never on any risk estimate.
No variance, covariance or correlation magnitude enters the weights at
all -- the covariance matrix isn't even computed. By contrast
``allocators.hrp.allocate`` (list bisection) and ``allocators.vihrp.allocate``
(tree walk) set every split by inverse variance,
``alpha = 1 - var_L / (var_L + var_R)``, and ``allocators.herc`` does the
same above the cut and ERC within clusters. Correlations affect HCAA only
through which assets get grouped together.

The 50/50 rule was chosen over "proportional to the number of
constituents" deliberately: splitting proportionally to subtree size at
every node telescopes to exactly equal weight per asset, making the tree
irrelevant, i.e. ``allocators.equal_weight``. 50/50 is the variant that
actually uses the hierarchy: a branch that splits off alone (e.g.
LTGILTBEES as a singleton vs. the other 11 assets) gets half the parent's
weight however small it is. That is HCAA's known weakness -- its weights
can concentrate heavily in isolated, singleton clusters -- and it is kept
as-is here because that is the published method this baseline exists to
represent.
"""

from __future__ import annotations

import pandas as pd

from allocators.herc import DEFAULT_N_CLUSTERS, cut_tree
from allocators.hrp import _validate_min_non_null_fraction, compute_correlation
from tree.cluster_tree import ClusterTree


def _distribute_weights(tree: ClusterTree, node, mass: float, cluster_nodes: set, weights: dict) -> None:
    """Halve ``mass`` at every split down to the cut, then share it equally within a cluster."""
    if tree.is_leaf(node) or node in cluster_nodes:
        assets = tree.assets_under(node)
        for ticker in assets:
            weights[ticker] = weights.get(ticker, 0.0) + mass / len(assets)
        return

    left, right = tree.children(node)
    _distribute_weights(tree, left, mass * 0.5, cluster_nodes, weights)
    _distribute_weights(tree, right, mass * 0.5, cluster_nodes, weights)


def allocate(
    returns_df: pd.DataFrame,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    linkage_method: str = "single",
) -> pd.Series:
    """Compute HCAA portfolio weights.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Daily simple returns, dates as rows, tickers as columns.
    n_clusters : int, optional
        Number of final clusters (same cut as ``allocators.herc``). With
        ``n_clusters = n_assets`` the 50/50 halving runs all the way down
        to single assets.
    linkage_method : str, optional
        Linkage method for ``ClusterTree.build``; ``"single"`` by default.

    Returns
    -------
    pd.Series
        Long-only weights indexed by ticker, summing to 1.0.

    Raises
    ------
    ValueError
        If any asset has less than ``MIN_NON_NULL_FRACTION`` non-null
        returns (same contract as ``allocators.hrp.allocate``) -- the
        correlation matrix behind the tree has the same pairwise-complete
        failure mode even though no covariance is used.
    """
    tickers = list(returns_df.columns)

    _validate_min_non_null_fraction(returns_df)

    corr = compute_correlation(returns_df)
    tree = ClusterTree.build(corr, linkage_method=linkage_method)
    cluster_nodes = set(cut_tree(tree, n_clusters))

    weights: dict = {}
    _distribute_weights(tree, tree.root, 1.0, cluster_nodes, weights)

    weights_series = pd.Series(weights).reindex(tickers)
    weights_series /= weights_series.sum()
    return weights_series
