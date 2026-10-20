"""Hierarchical Equal Risk Contribution (HERC) portfolio allocation.

Follows Raffinot (2018), "Hierarchical Clustering-Based Asset Allocation"
/ "The Hierarchical Equal Risk Contribution Portfolio", in three stages:

1. Tree clustering -- the exact same ``tree.cluster_tree.ClusterTree``
   (same correlation, same correlation-distance, same ``single`` linkage
   default) that ``allocators.hrp`` and ``allocators.vihrp`` build, so all
   three allocators see an identical tree for the same input.
2. Tree cut -- the dendrogram is cut into ``n_clusters`` final clusters by
   undoing its last ``n_clusters - 1`` merges (see ``cut_tree``). Top-down,
   weight is split between the two children of each of those top merges
   by inverse variance -- the same ``alpha = 1 - var_L / (var_L + var_R)``
   formula, with the same ``allocators.hrp._cluster_variance`` definition
   of a branch's variance, that HRP/VI-HRP use at every split.
3. Within each final cluster, weight is split by TRUE Equal Risk
   Contribution: every asset in the cluster contributes equally to the
   variance of the cluster's own sub-portfolio, computed from the
   cluster's full covariance submatrix (``equal_risk_contribution_weights``).
   This is solved numerically, not approximated by inverse-variance.

Choice of ``n_clusters`` (fixed ``DEFAULT_N_CLUSTERS = 3``)
--------------------------------------------------------
Raffinot selects k with the gap statistic. This project uses a fixed k=3
instead, because the empirical structure of this universe already pins it
down, and a fixed k is deterministic and directly testable. Cutting the
real rolling 252-day trees (2018-03 onward, every 20 days, 94 windows) at
k=3 yields exactly {GOLDBEES}, {LTGILTBEES}, {the 10 equities} in 84 of 94
windows -- the three asset classes that
``scripts/output/universe_structure_diagnosis.md`` identified as the source
of essentially all of HRP's diversification edge. Cutting deeper (k=4, 5)
under single linkage does not find new groups; it only peels off single
equities from the equity block (the chaining behavior of single linkage),
e.g. SUNPHARMA or ITC alone, so the extra "clusters" are not meaningful.
A gap statistic on 12 assets would mostly be choosing between those same
options with a random reference distribution on top. ``n_clusters`` is
still a parameter for anyone who wants a different cut.

How this differs from the other tree allocators
-----------------------------------------------
* ``allocators.hrp.allocate`` bisects the flat quasi-diagonalized *list*
  in halves all the way down to single assets, inverse-variance at every
  split. It never uses the tree's real (unbalanced) merge structure and
  never treats a group of assets as a unit with its own internal
  risk-parity solution.
* ``allocators.vihrp.allocate`` (with no views) walks the real tree, like
  HERC, but keeps splitting by inverse variance all the way down to
  single assets. HERC stops walking at the k-cluster cut and switches to
  ERC inside each cluster. That within-cluster ERC step is the key
  difference: inverse-variance splitting only ever looks at a branch's
  variance as a whole (and ``_cluster_variance`` builds that from
  inverse-variance weights that ignore correlation), whereas ERC uses
  every pairwise covariance inside the cluster, so two highly correlated
  members of a cluster are jointly down-weighted relative to a
  less-correlated one with the same volatility.

  Consequence (tested): with ``n_clusters = n_assets`` every cluster is a
  single asset, ERC is trivially 1.0, and HERC reproduces zero-view
  VI-HRP exactly. With ``n_clusters = 1`` HERC is pure ERC on the whole
  universe.

The missing-data contract and covariance estimation are reused from
``allocators.hrp`` unchanged (``MIN_NON_NULL_FRACTION`` check, pandas
pairwise-complete covariance).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from allocators.hrp import (
    _cluster_variance,
    _validate_min_non_null_fraction,
    compute_correlation,
    compute_covariance,
)
from tree.cluster_tree import ClusterTree

# See the module docstring for why 3 (asset-class structure of this universe).
DEFAULT_N_CLUSTERS = 3

# Eigenvalue floor (relative to the largest eigenvalue) used to repair a
# covariance matrix that pairwise-complete estimation has left slightly
# non-positive-definite. ERC and mean-variance both need a strictly PD
# matrix to have a unique solution.
MIN_EIGENVALUE_RATIO = 1e-10

# Tolerance on the ERC solution: max |RC_i - 1/n| across the cluster,
# where RC_i are the fractional risk contributions (summing to 1).
ERC_TOLERANCE = 1e-6


def ensure_positive_definite(cov: np.ndarray) -> np.ndarray:
    """Symmetrize ``cov`` and floor its eigenvalues so it is strictly PD.

    A no-op (up to symmetrization) for any well-conditioned sample
    covariance; only changes anything when pairwise-complete estimation
    has produced a (near-)singular or indefinite matrix.
    """
    cov = (np.asarray(cov, dtype=float) + np.asarray(cov, dtype=float).T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(cov)
    floor = max(eigvals.max(), 0.0) * MIN_EIGENVALUE_RATIO
    if eigvals.min() > floor:
        return cov
    eigvals = np.maximum(eigvals, floor if floor > 0 else 1e-18)
    return (eigvecs * eigvals) @ eigvecs.T


def risk_contributions(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Fractional risk contributions ``w_i (Cov w)_i / (w' Cov w)``, summing to 1."""
    weights = np.asarray(weights, dtype=float)
    marginal = cov @ weights
    total = float(weights @ marginal)
    return weights * marginal / total


def equal_risk_contribution_weights(cov: np.ndarray) -> np.ndarray:
    """Long-only weights (summing to 1) with equal risk contribution per asset.

    Uses the convex formulation of Spinu (2013): minimize
    ``0.5 y' C y - (1/n) sum(log y_i)`` over ``y > 0``. Its first-order
    condition is ``y_i (C y)_i = 1/n`` for every i, i.e. equal risk
    contributions, and it is strictly convex for PD ``C``, so the solution
    is unique; ``w = y / sum(y)`` is then the ERC portfolio (risk
    contributions are invariant to rescaling ``y``). ``C`` is ``cov``
    rescaled to unit average variance purely for numerical conditioning --
    ERC weights are invariant to scaling the covariance.

    Raises
    ------
    RuntimeError
        If the optimizer's solution does not equalize risk contributions
        to within ``ERC_TOLERANCE`` -- never silently returns a non-ERC
        portfolio.
    """
    cov = ensure_positive_definite(cov)
    n = cov.shape[0]
    if n == 1:
        return np.array([1.0])

    scaled = cov / np.mean(np.diag(cov))
    budget = 1.0 / n

    def objective(y):
        cy = scaled @ y
        return 0.5 * float(y @ cy) - budget * float(np.sum(np.log(y))), cy - budget / y

    # Inverse-volatility start: exact ERC when assets are uncorrelated.
    y0 = 1.0 / np.sqrt(np.diag(scaled))
    y0 *= np.sqrt(1.0 / float(y0 @ scaled @ y0))
    result = minimize(
        objective,
        y0,
        jac=True,
        method="L-BFGS-B",
        bounds=[(1e-12, None)] * n,
        options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 10_000},
    )

    weights = result.x / result.x.sum()
    deviation = np.max(np.abs(risk_contributions(weights, cov) - budget))
    if deviation > ERC_TOLERANCE:
        raise RuntimeError(
            f"ERC optimization did not converge: max risk-contribution deviation "
            f"{deviation:.2e} > {ERC_TOLERANCE:.0e} ({result.message})."
        )
    return weights


def cut_tree(tree: ClusterTree, n_clusters: int) -> list:
    """Nodes (ticker strings or internal node ids) forming a ``n_clusters`` cut.

    Undoes the tree's last ``n_clusters - 1`` merges: scipy numbers
    internal nodes in merge order (``n_leaves .. 2*n_leaves - 2``), so the
    last ``n_clusters - 1`` merges are exactly the internal nodes with id
    ``>= 2*n_leaves - n_clusters``. Every child of one of those nodes that
    isn't itself one of them is a final cluster. Always yields exactly
    ``min(n_clusters, n_leaves)`` clusters (unlike a distance-threshold
    cut, which can return fewer or more under tied merge heights).
    """
    n_clusters = max(1, min(int(n_clusters), tree.n_leaves))
    threshold = 2 * tree.n_leaves - n_clusters

    clusters = []

    def visit(node):
        if tree.is_leaf(node) or int(node) < threshold:
            clusters.append(node)
            return
        for child in tree.children(node):
            visit(child)

    visit(tree.root)
    return clusters


def _distribute_weights(
    tree: ClusterTree,
    node,
    mass: float,
    cov_df: pd.DataFrame,
    cluster_nodes: set,
    weights: dict,
) -> None:
    """Inverse-variance split down to the cut, then ERC inside each cluster."""
    if tree.is_leaf(node) or node in cluster_nodes:
        assets = tree.assets_under(node)
        erc = equal_risk_contribution_weights(cov_df.loc[assets, assets].values)
        for ticker, w in zip(assets, erc):
            weights[ticker] = weights.get(ticker, 0.0) + mass * w
        return

    left, right = tree.children(node)
    var_left = _cluster_variance(cov_df, tree.assets_under(left))
    var_right = _cluster_variance(cov_df, tree.assets_under(right))
    left_ratio = 1.0 - var_left / (var_left + var_right)  # more weight to lower variance

    _distribute_weights(tree, left, mass * left_ratio, cov_df, cluster_nodes, weights)
    _distribute_weights(tree, right, mass * (1.0 - left_ratio), cov_df, cluster_nodes, weights)


def allocate(
    returns_df: pd.DataFrame,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    linkage_method: str = "single",
) -> pd.Series:
    """Compute Hierarchical Equal Risk Contribution portfolio weights.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Daily simple returns, dates as rows, tickers as columns.
    n_clusters : int, optional
        Number of final clusters to cut the dendrogram into (see the
        module docstring for why the default is 3). Clipped to
        ``[1, n_assets]``.
    linkage_method : str, optional
        Linkage method for ``ClusterTree.build``; ``"single"`` by default,
        matching ``allocators.hrp``/``allocators.vihrp``.

    Returns
    -------
    pd.Series
        Long-only weights indexed by ticker (in ``returns_df``'s column
        order), summing to 1.0.

    Raises
    ------
    ValueError
        If any asset has less than ``MIN_NON_NULL_FRACTION`` non-null
        returns (same contract as ``allocators.hrp.allocate``).
    """
    tickers = list(returns_df.columns)

    _validate_min_non_null_fraction(returns_df)

    corr = compute_correlation(returns_df)
    cov = compute_covariance(returns_df)

    tree = ClusterTree.build(corr, linkage_method=linkage_method)
    cluster_nodes = set(cut_tree(tree, n_clusters))

    weights: dict = {}
    _distribute_weights(tree, tree.root, 1.0, cov, cluster_nodes, weights)

    weights_series = pd.Series(weights).reindex(tickers)
    weights_series /= weights_series.sum()
    return weights_series
