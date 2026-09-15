"""Plain Hierarchical Risk Parity (HRP) portfolio allocation.

Implements the Lopez de Prado (2016) HRP algorithm in four stages:

1. Tree clustering  -- hierarchical clustering on a correlation-distance
   matrix, wrapped by ``allocator.cluster_tree.ClusterTree`` (which itself
   calls ``scipy.cluster.hierarchy``, not implemented from scratch).
2. Quasi-diagonalization -- reorder assets so that similar assets sit
   next to each other, based on the clustering leaf order. This is
   ``ClusterTree.leaf_order``.
3. Recursive bisection -- walk the quasi-diagonalized asset order
   top-down, repeatedly splitting the *list* in half and allocating
   weight between the two halves using inverse-variance allocation,
   until every asset has a weight. See the note on ``recursive_bisection``
   for why this is list-bisection rather than a literal walk of
   ClusterTree's (generally unbalanced) merge tree.
4. ``allocate`` ties the above into a single entry point that returns a
   ``pd.Series`` of weights indexed by ticker, summing to 1.0.

No shrinkage, EWMA, or other covariance estimation tricks are used here
by design -- this is the "plain" HRP baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .cluster_tree import (
    ClusterTree,
    build_linkage,  # noqa: F401 -- re-exported: this module's former home
    correlation_distance,  # noqa: F401 -- re-exported: this module's former home
    quasi_diagonalize,  # noqa: F401 -- re-exported: this module's former home
)

# Minimum fraction of non-null observations required per asset, over the
# window actually used for a given allocate() call, before allocate() will
# proceed. See _validate_min_non_null_fraction() for why.
MIN_NON_NULL_FRACTION = 0.9


def compute_returns(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Compute simple (arithmetic) daily returns from a price DataFrame.

    Parameters
    ----------
    prices_df : pd.DataFrame
        Daily prices, dates as the index (rows) and tickers as columns.

    Returns
    -------
    pd.DataFrame
        Simple returns ``r_t = p_t / p_{t-1} - 1``, with the first row
        (which has no prior price to compare to) dropped.
    """
    returns_df = prices_df.pct_change()
    return returns_df.dropna(how="all")


def compute_correlation(returns_df: pd.DataFrame, window: int | None = None) -> pd.DataFrame:
    """Compute a sample correlation matrix of asset returns.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Daily returns, dates as rows, tickers as columns.
    window : int, optional
        If ``None`` (default), use the full sample. If given, use only
        the most recent ``window`` rows of ``returns_df`` (a simple
        rolling-window snapshot rather than a full rolling series, since
        HRP needs a single correlation matrix per allocation call).

    Returns
    -------
    pd.DataFrame
        Square correlation matrix indexed and columned by ticker.
    """
    data = _window_slice(returns_df, window)
    return data.corr()


def compute_covariance(returns_df: pd.DataFrame, window: int | None = None) -> pd.DataFrame:
    """Compute a sample covariance matrix of asset returns.

    Mirrors ``compute_correlation``'s windowing behavior so that the
    covariance and correlation matrices used by ``allocate`` are always
    computed over the same slice of data.
    """
    data = _window_slice(returns_df, window)
    return data.cov()


def _window_slice(returns_df: pd.DataFrame, window: int | None) -> pd.DataFrame:
    """Return the most recent ``window`` rows of ``returns_df``, or all of it if ``window`` is None."""
    return returns_df.tail(window) if window is not None else returns_df


def _validate_min_non_null_fraction(
    data: pd.DataFrame, min_fraction: float = MIN_NON_NULL_FRACTION
) -> None:
    """Raise ValueError if any column has too much missing data in ``data``.

    ``compute_correlation``/``compute_covariance`` rely on pandas' default
    pairwise-complete-observations behavior: for each pair of columns, rows
    where either is NaN are dropped just for that pair before the
    coefficient is computed. When different columns have different missing
    patterns, different pairs end up estimated from different, partially
    overlapping subsets of rows -- and a correlation matrix assembled that
    way is not guaranteed to be positive semi-definite, which can produce
    negative "distances" or otherwise nonsensical input to the clustering
    step. This check rejects that situation up front instead of letting it
    flow silently into ``build_linkage``.

    Requiring every column to individually clear ``min_fraction`` non-null
    also bounds the worst case: by inclusion-exclusion, any two columns
    that each clear a fraction ``f`` of non-null rows share at least
    ``2f - 1`` of their rows as jointly non-null. At the default
    ``min_fraction=0.9`` that guarantees at least 80% row overlap for
    every pair, which is enough for a well-conditioned correlation
    estimate.
    """
    n_rows = len(data)
    non_null_counts = data.notna().sum()
    non_null_fraction = non_null_counts / n_rows
    offending = non_null_fraction[non_null_fraction < min_fraction].sort_values()

    if offending.empty:
        return

    details = []
    for ticker in offending.index:
        missing = n_rows - int(non_null_counts[ticker])
        details.append(
            f"{ticker} ({non_null_fraction[ticker]:.1%} non-null, "
            f"{missing}/{n_rows} rows missing)"
        )

    raise ValueError(
        f"allocate() requires at least {min_fraction:.0%} non-null returns per "
        f"asset over the {n_rows}-row window; violated by: {'; '.join(details)}. "
        "Clean or exclude these assets before calling allocate()."
    )


def _inverse_variance_weights(cov_df: pd.DataFrame, cluster_items: list) -> pd.Series:
    """Inverse-variance weights for the assets in ``cluster_items``."""
    variances = np.diag(cov_df.loc[cluster_items, cluster_items].values)
    ivp = 1.0 / variances
    ivp /= ivp.sum()
    return pd.Series(ivp, index=cluster_items)


def _cluster_variance(cov_df: pd.DataFrame, cluster_items: list) -> float:
    """Variance of the inverse-variance portfolio formed from ``cluster_items``."""
    cov_slice = cov_df.loc[cluster_items, cluster_items]
    w = _inverse_variance_weights(cov_df, cluster_items).values.reshape(-1, 1)
    return float((w.T @ cov_slice.values @ w).item())


def recursive_bisection(cov_df: pd.DataFrame, sorted_tickers: list) -> pd.Series:
    """Allocate weights via top-down recursive bisection of the sorted asset list.

    Starting from the full, quasi-diagonalized ordering of assets, this
    repeatedly splits each cluster *list* exactly in half (floor/ceil for
    odd sizes), and allocates weight between the two halves in inverse
    proportion to each half's cluster variance (the lower-variance branch
    gets more weight). This continues until every cluster is a single
    asset. This is the classic de Prado HRP algorithm, and is exactly what
    PyPortfolioOpt's HRPOpt does too (confirmed by the exact match in
    scripts/validate_against_reference.py).

    Deliberately NOT a literal walk of the underlying merge tree: a real
    dendrogram is generally unbalanced (e.g. a root that splits 1 asset
    against 9, not 5 against 5), so recursing through
    ``ClusterTree.children()``/``.is_leaf()`` node-by-node -- respecting
    each actual subtree's size -- gives materially different splits, and
    therefore different weights, than bisecting the flat quasi-diagonalized
    *list*. Both are defensible HRP variants, but only list-bisection
    reproduces the validated, PyPortfolioOpt-matching behavior this
    function has always had, so that's what stays here. ClusterTree's
    node-traversal API is still available for anything that wants the
    literal-tree variant (e.g. attaching views to specific tree nodes).

    Parameters
    ----------
    cov_df : pd.DataFrame
        Full covariance matrix (used to compute per-cluster variances).
    sorted_tickers : list
        Tickers in quasi-diagonal order (``ClusterTree.leaf_order``).

    Returns
    -------
    pd.Series
        Weights indexed by ticker, summing to 1.0.
    """
    weights = pd.Series(1.0, index=sorted_tickers)
    clusters = [sorted_tickers]

    while clusters:
        # Bisect every current cluster of size > 1 into two halves.
        clusters = [
            cluster[start:end]
            for cluster in clusters
            for start, end in ((0, len(cluster) // 2), (len(cluster) // 2, len(cluster)))
            if len(cluster) > 1
        ]

        for i in range(0, len(clusters), 2):
            left = clusters[i]
            right = clusters[i + 1]

            var_left = _cluster_variance(cov_df, left)
            var_right = _cluster_variance(cov_df, right)

            alpha = 1.0 - var_left / (var_left + var_right)  # more weight to lower variance
            weights[left] *= alpha
            weights[right] *= 1.0 - alpha

    return weights


def allocate(
    returns_df: pd.DataFrame,
    window: int | None = None,
    linkage_method: str = "single",
) -> pd.Series:
    """Compute Hierarchical Risk Parity portfolio weights.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Daily simple returns, dates as rows, tickers as columns (e.g. the
        output of ``compute_returns``).
    window : int, optional
        Restrict the correlation/covariance estimation to the most
        recent ``window`` rows. Defaults to using the full sample.
    linkage_method : str, optional
        Linkage method passed to ``scipy.cluster.hierarchy.linkage``.
        Defaults to ``"single"``, the standard choice for HRP.

    Returns
    -------
    pd.Series
        Portfolio weights indexed by ticker, summing to 1.0. The
        result is invariant to the column order of ``returns_df``.

    Raises
    ------
    ValueError
        If any asset has less than ``MIN_NON_NULL_FRACTION`` (90%) non-null
        returns over the window actually used (the full sample, or the
        most recent ``window`` rows). The message names the offending
        asset(s), their non-null percentage, and how many rows are missing.

    Missing data (NaN) contract
    ----------------------------
    ``compute_correlation``/``compute_covariance`` compute pandas
    correlation/covariance, which default to pairwise-complete-observations:
    for each pair of columns, rows where either is NaN are dropped just for
    that pair. When different assets have different missing-data patterns,
    different pairs end up estimated from different, only partially
    overlapping subsets of rows, and the resulting correlation matrix is
    not guaranteed to be positive semi-definite -- it can produce
    nonsensical "distances" that reach clustering silently.

    To prevent that, ``allocate`` checks every asset's non-null fraction
    over the window up front and raises ``ValueError`` if any asset falls
    below ``MIN_NON_NULL_FRACTION`` (90%), naming the offending asset(s).
    Below that threshold, gaps remain silently tolerated via
    pairwise-complete correlation/covariance as before: no exception, no
    NaN in the returned weights, no asset dropped, just quietly fewer
    sample points behind the affected asset's estimates than other assets'.
    ``allocate`` never silently returns NaN weights and never silently
    drops an asset -- it either tolerates a gap below the threshold, or
    raises this explicit, named ``ValueError`` above it.

    See ``tests/test_hrp_nan_behavior.py`` for the experiments that
    established the pairwise-complete behavior, and
    ``_validate_min_non_null_fraction`` for why 90% was chosen.
    """
    tickers = list(returns_df.columns)

    _validate_min_non_null_fraction(_window_slice(returns_df, window))

    corr = compute_correlation(returns_df, window=window)
    cov = compute_covariance(returns_df, window=window)

    tree = ClusterTree.build(corr, linkage_method=linkage_method)
    sorted_tickers = tree.leaf_order

    weights = recursive_bisection(cov, sorted_tickers)
    weights = weights.reindex(tickers)
    weights /= weights.sum()

    return weights
