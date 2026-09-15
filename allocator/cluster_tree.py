"""Reusable hierarchical-clustering tree wrapper for HRP-family allocators.

Wraps the raw ``scipy.cluster.hierarchy.linkage`` output (plus the
quasi-diagonalization step de Prado's HRP relies on) behind a small,
node-based API, so callers don't need to decode scipy's linkage matrix
encoding (leaf ids 0..n-1, internal-cluster ids n..2n-2, each row
``[left_id, right_id, distance, cluster_size]``) by hand.

This module owns the clustering primitives (``correlation_distance``,
``build_linkage``, ``quasi_diagonalize``) that used to live inline in
``allocator/hrp.py``; ``hrp.py`` now imports them from here (re-exported
there for backward compatibility) rather than duplicating them, so the
plain-HRP allocator and any future tree-consumer (e.g. view-informed HRP)
are guaranteed to build the exact same tree from the same correlation
matrix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


def correlation_distance(corr_df: pd.DataFrame) -> pd.DataFrame:
    """Convert a correlation matrix into a correlation-distance matrix.

    Uses ``distance = sqrt(0.5 * (1 - corr))``, which is a proper metric
    (satisfies the triangle inequality) bounded in ``[0, 1]``.
    """
    dist = np.sqrt(0.5 * (1.0 - corr_df))
    # Numerical noise can leave the diagonal at ~1e-17 instead of exactly 0,
    # and can break exact symmetry; enforce both before use in squareform.
    dist_vals = np.array(dist.values, copy=True)
    np.fill_diagonal(dist_vals, 0.0)
    dist_vals = (dist_vals + dist_vals.T) / 2.0
    return pd.DataFrame(dist_vals, index=corr_df.index, columns=corr_df.columns)


def build_linkage(dist_df: pd.DataFrame, method: str = "single") -> np.ndarray:
    """Run hierarchical clustering on a distance matrix.

    Thin wrapper around ``scipy.cluster.hierarchy.linkage``: converts the
    square distance matrix to condensed form and clusters it. ``single``
    linkage is the standard choice in the HRP literature.
    """
    condensed = squareform(dist_df.values, checks=False)
    return linkage(condensed, method=method)


def quasi_diagonalize(link: np.ndarray) -> list[int]:
    """Return the leaf order (as integer positions) implied by a linkage tree.

    This recursively expands each cluster node in the linkage matrix into
    its constituent leaves, in the order the dendrogram would draw them,
    so that assets placed close together in the tree end up adjacent in
    the reordered correlation matrix (quasi-diagonalization).
    """
    link = link.astype(int)
    num_items = link[-1, 3]

    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)  # make space
        clusters = sort_ix[sort_ix >= num_items]
        i = clusters.index
        j = clusters.values - num_items
        sort_ix[i] = link[j, 0]  # left children replace the cluster
        right = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, right])
        sort_ix = sort_ix.sort_index()
        sort_ix.index = range(sort_ix.shape[0])

    return sort_ix.tolist()


def _looks_like_correlation_matrix(data: pd.DataFrame) -> bool:
    """Heuristic: square, same row/column labels in the same order, diag ~= 1.

    Used by ``ClusterTree.build`` to accept either a returns DataFrame or
    an already-computed correlation matrix without requiring the caller to
    say which. A returns DataFrame would have to coincidentally have as
    many rows as columns *and* have its date index exactly equal (in
    label and order) to its ticker columns *and* have every column's
    first value equal 1.0 to be mistaken for a correlation matrix -- not
    realistic in practice.
    """
    if data.shape[0] != data.shape[1]:
        return False
    if list(data.index) != list(data.columns):
        return False
    diag = np.diag(data.values)
    return bool(np.allclose(diag, 1.0, atol=1e-6))


class ClusterTree:
    """Wraps a scipy linkage tree with a node-based traversal API.

    Nodes are identified the same way scipy's linkage matrix identifies
    them: integers ``0..n_leaves-1`` are the original assets (in the order
    of the correlation matrix's columns as passed to ``build``), and
    integers ``n_leaves..2*n_leaves-2`` are internal clusters formed by
    each successive merge, with the largest id being the tree's root.

    As a convenience, ``.children()`` returns a leaf child as its ticker
    string directly rather than as a raw integer id, and ``.is_leaf()``
    accepts either representation -- so callers walking the tree rarely
    need to think about scipy's raw id scheme at all, while
    ``.linkage_matrix`` remains available for anyone who does.

    Construct via ``ClusterTree.build(...)``, not directly.
    """

    def __init__(self) -> None:
        self._linkage_matrix: np.ndarray | None = None
        self._tickers: list[str] = []  # original column order fed into build()
        self._leaf_order: list[str] = []  # quasi-diagonalized order

    @classmethod
    def build(
        cls,
        returns_dataframe_or_corr_matrix: pd.DataFrame,
        linkage_method: str = "single",
    ) -> "ClusterTree":
        """Build a ClusterTree from returns or from a precomputed correlation matrix.

        Parameters
        ----------
        returns_dataframe_or_corr_matrix : pd.DataFrame
            Either daily returns (dates as rows, tickers as columns) or an
            already-computed correlation matrix (square, symmetric,
            tickers as both index and columns). Which one was passed is
            detected via ``_looks_like_correlation_matrix``; if given raw
            returns, the full sample's correlation is used (no windowing
            or missing-data handling here -- callers needing that, like
            ``allocator.hrp.allocate``, should compute and pass in the
            correlation matrix themselves).
        linkage_method : str, optional
            Linkage method passed to ``scipy.cluster.hierarchy.linkage``.
            Defaults to ``"single"``, the standard choice for HRP.

        Returns
        -------
        ClusterTree
        """
        if _looks_like_correlation_matrix(returns_dataframe_or_corr_matrix):
            corr = returns_dataframe_or_corr_matrix
        else:
            corr = returns_dataframe_or_corr_matrix.corr()

        dist = correlation_distance(corr)
        link = build_linkage(dist, method=linkage_method)

        tickers = list(corr.columns)
        leaf_positions = quasi_diagonalize(link)

        tree = cls()
        tree._linkage_matrix = link
        tree._tickers = tickers
        tree._leaf_order = [tickers[i] for i in leaf_positions]
        return tree

    @property
    def linkage_matrix(self) -> np.ndarray:
        """The raw scipy linkage matrix this tree was built from."""
        return self._linkage_matrix

    @property
    def leaf_order(self) -> list[str]:
        """Tickers in quasi-diagonalized order (dendrogram left-to-right leaf order)."""
        return list(self._leaf_order)

    @property
    def n_leaves(self) -> int:
        return len(self._tickers)

    @property
    def root(self):
        """The top node of the tree (an internal node id, or the sole ticker if n_leaves == 1)."""
        if self.n_leaves <= 1:
            return self._tickers[0] if self._tickers else None
        return 2 * self.n_leaves - 2

    def is_leaf(self, node) -> bool:
        """True if ``node`` is a leaf -- either a ticker string or a raw leaf id < n_leaves."""
        if isinstance(node, str):
            return True
        return int(node) < self.n_leaves

    def children(self, node) -> tuple:
        """The two children of an internal node.

        Each child is returned as a ticker string if it's a leaf, or as an
        internal node id (int) otherwise -- so callers can recurse with
        ``is_leaf()`` without separately translating leaf ids to tickers.

        Raises
        ------
        ValueError
            If ``node`` is a leaf (leaves have no children).
        """
        if self.is_leaf(node):
            raise ValueError(f"leaf node {node!r} has no children")

        row = int(node) - self.n_leaves
        left_id, right_id = int(self._linkage_matrix[row, 0]), int(self._linkage_matrix[row, 1])
        return self._wrap(left_id), self._wrap(right_id)

    def _wrap(self, raw_child_id: int):
        """Translate a raw scipy child id into a ticker (leaf) or node id (internal)."""
        if raw_child_id < self.n_leaves:
            return self._tickers[raw_child_id]
        return raw_child_id

    def assets_under(self, node) -> list[str]:
        """All tickers in the subtree rooted at ``node``, as a flat list."""
        if self.is_leaf(node):
            return [node] if isinstance(node, str) else [self._tickers[int(node)]]

        left, right = self.children(node)
        return self.assets_under(left) + self.assets_under(right)
