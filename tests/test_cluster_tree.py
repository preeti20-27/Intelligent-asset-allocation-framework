"""Tests for allocator.cluster_tree.ClusterTree."""

import numpy as np
import pandas as pd
import pytest
from scipy.cluster.hierarchy import linkage as scipy_linkage
from scipy.spatial.distance import squareform

from allocator.cluster_tree import ClusterTree, correlation_distance


def _make_synthetic_returns(n_assets=8, n_days=500, seed=42) -> pd.DataFrame:
    """Same factor-model generator used in tests/test_hrp.py, for consistency."""
    rng = np.random.default_rng(seed)
    tickers = [f"ASSET_{i}" for i in range(n_assets)]

    n_factors = 3
    factor_loadings = rng.normal(0, 1, size=(n_assets, n_factors))
    factors = rng.normal(0, 0.01, size=(n_days, n_factors))
    idio = rng.normal(0, 0.005, size=(n_days, n_assets))
    returns = factors @ factor_loadings.T + idio

    return pd.DataFrame(returns, columns=tickers)


def _old_quasi_diagonalize_leaf_order(returns_df: pd.DataFrame) -> list:
    """Reimplements the exact pre-refactor logic that lived inline in hrp.py's
    allocate(): correlation -> distance -> scipy linkage -> quasi_diagonalize
    -> map integer leaf positions back to tickers. Used as an independent
    regression oracle for ClusterTree.leaf_order, so this test doesn't just
    compare ClusterTree against itself.
    """
    tickers = list(returns_df.columns)
    corr = returns_df.corr()
    dist = np.sqrt(0.5 * (1.0 - corr))
    dist_vals = np.array(dist.values, copy=True)
    np.fill_diagonal(dist_vals, 0.0)
    dist_vals = (dist_vals + dist_vals.T) / 2.0

    condensed = squareform(dist_vals, checks=False)
    link = scipy_linkage(condensed, method="single").astype(int)
    num_items = link[-1, 3]

    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        clusters = sort_ix[sort_ix >= num_items]
        i = clusters.index
        j = clusters.values - num_items
        sort_ix[i] = link[j, 0]
        right = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, right])
        sort_ix = sort_ix.sort_index()
        sort_ix.index = range(sort_ix.shape[0])

    return [tickers[i] for i in sort_ix.tolist()]


def test_build_from_returns_assets_under_root_contains_all_assets():
    returns_df = _make_synthetic_returns()
    tree = ClusterTree.build(returns_df)

    assert set(tree.assets_under(tree.root)) == set(returns_df.columns)
    assert len(tree.assets_under(tree.root)) == len(returns_df.columns)


def test_build_from_correlation_matrix_assets_under_root_contains_all_assets():
    returns_df = _make_synthetic_returns()
    corr = returns_df.corr()
    tree = ClusterTree.build(corr)

    assert set(tree.assets_under(tree.root)) == set(returns_df.columns)


def test_build_from_returns_and_from_corr_give_same_tree():
    returns_df = _make_synthetic_returns()
    corr = returns_df.corr()

    tree_from_returns = ClusterTree.build(returns_df)
    tree_from_corr = ClusterTree.build(corr)

    assert tree_from_returns.leaf_order == tree_from_corr.leaf_order
    np.testing.assert_array_equal(tree_from_returns.linkage_matrix, tree_from_corr.linkage_matrix)


def test_children_always_returns_exactly_two_items_for_internal_nodes():
    returns_df = _make_synthetic_returns()
    tree = ClusterTree.build(returns_df)

    def walk(node):
        if tree.is_leaf(node):
            return
        children = tree.children(node)
        assert len(children) == 2
        for child in children:
            walk(child)

    walk(tree.root)


def test_is_leaf_correctly_identifies_leaves_vs_internal_nodes():
    returns_df = _make_synthetic_returns()
    tree = ClusterTree.build(returns_df)

    # Every ticker, addressed by name, is a leaf.
    for ticker in returns_df.columns:
        assert tree.is_leaf(ticker)

    # Every raw leaf id (0..n_leaves-1) is a leaf.
    for leaf_id in range(tree.n_leaves):
        assert tree.is_leaf(leaf_id)

    # The root (with >1 asset) is internal, not a leaf.
    assert not tree.is_leaf(tree.root)

    # Walking down from the root, every non-leaf node found is correctly
    # reported as internal, and its children are correctly reported as
    # whatever is_leaf says they are.
    def check(node):
        if tree.is_leaf(node):
            return
        left, right = tree.children(node)
        for child in (left, right):
            if isinstance(child, str):
                assert tree.is_leaf(child)
            check(child)

    check(tree.root)


def test_leaf_order_matches_old_inline_hrp_logic():
    returns_df = _make_synthetic_returns()
    tree = ClusterTree.build(returns_df)

    expected_leaf_order = _old_quasi_diagonalize_leaf_order(returns_df)
    assert tree.leaf_order == expected_leaf_order


def test_linkage_matrix_matches_direct_scipy_call():
    returns_df = _make_synthetic_returns()
    corr = returns_df.corr()
    tree = ClusterTree.build(corr)

    dist = correlation_distance(corr)
    condensed = squareform(dist.values, checks=False)
    expected_linkage = scipy_linkage(condensed, method="single")

    np.testing.assert_array_equal(tree.linkage_matrix, expected_linkage)


def test_children_raises_on_leaf():
    returns_df = _make_synthetic_returns()
    tree = ClusterTree.build(returns_df)

    some_ticker = returns_df.columns[0]
    with pytest.raises(ValueError):
        tree.children(some_ticker)


def test_assets_under_leaf_returns_single_element_list():
    returns_df = _make_synthetic_returns()
    tree = ClusterTree.build(returns_df)

    some_ticker = returns_df.columns[0]
    assert tree.assets_under(some_ticker) == [some_ticker]


def test_assets_under_children_partition_parent_with_no_overlap():
    returns_df = _make_synthetic_returns()
    tree = ClusterTree.build(returns_df)

    def check_partition(node):
        if tree.is_leaf(node):
            return
        left, right = tree.children(node)
        left_assets = set(tree.assets_under(left))
        right_assets = set(tree.assets_under(right))

        assert left_assets.isdisjoint(right_assets)
        assert left_assets | right_assets == set(tree.assets_under(node))

        check_partition(left)
        check_partition(right)

    check_partition(tree.root)
