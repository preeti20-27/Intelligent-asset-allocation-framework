"""View objects for View-Informed HRP (VI-HRP): user-supplied directional
adjustments attached to specific nodes of a ``tree.cluster_tree.ClusterTree``.

A ``View`` doesn't reference a tree node directly (nodes are just integer
ids scoped to one specific tree instance, and aren't stable across
different builds/datasets) -- instead it names the exact set of asset
tickers the targeted node covers, and is matched against a tree at lookup
time via ``ClusterTree.assets_under(node)``.
"""

from __future__ import annotations

from dataclasses import dataclass

from tree.cluster_tree import ClusterTree


@dataclass
class View:
    """A single directional view attached to one specific ClusterTree node.

    Parameters
    ----------
    node_assets : frozenset[str]
        The exact set of asset tickers this view applies to. Matched
        against a tree node via ``ClusterTree.assets_under(node)`` -- a
        view targets a node only when ``node_assets`` equals that node's
        ``assets_under()`` exactly (as a set), not a subset or superset.
    tilt_strength : float
        Signed directional adjustment strength, in ``[-1, 1]``. Positive
        values tilt weight toward this node's "preferred" side of its
        parent split; negative values tilt away from it. The name and
        this description are deliberately generic -- semantics may extend
        in future versions.
    confidence : float
        In ``[0, 1]``. Scales how much ``tilt_strength`` actually shifts
        the split ratio: 0 means no effect (matches plain HRP exactly), 1
        means full effect.
    source : str
        Free-text description of where this view came from (e.g. an
        analyst note, a signal name, a user override).
    """

    node_assets: frozenset[str]
    tilt_strength: float
    confidence: float
    source: str

    def __post_init__(self) -> None:
        if not isinstance(self.node_assets, frozenset):
            self.node_assets = frozenset(self.node_assets)
        if not (-1.0 <= self.tilt_strength <= 1.0):
            raise ValueError(f"tilt_strength must be in [-1, 1], got {self.tilt_strength!r}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence!r}")


class ViewSet:
    """A collection of ``View`` objects, with node-matching lookup.

    Parameters
    ----------
    views : list[View], optional
        The views in this set. Defaults to an empty list (no views --
        every lookup returns ``None``, so an allocator using ``ViewSet()``
        behaves exactly as if no ``ViewSet`` were passed at all).
    """

    def __init__(self, views: "list[View] | None" = None):
        self.views = list(views) if views is not None else []

    def find_view_for_node(self, tree: ClusterTree, node) -> "View | None":
        """The View whose ``node_assets`` exactly matches ``tree.assets_under(node)``, or None.

        A view whose ``node_assets`` doesn't match any actual node in
        ``tree`` (e.g. because it names tickers that were never grouped
        into a single subtree together) simply never matches anything --
        it's silently inert, not an error.
        """
        node_assets = frozenset(tree.assets_under(node))
        for view in self.views:
            if view.node_assets == node_assets:
                return view
        return None
