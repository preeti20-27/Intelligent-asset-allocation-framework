"""Pluggable-view-generator interface for VI-HRP.

``allocators.vihrp.allocate`` only ever consumes a ``views.view.ViewSet``
-- it has no import dependency on ``views.momentum_generator`` or any
other specific signal, and never will (see ``allocators/vihrp.py``'s
module docstring). The piece that *does* need a stable shape, so a new
signal can be dropped in without the caller (a backtest script, a live
rebalancing job, ...) changing how it wires things together, is the
function that builds a ``ViewSet`` from market data in the first place.
``ViewGenerator`` is that shape.

``views.momentum_generator.generate_momentum_views`` is the first
implementation. It satisfies ``ViewGenerator`` as-is: its ``lookback`` and
``tilt_scale`` keyword arguments have defaults, so it's callable as a
plain ``(returns_df, tree) -> ViewSet`` the same as any other generator; a
caller wanting non-default hyperparameters binds them first (e.g.
``functools.partial(generate_momentum_views, lookback=40)``) and passes
the bound callable around instead of the module-level function.

A future generator (a carry signal, a value signal, an analyst-override
loader, an ensemble of several) plugs in the same way: implement
``(returns_df: pd.DataFrame, tree: ClusterTree) -> ViewSet``, bind any of
its own hyperparameters via ``functools.partial`` or a small wrapper
closure, and hand the result to anything that currently expects a
``ViewGenerator`` (or just call it directly to build a ``ViewSet`` and
pass that to ``allocators.vihrp.allocate``).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd

from tree.cluster_tree import ClusterTree
from views.view import ViewSet


@runtime_checkable
class ViewGenerator(Protocol):
    """Callable contract: trailing returns + a built tree in, a ``ViewSet`` out.

    ``returns_df`` : daily simple returns, dates as rows, tickers as
    columns -- the same panel the tree was built from (or a compatible
    one). ``tree`` : a ``tree.cluster_tree.ClusterTree`` already built over
    that universe. Any generator-specific parameters (lookback window,
    scaling constants, ...) are the implementation's own keyword
    arguments with defaults, not part of this contract -- bind them
    before passing the callable on, so every ``ViewGenerator`` can be
    invoked uniformly as ``generator(returns_df, tree)``.
    """

    def __call__(self, returns_df: pd.DataFrame, tree: ClusterTree) -> ViewSet: ...
