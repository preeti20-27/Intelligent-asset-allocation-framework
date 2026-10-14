"""Mean-reversion view generator for VI-HRP -- an alternative ``views.generator.ViewGenerator``.

Motivated directly by ``scripts/diagnose_regime.py``'s diagnosis of why
``views.momentum_generator.generate_momentum_views`` underperformed plain
HRP out-of-sample (see that script's committed output,
``scripts/output/regime_diagnosis.csv``): the momentum generator is built
on the premise that a cluster's trailing return predicts its *own*
future return with the *same* sign ("recent winners keep winning"). That
premise doesn't hold on this universe. The pooled correlation between a
cluster's trailing 40-day compounded return and its *next* 40-day
compounded return (``signal_persistence_corr_40d``) was measured as
**negative** in every in-sample slice tested -- subA (2019-04 to
2021-02): -0.022; subB (2021-03 to 2022-12): -0.084; full in-sample:
-0.036 -- i.e. a mild but consistent reversal tendency, not momentum, is
what the tuning-window data actually shows. (It's ~0 out-of-sample,
+0.015, which is the same diagnosis's other finding: the OOS window
looks calm/diversification-friendly with little cross-sectional
dispersion to exploit either way -- see that script's docstring and
output for the full picture, including why plain HRP alone did so well
there.)

This generator is the mirror image of ``generate_momentum_views``: same
tree walk, same ``tilt_strength``/``confidence`` formulas (reusing
``views.momentum_generator``'s own private helpers directly, not
reimplementing them -- ``_momentum_gap_significance``'s significance
test is direction-agnostic, since it's a two-tailed test on the same
|gap|, so it applies unchanged here), but it favors the side with the
LOWER trailing return instead of the higher one -- betting the gap
narrows rather than widens.

Deliberately reuses ``views.momentum_generator``'s already-calibrated
``LOOKBACK``/``TILT_SCALE`` (40 days / 0.20) as this generator's own
defaults, rather than running a separate grid search for it. That's a
controlled-comparison choice, not an oversight: it isolates "does
favoring the weaker side instead of the stronger one help" from "which
lookback/scale is best," which was already searched once for the
momentum case (``scripts/calibrate_momentum_generator.py``) -- running a
second full grid search for this generator's own hyperparameters is a
natural next step this module does not take (see this module's
docstring in context: this is the first alternative
``views.generator.ViewGenerator``, not a second full calibration pass).

Backtest result (``scripts/backtest_alternative_view_generators.py``,
committed output ``scripts/output/alternative_generator_comparison.csv``):
this does NOT fix the underperformance problem. It beats the momentum
generator on Sharpe in 2 of 4 segments (subB and OOS) but is dramatically
worse in subA -- Sharpe -0.159 and a -45.3% max drawdown, roughly double
plain HRP's -21.5% there -- and worse overall across the full in-sample
window (Sharpe 0.018 vs. momentum's 0.670). Plain HRP beats BOTH
generators on Sharpe in every one of the four segments tested, with no
exception. The likely reason mean-reversion is actively dangerous in
subA specifically: subA contains the Feb-Mar 2020 COVID crash, a period
of strong, violent trend continuation that the single pooled
``signal_persistence_corr_40d`` number for the whole subA window
(-0.022, averaged over ~2 years) doesn't capture -- a mean-reversion
tilt in a real crash means repeatedly adding to the side that's falling
fastest, the "catching a falling knife" failure mode. A single
window-averaged persistence correlation is too coarse a diagnostic to
guarantee a reversal tilt is *safe*, even when it's directionally
motivated by that same average. Treat this generator the same way as
``views.momentum_generator``: a documented, structurally-tested
signal that this project's own evidence says NOT to prefer over plain
HRP on this universe -- not a demonstrated improvement, and in subA's
case, a demonstrated regression even relative to the momentum generator
it was meant to improve on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tree.cluster_tree import ClusterTree
from views.momentum_generator import (
    LOOKBACK,
    MIN_TILT_STRENGTH_TO_EMIT,
    TILT_SCALE,
    _cluster_trailing_return,
    _data_availability_fraction,
    _momentum_gap_significance,
)
from views.view import View, ViewSet


def generate_mean_reversion_views(
    returns_df: pd.DataFrame,
    tree: ClusterTree,
    lookback: int = LOOKBACK,
    tilt_scale: float = TILT_SCALE,
) -> ViewSet:
    """Generate one View per tree split favoring the LOWER-trailing-return side.

    For every internal (non-leaf) node in ``tree``, compares its two
    children's trailing ``lookback``-day cluster returns exactly as
    ``views.momentum_generator.generate_momentum_views`` does, but the
    side favored by the emitted View (always with a *positive*
    ``tilt_strength``, matching ``allocators.vihrp``'s blending
    convention) is whichever child has the LOWER trailing return -- the
    opposite convention from the momentum generator.

    ``tilt_strength`` and ``confidence`` use the exact same formulas as
    ``generate_momentum_views`` (see that function's docstring): a
    linearly-scaled, capped ``|return_gap| / tilt_scale``, and a
    confidence that's the worst of data-availability (both sides) and
    the gap's statistical significance (``_momentum_gap_significance``,
    unchanged -- a two-tailed test doesn't care which side is "favored").
    Only the choice of *which side* to favor differs.

    Nodes whose resulting ``tilt_strength`` would round to below
    ``MIN_TILT_STRENGTH_TO_EMIT``, or whose return difference isn't
    finite, are skipped entirely -- same as the momentum generator.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Daily simple returns, dates as rows, tickers as columns.
    tree : ClusterTree
        Already-built tree over the same (or a compatible) universe.
    lookback : int, optional
        Trailing window, in trading days. Defaults to
        ``views.momentum_generator.LOOKBACK`` (reused, not independently
        calibrated for this generator -- see module docstring).
    tilt_scale : float, optional
        The return-gap magnitude that saturates tilt_strength at 1.0.
        Defaults to ``views.momentum_generator.TILT_SCALE`` (reused, not
        independently calibrated for this generator -- see module
        docstring).

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
            # Mirror of generate_momentum_views: favor the LOWER-return side.
            if diff > 0:
                favored_assets, unfavored_assets = right_assets, left_assets
            else:
                favored_assets, unfavored_assets = left_assets, right_assets

            tilt_strength = min(abs(diff) / tilt_scale, 1.0)

            if tilt_strength >= MIN_TILT_STRENGTH_TO_EMIT:
                confidence = min(
                    _data_availability_fraction(returns_df, favored_assets, lookback),
                    _data_availability_fraction(returns_df, unfavored_assets, lookback),
                    _momentum_gap_significance(returns_df, left_assets, right_assets, lookback),
                )
                views.append(
                    View(
                        node_assets=frozenset(favored_assets),
                        tilt_strength=tilt_strength,
                        confidence=confidence,
                        source=f"mean_reversion_{lookback}d",
                    )
                )

        walk(left)
        walk(right)

    walk(tree.root)
    return ViewSet(views)
