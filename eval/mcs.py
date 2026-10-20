"""Model Confidence Set (Hansen, Lunde & Nason, 2011) on Sharpe ratios,
plus Holm's step-down correction for families of pairwise tests.

What is implemented
-------------------
The full HLN MCS procedure (their Section 3, the ``T_max`` statistic with
the matching ``e_max`` elimination rule), not a simplified substitute:

1. Start with every model in ``M``.
2. For each model ``i`` in ``M`` compute its performance relative to the
   average of ``M``: ``d_i = SR_i - mean_{j in M} SR_j``, and, from the
   bootstrap, ``var(d_i)``. Standardize, ``t_i = -d_i / sqrt(var(d_i))``
   (sign flipped so that large ``t_i`` means "worse than average", the
   HLN loss convention).
3. ``T_max = max_i t_i``. Its null distribution comes from the same
   bootstrap draws, centered: ``T*_max = max_i -(d*_i - d_i) / sqrt(var(d_i))``.
   ``p = mean(T*_max >= T_max)`` tests ``H0: all models in M are equally
   good``.
4. Eliminate the worst model (argmax ``t_i``) and record its MCS p-value
   as ``max(p, every earlier p)`` -- HLN's running-max definition, which
   makes the MCS p-values monotone in elimination order. Repeat until one
   model remains (MCS p-value 1.0).

The ``(1 - alpha)`` Model Confidence Set is every model whose MCS p-value
is ``>= alpha``: the set that contains the best model(s) with probability
``>= 1 - alpha``.

The one adaptation from HLN: performance statistic = Sharpe ratio
------------------------------------------------------------------
HLN state the MCS for the mean of a per-period loss series. A Sharpe
ratio is not a per-period average, but it is a smooth function of the
first two moments of returns, and HLN's bootstrap implementation needs
only (a) a statistic per model and (b) its joint bootstrap distribution
across models, both of which the paired circular block bootstrap of
``eval.bootstrap`` provides -- the same block indices are applied to
every model's return series, so cross-model dependence and serial
dependence are both preserved. Ranking by Sharpe (rather than, say, mean
return) is what every comparison in this project reports, so the MCS
answers the question actually being asked: "which strategies can't be
statistically distinguished from the best one by Sharpe?".

Holm correction
---------------
``holm_adjust`` implements Holm (1979)'s step-down adjustment, which
controls the family-wise error rate for any dependence structure among
the tests. Used by ``scripts/apply_significance_tests.py`` to adjust the
pairwise ``eval.bootstrap.sharpe_difference_test`` p-values, as a
cross-check on the MCS that also says WHICH pairs differ.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from eval.bootstrap import (
    DEFAULT_N_BOOT,
    DEFAULT_SEED,
    annualized_sharpe,
    circular_block_indices,
    default_block_size,
)

DEFAULT_MCS_ALPHA = 0.10


def holm_adjust(p_values) -> np.ndarray:
    """Holm step-down adjusted p-values (same order as the input).

    ``p_adj[(k)] = max_{j <= k} min(1, (m - j + 1) * p_(j))`` over the
    ascending-sorted p-values. NaNs are left as NaN and excluded from the
    family size ``m``.
    """
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p, np.nan)
    valid = np.where(~np.isnan(p))[0]
    m = len(valid)
    if m == 0:
        return adjusted

    order = valid[np.argsort(p[valid], kind="stable")]
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adjusted[idx] = running
    return adjusted


def model_confidence_set(
    returns: pd.DataFrame,
    alpha: float = DEFAULT_MCS_ALPHA,
    block_size: int | None = None,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Hansen-Lunde-Nason Model Confidence Set over models' Sharpe ratios.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily simple returns, one column per model (strategy). Rows with
        any NaN are dropped, so every model is evaluated on the same days.
    alpha : float, optional
        MCS level: the returned ``in_mcs`` flags the ``(1 - alpha)`` MCS.
        HLN use 0.10 and 0.25; default 0.10.
    block_size, n_boot, seed : optional
        Circular block bootstrap settings (see ``eval.bootstrap``).

    Returns
    -------
    pd.DataFrame
        Indexed by model, sorted best-surviving first, with columns
        ``sharpe``, ``mcs_p_value``, ``elimination_order`` (1 = first
        eliminated; the last survivor gets ``n_models``) and ``in_mcs``.
    """
    data = returns.dropna()
    models = list(data.columns)
    if len(models) < 2:
        raise ValueError("Need at least 2 models for a Model Confidence Set.")
    values = data.to_numpy(dtype=float)
    n_obs = values.shape[0]
    if block_size is None:
        block_size = default_block_size(n_obs)

    sharpe = annualized_sharpe(values, axis=0)  # (n_models,)
    rng = np.random.default_rng(seed)
    indices = circular_block_indices(n_obs, block_size, n_boot, rng)
    boot_sharpe = np.column_stack(
        [annualized_sharpe(values[indices, k], axis=1) for k in range(len(models))]
    )  # (n_boot, n_models)

    alive = list(range(len(models)))
    p_values = {}
    order = {}
    running_p = 0.0
    step = 0

    while len(alive) > 1:
        step += 1
        sr = sharpe[alive]
        boot = boot_sharpe[:, alive]
        rel = sr - sr.mean()  # d_i: positive = better than average
        boot_rel = boot - boot.mean(axis=1, keepdims=True)
        sd = np.sqrt(np.mean((boot_rel - rel) ** 2, axis=0))
        sd = np.where(sd > 0, sd, np.inf)  # a model identical to the average can't be "worst"

        t_stats = -rel / sd  # large = worse
        t_max = t_stats.max()
        boot_t_max = (-(boot_rel - rel) / sd).max(axis=1)
        p = float(np.mean(boot_t_max >= t_max))

        running_p = max(running_p, p)
        worst = alive[int(np.argmax(t_stats))]
        p_values[worst] = running_p
        order[worst] = step
        alive.remove(worst)

    last = alive[0]
    p_values[last] = 1.0
    order[last] = len(models)

    result = pd.DataFrame(
        {
            "sharpe": sharpe,
            "mcs_p_value": [p_values[k] for k in range(len(models))],
            "elimination_order": [order[k] for k in range(len(models))],
        },
        index=pd.Index(models, name="model"),
    )
    result["in_mcs"] = result["mcs_p_value"] >= alpha
    return result.sort_values("elimination_order", ascending=False)
