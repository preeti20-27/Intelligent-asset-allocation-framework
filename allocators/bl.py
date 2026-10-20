"""Black-Litterman (BL) portfolio allocation (He & Litterman, 1999).

1. Equilibrium (prior) returns by reverse optimization:
   ``pi = delta * Sigma @ w_mkt``.
2. Optional views ``P @ mu = Q + eps, eps ~ N(0, Omega)`` blended into the
   prior to get the posterior mean and covariance:

       M      = tau Sigma P' (P tau Sigma P' + Omega)^-1
       mu_BL  = pi + M (Q - P pi)
       Sig_BL = Sigma + tau Sigma - M P tau Sigma

3. Mean-variance optimization over the posterior: maximize
   ``w' mu_BL - (delta / 2) w' Sig_BL w`` subject to ``sum(w) = 1`` (and
   ``w >= 0`` by default, matching the backtest's long-only constraint).

Choice of market proxy: equal weight
------------------------------------
There is no true market-cap benchmark for this 12-asset universe (10 NSE
equities + GOLDBEES + LTGILTBEES), so ``w_mkt`` must be a proxy. Equal
weight (``1/n``) is used: it is the only simple choice that is a
genuine long-only "market" holding every asset. The alternative named in
the project plan, an inverse-covariance-weighted proxy
(``w ∝ Sigma^-1 1``), was rejected because (a) it routinely has negative
entries, which no market portfolio has, and (b) reverse-optimizing it gives
``pi = delta * Sigma Sigma^-1 1 / c``, i.e. IDENTICAL implied returns for
every asset -- a degenerate "prior" that says nothing about equilibrium.

Zero-view behavior (the analogue of zero-view VI-HRP ~ plain HRP)
-----------------------------------------------------------------
With no views, ``mu_BL = pi`` and ``Sig_BL = (1 + tau) Sigma``. The
budget-constrained mean-variance optimum is then exactly

    w* = w_mkt / (1 + tau) + (tau / (1 + tau)) * w_minvar,

where ``w_minvar = Sigma^-1 1 / (1' Sigma^-1 1)`` -- i.e. the prior
allocation, with a small (``tau / (1 + tau)`` ~ 4.8% at ``tau = 0.05``)
tilt toward minimum variance coming from the extra estimation-uncertainty
term ``tau Sigma``. (This is He & Litterman's well-known result that
unconstrained BL weights without views are ``w_mkt / (1 + tau)``; the
budget constraint puts the missing ``tau / (1 + tau)`` into the
minimum-variance portfolio.) ``delta`` cancels out entirely in this case.
So zero-view BL is, by construction, a near-equal-weight portfolio: as a
backtest baseline it should behave very much like
``allocators.equal_weight``. Tested in ``tests/unit/test_bl.py``.

Units: everything is in the units of ``returns_df`` (daily simple
returns), so ``Q`` must be expressed as DAILY expected returns.
``DEFAULT_RISK_AVERSION = 2.5`` is the conventional He-Litterman value;
it only matters when views are supplied.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from allocators.herc import ensure_positive_definite
from allocators.hrp import _validate_min_non_null_fraction, compute_covariance

DEFAULT_TAU = 0.05
DEFAULT_RISK_AVERSION = 2.5


class BlackLittermanViews(NamedTuple):
    """Views in Black-Litterman form: ``P @ mu = Q`` with uncertainty ``omega``.

    Attributes
    ----------
    P : pd.DataFrame
        ``k x n`` pick matrix; columns are tickers (missing tickers are
        treated as 0), one row per view. E.g. an absolute view on INFY is
        a row with 1.0 under INFY; a relative "INFY beats TCS" view is
        +1 under INFY and -1 under TCS.
    Q : array-like
        Length-``k`` expected (DAILY) returns of each view portfolio.
    omega : array-like, optional
        ``k x k`` view-error covariance. Defaults to He & Litterman's
        ``diag(P tau Sigma P')`` -- each view as uncertain as the prior
        on the same portfolio.
    """

    P: pd.DataFrame
    Q: object
    omega: object = None


def implied_equilibrium_returns(
    cov: np.ndarray, market_weights: np.ndarray, risk_aversion: float = DEFAULT_RISK_AVERSION
) -> np.ndarray:
    """Reverse-optimized prior returns ``pi = delta * Sigma @ w_mkt``."""
    return risk_aversion * cov @ market_weights


def posterior(
    cov: np.ndarray,
    prior_returns: np.ndarray,
    P: np.ndarray | None = None,
    Q: np.ndarray | None = None,
    omega: np.ndarray | None = None,
    tau: float = DEFAULT_TAU,
) -> tuple[np.ndarray, np.ndarray]:
    """Black-Litterman posterior mean and covariance.

    With no views (``P`` is None or has zero rows) returns
    ``(prior_returns, (1 + tau) * cov)``.
    """
    tau_cov = tau * cov
    if P is None or len(P) == 0:
        return prior_returns.copy(), cov + tau_cov

    P = np.atleast_2d(np.asarray(P, dtype=float))
    Q = np.asarray(Q, dtype=float).reshape(-1)
    if omega is None:
        omega = np.diag(np.diag(P @ tau_cov @ P.T))
    omega = np.atleast_2d(np.asarray(omega, dtype=float))

    if P.shape[0] != Q.shape[0] or omega.shape != (P.shape[0], P.shape[0]):
        raise ValueError(
            f"Inconsistent view shapes: P {P.shape}, Q {Q.shape}, omega {omega.shape}."
        )

    gain = tau_cov @ P.T @ np.linalg.inv(P @ tau_cov @ P.T + omega)
    mu_bl = prior_returns + gain @ (Q - P @ prior_returns)
    cov_bl = cov + tau_cov - gain @ P @ tau_cov
    return mu_bl, (cov_bl + cov_bl.T) / 2.0


def mean_variance_weights(
    expected_returns: np.ndarray,
    cov: np.ndarray,
    risk_aversion: float = DEFAULT_RISK_AVERSION,
    long_only: bool = True,
) -> np.ndarray:
    """Maximize ``w' mu - (delta/2) w' Sigma w`` subject to ``sum(w) = 1`` (and ``w >= 0``).

    The budget-only problem is solved in closed form. The long-only
    problem is solved with SLSQP, warm-started from the closed-form
    solution clipped to the simplex; if that closed-form solution is
    already non-negative it is optimal for the long-only problem too and
    is returned directly.
    """
    n = len(expected_returns)
    inv_cov_ones = np.linalg.solve(cov, np.ones(n))
    inv_cov_mu = np.linalg.solve(cov, expected_returns)
    lagrange = (inv_cov_mu.sum() - risk_aversion) / inv_cov_ones.sum()
    unconstrained = (inv_cov_mu - lagrange * inv_cov_ones) / risk_aversion

    if not long_only or (unconstrained >= 0).all():
        return unconstrained

    # Rescale the objective to unit average variance for SLSQP's tolerances
    # (daily covariances are ~1e-4); this doesn't change the argmax.
    scale = np.mean(np.diag(cov))

    def neg_utility(w):
        cw = cov @ w
        value = -(w @ expected_returns - 0.5 * risk_aversion * w @ cw) / scale
        grad = -(expected_returns - risk_aversion * cw) / scale
        return value, grad

    start = np.clip(unconstrained, 0.0, None)
    start = start / start.sum() if start.sum() > 0 else np.full(n, 1.0 / n)
    result = minimize(
        neg_utility,
        start,
        jac=True,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(n)}],
        options={"ftol": 1e-15, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"Long-only mean-variance optimization failed: {result.message}")

    weights = np.clip(result.x, 0.0, None)
    return weights / weights.sum()


def allocate(
    returns_df: pd.DataFrame,
    views: BlackLittermanViews | None = None,
    tau: float = DEFAULT_TAU,
    risk_aversion: float = DEFAULT_RISK_AVERSION,
    long_only: bool = True,
) -> pd.Series:
    """Compute Black-Litterman mean-variance portfolio weights.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Daily simple returns, dates as rows, tickers as columns.
    views : BlackLittermanViews, optional
        Defaults to no views: pure equilibrium-return mean-variance
        optimization, which reduces to (approximately) the equal-weight
        prior -- see the module docstring.
    tau : float, optional
        Prior-uncertainty scalar (He & Litterman use 0.025-0.05).
    risk_aversion : float, optional
        ``delta``, used both to reverse-optimize the prior and in the
        mean-variance step.
    long_only : bool, optional
        Enforce ``w >= 0`` (default True, matching the backtest).

    Returns
    -------
    pd.Series
        Weights indexed by ticker, summing to 1.0.

    Raises
    ------
    ValueError
        If any asset has less than ``MIN_NON_NULL_FRACTION`` non-null
        returns (same contract as ``allocators.hrp.allocate``), or if the
        view arrays have inconsistent shapes.
    """
    tickers = list(returns_df.columns)
    if not tickers:
        raise ValueError("Cannot allocate: returns_df has no columns (tickers).")

    _validate_min_non_null_fraction(returns_df)

    cov = ensure_positive_definite(compute_covariance(returns_df).loc[tickers, tickers].values)
    market_weights = np.full(len(tickers), 1.0 / len(tickers))
    prior_returns = implied_equilibrium_returns(cov, market_weights, risk_aversion)

    if views is None:
        P = Q = omega = None
    else:
        unknown = sorted(set(views.P.columns) - set(tickers))
        if unknown:
            raise ValueError(f"View pick matrix references tickers not in returns_df: {unknown}.")
        P = views.P.reindex(columns=tickers, fill_value=0.0).values
        Q, omega = views.Q, views.omega

    mu_bl, cov_bl = posterior(cov, prior_returns, P, Q, omega, tau=tau)
    weights = mean_variance_weights(mu_bl, cov_bl, risk_aversion, long_only=long_only)

    weights_series = pd.Series(weights, index=tickers)
    weights_series /= weights_series.sum()
    return weights_series
