"""Tests for eval.mcs -- Model Confidence Set and Holm correction."""

import numpy as np
import pandas as pd
import pytest

from eval.mcs import holm_adjust, model_confidence_set


def _make_models(drifts, n_days=1000, corr=0.7, phi=0.1, seed=0) -> pd.DataFrame:
    """Correlated AR(1) daily return series (~1% vol), one per entry of ``drifts``."""
    rng = np.random.default_rng(seed)
    k = len(drifts)
    common = rng.normal(size=(n_days, 1))
    shocks = (np.sqrt(corr) * common + np.sqrt(1 - corr) * rng.normal(size=(n_days, k))) * 0.01
    x = np.zeros_like(shocks)
    for t in range(n_days):
        x[t] = shocks[t] + (phi * x[t - 1] if t else 0.0)
    return pd.DataFrame(
        x + np.asarray(drifts), index=pd.bdate_range("2020-01-01", periods=n_days),
        columns=[f"M{i}" for i in range(k)],
    )


def test_holm_adjust_known_values():
    # Sorted: 0.005*4=0.02, 0.01*3=0.03, 0.03*2=0.06, max(0.06, 0.04*1)=0.06
    np.testing.assert_allclose(holm_adjust([0.01, 0.04, 0.03, 0.005]), [0.03, 0.06, 0.06, 0.02])


def test_holm_adjust_caps_at_one_and_skips_nan():
    adjusted = holm_adjust([0.5, np.nan, 0.4])
    assert np.isnan(adjusted[1])
    np.testing.assert_allclose(adjusted[[0, 2]], [0.8, 0.8])
    np.testing.assert_allclose(holm_adjust([0.9, 0.8]), [1.0, 1.0])


def test_mcs_keeps_equally_good_models_and_drops_clearly_worse_ones():
    # M0-M2: equal, good drift; M3-M4: strongly negative drift.
    returns = _make_models([0.001, 0.001, 0.001, -0.001, -0.001], seed=1)
    result = model_confidence_set(returns, alpha=0.10)
    assert set(result.index[result["in_mcs"]]) == {"M0", "M1", "M2"}
    assert result.loc[["M3", "M4"], "mcs_p_value"].max() < 0.01


def test_mcs_keeps_everything_when_no_model_is_better():
    returns = _make_models([0.0004] * 5, seed=2)
    result = model_confidence_set(returns, alpha=0.10)
    assert result["in_mcs"].all()


def test_mcs_p_values_are_monotone_in_elimination_order_and_last_is_one():
    returns = _make_models([0.001, 0.0005, 0.0, -0.0005, -0.001], seed=3)
    result = model_confidence_set(returns).sort_values("elimination_order")
    p = result["mcs_p_value"].to_numpy()
    assert np.all(np.diff(p) >= 0)
    assert p[-1] == 1.0
    assert sorted(result["elimination_order"]) == list(range(1, 6))


def test_mcs_first_eliminated_has_lowest_sharpe_with_clear_ranking():
    returns = _make_models([0.0015, 0.0005, -0.0015], seed=4)
    result = model_confidence_set(returns)
    first_out = result["elimination_order"].idxmin()
    assert first_out == result["sharpe"].idxmin() == "M2"


def test_mcs_is_deterministic_for_seed():
    returns = _make_models([0.0005, 0.0, -0.0005], seed=5)
    pd.testing.assert_frame_equal(
        model_confidence_set(returns, n_boot=500, seed=1), model_confidence_set(returns, n_boot=500, seed=1)
    )


def test_mcs_requires_two_models():
    with pytest.raises(ValueError):
        model_confidence_set(_make_models([0.0]))
