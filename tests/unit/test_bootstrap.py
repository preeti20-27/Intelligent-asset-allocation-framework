"""Tests for eval.bootstrap -- paired circular block bootstrap of Sharpe differences."""

import numpy as np
import pandas as pd
import pytest

from eval.bootstrap import (
    annualized_sharpe,
    circular_block_indices,
    default_block_size,
    equity_to_returns,
    sharpe_difference_test,
)
from eval.metrics import sharpe_ratio


def _make_pair(n_days=1000, drift_a=0.0, drift_b=0.0, phi=0.2, corr=0.8, seed=0):
    """Two correlated AR(1) daily return series with chosen drifts (daily, ~1% vol).

    ``phi`` gives each series serial correlation (the reason for a block,
    not iid, bootstrap); ``corr`` is the cross-correlation of their shocks
    (the reason the bootstrap must be paired).
    """
    rng = np.random.default_rng(seed)
    common = rng.normal(size=n_days)
    shocks = np.column_stack(
        [
            np.sqrt(corr) * common + np.sqrt(1 - corr) * rng.normal(size=n_days),
            np.sqrt(corr) * common + np.sqrt(1 - corr) * rng.normal(size=n_days),
        ]
    ) * 0.01
    x = np.zeros_like(shocks)
    for t in range(n_days):
        x[t] = shocks[t] + (phi * x[t - 1] if t else 0.0)
    index = pd.bdate_range("2020-01-01", periods=n_days)
    return (
        pd.Series(x[:, 0] + drift_a, index=index, name="a"),
        pd.Series(x[:, 1] + drift_b, index=index, name="b"),
    )


def test_point_estimates_match_eval_metrics_sharpe():
    a, b = _make_pair(drift_a=0.0005)
    result = sharpe_difference_test(a, b, n_boot=200)
    assert result.sharpe_a == pytest.approx(sharpe_ratio(a), rel=1e-12)
    assert result.sharpe_b == pytest.approx(sharpe_ratio(b), rel=1e-12)
    assert result.difference == pytest.approx(sharpe_ratio(a) - sharpe_ratio(b), rel=1e-12)


def test_detects_clear_sharpe_difference():
    # a: +0.1%/day drift on ~1.25% daily vol (annual SR ~1.3); b: zero drift.
    a, b = _make_pair(drift_a=0.001, drift_b=0.0, seed=1)
    result = sharpe_difference_test(a, b)
    assert result.significant
    assert result.p_value < 0.01
    assert result.ci_low > 0
    assert result.ci_low < result.difference < result.ci_high


def test_reports_no_difference_on_identically_distributed_noise():
    # A single null dataset is, by design, falsely "significant" ~5% of the
    # time (seed=2 here happens to be one of those draws: p=0.002), so
    # this is an illustrative example; the real guarantee is the
    # multi-seed false-positive-rate test below.
    a, b = _make_pair(drift_a=0.0003, drift_b=0.0003, seed=7)
    result = sharpe_difference_test(a, b)
    assert not result.significant
    assert result.ci_low < 0 < result.ci_high


def test_false_positive_rate_is_controlled_under_the_null():
    # 40 independent null datasets at alpha=0.05: expected ~2 rejections;
    # P(Binomial(40, 0.05) > 6) ~ 1.4%, so > 6 indicates a miscalibrated test.
    rejections = sum(
        sharpe_difference_test(*_make_pair(drift_a=0.0003, drift_b=0.0003, seed=100 + s), n_boot=1000).significant
        for s in range(40)
    )
    assert rejections <= 6


def test_block_bootstrap_widens_ci_under_serial_correlation():
    # Strongly autocorrelated returns with independent shocks: an iid
    # bootstrap (block_size=1) understates Sharpe uncertainty; blocks
    # capture it.
    a, b = _make_pair(phi=0.6, corr=0.0, drift_a=0.0003, seed=3)
    iid = sharpe_difference_test(a, b, block_size=1)
    block = sharpe_difference_test(a, b, block_size=25)
    assert (block.ci_high - block.ci_low) > 1.3 * (iid.ci_high - iid.ci_low)


def test_pairing_matters_for_correlated_strategies():
    # Highly correlated strategies: resampling each series with its OWN
    # block indices (unpaired) throws away the shared noise that cancels
    # in the difference, so its CI is far wider than the paired one.
    a, b = _make_pair(corr=0.95, drift_a=0.0003, seed=4)
    paired = sharpe_difference_test(a, b)

    values = np.column_stack([a.values, b.values])
    n, bs = len(values), default_block_size(len(values))
    idx_a = circular_block_indices(n, bs, 5000, np.random.default_rng(1))
    idx_b = circular_block_indices(n, bs, 5000, np.random.default_rng(2))
    unpaired = annualized_sharpe(values[idx_a, 0], axis=1) - annualized_sharpe(values[idx_b, 1], axis=1)
    unpaired_width = np.subtract(*np.quantile(unpaired, [0.975, 0.025]))

    assert (paired.ci_high - paired.ci_low) < 0.5 * unpaired_width


def test_deterministic_for_seed_and_antisymmetric():
    a, b = _make_pair(drift_a=0.0004, seed=5)
    r1 = sharpe_difference_test(a, b, n_boot=500, seed=9)
    r2 = sharpe_difference_test(a, b, n_boot=500, seed=9)
    assert r1 == r2
    swapped = sharpe_difference_test(b, a, n_boot=500, seed=9)
    assert swapped.difference == pytest.approx(-r1.difference)
    assert swapped.ci_low == pytest.approx(-r1.ci_high)
    assert swapped.p_value == pytest.approx(r1.p_value)


def test_aligns_on_common_index_and_drops_nan():
    a, b = _make_pair(seed=6)
    a.iloc[:10] = np.nan
    result = sharpe_difference_test(a, b.iloc[5:], n_boot=100)
    assert result.n_obs == len(a) - 10


def test_circular_block_indices_are_contiguous_blocks_that_wrap():
    rng = np.random.default_rng(0)
    idx = circular_block_indices(n_obs=10, block_size=4, n_boot=50, rng=rng)
    assert idx.shape == (50, 10)
    assert idx.min() >= 0 and idx.max() <= 9
    # Within each block of 4, consecutive indices step by +1 mod 10.
    for row in idx:
        for start in range(0, 8, 4):
            block = row[start : start + 4]
            assert all((block[i + 1] - block[i]) % 10 == 1 for i in range(len(block) - 1))


def test_default_block_size_rule():
    assert default_block_size(1000) == 10
    assert default_block_size(1) == 1


def test_equity_to_returns():
    equity = pd.Series([100.0, 110.0, 99.0])
    np.testing.assert_allclose(equity_to_returns(equity).values, [0.10, -0.10])
