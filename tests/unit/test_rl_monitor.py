"""Unit tests for the Phase 2 RL Risk-Monitoring Overlay subsystem.

Covers:
- RiskOverlay constraint preservation (long-only, 25% equity cap, 5% ETF floor, sum=1.0)
- RiskOverlay action mechanics (HOLD, DERISK_MILD, DERISK_STRONG)
- State-to-observation transformation & feature extraction
- RiskMonitorEnv reset, step, date advancement, and reward formulation
- RLRiskAgent action selection, Q-value explainability, training loop, and save/load
"""

from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest
import torch

from monitor.env import (
    OBSERVATION_DIM,
    RiskMonitorEnv,
    state_to_observation,
)
from monitor.overlay import (
    ACTION_DERISK_MILD,
    ACTION_DERISK_STRONG,
    ACTION_HOLD,
    RiskOverlay,
    project_constraints,
)
from monitor.rl_agent import RLRiskAgent
from monitor.state import PortfolioState


# ---------------------------------------------------------------------------
# Fixtures & Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_portfolio_state():
    return PortfolioState(
        date="2024-03-15",
        weights={
            "RELIANCE": 0.20,
            "TCS": 0.20,
            "INFY": 0.20,
            "HDFCBANK": 0.20,
            "GOLDBEES": 0.10,
            "LTGILTBEES": 0.10,
        },
        weekly_return=0.015,
        monthly_return=0.032,
        drawdown_from_peak=-0.08,
        volatility=0.18,
        max_weight=0.20,
        portfolio_value=10_500_000.0,
    )


@pytest.fixture
def synthetic_market_data():
    """Create a synthetic 40-day panel for 4 equities and 2 ETFs."""
    dates = pd.bdate_range("2024-01-01", periods=40)
    tickers = ["EQUITY_A", "EQUITY_B", "EQUITY_C", "EQUITY_D", "GOLDBEES", "LTGILTBEES"]

    np.random.seed(42)
    returns_data = np.random.normal(loc=0.0005, scale=0.01, size=(len(dates), len(tickers)))
    # Add an engineered drawdown in the middle
    returns_data[15:25, :4] -= 0.02

    returns_df = pd.DataFrame(returns_data, index=dates, columns=tickers)

    # Base weights: 20% in each of 4 equities, 10% in each of 2 ETFs
    base_w_series = pd.Series(
        [0.20, 0.20, 0.20, 0.20, 0.10, 0.10], index=tickers
    )
    base_weights_df = pd.DataFrame(
        [base_w_series] * len(dates), index=dates
    )

    return returns_df, base_weights_df


# ---------------------------------------------------------------------------
# 1. Overlay Tests
# ---------------------------------------------------------------------------

def test_project_constraints_strict():
    """Verify project_constraints enforces caps, floors, long-only, and sum=1.0."""
    raw = pd.Series(
        {
            "EQUITY_A": 0.50,  # Violates 0.25 cap
            "EQUITY_B": 0.40,  # Violates 0.25 cap
            "GOLDBEES": 0.02,  # Violates 0.05 floor
            "LTGILTBEES": 0.08,
        }
    )
    projected = project_constraints(raw)

    assert np.isclose(projected.sum(), 1.0, atol=1e-7)
    assert (projected >= 0.0).all()
    assert projected["EQUITY_A"] <= 0.25 + 1e-7
    assert projected["EQUITY_B"] <= 0.25 + 1e-7
    assert projected["GOLDBEES"] >= 0.05 - 1e-7
    assert projected["LTGILTBEES"] >= 0.05 - 1e-7


def test_overlay_actions():
    """Verify overlay shifts mass from equities to ETFs progressively."""
    overlay = RiskOverlay()
    base_w = pd.Series(
        {
            "RELIANCE": 0.25,
            "TCS": 0.25,
            "INFY": 0.25,
            "GOLDBEES": 0.125,
            "LTGILTBEES": 0.125,
        }
    )

    w_hold = overlay.apply(base_w, action=ACTION_HOLD)
    w_mild = overlay.apply(base_w, action=ACTION_DERISK_MILD)
    w_strong = overlay.apply(base_w, action=ACTION_DERISK_STRONG)

    # Sum must always equal 1.0
    assert np.isclose(w_hold.sum(), 1.0)
    assert np.isclose(w_mild.sum(), 1.0)
    assert np.isclose(w_strong.sum(), 1.0)

    # ETF exposure must strictly increase as de-risking intensity increases
    etf_hold = w_hold["GOLDBEES"] + w_hold["LTGILTBEES"]
    etf_mild = w_mild["GOLDBEES"] + w_mild["LTGILTBEES"]
    etf_strong = w_strong["GOLDBEES"] + w_strong["LTGILTBEES"]

    assert etf_mild > etf_hold
    assert etf_strong > etf_mild

    # Equities must strictly decrease
    eq_hold = w_hold["RELIANCE"] + w_hold["TCS"] + w_hold["INFY"]
    eq_mild = w_mild["RELIANCE"] + w_mild["TCS"] + w_mild["INFY"]
    eq_strong = w_strong["RELIANCE"] + w_strong["TCS"] + w_strong["INFY"]

    assert eq_mild < eq_hold
    assert eq_strong < eq_mild


def test_overlay_invalid_action_rejected():
    overlay = RiskOverlay()
    base_w = pd.Series({"RELIANCE": 0.8, "GOLDBEES": 0.2})
    with pytest.raises(ValueError):
        overlay.apply(base_w, action=99)


# ---------------------------------------------------------------------------
# 2. State & Observation Tests
# ---------------------------------------------------------------------------

def test_state_to_observation_shape_and_values(sample_portfolio_state):
    obs = state_to_observation(sample_portfolio_state)

    assert isinstance(obs, np.ndarray)
    assert obs.shape == (OBSERVATION_DIM,)
    assert obs.dtype == np.float32

    # Feature 0: drawdown = -0.08
    assert np.isclose(obs[0], -0.08)
    # Feature 1: volatility = 0.18
    assert np.isclose(obs[1], 0.18)
    # Feature 5: equity exposure = 0.80
    assert np.isclose(obs[5], 0.80)
    # Feature 6: etf exposure = 0.20
    assert np.isclose(obs[6], 0.20)
    # Feature 7: breach gap = 0.0 (since -0.08 > -0.15 threshold)
    assert np.isclose(obs[7], 0.0)


def test_state_to_observation_severe_drawdown(sample_portfolio_state):
    severe_state = PortfolioState(
        date=sample_portfolio_state.date,
        weights=sample_portfolio_state.weights,
        weekly_return=-0.12,
        monthly_return=-0.22,
        drawdown_from_peak=-0.25,  # Breaches -0.15 threshold by -0.10
        volatility=0.35,
        max_weight=0.20,
        portfolio_value=8_000_000.0,
    )
    obs = state_to_observation(severe_state)
    # Breach gap = -0.25 - (-0.15) = -0.10
    assert np.isclose(obs[7], -0.10)


# ---------------------------------------------------------------------------
# 3. Environment Tests
# ---------------------------------------------------------------------------

def test_env_reset_and_step(synthetic_market_data):
    returns_df, base_weights_df = synthetic_market_data
    env = RiskMonitorEnv(returns_df, base_weights_df)

    obs, info = env.reset(seed=123)
    assert obs.shape == (OBSERVATION_DIM,)
    assert info["portfolio_value"] == 10_000_000.0
    assert info["drawdown"] == 0.0

    # Step forward with DERISK_MILD
    next_obs, reward, terminated, truncated, step_info = env.step(ACTION_DERISK_MILD)
    assert next_obs.shape == (OBSERVATION_DIM,)
    assert isinstance(reward, float)
    assert not terminated
    assert step_info["action"] == ACTION_DERISK_MILD
    assert "net_return" in step_info


def test_env_episode_completes(synthetic_market_data):
    returns_df, base_weights_df = synthetic_market_data
    env = RiskMonitorEnv(returns_df, base_weights_df)

    obs, _ = env.reset()
    done = False
    step_count = 0

    while not done:
        action = ACTION_HOLD
        obs, reward, done, _, _ = env.step(action)
        step_count += 1

    assert done
    assert step_count == len(env.dates) - 1


# ---------------------------------------------------------------------------
# 4. Agent Tests
# ---------------------------------------------------------------------------

def test_agent_act_deterministic():
    agent = RLRiskAgent(seed=42)
    obs = np.array([-0.05, 0.15, 0.01, 0.02, 0.25, 0.80, 0.20, 0.0], dtype=np.float32)

    # In evaluate mode, repeated calls must return identical action
    act1 = agent.act(obs, evaluate=True)
    act2 = agent.act(obs, evaluate=True)
    assert act1 == act2
    assert act1 in [ACTION_HOLD, ACTION_DERISK_MILD, ACTION_DERISK_STRONG]


def test_agent_explainable_q_values():
    agent = RLRiskAgent(seed=42)
    obs = np.array([-0.05, 0.15, 0.01, 0.02, 0.25, 0.80, 0.20, 0.0], dtype=np.float32)

    q_vals = agent.get_q_values(obs)
    assert isinstance(q_vals, dict)
    assert "HOLD" in q_vals
    assert "DERISK_MILD" in q_vals
    assert "DERISK_STRONG" in q_vals
    for val in q_vals.values():
        assert isinstance(val, float)


def test_agent_training_loop(synthetic_market_data):
    returns_df, base_weights_df = synthetic_market_data
    env = RiskMonitorEnv(returns_df, base_weights_df)
    agent = RLRiskAgent(hidden_dim=32, buffer_capacity=1000, seed=42)

    history = agent.train(env, total_episodes=4, batch_size=16)

    assert len(history["episode_rewards"]) == 4
    assert len(history["episode_losses"]) == 4
    assert history["epsilon_history"][-1] < history["epsilon_history"][0]


def test_agent_save_and_load():
    agent = RLRiskAgent(hidden_dim=32, seed=42)
    obs = np.array([-0.10, 0.22, -0.01, -0.04, 0.25, 0.75, 0.25, 0.0], dtype=np.float32)
    original_q = agent.get_q_values(obs)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_rl_agent.pt"
        agent.save(model_path)
        assert model_path.exists()

        new_agent = RLRiskAgent(hidden_dim=32, seed=999)
        new_agent.load(model_path)
        loaded_q = new_agent.get_q_values(obs)

        for k in original_q:
            assert np.isclose(original_q[k], loaded_q[k], atol=1e-5)
