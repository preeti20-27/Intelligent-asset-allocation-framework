"""Deep Q-Network (DQN) Reinforcement Learning Agent for Portfolio Risk Monitoring.

This module implements the RL decision engine of the risk-monitoring overlay.
It observes the current portfolio state (drawdown, trailing volatility, returns,
exposure metrics) and estimates the expected cumulative discounted reward (Q-value)
of each risk action (HOLD, DERISK_MILD, DERISK_STRONG).

Key Features:
- Neural Q-Network with experience replay and target network stability.
- Genuine gradient-based training via PyTorch (Adam optimizer, MSE Bellman loss).
- Epsilon-greedy exploration during training, deterministic greedy evaluation.
- Fully inspectable Q-values for explainability: users can see the exact numerical
  value the agent placed on maintaining exposure vs. triggering a defensive overlay.
- Model serialization (save / load).
"""

from __future__ import annotations

import random
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from monitor.base import BaseRiskEnv, BaseRLAgent
from monitor.env import OBSERVATION_DIM
from monitor.overlay import ACTION_NAMES


class QNetwork(nn.Module):
    """Feedforward neural network mapping state features to action Q-values."""

    def __init__(self, obs_dim: int = OBSERVATION_DIM, n_actions: int = 3, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """Bounded experience replay memory buffer for off-policy DQN updates."""

    def __init__(self, capacity: int = 5000):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = random.sample(self.buffer, batch_size)
        obs_b, act_b, rew_b, next_b, done_b = zip(*batch)

        obs_t = torch.tensor(np.array(obs_b), dtype=torch.float32)
        act_t = torch.tensor(act_b, dtype=torch.long).unsqueeze(1)
        rew_t = torch.tensor(rew_b, dtype=torch.float32).unsqueeze(1)
        next_t = torch.tensor(np.array(next_b), dtype=torch.float32)
        done_t = torch.tensor(done_b, dtype=torch.float32).unsqueeze(1)

        return obs_t, act_t, rew_t, next_t, done_t

    def __len__(self) -> int:
        return len(self.buffer)


class RLRiskAgent(BaseRLAgent):
    """DQN-based risk-monitoring agent.

    Parameters
    ----------
    obs_dim : int, default=OBSERVATION_DIM
        Dimension of the input state vector.
    n_actions : int, default=3
        Number of discrete risk actions (HOLD, DERISK_MILD, DERISK_STRONG).
    hidden_dim : int, default=64
        Width of neural network layers.
    lr : float, default=1e-3
        Learning rate for Adam optimizer.
    gamma : float, default=0.95
        Discount factor for future rewards.
    epsilon_start : float, default=1.0
        Initial exploration probability.
    epsilon_min : float, default=0.05
        Minimum exploration probability floor.
    epsilon_decay : float, default=0.97
        Multiplicative decay per training episode.
    buffer_capacity : int, default=5000
        Maximum transitions stored in replay buffer.
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        obs_dim: int = OBSERVATION_DIM,
        n_actions: int = 3,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.97,
        buffer_capacity: int = 5000,
        seed: int | None = None,
    ):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.q_net = QNetwork(obs_dim, n_actions, hidden_dim)
        self.target_net = QNetwork(obs_dim, n_actions, hidden_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.memory = ReplayBuffer(capacity=buffer_capacity)

    def act(self, observation: np.ndarray, evaluate: bool = False) -> int:
        """Select an action using epsilon-greedy (training) or greedy (evaluation).

        Parameters
        ----------
        observation : np.ndarray
            Current portfolio state feature vector.
        evaluate : bool, default=False
            If True, acts purely greedily (deterministic best action).

        Returns
        -------
        int
            Selected action index (0=HOLD, 1=DERISK_MILD, 2=DERISK_STRONG).
        """
        if not evaluate and random.random() < self.epsilon:
            return random.randrange(self.n_actions)

        with torch.no_grad():
            obs_t = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
            q_values = self.q_net(obs_t)
            return int(q_values.argmax(dim=1).item())

    def get_q_values(self, observation: np.ndarray) -> Dict[str, float]:
        """Inspect estimated Q-values for all actions given a state (for explainability).

        Returns a dictionary mapping action names to their expected discounted values.
        """
        with torch.no_grad():
            obs_t = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
            q_values = self.q_net(obs_t).squeeze(0).numpy()
            return {ACTION_NAMES.get(a, f"ACTION_{a}"): float(q_values[a]) for a in range(self.n_actions)}

    def train_step(self, batch_size: int = 32) -> float:
        """Sample a batch from replay buffer and perform one gradient descent step."""
        if len(self.memory) < batch_size:
            return 0.0

        obs_b, act_b, rew_b, next_b, done_b = self.memory.sample(batch_size)

        # Current Q-values for chosen actions
        q_current = self.q_net(obs_b).gather(1, act_b)

        # Target Q-values: y = r + gamma * max_a' Q_target(s', a') * (1 - done)
        with torch.no_grad():
            q_next_max = self.target_net(next_b).max(dim=1, keepdim=True)[0]
            q_target = rew_b + (1.0 - done_b) * self.gamma * q_next_max

        loss = self.criterion(q_current, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        return float(loss.item())

    def update_target_network(self) -> None:
        """Copy weights from primary Q-network to target network."""
        self.target_net.load_state_dict(self.q_net.state_dict())

    def train(
        self,
        env: BaseRiskEnv,
        total_episodes: int = 40,
        batch_size: int = 32,
        target_update_interval: int = 5,
    ) -> Dict[str, List[float]]:
        """Train the agent over multiple episodes on the given environment.

        Parameters
        ----------
        env : BaseRiskEnv
            Simulation environment.
        total_episodes : int, default=40
            Total number of training rollouts.
        batch_size : int, default=32
            Batch size for replay buffer updates.
        target_update_interval : int, default=5
            Episodes between syncing weights to the target network.

        Returns
        -------
        Dict[str, List[float]]
            Training metrics (episode_rewards, episode_losses, epsilon_history).
        """
        history = {
            "episode_rewards": [],
            "episode_losses": [],
            "epsilon_history": [],
        }

        for ep in range(1, total_episodes + 1):
            obs, _ = env.reset()
            done = False
            total_reward = 0.0
            losses = []

            while not done:
                action = self.act(obs, evaluate=False)
                next_obs, reward, done, _, _ = env.step(action)

                self.memory.push(obs, action, reward, next_obs, done)
                loss = self.train_step(batch_size=batch_size)
                if loss > 0.0:
                    losses.append(loss)

                obs = next_obs
                total_reward += reward

            # Decay exploration rate
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            if ep % target_update_interval == 0:
                self.update_target_network()

            avg_loss = float(np.mean(losses)) if losses else 0.0
            history["episode_rewards"].append(total_reward)
            history["episode_losses"].append(avg_loss)
            history["epsilon_history"].append(self.epsilon)

        return history

    def save(self, path: str | Path) -> None:
        """Save model weights and training parameters to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "q_net_state": self.q_net.state_dict(),
            "target_net_state": self.target_net.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "obs_dim": self.obs_dim,
            "n_actions": self.n_actions,
            "epsilon": self.epsilon,
            "gamma": self.gamma,
        }
        torch.save(payload, path)

    def load(self, path: str | Path) -> None:
        """Load model weights and parameters from disk."""
        path = Path(path)
        payload = torch.load(path, map_location=torch.device("cpu"), weights_only=True)
        self.q_net.load_state_dict(payload["q_net_state"])
        self.target_net.load_state_dict(payload["target_net_state"])
        if "optimizer_state" in payload:
            self.optimizer.load_state_dict(payload["optimizer_state"])
        self.epsilon = payload.get("epsilon", self.epsilon_min)
        self.gamma = payload.get("gamma", self.gamma)
        self.q_net.eval()
        self.target_net.eval()
