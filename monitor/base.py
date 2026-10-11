"""Abstract base classes and interfaces for the monitoring subsystem.

Following CONTRIBUTING.md architectural guidelines:
"No component imports another component's concrete class; components may
import only interfaces."
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from monitor.state import PortfolioState


class BaseRiskEnv(ABC):
    """Abstract interface for reinforcement learning risk environments."""

    @abstractmethod
    def reset(self, seed: int | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to the beginning of the episode.

        Returns
        -------
        Tuple[np.ndarray, Dict[str, Any]]
            Initial observation vector and metadata info dict.
        """
        pass

    @abstractmethod
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Advance the environment by one step given an action.

        Returns
        -------
        Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]
            (next_observation, reward, terminated, truncated, info)
        """
        pass


class BaseRLAgent(ABC):
    """Abstract interface for RL risk-monitoring agents."""

    @abstractmethod
    def act(self, observation: np.ndarray, evaluate: bool = False) -> int:
        """Select an action given an observation vector.

        Parameters
        ----------
        observation : np.ndarray
            Current portfolio state feature vector.
        evaluate : bool, default=False
            If True, choose the greedy/deterministic best action without exploration.

        Returns
        -------
        int
            Selected discrete action index.
        """
        pass

    @abstractmethod
    def train(self, env: BaseRiskEnv, total_episodes: int) -> Dict[str, Any]:
        """Train the agent in the provided environment.

        Parameters
        ----------
        env : BaseRiskEnv
            Environment to train on.
        total_episodes : int
            Number of training episodes.

        Returns
        -------
        Dict[str, Any]
            Training log metrics (e.g. episodic rewards, loss history).
        """
        pass

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist model weights and configuration to disk."""
        pass

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """Load model weights and configuration from disk."""
        pass


class BaseOverlay(ABC):
    """Abstract interface for portfolio risk overlays."""

    @abstractmethod
    def apply(
        self,
        base_weights: pd.Series,
        action: int,
        state: PortfolioState | None = None,
    ) -> pd.Series:
        """Adjust base portfolio weights according to the monitor action.

        Parameters
        ----------
        base_weights : pd.Series
            Raw weights output by the allocator (e.g. HRP / VI-HRP).
        action : int
            Risk decision selected by the monitor or RL agent.
        state : PortfolioState, optional
            Current portfolio state context.

        Returns
        -------
        pd.Series
            Adjusted, constraint-compliant target weights summing to 1.0.
        """
        pass


class BaseMonitor(ABC):
    """Abstract interface for passive or active portfolio monitors."""

    @abstractmethod
    def check_portfolio(self, state: PortfolioState) -> Any:
        """Evaluate current portfolio state and return monitoring decision."""
        pass
