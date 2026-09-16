"""Plotting utilities for backtest results."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: never try to open a GUI window
import matplotlib.pyplot as plt
import pandas as pd


def plot_equity_curve(equity_curve: pd.Series, title: str, save_path: str) -> None:
    """Save a line chart of portfolio value over time to ``save_path`` (PNG).

    Creates any missing parent directories of ``save_path``.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(equity_curve.index, equity_curve.values, color="#2b6cb0")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
