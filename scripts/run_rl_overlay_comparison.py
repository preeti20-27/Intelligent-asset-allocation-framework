"""Comparison script: Baseline HRP vs. HRP + RL Risk-Monitoring Overlay.

Runs an end-to-end evaluation using real market data:
1. Loads processed daily prices and returns.
2. Applies the same non-null-fraction warm-up logic as run_backtest.py.
3. Splits into an in-sample training period and an out-of-sample evaluation period
   to strictly prevent look-ahead bias and future-data leakage.
4. Generates base HRP weights.
5. Trains the RLRiskAgent strictly on the in-sample period.
6. Evaluates Baseline HRP vs. HRP + RL Overlay on:
   - Out-of-Sample Test Period (unseen data)
   - Full Period
7. Reports Total Return, Annualized Sharpe, Max Drawdown, Annualized Volatility,
   and Total Turnover.
8. Saves comparison plot and summary statistics.

Usage:
    python scripts/run_rl_overlay_comparison.py
    python scripts/run_rl_overlay_comparison.py --episodes 50 --split-date 2023-01-01
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from allocators.hrp import allocate as hrp_allocate  # noqa: E402
from backtest.calendar import DEFAULT_PRICES_PATH, weekly_rebalance_dates  # noqa: E402
from backtest.costs import transaction_cost  # noqa: E402
from backtest.engine import (  # noqa: E402
    DEFAULT_WINDOW,
    BacktestEngine,
    apply_constraints,
    load_default_returns,
)
from eval.metrics import max_drawdown, sharpe_ratio, total_return  # noqa: E402
from monitor.env import RiskMonitorEnv, state_to_observation  # noqa: E402
from monitor.overlay import (  # noqa: E402
    ACTION_DERISK_MILD,
    ACTION_DERISK_STRONG,
    ACTION_HOLD,
    ACTION_NAMES,
    RiskOverlay,
)
from monitor.rl_agent import RLRiskAgent  # noqa: E402


def compute_turnover(weights_series_list: list[pd.Series]) -> float:
    """Compute total one-way portfolio turnover across all daily states."""
    if len(weights_series_list) < 2:
        return 0.0
    total_turnover = 0.0
    prev_w = pd.Series(dtype=float)
    for w in weights_series_list:
        if len(prev_w) > 0:
            all_tickers = prev_w.index.union(w.index)
            diff = (w.reindex(all_tickers, fill_value=0.0) - prev_w.reindex(all_tickers, fill_value=0.0)).abs()
            total_turnover += float(diff.sum()) * 0.5  # one-way turnover
        prev_w = w
    return total_turnover


def precompute_base_hrp_weights(
    returns_df: pd.DataFrame,
    trading_days: pd.DatetimeIndex,
    rebalance_dates: set,
    window: int = DEFAULT_WINDOW,
) -> pd.DataFrame:
    """Precompute weekly HRP constrained weights held daily over the date range."""
    print("Pre-computing base HRP allocator weights for all rebalance dates...")
    current_weights = None
    daily_weights = []

    for date in trading_days:
        if current_weights is None or date in rebalance_dates:
            history = returns_df.loc[:date]
            window_returns = history.iloc[:-1].tail(window)
            raw_weights = hrp_allocate(window_returns)
            current_weights = apply_constraints(raw_weights)

        daily_weights.append(current_weights.copy())

    base_weights_df = pd.DataFrame(daily_weights, index=trading_days)
    print(f"Base weights precomputed for {len(base_weights_df)} trading days.")
    return base_weights_df


def run_overlaid_backtest(
    returns_df: pd.DataFrame,
    base_weights_df: pd.DataFrame,
    agent: RLRiskAgent | None,
    overlay: RiskOverlay,
    trading_days: pd.DatetimeIndex,
) -> tuple[pd.Series, list, list[int]]:
    """Execute a backtest using the given base weights, optionally with an active RL overlay."""
    engine = BacktestEngine(returns_df)
    states = []
    actions = []
    current_state = None

    for date in trading_days:
        base_w = base_weights_df.loc[date]

        if agent is not None and current_state is not None:
            obs = state_to_observation(current_state)
            action = agent.act(obs, evaluate=True)
        else:
            action = ACTION_HOLD

        actions.append(action)
        effective_w = overlay.apply(base_w, action=action, state=current_state)
        current_state = engine.step(date, effective_w)
        states.append(current_state)

    equity_curve = engine._value_history.copy()
    return equity_curve, states, actions


def print_metrics_table(
    title: str,
    baseline_equity: pd.Series,
    baseline_states: list,
    overlay_equity: pd.Series,
    overlay_states: list,
    actions: list[int],
):
    """Print formatted comparative performance table."""
    base_ret = total_return(baseline_equity)
    over_ret = total_return(overlay_equity)

    base_daily = baseline_equity.pct_change().dropna()
    over_daily = overlay_equity.pct_change().dropna()

    base_sharpe = sharpe_ratio(base_daily)
    over_sharpe = sharpe_ratio(over_daily)

    base_mdd_res = max_drawdown(baseline_equity)
    over_mdd_res = max_drawdown(overlay_equity)

    base_vol = float(base_daily.std() * np.sqrt(252))
    over_vol = float(over_daily.std() * np.sqrt(252))

    base_turnover = compute_turnover([pd.Series(s.weights) for s in baseline_states])
    over_turnover = compute_turnover([pd.Series(s.weights) for s in overlay_states])

    print(f"\n=======================================================")
    print(f" {title}")
    print(f"=======================================================")
    print(f"{'Metric':<24} | {'Baseline HRP':<14} | {'HRP + RL Overlay':<16}")
    print(f"-------------------------+----------------+-----------------")
    print(f"{'Total Return':<24} | {base_ret*100:>13.2f}% | {over_ret*100:>15.2f}%")
    print(f"{'Annualized Sharpe':<24} | {base_sharpe:>14.2f} | {over_sharpe:>16.2f}")
    print(f"{'Max Drawdown':<24} | {base_mdd_res.max_drawdown*100:>13.2f}% | {over_mdd_res.max_drawdown*100:>15.2f}%")
    print(f"{'Annualized Volatility':<24} | {base_vol*100:>13.2f}% | {over_vol*100:>15.2f}%")
    print(f"{'Total Turnover (x)':<24} | {base_turnover:>14.2f} | {over_turnover:>16.2f}")
    print(f"-------------------------+----------------+-----------------")

    action_counts = Counter(actions)
    n_days = len(actions)
    print("RL Overlay Action Distribution:")
    for a in [ACTION_HOLD, ACTION_DERISK_MILD, ACTION_DERISK_STRONG]:
        name = ACTION_NAMES[a]
        cnt = action_counts.get(a, 0)
        pct = (cnt / n_days) * 100 if n_days > 0 else 0.0
        print(f"  {name:<16}: {cnt:>5} days ({pct:5.1f}%)")
    print("=======================================================\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episodes",
        type=int,
        default=40,
        help="Number of training episodes for the RL agent (default 40).",
    )
    parser.add_argument(
        "--split-date",
        type=str,
        default="2023-01-02",
        help="Train/test split boundary date (default 2023-01-02).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=DEFAULT_WINDOW,
        help=f"Rolling lookback window fed to HRP (default {DEFAULT_WINDOW}).",
    )
    args = parser.parse_args()

    prices_df = pd.read_parquet(DEFAULT_PRICES_PATH)
    returns_df = load_default_returns()

    trading_days = pd.DatetimeIndex(sorted(prices_df.index.unique()))
    rebalance_dates = set(pd.DatetimeIndex(weekly_rebalance_dates(trading_days)))

    # Warm-up check (same as run_backtest.py)
    rolling_non_null_frac = returns_df.notna().rolling(args.window).mean()
    window_ok = (rolling_non_null_frac >= 0.90).all(axis=1).to_numpy()
    bad_positions = np.where(~window_ok)[0]
    last_bad_window_end = int(bad_positions[-1]) if len(bad_positions) else args.window - 2
    start_pos = last_bad_window_end + 2

    start_date = returns_df.index[start_pos]
    end_date = returns_df.index[-1]
    split_date = pd.Timestamp(args.split_date)

    available_days = trading_days[(trading_days >= start_date) & (trading_days <= end_date)]
    train_days = available_days[available_days < split_date]
    test_days = available_days[available_days >= split_date]

    print(f"Full backtest date range : {start_date.date()} to {end_date.date()} ({len(available_days)} days)")
    print(f"In-Sample Train range    : {train_days[0].date()} to {train_days[-1].date()} ({len(train_days)} days)")
    print(f"Out-of-Sample Test range : {test_days[0].date()} to {test_days[-1].date()} ({len(test_days)} days)")

    # 1. Precompute base weights over the full history
    base_weights_df = precompute_base_hrp_weights(
        returns_df=returns_df,
        trading_days=available_days,
        rebalance_dates=rebalance_dates,
        window=args.window,
    )

    overlay = RiskOverlay()

    # 2. Train the RL agent strictly on the training period
    print(f"\nTraining RL Risk Agent for {args.episodes} episodes on in-sample data ({len(train_days)} days)...")
    train_env = RiskMonitorEnv(
        returns_df=returns_df.loc[train_days],
        base_weights_df=base_weights_df.loc[train_days],
        overlay=overlay,
        drawdown_penalty_coef=2.5,
        vol_penalty_coef=0.8,
        vol_target=0.18,
        action_switch_penalty=0.001,
        return_scale=100.0,
    )

    agent = RLRiskAgent(seed=42, lr=1e-3, gamma=0.95, epsilon_start=1.0, epsilon_decay=0.95)
    train_history = agent.train(train_env, total_episodes=args.episodes, batch_size=32)

    avg_final_loss = np.mean(train_history["episode_losses"][-5:]) if train_history["episode_losses"] else 0.0
    print(f"Training completed. Final average loss: {avg_final_loss:.5f}, Final epsilon: {agent.epsilon:.3f}")

    # Save trained model
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    model_save_path = output_dir / "rl_risk_agent.pt"
    agent.save(model_save_path)
    print(f"Trained model checkpoint saved to: {model_save_path}")

    # 3. Out-of-Sample (Test Set) Evaluation
    print("\nRunning Out-of-Sample (Unseen Test Data) Backtest...")
    base_test_equity, base_test_states, _ = run_overlaid_backtest(
        returns_df=returns_df,
        base_weights_df=base_weights_df,
        agent=None,  # Pure baseline
        overlay=overlay,
        trading_days=test_days,
    )
    over_test_equity, over_test_states, test_actions = run_overlaid_backtest(
        returns_df=returns_df,
        base_weights_df=base_weights_df,
        agent=agent,  # RL overlay active
        overlay=overlay,
        trading_days=test_days,
    )

    print_metrics_table(
        title=f"OUT-OF-SAMPLE TEST RESULTS ({test_days[0].date()} to {test_days[-1].date()})",
        baseline_equity=base_test_equity,
        baseline_states=base_test_states,
        overlay_equity=over_test_equity,
        overlay_states=over_test_states,
        actions=test_actions,
    )

    # 4. Full Period Evaluation
    print("Running Full-Period Backtest...")
    base_full_equity, base_full_states, _ = run_overlaid_backtest(
        returns_df=returns_df,
        base_weights_df=base_weights_df,
        agent=None,
        overlay=overlay,
        trading_days=available_days,
    )
    over_full_equity, over_full_states, full_actions = run_overlaid_backtest(
        returns_df=returns_df,
        base_weights_df=base_weights_df,
        agent=agent,
        overlay=overlay,
        trading_days=available_days,
    )

    print_metrics_table(
        title=f"FULL PERIOD RESULTS ({available_days[0].date()} to {available_days[-1].date()})",
        baseline_equity=base_full_equity,
        baseline_states=base_full_states,
        overlay_equity=over_full_equity,
        overlay_states=over_full_states,
        actions=full_actions,
    )

    # 5. Plot Comparison
    plt.figure(figsize=(12, 6))
    plt.plot(base_full_equity.index, base_full_equity / base_full_equity.iloc[0], label="Baseline HRP", color="steelblue", linewidth=1.5)
    plt.plot(over_full_equity.index, over_full_equity / over_full_equity.iloc[0], label="HRP + RL Risk Overlay", color="darkgreen", linewidth=1.5)
    plt.axvline(split_date, color="crimson", linestyle="--", alpha=0.7, label=f"Out-of-Sample Split ({split_date.date()})")
    plt.title("Equity Curve: Baseline HRP vs. HRP + RL Risk-Monitoring Overlay", fontsize=14, fontweight="bold")
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Normalized Value (Base = 1.0)", fontsize=11)
    plt.legend(loc="upper left", frameon=True)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    plot_path = output_dir / "equity_curve_hrp_vs_rl_overlay.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Comparison plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
