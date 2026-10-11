# Monitoring Segment

Portfolio monitoring, reinforcement learning, rules, and overlay components.

---

## Architecture Overview

The monitoring subsystem operates at two levels:

1. **Phase 1: Rule-Based Passive Surveillance** (`monitor.py`, `state.py`, `rules.py`, `logger.py`)
   * Observes daily portfolio health and flags threshold breaches to `monitor_warnings.log`.
2. **Phase 2: Reinforcement Learning Risk-Monitoring Overlay** (`base.py`, `env.py`, `rl_agent.py`, `overlay.py`)
   * Active macro overlay operating on top of base allocator weights (HRP / VI-HRP).
   * Dynamically shifts risk exposure between equities and defensive ETFs (`GOLDBEES`, `LTGILTBEES`) when risk conditions deteriorate, while strictly respecting all portfolio constraints.

```
                    +-----------------------------+
                    | Allocator (HRP / VI-HRP)    |
                    +--------------+--------------+
                                   | Base Weights (w)
                                   v
+------------------+        +--------------+        +--------------------------+
|  PortfolioState  |------->|   RL Agent   |------->|   RiskOverlay            |
| (drawdown, vol,  |        | (DQN Policy) | Action | - Adjusts equity/ETF mix |
|  trailing rets)  |        +--------------+        | - Applies constraints    |
+------------------+                                +------------+-------------+
                                                                 | Target Weights (w*)
                                                                 v
                                                    +--------------------------+
                                                    | BacktestEngine.step()    |
                                                    +--------------------------+
```

---

## Phase 1: Rule-Based Passive Monitor

### Input
The monitor receives a [`PortfolioState`](file:///c:/Users/v/Downloads/VIT%20Downloads/Intelligent-asset-allocation-framework-1/monitor/state.py) containing:
- `date`: Trading date string
- `weights`: Current portfolio asset weights dict
- `weekly_return`: Trailing 5-day return
- `monthly_return`: Trailing 21-day return
- `drawdown_from_peak`: Portfolio drawdown from cumulative high-water mark
- `volatility`: Trailing 20-day annualized return standard deviation
- `max_weight`: Highest individual asset weight
- `portfolio_value`: Total portfolio mark-to-market value

### Rules
- **Drawdown Rule**: If drawdown from peak is $\le -15\%$, returns `WARN_DRAWDOWN` and logs to `monitor_warnings.log`.
- **Concentration Rule**: If maximum single-asset weight is $\ge 40\%$, returns `WARN_CONCENTRATION` and logs to `monitor_warnings.log`.
- **Default**: Returns `HOLD`.

---

## Phase 2: Reinforcement Learning Risk Overlay

### 1. State Representation (`monitor/env.py`)
The environment converts each `PortfolioState` into a fixed 8-dimensional normalized feature vector:
1. `drawdown_from_peak`: Portfolio drawdown fraction in $[-1.0, 0.0]$.
2. `trailing_annualized_vol`: 20-day annualized volatility.
3. `weekly_return`: Trailing 5-day return.
4. `monthly_return`: Trailing 21-day return.
5. `max_weight`: Largest individual asset weight.
6. `equity_exposure`: Sum of weights allocated to equities.
7. `etf_exposure`: Sum of weights allocated to defensive ETFs.
8. `drawdown_breach_gap`: Distance breaching past the Phase 1 $-15\%$ drawdown threshold (`min(0.0, drawdown - (-0.15))`).

### 2. Action Space (`monitor/overlay.py`)
The RL agent outputs one of three discrete risk decisions:
* `ACTION_HOLD (0)`: Maintain base allocator weights without modification.
* `ACTION_DERISK_MILD (1)`: Mild risk-off tilt. Proportionally shifts $15\%$ of total equity exposure into safe-haven ETFs (`GOLDBEES` and `LTGILTBEES`).
* `ACTION_DERISK_STRONG (2)`: Strong risk-off tilt. Proportionally shifts $35\%$ of total equity exposure into safe-haven ETFs.

### 3. Constraint Preservation
Every weight vector produced by [`RiskOverlay.apply()`](file:///c:/Users/v/Downloads/VIT%20Downloads/Intelligent-asset-allocation-framework-1/monitor/overlay.py) is projected through the repository's iterative water-filling constraint projection:
- Long-only ($w_i \ge 0$).
- Equities capped at $\le 25\%$ (`MAX_EQUITY_WEIGHT`).
- ETFs floored at $\ge 5\%$ each (`MIN_ETF_WEIGHT` for `GOLDBEES` and `LTGILTBEES`).
- Sum of weights strictly equals $1.0$.

### 4. Risk-Adjusted Reward (`monitor/env.py`)
The agent's objective balances capital compounding against downside risk:
$$R_t = \text{Net Return} \times 100 - \lambda_{\text{dd\_base}} |\text{Drawdown}| - \lambda_{\text{breach}} \mathbb{I}(\text{DD} < -0.15)(\text{DD} + 0.15)^2 - \lambda_{\text{vol}} \max(0, \sigma - 0.18) - \lambda_{\text{switch}} \mathbb{I}(A_t \neq A_{t-1})$$

### 5. Learning Algorithm & Explainability (`monitor/rl_agent.py`)
* Implements a Deep Q-Network (DQN) with an experience replay buffer and target network.
* Neural Network: Multi-layer perceptron with layer normalization and ReLU activations.
* Epsilon-greedy exploration during training, deterministic greedy evaluation.
* **Explainability**: [`RLRiskAgent.get_q_values(obs)`](file:///c:/Users/v/Downloads/VIT%20Downloads/Intelligent-asset-allocation-framework-1/monitor/rl_agent.py) outputs expected discounted values for each action, allowing transparent inspection of why a specific risk action was chosen.
* Model serialization via `agent.save(path)` and `agent.load(path)`.

---

## How to Run

### Run Unit Tests
```bash
python -m pytest tests/unit/test_monitor.py tests/unit/test_rl_monitor.py -v
```

### Run Full Test Suite (All 90 Unit Tests)
```bash
python -m pytest
```

### Run Baseline HRP vs. RL Overlay Comparison Backtest
Trains the agent strictly on the in-sample training period (no look-ahead bias) and evaluates on unseen out-of-sample test data:
```bash
python scripts/run_rl_overlay_comparison.py --episodes 40 --split-date 2023-01-02
```

Outputs:
* Performance summary table (Total Return, Sharpe, Max Drawdown, Volatility, Turnover).
* Action distribution (% of time in HOLD, DERISK_MILD, DERISK_STRONG).
* Checkpoint: `outputs/rl_risk_agent.pt`
* Plot: `outputs/equity_curve_hrp_vs_rl_overlay.png`
