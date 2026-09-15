# Monitoring Segment

Portfolio monitoring, reinforcement learning, rules, and overlay components.

## Phase 1

`monitor.py`/`state.py`/`rules.py`/`logger.py` implement the rule-based
monitoring stub for the Intelligent Asset Allocation Framework. This is a
placeholder for the reinforcement learning monitoring layer planned for
Phase 2 (`rl_agent.py`, `env.py`, `overlay.py`, `base.py`).

## Input

The monitor receives a `PortfolioState` containing:

- Date
- Portfolio weights
- Weekly return
- Monthly return
- Drawdown from peak
- Volatility
- Maximum asset weight
- Portfolio value

## Function

```python
check_portfolio(state) -> action
```

## Actions

The monitor can return:

- `HOLD`
- `WARN_DRAWDOWN`
- `WARN_CONCENTRATION`

## Rules

**Drawdown**: if portfolio drawdown from its peak is at least 15%, `WARN_DRAWDOWN`.

**Concentration**: if the maximum asset weight is at least 40%, `WARN_CONCENTRATION`.

**Otherwise**: `HOLD`.

## Logging

Whenever a warning is triggered, the monitor records the warning and
relevant portfolio information in `monitor_warnings.log`.

## Testing

The monitoring logic is tested using pytest, in `tests/unit/test_monitor.py`.

## Phase 2

The rule-based monitor will eventually be replaced by a reinforcement
learning monitoring agent after real Phase 1 backtest results are available.
