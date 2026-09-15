# Monitoring Segment

## Phase 1

This module implements the rule-based monitoring stub for the
Intelligent Asset Allocation Framework.

The Phase 1 monitor is a placeholder for the reinforcement
learning monitoring layer planned for Phase 2.

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
Actions

The monitor can return:

HOLD
WARN_DRAWDOWN
WARN_CONCENTRATION
Rules
Drawdown

If portfolio drawdown from its peak is at least 15%:

WARN_DRAWDOWN
Concentration

If the maximum asset weight is at least 40%:

WARN_CONCENTRATION
Otherwise
HOLD
Logging

Whenever a warning is triggered, the monitor records the
warning and relevant portfolio information in:

monitor_warnings.log
Testing

The monitoring logic is tested using pytest.

Run:

python -m pytest -v
Phase 2

The rule-based monitor will eventually be replaced by a
reinforcement learning monitoring agent after real Phase 1
backtest results are available.