# Strategy explorer (interface)

A point-and-click app for running and comparing the project's portfolio
allocation strategies without touching code.

## What we learned

**Read this before reading any chart.** In this project's significance testing
(`scripts/output/significance_results.csv`), the one statistically proven result
is that **plain HRP has a significantly higher out-of-sample Sharpe ratio
(2023-01 to 2026-09) than naive baselines**: equal weight and Black-Litterman
with no views (and also the mean-reversion VI-HRP signal).
**No tested VI-HRP view signal beats plain HRP with statistical significance in
any tested period.**

A strategy showing a higher Sharpe ratio in the app is a point estimate, not
evidence that it is better. Use Compare mode to see whether a difference is
significant. The app's banner is generated from the CSV at startup, so it always
reflects the committed results, even if this README goes out of date.

## How to run

From the project root, with the project's virtual environment active:

```
pip install streamlit      # once; not yet in pyproject.toml
streamlit run interface/app.py
```

The app opens in your browser. The first time you choose a strategy, its backtest
takes about a minute (a progress bar shows which one is running). Results are then
cached on disk, so later runs are instant. The cache is cleared automatically
whenever an allocator, a view signal, the backtest engine or the data changes.

## What it does

**Sidebar controls**
- **Mode**: *Single strategy* or *Compare strategies*.
- **Tested period**: a shortcut to one of the four periods the significance tests
  covered, or *Custom*.
- **Date range**: limited to what the data supports, 2019-04-05 to 2026-09-11.
  The start is the first date with a full year of history for every asset, and the
  range must cover at least 60 trading days.
- **Rebalance frequency**: Weekly (what every committed result used) or Monthly.
- **Allocator** (single mode): Plain HRP, Equal weight, Black-Litterman (no
  views), HERC, HCAA, or VI-HRP. Picking VI-HRP shows a **View signal** choice:
  Momentum, Mean reversion, Standardized momentum + reversal, or Volatility
  regime.
- **Strategies** (compare mode): any two or more of the nine strategies.

**Single strategy** shows the Sharpe ratio, total return, annualized volatility and
maximum drawdown for the chosen period. It also shows the equity curve against
plain HRP, the portfolio weights held on the last day of the period, and a test of
whether the Sharpe difference from plain HRP is statistically significant.

**Compare strategies** shows every selected strategy's equity curve, a metrics
table, and a significance test for every pair:
- If you chose one of the four tested periods with weekly rebalancing, the app
  shows the committed results from the CSV.
- Otherwise it runs the same block-bootstrap test live (5,000 resamples per pair).
  A "Computing..." spinner shows while it runs.
- p-values are Holm-adjusted for comparing several pairs at once.

## How it is built

- `core.py` holds all the logic and imports no Streamlit. It only calls the
  project's existing, tested functions (`allocators/`, `views/`,
  `backtest/engine.py`, `eval/`), wired exactly as
  `scripts/apply_significance_tests.py` wires them. Each strategy is backtested once
  over the full history, and your date range is a slice of that run, the same
  method the committed results used. So a tested period reproduces the committed
  numbers exactly.
- `app.py` only lays out what `core.py` returns.
- Tests: `tests/unit/test_interface.py` (logic only; rendering is not tested).
