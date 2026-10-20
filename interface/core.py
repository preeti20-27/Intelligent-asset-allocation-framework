"""UI-independent logic behind ``interface/app.py``.

Everything the Streamlit app needs that is not rendering lives here, so it
can be unit-tested without Streamlit (``tests/unit/test_interface.py``).
Nothing in this module implements allocation, backtesting, metrics or
statistics: it only selects and wires the project's existing, tested
public functions --

* allocators: ``allocators.{hrp,equal_weight,bl,herc,hcaa,vihrp}.allocate``
* view signals: the four ``views.*`` ``ViewGenerator`` implementations
* backtest: ``backtest.engine.BacktestEngine.run`` and
  ``backtest.calendar.weekly_rebalance_dates``
* metrics: ``eval.metrics.{sharpe_ratio,max_drawdown,total_return}``
* significance: ``eval.bootstrap.sharpe_difference_test``,
  ``eval.mcs.holm_adjust``, and the committed
  ``scripts/output/significance_results.csv``

-- and every strategy is wired EXACTLY as
``scripts/apply_significance_tests.py`` wired it (same generator
hyperparameters, same long-history feed for the standardized generator,
same warm-up start, same segment boundaries, same bootstrap seed and
resample count). That is what makes the committed CSV a valid answer for
a user whose selection matches a tested segment, and makes a live
bootstrap on such a selection reproduce the CSV exactly.

How a backtest maps to the user's date range: each strategy is backtested
ONCE, continuously, from the first feasible date (the warm-up start) to
the end of the data, and the user's range is a slice of that one equity
curve -- the same methodology every comparison script in this project
used. The alternative (starting a fresh backtest at the user's start
date) would make the first week's weights and transaction costs depend
on where the user happened to click, and would not reproduce the tested
numbers.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, ContextManager

import numpy as np
import pandas as pd

from allocators.bl import allocate as bl_allocate
from allocators.equal_weight import allocate as equal_weight_allocate
from allocators.hcaa import allocate as hcaa_allocate
from allocators.herc import allocate as herc_allocate
from allocators.hrp import allocate as hrp_allocate
from allocators.vihrp import allocate as vihrp_allocate
from backtest.calendar import weekly_rebalance_dates
from backtest.engine import DEFAULT_WINDOW, BacktestEngine
from eval.bootstrap import sharpe_difference_test
from eval.mcs import holm_adjust
from eval.metrics import max_drawdown, sharpe_ratio, total_return
from tree.cluster_tree import ClusterTree
from views.mean_reversion_generator import generate_mean_reversion_views
from views.momentum_generator import LOOKBACK, TILT_SCALE, generate_momentum_views
from views.standardized_momentum_reversal_generator import (
    MOMENTUM_FORMATION_DAYS,
    MOMENTUM_SKIP_DAYS,
    REVERSAL_FORMATION_DAYS,
    TILT_SCALE as STD_TILT_SCALE,
    generate_standardized_momentum_reversal_views,
)
from views.volatility_regime_generator import (
    LONG_VOL_WINDOW,
    SHORT_VOL_WINDOW,
    TILT_SCALE as VOL_TILT_SCALE,
    generate_volatility_regime_views,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SIGNIFICANCE_CSV_PATH = PROJECT_ROOT / "scripts" / "output" / "significance_results.csv"

TRADING_DAYS_PER_YEAR = 252
ALPHA = 0.05

# Must match scripts/apply_significance_tests.py so committed and live
# results are produced by the identical procedure.
N_BOOT = 5000
BOOTSTRAP_SEED = 20261020
IN_SAMPLE_END = pd.Timestamp("2022-12-31")
SUB_SPLIT = pd.Timestamp("2021-02-28")
OOS_SEGMENT = "OOS_2023-01_to_2026-09"
SEGMENT_LABELS = {
    "subA_2019-04_to_2021-02": "In-sample A, 2019-04 to 2021-02 (incl. COVID crash)",
    "subB_2021-03_to_2022-12": "In-sample B, 2021-03 to 2022-12",
    "full_in_sample": "Full in-sample, 2019-04 to 2022-12",
    OOS_SEGMENT: "Out-of-sample, 2023-01 to 2026-09",
}

# Shortest range the UI accepts: below ~3 months a Sharpe ratio (and a
# block bootstrap of it) is too noisy to say anything.
MIN_RANGE_TRADING_DAYS = 60

PLAIN_HRP = "plain_hrp"
VIHRP = "vihrp"

# Selectable allocators, in display order. "vihrp" is a family that needs a
# signal; the rest are complete strategies on their own.
ALLOCATOR_LABELS = {
    PLAIN_HRP: "Plain HRP",
    "equal_weight": "Equal weight",
    "black_litterman": "Black-Litterman (no views)",
    "herc": "HERC",
    "hcaa": "HCAA",
    VIHRP: "VI-HRP (view-informed HRP)",
}

# Looked up at call time (not captured at import), so tests can substitute
# stubs with monkeypatch.setitem and observe exactly what gets called.
BASE_ALLOCATORS: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    PLAIN_HRP: hrp_allocate,
    "equal_weight": equal_weight_allocate,
    "black_litterman": bl_allocate,
    "herc": herc_allocate,
    "hcaa": hcaa_allocate,
}


@dataclass(frozen=True)
class SignalSpec:
    label: str
    generator: Callable
    kwargs: dict = field(default_factory=dict)
    # Generators that need more history than the engine's rolling window
    # get a longer trailing slice of the full panel (as in the script).
    history_days: int | None = None


SIGNALS: dict[str, SignalSpec] = {
    "momentum": SignalSpec(
        "Momentum", generate_momentum_views, {"lookback": LOOKBACK, "tilt_scale": TILT_SCALE}
    ),
    "mean_reversion": SignalSpec(
        "Mean reversion", generate_mean_reversion_views, {"lookback": LOOKBACK, "tilt_scale": TILT_SCALE}
    ),
    "standardized": SignalSpec(
        "Standardized momentum + reversal",
        generate_standardized_momentum_reversal_views,
        {
            "momentum_formation_days": MOMENTUM_FORMATION_DAYS,
            "momentum_skip_days": MOMENTUM_SKIP_DAYS,
            "reversal_formation_days": REVERSAL_FORMATION_DAYS,
            "tilt_scale": STD_TILT_SCALE,
        },
        history_days=REVERSAL_FORMATION_DAYS + 50,
    ),
    "volatility_regime": SignalSpec(
        "Volatility regime",
        generate_volatility_regime_views,
        {"short_vol_window": SHORT_VOL_WINDOW, "long_vol_window": LONG_VOL_WINDOW, "tilt_scale": VOL_TILT_SCALE},
    ),
}

REBALANCE_FREQUENCIES = {"weekly": "Weekly (tested)", "monthly": "Monthly"}


# ---------------------------------------------------------------------------
# Strategy selection -> allocator function
# ---------------------------------------------------------------------------

def strategy_key(allocator: str, signal: str | None = None) -> str:
    """Canonical strategy id, matching the names in significance_results.csv."""
    if allocator == VIHRP:
        if signal not in SIGNALS:
            raise ValueError(f"VI-HRP needs a view signal, one of {list(SIGNALS)}; got {signal!r}.")
        return f"vihrp_{signal}"
    if allocator not in BASE_ALLOCATORS:
        raise ValueError(f"Unknown allocator {allocator!r}; expected one of {list(ALLOCATOR_LABELS)}.")
    if signal is not None:
        raise ValueError(f"A view signal only applies to VI-HRP, not {allocator!r}.")
    return allocator


def parse_strategy_key(key: str) -> tuple[str, str | None]:
    """Inverse of ``strategy_key``."""
    if key.startswith("vihrp_"):
        signal = key[len("vihrp_"):]
        strategy_key(VIHRP, signal)  # validates
        return VIHRP, signal
    strategy_key(key)
    return key, None


ALL_STRATEGIES = [strategy_key(a) for a in BASE_ALLOCATORS] + [strategy_key(VIHRP, s) for s in SIGNALS]


def strategy_label(key: str) -> str:
    allocator, signal = parse_strategy_key(key)
    if signal is None:
        return ALLOCATOR_LABELS[allocator]
    return f"VI-HRP: {SIGNALS[signal].label}"


def build_allocator_fn(key: str, returns_full: pd.DataFrame | None = None) -> Callable[[pd.DataFrame], pd.Series]:
    """The ``allocator_fn(window_returns) -> weights`` BacktestEngine.run expects.

    Base allocators are passed through unchanged. VI-HRP builds the
    ClusterTree over the engine's window, asks the chosen ViewGenerator
    for views (with its calibrated hyperparameters), and hands them to
    ``allocators.vihrp.allocate`` -- identical to the backtest scripts.
    ``returns_full`` is required only for signals with ``history_days``.
    """
    allocator, signal = parse_strategy_key(key)
    if signal is None:
        base = BASE_ALLOCATORS[allocator]
        return lambda window_returns: base(window_returns)

    spec = SIGNALS[signal]
    if spec.history_days is not None and returns_full is None:
        raise ValueError(f"The {spec.label} signal needs the full returns panel (returns_full).")

    def allocator_fn(window_returns: pd.DataFrame) -> pd.Series:
        tree = ClusterTree.build(window_returns.corr())
        signal_input = window_returns
        if spec.history_days is not None:
            signal_input = returns_full.loc[: window_returns.index[-1]].tail(spec.history_days)
        views = spec.generator(signal_input, tree, **spec.kwargs)
        return vihrp_allocate(window_returns, views=views)

    return allocator_fn


# ---------------------------------------------------------------------------
# Calendar, date-range bounds and validation
# ---------------------------------------------------------------------------

def rebalance_dates(frequency: str, trading_days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Rebalance schedule for the engine's ``rebalance_dates`` argument.

    Weekly uses ``backtest.calendar.weekly_rebalance_dates`` (the schedule
    every committed result used). Monthly is the first trading day of each
    calendar month -- a different input to the same engine, not a
    different engine.
    """
    trading_days = pd.DatetimeIndex(trading_days)
    if frequency == "weekly":
        return weekly_rebalance_dates(trading_days)
    if frequency == "monthly":
        first = pd.Series(trading_days, index=trading_days).groupby(trading_days.to_period("M")).min()
        return pd.DatetimeIndex(sorted(first.values))
    raise ValueError(f"Unknown rebalance frequency {frequency!r}; expected one of {list(REBALANCE_FREQUENCIES)}.")


def find_warmup_start(returns_df: pd.DataFrame, window: int = DEFAULT_WINDOW) -> pd.Timestamp:
    """First date whose trailing ``window`` passes every allocator's 90% non-null check.

    Same rule as the backtest/significance scripts' ``_find_warmup_start``.
    """
    rolling_non_null_frac = returns_df.notna().rolling(window).mean()
    window_ok = (rolling_non_null_frac >= 0.90).all(axis=1).to_numpy()
    bad_positions = np.where(~window_ok)[0]
    last_bad_window_end = int(bad_positions[-1]) if len(bad_positions) else window - 2
    return returns_df.index[last_bad_window_end + 2]


def date_bounds(returns_df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The selectable range: warm-up start .. last date in the returns panel."""
    return find_warmup_start(returns_df), pd.Timestamp(returns_df.index[-1])


def validate_date_range(
    start, end, returns_df: pd.DataFrame, min_trading_days: int = MIN_RANGE_TRADING_DAYS
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Check a user range against the data; return it snapped to trading days.

    Raises ``ValueError`` (with a message fit to show the user) if the
    range is reversed, falls outside ``date_bounds``, or holds fewer than
    ``min_trading_days`` trading days. Otherwise returns the first and last
    trading days inside ``[start, end]``.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    lo, hi = date_bounds(returns_df)
    if start > end:
        raise ValueError(f"Start date {start.date()} is after end date {end.date()}.")
    if start < lo:
        raise ValueError(
            f"Start date {start.date()} is before {lo.date()}, the first date with a full "
            f"{DEFAULT_WINDOW}-day history for every asset."
        )
    if end > hi:
        raise ValueError(f"End date {end.date()} is after the last date in the data ({hi.date()}).")
    days = returns_df.index[(returns_df.index >= start) & (returns_df.index <= end)]
    if len(days) < min_trading_days:
        raise ValueError(
            f"The range holds {len(days)} trading days; at least {min_trading_days} are needed "
            "for a meaningful Sharpe ratio."
        )
    return pd.Timestamp(days[0]), pd.Timestamp(days[-1])


def tested_segments(returns_df: pd.DataFrame) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """The four segments in significance_results.csv, snapped to trading days.

    Same boundaries as scripts/apply_significance_tests.py.
    """
    days = returns_df.index
    start, end = date_bounds(returns_df)
    sub_a_end = days[days <= SUB_SPLIT][-1]
    sub_b_start = days[days > SUB_SPLIT][0]
    in_sample_end = days[days <= IN_SAMPLE_END][-1]
    oos_start = days[days > IN_SAMPLE_END][0]
    return {
        "subA_2019-04_to_2021-02": (start, sub_a_end),
        "subB_2021-03_to_2022-12": (sub_b_start, in_sample_end),
        "full_in_sample": (start, in_sample_end),
        OOS_SEGMENT: (oos_start, end),
    }


def match_segment(start, end, segments: dict) -> str | None:
    """Name of the tested segment covering exactly the same trading days, else None.

    ``start``/``end`` must already be snapped to trading days
    (``validate_date_range`` output).
    """
    for name, (seg_start, seg_end) in segments.items():
        if pd.Timestamp(start) == seg_start and pd.Timestamp(end) == seg_end:
            return name
    return None


# ---------------------------------------------------------------------------
# Backtest and metrics
# ---------------------------------------------------------------------------

CODE_DIRECTORIES = ("allocators", "views", "backtest", "tree", "eval")


def code_fingerprint(root: Path = PROJECT_ROOT) -> str:
    """Hash of every source file a backtest result depends on, plus the data.

    The app keys its on-disk backtest cache on this, so editing an
    allocator, a generator, the engine, or the returns data invalidates
    cached results instead of silently showing stale numbers.
    """
    import hashlib

    digest = hashlib.sha256()
    paths = [p for d in CODE_DIRECTORIES for p in sorted((root / d).rglob("*.py"))]
    paths += [Path(__file__).resolve()]
    paths += [root / "data" / "processed" / name for name in ("returns.parquet", "prices.parquet")]
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


@dataclass
class BacktestResult:
    equity: pd.Series  # daily portfolio value
    weights: pd.DataFrame  # held weights per day (post-constraints), dates x tickers


def run_full_backtest(
    key: str,
    frequency: str,
    returns_df: pd.DataFrame,
    trading_days: pd.DatetimeIndex,
    engine_cls=BacktestEngine,
) -> BacktestResult:
    """Backtest one strategy from the warm-up start to the end of the data."""
    start, end = date_bounds(returns_df)
    engine = engine_cls(returns_df)
    equity, states = engine.run(
        start_date=start,
        end_date=end,
        allocator_fn=build_allocator_fn(key, returns_full=returns_df),
        window=DEFAULT_WINDOW,
        rebalance_dates=rebalance_dates(frequency, trading_days),
    )
    weights = pd.DataFrame([s.weights for s in states], index=equity.index)
    return BacktestResult(equity=equity, weights=weights)


def range_returns(equity: pd.Series, start, end) -> pd.Series:
    """Daily returns over ``[start, end]``, sliced exactly as the scripts slice segments."""
    return equity.loc[start:end].pct_change().dropna()


def range_metrics(equity: pd.Series, start, end) -> dict:
    """Sharpe, max drawdown, annualized volatility and total return over the range.

    Sharpe, drawdown and total return are ``eval.metrics``' functions.
    ``eval.metrics`` has no volatility function, so volatility is the
    annualized sample std of daily returns -- the same formula the
    comparison scripts report as ``vol_pct``.
    """
    segment = equity.loc[start:end]
    daily = range_returns(equity, start, end)
    dd = max_drawdown(segment)
    return {
        "sharpe": sharpe_ratio(daily),
        "max_drawdown": dd.max_drawdown,
        "volatility": float(daily.std() * np.sqrt(TRADING_DAYS_PER_YEAR)),
        "total_return": total_return(segment),
    }


def weights_on(weights: pd.DataFrame, date) -> pd.Series:
    """The weights held on the last trading day at or before ``date``."""
    held = weights.loc[:date]
    if held.empty:
        raise ValueError(f"No weights held on or before {pd.Timestamp(date).date()}.")
    return held.iloc[-1].fillna(0.0)


# ---------------------------------------------------------------------------
# Significance: committed CSV lookup, with live bootstrap fallback
# ---------------------------------------------------------------------------

def load_significance_results(path: str | Path = SIGNIFICANCE_CSV_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def lookup_committed_pair(results: pd.DataFrame, segment: str, key_a: str, key_b: str) -> dict | None:
    """The committed pairwise test for (a, b) in ``segment``, oriented as a - b; None if absent."""
    pairwise = results[(results["test"] == "pairwise") & (results["segment"] == segment)]
    row = pairwise[(pairwise["strategy_a"] == key_a) & (pairwise["strategy_b"] == key_b)]
    flipped = False
    if row.empty:
        row = pairwise[(pairwise["strategy_a"] == key_b) & (pairwise["strategy_b"] == key_a)]
        flipped = True
    if row.empty:
        return None
    r = row.iloc[0]
    sign = -1.0 if flipped else 1.0
    return {
        "sharpe_a": float(r["sharpe_b"] if flipped else r["sharpe_a"]),
        "sharpe_b": float(r["sharpe_a"] if flipped else r["sharpe_b"]),
        "sharpe_diff": sign * float(r["sharpe_diff"]),
        "ci95_low": float(-r["ci95_high"] if flipped else r["ci95_low"]),
        "ci95_high": float(-r["ci95_low"] if flipped else r["ci95_high"]),
        "p_value": float(r["p_value"]),
    }


def pairwise_significance(
    strategy_returns: dict[str, pd.Series],
    segment: str | None,
    frequency: str,
    results: pd.DataFrame | None,
    progress: Callable[[], ContextManager] = nullcontext,
    test_fn: Callable = None,
) -> pd.DataFrame:
    """Sharpe-difference tests for every pair of the selected strategies.

    Uses the committed result when the selection is exactly a tested
    configuration -- weekly rebalancing AND ``segment`` is a tested
    segment AND the pair is in the CSV -- and otherwise runs the same
    paired block bootstrap live (``eval.bootstrap.sharpe_difference_test``,
    same resample count and seed as the committed run). ``progress`` is
    entered once around all live tests (the app passes a "computing..."
    spinner); it is not entered at all if every pair was looked up.

    p-values are Holm-adjusted over the pairs shown (``eval.mcs.holm_adjust``),
    since comparing several strategies at once is a multiple comparison.
    """
    test_fn = test_fn or sharpe_difference_test
    keys = list(strategy_returns)
    pairs = [(a, b) for i, a in enumerate(keys) for b in keys[i + 1:]]
    use_committed = frequency == "weekly" and segment is not None and results is not None

    rows, live = [], []
    for a, b in pairs:
        committed = lookup_committed_pair(results, segment, a, b) if use_committed else None
        if committed is not None:
            rows.append({"strategy_a": a, "strategy_b": b, "source": "committed", **committed})
        else:
            rows.append(None)
            live.append((len(rows) - 1, a, b))

    if live:
        with progress():
            for position, a, b in live:
                r = test_fn(strategy_returns[a], strategy_returns[b], n_boot=N_BOOT, alpha=ALPHA, seed=BOOTSTRAP_SEED)
                rows[position] = {
                    "strategy_a": a, "strategy_b": b, "source": "live bootstrap",
                    "sharpe_a": r.sharpe_a, "sharpe_b": r.sharpe_b, "sharpe_diff": r.difference,
                    "ci95_low": r.ci_low, "ci95_high": r.ci_high, "p_value": r.p_value,
                }

    table = pd.DataFrame(
        rows,
        columns=["strategy_a", "strategy_b", "source", "sharpe_a", "sharpe_b", "sharpe_diff",
                 "ci95_low", "ci95_high", "p_value"],
    )
    table["p_holm"] = holm_adjust(table["p_value"]) if len(table) else []
    table["significant"] = table["p_holm"] < ALPHA
    return table


# ---------------------------------------------------------------------------
# The project's honest finding, derived from the committed CSV
# ---------------------------------------------------------------------------

@dataclass
class Findings:
    oos_segment: str
    hrp_significantly_beats: list[dict]  # OOS, Holm (vs-HRP family) p < ALPHA, HRP higher
    vihrp_significantly_beats_hrp: list[dict]  # any segment, Holm p < ALPHA, VI-HRP higher
    vihrp_raw_beats_hrp: list[dict]  # same, uncorrected p < ALPHA
    best_vihrp_vs_hrp: dict | None  # largest VI-HRP Sharpe edge over HRP in any segment


def vs_plain_hrp(results: pd.DataFrame) -> pd.DataFrame:
    """Every committed pairwise test involving plain HRP, oriented as HRP minus the other strategy.

    Works whichever column (``strategy_a`` or ``strategy_b``) plain HRP
    occupies in the CSV.
    """
    pairwise = results[results["test"] == "pairwise"]
    hrp_rows = pairwise[(pairwise["strategy_a"] == PLAIN_HRP) | (pairwise["strategy_b"] == PLAIN_HRP)]
    oriented = []
    for _, r in hrp_rows.iterrows():
        hrp_first = r["strategy_a"] == PLAIN_HRP
        oriented.append({
            "segment": r["segment"],
            "other": r["strategy_b"] if hrp_first else r["strategy_a"],
            "hrp_sharpe": float(r["sharpe_a"] if hrp_first else r["sharpe_b"]),
            "other_sharpe": float(r["sharpe_b"] if hrp_first else r["sharpe_a"]),
            "hrp_minus_other": (1.0 if hrp_first else -1.0) * float(r["sharpe_diff"]),
            "p_value": float(r["p_value"]),
            "p_holm": float(r["p_holm_vs_plain_hrp"]),
        })
    columns = ["segment", "other", "hrp_sharpe", "other_sharpe", "hrp_minus_other", "p_value", "p_holm"]
    return pd.DataFrame(oriented, columns=columns)


def honest_findings(results: pd.DataFrame) -> Findings:
    """Compute the headline finding from the committed significance CSV.

    Nothing here is hardcoded beyond which segment is out-of-sample: if
    the CSV changes, the finding changes with it.
    """
    oriented = vs_plain_hrp(results)

    hrp_wins = oriented[
        (oriented["segment"] == OOS_SEGMENT) & (oriented["hrp_minus_other"] > 0) & (oriented["p_holm"] < ALPHA)
    ].sort_values("p_holm")

    vihrp = oriented[oriented["other"].str.startswith("vihrp_") & (oriented["hrp_minus_other"] < 0)]
    best = None
    if len(vihrp):
        best = vihrp.sort_values("hrp_minus_other").iloc[0].to_dict()

    return Findings(
        oos_segment=OOS_SEGMENT,
        hrp_significantly_beats=hrp_wins.to_dict("records"),
        vihrp_significantly_beats_hrp=vihrp[vihrp["p_holm"] < ALPHA].to_dict("records"),
        vihrp_raw_beats_hrp=vihrp[vihrp["p_value"] < ALPHA].to_dict("records"),
        best_vihrp_vs_hrp=best,
    )


def findings_markdown(findings: Findings) -> str:
    """The headline finding as display text, built only from ``findings``."""
    lines = []
    if findings.hrp_significantly_beats:
        beats = "; ".join(
            f"**{strategy_label(r['other'])}** ({r['hrp_sharpe']:.2f} vs {r['other_sharpe']:.2f}, "
            f"Holm-adjusted p = {r['p_holm']:.3f})"
            for r in findings.hrp_significantly_beats
        )
        lines.append(
            f"**{SEGMENT_LABELS[findings.oos_segment]}: plain HRP has a statistically significantly "
            f"higher Sharpe ratio than** {beats}."
        )
    else:
        lines.append("**Out-of-sample, plain HRP does not significantly beat any other strategy.**")

    if findings.vihrp_significantly_beats_hrp:
        names = ", ".join(sorted({strategy_label(r["other"]) for r in findings.vihrp_significantly_beats_hrp}))
        lines.append(f"**A VI-HRP signal significantly beats plain HRP in at least one segment:** {names}.")
    else:
        detail = ""
        if findings.best_vihrp_vs_hrp is not None:
            b = findings.best_vihrp_vs_hrp
            detail = (
                f" The largest VI-HRP edge in any tested segment was {strategy_label(b['other'])} "
                f"({SEGMENT_LABELS.get(b['segment'], b['segment'])}): +{-b['hrp_minus_other']:.2f} Sharpe, p = {b['p_value']:.2f}."
            )
        raw = "" if findings.vihrp_raw_beats_hrp else " (not even before multiple-comparison correction)"
        lines.append(
            f"**No tested VI-HRP signal beats plain HRP with statistical significance in any segment"
            f"{raw}.**{detail}"
        )
    lines.append(
        "A higher Sharpe ratio in a chart below is a point estimate, not evidence that one "
        "strategy is better; use Compare mode for the significance test."
    )
    return "\n\n".join(lines)
