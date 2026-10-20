"""Streamlit front end for running and comparing the project's allocation strategies.

Run from the project root:

    streamlit run interface/app.py

Rendering only. Every decision (which allocator/signal to call, date
validation, backtest wiring, metrics, significance lookup vs. live
bootstrap, the headline finding) lives in ``interface/core.py``, which
only calls the project's existing, tested functions.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from backtest.calendar import load_trading_days  # noqa: E402
from backtest.engine import load_default_returns  # noqa: E402
from interface import core  # noqa: E402

# ---------------------------------------------------------------------------
# Colors: one fixed color per strategy (never by rank or selection order),
# from the validated 8-hue categorical palette; plain HRP, the baseline in
# every view, is a neutral dashed line instead of a ninth hue.
# ---------------------------------------------------------------------------

PALETTE = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
}
BASELINE_INK = {"light": "#52514e", "dark": "#c3c2b7"}


def _theme() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:
        return "light"


def strategy_colors() -> dict[str, str]:
    theme = _theme()
    others = [k for k in core.ALL_STRATEGIES if k != core.PLAIN_HRP]
    colors = dict(zip(others, PALETTE[theme]))
    colors[core.PLAIN_HRP] = BASELINE_INK[theme]
    return colors


# ---------------------------------------------------------------------------
# Data and cached backtests
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_data():
    return load_default_returns(), load_trading_days()


@st.cache_data(show_spinner=False)
def fingerprint() -> str:
    return core.code_fingerprint()


@st.cache_data(persist="disk", show_spinner=False)
def cached_backtest(key: str, frequency: str, code_version: str):
    """One full-history backtest per (strategy, frequency, code version)."""
    returns_df, trading_days = load_data()
    result = core.run_full_backtest(key, frequency, returns_df, trading_days)
    return result.equity, result.weights


@st.cache_data(show_spinner=False)
def significance_results():
    return core.load_significance_results()


def run_backtests(keys: list[str], frequency: str) -> dict:
    """Backtests for ``keys``, with visible progress for any not yet cached."""
    results = {}
    progress = st.progress(0.0, text="Preparing backtests...")
    for i, key in enumerate(keys):
        progress.progress(
            i / len(keys),
            text=f"Backtesting {core.strategy_label(key)} ({i + 1}/{len(keys)}). "
            "About a minute each the first time; cached afterwards.",
        )
        results[key] = cached_backtest(key, frequency, fingerprint())
    progress.empty()
    return results


# ---------------------------------------------------------------------------
# Charts and tables
# ---------------------------------------------------------------------------

def equity_chart(curves: dict[str, pd.Series], start, end) -> alt.Chart:
    """Growth of 1 over the range, one line per strategy, crosshair tooltip."""
    colors = strategy_colors()
    frames = []
    for key, equity in curves.items():
        segment = equity.loc[start:end]
        frames.append(pd.DataFrame({
            "date": segment.index, "value": (segment / segment.iloc[0]).values,
            "strategy": core.strategy_label(key), "key": key,
        }))
    data = pd.concat(frames, ignore_index=True)
    labels = [core.strategy_label(k) for k in curves]
    color = alt.Color(
        "strategy:N",
        scale=alt.Scale(domain=labels, range=[colors[k] for k in curves]),
        legend=alt.Legend(title=None, orient="top"),
    )
    dash = alt.StrokeDash(
        "strategy:N",
        scale=alt.Scale(domain=labels, range=[[6, 4] if k == core.PLAIN_HRP else [1, 0] for k in curves]),
        legend=None,
    )
    base = alt.Chart(data).encode(x=alt.X("date:T", title=None))
    lines = base.mark_line(strokeWidth=2).encode(
        y=alt.Y("value:Q", title="Growth of 1", scale=alt.Scale(zero=False)), color=color, strokeDash=dash
    )

    hover = alt.selection_point(fields=["date"], nearest=True, on="pointerover", empty=False)
    wide = data.pivot_table(index="date", columns="strategy", values="value").reset_index()
    tooltip = [alt.Tooltip("date:T", title="Date")] + [
        alt.Tooltip(field=label, type="quantitative", title=label, format=".3f") for label in labels
    ]
    rule = (
        alt.Chart(wide).mark_rule(strokeWidth=1, opacity=0.5)
        .encode(x="date:T", opacity=alt.condition(hover, alt.value(0.5), alt.value(0)), tooltip=tooltip)
        .add_params(hover)
    )
    points = base.mark_point(size=60, filled=True).encode(
        y="value:Q", color=color, opacity=alt.condition(hover, alt.value(1), alt.value(0))
    ).transform_filter(hover)

    layers = [lines, rule, points]
    if len(curves) <= 4:  # direct-label line ends when there are few enough not to collide
        last = data.sort_values("date").groupby("strategy").tail(1)
        layers.append(
            alt.Chart(last).mark_text(align="left", dx=6, fontSize=12)
            .encode(x="date:T", y="value:Q", text="strategy:N")
        )
    return alt.layer(*layers).properties(height=380)


def weights_chart(weights: pd.Series, key: str) -> alt.Chart:
    data = pd.DataFrame({"ticker": weights.index, "weight": weights.values}).sort_values("weight", ascending=False)
    return (
        alt.Chart(data)
        .mark_bar(color=strategy_colors()[key], cornerRadiusEnd=4)
        .encode(
            y=alt.Y("ticker:N", sort=None, title=None),
            x=alt.X("weight:Q", title="Portfolio weight", axis=alt.Axis(format="%")),
            tooltip=[alt.Tooltip("ticker:N", title="Asset"), alt.Tooltip("weight:Q", title="Weight", format=".1%")],
        )
        .properties(height=28 * len(data))
    )


def metrics_table(curves: dict[str, pd.Series], start, end) -> pd.DataFrame:
    rows = []
    for key, equity in curves.items():
        m = core.range_metrics(equity, start, end)
        rows.append({
            "Strategy": core.strategy_label(key),
            "Sharpe": round(m["sharpe"], 2),
            "Total return": f"{m['total_return']:.1%}",
            "Volatility (ann.)": f"{m['volatility']:.1%}",
            "Max drawdown": f"{m['max_drawdown']:.1%}",
        })
    return pd.DataFrame(rows).set_index("Strategy")


def significance_section(curves: dict, start, end, frequency: str, segment: str | None) -> None:
    returns = {key: core.range_returns(equity, start, end) for key, equity in curves.items()}
    table = core.pairwise_significance(
        returns, segment, frequency, significance_results(),
        progress=lambda: st.spinner(
            "Computing block-bootstrap significance tests (5,000 resamples per pair)..."
        ),
    )
    if segment is not None and frequency == "weekly":
        st.caption(
            f"Your range is exactly the tested segment **{core.SEGMENT_LABELS[segment]}** with weekly "
            "rebalancing, so these are the committed results from scripts/output/significance_results.csv."
        )
    else:
        reason = "monthly rebalancing was never tested" if frequency != "weekly" else (
            "your range does not exactly match one of the four tested segments"
        )
        st.caption(
            f"Computed live because {reason}: a paired circular block bootstrap of the Sharpe "
            "difference (eval/bootstrap.py), the same procedure used for the committed results."
        )

    display = pd.DataFrame({
        "Strategy A": table["strategy_a"].map(core.strategy_label),
        "Strategy B": table["strategy_b"].map(core.strategy_label),
        "Sharpe A - B": table["sharpe_diff"].round(2),
        "95% CI": [f"[{lo:.2f}, {hi:.2f}]" for lo, hi in zip(table["ci95_low"], table["ci95_high"])],
        "p-value": table["p_value"].round(3),
        "p (Holm)": table["p_holm"].round(3),
        "Significant at 5%?": table["significant"].map({True: "Yes", False: "No"}),
        "Source": table["source"],
    })
    st.dataframe(display, hide_index=True, width="stretch")
    n_sig = int(table["significant"].sum())
    if n_sig == 0:
        st.markdown(
            "**None of these Sharpe differences is statistically significant** - over this range the "
            "data cannot tell these strategies apart, whatever the point estimates above say."
        )
    else:
        st.markdown(
            f"**{n_sig} of {len(table)} differences {'is' if n_sig == 1 else 'are'} significant** after "
            "Holm correction across the pairs shown."
        )
    if len(table) > 1:
        st.caption("p (Holm) is adjusted for testing several pairs at once; judge significance by it.")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Strategy Explorer", layout="wide")
st.title("Portfolio strategy explorer")
st.caption(
    "Backtest and compare the project's allocation strategies on 10 NSE equities, a gold ETF and a "
    "gilt ETF, with transaction costs and long-only constraints (max 25% per stock, min 5% per ETF)."
)

findings = core.honest_findings(significance_results())
st.info(core.findings_markdown(findings), icon=":material/fact_check:")

with st.expander("What we learned: every strategy vs. plain HRP, all tested segments"):
    table = core.vs_plain_hrp(significance_results())
    st.dataframe(
        pd.DataFrame({
            "Segment": table["segment"].map(core.SEGMENT_LABELS),
            "Strategy": table["other"].map(core.strategy_label),
            "Plain HRP Sharpe": table["hrp_sharpe"],
            "Strategy Sharpe": table["other_sharpe"],
            "HRP - strategy": table["hrp_minus_other"],
            "p-value": table["p_value"],
            "p (Holm)": table["p_holm"],
            "Significant at 5%?": (table["p_holm"] < core.ALPHA).map({True: "Yes", False: "No"}),
        }).round(3),
        hide_index=True, width="stretch",
    )
    st.caption(
        "Source: scripts/output/significance_results.csv (paired block bootstrap, Holm-corrected over "
        "the 8 comparisons with plain HRP per segment). In-sample p-values for the VI-HRP signals are "
        "optimistic: their parameters were tuned on that same window."
    )

returns_df, trading_days = load_data()
lo, hi = core.date_bounds(returns_df)
segments = core.tested_segments(returns_df)

with st.sidebar:
    mode = st.radio("Mode", ["Single strategy", "Compare strategies"])

    preset_options = ["Custom"] + list(segments)

    def _apply_preset():
        choice = st.session_state["preset"]
        if choice != "Custom":
            s, e = segments[choice]
            st.session_state["date_range"] = (s.date(), e.date())

    st.selectbox(
        "Tested period", preset_options, key="preset", on_change=_apply_preset,
        format_func=lambda k: k if k == "Custom" else core.SEGMENT_LABELS[k],
        help="Pick one of the four periods the significance tests were run on, or set your own range.",
    )
    if "date_range" not in st.session_state:
        st.session_state["date_range"] = (lo.date(), hi.date())
    date_range = st.date_input(
        "Date range", min_value=lo.date(), max_value=hi.date(),
        key="date_range",
        help=f"Data allows {lo.date()} to {hi.date()} (the start is the first date with a full year of "
        "history for every asset).",
    )
    frequency = st.selectbox(
        "Rebalance frequency", list(core.REBALANCE_FREQUENCIES),
        format_func=core.REBALANCE_FREQUENCIES.get,
        help="All committed results used weekly rebalancing.",
    )

if not isinstance(date_range, (tuple, list)) or len(date_range) != 2:
    st.info("Pick an end date to finish the date range.")
    st.stop()
try:
    start, end = core.validate_date_range(date_range[0], date_range[1], returns_df)
except ValueError as exc:
    st.error(str(exc))
    st.stop()
segment = core.match_segment(start, end, segments)

if mode == "Single strategy":
    with st.sidebar:
        allocator = st.selectbox("Allocator", list(core.ALLOCATOR_LABELS), format_func=core.ALLOCATOR_LABELS.get)
        signal = None
        if allocator == core.VIHRP:
            signal = st.selectbox(
                "View signal", list(core.SIGNALS), format_func=lambda s: core.SIGNALS[s].label,
                help="The signal VI-HRP uses to tilt plain HRP's splits.",
            )
    key = core.strategy_key(allocator, signal)
    keys = [key] if key == core.PLAIN_HRP else [key, core.PLAIN_HRP]
    backtests = run_backtests(keys, frequency)
    curves = {k: equity for k, (equity, _w) in backtests.items()}

    st.subheader(core.strategy_label(key))
    st.caption(f"{start.date()} to {end.date()}, {core.REBALANCE_FREQUENCIES[frequency].lower()} rebalancing")
    m = core.range_metrics(curves[key], start, end)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sharpe ratio", f"{m['sharpe']:.2f}")
    c2.metric("Total return", f"{m['total_return']:.1%}")
    c3.metric("Volatility (annualized)", f"{m['volatility']:.1%}")
    c4.metric("Max drawdown", f"{m['max_drawdown']:.1%}")

    left, right = st.columns([2, 1])
    with left:
        st.markdown("**Equity curve vs. plain HRP**" if key != core.PLAIN_HRP else "**Equity curve**")
        st.altair_chart(equity_chart(curves, start, end), width="stretch")
    with right:
        weights = core.weights_on(backtests[key][1], end)
        st.markdown(f"**Weights held on {end.date()}**")
        st.altair_chart(weights_chart(weights, key), width="stretch")
        with st.expander("Weights as a table"):
            st.dataframe(weights.sort_values(ascending=False).map("{:.1%}".format).rename("Weight"))

    if key != core.PLAIN_HRP:
        st.markdown("**Against plain HRP**")
        st.dataframe(metrics_table(curves, start, end), width="stretch")
        significance_section(curves, start, end, frequency, segment)

else:
    with st.sidebar:
        selected = st.multiselect(
            "Strategies", core.ALL_STRATEGIES,
            default=[core.PLAIN_HRP, "equal_weight", "vihrp_volatility_regime"],
            format_func=core.strategy_label,
        )
    if len(selected) < 2:
        st.info("Select at least two strategies to compare.")
        st.stop()
    backtests = run_backtests(selected, frequency)
    curves = {k: equity for k, (equity, _w) in backtests.items()}

    st.subheader(f"Comparing {len(selected)} strategies")
    st.caption(f"{start.date()} to {end.date()}, {core.REBALANCE_FREQUENCIES[frequency].lower()} rebalancing")
    st.altair_chart(equity_chart(curves, start, end), width="stretch")
    st.markdown("**Metrics**")
    st.dataframe(metrics_table(curves, start, end), width="stretch")
    st.markdown("**Is the Sharpe difference statistically significant?**")
    significance_section(curves, start, end, frequency, segment)
    with st.expander(f"Weights held on {end.date()}"):
        table = pd.DataFrame({core.strategy_label(k): core.weights_on(w, end) for k, (_e, w) in backtests.items()})
        st.dataframe(table.map("{:.1%}".format), width="stretch")
