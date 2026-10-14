# Universe structure diagnosis

Why has every tested VI-HRP signal (momentum, mean-reversion, standardized
momentum+reversal) lost to plain HRP on this 12-asset universe (10 NSE
equities + GOLDBEES + LTGILTBEES), even after calibration? Three
hypotheses, tested against `data/processed/returns.parquet` and the real
backtest engine via `scripts/diagnose_universe_structure.py`. Full data:
`universe_correlation_matrix.csv`, `universe_correlation_summary.csv`,
`universe_tree_structure.csv`, `universe_variance_decomposition.csv`.

## 2a. Pairwise correlation

Mean equity-equity correlation (full sample): **0.277** (range across
segments: 0.254-0.365). This is *moderate*, not high — nowhere near the
task's own suggested ">0.5" bar for "too intercorrelated to have
independent variation." GOLDBEES and LTGILTBEES are both essentially
uncorrelated with the equities (mean ≈ -0.01 to +0.11) and with each
other (-0.015 to +0.127 across segments) — genuine, distinct
diversifiers, not quasi-equities.

Taken at face value, raw correlation does **not** support hypothesis
(b) as stated. But see the variance-decomposition section below: a
*different*, more direct test of "is there exploitable structure among
these 10 equities" tells a less reassuring story than the correlation
number alone.

## 2b. Tree structure

The actual `ClusterTree` (identical construction to `allocators.hrp`/
`vihrp`) has the **exact same shape in every segment tested** — full
sample, subA, subB, full in-sample, and OOS all agree:

```
ROOT (node 22): GOLDBEES (1)  vs  everything else (11)
  node 21:      LTGILTBEES (1)  vs  the 10 equities
    node 20:    {INFY, TCS} (2)  vs  the other 8 equities
      node 19:  SUNPHARMA (1)  vs  7 remaining equities
        node 18: BHARTIARTL (1)  vs  6 remaining
          node 17: ITC (1)  vs  5 remaining
            node 16: RELIANCE (1)  vs  4 remaining
              node 15: MARUTI (1)  vs  3 remaining
                node 14: TATASTEEL (1)  vs  2 remaining
                  node 13: HDFCBANK (1)  vs  LT (1)
    node 12:    INFY (1)  vs  TCS (1)
```

Two findings, both structural and both robust across every regime
tested (not an artifact of one time window):

1. **The two asset-class boundaries sit at the very top of the tree,
   every single time.** The root split is always GOLDBEES vs. everyone
   else; the very next split is always LTGILTBEES vs. the 10 equities.
   Both asset-class splits resolve *before any equity-internal
   structure at all* — of the 11 total splits, exactly 2 are
   asset-class splits and they are the two highest-level ones (nodes 22
   and 21), carrying the most multiplicative portfolio mass of any
   nodes in the tree.
2. **Every equity-vs-equity split is a 1-vs-many "peel" structure**, not
   a balanced binary split: {INFY,TCS} vs. 8 others, then one equity at
   a time peeled off a shrinking remainder (SUNPHARMA vs. 7, BHARTIARTL
   vs. 6, ITC vs. 5, ...). Half of every such split (the single peeled
   asset) carries an undiluted signal; the other half (the shrinking
   "remaining basket," up to 8 equities at once) averages whatever
   signal those equities have, diluting any one of them that isn't the
   asset about to be peeled off next.

## 2c/2d. Variance/Sharpe decomposition

Four-rung ladder, backtested with the real engine (weekly rebalance,
real transaction costs): **EW12** (equal-weight, no class awareness) →
**NAIVE3WAY** (1/3-1/3-1/3 across equity/gold/gilt, equal-weight within
equities) → **HRP_TOP_NAIVE_EQ** (plain HRP's own real class-level mass,
equal-weight within equities) → **PLAIN_HRP** (HRP's real weights,
unmodified).

| Segment | EW12 Sharpe (vol%) | NAIVE3WAY Sharpe (vol%) | HRP_TOP_NAIVE_EQ Sharpe (vol%) | PLAIN_HRP Sharpe (vol%) |
|---|---|---|---|---|
| subA (2019-04→2021-02) | 0.752 (21.05%) | 0.008 (26.92%) | 0.909 (22.32%) | 0.774 (22.15%) |
| subB (2021-03→2022-12) | 1.410 (13.16%) | 1.625 (6.82%) | 1.280 (6.17%) | 1.293 (5.95%) |
| Full in-sample | 0.986 (17.60%) | 0.284 (19.76%) | 0.864 (16.47%) | 0.768 (16.32%) |
| OOS (2023-01→2026-09) | 0.958 (10.96%) | 1.752 (8.55%) | 2.557 (3.43%) | 2.601 (3.38%) |

**The HRP_TOP_NAIVE_EQ → PLAIN_HRP step (equity-internal risk-parity
weighting, holding the real class-level ratio fixed) is tiny in
*absolute* terms in every single segment**: a 0.17, 0.22, 0.15, and
0.05 percentage-point reduction in annualized volatility, respectively.
By contrast, the NAIVE3WAY → HRP_TOP_NAIVE_EQ step (getting the
class-level *ratio* right, via data, instead of an arbitrary 1/3-1/3-1/3)
is large in OOS (8.55% → 3.43% vol, more than half the total gap to
PLAIN_HRP) and in full in-sample (19.76% → 16.47%). This is the cleanest,
most direct confirmation of hypothesis (a): **the asset-class-level
ratio decision, not the equity-internal weighting, carries essentially
all of HRP's structural edge, and that's most true in exactly the
regime (OOS) where every tested signal failed worst.**

(Fractional variance-reduction attributions are reported in
`universe_variance_decomposition.csv`, but several segments have a
small or sign-flipping EW12→PLAIN_HRP denominator, which blows the
*fractions* up to nonsensical values -- e.g. full in-sample shows
"-184%" / "273%" / "12%". The *absolute* vol-point deltas above are the
reliable number; the fractions should not be read as precise
percentages in every segment, only as a directional signal where the
denominator is well-behaved, like OOS.)

**subA's anomaly, explained, not a bug:** NAIVE3WAY is dramatically
*worse* than even doing nothing (EW12) in subA -- Sharpe 0.008, vol
26.92% (higher than EW12's 21.05%, despite having far less equity
exposure). Checking per-asset volatility in that window explains it
directly: **GOLDBEES had 74.6% annualized volatility in subA** -- nearly
double the most volatile equity (TATASTEEL, 45.8%), almost certainly the
well-documented March-2020 gold-ETF NAV/premium dislocation seen
globally during the COVID liquidity crisis. An arbitrary, non-adaptive
33% allocation to GOLDBEES during that specific window was not a
safe-haven hedge; it was the single riskiest position in the portfolio.
This is itself evidence *for* hypothesis (a): the value of HRP's
class-level ratio isn't just "diversify across classes" in the abstract,
it's "size each class's exposure by its *actual, currently realized*
risk" -- something a fixed naive ratio cannot do, and something a
return-based (not volatility-based) signal, as currently designed, also
does not directly target.

## Verdict on the three hypotheses

**(a) Plain HRP's edge comes almost entirely from cross-asset-class
diversification that any tilt necessarily sacrifices some of.**
**Strongly supported.** The asset-class boundary sits at the top of the
tree in every regime tested, carries the most portfolio-mass-weighted
influence of any split, and the variance decomposition shows the
equity-internal layer contributing a near-zero absolute share of HRP's
total edge in every single segment (0.05-0.22 vol points), with the
class-level ratio (sized by real, time-varying volatility) carrying the
rest. This is the dominant, best-supported explanation.

**(b) The 10-equity sub-universe is too internally correlated to have
exploitable cross-sectional dispersion.**
**Not supported by raw correlation (0.28-0.37 is moderate, not high),
but partially supported by a more direct test.** HRP's OWN
correlation/volatility-optimized weighting *within* the equity block
barely improves on naive equal-weight within that same block, in every
regime (the same tiny 0.05-0.22 vol-point deltas above) -- including in
regimes (OOS) where equity-equity correlation is at its *lowest*
measured level (0.254), which should be the most favorable case for
exploitable dispersion if correlation were the limiting factor. That
HRP's own optimizer still finds little to work with there, independent
of the correlation level, suggests the limiting factor isn't pairwise
correlation specifically -- it's that these 10 names (all large-cap,
liquid, broadly similar risk profiles) don't have much *exploitable risk
heterogeneity* for a weighting scheme to act on, which is a related but
distinct claim from "they're too correlated." Call this hypothesis
*redirected, not confirmed*: correlation isn't the bottleneck, but
homogeneity in a different sense (risk profile similarity among
large-cap Indian equities) plausibly is.

**(c) VI-HRP's cluster-level tilting washes out asset-specific signal by
averaging with uninformative assets in the same node.**
**Structurally real, but secondary.** Every equity-vs-equity split is a
1-vs-many "peel," so half of every such split genuinely does average up
to 8 equities together, diluting whichever of them isn't the one about
to be peeled off. This is a real, demonstrated structural property, not
a hypothetical one. But its practical impact is likely secondary to (a)
and (b): even a perfectly undiluted equity-internal signal would be
acting on a layer that contributes almost no variance reduction to
begin with (per the decomposition above), so dilution compounds an
already-low-ceiling problem rather than being the primary cause of the
three generators' failures.

## Implications for where a future signal should operate

1. **The highest-value target is the asset-class-level split, not the
   equity-internal layer** -- but the value there comes from *sizing by
   realized volatility/correlation*, which is exactly what HRP's
   inverse-variance mechanism already does well. A trailing-RETURN
   momentum/reversal signal (what all three prior generators used, even
   at the asset-class nodes they do walk through) is a weaker tool for
   that specific job than a volatility/regime-aware signal would be --
   subA's GOLDBEES vol spike is a concrete example of a risk-based
   signal (rising realized vol) that would have been informative exactly
   when a return-based signal gave no particular warning. A future
   signal aimed at adding value here should target volatility/regime
   shifts in the asset-class aggregates, not trailing returns.
2. **The equity-internal layer has a low ceiling with this specific
   10-name universe, independent of signal quality.** Any future
   signal operating only within the equity block (as all three tested
   signals effectively did, in aggregate) is fighting for a tiny slice
   of total achievable variance reduction (0.05-0.22 vol points across
   every regime tested) -- even a perfect signal there has little room
   to move the needle on *portfolio-level* risk-adjusted return.
3. **Widening the equity sub-universe is a reasonable, complementary
   next step**, not a replacement for (1): more names, more sectors, or
   a wider risk-profile spread (mid-caps, different sector mixes) could
   increase the equity-internal layer's own exploitable dispersion
   (addressing the homogeneity concern under (b)), which would raise the
   ceiling for any future equity-internal signal -- but per the
   decomposition above, that ceiling only matters much once the
   asset-class-level allocation is already being done well, since that
   layer dominates total portfolio risk by a wide margin in this data.
