"""Block-bootstrap significance + exposure analysis for the RL overlay vs baseline HRP.
Uses the trained checkpoints as-is (no retraining) and eval/bootstrap.py with the
same settings as scripts/apply_significance_tests.py."""
import sys; sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
from collections import Counter
import numpy as np, pandas as pd
import run_rl_overlay_comparison as m
from backtest.calendar import DEFAULT_PRICES_PATH, weekly_rebalance_dates
from backtest.costs import ETF_TICKERS, transaction_cost
from eval.bootstrap import (circular_block_indices, default_block_size, equity_to_returns,
                            sharpe_difference_test)
N_BOOT, SEED, ALPHA = 5000, 20261020, 0.05
ck50, ck150, out = sys.argv[1], sys.argv[2], sys.argv[3]

prices = pd.read_parquet(DEFAULT_PRICES_PATH); r = m.load_default_returns()
td = pd.DatetimeIndex(sorted(prices.index.unique())); rb = set(pd.DatetimeIndex(weekly_rebalance_dates(td)))
days = td[td >= pd.Timestamp('2019-04-05')]
bw = m.precompute_base_hrp_weights(r, days, rb)
ov = m.RiskOverlay()

class Fixed:
    def __init__(s, a): s.a = a
    def act(s, o, evaluate=True): return s.a
def load(p):
    a = m.RLRiskAgent(); a.load(p); return a
policies = {'RL_150ep': load(ck150), 'RL_50ep': load(ck50),
            'always_MILD': Fixed(1), 'always_STRONG': Fixed(2)}

def mdd_from_returns(x):  # x: (..., n) daily returns -> max drawdown along last axis
    eq = np.cumprod(1 + x, axis=-1); return (eq / np.maximum.accumulate(eq, axis=-1) - 1).min(axis=-1)
def cost_series(states):
    ws = [pd.Series(s.weights) for s in states]
    return np.array([transaction_cost(ws[i-1], ws[i]) for i in range(1, len(ws))])
def eq_frac(w): w = pd.Series(w); return float(w[~w.index.isin(ETF_TICKERS)].sum())

segments = {'OOS 2023-01..2026-09': days[days >= '2023-01-02'], 'FULL 2019-04..2026-09': days,
            'IN-SAMPLE 2019-04..2022-12': days[days < '2023-01-02']}
rows, expo_rows = [], []
for seg, d in segments.items():
    base_eq, base_st, _ = m.run_overlaid_backtest(r, bw, None, ov, d)
    base_ret = equity_to_returns(base_eq); base_cost = cost_series(base_st)
    for name, ag in policies.items():
        eq, st, acts = m.run_overlaid_backtest(r, bw, ag, ov, d)
        ret = equity_to_returns(eq)
        t = sharpe_difference_test(ret, base_ret, n_boot=N_BOOT, alpha=ALPHA, seed=SEED)
        # supplementary: paired block bootstrap of max-drawdown difference (same index scheme)
        data = pd.concat([ret, base_ret], axis=1, join='inner').dropna().to_numpy()
        idx = circular_block_indices(len(data), default_block_size(len(data)), N_BOOT, np.random.default_rng(SEED))
        dd_obs = mdd_from_returns(data[:, 0]) - mdd_from_returns(data[:, 1])
        dd_boot = mdd_from_returns(data[idx, 0]) - mdd_from_returns(data[idx, 1])
        dd_p = float(np.mean(np.abs(dd_boot - dd_obs) >= abs(dd_obs)))
        cost = cost_series(st); yrs = len(ret) / 252
        c = Counter(acts); n = len(acts)
        rows.append(dict(segment=seg, policy=name, sharpe=t.sharpe_a, sharpe_base=t.sharpe_b,
            sharpe_diff=t.difference, ci_low=t.ci_low, ci_high=t.ci_high, p_value=t.p_value,
            significant=t.significant, mdd_diff_pp=dd_obs*100,
            mdd_diff_ci=f"[{np.quantile(dd_boot,.025)*100:.2f}, {np.quantile(dd_boot,.975)*100:.2f}]",
            mdd_p=dd_p, cost_drag_bps_yr=(cost.sum()-base_cost.sum())/yrs*1e4,
            ann_ret_diff_bps=(ret.mean()-base_ret.mean())*252*1e4,
            hold=c[0]/n, mild=c[1]/n, strong=c[2]/n, block=t.block_size, n_obs=t.n_obs))
        if name == 'RL_150ep' and seg.startswith('FULL'):
            for s, a, dt in zip(st, acts, d):
                if dt in rb:
                    bwr = bw.loc[dt]
                    expo_rows.append(dict(date=dt, base_equity=eq_frac(bwr), base_gilt=bwr['LTGILTBEES'],
                        base_gold=bwr['GOLDBEES'], rl_equity=eq_frac(s.weights), action=a))
res = pd.DataFrame(rows); res.to_csv(f"{out}/rl_overlay_significance.csv", index=False)
ex = pd.DataFrame(expo_rows).set_index('date'); ex.to_csv(f"{out}/rl_overlay_exposure_by_rebalance.csv")
pd.set_option('display.width', 250)
print(res[['segment','policy','sharpe','sharpe_base','sharpe_diff','ci_low','ci_high','p_value','significant']].round(4).to_string(index=False))
print(); print(res[['segment','policy','mdd_diff_pp','mdd_diff_ci','mdd_p','cost_drag_bps_yr','ann_ret_diff_bps','hold','mild','strong','block','n_obs']].round(3).to_string(index=False))
ex['moved_pp'] = (ex.base_equity - ex.rl_equity) * 100
ex['year'] = ex.index.year
print(); print(ex.groupby('year').agg(rebalances=('action','size'), base_equity_mean=('base_equity','mean'),
    base_equity_min=('base_equity','min'), base_equity_max=('base_equity','max'), gilt_mean=('base_gilt','mean'),
    gold_mean=('base_gold','mean'), rl_equity_mean=('rl_equity','mean'), moved_pp_mean=('moved_pp','mean'),
    moved_pp_max=('moved_pp','max'), pct_hold=('action', lambda a: (a==0).mean()),
    pct_mild=('action', lambda a: (a==1).mean()), pct_strong=('action', lambda a: (a==2).mean())).round(3).to_string())
print('\nshare of rebalances with base equity >= 20%:', round((ex.base_equity >= .20).mean(), 3),
      ' (OOS only:', round((ex.loc['2023-01-02':].base_equity >= .20).mean(), 3), ')')
