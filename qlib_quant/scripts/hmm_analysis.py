#!/usr/bin/env python3
"""HMM regime detection - using returns directly with better initialization"""
import pandas as pd
import numpy as np
from scipy import stats
from hmmlearn import hmm
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("验证3: HMM 市场体制检测 (使用收益率直接建模, 3-state)")

df = pd.read_parquet('data/qlib_data/cn_data/factor_data.parquet')
factors = ['book_to_market', 'roa_fina', 'ebit_to_mv', 'ocf_to_ev', 'retained_earnings', 'turnover_rate_f']

# Use median market cap daily log return as market proxy
daily_mv = df.groupby('datetime')['circ_mv'].median().reset_index()
daily_mv.columns = ['datetime', 'median_mv']
daily_mv = daily_mv.dropna(subset=['median_mv']).sort_values('datetime').reset_index(drop=True)
daily_mv['log_mv'] = np.log(daily_mv['median_mv'])
daily_mv['ret_1d'] = daily_mv['log_mv'].diff()
daily_mv = daily_mv.dropna(subset=['ret_1d'])

# Standardize returns for HMM
ret = daily_mv['ret_1d'].values.reshape(-1, 1)
ret_std = (ret - ret.mean()) / ret.std()

print(f"输入数据: {len(ret_std)} 天, 收益率std={ret.std():.6f}")

# Try 3-state HMM with better params
best_bic = np.inf
best_model = None
best_n = 2

for n_states in [2, 3]:
    try:
        model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type='diag',
            n_iter=500,
            random_state=42,
            tol=1e-4,
            init_params='stmc',  # don't init means
        )
        # Initialize means manually
        quantiles = np.quantile(ret_std.flatten(), np.linspace(0.15, 0.85, n_states))
        model.means_ = quantiles.reshape(-1, 1)
        model.startprob_ = np.ones(n_states) / n_states
        model.transmat_ = np.ones((n_states, n_states)) / n_states * 0.8
        np.fill_diagonal(model.transmat_, 0.95)
        model.transmat_ = model.transmat_ / model.transmat_.sum(axis=1, keepdims=True)

        model.fit(ret_std)
        bic = -2 * model.score(ret_std) + n_states * np.log(len(ret_std))
        print(f"  {n_states}-state HMM: log-likelihood={model.score(ret_std):.1f}, BIC={bic:.1f}")

        if bic < best_bic:
            best_bic = bic
            best_model = model
            best_n = n_states
    except Exception as e:
        print(f"  {n_states}-state failed: {e}")

print(f"\n最优模型: {best_n}-state HMM (BIC={best_bic:.1f})")

model = best_model
states = model.predict(ret_std)

state_means = model.means_.flatten()
state_stds = np.sqrt(model.covars_.flatten())

print(f"\n状态参数:")
for i in range(best_n):
    print(f"  State {i}: mean={state_means[i]:.4f}std, std={state_stds[i]:.4f}, 占比={(states==i).sum()/len(states)*100:.1f}%")

# Order states by mean return: bear < neutral < bull
order = np.argsort(state_means)
print(f"\n按收益率排序: {order} (低→高)")

# Group into high/low regime for comparison
# If 3 states: state 0 (lowest) = bear, state 2 (highest) = bull, compare bear vs bull
if best_n == 3:
    bear_state = order[0]
    bull_state = order[2]
    neutral_state = order[1]
    print(f"Bear state: {bear_state}, Neutral: {neutral_state}, Bull: {bull_state}")
else:
    bear_state = order[0]
    bull_state = order[1]

print(f"\n转移概率矩阵:")
for i in range(best_n):
    parts = [f"  State {i} -> {j}: {model.transmat_[i][j]:.4f}" for j in range(best_n)]
    print("  " + " | ".join(parts))

for i in range(best_n):
    stay = 1.0 / (1.0 - model.transmat_[i][i]) if model.transmat_[i][i] < 1 else float('inf')
    print(f"  State {i} 平均驻留: {stay:.1f}天")

# Build regime map
daily_mv['hmm_state'] = states

# For 3-state: compare bear vs bull
daily_mv['regime_compare'] = 'other'
if best_n == 3:
    daily_mv.loc[daily_mv['hmm_state'] == bear_state, 'regime_compare'] = 'bear'
    daily_mv.loc[daily_mv['hmm_state'] == bull_state, 'regime_compare'] = 'bull'
else:
    daily_mv.loc[daily_mv['hmm_state'] == bear_state, 'regime_compare'] = 'bear'
    daily_mv.loc[daily_mv['hmm_state'] == bull_state, 'regime_compare'] = 'bull'

regime_map = dict(zip(daily_mv['datetime'], daily_mv['regime_compare']))

# Compute forward returns
df_sorted = df.sort_values(['instrument', 'datetime']).copy()
df_sorted['fwd_ret_20d'] = df_sorted.groupby('instrument')['circ_mv'].pct_change(-20)
df_ic = df_sorted.dropna(subset=['fwd_ret_20d'])

sample_dates = sorted(df_ic['datetime'].unique())[20:-20:3]
matched = [d for d in sample_dates if d in regime_map]
bear_dates = [d for d in matched if regime_map[d] == 'bear']
bull_dates = [d for d in matched if regime_map[d] == 'bull']
print(f"\n截面日: Bear={len(bear_dates)}, Bull={len(bull_dates)}, Other={len(matched)-len(bear_dates)-len(bull_dates)}")

# Calculate IC by regime
bear_ics = {f: [] for f in factors}
bull_ics = {f: [] for f in factors}

for date in matched:
    regime = regime_map[date]
    if regime == 'other':
        continue
    sub = df_ic[df_ic['datetime'] == date].dropna(subset=factors)
    if len(sub) < 100:
        continue
    for f in factors:
        valid = sub[[f, 'fwd_ret_20d']].dropna()
        if len(valid) < 50:
            continue
        ic, _ = stats.spearmanr(valid[f], valid['fwd_ret_20d'])
        if np.isnan(ic):
            continue
        if regime == 'bear':
            bear_ics[f].append(ic)
        else:
            bull_ics[f].append(ic)

print(f"\n{'Factor':<22} {'Bear_IC':>9} {'Bear_IR':>9} {'Bull_IC':>9} {'Bull_IR':>9} | {'DeltaIC':>8} {'t':>8} {'p':>8}")
print("-" * 100)

results = []
for f in factors:
    b_ic = bear_ics[f]
    l_ic = bull_ics[f]

    b_mean = np.mean(b_ic) if b_ic else 0
    b_std = np.std(b_ic) if b_ic else 1
    b_ir = b_mean / b_std if b_std > 0 else 0

    l_mean = np.mean(l_ic) if l_ic else 0
    l_std = np.std(l_ic) if l_ic else 1
    l_ir = l_mean / l_std if l_std > 0 else 0

    delta = l_mean - b_mean

    if len(b_ic) > 2 and len(l_ic) > 2:
        t_stat, p_val = stats.ttest_ind(b_ic, l_ic)
    else:
        t_stat, p_val = 0, 1

    print(f"{f:<22} {b_mean:>+9.4f} {b_ir:>+9.3f} {l_mean:>+9.4f} {l_ir:>+9.3f} | {delta:>+8.4f} {t_stat:>+8.3f} {p_val:>8.4f}")
    results.append({
        'factor': f, 'bear_ir': b_ir, 'bull_ir': l_ir,
        'delta': delta, 't': t_stat, 'p': p_val,
        'bear_n': len(b_ic), 'bull_n': len(l_ic)
    })

print("\n结论:")
sig = [r for r in results if r['p'] < 0.05]
print(f"Bear/Bull体制下IC差异显著(p<0.05): {len(sig)} / {len(factors)}")
for r in sig:
    print(f"  {r['factor']}: DeltaIC={r['delta']:+.4f}, t={r['t']:+.3f}, p={r['p']:.4f}")

# Year breakdown
print("\nHMM体制年度分布:")
for y in sorted(daily_mv['datetime'].dt.year.unique()):
    mask = daily_mv['datetime'].dt.year == y
    y_data = daily_mv[mask]
    if len(y_data) == 0:
        continue
    parts = []
    for i in range(best_n):
        pct = (y_data['hmm_state'] == i).sum() / len(y_data) * 100
        label = ['LowRet', 'MidRet', 'HighRet'][order[i]] if best_n == 3 else ['LowRet', 'HighRet'][order[i]]
        parts.append(f"{label}={pct:.0f}%")
    print(f"  {y}: {', '.join(parts)}, N={len(y_data)}")

print("\n验证3完成")
