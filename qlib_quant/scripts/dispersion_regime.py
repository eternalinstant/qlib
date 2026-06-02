#!/usr/bin/env python3
"""HMM-based regime detection using cross-sectional return dispersion (more robust signal)"""
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("验证3 (替代方案): 基于截面收益离散度的体制检测")
print("方法: 用每日截面收益率的标准差作为市场体制指标，分3组")

df = pd.read_parquet('data/qlib_data/cn_data/factor_data.parquet')
factors = ['book_to_market', 'roa_fina', 'ebit_to_mv', 'ocf_to_ev', 'retained_earnings', 'turnover_rate_f']

# Compute daily cross-sectional return dispersion
# Use forward 20d return proxy: (next_day_mv / today_mv) - 1 per stock
df_sorted = df.sort_values(['instrument', 'datetime']).copy()
df_sorted['ret_1d'] = df_sorted.groupby('instrument')['circ_mv'].pct_change()

# Cross-sectional dispersion of daily returns
dispersion = df_sorted.groupby('datetime')['ret_1d'].agg(['std', 'mean']).reset_index()
dispersion.columns = ['datetime', 'dispersion', 'mkt_ret']
dispersion = dispersion.dropna()

# Use 60-day rolling average of dispersion for smoother regime signal
dispersion['disp_60d'] = dispersion['dispersion'].rolling(60).mean()
dispersion = dispersion.dropna()

print(f"离散度数据: {len(dispersion)} 天")
print(f"离散度范围: {dispersion['disp_60d'].min():.4f} ~ {dispersion['disp_60d'].max():.4f}")
print(f"离散度中位数: {dispersion['disp_60d'].median():.4f}")

# Split into 3 regimes by dispersion terciles
q33 = dispersion['disp_60d'].quantile(0.33)
q67 = dispersion['disp_60d'].quantile(0.67)
print(f"Tercile thresholds: Q33={q33:.4f}, Q67={q67:.4f}")

dispersion['regime'] = np.where(
    dispersion['disp_60d'] > q67, 'high_disp',
    np.where(dispersion['disp_60d'] > q33, 'mid_disp', 'low_disp')
)

regime_map = dict(zip(dispersion['datetime'], dispersion['regime']))
regime_counts = dispersion['regime'].value_counts()
print(f"体制分布: {dict(regime_counts)}")

# Compute forward 20d returns
df_sorted2 = df.sort_values(['instrument', 'datetime']).copy()
df_sorted2['fwd_ret_20d'] = df_sorted2.groupby('instrument')['circ_mv'].pct_change(-20)
df_ic = df_sorted2.dropna(subset=['fwd_ret_20d'])

sample_dates = sorted(df_ic['datetime'].unique())[20:-20:3]
matched = [d for d in sample_dates if d in regime_map]
print(f"截面日: {len(matched)}")

# Calculate IC by regime
regime_ics = {r: {f: [] for f in factors} for r in ['low_disp', 'mid_disp', 'high_disp']}

for date in matched:
    regime = regime_map[date]
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
        regime_ics[regime][f].append(ic)

print(f"\n{'Factor':<22} {'LowDisp_IC':>10} {'LowDisp_IR':>10} {'MidDisp_IC':>10} {'MidDisp_IR':>10} {'HiDisp_IC':>10} {'HiDisp_IR':>10}")
print("-" * 115)

results = []
for f in factors:
    row = f"{f:<22}"
    for r in ['low_disp', 'mid_disp', 'high_disp']:
        ics = regime_ics[r][f]
        m = np.mean(ics) if ics else 0
        s = np.std(ics) if ics else 1
        ir = m / s if s > 0 else 0
        row += f" {m:>+10.4f} {ir:>+10.3f}"
    print(row)

# Statistical tests: high vs low dispersion
print(f"\n{'Factor':<22} {'Hi-Low ΔIC':>10} {'t_stat':>8} {'p_val':>8} {'结论':>10}")
print("-" * 65)

for f in factors:
    hi = regime_ics['high_disp'][f]
    lo = regime_ics['low_disp'][f]
    delta = np.mean(hi) - np.mean(lo) if hi and lo else 0
    if len(hi) > 2 and len(lo) > 2:
        t, p = stats.ttest_ind(hi, lo)
    else:
        t, p = 0, 1
    conclusion = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else ""))
    print(f"{f:<22} {delta:>+10.4f} {t:>+8.3f} {p:>8.4f} {conclusion:>10}")
    results.append({'factor': f, 'delta': delta, 't': t, 'p': p})

print("\n关键发现:")
sig = [r for r in results if r['p'] < 0.05]
print(f"高/低离散度体制下IC差异显著(p<0.05): {len(sig)} / {len(factors)}")
for r in sig:
    print(f"  {r['factor']}: Hi-Low ΔIC={r['delta']:+.4f}, t={r['t']:+.3f}, p={r['p']:.4f}")

# Additional: correlation between dispersion and factor IC time series
print("\n离散度与因子IC的时间序列相关性:")
for f in factors:
    # Build time series of (dispersion, IC) pairs
    pairs = []
    for date in matched:
        regime = regime_map[date]
        sub = df_ic[df_ic['datetime'] == date].dropna(subset=factors)
        if len(sub) < 100:
            continue
        valid = sub[[f, 'fwd_ret_20d']].dropna()
        if len(valid) < 50:
            continue
        ic, _ = stats.spearmanr(valid[f], valid['fwd_ret_20d'])
        if np.isnan(ic):
            continue
        d = dispersion[dispersion['datetime'] == date]['disp_60d']
        if len(d) > 0:
            pairs.append((d.values[0], ic))

    if len(pairs) > 30:
        d_arr = [p[0] for p in pairs]
        ic_arr = [p[1] for p in pairs]
        corr, p_corr = stats.spearmanr(d_arr, ic_arr)
        print(f"  {f:<22}: rho={corr:+.4f}, p={p_corr:.4f} {'***' if p_corr < 0.01 else ('**' if p_corr < 0.05 else ('*' if p_corr < 0.1 else ''))}")

# Year-by-year regime distribution
print("\n体制年度分布:")
for y in sorted(dispersion['datetime'].dt.year.unique()):
    mask = dispersion['datetime'].dt.year == y
    y_data = dispersion[mask]
    if len(y_data) == 0:
        continue
    parts = []
    for r in ['low_disp', 'mid_disp', 'high_disp']:
        pct = (y_data['regime'] == r).sum() / len(y_data) * 100
        parts.append(f"{r[:3]}={pct:.0f}%")
    print(f"  {y}: {', '.join(parts)}, N={len(y_data)}")

print("\n验证3替代方案完成")
