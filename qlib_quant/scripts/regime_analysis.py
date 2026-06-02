#!/usr/bin/env python3
"""
第9轮研究验证：波动率体制因子IC分析 + 回撤因子择时 + HMM体制检测
基于论文:
1. Luo, Wang, Jussa (2025) - Dynamic allocation: extremes, tail dependence, and regime Shifts
2. Park (2023) - PCA and HMM for Forecasting Stock Returns
3. Choi (2014) - Maximum drawdown, recovery, and momentum
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load data
print("=" * 80)
print("加载因子数据...")
df = pd.read_parquet('data/qlib_data/cn_data/factor_data.parquet')
print(f"数据量: {df.shape[0]:,} 行 × {df.shape[1]} 列")
print(f"日期范围: {df['datetime'].min()} ~ {df['datetime'].max()}")
print(f"股票数: {df['instrument'].nunique()}")
print(f"交易日数: {df['datetime'].nunique()}")

# Key factors
factors = ['book_to_market', 'roa_fina', 'ebit_to_mv', 'ocf_to_ev', 'retained_earnings', 'turnover_rate_f']

# ==========================================
# 验证1: 波动率体制对因子IC的影响
# ==========================================
print("\n" + "=" * 80)
print("验证1: 波动率体制(高/低波动)对因子IC的影响")
print("方法: 用60日滚动volatility的中位数划分高/低波动体制")

# 用中位数市值变化率的波动率作为市场状态指标
daily_mv = df.groupby('datetime')['circ_mv'].median().reset_index()
daily_mv.columns = ['datetime', 'median_mv']
daily_mv['log_mv'] = np.log(daily_mv['median_mv'].replace(0, np.nan).dropna())
daily_mv = daily_mv.dropna()
daily_mv['ret_20d'] = daily_mv['log_mv'].diff(20)
daily_mv['vol_60d'] = daily_mv['ret_20d'].rolling(60).std()
vol_median = daily_mv['vol_60d'].median()
daily_mv['regime'] = np.where(daily_mv['vol_60d'] > vol_median, 'high_vol', 'low_vol')

print(f"波动率中位数: {vol_median:.6f}")
print(f"高波动体制天数: {(daily_mv['regime']=='high_vol').sum()}")
print(f"低波动体制天数: {(daily_mv['regime']=='low_vol').sum()}")

# 计算20日前瞻收益率
df_sorted = df.sort_values(['instrument', 'datetime']).copy()
df_sorted['fwd_ret_20d'] = df_sorted.groupby('instrument')['circ_mv'].pct_change(-20)
df_ic = df_sorted.dropna(subset=['fwd_ret_20d'])

# 每个日期、每个因子计算Spearman IC
regime_ics = {'high_vol': {f: [] for f in factors}, 'low_vol': {f: [] for f in factors}}
all_ics = {f: [] for f in factors}
regime_map = dict(zip(daily_mv['datetime'], daily_mv['regime']))

sample_dates = sorted(df_ic['datetime'].unique())[20:-20:3]
print(f"\n计算IC (使用 {len(sample_dates)} 个截面日)...")

for date in sample_dates:
    if date not in regime_map:
        continue
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
        all_ics[f].append((date, ic))
        if regime in regime_ics:
            regime_ics[regime][f].append(ic)

print("\n结果:")
print(f"{'因子':<22} {'全期IC':>8} {'全期IR':>7} | {'高波IC':>8} {'高波IR':>7} {'低波IC':>8} {'低波IR':>7} | {'ΔIC':>7} {'t':>8} {'p':>8}")
print("-" * 115)

results_regime = []
for f in factors:
    all_ic_vals = [x[1] for x in all_ics[f]]
    hv_ic = regime_ics['high_vol'][f]
    lv_ic = regime_ics['low_vol'][f]

    all_mean = np.mean(all_ic_vals) if all_ic_vals else 0
    all_std = np.std(all_ic_vals) if all_ic_vals else 1
    all_ir = all_mean / all_std if all_std > 0 else 0

    hv_mean = np.mean(hv_ic) if hv_ic else 0
    hv_std = np.std(hv_ic) if hv_ic else 1
    hv_ir = hv_mean / hv_std if hv_std > 0 else 0

    lv_mean = np.mean(lv_ic) if lv_ic else 0
    lv_std = np.std(lv_ic) if lv_ic else 1
    lv_ir = lv_mean / lv_std if lv_std > 0 else 0

    delta_ic = lv_mean - hv_mean

    if len(hv_ic) > 2 and len(lv_ic) > 2:
        t_stat, p_val = stats.ttest_ind(hv_ic, lv_ic)
    else:
        t_stat, p_val = 0, 1

    print(f"{f:<22} {all_mean:>+8.4f} {all_ir:>+7.3f} | {hv_mean:>+8.4f} {hv_ir:>+7.3f} {lv_mean:>+8.4f} {lv_ir:>+7.3f} | {delta_ic:>+7.4f} {t_stat:>+8.3f} {p_val:>8.4f}")
    results_regime.append({
        'factor': f, 'all_ir': all_ir, 'hv_ir': hv_ir, 'lv_ir': lv_ir,
        'delta_ic': delta_ic, 't_stat': t_stat, 'p_val': p_val,
        'hv_n': len(hv_ic), 'lv_n': len(lv_ic)
    })

print("\n结论:")
sig_factors = [r for r in results_regime if r['p_val'] < 0.05]
print(f"高/低波动体制下IC差异显著的因子(p<0.05): {len(sig_factors)} / {len(factors)}")
for r in sig_factors:
    print(f"  {r['factor']}: ΔIC={r['delta_ic']:+.4f}, t={r['t_stat']:+.3f}, p={r['p_val']:.4f}")

# ==========================================
# 验证2: 回撤因子择时 — 回撤后因子IC是否更强
# ==========================================
print("\n" + "=" * 80)
print("验证2: 回撤因子择时 (基于Choi 2014 最大回撤理论)")
print("方法: 定义市场处于回撤状态(当前价格 < 60日最高价 × 0.95)，分组计算因子IC")

# 用中位市值作为市场指数代理
market = daily_mv.set_index('datetime')['median_mv']
market_60d_max = market.rolling(60).max()
drawdown = (market - market_60d_max) / market_60d_max
daily_mv2 = daily_mv.copy()
daily_mv2['drawdown'] = drawdown.values

# 定义回撤体制: drawdown < -5% 为回撤期
dd_threshold = -0.05
daily_mv2['dd_regime'] = np.where(daily_mv2['drawdown'] < dd_threshold, 'drawdown', 'normal')

print(f"回撤阈值: {dd_threshold:.0%}")
print(f"回撤期天数: {(daily_mv2['dd_regime']=='drawdown').sum()}")
print(f"正常期天数: {(daily_mv2['dd_regime']=='normal').sum()}")
print(f"平均回撤幅度: {daily_mv2['drawdown'].mean():.4f}")
print(f"最大回撤幅度: {daily_mv2['drawdown'].min():.4f}")

dd_regime_map = dict(zip(daily_mv2['datetime'], daily_mv2['dd_regime']))
dd_ics = {'drawdown': {f: [] for f in factors}, 'normal': {f: [] for f in factors}}

for date in sample_dates:
    if date not in dd_regime_map:
        continue
    regime = dd_regime_map[date]
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
        if regime in dd_ics:
            dd_ics[regime][f].append(ic)

print(f"\n{'因子':<22} {'回撤期IC':>9} {'回撤期IR':>9} {'正常期IC':>9} {'正常期IR':>9} | {'ΔIC':>7} {'t':>8} {'p':>8}")
print("-" * 100)

results_dd = []
for f in factors:
    dd_ic = dd_ics['drawdown'][f]
    nm_ic = dd_ics['normal'][f]

    dd_mean = np.mean(dd_ic) if dd_ic else 0
    dd_std = np.std(dd_ic) if dd_ic else 1
    dd_ir = dd_mean / dd_std if dd_std > 0 else 0

    nm_mean = np.mean(nm_ic) if nm_ic else 0
    nm_std = np.std(nm_ic) if nm_ic else 1
    nm_ir = nm_mean / nm_std if nm_std > 0 else 0

    delta_ic = dd_mean - nm_mean

    if len(dd_ic) > 2 and len(nm_ic) > 2:
        t_stat, p_val = stats.ttest_ind(dd_ic, nm_ic)
    else:
        t_stat, p_val = 0, 1

    print(f"{f:<22} {dd_mean:>+9.4f} {dd_ir:>+9.3f} {nm_mean:>+9.4f} {nm_ir:>+9.3f} | {delta_ic:>+7.4f} {t_stat:>+8.3f} {p_val:>8.4f}")
    results_dd.append({
        'factor': f, 'dd_ir': dd_ir, 'nm_ir': nm_ir,
        'delta_ic': delta_ic, 't_stat': t_stat, 'p_val': p_val,
        'dd_n': len(dd_ic), 'nm_n': len(nm_ic)
    })

print("\n结论:")
sig_dd = [r for r in results_dd if r['p_val'] < 0.05]
print(f"回撤/正常期IC差异显著的因子(p<0.05): {len(sig_dd)} / {len(factors)}")
for r in sig_dd:
    print(f"  {r['factor']}: ΔIC={r['delta_ic']:+.4f}, t={r['t_stat']:+.3f}, p={r['p_val']:.4f}")

# ==========================================
# 验证3: 2-state Gaussian HMM 市场体制检测
# ==========================================
print("\n" + "=" * 80)
print("验证3: 2-state Gaussian HMM 市场体制检测 (基于Park 2023)")
print("方法: 对中位市值日收益率拟合2-state HMM，分析不同体制下因子IC")

try:
    from hmmlearn import hmm

    # 准备收益率序列
    ret_series = daily_mv2['ret_20d'].dropna().values.reshape(-1, 1)
    ret_series = ret_series[~np.isnan(ret_series).flatten()]

    # Fit 2-state Gaussian HMM
    model = hmm.GaussianHMM(n_components=2, covariance_type='full', n_iter=200, random_state=42)
    model.fit(ret_series)
    states = model.predict(ret_series)

    # 确定哪个state是高波动
    state_means = model.means_.flatten()
    state_vars = [np.cov(ret_series[states == i].flatten()) for i in range(2)]
    high_vol_state = np.argmax(state_vars)
    low_vol_state = 1 - high_vol_state

    print(f"HMM State 0: mean={state_means[0]:.6f}, var={state_vars[0][0][0]:.8f}")
    print(f"HMM State 1: mean={state_means[1]:.6f}, var={state_vars[1][0][0]:.8f}")
    print(f"高波动体制: State {high_vol_state}, 低波动体制: State {low_vol_state}")

    # Transition matrix
    print(f"\n转移概率矩阵:")
    print(f"  State 0 → State 0: {model.transmat_[0][0]:.4f}, State 0 → State 1: {model.transmat_[0][1]:.4f}")
    print(f"  State 1 → State 0: {model.transmat_[1][0]:.4f}, State 1 → State 1: {model.transmat_[1][1]:.4f}")

    # 平均驻留时间
    avg_stay_0 = 1.0 / (1.0 - model.transmat_[0][0]) if model.transmat_[0][0] < 1 else float('inf')
    avg_stay_1 = 1.0 / (1.0 - model.transmat_[1][1]) if model.transmat_[1][1] < 1 else float('inf')
    print(f"平均驻留时间: State 0={avg_stay_0:.1f}天, State 1={avg_stay_1:.1f}天")

    # State proportions
    state_0_pct = (states == 0).sum() / len(states) * 100
    state_1_pct = (states == 1).sum() / len(states) * 100
    print(f"状态占比: State 0={state_0_pct:.1f}%, State 1={state_1_pct:.1f}%")

    # Align HMM states with dates
    valid_dates = daily_mv2.dropna(subset=['ret_20d'])['datetime'].values
    hmm_regime_map = {}
    for i, d in enumerate(valid_dates):
        if i < len(states):
            hmm_regime_map[d] = 'hmm_high' if states[i] == high_vol_state else 'hmm_low'

    # Calculate IC by HMM regime
    hmm_ics = {'hmm_high': {f: [] for f in factors}, 'hmm_low': {f: [] for f in factors}}

    for date in sample_dates:
        if date not in hmm_regime_map:
            continue
        regime = hmm_regime_map[date]
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
            if regime in hmm_ics:
                hmm_ics[regime][f].append(ic)

    print(f"\n{'因子':<22} {'HMM高波IC':>10} {'HMM高波IR':>10} {'HMM低波IC':>10} {'HMM低波IR':>10} | {'ΔIC':>7} {'t':>8} {'p':>8}")
    print("-" * 108)

    results_hmm = []
    for f in factors:
        hv_ic = hmm_ics['hmm_high'][f]
        lv_ic = hmm_ics['hmm_low'][f]

        hv_mean = np.mean(hv_ic) if hv_ic else 0
        hv_std = np.std(hv_ic) if hv_ic else 1
        hv_ir = hv_mean / hv_std if hv_std > 0 else 0

        lv_mean = np.mean(lv_ic) if lv_ic else 0
        lv_std = np.std(lv_ic) if lv_ic else 1
        lv_ir = lv_mean / lv_std if lv_std > 0 else 0

        delta_ic = lv_mean - hv_mean

        if len(hv_ic) > 2 and len(lv_ic) > 2:
            t_stat, p_val = stats.ttest_ind(hv_ic, lv_ic)
        else:
            t_stat, p_val = 0, 1

        print(f"{f:<22} {hv_mean:>+10.4f} {hv_ir:>+10.3f} {lv_mean:>+10.4f} {lv_ir:>+10.3f} | {delta_ic:>+7.4f} {t_stat:>+8.3f} {p_val:>8.4f}")
        results_hmm.append({
            'factor': f, 'hv_ir': hv_ir, 'lv_ir': lv_ir,
            'delta_ic': delta_ic, 't_stat': t_stat, 'p_val': p_val,
            'hv_n': len(hv_ic), 'lv_n': len(lv_ic)
        })

    print("\n结论:")
    sig_hmm = [r for r in results_hmm if r['p_val'] < 0.05]
    print(f"HMM体制下IC差异显著的因子(p<0.05): {len(sig_hmm)} / {len(factors)}")
    for r in sig_hmm:
        print(f"  {r['factor']}: ΔIC={r['delta_ic']:+.4f}, t={r['t_stat']:+.3f}, p={r['p_val']:.4f}")

except ImportError:
    print("hmmlearn 未安装，跳过HMM验证")
    print("安装命令: pip install hmmlearn")

# ==========================================
# 验证4: 因子IC的滚动稳定性与衰减分析
# ==========================================
print("\n" + "=" * 80)
print("验证4: 因子IC的年度衰减分析 (基于Lee 2025 Alpha Decay理论)")
print("方法: 按年度分组计算因子IC/IR，检测是否存在alpha衰减趋势")

# Get year from dates
for f in factors:
    ic_by_year = {}
    for date, ic in all_ics[f]:
        year = date.year
        if year not in ic_by_year:
            ic_by_year[year] = []
        ic_by_year[year].append(ic)

    year_stats = {}
    for year in sorted(ic_by_year.keys()):
        ics = ic_by_year[year]
        year_stats[year] = {
            'mean': np.mean(ics),
            'std': np.std(ics),
            'ir': np.mean(ics) / np.std(ics) if np.std(ics) > 0 else 0,
            'n': len(ics),
            'pct_positive': sum(1 for x in ics if x > 0) / len(ics) * 100
        }

    years = sorted(year_stats.keys())
    irs = [year_stats[y]['ir'] for y in years]

    if len(years) >= 3:
        slope, intercept, r_val, p_val, std_err = stats.linregress(range(len(years)), irs)
        trend = "衰减↓" if slope < -0.01 else ("增强↑" if slope > 0.01 else "稳定→")
    else:
        slope, r_val, p_val = 0, 0, 1
        trend = "N/A"

    print(f"\n{f} — IR趋势: {trend} (slope={slope:+.4f}, R²={r_val**2:.3f}, p={p_val:.4f})")
    print(f"  {'年份':>6} {'IC均值':>8} {'IC_std':>8} {'IR':>8} {'N':>5} {'正IC%':>7}")
    for y in years:
        s = year_stats[y]
        print(f"  {y:>6} {s['mean']:>+8.4f} {s['std']:>8.4f} {s['ir']:>+8.3f} {s['n']:>5} {s['pct_positive']:>6.1f}%")

print("\n" + "=" * 80)
print("全部验证完成")
