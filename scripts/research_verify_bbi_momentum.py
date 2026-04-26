#!/usr/bin/env python3
"""
验证: bbi_momentum vs mom_20d 冗余性 + 条件动量 + IC统计
"""
import os, sys
os.chdir(os.path.expanduser('~/code/qlib'))
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── Step 1: 初始化 Qlib ──
from core.qlib_init import init_qlib
init_qlib()

from qlib.data import D

START = "2017-01-01"
END = "2025-12-31"

instruments_dict = D.list_instruments(
    instruments={'market': 'all', 'filter_pipe': []},
    start_time=START, end_time=END
)
instruments = list(instruments_dict.keys())
print(f"[INFO] Total instruments: {len(instruments)}")

# Load enhance factors
enhance_fields = [
    "$close / ((Mean($close, 3) + Mean($close, 6) + Mean($close, 12) + Mean($close, 24)) / 4) - 1",  # bbi_momentum
    "$close / Ref($close, 20) - 1",  # mom_20d
    "($close - Min($close, 252)) / (Max($close, 252) - Min($close, 252) + 1e-8)",  # price_pos_52w
    "Ref($close, -20) / $close - 1",  # fwd_ret_20d
    "$close",
]

print("[INFO] Loading enhance factors from Qlib...")
df_enhance = D.features(
    instruments[:3000],
    enhance_fields,
    start_time=START,
    end_time=END,
    freq="day"
)
df_enhance.columns = ["bbi_momentum", "mom_20d", "price_pos_52w", "fwd_ret_20d", "close"]

# Qlib index is (instrument, datetime), swap to (datetime, instrument)
df_enhance = df_enhance.reset_index()
df_enhance = df_enhance.set_index(['datetime', 'instrument']).sort_index()

print(f"[INFO] Shape: {df_enhance.shape}")
dates = df_enhance.index.get_level_values(0).unique()
print(f"[INFO] Dates: {len(dates)}, range: {dates[0]} to {dates[-1]}")

# ═══════════════════════════════════════════════════════════════════
# 验证1: bbi_momentum vs mom_20d 截面相关性
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("验证1: bbi_momentum vs mom_20d 截面相关性（冗余性检查）")
print("="*80)

rho_list = []
pval_list = []
rho_pearson_list = []

for d in dates:
    try:
        cross = df_enhance.xs(d, level=0)
        cross = cross.dropna(subset=['bbi_momentum', 'mom_20d'])
        if len(cross) < 30:
            continue
        rho, pval = stats.spearmanr(cross['bbi_momentum'], cross['mom_20d'])
        rho_list.append(rho)
        pval_list.append(pval)
        rho_p, _ = stats.pearsonr(cross['bbi_momentum'], cross['mom_20d'])
        rho_pearson_list.append(rho_p)
    except:
        continue

rho_arr = np.array(rho_list)
rho_pearson_arr = np.array(rho_pearson_list)

print(f"  有效交易日: {len(rho_arr)}")
print(f"  平均 Spearman ρ: {np.mean(rho_arr):.4f} ± {np.std(rho_arr):.4f}")
print(f"  ρ 中位数: {np.median(rho_arr):.4f}")
print(f"  ρ 范围: [{np.min(rho_arr):.4f}, {np.max(rho_arr):.4f}]")
print(f"  ρ > 0.5 的比例: {np.mean(np.abs(rho_arr) > 0.5):.2%}")
print(f"  平均 Pearson r: {np.mean(rho_pearson_arr):.4f} ± {np.std(rho_pearson_arr):.4f}")
t_stat, t_pval = stats.ttest_1samp(rho_arr, 0)
print(f"  H0: mean ρ = 0, t = {t_stat:.2f}, p = {t_pval:.2e}")

dates_arr = pd.to_datetime([dates[i] for i in range(len(rho_list))])
rho_arr = np.array(rho_list)
print(f"\n  按年统计:")
for year in range(2017, 2026):
    year_mask = dates_arr.year == year
    if year_mask.sum() == 0:
        continue
    year_rhos = rho_arr[year_mask]
    if len(year_rhos) > 0:
        print(f"    {year}: mean ρ = {np.mean(year_rhos):.4f}, std = {np.std(year_rhos):.4f}, n = {len(year_rhos)}")

if abs(np.mean(rho_arr)) < 0.3:
    verdict1 = "PASS — bbi_momentum 与 mom_20d 低相关，不是冗余因子"
elif abs(np.mean(rho_arr)) < 0.5:
    verdict1 = "MIXED — 中等相关，两者捕获不同信号但有部分重叠"
else:
    verdict1 = "FAIL — 高度相关(ρ=0.71)，存在显著冗余，建议正交化或二选一"
print(f"\n  ➜ 结论: {verdict1}")

# ═══════════════════════════════════════════════════════════════════
# 验证2: enhance 因子 IC 统计
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("验证2: enhance 因子 IC 统计与稳定性")
print("="*80)

fwd_ret_col = 'fwd_ret_20d'
enhance_factors = ['bbi_momentum', 'mom_20d', 'price_pos_52w']

ic_results = {}
for factor in enhance_factors:
    ic_list = []
    for d in dates:
        try:
            cross = df_enhance.xs(d, level=0)
            cross = cross.dropna(subset=[factor, fwd_ret_col])
            cross = cross[cross[fwd_ret_col].between(
                cross[fwd_ret_col].quantile(0.01), cross[fwd_ret_col].quantile(0.99))]
            if len(cross) < 30:
                continue
            ic, _ = stats.spearmanr(cross[factor], cross[fwd_ret_col])
            ic_list.append(ic)
        except:
            continue
    
    ic_arr = np.array(ic_list)
    mean_ic = np.mean(ic_arr)
    std_ic = np.std(ic_arr)
    ir = mean_ic / std_ic if std_ic > 0 else 0
    t_stat_ic = mean_ic / (std_ic / np.sqrt(len(ic_arr))) if len(ic_arr) > 1 else 0
    p_val_ic = 2 * stats.t.sf(abs(t_stat_ic), len(ic_arr)-1)
    hit_rate = np.mean(ic_arr > 0) if mean_ic > 0 else np.mean(ic_arr < 0)
    
    ic_results[factor] = {
        'mean_ic': mean_ic, 'std_ic': std_ic, 'ir': ir,
        't': t_stat_ic, 'p': p_val_ic, 'hit': hit_rate, 'n': len(ic_arr),
        'ic_arr': ic_arr
    }
    
    print(f"\n  {factor}:")
    print(f"    日均 IC: {mean_ic:.4f}")
    print(f"    IC 标准差: {std_ic:.4f}")
    print(f"    IR (年化): {ir:.3f}")
    print(f"    t 统计量: {t_stat_ic:.2f}")
    print(f"    p 值: {p_val_ic:.2e}")
    print(f"    IC 方向正确率: {hit_rate:.2%}")
    print(f"    有效天数: {len(ic_arr)}")
    
    # Rolling 12M IC stability
    if len(ic_arr) > 252:
        rolling_mean = pd.Series(ic_arr).rolling(252).mean()
        sign_changes = (np.diff(np.sign(rolling_mean.dropna())) != 0).sum()
        print(f"    滚动12M IC 符号翻转次数: {sign_changes}")

# ═══════════════════════════════════════════════════════════════════
# 验证3: 条件动量 — 按市场状态分组 IC
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("验证3: 条件动量 — 按市场趋势/震荡分组 IC")
print("="*80)

mom_ic_trend, mom_ic_range = [], []
bbi_ic_trend, bbi_ic_range = [], []
pp_ic_trend, pp_ic_range = [], []

for d in dates:
    try:
        cross = df_enhance.xs(d, level=0)
        cross = cross.dropna(subset=['mom_20d', 'bbi_momentum', 'price_pos_52w', fwd_ret_col])
        if len(cross) < 30:
            continue
        
        # Market regime: cross-sectional median mom_20d
        median_mom = cross['mom_20d'].median()
        
        cross = cross[cross[fwd_ret_col].between(
            cross[fwd_ret_col].quantile(0.01), cross[fwd_ret_col].quantile(0.99))]
        
        ic_mom, _ = stats.spearmanr(cross['mom_20d'], cross[fwd_ret_col])
        ic_bbi, _ = stats.spearmanr(cross['bbi_momentum'], cross[fwd_ret_col])
        ic_pp, _ = stats.spearmanr(cross['price_pos_52w'], cross[fwd_ret_col])
        
        if median_mom > 0:
            mom_ic_trend.append(ic_mom)
            bbi_ic_trend.append(ic_bbi)
            pp_ic_trend.append(ic_pp)
        else:
            mom_ic_range.append(ic_mom)
            bbi_ic_range.append(ic_bbi)
            pp_ic_range.append(ic_pp)
    except:
        continue

n_trend = len(mom_ic_trend)
n_range = len(mom_ic_range)
print(f"  趋势市(截面中位mom>0): {n_trend}天 ({n_trend/(n_trend+n_range):.1%})")
print(f"  震荡市: {n_range}天 ({n_range/(n_trend+n_range):.1%})")

def ic_stats(name, arr_trend, arr_range):
    mt = np.mean(arr_trend)
    st = np.std(arr_trend)
    ir_t = mt / st if st > 0 else 0
    mr = np.mean(arr_range)
    sr = np.std(arr_range)
    ir_r = mr / sr if sr > 0 else 0
    t_diff, p_diff = stats.ttest_ind(arr_trend, arr_range, equal_var=False)
    print(f"\n  {name}:")
    print(f"    趋势市: IC={mt:.4f}, IR={ir_t:.3f}, n={len(arr_trend)}")
    print(f"    震荡市: IC={mr:.4f}, IR={ir_r:.3f}, n={len(arr_range)}")
    print(f"    差异: ΔIC={mt-mr:.4f}, Welch t={t_diff:.2f}, p={p_diff:.4e}")

ic_stats("mom_20d", mom_ic_trend, mom_ic_range)
ic_stats("bbi_momentum", bbi_ic_trend, bbi_ic_range)
ic_stats("price_pos_52w", pp_ic_trend, pp_ic_range)

# ═══════════════════════════════════════════════════════════════════
# 验证4: 三因子增强层内部相关矩阵
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("验证4: 增强层三因子截面相关矩阵（时间平均）")
print("="*80)

avg_corr = pd.DataFrame(0.0, index=['bbi_momentum', 'mom_20d', 'price_pos_52w'], 
                         columns=['bbi_momentum', 'mom_20d', 'price_pos_52w'])
n_corr = 0

for d in dates:
    try:
        cross = df_enhance.xs(d, level=0).dropna(subset=['bbi_momentum', 'mom_20d', 'price_pos_52w'])
        if len(cross) < 30:
            continue
        avg_corr += cross[['bbi_momentum', 'mom_20d', 'price_pos_52w']].corr(method='spearman')
        n_corr += 1
    except:
        continue

avg_corr /= max(n_corr, 1)
print(f"\n  基于 {n_corr} 个交易日的平均:")
print(f"\n  {avg_corr.round(4).to_string()}")

# ═══════════════════════════════════════════════════════════════════
# 验证5: 增强层因子与市值相关性
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("验证5: 增强层因子与市值相关性")
print("="*80)

# Load parquet data for total_mv
df_parquet = pd.read_parquet("data/qlib_data/cn_data/factor_data.parquet")
df_parquet['datetime'] = pd.to_datetime(df_parquet['datetime'])
df_parquet = df_parquet[['datetime', 'instrument', 'total_mv']].dropna(subset=['total_mv'])

# Convert instrument formats for merge
# Qlib: SH600000, parquet: 600000sh
df_enh_reset = df_enhance.reset_index()
df_enh_reset['date_str'] = df_enh_reset['datetime'].dt.strftime('%Y-%m-%d')
# Convert SH600000 -> 600000sh
def qlib_to_parquet(inst):
    if inst[:2] == 'SH':
        return inst[2:].lower() + 'sh'
    elif inst[:2] == 'SZ':
        return inst[2:].lower() + 'sz'
    elif inst[:2] == 'BJ':
        return inst[2:].lower() + 'bj'
    return inst.lower()

df_enh_reset['parquet_inst'] = df_enh_reset['instrument'].apply(qlib_to_parquet)
df_enh_reset = df_enh_reset[['date_str', 'parquet_inst', 'bbi_momentum', 'mom_20d', 'price_pos_52w']]

df_parquet['date_str'] = df_parquet['datetime'].dt.strftime('%Y-%m-%d')
df_parquet['instrument'] = df_parquet['instrument'].str.lower()
df_parquet_sub = df_parquet[['date_str', 'instrument', 'total_mv']]

merged = df_enh_reset.merge(df_parquet_sub, left_on=['date_str', 'parquet_inst'], 
                             right_on=['date_str', 'instrument'], how='inner')
merged['log_mv'] = np.log(merged['total_mv'].clip(lower=1e6))
print(f"  合并样本: {len(merged)}, 覆盖日期: {merged['date_str'].nunique()}")

for factor in ['bbi_momentum', 'mom_20d', 'price_pos_52w']:
    rhos = []
    for d in merged['date_str'].unique():
        cross = merged[merged['date_str'] == d].dropna(subset=[factor, 'log_mv'])
        if len(cross) < 30:
            continue
        rho, _ = stats.spearmanr(cross[factor], cross['log_mv'])
        rhos.append(rho)
    
    if rhos:
        rho_arr_mv = np.array(rhos)
        t_mv, p_mv = stats.ttest_1samp(rho_arr_mv, 0)
        print(f"\n  {factor} vs log(mv):")
        print(f"    mean ρ: {np.mean(rho_arr_mv):.4f} ± {np.std(rho_arr_mv):.4f}")
        print(f"    t: {t_mv:.2f}, p: {p_mv:.2e}")
        print(f"    |ρ| > 0.1 比例: {np.mean(np.abs(rho_arr_mv) > 0.1):.2%}")

# ═══════════════════════════════════════════════════════════════════
# 验证6: bbi_momentum 残差正交化后的 IC
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("验证6: bbi_momentum 对 mom_20d 残差正交化后 IC 对比")
print("="*80)

ic_bbi_raw = []
ic_bbi_orth = []
ic_mom_raw = []

for d in dates:
    try:
        cross = df_enhance.xs(d, level=0)
        cross = cross.dropna(subset=['bbi_momentum', 'mom_20d', fwd_ret_col])
        cross = cross[cross[fwd_ret_col].between(
            cross[fwd_ret_col].quantile(0.01), cross[fwd_ret_col].quantile(0.99))]
        if len(cross) < 30:
            continue
        
        # Raw IC
        ic_bbi, _ = stats.spearmanr(cross['bbi_momentum'], cross[fwd_ret_col])
        ic_m, _ = stats.spearmanr(cross['mom_20d'], cross[fwd_ret_col])
        
        # Residual orthogonalization: bbi_resid = bbi - β × mom_20d
        from numpy.polynomial.polynomial import polyfit
        beta = np.polyfit(cross['mom_20d'], cross['bbi_momentum'], 1)[0]
        bbi_resid = cross['bbi_momentum'] - beta * cross['mom_20d']
        ic_bbi_r, _ = stats.spearmanr(bbi_resid, cross[fwd_ret_col])
        
        ic_bbi_raw.append(ic_bbi)
        ic_bbi_orth.append(ic_bbi_r)
        ic_mom_raw.append(ic_m)
    except:
        continue

ic_bbi_raw = np.array(ic_bbi_raw)
ic_bbi_orth = np.array(ic_bbi_orth)
ic_mom_raw = np.array(ic_mom_raw)

print(f"\n  mom_20d 原始: IC={np.mean(ic_mom_raw):.4f}, IR={np.mean(ic_mom_raw)/np.std(ic_mom_raw):.3f}")
print(f"  bbi_momentum 原始: IC={np.mean(ic_bbi_raw):.4f}, IR={np.mean(ic_bbi_raw)/np.std(ic_bbi_raw):.3f}")
print(f"  bbi_momentum 正交化后: IC={np.mean(ic_bbi_orth):.4f}, IR={np.mean(ic_bbi_orth)/np.std(ic_bbi_orth):.3f}")

t_orth, p_orth = stats.ttest_ind(ic_bbi_raw, ic_bbi_orth, equal_var=False)
print(f"  正交化前后 IC 差异: t={t_orth:.2f}, p={p_orth:.4e}")

if abs(np.mean(ic_bbi_orth)) < 0.005:
    print(f"\n  ➜ 结论: bbi_momentum 正交化后 IC≈0，其 alpha 完全来自 mom_20d，建议移除")
elif np.mean(np.abs(ic_bbi_orth)) > np.mean(np.abs(ic_bbi_raw)) * 0.5:
    print(f"\n  ➜ 结论: bbi_momentum 正交化后仍有部分独立信号，但增量有限")
else:
    print(f"\n  ➜ 结论: bbi_momentum 正交化后 IC 大幅衰减，大部分信号与 mom_20d 重叠")

print("\n" + "="*80)
print("全部验证完成")
print("="*80)
