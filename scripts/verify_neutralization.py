"""
验证：市值中性化方法研究
包括：因子-市值相关性、中性化效果、因子IC稳定性、层间权重理论值
"""
import os
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_parquet(os.path.expanduser('~/code/qlib/data/qlib_data/cn_data/factor_data.parquet'))
print(f"=== 数据概览 ===")
print(f"Shape: {df.shape}")
print(f"Date range: {df['datetime'].min()} ~ {df['datetime'].max()}")
print(f"Stocks per date: {df.groupby('datetime')['instrument'].count().mean():.0f}")
print()

# Factor columns from factors.py
parquet_factors_alpha = ['roa_fina', 'book_to_market', 'ebit_to_mv', 'ocf_to_ev', 'retained_earnings']
parquet_factors_risk = ['turnover_rate_f']
all_parquet_factors = parquet_factors_alpha + parquet_factors_risk

market_cap_col = 'total_mv'

# Clean data: remove inf/nan
clean_mask = np.isfinite(df[market_cap_col])
for f in all_parquet_factors:
    if f in df.columns:
        clean_mask &= np.isfinite(df[f])

df_clean = df[clean_mask].copy()
print(f"Clean data shape: {df_clean.shape}")

# ─── 1. 各因子与市值的相关性（月度截面均值）───
print("\n" + "="*70)
print("1. 各因子与市值(total_mv)的截面相关系数（月度均值）")
print("="*70)

df_clean['month'] = df_clean['datetime'].dt.to_period('M')

cap_corr_results = {}
for f in all_parquet_factors:
    if f not in df_clean.columns:
        continue
    monthly_corr = df_clean.groupby('month').apply(
        lambda g: g[f].corr(g[market_cap_col])
    )
    cap_corr_results[f] = {
        'mean': monthly_corr.mean(),
        'std': monthly_corr.std(),
        't_stat': monthly_corr.mean() / monthly_corr.std() * np.sqrt(len(monthly_corr)),
        'pct_positive': (monthly_corr > 0).mean(),
    }
    print(f"  {f:25s}: ρ={cap_corr_results[f]['mean']:+.4f} ± {cap_corr_results[f]['std']:.4f}, "
          f"t={cap_corr_results[f]['t_stat']:.2f}, "
          f"正相关月份={cap_corr_results[f]['pct_positive']:.1%}")

# ─── 2. 市值中性化前后对比 ───
print("\n" + "="*70)
print("2. 市值中性化效果（OLS残差法: factor ~ log(market_cap)）")
print("="*70)

for f in all_parquet_factors:
    if f not in df_clean.columns:
        continue
    
    # Monthly cross-sectional OLS residual
    def neutralize(g):
        if len(g) < 30:
            return pd.Series(np.nan, index=g.index)
        log_cap = np.log(g[market_cap_col])
        y = g[f]
        slope, intercept, r_val, p_val, std_err = stats.linregress(log_cap, y)
        residual = y - (slope * log_cap + intercept)
        return residual
    
    df_clean[f'_neut_{f}'] = df_clean.groupby('month').apply(neutralize).reset_index(level=0, drop=True)

print(f"{'Factor':25s} {'原始ρ(cap)':>12s} {'中性化ρ(cap)':>14s} {'消除比例':>10s}")
print("-"*65)
for f in all_parquet_factors:
    if f not in df_clean.columns:
        continue
    orig_corr = df_clean[f].corr(df_clean[market_cap_col])
    neut_corr = df_clean[f'_neut_{f}'].corr(df_clean[market_cap_col])
    reduction = 1 - abs(neut_corr) / abs(orig_corr) if abs(orig_corr) > 1e-8 else 0
    print(f"  {f:25s} {orig_corr:+12.4f} {neut_corr:+14.6f} {reduction:10.1%}")

# ─── 3. 因子IC稳定性分析（月度IC统计）───
print("\n" + "="*70)
print("3. 因子IC稳定性分析（截面Spearman秩相关，月度收益率）")
print("="*70)

# Compute forward returns using circ_mv as price proxy
df_sorted = df_clean.sort_values(['instrument', 'datetime']).copy()

# 20-day forward return: use circ_mv changes
df_sorted['fwd_ret'] = df_sorted.groupby('instrument')['circ_mv'].shift(-20) / df_sorted['circ_mv'] - 1

# Compute IC: Spearman rank correlation between factor and forward return
print("\nUsing 20-day forward return (circ_mv proxy) for IC:")
print(f"{'Factor':25s} {'IC均值':>10s} {'IC标准差':>10s} {'IR':>8s} {'t(IC)':>8s} {'IC>0%':>8s}")
print("-"*75)

ic_results = {}
for f in all_parquet_factors:
    if f not in df_clean.columns:
        continue
    # Monthly cross-sectional IC
    def compute_ic(g):
        if len(g) < 30 or g['fwd_ret'].isna().sum() > len(g) * 0.3:
            return np.nan
        ic, _ = stats.spearmanr(g[f], g['fwd_ret'])
        return ic
    
    monthly_ic = df_sorted.groupby('month').apply(compute_ic)
    monthly_ic = monthly_ic.dropna()
    
    if len(monthly_ic) > 0:
        ic_mean = monthly_ic.mean()
        ic_std = monthly_ic.std()
        ir = ic_mean / ic_std if ic_std > 0 else 0
        t_stat = ic_mean / ic_std * np.sqrt(len(monthly_ic)) if ic_std > 0 else 0
        pct_pos = (monthly_ic > 0).mean()
        ic_results[f] = {'mean': ic_mean, 'std': ic_std, 'ir': ir, 't': t_stat, 'pct_pos': pct_pos}
        print(f"  {f:25s} {ic_mean:+10.4f} {ic_std:10.4f} {ir:+8.3f} {t_stat:+8.2f} {pct_pos:8.1%}")

# ─── 4. 中性化前后IC对比 ───
print("\n" + "="*70)
print("4. 中性化前后IC对比")
print("="*70)
print(f"{'Factor':25s} {'原始IC':>10s} {'原始IR':>8s} {'中性化IC':>10s} {'中性化IR':>10s} {'IR变化':>8s}")
print("-"*80)

for f in all_parquet_factors:
    if f not in ic_results:
        continue
    orig = ic_results[f]
    
    def compute_ic_neut(g):
        col = f'_neut_{f}'
        if col not in g.columns or len(g) < 30 or g['fwd_ret'].isna().sum() > len(g) * 0.3:
            return np.nan
        ic, _ = stats.spearmanr(g[col], g['fwd_ret'])
        return ic
    
    monthly_ic_neut = df_sorted.groupby('month').apply(compute_ic_neut).dropna()
    
    if len(monthly_ic_neut) > 0:
        neut_mean = monthly_ic_neut.mean()
        neut_std = monthly_ic_neut.std()
        neut_ir = neut_mean / neut_std if neut_std > 0 else 0
        ir_change = (neut_ir - orig['ir']) / abs(orig['ir']) if abs(orig['ir']) > 1e-8 else 0
        print(f"  {f:25s} {orig['mean']:+10.4f} {orig['ir']:+8.3f} {neut_mean:+10.4f} {neut_ir:+10.3f} {ir_change:+8.1%}")

# ─── 5. Grinold公式：理论最优层间权重 ───
print("\n" + "="*70)
print("5. Grinold公式：理论最优层间权重 w ∝ IR × √N")
print("="*70)

# Using the IR values from factors.py metadata + IC analysis
factor_metadata = {
    'alpha': {'factors': ['roa_fina', 'book_to_market', 'ebit_to_mv', 'ocf_to_ev', 'retained_earnings'],
              'N': 5},
    'risk': {'factors': ['turnover_rate_f'], 'N': 1},
    'enhance': {'factors': ['bbi_momentum', 'price_pos_52w', 'mom_20d'], 'N': 3},
}

# IR from factors.py
ir_from_meta = {
    'alpha': 0.30,  # avg of [0.30, 0.14, 0.33, 0.33, 0.33]
    'risk': 0.37,   # avg of [0.31, 0.42]
    'enhance': 0.29,  # avg of [0.26, 0.24, 0.38]
}

print("\n基于 factors.py IR 元数据:")
raw_weights = {}
for cat in ['alpha', 'risk', 'enhance']:
    N = factor_metadata[cat]['N']
    IR = ir_from_meta[cat]
    w = IR * np.sqrt(N)
    raw_weights[cat] = w
    print(f"  {cat:10s}: IR={IR:.3f}, N={N}, w_raw = {IR:.3f} × √{N} = {w:.4f}")

total = sum(raw_weights.values())
optimal_weights = {k: v/total for k, v in raw_weights.items()}
print(f"\n理论最优权重: alpha={optimal_weights['alpha']:.3f}, risk={optimal_weights['risk']:.3f}, enhance={optimal_weights['enhance']:.3f}")
print(f"当前默认权重: alpha=0.550, risk=0.200, enhance=0.250")
print(f"\n差异分析:")
for cat in ['alpha', 'risk', 'enhance']:
    current = {'alpha': 0.55, 'risk': 0.20, 'enhance': 0.25}[cat]
    delta = optimal_weights[cat] - current
    print(f"  {cat:10s}: 理论={optimal_weights[cat]:.3f}, 当前={current:.3f}, 差异={delta:+.3f}")

# ─── 6. 因子间相关性矩阵 ───
print("\n" + "="*70)
print("6. 因子间截面相关性矩阵（月度均值）")
print("="*70)

factor_cols = [f for f in all_parquet_factors if f in df_clean.columns]
corr_matrix = pd.DataFrame(index=factor_cols, columns=factor_cols, dtype=float)

for i, f1 in enumerate(factor_cols):
    for j, f2 in enumerate(factor_cols):
        if i > j:
            corr_matrix.loc[f1, f2] = corr_matrix.loc[f2, f1]
            continue
        monthly_corr = df_clean.groupby('month').apply(
            lambda g: g[f1].corr(g[f2])
        )
        corr_matrix.loc[f1, f2] = monthly_corr.mean()

print(corr_matrix.round(4).to_string())

# ─── 7. 条件分析：大/小市值分组IC差异 ───
print("\n" + "="*70)
print("7. 按市值分组（大/小盘）的因子IC差异")
print("="*70)

# Split by median market cap within each month
df_sorted['cap_group'] = df_sorted.groupby('month')[market_cap_col].transform(
    lambda x: pd.qcut(x, 2, labels=['small', 'large'], duplicates='drop')
)

for f in all_parquet_factors[:3]:  # just top 3 to save space
    if f not in df_clean.columns:
        continue
    print(f"\n  {f}:")
    for grp in ['small', 'large']:
        def compute_ic_grp(g):
            if len(g) < 20 or g['fwd_ret'].isna().sum() > len(g) * 0.3:
                return np.nan
            ic, _ = stats.spearmanr(g[f], g['fwd_ret'])
            return ic
        
        grp_data = df_sorted[df_sorted['cap_group'] == grp]
        monthly_ic = grp_data.groupby('month').apply(compute_ic_grp).dropna()
        if len(monthly_ic) > 0:
            print(f"    {grp:6s}: IC={monthly_ic.mean():+.4f}, IR={monthly_ic.mean()/monthly_ic.std():+.3f}, n={len(monthly_ic)}")

print("\n" + "="*70)
print("=== 验证完成 ===")
