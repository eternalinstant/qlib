"""
验证脚本v3: 条件动量、risk层权重、因子timing
"""
import pandas as pd
import numpy as np
from scipy import stats
from collections import defaultdict
import struct, os
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "data/qlib_data/cn_data"

# Read calendar
with open(f'{DATA_DIR}/calendars/day.txt') as f:
    calendar = [l.strip() for l in f if l.strip()]
n_days = len(calendar)
dates = pd.to_datetime(calendar)
print(f"Calendar: {n_days} days ({calendar[0]} to {calendar[-1]})")

# Read instruments (tab-separated: CODE\tstart\tend)
with open(f'{DATA_DIR}/instruments/all.txt') as f:
    instruments = [l.strip().split('\t')[0].lower() for l in f if l.strip()]
print(f"Total instruments: {len(instruments)}")

# Read close prices from binary files
print("Reading close prices...")
close_series = {}
feature_dir = f'{DATA_DIR}/features'
for inst in instruments:
    fpath = f'{feature_dir}/{inst}/close.day.bin'
    if not os.path.exists(fpath):
        continue
    with open(fpath, 'rb') as f:
        raw = f.read()
    if len(raw) < 8:
        continue
    start_idx = int(np.frombuffer(raw[:4], dtype='<f')[0])
    data = np.frombuffer(raw[4:], dtype='<f').astype(np.float64)
    full = np.full(n_days, np.nan)
    end_idx = min(start_idx + len(data), n_days)
    full[start_idx:end_idx] = data[:end_idx - start_idx]
    full[full == 0] = np.nan
    
    if np.sum(~np.isnan(full)) < 200:
        continue
    close_series[inst] = pd.Series(full, index=dates)

close_df = pd.DataFrame(close_series)
# Convert index to DatetimeIndex
close_df.index = pd.to_datetime(close_df.index)
print(f"Close matrix: {close_df.shape}, non-null: {close_df.count().sum().sum()}")

# ============================================================================
# Compute factors
# ============================================================================
print("Computing factors...")
ret_1d = close_df.pct_change()
mom_20d = close_df / close_df.shift(20) - 1
ma3 = close_df.rolling(3, min_periods=3).mean()
ma6 = close_df.rolling(6, min_periods=6).mean()
ma12 = close_df.rolling(12, min_periods=12).mean()
ma24 = close_df.rolling(24, min_periods=24).mean()
bbi_momentum = close_df / ((ma3 + ma6 + ma12 + ma24) / 4) - 1
min252 = close_df.rolling(252, min_periods=60).min()
max252 = close_df.rolling(252, min_periods=60).max()
price_pos_52w = (close_df - min252) / (max252 - min252 + 1e-8)
vol_std_20d_neg = -ret_1d.rolling(20, min_periods=20).std()
fwd_ret_20d = close_df.pct_change(-20).shift(-20)

# Market regime detection
ret_mean_60 = ret_1d.rolling(60, min_periods=20).mean()
ret_std_60 = ret_1d.rolling(60, min_periods=20).std()
trend_strength = ret_mean_60 / (ret_std_60 + 1e-8)
daily_market_trend = trend_strength.median(axis=1)
threshold = daily_market_trend.abs().quantile(0.5)
daily_regime = pd.Series(np.where(daily_market_trend.abs() > threshold, 'trend', 'consolidation'), index=close_df.index)
print(f"Regime: trend={sum(daily_regime=='trend')} days, consolidation={sum(daily_regime=='consolidation')} days")

# ============================================================================
# Helpers
# ============================================================================
def compute_daily_ic(factor_df, fwd_df, min_valid=100):
    results = []
    idx = factor_df.index
    start_i = 60
    end_i = len(idx) - 20
    for i in range(start_i, end_i):
        d = idx[i]
        fac = factor_df.iloc[i].values.astype(float)
        fwd = fwd_df.iloc[i].values.astype(float)
        valid = (~np.isnan(fac)) & (~np.isnan(fwd)) & (np.abs(fwd) < 1)
        if valid.sum() > min_valid:
            ic, _ = stats.spearmanr(fac[valid], fwd[valid])
            if not np.isnan(ic):
                results.append((d, ic))
    return results

def agg_by_regime(daily_ics, regime_series):
    rm = defaultdict(list)
    for d, ic in daily_ics:
        r = regime_series.get(d, 'unknown')
        ym = d.to_period('M')
        rm[(str(ym), r)].append(ic)
    return rm

def agg_monthly(ics_by_ym):
    ma = defaultdict(list)
    for (ym, r), ics in ics_by_ym.items():
        ma[ym].extend(ics)
    return {ym: np.mean(ics) for ym, ics in ma.items()}

# ============================================================================
# 验证1: 条件动量 IC by regime
# ============================================================================
print("\n" + "=" * 70)
print("验证1: 条件动量 — 按市场状态分组的因子IC")
print("=" * 70)

qlib_factors = {
    'mom_20d': mom_20d,
    'bbi_momentum': bbi_momentum,
    'price_pos_52w': price_pos_52w,
    'vol_std_20d_neg': vol_std_20d_neg,
}

all_regime_results = {}
for fname, fdf in qlib_factors.items():
    daily_ics = compute_daily_ic(fdf, fwd_ret_20d)
    rm = agg_by_regime(daily_ics, daily_regime)
    all_regime_results[fname] = rm
    
    print(f"\n  {fname} ({len(daily_ics)} daily obs):")
    for regime in ['trend', 'consolidation']:
        monthly_ics = [np.mean(ics) for (ym, r), ics in rm.items() if r == regime]
        if len(monthly_ics) > 6:
            ics = np.array(monthly_ics)
            m = np.mean(ics); s = np.std(ics)
            ir = m/s if s > 0 else 0
            t = m / (s / np.sqrt(len(ics)))
            p = 2 * (1 - stats.t.cdf(abs(t), len(ics)-1))
            pct = (ics > 0).mean() * 100
            print(f"    {regime}: mean_IC={m:+.4f}, std={s:.4f}, IR={ir:+.3f}, t={t:+.2f}, p={p:.4f}, pos={pct:.1f}% ({len(ics)}m)")
    
    t_ics = [np.mean(ics) for (ym, r), ics in rm.items() if r == 'trend']
    c_ics = [np.mean(ics) for (ym, r), ics in rm.items() if r == 'consolidation']
    if len(t_ics) > 6 and len(c_ics) > 6:
        t_diff, p_diff = stats.ttest_ind(t_ics, c_ics, equal_var=False)
        print(f"    Welch test: t={t_diff:+.3f}, p={p_diff:.4f}")

# ============================================================================
# 验证2: Parquet因子 + Grinold权重
# ============================================================================
print("\n" + "=" * 70)
print("验证2: 各层因子IR对比 + Grinold理论权重")
print("=" * 70)

df = pd.read_parquet(f'{DATA_DIR}/factor_data.parquet')
df['ym'] = df['datetime'].dt.to_period('M')
# Normalize instrument codes to lowercase
df['inst_lower'] = df['instrument'].str.lower()
print(f"Parquet: {len(df)} rows")

# Build date-to-index mapping for close_df
date_to_idx = {d: i for i, d in enumerate(close_df.index)}

parquet_factors = ['roa_fina', 'book_to_market', 'ebit_to_mv', 'ocf_to_ev', 
                   'retained_earnings', 'turnover_rate_f']
factor_cats = {
    'roa_fina': 'alpha', 'book_to_market': 'alpha', 'ebit_to_mv': 'alpha',
    'ocf_to_ev': 'alpha', 'retained_earnings': 'alpha', 'turnover_rate_f': 'risk'
}

all_ir = []

for factor in parquet_factors:
    monthly_ics = []
    for ym, grp in df.groupby('ym'):
        last_date = grp['datetime'].max()
        ts = pd.Timestamp(last_date)
        if ts not in date_to_idx:
            # Find closest prior date
            candidates = [d for d in date_to_idx if d <= ts]
            if not candidates: continue
            ts = max(candidates)
        i = date_to_idx[ts]
        if i + 20 >= len(close_df.index): continue
        
        fwd = fwd_ret_20d.iloc[i].values.astype(float)
        fac_vals = grp.groupby('inst_lower')[factor].last()
        common = fac_vals.index.intersection(close_df.columns)
        if len(common) < 100: continue
        
        fv = fac_vals[common].values.astype(float)
        rv = fwd[close_df.columns.get_indexer(common)].astype(float)
        valid = (~np.isnan(fv)) & (~np.isnan(rv)) & (np.abs(rv) < 1)
        if valid.sum() > 100:
            ic, _ = stats.spearmanr(fv[valid], rv[valid])
            if not np.isnan(ic):
                monthly_ics.append(ic)
    
    if len(monthly_ics) > 6:
        ics = np.array(monthly_ics)
        m = np.mean(ics); s = np.std(ics)
        n_m = len(ics)
        t_stat = m / (s / np.sqrt(n_m))
        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), n_m - 1))
        all_ir.append({
            'factor': factor, 'category': factor_cats[factor],
            'mean_ic': m, 'std_ic': s, 'IR': m/s if s > 0 else 0,
            't_stat': t_stat, 'p_val': p_val,
            'pct_pos': (ics > 0).mean() * 100, 'n_months': n_m
        })
        print(f"  {factor}: IC={m:+.4f}, std={s:.4f}, IR={m/s:+.3f}, t={t_stat:+.2f}, p={p_val:.4f} ({n_m}m)")
    else:
        print(f"  {factor}: insufficient data ({len(monthly_ics)} months)")

qlib_cat_map = {'mom_20d': 'enhance', 'bbi_momentum': 'enhance',
                'price_pos_52w': 'enhance', 'vol_std_20d_neg': 'risk'}

for fname, fdf in qlib_factors.items():
    rm = all_regime_results[fname]
    monthly_avg = agg_monthly(rm)
    ics = np.array(list(monthly_avg.values()))
    if len(ics) > 6:
        m = np.mean(ics); s = np.std(ics)
        n_m = len(ics)
        t_stat = m / (s / np.sqrt(n_m))
        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), n_m - 1))
        all_ir.append({
            'factor': fname, 'category': qlib_cat_map[fname],
            'mean_ic': m, 'std_ic': s, 'IR': m/s if s > 0 else 0,
            't_stat': t_stat, 'p_val': p_val,
            'pct_pos': (ics > 0).mean() * 100, 'n_months': n_m
        })
        print(f"  {fname}: IC={m:+.4f}, std={s:.4f}, IR={m/s:+.3f}, t={t_stat:+.2f}, p={p_val:.4f} ({n_m}m)")

ir_df = pd.DataFrame(all_ir)
print("\n--- Full IR Summary ---")
print(ir_df.sort_values('IR', ascending=False).to_string(index=False))

# Grinold weights
print("\n--- Grinold Theoretical Layer Weights (w ∝ IR × √N) ---")
layer_data = {}
for cat in ['alpha', 'risk', 'enhance']:
    cf = ir_df[ir_df['category'] == cat]
    if len(cf) > 0:
        avg_ir = cf['IR'].mean()
        n = len(cf)
        w = avg_ir * np.sqrt(n)
        layer_data[cat] = {'avg_ir': avg_ir, 'n': n, 'w': w}
        print(f"  {cat.upper()}: avg_IR={avg_ir:+.3f}, N={n}, w∝IR×√N={w:+.4f}")

total_w = sum(abs(v['w']) for v in layer_data.values())
if total_w > 0:
    print(f"\n  Normalized theoretical weights vs current config:")
    for cat, v in layer_data.items():
        nw = abs(v['w']) / total_w
        current = {'alpha': 0.550, 'risk': 0.200, 'enhance': 0.250}.get(cat, 0)
        print(f"    {cat}: Grinold={nw:.3f}, current={current:.3f}, diff={nw-current:+.3f}")

# ============================================================================
# 验证3: 因子IC可预测性
# ============================================================================
print("\n" + "=" * 70)
print("验证3: 因子IC可预测性 (Factor Timing)")
print("=" * 70)

factor_monthly_ts = {}

for factor in parquet_factors:
    ics_dict = {}
    for ym, grp in df.groupby('ym'):
        last_date = grp['datetime'].max()
        ts = pd.Timestamp(last_date)
        if ts not in date_to_idx:
            candidates = [d for d in date_to_idx if d <= ts]
            if not candidates: continue
            ts = max(candidates)
        i = date_to_idx[ts]
        if i + 20 >= len(close_df.index): continue
        
        fwd = fwd_ret_20d.iloc[i].values.astype(float)
        fac_vals = grp.groupby('inst_lower')[factor].last()
        common = fac_vals.index.intersection(close_df.columns)
        if len(common) < 100: continue
        
        fv = fac_vals[common].values.astype(float)
        rv = fwd[close_df.columns.get_indexer(common)].astype(float)
        valid = (~np.isnan(fv)) & (~np.isnan(rv)) & (np.abs(rv) < 1)
        if valid.sum() > 100:
            ic, _ = stats.spearmanr(fv[valid], rv[valid])
            if not np.isnan(ic):
                ics_dict[str(ym)] = ic
    factor_monthly_ts[factor] = ics_dict

for fname in qlib_factors:
    rm = all_regime_results[fname]
    monthly_avg = agg_monthly(rm)
    factor_monthly_ts[fname] = monthly_avg

print("\nFactor IC Autocorrelation (AR(1)):")
print(f"| {'Factor':<22} | {'AR(1)':>7} | {'t-stat':>7} | {'p-val':>7} | Interpretation |")
print(f"|{'-'*24}|{'-'*9}|{'-'*9}|{'-'*9}|----------------|")
for factor in sorted(factor_monthly_ts.keys()):
    d = factor_monthly_ts[factor]
    if len(d) < 12: continue
    months = sorted(d.keys())
    ics = np.array([d[m] for m in months])
    ac = np.corrcoef(ics[:-1], ics[1:])[0, 1]
    if np.isnan(ac): ac = 0.0
    n = len(ics) - 1
    se = 1.0 / np.sqrt(n)
    t = ac / se
    p = 2 * (1 - stats.norm.cdf(abs(t)))
    interp = "PREDICTABLE" if abs(ac) > 0.3 else ("weakly predictable" if abs(ac) > 0.15 else "NOT predictable")
    print(f"| {factor:<22} | {ac:>+7.3f} | {t:>+7.2f} | {p:>7.4f} | {interp} |")

# Cross-factor IC correlation
print("\nFactor IC cross-correlation (|ρ|>0.5 highlighted):")
fnames = sorted(factor_monthly_ts.keys())
ic_cross = pd.DataFrame(index=fnames, columns=fnames, dtype=float)
for f1 in fnames:
    for f2 in fnames:
        d1, d2 = factor_monthly_ts[f1], factor_monthly_ts[f2]
        common = sorted(set(d1.keys()) & set(d2.keys()))
        if len(common) > 12:
            v1 = np.array([d1[m] for m in common])
            v2 = np.array([d2[m] for m in common])
            ic_cross.loc[f1, f2] = np.corrcoef(v1, v2)[0, 1]
print(ic_cross.round(2).to_string())

print("\n\n=== ALL VERIFICATIONS COMPLETE ===")
