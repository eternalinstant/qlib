#!/usr/bin/env python3
"""
年度 Top10 因子扫描
计算所有因子的年度 Rank IC，输出每年 top10
"""
import pandas as pd
import numpy as np
from pathlib import Path
import os, sys, time, warnings
warnings.filterwarnings('ignore')

PROJECT = Path('/Users/sxt/code/qlib')
RAW_DIR = PROJECT / 'data' / 'qlib_data' / 'raw_data'
FACTOR_PATH = PROJECT / 'data' / 'qlib_data' / 'cn_data' / 'factor_data.parquet'
OUTPUT = Path('/tmp/factor_yearly_ic.csv')

t0 = time.time()
print("=" * 60)
print("Step 1: 加载 close price panel")
print("=" * 60)

# --- Load close prices from raw_data ---
raw_files = sorted(RAW_DIR.glob('*.parquet'))
print(f"  {len(raw_files)} 个 raw_data 文件")

close_dict = {}
for f in raw_files:
    try:
        df = pd.read_parquet(f, columns=['date', 'close', 'symbol'])
        if df.empty:
            continue
        # Convert symbol 600000.SH -> sh600000
        sym = df['symbol'].iloc[0]
        if sym.endswith('.SH'):
            qlib_sym = 'sh' + sym.replace('.SH', '')
        elif sym.endswith('.SZ'):
            qlib_sym = 'sz' + sym.replace('.SZ', '')
        elif sym.endswith('.BJ'):
            qlib_sym = 'bj' + sym.replace('.BJ', '')
        else:
            continue
        df = df.set_index('date')
        close_dict[qlib_sym] = df['close']
    except:
        continue

close = pd.DataFrame(close_dict)
close.index = pd.to_datetime(close.index)
# Filter to A-share main board (exclude BJ)
close = close[[c for c in close.columns if not c.startswith('bj')]]
print(f"  close panel: {close.shape}, {close.index.min()} ~ {close.index.max()}")

# Remove stocks with too few observations
min_obs = 252  # at least 1 year
valid = close.count() >= min_obs
close = close[valid[valid].index]
print(f"  after filter: {close.shape}")

print(f"\n  elapsed: {time.time()-t0:.1f}s")

# --- Step 2: Compute technical factors ---
print("\n" + "=" * 60)
print("Step 2: 计算技术因子")
print("=" * 60)

factors_tech = {}

# 20-day momentum (ROC20)
factors_tech['mom_20d'] = close.pct_change(20)

# 60-day momentum
factors_tech['mom_60d'] = close.pct_change(60)

# 120-day momentum
factors_tech['mom_120d'] = close.pct_change(120)

# 20-day volatility (daily return std)
daily_ret = close.pct_change()
factors_tech['vol_20d'] = daily_ret.rolling(20).std()

# 60-day volatility
factors_tech['vol_60d'] = daily_ret.rolling(60).std()

# 20-day mean reversion (negative of recent return)
factors_tech['rev_5d'] = -close.pct_change(5)

# 52-week high distance
rolling_max_252 = close.rolling(252, min_periods=60).max()
factors_tech['dist_52w_high'] = close / rolling_max_252 - 1.0

# 20-day SMA cross
sma_20 = close.rolling(20).mean()
sma_60 = close.rolling(60).mean()
factors_tech['sma_cross'] = (sma_20 / sma_60 - 1.0)

# Amihud illiquidity (|return| / volume)
amount_proxy = close * 100  # rough proxy
illiq_daily = daily_ret.abs() / (amount_proxy + 1e-8)
factors_tech['illiq_20d'] = illiq_daily.rolling(20).mean()

# Max drawdown over 60 days
rolling_min_60 = close.rolling(60).min()
factors_tech['dd_60d'] = close / rolling_min_60 - 1.0

print(f"  计算了 {len(factors_tech)} 个技术因子")
print(f"  elapsed: {time.time()-t0:.1f}s")

# --- Step 3: Load fundamental factors ---
print("\n" + "=" * 60)
print("Step 3: 加载基本面因子")
print("=" * 60)

fd = pd.read_parquet(FACTOR_PATH)
# Keep only relevant factor columns (exclude metadata)
meta_cols = {'instrument', 'datetime'}
factor_cols = [c for c in fd.columns if c not in meta_cols]
print(f"  factor_data: {len(factor_cols)} 个因子列")

# Convert to panel: index=datetime, columns=instrument
fd['datetime'] = pd.to_datetime(fd['datetime'])
fd = fd.set_index(['datetime', 'instrument'])

# Build panels for fundamental factors
factors_fund = {}
for col in factor_cols:
    try:
        panel = fd[col].unstack('instrument')
        # Ensure consistent column naming
        panel.columns = [c.lower() for c in panel.columns]
        factors_fund[col] = panel
    except:
        pass

print(f"  成功转换 {len(factors_fund)} 个基本面因子面板")
print(f"  elapsed: {time.time()-t0:.1f}s")

# --- Step 4: Compute forward returns (label) ---
print("\n" + "=" * 60)
print("Step 4: 计算前瞻收益 (20日)")
print("=" * 60)

# Forward 20-day return
fwd_ret_20 = close.shift(-20) / close - 1.0
fwd_ret_20 = fwd_ret_20.stack()
fwd_ret_20.name = 'fwd_ret_20'
print(f"  fwd_ret_20: {len(fwd_ret_20)} 条")

# Also 5-day forward return
fwd_ret_5 = close.shift(-5) / close - 1.0
fwd_ret_5 = fwd_ret_5.stack()
fwd_ret_5.name = 'fwd_ret_5'

# --- Step 5: Compute Rank IC by year ---
print("\n" + "=" * 60)
print("Step 5: 计算年度 Rank IC")
print("=" * 60)

# Combine all factor panels
all_factors = {}
all_factors.update(factors_tech)
all_factors.update(factors_fund)

print(f"  总计 {len(all_factors)} 个因子")

# Get years
all_dates = close.index
years = sorted(set(all_dates.year))
years = [y for y in years if y >= 2016 and y <= 2025]
print(f"  年份: {years}")

results = []

for year in years:
    year_mask = fwd_ret_20.index.get_level_values(0).year == year
    yr_ret = fwd_ret_20[year_mask]
    
    if len(yr_ret) < 1000:
        print(f"  {year}: 数据不足 ({len(yr_ret)}), 跳过")
        continue
    
    ic_dict = {}
    for fname, fpanel in all_factors.items():
        try:
            # Stack the factor panel
            f_series = fpanel.stack()
            # Align with forward returns
            combined = pd.DataFrame({'factor': f_series, 'ret': yr_ret}).dropna()
            if len(combined) < 500:
                continue
            # Daily rank IC then average
            combined = combined.reset_index()
            combined.columns = ['date', 'stock', 'factor', 'ret']
            daily_ic = combined.groupby('date').apply(
                lambda g: g['factor'].corr(g['ret'], method='spearman') if len(g) >= 10 else np.nan
            )
            mean_ic = daily_ic.mean()
            ic_std = daily_ic.std()
            icir = mean_ic / ic_std * np.sqrt(252) if ic_std > 0 else 0
            
            ic_dict[fname] = {'ic': mean_ic, 'icir': icir, 'n_days': len(daily_ic)}
        except Exception as e:
            pass
    
    # Sort by absolute IC
    sorted_ic = sorted(ic_dict.items(), key=lambda x: abs(x[1]['ic']), reverse=True)
    
    print(f"\n  {year} 年 Top 10 (共 {len(ic_dict)} 因子):")
    for rank, (fname, stats) in enumerate(sorted_ic[:10], 1):
        print(f"    {rank:2d}. {fname:30s} IC={stats['ic']:+.4f}  ICIR={stats['icir']:+.2f}")
        results.append({
            'year': year,
            'rank': rank,
            'factor': fname,
            'rank_ic': stats['ic'],
            'icir': stats['icir'],
        })

# Save results
res_df = pd.DataFrame(results)
res_df.to_csv(OUTPUT, index=False)
print(f"\n{'='*60}")
print(f"✅ 完成! 结果保存到 {OUTPUT}")
print(f"   总耗时: {time.time()-t0:.1f}s")
