"""Alpha360 因子年度 IC 拆解"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from modules.modeling.predictive_signal import _alpha360_feature_map, _resolve_alpha158_instruments, init_qlib, load_features_safe

init_qlib()
fmap = _alpha360_feature_map()

# Select representative factors per category per window
windows = [1, 3, 5, 8, 11, 12, 13, 14, 15, 20, 30, 40, 59]
categories = ['LOW', 'CLOSE', 'HIGH', 'VWAP', 'VOLUME']
factors = []
for cat in categories:
    for w in windows:
        name = f'{cat}{w}'
        if name in fmap:
            factors.append(name)
factors = sorted(set(factors))
print(f'Computing per-year IC for {len(factors)} factors...')

# Load data
instruments = _resolve_alpha158_instruments(
    '2019-01-01', '2026-05-01',
    {'alpha158_universe': 'csi300'}, {'universe': 'csi300'},
)
from qlib.data import D
if isinstance(instruments, str):
    instruments = D.instruments(market=instruments)

expressions = [fmap[k] for k in factors]
df_raw = load_features_safe(instruments, expressions, start_time='2018-06-01', end_time='2026-06-01', freq='day')
if list(df_raw.index.names) == ['instrument', 'datetime']:
    df_raw = df_raw.swaplevel().sort_index()
df_raw.columns = factors

close = load_features_safe(instruments, ['$close'], start_time='2018-06-01', end_time='2026-07-01', freq='day')
if list(close.index.names) == ['instrument', 'datetime']:
    close = close.swaplevel().sort_index()
close_series = close.iloc[:, 0].astype(float)

# 20-day forward return labels
future = close_series.groupby(level='instrument').shift(-20)
labels = future / close_series - 1.0

aligned = df_raw.copy()
aligned['year'] = aligned.index.get_level_values('datetime').year
aligned['label'] = labels.reindex(aligned.index)

# Compute rank IC by year per factor
years = list(range(2019, 2027))
results = {col: {} for col in factors}
for col in factors:
    for yr in years:
        subset = aligned[aligned['year'] == yr].dropna(subset=[col, 'label'])
        if len(subset) < 50:
            results[col][yr] = np.nan
        else:
            results[col][yr] = subset[col].rank().corr(subset['label'].rank())

# ==== PRINT TABLES ====

# Table 1: Key factors x year
print()
print("=" * 85)
print("价格类代表因子 年度 Rank IC (20-day forward)")
print("=" * 85)
price_factors = [f for f in factors if not f.startswith('VOLUME')]
# Pick best per category
show = []
for cat in ['LOW', 'CLOSE', 'HIGH', 'VWAP']:
    for w in [11, 15, 20, 30, 59]:
        name = f'{cat}{w}'
        if name in factors:
            show.append(name)
show = sorted(set(show))

header = f"{'':>12}"
for yr in years:
    header += f" {yr:>7}"
print(header)
print("-" * 85)
for col in show:
    row = f"{col:>12}"
    vals = [results[col].get(yr, np.nan) for yr in years]
    best_yr = np.argmax(np.abs(vals))
    for i, v in enumerate(vals):
        marker = " *" if i == best_yr else "  "
        if np.isnan(v):
            row += f" {'NA':>5}  "
        else:
            row += f" {v:>+.4f}{marker}"
    print(row)

# Table 2: VOLUME
print()
print("=" * 85)
print("成交量因子 年度 Rank IC (20-day forward)")
print("=" * 85)
vol_factors = [f for f in factors if f.startswith('VOLUME') and int(f.replace('VOLUME','')) in [11,15,20,30,59]]
header = f"{'':>12}"
for yr in years:
    header += f" {yr:>7}"
print(header)
print("-" * 85)
for col in sorted(vol_factors, key=lambda x: int(x.replace('VOLUME',''))):
    row = f"{col:>12}"
    vals = [results[col].get(yr, np.nan) for yr in years]
    for v in vals:
        if np.isnan(v):
            row += f" {'NA':>5}  "
        else:
            row += f" {v:>+.4f}  "
    print(row)

# Table 3: Category average by year
print()
print("=" * 85)
print("各类别平均 IC by year (11-15天窗口)")
print("=" * 85)
header = f"{'':>10}"
for yr in years:
    header += f" {yr:>7}"
print(header)
print("-" * 85)
for cat in ['LOW', 'CLOSE', 'HIGH', 'VWAP', 'VOLUME']:
    cat_avg = {}
    for yr in years:
        vals = []
        for w in [11, 12, 13, 14, 15]:
            name = f'{cat}{w}'
            if name in results:
                v = results[name].get(yr, np.nan)
                if not np.isnan(v):
                    vals.append(v)
        cat_avg[yr] = np.mean(vals) if vals else np.nan
    row = f"{cat:>10}"
    for yr in years:
        v = cat_avg[yr]
        if np.isnan(v):
            row += f" {'NA':>5}  "
        else:
            row += f" {v:>+7.4f}"
    print(row)

# Save
with open('/tmp/scan_factor/alpha360_yearly_ic.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: /tmp/scan_factor/alpha360_yearly_ic.json")
