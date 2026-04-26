"""
验证：bbi_momentum vs mom_20d 冗余性 + PCA正交化
使用 qlib 二进制数据计算 enhance 因子
"""
import pandas as pd
import numpy as np
from scipy import stats
import struct
import os
import warnings
warnings.filterwarnings('ignore')

def read_qlib_bin(filepath):
    """Read qlib .day.bin file (float32 series with date index)"""
    data = []
    with open(filepath, 'rb') as f:
        while True:
            chunk = f.read(4)
            if not chunk or len(chunk) < 4:
                break
            val = struct.unpack('f', chunk)[0]
            data.append(val)
    return np.array(data, dtype=np.float32)

def load_qlib_features(feat_dir, instrument, fields=['close', 'high', 'low', 'open', 'total_mv']):
    """Load qlib binary features for a single instrument"""
    result = {}
    for field in fields:
        filepath = os.path.join(feat_dir, instrument, f'{field}.day.bin')
        if os.path.exists(filepath):
            result[field] = read_qlib_bin(filepath)
    return result

# Load calendar
cal_dir = os.path.expanduser('~/code/qlib/data/qlib_data/cn_data/calendars/')
day_file = os.path.join(cal_dir, 'day.txt')
with open(day_file, 'r') as f:
    calendar = [line.strip() for line in f.readlines()]
print(f"Calendar: {len(calendar)} days, {calendar[0]} ~ {calendar[-1]}")

# Load instruments (csi300)
inst_dir = os.path.expanduser('~/code/qlib/data/qlib_data/cn_data/instruments/')
all_instruments = set()
for fn in os.listdir(inst_dir):
    if fn.endswith('.txt'):
        with open(os.path.join(inst_dir, fn), 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    all_instruments.add(parts[0])

feat_dir = os.path.expanduser('~/code/qlib/data/qlib_data/cn_data/features/')

# Sample a subset of instruments for efficiency (limit to first 500 that have data)
sample_stocks = []
for inst in sorted(all_instruments):
    if inst.startswith('SH') or inst.startswith('SZ'):
        inst_feat_dir = os.path.join(feat_dir, inst.lower())
        if os.path.exists(inst_feat_dir) and os.path.exists(os.path.join(inst_feat_dir, 'close.day.bin')):
            sample_stocks.append(inst.lower())
            if len(sample_stocks) >= 500:
                break

print(f"Sample stocks: {len(sample_stocks)}")

# Compute bbi_momentum and mom_20d for each stock
results = []
for i, inst in enumerate(sample_stocks):
    try:
        data = load_qlib_features(feat_dir, inst, ['close', 'total_mv'])
        if 'close' not in data or len(data['close']) < 30:
            continue
        
        close = data['close'].astype(np.float64)
        n = len(close)
        # Use only as many dates as we have data for
        n_dates = min(n, len(calendar))
        
        # bbi_momentum = close / mean(MA3, MA6, MA12, MA24) - 1
        ma3 = pd.Series(close).rolling(3).mean().values
        ma6 = pd.Series(close).rolling(6).mean().values
        ma12 = pd.Series(close).rolling(12).mean().values
        ma24 = pd.Series(close).rolling(24).mean().values
        bbi_ma = (ma3 + ma6 + ma12 + ma24) / 4
        bbi_mom = close / bbi_ma - 1
        
        # mom_20d = close / ref(close, 20) - 1
        mom_20d = close / np.roll(close, 20) - 1
        mom_20d[:20] = np.nan
        
        # market cap
        mv = data['total_mv'].astype(np.float64) if 'total_mv' in data else np.full(n, np.nan)
        
        for j in range(24, n_dates):
            if np.isnan(close[j]) or np.isnan(bbi_mom[j]) or np.isnan(mom_20d[j]):
                continue
            results.append({
                'datetime': calendar[j],
                'instrument': inst,
                'bbi_momentum': bbi_mom[j],
                'mom_20d': mom_20d[j],
                'total_mv': mv[j] if j < len(mv) else np.nan,
            })
    except Exception as e:
        continue
    
    if (i + 1) % 100 == 0:
        print(f"  Processed {i+1}/{len(sample_stocks)} stocks...")

df = pd.DataFrame(results)
df['datetime'] = pd.to_datetime(df['datetime'])
print(f"\nComputed factors: {df.shape}")

# ─── A. bbi_momentum vs mom_20d 截面相关系数 ───
print("\n" + "="*70)
print("A. bbi_momentum vs mom_20d 截面相关系数（月度均值）")
print("="*70)

df['month'] = df['datetime'].dt.to_period('M')

monthly_corr = df.groupby('month').apply(
    lambda g: g['bbi_momentum'].corr(g['mom_20d'])
).dropna()
print(f"  月度均值相关系数: ρ = {monthly_corr.mean():.4f} ± {monthly_corr.std():.4f}")
print(f"  t统计量: {monthly_corr.mean()/monthly_corr.std()*np.sqrt(len(monthly_corr)):.2f}")
print(f"  相关系数范围: [{monthly_corr.min():.4f}, {monthly_corr.max():.4f}]")
print(f"  ρ > 0.7 的月份占比: {(monthly_corr > 0.7).mean():.1%}")
print(f"  ρ > 0.5 的月份占比: {(monthly_corr > 0.5).mean():.1%}")

# Overall correlation
overall_corr = df['bbi_momentum'].corr(df['mom_20d'])
print(f"  总体相关系数: ρ = {overall_corr:.4f}")

# ─── B. PCA正交化测试 ───
print("\n" + "="*70)
print("B. PCA正交化：enhance因子组")
print("="*70)

# Monthly PCA
from sklearn.decomposition import PCA

def pca_monthly(g):
    if len(g) < 50:
        return None
    X = g[['bbi_momentum', 'mom_20d']].dropna()
    if len(X) < 50:
        return None
    # Standardize
    X_std = (X - X.mean()) / X.std()
    pca = PCA(n_components=2)
    pca.fit(X_std)
    return pd.Series({
        'pc1_explained': pca.explained_variance_ratio_[0],
        'pc2_explained': pca.explained_variance_ratio_[1],
        'n_stocks': len(X),
    })

pca_results = df.groupby('month').apply(pca_monthly).dropna()
if len(pca_results) > 0:
    pc1 = pca_results['pc1_explained']
    print(f"  PC1 解释方差比（月度均值）: {pc1.mean():.4f} ± {pc1.std():.4f}")
    print(f"  PC2 解释方差比（月度均值）: {pca_results['pc2_explained'].mean():.4f} ± {pca_results['pc2_explained'].std():.4f}")
    print(f"  结论: PC1平均解释{pc1.mean():.1%}方差 → 因子存在{'强' if pc1.mean() > 0.7 else '中等' if pc1.mean() > 0.5 else '弱'}冗余")

# ─── C. bbi_momentum 和 mom_20d 各自与市值的相关性 ───
print("\n" + "="*70)
print("C. enhance因子与市值的相关性")
print("="*70)

for f in ['bbi_momentum', 'mom_20d']:
    valid = df.dropna(subset=[f, 'total_mv'])
    monthly_corr_mv = valid.groupby('month').apply(
        lambda g: g[f].corr(g['total_mv'])
    ).dropna()
    print(f"  {f:20s}: ρ(cap) = {monthly_corr_mv.mean():+.4f} ± {monthly_corr_mv.std():.4f}, "
          f"t={monthly_corr_mv.mean()/monthly_corr_mv.std()*np.sqrt(len(monthly_corr_mv)):.2f}")

print("\n=== enhance因子验证完成 ===")
