"""
验证脚本: 绕过 qlib API，直接读取 binary + parquet 数据
验证: 因子冗余性、IC稳定性、PCA正交化、层间权重优化、条件动量
"""
import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import struct
import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ==================================================================
# 数据加载
# ==================================================================
DATA_DIR = os.path.expanduser("~/code/qlib/data/qlib_data/cn_data")

# Read calendar
with open(os.path.join(DATA_DIR, "calendars", "day.txt")) as f:
    calendar = [line.strip() for line in f if line.strip()]

# Read instruments
instruments = []
with open(os.path.join(DATA_DIR, "instruments", "stocks_with_ohlcv.txt")) as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 3:
            instruments.append((parts[0], parts[1], parts[2]))
print(f"Calendar: {len(calendar)} days")
print(f"Instruments: {len(instruments)}")

# Read parquet factor data
print("Loading factor_data.parquet...")
factor_df = pd.read_parquet(os.path.join(DATA_DIR, "factor_data.parquet"))
factor_df['datetime'] = pd.to_datetime(factor_df['datetime'])
print(f"Factor data: {len(factor_df):,} rows")

# Create calendar index for binary data
cal_idx = pd.DatetimeIndex(calendar)
cal_start = cal_idx[0]

# ==================================================================
# Read binary feature data for all stocks (close, high, low)
# ==================================================================
def read_binary_feature(instrument, feature_name):
    """Read a qlib binary feature file, return aligned to calendar."""
    feature_file = os.path.join(DATA_DIR, "features", instrument, f"{feature_name}.day.bin")
    if not os.path.exists(feature_file):
        return None
    with open(feature_file, 'rb') as f:
        data = f.read()
    n_values = len(data) // 4
    values = np.array(struct.unpack(f'<{n_values}f', data), dtype=np.float32)
    # First value is placeholder (0.0), skip it
    values = values[1:]
    # Align to calendar
    if len(values) > len(calendar):
        values = values[:len(calendar)]
    elif len(values) < len(calendar):
        pad = np.full(len(calendar) - len(values), np.nan, dtype=np.float32)
        values = np.concatenate([pad, values])
    return pd.Series(values, index=cal_idx)

# Read close prices for all instruments (sample first for speed)
print("Reading binary price data for all instruments...")
price_data = {}
for inst, start, end in instruments:
    close = read_binary_feature(inst, "close")
    if close is not None and close.notna().sum() > 100:
        price_data[inst] = close

print(f"Loaded close prices for {len(price_data)} instruments")

# Build a wide DataFrame of close prices: date x instrument
# Use a subset of dates for efficiency: 2018-01-01 to 2026-03-31
start_date = pd.Timestamp('2018-01-01')
end_date = pd.Timestamp('2026-03-31')

# Filter instruments that have data in range
valid_instruments = []
for inst, series in price_data.items():
    mask = (series.index >= start_date) & (series.index <= end_date)
    if mask.sum() > 200:  # at least 200 trading days
        valid_instruments.append(inst)

print(f"Valid instruments in date range: {len(valid_instruments)}")

# Build panel data: dict of {date: {instrument: close}}
# Process in chunks to manage memory
dates_in_range = cal_idx[(cal_idx >= start_date) & (cal_idx <= end_date)]
print(f"Date range: {len(dates_in_range)} days")

# Build DataFrame: rows=date, columns=instrument, values=close
print("Building price matrix...")
close_matrix = pd.DataFrame(index=dates_in_range, columns=valid_instruments, dtype=np.float32)

for inst in valid_instruments:
    s = price_data[inst]
    mask = s.index.isin(dates_in_range)
    close_matrix.loc[s.index[mask], inst] = s[mask].values

print(f"Price matrix: {close_matrix.shape}")
# Free memory
del price_data

# ==================================================================
# 计算因子
# ==================================================================
print("\nComputing factors...")

# mom_20d = close / ref(close, 20) - 1
mom_20d = close_matrix / close_matrix.shift(20) - 1

# bbi_momentum = close / ((MA3 + MA6 + MA12 + MA24) / 4) - 1
ma3 = close_matrix.rolling(3).mean()
ma6 = close_matrix.rolling(6).mean()
ma12 = close_matrix.rolling(12).mean()
ma24 = close_matrix.rolling(24).mean()
bbi_ma = (ma3 + ma6 + ma12 + ma24) / 4
bbi_momentum = close_matrix / bbi_ma - 1

# price_pos_52w = (close - min_252) / (max_252 - min_252 + 1e-8)
min_252 = close_matrix.rolling(252).min()
max_252 = close_matrix.rolling(252).max()
price_pos_52w = (close_matrix - min_252) / (max_252 - min_252 + 1e-8)

# Forward return 20d
fwd_ret_20d = close_matrix.shift(-20) / close_matrix - 1

print("Factors computed.")

# ==================================================================
# 验证1: bbi_momentum 与 mom_20d 冗余性
# ==================================================================
print("\n" + "=" * 70)
print("验证1: bbi_momentum 与 mom_20d 冗余性分析")
print("=" * 70)

corr_list = []
for date in dates_in_range:
    if date not in mom_20d.index:
        continue
    m = mom_20d.loc[date].dropna()
    b = bbi_momentum.loc[date].dropna()
    common = m.index.intersection(b.index)
    if len(common) < 30:
        continue
    c = m[common].corr(b[common])
    if not np.isnan(c):
        corr_list.append({'date': date, 'corr': c})

corr_df = pd.DataFrame(corr_list)
mean_corr = corr_df['corr'].mean()
std_corr = corr_df['corr'].std()
t_stat = mean_corr / (std_corr / np.sqrt(len(corr_df)))

print(f"bbi_momentum vs mom_20d 截面相关系数:")
print(f"  月均 ρ = {mean_corr:.4f}")
print(f"  标准差 σ = {std_corr:.4f}")
print(f"  t统计量 = {t_stat:.2f}")
print(f"  正相关月份占比 = {(corr_df['corr'] > 0).mean():.1%}")
print(f"  ρ > 0.5 的月份占比 = {(corr_df['corr'] > 0.5).mean():.1%}")
print(f"  ρ > 0.7 的月份占比 = {(corr_df['corr'] > 0.7).mean():.1%}")
print(f"  5%分位: {corr_df['corr'].quantile(0.05):.4f}")
print(f"  中位数: {corr_df['corr'].median():.4f}")
print(f"  95%分位: {corr_df['corr'].quantile(0.95):.4f}")
mom_redundancy = "HIGH" if mean_corr > 0.6 else ("MODERATE" if mean_corr > 0.3 else "LOW")
print(f"  冗余度判定: {mom_redundancy}")

# ==================================================================
# 验证2: 因子 IC 滚动稳定性
# ==================================================================
print("\n" + "=" * 70)
print("验证2: 因子 IC 滚动稳定性分析")
print("=" * 70)

# 从 parquet 获取基本面因子，merge 到日期×股票格式
factor_df_pivot = factor_df.pivot_table(index='datetime', columns='instrument', values=['roa_fina', 'book_to_market', 'ebit_to_mv', 'ocf_to_ev', 'retained_earnings', 'turnover_rate_f', 'total_mv'])

factor_names_ic = {
    'mom_20d': mom_20d,
    'bbi_momentum': bbi_momentum,
    'price_pos_52w': price_pos_52w,
}

# Add parquet factors
for col in ['roa_fina', 'book_to_market', 'ebit_to_mv', 'ocf_to_ev', 'retained_earnings', 'turnover_rate_f', 'total_mv']:
    if col in factor_df_pivot.columns:
        s = factor_df_pivot[col]
        # Align index
        factor_names_ic[col] = s.reindex(dates_in_range)

all_factor_names = list(factor_names_ic.keys())
ic_results = {}

for factor_name, factor_mat in factor_names_ic.items():
    ic_series = []
    for date in dates_in_range:
        if date not in factor_mat.index or date not in fwd_ret_20d.index:
            continue
        f = factor_mat.loc[date].dropna()
        r = fwd_ret_20d.loc[date].dropna()
        common = f.index.intersection(r.index)
        if len(common) < 30:
            continue
        # Spearman rank IC
        ic = f[common].corr(r[common], method='spearman')
        ic_series.append({'date': date, 'IC': ic})
    
    ic_s = pd.DataFrame(ic_series)
    if len(ic_s) > 0:
        mean_ic = ic_s['IC'].mean()
        ic_std = ic_s['IC'].std()
        icir = mean_ic / ic_std if ic_std > 0 else 0
        ic_positive = (ic_s['IC'] > 0).mean()
        t_stat_ic = mean_ic / (ic_std / np.sqrt(len(ic_s))) if ic_std > 0 else 0
        
        rolling_ic = ic_s.set_index('date')['IC'].rolling(12).mean()
        rolling_std = ic_s.set_index('date')['IC'].rolling(12).std()
        rolling_icir = (rolling_ic / rolling_std).dropna()
        
        ic_results[factor_name] = {
            'mean_ic': mean_ic, 'icir': icir, 'ic_positive': ic_positive,
            't_stat': t_stat_ic, 'n_months': len(ic_s),
            'rolling_icir_mean': rolling_icir.mean() if len(rolling_icir) > 0 else np.nan,
            'rolling_icir_min': rolling_icir.min() if len(rolling_icir) > 0 else np.nan,
        }

print(f"\n{'因子':<22} {'Mean IC':>8} {'ICIR':>7} {'t统计量':>8} {'IC>0%':>7} {'月份':>5} {'R-ICIR':>8} {'R-Min':>8}")
print("-" * 88)
for name, r in sorted(ic_results.items(), key=lambda x: abs(x[1]['icir']), reverse=True):
    print(f"{name:<22} {r['mean_ic']:>8.4f} {r['icir']:>7.3f} {r['t_stat']:>8.2f} {r['ic_positive']:>6.1%} {r['n_months']:>5d} {r['rolling_icir_mean']:>8.3f} {r['rolling_icir_min']:>8.3f}")

# ==================================================================
# 验证2b: 市值暴露检查 — 各因子与 total_mv 的截面相关
# ==================================================================
print("\n" + "=" * 70)
print("验证2b: 市值暴露 — 各因子与 total_mv 的截面相关性")
print("=" * 70)

mv_factor = factor_df_pivot['total_mv'].reindex(dates_in_range) if 'total_mv' in factor_df_pivot.columns else None

mv_corr_results = {}
check_factors = {
    'mom_20d': mom_20d,
    'bbi_momentum': bbi_momentum,
    'price_pos_52w': price_pos_52w,
}
for col in ['roa_fina', 'book_to_market', 'ebit_to_mv', 'ocf_to_ev', 'retained_earnings', 'turnover_rate_f']:
    if col in factor_df_pivot.columns:
        check_factors[col] = factor_df_pivot[col].reindex(dates_in_range)

if mv_factor is not None:
    for fname, fmat in check_factors.items():
        corr_list_mv = []
        for date in dates_in_range[24::20]:  # sample every 20 days for speed
            if date not in fmat.index or date not in mv_factor.index:
                continue
            f = fmat.loc[date].dropna()
            mv = mv_factor.loc[date].dropna()
            common = f.index.intersection(mv.index)
            if len(common) < 30:
                continue
            c = f[common].corr(mv[common])
            if not np.isnan(c):
                corr_list_mv.append(c)
        
        if corr_list_mv:
            mv_corr_results[fname] = {
                'mean_corr': np.mean(corr_list_mv),
                'std': np.std(corr_list_mv),
                'pct_pos': np.mean([c > 0 for c in corr_list_mv]),
            }
    
    print(f"\n{'因子':<22} {'与市值ρ':>10} {'标准差':>8} {'正相关%':>8}")
    print("-" * 52)
    for name, r in sorted(mv_corr_results.items(), key=lambda x: abs(x[1]['mean_corr']), reverse=True):
        flag = " ⚠️需中性化" if abs(r['mean_corr']) > 0.1 else ""
        print(f"{name:<22} {r['mean_corr']:>10.4f} {r['std']:>8.4f} {r['pct_pos']:>7.1%}{flag}")

# ==================================================================
# 验证3: PCA 因子正交化效果
# ==================================================================
print("\n" + "=" * 70)
print("验证3: PCA 因子正交化 + 残差正交化效果")
print("=" * 70)

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

enhance_names = ['mom_20d', 'bbi_momentum', 'price_pos_52w']
enhance_mats = [mom_20d, bbi_momentum, price_pos_52w]

# 逐月 PCA (sample every 20 days)
pca_explained = []
for date in dates_in_range[24::20]:
    all_data = []
    for fm in enhance_mats:
        if date in fm.index:
            all_data.append(fm.loc[date].dropna())
    if len(all_data) != 3:
        continue
    common_idx = all_data[0].index
    for d in all_data[1:]:
        common_idx = common_idx.intersection(d.index)
    if len(common_idx) < 50:
        continue
    
    mat = pd.DataFrame({n: d[common_idx] for n, d in zip(enhance_names, all_data)})
    scaler = StandardScaler()
    scaled = scaler.fit_transform(mat)
    pca = PCA()
    pca.fit(scaled)
    pca_explained.append({
        'date': date,
        'PC1': pca.explained_variance_ratio_[0],
        'PC2': pca.explained_variance_ratio_[1],
        'PC3': pca.explained_variance_ratio_[2],
    })

pca_exp_df = pd.DataFrame(pca_explained)
if len(pca_exp_df) > 0:
    print(f"\nPCA 解释方差比 (月均值, {len(pca_exp_df)} 个采样日):")
    print(f"  PC1: {pca_exp_df['PC1'].mean():.4f} ± {pca_exp_df['PC1'].std():.4f}")
    print(f"  PC2: {pca_exp_df['PC2'].mean():.4f} ± {pca_exp_df['PC2'].std():.4f}")
    print(f"  PC3: {pca_exp_df['PC3'].mean():.4f} ± {pca_exp_df['PC3'].std():.4f}")
    
    pc1_mean = pca_exp_df['PC1'].mean()
    print(f"\n  PC1 平均解释方差: {pc1_mean:.1%}")
    if pc1_mean > 0.80:
        print("  → 三个 enhance 因子高度冗余，建议用 PC1 替代或正交化处理")
    elif pc1_mean > 0.60:
        print("  → 存在显著冗余，建议 PCA 正交化")
    else:
        print("  → 冗余度较低，三个因子提供独立信息")

# 残差正交化: bbi_momentum 对 mom_20d 回归
print("\n残差正交化 (bbi_momentum ~ mom_20d):")
ortho_ic_list = []
for date in dates_in_range[24::20]:
    m = mom_20d.loc[date].dropna() if date in mom_20d.index else pd.Series(dtype=float)
    b = bbi_momentum.loc[date].dropna() if date in bbi_momentum.index else pd.Series(dtype=float)
    r = fwd_ret_20d.loc[date].dropna() if date in fwd_ret_20d.index else pd.Series(dtype=float)
    
    common = m.index.intersection(b.index).intersection(r.index)
    if len(common) < 50:
        continue
    
    X = m[common].values.reshape(-1, 1)
    y = b[common].values
    reg = LinearRegression().fit(X, y)
    residual = y - reg.predict(X)
    
    ic_orig = np.corrcoef(y, r[common].values)[0, 1]
    ic_resid = np.corrcoef(residual, r[common].values)[0, 1]
    ortho_ic_list.append({
        'IC_orig': ic_orig, 'IC_resid': ic_resid,
        'beta': reg.coef_[0], 'r2': reg.score(X, y),
    })

if ortho_ic_list:
    ortho_df = pd.DataFrame(ortho_ic_list)
    print(f"  回归 β = {ortho_df['beta'].mean():.4f} ± {ortho_df['beta'].std():.4f}")
    print(f"  R² = {ortho_df['r2'].mean():.4f} ± {ortho_df['r2'].std():.4f}")
    print(f"  bbi_momentum 原始 IC = {ortho_df['IC_orig'].mean():.4f}")
    print(f"  正交化后残差 IC = {ortho_df['IC_resid'].mean():.4f}")
    ic_retained = abs(ortho_df['IC_resid'].mean()) / abs(ortho_df['IC_orig'].mean()) * 100 if ortho_df['IC_orig'].mean() != 0 else 0
    print(f"  IC 保留比例: {ic_retained:.1f}%")

# ==================================================================
# 验证4: 层间权重理论最优值 vs 当前 0.55/0.20/0.25
# ==================================================================
print("\n" + "=" * 70)
print("验证4: 层间权重理论最优值 (Grinold w ∝ IR × √N)")
print("=" * 70)

layers = {
    'alpha': ['roa_fina', 'book_to_market', 'ebit_to_mv', 'ocf_to_ev', 'retained_earnings'],
    'risk': ['turnover_rate_f'],
    'enhance': ['mom_20d', 'bbi_momentum', 'price_pos_52w'],
}

layer_stats = {}
for layer_name, factors in layers.items():
    icirs = [ic_results[f]['icir'] for f in factors if f in ic_results]
    mean_icir = np.mean(icirs) if icirs else 0
    n = len(factors)
    layer_stats[layer_name] = {
        'n_factors': n,
        'mean_icir': mean_icir,
        'theoretical_weight': abs(mean_icir) * np.sqrt(n),
    }

print(f"\n{'层':<12} {'因子数':>6} {'Mean ICIR':>10} {'w_theory':>10}")
print("-" * 42)
total_w = 0
for layer_name in ['alpha', 'risk', 'enhance']:
    s = layer_stats[layer_name]
    total_w += s['theoretical_weight']
    print(f"{layer_name:<12} {s['n_factors']:>6d} {s['mean_icir']:>10.3f} {s['theoretical_weight']:>10.3f}")

if total_w > 0:
    print(f"\n理论最优权重 (归一化后):")
    for layer_name in ['alpha', 'risk', 'enhance']:
        s = layer_stats[layer_name]
        w = s['theoretical_weight'] / total_w
        print(f"  {layer_name}: {w:.2%}")

    print(f"\n当前权重: alpha=0.55, risk=0.20, enhance=0.25")
    print(f"理论权重 vs 当前权重对比:")
    for layer_name, current in [('alpha', 0.55), ('risk', 0.20), ('enhance', 0.25)]:
        s = layer_stats[layer_name]
        w_theory = s['theoretical_weight'] / total_w
        diff = w_theory - current
        direction = "↑应增加" if diff > 0.05 else ("↓应减少" if diff < -0.05 else "≈合适")
        print(f"  {layer_name}: 理论={w_theory:.2%}, 当前={current:.0%}, {direction}")

# ==================================================================
# 验证5: 条件动量 — 按市场状态分组
# ==================================================================
print("\n" + "=" * 70)
print("验证5: 条件动量 — 按市场状态(趋势/震荡)分组 IC")
print("=" * 70)

# 用全市场等权均值作为市场指数 proxy
market_ret = mom_20d.mean(axis=1)  # cross-sectional mean of mom_20d as market trend proxy
# 更好的: 60日累计收益率
market_60d_ret = close_matrix.mean(axis=1).pct_change(60).shift(-60)  # proxy

# 或者用 close_matrix 的 60d return
market_index = close_matrix.mean(axis=1)
market_trend_60d = market_index / market_index.shift(60) - 1

trend_ic_list = []
down_ic_list = []
sideways_ic_list = []

for date in dates_in_range[60::5]:
    if date not in mom_20d.index or date not in fwd_ret_20d.index:
        continue
    if date not in market_trend_60d.index:
        continue
    
    mt = market_trend_60d.loc[date]
    if np.isnan(mt):
        continue
    
    m = mom_20d.loc[date].dropna()
    r = fwd_ret_20d.loc[date].dropna()
    common = m.index.intersection(r.index)
    if len(common) < 30:
        continue
    ic = m[common].corr(r[common], method='spearman')
    
    if mt > 0.05:
        trend_ic_list.append(ic)
    elif mt < -0.05:
        down_ic_list.append(ic)
    else:
        sideways_ic_list.append(ic)

def print_conditional(name, ic_list):
    if len(ic_list) > 0:
        arr = np.array(ic_list)
        arr = arr[~np.isnan(arr)]
        if len(arr) > 0:
            print(f"  {name}:")
            print(f"    样本数: {len(arr)}")
            print(f"    mom_20d 平均 IC = {np.mean(arr):.4f}")
            print(f"    ICIR = {np.mean(arr)/np.std(arr):.3f}" if np.std(arr) > 0 else "")
            return np.mean(arr), np.mean(arr)/np.std(arr) if np.std(arr) > 0 else 0
    return None, None

print("\n按市场状态分组:")
print(f"  上升趋势 (60d ret > 5%):")
print_conditional("  上升趋势", trend_ic_list)
print(f"  下降趋势 (60d ret < -5%):")
print_conditional("  下降趋势", down_ic_list)
print(f"  震荡市场 (-5% ~ 5%):")
print_conditional("  震荡市场", sideways_ic_list)

# Also check bbi_momentum conditional
print("\n--- bbi_momentum 条件 IC ---")
bbi_trend_ic, bbi_down_ic, bbi_sideways_ic = [], [], []
for date in dates_in_range[60::5]:
    if date not in bbi_momentum.index or date not in fwd_ret_20d.index or date not in market_trend_60d.index:
        continue
    mt = market_trend_60d.loc[date]
    if np.isnan(mt):
        continue
    b = bbi_momentum.loc[date].dropna()
    r = fwd_ret_20d.loc[date].dropna()
    common = b.index.intersection(r.index)
    if len(common) < 30:
        continue
    ic = b[common].corr(r[common], method='spearman')
    if mt > 0.05:
        bbi_trend_ic.append(ic)
    elif mt < -0.05:
        bbi_down_ic.append(ic)
    else:
        bbi_sideways_ic.append(ic)

print_conditional("  上升趋势", bbi_trend_ic)
print_conditional("  下降趋势", bbi_down_ic)
print_conditional("  震荡市场", bbi_sideways_ic)

print("\n" + "=" * 70)
print("所有验证完成")
print("=" * 70)
