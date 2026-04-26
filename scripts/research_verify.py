"""
验证脚本: 因子冗余性、IC稳定性、PCA正交化、层间权重优化
"""
import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

import qlib
from qlib.data import D

qlib.init(provider_uri=os.path.expanduser("~/code/qlib/data/qlib_data/cn_data"))

# 使用 stocks_with_ohlcv 作为 instrument
instruments = D.instruments("stocks_with_ohlcv")

# ==================================================================
# 验证1: bbi_momentum 与 mom_20d 冗余性
# ==================================================================
print("=" * 70)
print("验证1: bbi_momentum 与 mom_20d 冗余性分析")
print("=" * 70)

fields = [
    "$close/Ref($close,20)-1", 
    "$close/((Mean($close,3)+Mean($close,6)+Mean($close,12)+Mean($close,24))/4)-1"
]
names = ["mom_20d", "bbi_momentum"]
price_df = D.features(instruments, fields, start_time="2018-01-01", end_time="2026-04-15")
price_df.columns = names
price_df = price_df.dropna(subset=['mom_20d', 'bbi_momentum'])
print(f"Qlib 因子数据: {len(price_df):,} 行")

corr_list = []
for date, grp in price_df.groupby(level=0):
    vals = grp.droplevel(0)
    if len(vals) < 30:
        continue
    c = vals['mom_20d'].corr(vals['bbi_momentum'])
    if not np.isnan(c):
        corr_list.append({'date': str(date), 'corr': c})

corr_df = pd.DataFrame(corr_list)
mean_corr = corr_df['corr'].mean()
std_corr = corr_df['corr'].std()
t_stat = mean_corr / (std_corr / np.sqrt(len(corr_df)))

print(f"\nbbi_momentum vs mom_20d 截面相关系数:")
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

all_fields = [
    "Ref($close,-20)/$close-1",
    "roa_fina",
    "book_to_market",
    "ebit_to_mv",
    "ocf_to_ev",
    "retained_earnings",
    "turnover_rate_f",
    "$close/Ref($close,20)-1",
    "$close/((Mean($close,3)+Mean($close,6)+Mean($close,12)+Mean($close,24))/4)-1",
    "($close-Min($close,252))/(Max($close,252)-Min($close,252)+1e-8)",
]
all_names = [
    'fwd_ret_20d', 'roa_fina', 'book_to_market', 'ebit_to_mv', 'ocf_to_ev',
    'retained_earnings', 'turnover_rate_f', 'mom_20d', 'bbi_momentum', 'price_pos_52w'
]

ic_df = D.features(instruments, all_fields, start_time="2018-01-01", end_time="2026-03-31")
ic_df.columns = all_names
ic_df = ic_df.dropna(subset=['fwd_ret_20d'])
print(f"IC 计算数据: {len(ic_df):,} 行")

factor_names = [n for n in all_names if n != 'fwd_ret_20d']
ic_results = {}

for factor_name in factor_names:
    ic_series = []
    for date, grp in ic_df.groupby(level=0):
        vals = grp.droplevel(0)
        valid = vals[[factor_name, 'fwd_ret_20d']].dropna()
        if len(valid) < 30:
            continue
        ic = valid[factor_name].corr(valid['fwd_ret_20d'], method='spearman')
        ic_series.append({'date': str(date), 'IC': ic})
    
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
# 验证3: PCA 因子正交化效果
# ==================================================================
print("\n" + "=" * 70)
print("验证3: PCA 因子正交化 + 残差正交化效果")
print("=" * 70)

enhance_factors = ['mom_20d', 'bbi_momentum', 'price_pos_52w']
pca_df = ic_df[enhance_factors].dropna()
print(f"Enhance 因子数据: {len(pca_df):,} 行")

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# 逐月做 PCA
pca_explained = []
for date, grp in pca_df.groupby(level=0):
    vals = grp.droplevel(0)
    if len(vals) < 50:
        continue
    scaler = StandardScaler()
    scaled = scaler.fit_transform(vals)
    pca = PCA()
    pca.fit(scaled)
    pca_explained.append({
        'date': str(date),
        'PC1': pca.explained_variance_ratio_[0],
        'PC2': pca.explained_variance_ratio_[1],
        'PC3': pca.explained_variance_ratio_[2],
    })

pca_exp_df = pd.DataFrame(pca_explained)
print(f"\nPCA 解释方差比 (月均值):")
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

# 残差正交化: bbi_momentum 对 mom_20d 回归取残差
print("\n残差正交化 (bbi_momentum ~ mom_20d):")
ortho_ic_list = []
for date, grp in ic_df.groupby(level=0):
    vals = grp.droplevel(0)
    valid = vals[['mom_20d', 'bbi_momentum', 'fwd_ret_20d']].dropna()
    if len(valid) < 50:
        continue
    X = valid['mom_20d'].values.reshape(-1, 1)
    y = valid['bbi_momentum'].values
    reg = LinearRegression().fit(X, y)
    residual = y - reg.predict(X)
    
    ic_orig = np.corrcoef(y, valid['fwd_ret_20d'].values)[0, 1]
    ic_resid = np.corrcoef(residual, valid['fwd_ret_20d'].values)[0, 1]
    ortho_ic_list.append({
        'date': str(date),
        'IC_orig': ic_orig,
        'IC_resid': ic_resid,
        'beta': reg.coef_[0],
        'r2': reg.score(X, y),
    })

ortho_df = pd.DataFrame(ortho_ic_list)
print(f"  回归 β = {ortho_df['beta'].mean():.4f} ± {ortho_df['beta'].std():.4f}")
print(f"  R² = {ortho_df['r2'].mean():.4f} ± {ortho_df['r2'].std():.4f}")
print(f"  bbi_momentum 原始 IC = {ortho_df['IC_orig'].mean():.4f}")
print(f"  正交化后残差 IC = {ortho_df['IC_resid'].mean():.4f}")
ic_retained = abs(ortho_df['IC_resid'].mean()) / abs(ortho_df['IC_orig'].mean()) * 100 if ortho_df['IC_orig'].mean() != 0 else 0
print(f"  IC 保留比例: {ic_retained:.1f}%")

# 也对 price_pos_52w 正交化
print("\n残差正交化 (price_pos_52w ~ mom_20d):")
ortho_ic_list2 = []
for date, grp in ic_df.groupby(level=0):
    vals = grp.droplevel(0)
    valid = vals[['mom_20d', 'price_pos_52w', 'fwd_ret_20d']].dropna()
    if len(valid) < 50:
        continue
    X = valid['mom_20d'].values.reshape(-1, 1)
    y = valid['price_pos_52w'].values
    reg = LinearRegression().fit(X, y)
    residual = y - reg.predict(X)
    
    ic_orig = np.corrcoef(y, valid['fwd_ret_20d'].values)[0, 1]
    ic_resid = np.corrcoef(residual, valid['fwd_ret_20d'].values)[0, 1]
    ortho_ic_list2.append({
        'IC_orig': ic_orig, 'IC_resid': ic_resid,
        'beta': reg.coef_[0], 'r2': reg.score(X, y),
    })

ortho_df2 = pd.DataFrame(ortho_ic_list2)
print(f"  回归 β = {ortho_df2['beta'].mean():.4f} ± {ortho_df2['beta'].std():.4f}")
print(f"  R² = {ortho_df2['r2'].mean():.4f} ± {ortho_df2['r2'].std():.4f}")
print(f"  price_pos_52w 原始 IC = {ortho_df2['IC_orig'].mean():.4f}")
print(f"  正交化后残差 IC = {ortho_df2['IC_resid'].mean():.4f}")
ic_retained2 = abs(ortho_df2['IC_resid'].mean()) / abs(ortho_df2['IC_orig'].mean()) * 100 if ortho_df2['IC_orig'].mean() != 0 else 0
print(f"  IC 保留比例: {ic_retained2:.1f}%")

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

print(f"\n理论最优权重 (归一化后):")
for layer_name in ['alpha', 'risk', 'enhance']:
    s = layer_stats[layer_name]
    w = s['theoretical_weight'] / total_w if total_w > 0 else 0
    print(f"  {layer_name}: {w:.2%}")

print(f"\n当前权重: alpha=0.55, risk=0.20, enhance=0.25")
print(f"理论权重 vs 当前权重对比:")
for layer_name, current in [('alpha', 0.55), ('risk', 0.20), ('enhance', 0.25)]:
    s = layer_stats[layer_name]
    w_theory = s['theoretical_weight'] / total_w if total_w > 0 else 0
    diff = w_theory - current
    direction = "↑应增加" if diff > 0.05 else ("↓应减少" if diff < -0.05 else "≈合适")
    print(f"  {layer_name}: 理论={w_theory:.2%}, 当前={current:.0%}, {direction}")

# ==================================================================
# 验证5: 条件动量 — 按市场状态分组
# ==================================================================
print("\n" + "=" * 70)
print("验证5: 条件动量 — 按市场状态(趋势/震荡)分组 IC")
print("=" * 70)

# 用 CSI 300 近60日收益率趋势判断市场状态
market_fields = ["$close/Ref($close,60)-1"]
market_names = ["market_trend_60d"]
market_df = D.features(D.instruments("stocks_with_ohlcv"), market_fields, start_time="2018-01-01", end_time="2026-03-31")
market_df.columns = market_names
# 用 SH000300 的第一个日期作为市场状态
# 简化：用全市场均值作为 proxy
market_state = market_df.groupby(level=0).mean()

# 合并 IC 数据
trend_ic_list = []
sideways_ic_list = []

for date, grp in ic_df.groupby(level=0):
    vals = grp.droplevel(0)
    valid = vals[['mom_20d', 'fwd_ret_20d']].dropna()
    if len(valid) < 30:
        continue
    
    # 获取市场状态
    if str(date) in market_state.index:
        mt = market_state.loc[str(date), 'market_trend_60d']
    else:
        continue
    
    ic = valid['mom_20d'].corr(valid['fwd_ret_20d'], method='spearman')
    
    if mt > 0.05:  # 上升趋势
        trend_ic_list.append({'date': str(date), 'IC': ic})
    elif mt < -0.05:  # 下降趋势
        sideways_ic_list.append({'date': str(date), 'IC': ic})

trend_df = pd.DataFrame(trend_ic_list)
sideways_df = pd.DataFrame(sideways_ic_list)

if len(trend_df) > 0:
    print(f"\n  上升趋势 (60d ret > 5%):")
    print(f"    月份数: {len(trend_df)}")
    print(f"    mom_20d 平均 IC = {trend_df['IC'].mean():.4f}")
    print(f"    ICIR = {trend_df['IC'].mean() / trend_df['IC'].std():.3f}")

if len(sideways_df) > 0:
    print(f"\n  下降趋势 (60d ret < -5%):")
    print(f"    月份数: {len(sideways_df)}")
    print(f"    mom_20d 平均 IC = {sideways_df['IC'].mean():.4f}")
    print(f"    ICIR = {sideways_df['IC'].mean() / sideways_df['IC'].std():.3f}")

# 中间状态
middle_ic_list = []
for date, grp in ic_df.groupby(level=0):
    vals = grp.droplevel(0)
    valid = vals[['mom_20d', 'fwd_ret_20d']].dropna()
    if len(valid) < 30:
        continue
    if str(date) in market_state.index:
        mt = market_state.loc[str(date), 'market_trend_60d']
    else:
        continue
    ic = valid['mom_20d'].corr(valid['fwd_ret_20d'], method='spearman')
    if -0.05 <= mt <= 0.05:
        middle_ic_list.append({'date': str(date), 'IC': ic})

middle_df = pd.DataFrame(middle_ic_list)
if len(middle_df) > 0:
    print(f"\n  震荡市场 (-5% < 60d ret < 5%):")
    print(f"    月份数: {len(middle_df)}")
    print(f"    mom_20d 平均 IC = {middle_df['IC'].mean():.4f}")
    print(f"    ICIR = {middle_df['IC'].mean() / middle_df['IC'].std():.3f}")

print("\n" + "=" * 70)
print("验证完成")
print("=" * 70)
