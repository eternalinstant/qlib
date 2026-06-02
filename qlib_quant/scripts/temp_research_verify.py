"""A股因子市值中性化验证 - 完整版"""
import warnings
warnings.filterwarnings('ignore')
import sys
import os
import time
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA

def log(msg):
    print(msg, flush=True)

t0 = time.time()
log("=" * 70)
log("A股因子市值中性化验证报告")
log("=" * 70)

# ============================================================
# 1. 加载数据
# ============================================================
log("\n[1] 加载数据...")

cal = pd.read_csv('data/qlib_data/cn_data/calendars/day.txt', header=None, names=['date'])
cal['date'] = pd.to_datetime(cal['date'])
cal_index = cal['date'].values

factor_df = pd.read_parquet('data/qlib_data/cn_data/factor_data.parquet')
factor_df['datetime'] = pd.to_datetime(factor_df['datetime'])

feat_dir = 'data/qlib_data/cn_data/features'
stock_dirs = sorted(os.listdir(feat_dir))

close_frames = []
for i, stock in enumerate(stock_dirs):
    bin_path = os.path.join(feat_dir, stock, 'close.day.bin')
    if not os.path.exists(bin_path):
        continue
    with open(bin_path, 'rb') as f:
        data = f.read()
    if len(data) < 8:
        continue
    n_values = len(data) // 4 - 1
    start_idx = int(np.frombuffer(data[:4], dtype='<f4')[0])
    values = np.frombuffer(data[4:], dtype='<f4')[:n_values]
    dates = cal_index[start_idx:start_idx + n_values]
    if len(dates) != n_values:
        continue
    df = pd.DataFrame({'date': dates, 'close': values, 'instrument': stock})
    close_frames.append(df)

close_df = pd.concat(close_frames, ignore_index=True)

def feat_to_factor(s):
    s = s.lower()
    for prefix in ['sh', 'sz', 'bj']:
        if s.startswith(prefix):
            return s[len(prefix):] + prefix
    return s

close_df['instrument'] = close_df['instrument'].apply(feat_to_factor)
close_df.rename(columns={'date': 'datetime'}, inplace=True)
close_df['datetime'] = pd.to_datetime(close_df['datetime'])

merged = factor_df.merge(close_df[['instrument', 'datetime', 'close']], on=['instrument', 'datetime'], how='left')
merged = merged[merged['datetime'] >= '2019-01-01'].copy()
merged = merged.sort_values(['instrument', 'datetime']).reset_index(drop=True)
log(f"  合并后: {merged.shape[0]:,} 行, close非空: {merged['close'].notna().sum():,}")

# ============================================================
# 2. 计算技术因子
# ============================================================
log("\n[2] 计算技术因子...")

merged['mom_20d'] = merged.groupby('instrument')['close'].pct_change(20, fill_method=None)

for w in [3, 6, 12, 24]:
    merged[f'_ma_{w}'] = merged.groupby('instrument')['close'].transform(
        lambda x: x.rolling(w, min_periods=w).mean()
    )
merged['bbi_momentum'] = merged['close'] / (
    (merged['_ma_3'] + merged['_ma_6'] + merged['_ma_12'] + merged['_ma_24']) / 4
) - 1

merged['_min252'] = merged.groupby('instrument')['close'].transform(
    lambda x: x.rolling(252, min_periods=1).min()
)
merged['_max252'] = merged.groupby('instrument')['close'].transform(
    lambda x: x.rolling(252, min_periods=1).max()
)
merged['price_pos_52w'] = (merged['close'] - merged['_min252']) / (merged['_max252'] - merged['_min252'] + 1e-8)

_daily_ret = merged.groupby('instrument')['close'].pct_change(1, fill_method=None)
merged['vol_std_20d'] = _daily_ret.groupby(merged['instrument']).transform(
    lambda x: x.rolling(20, min_periods=20).std()
)

merged['fwd_ret_20d'] = merged.groupby('instrument')['close'].pct_change(20, fill_method=None).shift(-20)

merged.drop(columns=['_ma_3', '_ma_6', '_ma_12', '_ma_24', '_min252', '_max252'], inplace=True, errors='ignore')

log(f"  因子计算完成, 耗时 {time.time()-t0:.1f}s")

mv_col = 'total_mv'
check_factors = ['roa_fina', 'book_to_market', 'ebit_to_mv', 'ocf_to_ev', 'retained_earnings',
                 'turnover_rate_f', 'mom_20d', 'bbi_momentum', 'price_pos_52w', 'vol_std_20d']

# ============================================================
# 验证1: 各因子与市值的相关性
# ============================================================
log("\n" + "=" * 70)
log("验证1: 各因子与市值的相关性 (Spearman, 截面平均)")
log("=" * 70)

mv_corr_results = {}
for fc in check_factors:
    if fc not in merged.columns:
        continue
    corrs = []
    for dt, grp in merged.groupby('datetime'):
        valid = grp[[fc, mv_col]].dropna()
        if len(valid) < 30:
            continue
        r, _ = stats.spearmanr(valid[fc], valid[mv_col])
        if not np.isnan(r):
            corrs.append(r)
    if corrs:
        arr = np.array(corrs)
        mean_r = np.mean(arr)
        se = np.std(arr, ddof=1) / np.sqrt(len(arr))
        t_stat = mean_r / se
        significance = "***" if abs(t_stat) > 3.29 else "**" if abs(t_stat) > 2.58 else "*" if abs(t_stat) > 1.96 else "n.s."
        flag = "NEED" if abs(mean_r) > 0.1 and abs(t_stat) > 1.96 else "OK"
        mv_corr_results[fc] = {'mean_corr': mean_r, 't_stat': t_stat, 'flag': flag}
        log(f"  {fc:25s}: r={mean_r:+.4f}, t={t_stat:+7.2f} {significance:>4s}  {flag}")

# ============================================================
# 验证2: bbi_momentum vs mom_20d
# ============================================================
log("\n" + "=" * 70)
log("验证2: bbi_momentum vs mom_20d 冗余检查")
log("=" * 70)

corrs_bm = []
for dt, grp in merged.groupby('datetime'):
    valid = grp[['bbi_momentum', 'mom_20d']].dropna()
    if len(valid) < 30:
        continue
    r, _ = stats.spearmanr(valid['bbi_momentum'], valid['mom_20d'])
    if not np.isnan(r):
        corrs_bm.append(r)

if corrs_bm:
    arr = np.array(corrs_bm)
    mean_c = np.mean(arr)
    t_s = mean_c / (np.std(arr, ddof=1) / np.sqrt(len(arr)))
    log(f"  平均 Spearman 相关: {mean_c:.4f}")
    log(f"  t 统计量: {t_s:.2f} ({len(arr)} 期)")
    log(f"  分布: min={arr.min():.4f}, p25={np.percentile(arr,25):.4f}, med={np.median(arr):.4f}, p75={np.percentile(arr,75):.4f}, max={arr.max():.4f}")
    log(f"  r>0.7: {(arr > 0.7).mean():.1%}, r>0.5: {(arr > 0.5).mean():.1%}")
    if mean_c > 0.7:
        log(f"  结论: FAIL - 高度冗余 (r={mean_c:.3f})")
    elif mean_c > 0.5:
        log(f"  结论: MIXED - 中度冗余 (r={mean_c:.3f})")
    else:
        log(f"  结论: PASS (r={mean_c:.3f})")
else:
    log("  无有效数据")

# ============================================================
# 验证3: 因子 IC
# ============================================================
log("\n" + "=" * 70)
log("验证3: 因子 IC 与稳定性")
log("=" * 70)

ic_results = {}
for fc in check_factors:
    if fc not in merged.columns:
        continue
    ics = []
    for dt, grp in merged.groupby('datetime'):
        valid = grp[[fc, 'fwd_ret_20d']].dropna()
        if len(valid) < 30:
            continue
        r, _ = stats.spearmanr(valid[fc], valid['fwd_ret_20d'])
        if not np.isnan(r):
            ics.append(r)
    
    if not ics:
        log(f"  {fc}: 无有效 IC 数据")
        continue
    arr = np.array(ics)
    mean_ic = np.mean(arr)
    ic_std = np.std(arr)
    icir = mean_ic / ic_std if ic_std > 0 else 0
    t_stat = mean_ic / (ic_std / np.sqrt(len(arr)))
    ic_pos = (arr > 0).mean()
    
    ic_results[fc] = {'mean_ic': mean_ic, 'icir': icir, 't_stat': t_stat, 'n': len(arr), 'ic_pos': ic_pos}
    sig = "***" if abs(t_stat) > 3.29 else "**" if abs(t_stat) > 2.58 else "*" if abs(t_stat) > 1.96 else "n.s."
    log(f"  {fc:25s}: IC={mean_ic:+.4f}, ICIR={icir:+.3f}, t={t_stat:+.2f} {sig:>4s}, IC>0={ic_pos:.1%} ({len(arr)}期)")

# ============================================================
# 验证4: 市值中性化前后 IC
# ============================================================
log("\n" + "=" * 70)
log("验证4: 市值中性化前后 IC 对比 (OLS log_mv 残差法)")
log("=" * 70)

neutral_factors = ['roa_fina', 'book_to_market', 'ebit_to_mv', 'ocf_to_ev', 
                   'retained_earnings', 'turnover_rate_f', 'mom_20d', 'bbi_momentum']

log(f"  {'因子':25s} {'IC前':>8s} {'ICIR前':>8s} {'IC后':>8s} {'ICIR后':>8s} {'ΔICIR':>8s} {'t_diff':>8s}")
log(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

for fc in neutral_factors:
    if fc not in merged.columns:
        continue
    ic_b, ic_a = [], []
    for dt, grp in merged.groupby('datetime'):
        valid = grp[[fc, 'fwd_ret_20d', mv_col]].dropna()
        if len(valid) < 50:
            continue
        rb, _ = stats.spearmanr(valid[fc], valid['fwd_ret_20d'])
        if np.isnan(rb):
            continue
        log_mv = np.log(valid[mv_col].values)
        X = np.column_stack([np.ones(len(log_mv)), log_mv])
        y = valid[fc].values
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        res = y - X @ beta
        ra, _ = stats.spearmanr(res, valid['fwd_ret_20d'])
        ic_b.append(rb)
        ic_a.append(ra)
    
    mb, ma = np.mean(ic_b), np.mean(ic_a)
    sb, sa = np.std(ic_b), np.std(ic_a)
    ir_b = mb/sb if sb > 0 else 0
    ir_a = ma/sa if sa > 0 else 0
    delta = ir_a - ir_b
    diff = np.array(ic_a) - np.array(ic_b)
    td = np.mean(diff) / (np.std(diff, ddof=1) / np.sqrt(len(diff)))
    log(f"  {fc:25s} {mb:+8.4f} {ir_b:+8.3f} {ma:+8.4f} {ir_a:+8.3f} {delta:+8.3f} {td:+8.2f}")

# ============================================================
# 验证5: Grinold 层间权重
# ============================================================
log("\n" + "=" * 70)
log("验证5: Grinold 理论最优层间权重 (w ∝ IR × √N)")
log("=" * 70)

alpha_ics = [abs(ic_results.get(f, {}).get('icir', 0)) for f in ['roa_fina', 'book_to_market', 'ebit_to_mv', 'ocf_to_ev', 'retained_earnings'] if f in ic_results]
risk_ics = [abs(ic_results.get(f, {}).get('icir', 0)) for f in ['turnover_rate_f', 'vol_std_20d'] if f in ic_results]
enhance_ics = [abs(ic_results.get(f, {}).get('icir', 0)) for f in ['mom_20d', 'bbi_momentum', 'price_pos_52w'] if f in ic_results]

log(f"  Alpha 层 |ICIR|: {alpha_ics}")
log(f"  Risk 层 |ICIR|: {risk_ics}")
log(f"  Enhance 层 |ICIR|: {enhance_ics}")

def grinold(icir_list):
    return np.mean(icir_list) * np.sqrt(len(icir_list))

w_a, w_r, w_e = grinold(alpha_ics), grinold(risk_ics), grinold(enhance_ics)
total = w_a + w_r + w_e
wa, wr, we = w_a/total, w_r/total, w_e/total

log(f"\n  理论最优: Alpha={wa:.3f}, Risk={wr:.3f}, Enhance={we:.3f}")
log(f"  当前配置: Alpha=0.55, Risk=0.20, Enhance=0.25")
log(f"  差异:     Alpha={wa-0.55:+.3f}, Risk={wr-0.20:+.3f}, Enhance={we-0.25:+.3f}")

# ============================================================
# 验证6: PCA
# ============================================================
log("\n" + "=" * 70)
log("验证6: PCA 因子正交化")
log("=" * 70)

pca_factors = ['roa_fina', 'book_to_market', 'ebit_to_mv', 'ocf_to_ev', 'retained_earnings',
               'turnover_rate_f', 'mom_20d', 'bbi_momentum', 'price_pos_52w', 'vol_std_20d']

recent = merged[merged['datetime'] >= '2025-01-01']
all_std = []
for dt, grp in recent.groupby('datetime'):
    valid = grp[pca_factors].dropna()
    if len(valid) < 50:
        continue
    s = (valid - valid.mean()) / valid.std()
    all_std.append(s)

combined = pd.concat(all_std, ignore_index=True).dropna()
pca = PCA()
pca.fit(combined)

ev = pca.explained_variance_ratio_
cv = np.cumsum(ev)
log(f"  前10主成分解释方差:")
for i in range(min(10, len(ev))):
    log(f"    PC{i+1}: {ev[i]:.4f} (累计: {cv[i]:.4f})")

log(f"\n  PC1 解释: {ev[0]:.2%}, 前3累计: {cv[2]:.2%}, 前5累计: {cv[4]:.2%}")

log(f"\n  PC1 因子载荷 (判断市值主导):")
for i, fc in enumerate(pca_factors):
    m = " <-- SIZE" if abs(pca.components_[0][i]) > 0.3 else ""
    log(f"    {fc:25s}: {pca.components_[0][i]:+.4f}{m}")

log(f"\n总耗时: {time.time()-t0:.1f}s")
log("=" * 70)
log("全部验证完成")
log("=" * 70)
