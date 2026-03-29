#!/usr/bin/env python3
"""
因子相关性与聚类分析
- 从 Top-K 回测结果筛选夏普≥0.5的因子
- 计算截面相关矩阵
- 层次聚类 + 因子组合优化建议
"""
import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.qlib_init import init_qlib, load_features_safe
from core.universe import filter_instruments

# 使用 $high/$low 的因子列表（可能损坏）
HIGH_LOW_FACTORS = {
    "amplitude", "amplitude_ma20", "ar_indicator", "br_indicator",
    "bull_power", "bear_power", "cr_indicator", "atr_6d", "atr_14d",
    "mfi_proxy",
}

START_DATE = "2024-01-01"
END_DATE = "2026-03-08"


def load_topk_results():
    """加载 Top-K 回测结果，筛选夏普≥0.5的因子"""
    csv_path = PROJECT_ROOT / "results" / "single_factor_topk_2024_2026.csv"
    df = pd.read_csv(csv_path)
    # 筛选夏普≥0.5
    selected = df[df["夏普比率"] >= 0.5].copy()
    selected = selected.sort_values("夏普比率", ascending=False)
    print(f"[筛选] 夏普≥0.5 的因子: {len(selected)} 个（共 {len(df)} 个）")
    print(f"  夏普范围: {selected['夏普比率'].min():.2f} ~ {selected['夏普比率'].max():.2f}")
    return selected


def get_factor_expressions(factor_names):
    """从 factor_scan.py 获取因子表达式，只取需要的"""
    # 复用 factor_scan 的 get_all_factors
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from factor_scan import get_all_factors
    all_factors = get_all_factors()

    selected = {}
    missing = []
    for name in factor_names:
        if name in all_factors:
            selected[name] = all_factors[name]
        else:
            missing.append(name)

    if missing:
        print(f"  [警告] {len(missing)} 个因子无表达式（可能是parquet源）: {missing}")
    return selected


def load_factor_data(factor_exprs, valid_instruments):
    """分批加载因子数据"""
    items = list(factor_exprs.items())
    batch_size = 5
    df_parts = []
    failed = []

    for b in range(0, len(items), batch_size):
        batch = items[b:b + batch_size]
        names = [n for n, _ in batch]
        exprs = [e for _, e in batch]
        try:
            df_part = load_features_safe(
                valid_instruments, exprs,
                start_time=START_DATE, end_time=END_DATE, freq="day"
            )
            df_part.columns = names
            df_parts.append(df_part)
            print(f"  [{min(b + batch_size, len(items))}/{len(items)}] {names} OK")
        except Exception:
            for name, expr in batch:
                try:
                    df_single = load_features_safe(
                        valid_instruments, [expr],
                        start_time=START_DATE, end_time=END_DATE, freq="day"
                    )
                    df_single.columns = [name]
                    df_parts.append(df_single)
                except Exception as e2:
                    failed.append((name, str(e2)[:80]))
                    print(f"  FAIL: {name} - {str(e2)[:80]}")

    if failed:
        print(f"\n失败因子: {len(failed)}")
        for name, err in failed:
            print(f"  {name}: {err}")

    df_all = pd.concat(df_parts, axis=1) if df_parts else pd.DataFrame()
    print(f"\n成功加载: {len(df_all.columns)} 个因子, {len(df_all)} 行")
    return df_all


def compute_cross_sectional_corr(df_factors):
    """计算因子截面相关性（按日期取平均 Spearman 相关）"""
    factors = list(df_factors.columns)
    n = len(factors)
    print(f"\n[相关性] 计算 {n} 个因子的截面相关矩阵...")

    # Index is (instrument, datetime) - groupby datetime level
    grouped = df_factors.groupby(level="datetime")
    total_dates = len(grouped)
    print(f"  日期数: {total_dates}")

    # 按日期计算截面 rank 相关，取平均
    corr_sum = np.zeros((n, n))
    valid_count = np.zeros((n, n))

    processed = 0
    for dt, cross in grouped:
        if len(cross) < 50:
            continue
        # rank 转换
        ranked = cross.rank(pct=True)
        # 计算相关
        c = ranked.corr(method="pearson")  # rank后pearson ≈ spearman
        if c.shape == (n, n):
            vals = c.values
            mask = ~np.isnan(vals)
            corr_sum[mask] += vals[mask]
            valid_count[mask] += 1
        processed += 1

    if processed == 0:
        print("  [错误] 无有效截面数据")
        return pd.DataFrame()

    # 避免除0
    valid_count[valid_count == 0] = 1
    corr_mean = corr_sum / valid_count
    corr_df = pd.DataFrame(corr_mean, index=factors, columns=factors)
    # 对角线设为 1
    np.fill_diagonal(corr_df.values, 1.0)
    print(f"  有效截面: {processed} 天")
    return corr_df


def hierarchical_clustering(corr_df, threshold=0.8):
    """层次聚类，识别高相关因子组"""
    print(f"\n[聚类] 层次聚类 (|corr| > {threshold} 为同组)...")

    # 距离矩阵 = 1 - |corr|
    dist_matrix = 1 - corr_df.abs().values
    np.fill_diagonal(dist_matrix, 0)
    # 确保对称且非负
    dist_matrix = (dist_matrix + dist_matrix.T) / 2
    dist_matrix = np.maximum(dist_matrix, 0)

    condensed = squareform(dist_matrix)
    Z = linkage(condensed, method="complete")

    # 按 threshold 切割：距离 < (1 - threshold) 的归为一组
    labels = fcluster(Z, t=1 - threshold, criterion="distance")

    factors = list(corr_df.index)
    clusters = {}
    for factor, label in zip(factors, labels):
        clusters.setdefault(label, []).append(factor)

    return clusters, Z, labels


def print_correlation_heatmap(corr_df):
    """用文本方式打印相关矩阵（简化版）"""
    print("\n" + "=" * 100)
    print("  因子截面相关矩阵 (平均 Spearman)")
    print("=" * 100)

    factors = list(corr_df.index)
    n = len(factors)

    # 缩短因子名
    short_names = []
    for f in factors:
        s = f[:16]
        short_names.append(s)

    # 打印表头
    header = f"{'因子':<18}" + "".join(f"{s[:8]:>9}" for s in short_names)
    print(header)
    print("-" * len(header))

    for i, (factor, short) in enumerate(zip(factors, short_names)):
        row = f"{short:<18}"
        for j in range(n):
            val = corr_df.iloc[i, j]
            if i == j:
                row += f"{'1.00':>9}"
            elif abs(val) >= 0.8:
                row += f"{val:>8.2f}*"
            elif abs(val) >= 0.5:
                row += f"{val:>8.2f}+"
            else:
                row += f"{val:>9.2f}"
        print(row)

    print("\n  * = |corr| ≥ 0.8 (高度相关)")
    print("  + = |corr| ≥ 0.5 (中度相关)")


def print_high_corr_pairs(corr_df, threshold=0.8):
    """列出所有高相关因子对"""
    print(f"\n[高相关因子对] |corr| ≥ {threshold}:")
    print("-" * 70)
    factors = list(corr_df.index)
    pairs = []
    for i in range(len(factors)):
        for j in range(i + 1, len(factors)):
            val = corr_df.iloc[i, j]
            if abs(val) >= threshold:
                pairs.append((factors[i], factors[j], val))
    pairs.sort(key=lambda x: -abs(x[2]))

    if not pairs:
        print("  无高相关因子对")
    else:
        for f1, f2, v in pairs:
            print(f"  {f1:<25} <-> {f2:<25} corr={v:+.3f}")
    print(f"\n  共 {len(pairs)} 对")
    return pairs


def optimize_combination(clusters, topk_results, corr_df):
    """从每个聚类中选夏普最高的代表因子"""
    print("\n" + "=" * 100)
    print("  因子组合优化建议")
    print("=" * 100)

    # 建立因子→夏普的映射
    sharpe_map = dict(zip(topk_results["因子"], topk_results["夏普比率"]))
    direction_map = dict(zip(topk_results["因子"], topk_results["方向"]))
    annual_return_map = dict(zip(topk_results["因子"], topk_results["年化收益"]))

    representatives = []

    # 按聚类大小降序
    sorted_clusters = sorted(clusters.items(), key=lambda x: -len(x[1]))

    for cid, members in sorted_clusters:
        # 按夏普排序
        scored = [(f, sharpe_map.get(f, 0)) for f in members]
        scored.sort(key=lambda x: -x[1])
        best = scored[0]
        representatives.append(best[0])

        hl_mark = lambda f: " [!$high/$low]" if f in HIGH_LOW_FACTORS else ""
        print(f"\n  聚类 {cid} ({len(members)} 个因子):")
        for f, s in scored:
            marker = " <-- 代表" if f == best[0] else ""
            dir_str = direction_map.get(f, "?")
            ret_str = annual_return_map.get(f, 0)
            print(f"    {f:<28} 夏普={s:>6.2f}  年化={ret_str:>8.2f}  方向={dir_str}{hl_mark(f)}{marker}")

    # 验证代表因子之间的互相关
    print(f"\n{'─' * 70}")
    print(f"  推荐因子组合 ({len(representatives)} 个代表因子):")
    print(f"{'─' * 70}")

    # 只保留在 corr_df 中存在的代表因子
    valid_reps = [f for f in representatives if f in corr_df.index]

    for f in valid_reps:
        s = sharpe_map.get(f, 0)
        d = direction_map.get(f, "?")
        r = annual_return_map.get(f, 0)
        hl = " [!$high/$low可能损坏]" if f in HIGH_LOW_FACTORS else ""
        print(f"    {f:<28} 夏普={s:>6.2f}  年化={r:>8.2f}  方向={d}{hl}")

    # 代表因子互相关矩阵
    if len(valid_reps) > 1:
        sub_corr = corr_df.loc[valid_reps, valid_reps]
        print(f"\n  代表因子互相关矩阵:")
        short = [f[:14] for f in valid_reps]
        header = f"{'':>16}" + "".join(f"{s:>16}" for s in short)
        print(header)
        for i, f in enumerate(valid_reps):
            row = f"{short[i]:>16}"
            for j in range(len(valid_reps)):
                val = sub_corr.iloc[i, j]
                if i == j:
                    row += f"{'1.00':>16}"
                else:
                    row += f"{val:>16.3f}"
            print(row)

        # 检查是否有代表因子之间高相关
        issues = []
        for i in range(len(valid_reps)):
            for j in range(i + 1, len(valid_reps)):
                val = sub_corr.iloc[i, j]
                if abs(val) >= 0.5:
                    issues.append((valid_reps[i], valid_reps[j], val))

        if issues:
            print(f"\n  [!] 代表因子间仍有中高相关:")
            for f1, f2, v in issues:
                print(f"      {f1} <-> {f2}: {v:+.3f}")
            print("  建议：可考虑进一步精简或降权处理")
        else:
            print(f"\n  [OK] 代表因子间互相关均 < 0.5，组合分散度良好")

    return valid_reps


def main():
    import time
    start = time.time()

    # 1. 加载 Top-K 结果，筛选夏普≥0.5
    topk_results = load_topk_results()
    factor_names = topk_results["因子"].tolist()

    print(f"\n筛选因子列表:")
    for i, row in topk_results.iterrows():
        hl = " [!$high/$low]" if row["因子"] in HIGH_LOW_FACTORS else ""
        print(f"  {row['因子']:<28} 夏普={row['夏普比率']:>6.2f}  年化={row['年化收益']:>8.2f}  方向={row['方向']}{hl}")

    # 2. 获取因子表达式并加载数据
    factor_exprs = get_factor_expressions(factor_names)
    print(f"\n[数据加载] 初始化 Qlib...")
    init_qlib()
    from qlib.data import D

    print("[数据加载] 获取股票列表...")
    instruments = D.instruments(market="all")
    df_close = load_features_safe(
        instruments, ["$close"],
        start_time=START_DATE, end_time=END_DATE, freq="day"
    )
    valid = filter_instruments(
        df_close.index.get_level_values("instrument").unique().tolist()
    )
    print(f"  股票数: {len(valid)}")

    print("\n[数据加载] 加载因子数据...")
    df_factors = load_factor_data(factor_exprs, valid)

    if df_factors.empty:
        print("[错误] 无因子数据，退出")
        return

    # 3. 计算截面相关矩阵
    corr_df = compute_cross_sectional_corr(df_factors)
    if corr_df.empty:
        print("[错误] 无法计算相关矩阵，退出")
        return

    # 打印相关矩阵
    print_correlation_heatmap(corr_df)

    # 列出高相关因子对
    high_pairs = print_high_corr_pairs(corr_df, threshold=0.8)

    # 4. 层次聚类
    clusters, Z, labels = hierarchical_clustering(corr_df, threshold=0.8)

    print(f"\n  聚类结果 ({len(clusters)} 组):")
    for cid, members in sorted(clusters.items(), key=lambda x: -len(x[1])):
        print(f"    组{cid} ({len(members)}个): {', '.join(members)}")

    # 5. 因子组合优化
    recommended = optimize_combination(clusters, topk_results, corr_df)

    # 6. 保存结果
    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    corr_path = out_dir / "factor_correlation_matrix.csv"
    corr_df.to_csv(corr_path, float_format="%.4f")
    print(f"\n[保存] 相关矩阵: {corr_path}")

    cluster_rows = []
    for cid, members in clusters.items():
        for m in members:
            cluster_rows.append({"cluster": cid, "factor": m, "is_representative": m in recommended})
    cluster_df = pd.DataFrame(cluster_rows)
    cluster_path = out_dir / "factor_clusters.csv"
    cluster_df.to_csv(cluster_path, index=False)
    print(f"[保存] 聚类结果: {cluster_path}")

    elapsed = time.time() - start
    print(f"\n总耗时: {elapsed:.1f}秒")


if __name__ == "__main__":
    main()
