#!/usr/bin/env python3
"""
周频因子相关性去重

基于 weekly_factor_rank 的结果文件，读取前 N 个周频因子表达式，
计算平均截面 rank correlation，并按原排名做贪心去重。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config.config import CONFIG
from core.qlib_init import init_qlib, load_features_safe
from core.universe import filter_instruments
from qlib.data import D


PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
RESULTS_DIR = PROJECT_ROOT / "results"


def _resolve_end_date(value: str) -> str:
    if not value or value == "auto":
        return pd.Timestamp.today().strftime("%Y-%m-%d")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="results/weekly_factor_rank_2019_2026_k15_b20_c2_s1_ws5.csv",
        help="周频因子排行 CSV 路径",
    )
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default=_resolve_end_date(CONFIG.get("end_date", "auto")))
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--corr-threshold", type=float, default=0.8)
    parser.add_argument("--min-cross-section", type=int, default=50)
    parser.add_argument("--output-tag", default="")
    return parser.parse_args()


def _load_leaders(path: Path, top_n: int) -> pd.DataFrame:
    df = pd.read_csv(path).sort_values("排名").head(top_n).copy()
    required = {"排名", "因子", "方向", "夏普比率", "年化收益", "周频表达式"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"结果文件缺少列: {sorted(missing)}")
    return df


def _load_factor_values(factor_df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    init_qlib()
    instruments = D.instruments(market="all")
    inst_list = list(D.list_instruments(instruments, start, end).keys())
    inst_list = filter_instruments(inst_list, exclude_st=False)

    names = factor_df["因子"].tolist()
    exprs = factor_df["周频表达式"].tolist()
    df = load_features_safe(inst_list, exprs, start_time=start, end_time=end, freq="day")
    df.columns = names
    return df


def _average_rank_corr(df: pd.DataFrame, min_cross_section: int) -> tuple[pd.DataFrame, int]:
    factor_names = list(df.columns)
    corr_sum = np.zeros((len(factor_names), len(factor_names)))
    valid_count = np.zeros((len(factor_names), len(factor_names)))
    used_dates = 0

    for _, cross in df.groupby(level="datetime"):
        if len(cross) < min_cross_section:
            continue
        ranked = cross.rank(pct=True)
        corr = ranked.corr(method="pearson").values
        mask = ~np.isnan(corr)
        corr_sum[mask] += corr[mask]
        valid_count[mask] += 1
        used_dates += 1

    valid_count[valid_count == 0] = 1
    avg_corr = corr_sum / valid_count
    corr_df = pd.DataFrame(avg_corr, index=factor_names, columns=factor_names)
    np.fill_diagonal(corr_df.values, 1.0)
    return corr_df, used_dates


def _select_candidates(leaders: pd.DataFrame, corr_df: pd.DataFrame, threshold: float) -> tuple[list[str], list[tuple[str, str, float]]]:
    factor_names = leaders["因子"].tolist()
    high_pairs: list[tuple[str, str, float]] = []
    for i in range(len(factor_names)):
        for j in range(i + 1, len(factor_names)):
            value = float(corr_df.iloc[i, j])
            if abs(value) >= threshold:
                high_pairs.append((factor_names[i], factor_names[j], value))

    high_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    selected: list[str] = []
    for _, row in leaders.sort_values("排名").iterrows():
        factor = row["因子"]
        if all(abs(float(corr_df.loc[factor, kept])) < threshold for kept in selected):
            selected.append(factor)

    return selected, high_pairs


def _output_paths(source_path: Path, top_n: int, tag: str) -> tuple[Path, Path, Path]:
    stem = source_path.stem
    suffix = f"_top{top_n}"
    if tag:
        suffix += f"_{tag}"
    corr_csv = RESULTS_DIR / f"{stem}_corr{suffix}.csv"
    candidates_csv = RESULTS_DIR / f"{stem}_candidates{suffix}.csv"
    summary_md = RESULTS_DIR / f"{stem}_candidates{suffix}.md"
    return corr_csv, candidates_csv, summary_md


def _write_summary(
    leaders: pd.DataFrame,
    selected: list[str],
    high_pairs: list[tuple[str, str, float]],
    used_dates: int,
    threshold: float,
    summary_path: Path,
) -> None:
    lines = [
        "# 周频因子相关性去重摘要",
        "",
        f"- 排行样本: 前 {len(leaders)} 名周频因子",
        f"- 平均截面相关性阈值: {threshold:.2f}",
        f"- 使用截面日期数: {used_dates}",
        "",
        "## 入选候选池",
        "",
    ]

    chosen = leaders[leaders["因子"].isin(selected)].sort_values("排名")
    for _, row in chosen.iterrows():
        lines.append(
            f"- {row['因子']} | {row['方向']} | Sharpe {row['夏普比率']:.3f} | Annual {row['年化收益']:.2%}"
        )

    if high_pairs:
        lines.extend(["", "## 高相关因子对", ""])
        for a, b, value in high_pairs[:20]:
            lines.append(f"- {a} vs {b}: {value:.3f}")

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    source_path = Path(args.input)
    if not source_path.is_absolute():
        source_path = PROJECT_ROOT / source_path

    leaders = _load_leaders(source_path, args.top_n)
    print(f"[leaders] {leaders['因子'].tolist()}")

    df = _load_factor_values(leaders, args.start, args.end)
    print(f"[loaded] {df.shape}")

    corr_df, used_dates = _average_rank_corr(df, min_cross_section=args.min_cross_section)
    selected, high_pairs = _select_candidates(leaders, corr_df, threshold=args.corr_threshold)

    leaders["是否入选"] = leaders["因子"].isin(selected)

    corr_csv, candidates_csv, summary_md = _output_paths(source_path, args.top_n, args.output_tag.strip())
    corr_df.to_csv(corr_csv, float_format="%.6f")
    leaders.to_csv(candidates_csv, index=False, encoding="utf-8-sig", float_format="%.6f")
    _write_summary(leaders, selected, high_pairs, used_dates, args.corr_threshold, summary_md)

    print("\n[selected]")
    for factor in selected:
        row = leaders.loc[leaders["因子"] == factor].iloc[0]
        print(f"{factor},{row['方向']},{row['夏普比率']:.3f},{row['年化收益']:.4f}")

    print("\n[high_corr_pairs]")
    for a, b, value in high_pairs[:20]:
        print(f"{a},{b},{value:.3f}")

    print(f"\n[corr_csv] {corr_csv}")
    print(f"[candidates_csv] {candidates_csv}")
    print(f"[summary_md] {summary_md}")
    print(f"[used_dates] {used_dates}")


if __name__ == "__main__":
    main()
