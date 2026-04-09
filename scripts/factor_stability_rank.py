#!/usr/bin/env python3
"""
正式单因子稳定性排行

定义：
- 使用 formal_single_factor_results.csv 中已经跑完的正式单因子结果
- 从每个 selection_file 读取相邻调仓日的持仓集合
- 以相邻两期持仓集合的 Jaccard 重合度作为主稳定性指标
- 同时输出保留率 / 替换率 / 平均新增和移除数量

输出：
- results/formal_single_factor_stability.csv
- results/formal_single_factor_stability_day_top10.csv
- results/formal_single_factor_stability_week_top10.csv
- results/formal_single_factor_stability_summary.md
"""

from __future__ import annotations

from pathlib import Path
from statistics import mean, median

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

FORMAL_RESULTS_CSV = RESULTS_DIR / "formal_single_factor_results.csv"
STABILITY_CSV = RESULTS_DIR / "formal_single_factor_stability.csv"
DAY_TOP10_CSV = RESULTS_DIR / "formal_single_factor_stability_day_top10.csv"
WEEK_TOP10_CSV = RESULTS_DIR / "formal_single_factor_stability_week_top10.csv"
SUMMARY_MD = RESULTS_DIR / "formal_single_factor_stability_summary.md"


def _load_holdings(selection_file: Path) -> list[set[str]]:
    df = pd.read_csv(selection_file, usecols=["date", "symbol"])
    if df.empty:
        return []
    grouped = df.groupby("date", sort=True)["symbol"].agg(lambda s: set(map(str, s)))
    return grouped.tolist()


def _calc_metrics(holdings: list[set[str]]) -> dict[str, float]:
    jaccard_list: list[float] = []
    retain_prev_list: list[float] = []
    replace_ratio_list: list[float] = []
    added_count_list: list[int] = []
    removed_count_list: list[int] = []
    bucket_size_list: list[int] = []

    for prev_set, curr_set in zip(holdings, holdings[1:]):
        inter = len(prev_set & curr_set)
        union = len(prev_set | curr_set)
        jaccard = inter / union if union else 1.0
        retain_prev = inter / len(prev_set) if prev_set else 1.0
        replace_ratio = 1.0 - retain_prev

        jaccard_list.append(jaccard)
        retain_prev_list.append(retain_prev)
        replace_ratio_list.append(replace_ratio)
        added_count_list.append(len(curr_set - prev_set))
        removed_count_list.append(len(prev_set - curr_set))
        bucket_size_list.append(len(curr_set))

    return {
        "mean_jaccard": mean(jaccard_list),
        "median_jaccard": median(jaccard_list),
        "min_jaccard": min(jaccard_list),
        "mean_retain_prev": mean(retain_prev_list),
        "mean_replace_ratio": mean(replace_ratio_list),
        "mean_added_count": mean(added_count_list),
        "mean_removed_count": mean(removed_count_list),
        "avg_bucket_size": mean(bucket_size_list),
        "n_rebalance_steps": len(jaccard_list),
        "n_rebalance_dates": len(holdings),
    }


def build_stability_table() -> pd.DataFrame:
    results_df = pd.read_csv(FORMAL_RESULTS_CSV)
    rows: list[dict[str, object]] = []

    for item in results_df.itertuples(index=False):
        if getattr(item, "status") != "success":
            continue
        selection_file = PROJECT_ROOT / str(getattr(item, "selection_file"))
        if not selection_file.exists():
            continue

        holdings = _load_holdings(selection_file)
        if len(holdings) < 2:
            continue

        metrics = _calc_metrics(holdings)
        rows.append(
            {
                "freq": getattr(item, "freq"),
                "factor_name": getattr(item, "factor_name"),
                "direction": getattr(item, "direction"),
                "strategy_name": getattr(item, "strategy_name"),
                "strategy_yaml": getattr(item, "strategy_yaml"),
                "selection_file": getattr(item, "selection_file"),
                "annual_return": getattr(item, "annual_return"),
                "sharpe_ratio": getattr(item, "sharpe_ratio"),
                "max_drawdown": getattr(item, "max_drawdown"),
                **metrics,
            }
        )

    df = pd.DataFrame(rows)
    return df.sort_values(
        by=[
            "mean_jaccard",
            "median_jaccard",
            "mean_retain_prev",
            "mean_replace_ratio",
            "min_jaccard",
        ],
        ascending=[False, False, False, True, False],
        kind="mergesort",
    ).reset_index(drop=True)


def _dedupe_by_factor(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(subset=["freq", "factor_name"], keep="first").reset_index(drop=True)


def _fmt_pct(v: float) -> str:
    return f"{v:.2%}"


def _write_summary(full_df: pd.DataFrame, day_top10: pd.DataFrame, week_top10: pd.DataFrame) -> None:
    lines = [
        "# 正式单因子稳定性排行",
        "",
        "口径：",
        "- 只使用 `results/formal_single_factor_results.csv` 中 `status=success` 的正式单因子结果",
        "- 主指标是相邻调仓日持仓集合的 `mean_jaccard`",
        "- `mean_replace_ratio = 1 - mean_retain_prev`，表示平均有多少上一期持仓被替换掉",
        "",
        f"- 总策略数：`{len(full_df)}`",
        f"- 日频唯一因子数：`{full_df[full_df['freq'] == 'day']['factor_name'].nunique()}`",
        f"- 周频唯一因子数：`{full_df[full_df['freq'] == 'week']['factor_name'].nunique()}`",
        "",
        "## 日频 Top 10",
        "",
        day_top10.to_markdown(index=False),
        "",
        "## 周频 Top 10",
        "",
        week_top10.to_markdown(index=False),
        "",
    ]
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    full_df = build_stability_table()
    full_df.to_csv(STABILITY_CSV, index=False)

    display_cols = [
        "factor_name",
        "direction",
        "mean_jaccard",
        "mean_retain_prev",
        "mean_replace_ratio",
        "mean_added_count",
        "mean_removed_count",
        "annual_return",
        "sharpe_ratio",
        "strategy_name",
    ]

    day_top10 = _dedupe_by_factor(full_df[full_df["freq"] == "day"]).head(10).copy()
    week_top10 = _dedupe_by_factor(full_df[full_df["freq"] == "week"]).head(10).copy()

    for frame in (day_top10, week_top10):
        for col in ["mean_jaccard", "mean_retain_prev", "mean_replace_ratio", "annual_return"]:
            frame[col] = frame[col].map(_fmt_pct)
        for col in ["mean_added_count", "mean_removed_count", "sharpe_ratio"]:
            frame[col] = frame[col].map(lambda v: f"{v:.4f}")

    day_top10[display_cols].to_csv(DAY_TOP10_CSV, index=False)
    week_top10[display_cols].to_csv(WEEK_TOP10_CSV, index=False)
    _write_summary(full_df, day_top10[display_cols], week_top10[display_cols])


if __name__ == "__main__":
    main()
