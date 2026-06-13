#!/usr/bin/env python3
"""
年度拆分回测对比：core_12 vs full_24（或其他策略组合）。

按年度对比多个策略的年化收益、最大回撤、夏普比率。

用法:
    python3 scripts/yearly_split_backtest.py \
      --configs strategy/configs/models/qvf_alpha158_core12.yaml,strategy/configs/models/qvf_alpha158_hybrid24_prune_v2.yaml

    python3 scripts/yearly_split_backtest.py \
      --configs strategy/configs/models/qvf_alpha158_core12.yaml,strategy/configs/models/qvf_alpha158_hybrid24_prune_v2.yaml \
      --output results/yearly_split/comparison.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def period_metrics(returns: pd.Series) -> dict:
    returns = pd.Series(returns, dtype=float).dropna()
    if returns.empty:
        return {"annual_return": 0.0, "max_drawdown": 0.0, "sharpe": 0.0, "trading_days": 0}
    nav = (1.0 + returns).cumprod()
    max_drawdown = float((nav / nav.cummax() - 1.0).min())
    std = float(returns.std(ddof=0))
    total_return = float(nav.iloc[-1] - 1.0)
    n_days = len(returns)
    if n_days > 0:
        annual_return = (1.0 + total_return) ** (252 / n_days) - 1.0
    else:
        annual_return = 0.0
    sharpe = float(returns.mean() / std * np.sqrt(252)) if std > 0 else 0.0
    return {
        "annual_return": annual_return,
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "trading_days": n_days,
    }


def load_overlay_returns(config_path: Path) -> pd.Series | None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    root = Path(cfg["output"]["root"]).expanduser()
    overlay_path = root / "overlay_results.csv"
    if not overlay_path.exists():
        # 尝试回测 CSV
        bt_files = sorted(root.glob("backtest_*.csv"))
        if bt_files:
            return _load_backtest_csv(bt_files[-1])
        print(f"  WARNING: {cfg['name']} 无 overlay_results.csv")
        return None

    frame = pd.read_csv(overlay_path, parse_dates=["date"])
    col = "overlay_return" if "overlay_return" in frame.columns else "return"
    returns = frame.sort_values("date").drop_duplicates("date").set_index("date")[col]
    return returns.astype(float)


def _load_backtest_csv(path: Path) -> pd.Series | None:
    frame = pd.read_csv(path, parse_dates=["date"])
    if "return" not in frame.columns:
        return None
    returns = frame.sort_values("date").drop_duplicates("date").set_index("date")["return"]
    return returns.astype(float)


def yearly_split(returns: pd.Series) -> pd.DataFrame:
    rows = []
    for year, group in returns.groupby(returns.index.year):
        m = period_metrics(group)
        rows.append({"year": int(year), **m})
    # 添加整体
    total = period_metrics(returns)
    rows.append({"year": "total", **total})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="年度拆分回测对比")
    parser.add_argument(
        "--configs",
        required=True,
        help="逗号分隔的 YAML 配置路径",
    )
    parser.add_argument(
        "--output",
        default="results/yearly_split/comparison.csv",
        help="输出路径",
    )
    parser.add_argument(
        "--holdout-start",
        default="2024-01-01",
        help="Holdout 起始日期",
    )
    args = parser.parse_args()

    config_paths = [Path(p.strip()) for p in args.configs.split(",")]
    all_dfs = {}

    for cfg_path in config_paths:
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        name = cfg["name"]
        print(f"加载: {name}")
        returns = load_overlay_returns(cfg_path)
        if returns is None:
            continue
        yearly = yearly_split(returns)
        yearly["strategy"] = name
        all_dfs[name] = yearly
        print(f"  日期范围: {returns.index.min().date()} ~ {returns.index.max().date()}")

    if not all_dfs:
        print("无可用数据")
        sys.exit(1)

    # 合并输出
    combined = pd.concat(all_dfs.values(), ignore_index=True)
    combined = combined[["strategy", "year", "trading_days", "total_return", "annual_return", "max_drawdown", "sharpe"]]

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    # 打印对比表
    print(f"\n{'=' * 80}")
    print("年度拆分对比")
    print(f"{'=' * 80}")

    strategies = list(all_dfs.keys())
    for strat_df in all_dfs.values():
        name = strat_df["strategy"].iloc[0]
        print(f"\n--- {name} ---")
        for _, row in strat_df.iterrows():
            year = row["year"]
            ann = row["annual_return"]
            dd = row["max_drawdown"]
            sh = row["sharpe"]
            days = int(row["trading_days"])
            print(f"  {year:>6}: 年化={ann:>+7.2%}  回撤={dd:>+7.2%}  夏普={sh:>5.2f}  交易日={days}")

    # 策略对比摘要
    print(f"\n{'=' * 80}")
    print("Holdout 对比")
    print(f"{'=' * 80}")
    holdout_start = pd.Timestamp(args.holdout_start)
    for name, strat_df in all_dfs.items():
        matching = [p for p in config_paths if name in str(p)]
        if not matching:
            continue
        returns = load_overlay_returns(matching[0])
        if returns is None:
            continue
        holdout = returns[returns.index >= holdout_start]
        if holdout.empty:
            continue
        m = period_metrics(holdout)
        print(f"  {name:>40}: 年化={m['annual_return']:>+7.2%}  回撤={m['max_drawdown']:>+7.2%}  夏普={m['sharpe']:>5.2f}")

    print(f"\n输出: {output_path}")


if __name__ == "__main__":
    main()
