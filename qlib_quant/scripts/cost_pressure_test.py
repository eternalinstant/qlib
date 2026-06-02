#!/usr/bin/env python3
"""
交易成本压力测试：评估策略在不同成本假设下的表现。

测试档位：0bp / 5bp / 10bp / 15bp / 20bp / 30bp
成本以单边 total cost (买+卖) 计。

用法:
    python3 scripts/cost_pressure_test.py \
      --config config/models/qvf_alpha158_core12.yaml

    python3 scripts/cost_pressure_test.py \
      --config config/models/qvf_alpha158_core12.yaml \
      --cost-levels 0,5,10,15,20,30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 默认单边成本档位（单位：bp）
DEFAULT_COST_LEVELS = [0, 5, 10, 15, 20, 30]


def period_metrics(returns: pd.Series) -> dict:
    returns = pd.Series(returns, dtype=float).dropna()
    if returns.empty:
        return {"annual_return": 0.0, "max_drawdown": 0.0, "sharpe": 0.0, "trading_days": 0}
    nav = (1.0 + returns).cumprod()
    max_drawdown = float((nav / nav.cummax() - 1.0).min())
    std = float(returns.std(ddof=0))
    total_return = float(nav.iloc[-1] - 1.0)
    n_days = len(returns)
    annual_return = (1.0 + total_return) ** (252 / n_days) - 1.0 if n_days > 0 else 0.0
    sharpe = float(returns.mean() / std * np.sqrt(252)) if std > 0 else 0.0
    return {
        "annual_return": float(annual_return),
        "total_return": float(total_return),
        "max_drawdown": float(max_drawdown),
        "sharpe": float(sharpe),
        "trading_days": n_days,
    }


def load_overlay_data(config_path: Path) -> tuple[pd.DataFrame, dict]:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    root = Path(cfg["output"]["root"]).expanduser()
    overlay_path = root / "overlay_results.csv"

    if not overlay_path.exists():
        raise FileNotFoundError(f"未找到 overlay_results.csv: {overlay_path}")

    df = pd.read_csv(overlay_path, parse_dates=["date"]).sort_values("date")

    # 提取当前成本配置
    trading = cfg.get("trading", {})
    current_cost = {
        "buy_commission_rate": trading.get("buy_commission_rate", 0.0003),
        "sell_commission_rate": trading.get("sell_commission_rate", 0.0003),
        "sell_stamp_tax_rate": trading.get("sell_stamp_tax_rate", 0.001),
        "slippage_bps": trading.get("slippage_bps", 5.0),
        "impact_bps": trading.get("impact_bps", 5.0),
    }
    # 计算当前单边综合成本（买+卖）
    buy_side = current_cost["buy_commission_rate"] + current_cost["slippage_bps"] / 10000 + current_cost["impact_bps"] / 10000
    sell_side = current_cost["sell_commission_rate"] + current_cost["sell_stamp_tax_rate"] + current_cost["slippage_bps"] / 10000 + current_cost["impact_bps"] / 10000
    current_total_bps = (buy_side + sell_side) * 10000

    return df, {
        "name": cfg["name"],
        "current_cost_bps": current_total_bps,
        "selection_freq": cfg.get("selection", {}).get("freq", "unknown"),
        "topk": cfg.get("selection", {}).get("topk", 0),
    }


def estimate_turnover(overlay_df: pd.DataFrame) -> pd.Series:
    """从 overlay 数据估算日换手率。

    通过 exposure 变化估算仓位调整幅度。
    如果有 base_return 列，可以用 exposure 变化近似换手。
    """
    if "exposure" in overlay_df.columns:
        exposure_changes = overlay_df["exposure"].diff().abs().fillna(0)
        # 估算换手：exposure 变化意味着仓位调整
        return exposure_changes
    return pd.Series(0.0, index=overlay_df.index)


def run_pressure_test(config_path: Path, cost_levels: list[int]) -> pd.DataFrame:
    overlay_df, info = load_overlay_data(config_path)
    print(f"策略: {info['name']}")
    print(f"当前配置单边成本: {info['current_cost_bps']:.1f} bp")
    print(f"调仓频率: {info['selection_freq']}, topk: {info['topk']}")

    # 使用 base_return + overlay_return 重建
    base_col = "base_return" if "base_return" in overlay_df.columns else None
    overlay_col = "overlay_return" if "overlay_return" in overlay_df.columns else "return"

    # 估算换手成本
    turnover = estimate_turnover(overlay_df)
    avg_turnover = float(turnover.mean())

    # 当前成本下的实际收益
    current_overlay = overlay_df.set_index("date")[overlay_col].astype(float)

    # 当前成本对应的 bp
    current_bps = info["current_cost_bps"]

    rows = []
    for target_bps in cost_levels:
        # 调整成本差异
        cost_diff_bps = target_bps - current_bps
        # 日均额外成本 = 成本差异 * 日换手率
        daily_cost_adjustment = (cost_diff_bps / 10000) * avg_turnover

        adjusted = current_overlay - daily_cost_adjustment
        m = period_metrics(adjusted)
        rows.append({
            "cost_bps": target_bps,
            **m,
        })

    return pd.DataFrame(rows), info


def main():
    parser = argparse.ArgumentParser(description="交易成本压力测试")
    parser.add_argument("--config", required=True, help="YAML 配置路径")
    parser.add_argument("--cost-levels", default="0,5,10,15,20,30", help="逗号分隔的成本档位（bp）")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    cost_levels = [int(x.strip()) for x in args.cost_levels.split(",")]
    config_path = Path(args.config)

    result_df, info = run_pressure_test(config_path, cost_levels)

    # 打印结果
    print(f"\n{'=' * 70}")
    print(f"成本压力测试: {info['name']}")
    print(f"{'=' * 70}")
    print(f"{'成本(bp)':>10} {'年化收益':>10} {'最大回撤':>10} {'夏普':>8} {'交易日':>8}")
    print("-" * 70)
    for _, row in result_df.iterrows():
        print(f"{row['cost_bps']:>10.0f} {row['annual_return']:>+10.2%} {row['max_drawdown']:>+10.2%} {row['sharpe']:>8.2f} {row['trading_days']:>8.0f}")

    # 关键判断
    zero_cost = result_df[result_df["cost_bps"] == 0]
    current_cost = result_df[result_df["cost_bps"] == round(info["current_cost_bps"])]
    high_cost = result_df[result_df["cost_bps"] == 20]

    if not zero_cost.empty and not high_cost.empty:
        zero_ann = zero_cost.iloc[0]["annual_return"]
        high_ann = high_cost.iloc[0]["annual_return"]
        cost_drag = zero_ann - high_ann
        print(f"\n成本拖累 (0bp → 20bp): {cost_drag:.2%}")
        if high_ann > 0.05:
            print("结论: 20bp 成本下年化仍 > 5%, 策略成本容忍度良好")
        elif high_ann > 0:
            print("结论: 20bp 成本下年化仍为正, 但利润空间有限")
        else:
            print("结论: 20bp 成本下年化为负, 策略对成本高度敏感")

    # 保存
    if args.output:
        output_path = PROJECT_ROOT / args.output
    else:
        name = info["name"]
        output_path = PROJECT_ROOT / "results" / "cost_pressure" / f"{name}_cost_pressure.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)
    print(f"\n输出: {output_path}")


if __name__ == "__main__":
    main()
