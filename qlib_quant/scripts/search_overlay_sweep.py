#!/usr/bin/env python3
"""快速 overlay 参数扫描 —— 复用已有回测的 base_return，不重训练模型。"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.modeling.portfolio_overlay import OverlayConfig, compute_overlay_frame


def _annualized_return(total_return: float, days: int) -> float:
    gross = 1.0 + total_return
    if gross <= 0 or days <= 0:
        return -1.0
    return gross ** (365.0 / days) - 1.0


def load_base_returns(overlay_csv: Path) -> pd.Series:
    """从已有 overlay_results.csv 提取 base_return。"""
    df = pd.read_csv(overlay_csv, parse_dates=["date"])
    return df.set_index("date")["base_return"].astype(float).sort_index()


def evaluate_overlay(base_returns: pd.Series, overlay_cfg: dict, oos_start: str = "2024-01-01") -> dict:
    """应用 overlay 配置，返回 full + OOS 指标。"""
    config = OverlayConfig(
        target_vol=overlay_cfg.get("target_vol"),
        vol_lookback=int(overlay_cfg.get("vol_lookback", 20)),
        dd_soft=overlay_cfg.get("dd_soft"),
        dd_hard=overlay_cfg.get("dd_hard"),
        soft_exposure=float(overlay_cfg.get("soft_exposure", 0.95)),
        hard_exposure=float(overlay_cfg.get("hard_exposure", 0.80)),
        trend_lookback=int(overlay_cfg.get("trend_lookback", 0)),
        trend_exposure=float(overlay_cfg.get("trend_exposure", 0.90)),
        exposure_min=float(overlay_cfg.get("exposure_min", 0.0)),
        exposure_max=float(overlay_cfg.get("exposure_max", 1.0)),
    )
    frame = compute_overlay_frame(base_returns, 0.03 / 252, config)

    def _metrics(sub_frame: pd.DataFrame) -> dict:
        if sub_frame.empty or len(sub_frame) < 5:
            return {"annual_return": 0.0, "max_drawdown": 0.0, "sharpe_ratio": 0.0, "avg_exposure": 0.0}
        nav = sub_frame["portfolio_value"]
        rets = sub_frame["overlay_return"].astype(float)
        days = int((nav.index[-1] - nav.index[0]).days)
        total_ret = float(nav.iloc[-1]) / (float(nav.iloc[0]) / (1 + float(rets.iloc[0]))) - 1.0
        ann_ret = _annualized_return(total_ret, days)
        max_dd = float((nav / nav.cummax() - 1.0).min())
        std = float(rets.std(ddof=0))
        sharpe = float(rets.mean()) / std * math.sqrt(252) if std > 0 else 0.0
        avg_exp = float(sub_frame["exposure"].mean())
        return {"annual_return": ann_ret, "max_drawdown": max_dd, "sharpe_ratio": sharpe, "avg_exposure": avg_exp}

    full = _metrics(frame)
    oos_mask = frame.index >= pd.Timestamp(oos_start)
    oos = _metrics(frame.loc[oos_mask])
    return {"full": full, "oos": oos}


# 网格定义
OVERLAY_GRIDS = {
    "aggressive_dd": {
        "target_vol": [0.12, 0.15, 0.19],
        "dd_soft": [0.010, 0.015, 0.020],
        "dd_hard": [0.025, 0.035, 0.045],
        "hard_exposure": [0.25, 0.35, 0.45],
        "soft_exposure": [0.65, 0.75, 0.82],
        "trend_lookback": [0, 20, 40],
        "trend_exposure": [0.60, 0.70, 0.80],
    },
    "conservative": {
        "target_vol": [0.10, 0.13, 0.16],
        "dd_soft": [0.008, 0.012, 0.018],
        "dd_hard": [0.020, 0.030, 0.040],
        "hard_exposure": [0.20, 0.30, 0.40],
        "soft_exposure": [0.60, 0.70, 0.80],
        "trend_lookback": [20, 40, 60],
        "trend_exposure": [0.55, 0.65, 0.75],
    },
    "targeted": [
        # 精选组合，基于历史经验
        {"target_vol": 0.19, "dd_soft": 0.025, "dd_hard": 0.05, "hard_exposure": 0.48, "soft_exposure": 0.82, "trend_lookback": 20, "trend_exposure": 0.75},
        {"target_vol": 0.15, "dd_soft": 0.020, "dd_hard": 0.04, "hard_exposure": 0.40, "soft_exposure": 0.75, "trend_lookback": 20, "trend_exposure": 0.70},
        {"target_vol": 0.12, "dd_soft": 0.015, "dd_hard": 0.03, "hard_exposure": 0.30, "soft_exposure": 0.70, "trend_lookback": 20, "trend_exposure": 0.65},
        {"target_vol": 0.19, "dd_soft": 0.015, "dd_hard": 0.03, "hard_exposure": 0.35, "soft_exposure": 0.70, "trend_lookback": 20, "trend_exposure": 0.70},
        {"target_vol": 0.19, "dd_soft": 0.010, "dd_hard": 0.025, "hard_exposure": 0.25, "soft_exposure": 0.65, "trend_lookback": 20, "trend_exposure": 0.60},
        {"target_vol": 0.15, "dd_soft": 0.010, "dd_hard": 0.025, "hard_exposure": 0.25, "soft_exposure": 0.65, "trend_lookback": 20, "trend_exposure": 0.60},
        {"target_vol": 0.12, "dd_soft": 0.010, "dd_hard": 0.020, "hard_exposure": 0.20, "soft_exposure": 0.60, "trend_lookback": 40, "trend_exposure": 0.55},
        {"target_vol": 0.19, "dd_soft": 0.020, "dd_hard": 0.04, "hard_exposure": 0.40, "soft_exposure": 0.75, "trend_lookback": 40, "trend_exposure": 0.70},
        {"target_vol": 0.15, "dd_soft": 0.015, "dd_hard": 0.035, "hard_exposure": 0.35, "soft_exposure": 0.70, "trend_lookback": 40, "trend_exposure": 0.65},
        {"target_vol": 0.10, "dd_soft": 0.008, "dd_hard": 0.020, "hard_exposure": 0.20, "soft_exposure": 0.55, "trend_lookback": 40, "trend_exposure": 0.50},
    ],
}


def generate_grid_configs(grid_name: str) -> list[dict]:
    if grid_name == "targeted":
        return OVERLAY_GRIDS["targeted"]

    grid = OVERLAY_GRIDS[grid_name]
    keys = list(grid.keys())
    values = [grid[k] if isinstance(grid[k], list) else [grid[k]] for k in keys]
    configs = []
    for combo in itertools.product(*values):
        configs.append(dict(zip(keys, combo)))
    return configs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-overlay-csv", default="results/model_signals/hybrid10_cashflow_quality10_qlib_k8_d2/overlay_results.csv")
    parser.add_argument("--grid", choices=["aggressive_dd", "conservative", "targeted", "all"], default="targeted")
    parser.add_argument("--search-name", default="overlay_sweep_v1")
    parser.add_argument("--oos-start", default="2024-01-01")
    return parser.parse_args()


def main():
    args = parse_args()
    csv_path = PROJECT_ROOT / args.base_overlay_csv
    if not csv_path.exists():
        print(f"[ERROR] 找不到 {csv_path}")
        sys.exit(1)

    base_returns = load_base_returns(csv_path)
    print(f"[INFO] 加载 {len(base_returns)} 天 base_return")

    out_root = PROJECT_ROOT / "results" / "model_signals" / "search_runs" / args.search_name
    out_root.mkdir(parents=True, exist_ok=True)

    grid_names = ["targeted", "aggressive_dd", "conservative"] if args.grid == "all" else [args.grid]

    all_rows = []
    for grid_name in grid_names:
        configs = generate_grid_configs(grid_name)
        print(f"[INFO] {grid_name}: {len(configs)} 种 overlay 配置")
        for i, cfg in enumerate(configs):
            result = evaluate_overlay(base_returns, cfg, oos_start=args.oos_start)
            tag = f"{grid_name}_{i:04d}"
            row = {
                "tag": tag,
                "grid": grid_name,
                **{f"ov_{k}": v for k, v in cfg.items()},
                "full_annual_return": result["full"]["annual_return"],
                "full_max_drawdown": result["full"]["max_drawdown"],
                "full_sharpe_ratio": result["full"]["sharpe_ratio"],
                "full_avg_exposure": result["full"]["avg_exposure"],
                "oos_annual_return": result["oos"]["annual_return"],
                "oos_max_drawdown": result["oos"]["max_drawdown"],
                "oos_sharpe_ratio": result["oos"]["sharpe_ratio"],
                "oos_avg_exposure": result["oos"]["avg_exposure"],
            }
            all_rows.append(row)

    df = pd.DataFrame(all_rows)

    # 打分: OOS夏普优先, 回撤次之
    df["score"] = (
        df["oos_sharpe_ratio"] * 0.4
        + df["oos_annual_return"].clip(lower=-0.5, upper=0.5) * 2.0
        + df["oos_max_drawdown"].clip(lower=-0.30, upper=0) * (-3.0)  # 回撤越小越好
    )
    df = df.sort_values("score", ascending=False)

    out_csv = out_root / "overlay_sweep_results.csv"
    df.to_csv(out_csv, index=False)

    # 打印 Top 10
    print(f"\n[OK] 扫描完成: {out_csv}")
    print(f"共 {len(df)} 种配置\n")
    print("=" * 120)
    print(f"{'排名':>4} {'OOS年化':>8} {'OOS夏普':>8} {'OOS回撤':>8} {'全期年化':>8} {'全期回撤':>8} {'全期夏普':>8} {'得分':>6} {'配置'}")
    print("-" * 120)
    for rank, (_, row) in enumerate(df.head(20).iterrows(), 1):
        ov_keys = [k for k in row.index if k.startswith("ov_")]
        ov_str = " ".join(f"{k[3:]}={row[k]}" for k in ov_keys)
        print(
            f"{rank:4d} {row['oos_annual_return']:8.2%} {row['oos_sharpe_ratio']:8.3f} "
            f"{row['oos_max_drawdown']:8.2%} {row['full_annual_return']:8.2%} "
            f"{row['full_max_drawdown']:8.2%} {row['full_sharpe_ratio']:8.3f} "
            f"{row['score']:6.2f} {ov_str}"
        )


if __name__ == "__main__":
    main()
