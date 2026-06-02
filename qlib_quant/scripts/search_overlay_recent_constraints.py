#!/usr/bin/env python3
"""Overlay 参数约束扫描。

这个脚本用于复用已有模型/选股的 base_return，只搜索组合风控参数。
目标是把“全周期”和“近期”约束放在同一张表里，避免只看单段结果。
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.modeling.portfolio_overlay import OverlayConfig, compute_overlay_frame


def annualized_return(total_return: float, days: int) -> float:
    gross = 1.0 + float(total_return)
    if gross <= 0 or days <= 0:
        return -1.0
    return gross ** (365.0 / days) - 1.0


def load_overlay_inputs(path: Path) -> tuple[pd.Series, pd.Series | float]:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").set_index("date")
    if "base_return" not in df.columns:
        raise ValueError(f"{path} 缺少 base_return 列")
    base_returns = df["base_return"].astype(float)
    if "bond_return" in df.columns:
        bond_returns = df["bond_return"].astype(float)
    else:
        bond_returns = 0.03 / 252
    return base_returns, bond_returns


def segment_metrics(frame: pd.DataFrame, start: str, end: str) -> dict[str, float]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    sub = frame.loc[(frame.index >= start_ts) & (frame.index <= end_ts)]
    if sub.empty or len(sub) < 5:
        return {
            "annual_return": 0.0,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "avg_exposure": 0.0,
        }

    nav = sub["portfolio_value"].astype(float)
    rets = sub["overlay_return"].astype(float)
    first_nav_before_return = float(nav.iloc[0]) / (1.0 + float(rets.iloc[0]))
    total_return = float(nav.iloc[-1]) / first_nav_before_return - 1.0
    days = int((nav.index[-1] - nav.index[0]).days)
    std = float(rets.std(ddof=0))
    sharpe = float(rets.mean()) / std * math.sqrt(252.0) if std > 0 else 0.0
    max_drawdown = float((nav / nav.cummax() - 1.0).min())
    return {
        "annual_return": annualized_return(total_return, days),
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe,
        "avg_exposure": float(sub["exposure"].mean()),
    }


def focused_overlay_configs() -> list[dict]:
    """小而集中的参数网格，避免无意义全网格爆炸。"""
    risk_profiles = [
        {"dd_soft": 0.010, "dd_hard": 0.025, "soft_exposure": 0.65, "hard_exposure": 0.25},
        {"dd_soft": 0.012, "dd_hard": 0.028, "soft_exposure": 0.68, "hard_exposure": 0.30},
        {"dd_soft": 0.015, "dd_hard": 0.030, "soft_exposure": 0.70, "hard_exposure": 0.35},
        {"dd_soft": 0.015, "dd_hard": 0.035, "soft_exposure": 0.72, "hard_exposure": 0.38},
        {"dd_soft": 0.018, "dd_hard": 0.035, "soft_exposure": 0.75, "hard_exposure": 0.40},
        {"dd_soft": 0.020, "dd_hard": 0.040, "soft_exposure": 0.78, "hard_exposure": 0.45},
        {"dd_soft": 0.025, "dd_hard": 0.050, "soft_exposure": 0.82, "hard_exposure": 0.48},
        {"dd_soft": 0.030, "dd_hard": 0.060, "soft_exposure": 0.85, "hard_exposure": 0.55},
    ]
    trend_profiles = [
        {"trend_lookback": 0, "trend_exposure": 1.0},
        {"trend_lookback": 20, "trend_exposure": 0.65},
        {"trend_lookback": 20, "trend_exposure": 0.75},
        {"trend_lookback": 40, "trend_exposure": 0.65},
        {"trend_lookback": 40, "trend_exposure": 0.75},
    ]
    target_vols = [0.15, 0.17, 0.19, 0.21]

    configs: list[dict] = []
    for target_vol in target_vols:
        for risk in risk_profiles:
            for trend in trend_profiles:
                cfg = {
                    "target_vol": target_vol,
                    "vol_lookback": 20,
                    "exposure_min": 0.0,
                    "exposure_max": 1.0,
                    **risk,
                    **trend,
                }
                configs.append(cfg)
    return configs


def evaluate_path(path: Path, args: argparse.Namespace) -> pd.DataFrame:
    base_returns, bond_returns = load_overlay_inputs(path)
    rows: list[dict] = []
    for idx, cfg in enumerate(focused_overlay_configs()):
        overlay_cfg = OverlayConfig(**cfg)
        frame = compute_overlay_frame(base_returns, bond_returns, overlay_cfg)
        full = segment_metrics(frame, args.full_start, args.end)
        holdout = segment_metrics(frame, args.holdout_start, args.end)
        recent = segment_metrics(frame, args.recent_start, args.end)
        ytd = segment_metrics(frame, args.ytd_start, args.end)
        score = (
            full["annual_return"] * 2.0
            + holdout["annual_return"] * 1.0
            + recent["annual_return"] * 0.75
            + ytd["annual_return"] * 0.50
            + full["max_drawdown"] * 2.5
            + recent["max_drawdown"] * 1.0
        )
        hit = (
            full["annual_return"] >= args.min_full_ann
            and full["max_drawdown"] >= -abs(args.max_full_dd)
            and recent["annual_return"] >= args.min_recent_ann
            and ytd["annual_return"] >= args.min_ytd_ann
        )
        rows.append(
            {
                "source": str(path.relative_to(PROJECT_ROOT)),
                "tag": f"cfg_{idx:04d}",
                "hit": hit,
                "score": score,
                **{f"ov_{k}": v for k, v in cfg.items()},
                "full_annual_return": full["annual_return"],
                "full_total_return": full["total_return"],
                "full_max_drawdown": full["max_drawdown"],
                "full_sharpe_ratio": full["sharpe_ratio"],
                "full_avg_exposure": full["avg_exposure"],
                "holdout_annual_return": holdout["annual_return"],
                "holdout_max_drawdown": holdout["max_drawdown"],
                "holdout_sharpe_ratio": holdout["sharpe_ratio"],
                "recent_annual_return": recent["annual_return"],
                "recent_max_drawdown": recent["max_drawdown"],
                "recent_sharpe_ratio": recent["sharpe_ratio"],
                "ytd_annual_return": ytd["annual_return"],
                "ytd_total_return": ytd["total_return"],
                "ytd_max_drawdown": ytd["max_drawdown"],
                "ytd_sharpe_ratio": ytd["sharpe_ratio"],
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-overlay-csv", nargs="+", required=True)
    parser.add_argument("--search-name", required=True)
    parser.add_argument("--full-start", default="2019-01-01")
    parser.add_argument("--holdout-start", default="2024-01-01")
    parser.add_argument("--recent-start", default="2025-10-01")
    parser.add_argument("--ytd-start", default="2026-01-01")
    parser.add_argument("--end", default="2026-04-15")
    parser.add_argument("--min-full-ann", type=float, default=0.25)
    parser.add_argument("--max-full-dd", type=float, default=0.15)
    parser.add_argument("--min-recent-ann", type=float, default=0.0)
    parser.add_argument("--min-ytd-ann", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = [PROJECT_ROOT / p for p in args.base_overlay_csv]
    missing = [p for p in paths if not p.exists()]
    if missing:
        for path in missing:
            print(f"[ERROR] 找不到 {path}")
        raise SystemExit(1)

    out_dir = PROJECT_ROOT / "results" / "model_signals" / "search_runs" / args.search_name
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for path in paths:
        print(f"[INFO] scan {path.relative_to(PROJECT_ROOT)}")
        frames.append(evaluate_path(path, args))

    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(["hit", "score"], ascending=[False, False])
    out_csv = out_dir / "overlay_recent_constraints.csv"
    result.to_csv(out_csv, index=False)

    hits = result[result["hit"]]
    print(f"[OK] saved {out_csv}")
    print(f"[INFO] configs={len(result)} hits={len(hits)}")
    show = hits if not hits.empty else result
    cols = [
        "source",
        "tag",
        "full_annual_return",
        "full_max_drawdown",
        "holdout_annual_return",
        "holdout_max_drawdown",
        "recent_annual_return",
        "recent_max_drawdown",
        "ytd_annual_return",
        "ytd_max_drawdown",
        "ov_target_vol",
        "ov_dd_soft",
        "ov_dd_hard",
        "ov_soft_exposure",
        "ov_hard_exposure",
        "ov_trend_lookback",
        "ov_trend_exposure",
    ]
    print(show[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
