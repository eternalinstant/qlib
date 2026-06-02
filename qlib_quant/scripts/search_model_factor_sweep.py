#!/usr/bin/env python3
"""综合搜索: 因子组合 × 模型参数 × topk, 带优化 overlay。

目标: 年化 25%+ / 回撤 -10% 左右
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.modeling.predictive_signal import (
    backtest_from_config,
    load_predictive_config,
    save_json,
    score_from_config,
    train_from_config,
)
from modules.modeling.portfolio_overlay import OverlayConfig, compute_overlay_frame

ALPHA158_TECH = {"BETA60", "MA60", "ROC20", "RSV20", "CORD10", "CORD20", "WVMA10", "VSUMD10", "RANK20"}

# 精选因子组合 — 基于历史最优 + 新候选
FACTOR_BUNDLES = {
    # 最优基线
    "cq10": {
        "parquet": ["ocf_to_ev", "fcff_to_mv", "roe_fina", "current_ratio_fina", "n_cashflow_act", "rank_value_profit_core", "rank_balance_core", "qvf_core_interaction"],
        "alpha158": ["ROC20", "CORD20"],
    },
    "tech10": {
        "parquet": ["ebit_to_mv", "roe_fina", "operate_profit_inc", "rank_value_profit_core", "rank_growth_quality_core", "rank_flow_momentum_core", "qvf_core_interaction"],
        "alpha158": ["ROC20", "RANK20", "CORD20"],
    },
    # 精简版 — 减少因子数可能提升OOS
    "cq7": {
        "parquet": ["ocf_to_ev", "fcff_to_mv", "roe_fina", "current_ratio_fina", "rank_value_profit_core", "qvf_core_interaction"],
        "alpha158": ["ROC20"],
    },
    "cq8_rsv": {
        "parquet": ["ocf_to_ev", "fcff_to_mv", "roe_fina", "current_ratio_fina", "n_cashflow_act", "rank_value_profit_core", "qvf_core_interaction"],
        "alpha158": ["ROC20", "RSV20"],
    },
    "cq8_rank": {
        "parquet": ["ocf_to_ev", "fcff_to_mv", "roe_fina", "current_ratio_fina", "n_cashflow_act", "rank_value_profit_core", "qvf_core_interaction"],
        "alpha158": ["ROC20", "RANK20"],
    },
    # 增强版 — 加入BETA60
    "cq10_beta": {
        "parquet": ["ocf_to_ev", "fcff_to_mv", "roe_fina", "current_ratio_fina", "n_cashflow_act", "rank_value_profit_core", "rank_balance_core", "qvf_core_interaction"],
        "alpha158": ["ROC20", "CORD20", "BETA60"],
    },
    # 纯价值质量
    "vq6": {
        "parquet": ["ebit_to_mv", "ocf_to_ev", "roe_fina", "current_ratio_fina", "rank_value_profit_core", "qvf_core_interaction"],
        "alpha158": ["ROC20"],
    },
    # 质量增强
    "qe8": {
        "parquet": ["ocf_to_ev", "fcff_to_mv", "roe_fina", "roa_fina", "current_ratio_fina", "rank_profitability_quality_core", "rank_balance_core", "qvf_core_interaction"],
        "alpha158": ["ROC20", "CORD20"],
    },
}

# 模型参数网格
MODEL_PARAMS = {
    "m_d4_n300": {"learning_rate": 0.05, "n_estimators": 300, "max_depth": 4, "min_child_samples": 32, "random_state": 42, "verbosity": -1},
    "m_d3_n200": {"learning_rate": 0.05, "n_estimators": 200, "max_depth": 3, "min_child_samples": 32, "random_state": 42, "verbosity": -1},
    "m_d3_n100": {"learning_rate": 0.05, "n_estimators": 100, "max_depth": 3, "min_child_samples": 32, "random_state": 42, "verbosity": -1},
    "m_d4_n200": {"learning_rate": 0.05, "n_estimators": 200, "max_depth": 4, "min_child_samples": 32, "random_state": 42, "verbosity": -1},
    "m_d4_n100": {"learning_rate": 0.05, "n_estimators": 100, "max_depth": 4, "min_child_samples": 32, "random_state": 42, "verbosity": -1},
    "m_d3_n300_c64": {"learning_rate": 0.05, "n_estimators": 300, "max_depth": 3, "min_child_samples": 64, "random_state": 42, "verbosity": -1},
}

# topk/n_drop 组合
TOPK_PRESETS = {
    "k6d2": {"topk": 6, "n_drop": 2},
    "k6d1": {"topk": 6, "n_drop": 1},
    "k8d2": {"topk": 8, "n_drop": 2},
    "k10d3": {"topk": 10, "n_drop": 3},
}

# 最佳 overlay 配置 (从 cross-sweep 得到)
OVERLAY_CONFIGS = {
    "very_tight": {
        "target_vol": 0.19, "dd_soft": 0.010, "dd_hard": 0.025,
        "hard_exposure": 0.25, "soft_exposure": 0.65,
        "trend_lookback": 20, "trend_exposure": 0.60,
        "vol_lookback": 20, "exposure_min": 0.0, "exposure_max": 1.0,
    },
    "tight_dd": {
        "target_vol": 0.19, "dd_soft": 0.015, "dd_hard": 0.030,
        "hard_exposure": 0.35, "soft_exposure": 0.70,
        "trend_lookback": 20, "trend_exposure": 0.70,
        "vol_lookback": 20, "exposure_min": 0.0, "exposure_max": 1.0,
    },
    "trend40": {
        "target_vol": 0.19, "dd_soft": 0.020, "dd_hard": 0.040,
        "hard_exposure": 0.40, "soft_exposure": 0.75,
        "trend_lookback": 40, "trend_exposure": 0.70,
        "vol_lookback": 20, "exposure_min": 0.0, "exposure_max": 1.0,
    },
}


def search_root(name: str) -> Path:
    return PROJECT_ROOT / "results" / "model_signals" / "search_runs" / name


def build_cfg(
    base_cfg: dict,
    bundle_name: str,
    model_name: str,
    preset_name: str,
    overlay_name: str,
    output_root: Path,
) -> dict:
    bundle = FACTOR_BUNDLES[bundle_name]
    preset = TOPK_PRESETS[preset_name]
    model_p = MODEL_PARAMS[model_name]
    overlay_p = OVERLAY_CONFIGS[overlay_name]

    cfg = copy.deepcopy(base_cfg)
    cfg["name"] = f"mf_{bundle_name}_{model_name}_{preset_name}_{overlay_name}"

    cfg["data"]["source"] = "hybrid"
    cfg["data"]["parquet_feature_columns"] = list(bundle["parquet"])
    cfg["data"]["alpha158_feature_columns"] = list(bundle["alpha158"])

    cfg["model"]["params"] = dict(model_p)

    topk = int(preset["topk"])
    n_drop = int(preset["n_drop"])
    sel = cfg["selection"]
    sel["topk"] = topk
    sel["sticky"] = max(topk - n_drop, 0)
    sel["churn_limit"] = n_drop
    sel["buffer"] = 0
    sel["threshold"] = 0.0
    sel["margin_stable"] = False
    sel["score_smoothing_days"] = 1
    sel["entry_persist_days"] = 1
    sel["exit_persist_days"] = 1
    sel["min_hold_days"] = 0
    sel["entry_rank"] = None
    sel["exit_rank"] = None

    cfg["overlay"]["enabled"] = True
    cfg["overlay"].update(overlay_p)

    cfg["output"]["root"] = str(output_root / cfg["name"])
    return cfg


def evaluate_overlay_window(path: Path, start_date: str, end_date: str) -> dict:
    if not path.exists():
        return {"annual_return": 0.0, "max_drawdown": 0.0, "sharpe_ratio": 0.0}
    frame = pd.read_csv(path, parse_dates=["date"])
    mask = (frame["date"] >= pd.Timestamp(start_date)) & (frame["date"] <= pd.Timestamp(end_date))
    frame = frame.loc[mask]
    if frame.empty or len(frame) < 5:
        return {"annual_return": 0.0, "max_drawdown": 0.0, "sharpe_ratio": 0.0}
    first_val = float(frame["portfolio_value"].iloc[0]) / (1 + float(frame["overlay_return"].iloc[0]))
    last_val = float(frame["portfolio_value"].iloc[-1])
    total = last_val / first_val - 1.0
    days = max((frame["date"].iloc[-1] - frame["date"].iloc[0]).days, 1)
    ann = (1 + total) ** (365 / days) - 1 if (1 + total) > 0 else -1
    dd = float(frame["drawdown"].min())
    rets = frame["overlay_return"].astype(float)
    std = float(rets.std(ddof=0))
    sharpe = float(rets.mean()) / std * math.sqrt(252) if std > 0 else 0
    return {"annual_return": ann, "max_drawdown": dd, "sharpe_ratio": sharpe}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-config", default="config/models/hybrid10_cashflow_quality10_qlib_k8_d2.yaml")
    p.add_argument("--search-name", default="model_factor_sweep_v1")
    p.add_argument("--bundles", default="")  # empty=all
    p.add_argument("--models", default="")  # empty=all
    p.add_argument("--presets", default="k6d2,k6d1")
    p.add_argument("--overlays", default="very_tight,tight_dd")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--offset", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    base_cfg = load_predictive_config(args.base_config)
    root = search_root(args.search_name)
    root.mkdir(parents=True, exist_ok=True)

    bundles = [x.strip() for x in args.bundles.split(",") if x.strip()] or list(FACTOR_BUNDLES.keys())
    models = [x.strip() for x in args.models.split(",") if x.strip()] or list(MODEL_PARAMS.keys())
    presets = [x.strip() for x in args.presets.split(",") if x.strip()]
    overlays = [x.strip() for x in args.overlays.split(",") if x.strip()]

    candidates = [(b, m, p, o) for b in bundles for m in models for p in presets for o in overlays]
    if args.offset:
        candidates = candidates[args.offset:]
    if args.limit:
        candidates = candidates[:args.limit]

    print(f"[INFO] 共 {len(candidates)} 候选配置")
    print(f"[INFO] bundles={bundles} models={models} presets={presets} overlays={overlays}")

    rows = []
    for i, (bname, mname, pname, oname) in enumerate(candidates):
        cfg = build_cfg(base_cfg, bname, mname, pname, oname, root)
        tag = cfg["name"]
        started = time.time()
        print(f"[{i+1}/{len(candidates)}] {tag}")
        try:
            train_summary = train_from_config(cfg)
            score_from_config(cfg)
            _, bt_summary = backtest_from_config(cfg, engine="qlib")

            full = evaluate_overlay_window(
                Path(cfg["output"]["root"]) / "overlay_results.csv",
                str(cfg["data"]["start_date"]),
                str(cfg["data"]["end_date"]),
            )
            oos = evaluate_overlay_window(
                Path(cfg["output"]["root"]) / "overlay_results.csv",
                "2024-01-01",
                str(cfg["data"]["end_date"]),
            )

            metrics = train_summary.get("metrics", {})
            rows.append({
                "name": tag,
                "bundle": bname, "model": mname, "preset": pname, "overlay": oname,
                "n_factors": len(FACTOR_BUNDLES[bname]["parquet"]) + len(FACTOR_BUNDLES[bname]["alpha158"]),
                "topk": TOPK_PRESETS[pname]["topk"], "n_drop": TOPK_PRESETS[pname]["n_drop"],
                "valid_rank_ic": float(metrics.get("valid_mean_rank_ic", 0)),
                "full_ann": full["annual_return"], "full_dd": full["max_drawdown"], "full_sh": full["sharpe_ratio"],
                "oos_ann": oos["annual_return"], "oos_dd": oos["max_drawdown"], "oos_sh": oos["sharpe_ratio"],
                "elapsed": round(time.time() - started, 1),
            })

            hit = " <<<" if full["max_drawdown"] > -0.105 else ""
            print(f"  → full={full['annual_return']:.2%}/{full['max_drawdown']:.2%} oos={oos['annual_return']:.2%}/{oos['max_drawdown']:.2%}{hit}")
        except Exception as e:
            print(f"  [ERR] {e}")
            rows.append({
                "name": tag, "bundle": bname, "model": mname, "preset": pname, "overlay": oname,
                "n_factors": 0, "topk": 0, "n_drop": 0, "valid_rank_ic": 0,
                "full_ann": 0, "full_dd": 0, "full_sh": 0,
                "oos_ann": 0, "oos_dd": 0, "oos_sh": 0,
                "elapsed": round(time.time() - started, 1),
                "error": str(e),
            })

    df = pd.DataFrame(rows)
    df["score"] = (
        df["oos_sh"] * 0.3
        + df["oos_ann"].clip(lower=-0.5, upper=0.5) * 2.0
        + (1 + df["full_dd"].clip(lower=-0.3, upper=0)) * 3.0
    )
    df = df.sort_values("score", ascending=False)
    df.to_csv(root / "summary.csv", index=False)
    save_json({"search_name": args.search_name, "total": len(rows), "best": df.iloc[0].to_dict() if not df.empty else None}, root / "summary.json")

    print(f"\n[OK] 搜索完成: {root / 'summary.csv'}")
    if not df.empty:
        best = df.iloc[0]
        print(f"[BEST] {best['name']} full={best['full_ann']:.2%}/{best['full_dd']:.2%} oos={best['oos_ann']:.2%}/{best['oos_dd']:.2%}")


if __name__ == "__main__":
    main()
