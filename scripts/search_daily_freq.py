#!/usr/bin/env python3
"""搜索日频策略：多因子组合 × topk × sticky × overlay参数。

基于 qlib 模型信号框架，搜索 freq=day 下的最优配置。
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.modeling.predictive_signal import (
    backtest_from_config,
    load_predictive_config,
    save_json,
    score_from_config,
    train_from_config,
)

ALPHA158_BASE = {
    "kbar": {},
    "price": {"windows": [0], "feature": ["OPEN", "HIGH", "LOW", "VWAP"]},
    "rolling": {
        "windows": [10, 20, 60],
        "include": [
            "ROC", "MA", "STD", "BETA", "RSQR", "RESI",
            "MAX", "LOW", "RANK", "RSV", "IMAX", "IMIN",
            "IMXD", "CORR", "CORD", "CNTP", "CNTD", "SUMP",
            "VSTD", "WVMA", "VSUMD",
        ],
    },
}

# 因子组合：每组 ≤ 6 因子
FEATURE_BUNDLES = {
    "momentum5": ["ROC20", "RSV20", "RANK20", "CORD20", "VSUMD20"],
    "trend5": ["MA20", "ROC20", "BETA20", "RSQR20", "RESI20"],
    "breakout5": ["MAX20", "MIN20", "RSV20", "IMAX20", "IMIN20"],
    "behavioral5": ["CNTP20", "CNTD20", "SUMP20", "VSUMD20", "ROC20"],
    "price_vol5": ["CORR20", "CORD20", "WVMA20", "VSUMD20", "VSTD20"],
    "kbar5": ["KMID", "KLEN", "KSFT", "CORD20", "VSUMD20"],
    "momentum3": ["ROC20", "RSV20", "RANK20"],
    "trend3": ["MA20", "ROC20", "BETA20"],
    "rocvol3": ["ROC20", "CORD20", "VSUMD20"],
    "rankvol3": ["RANK20", "RSV20", "VSUMD20"],
    "composite6": ["ROC20", "CORD20", "RANK20", "RSV20", "VSUMD20", "MA20"],
    "composite8": ["ROC20", "CORD20", "RANK20", "RSV20", "VSUMD20", "MA20", "WVMA20", "BETA20"],
}

# 日频关键参数网格
DAILY_GRID = {
    "topk": [4, 6, 8],
    "sticky": [10, 20, 40],
    "horizon_days": [3, 5],
}

# Overlay配置预设
OVERLAY_PRESETS = {
    "tight": {
        "target_vol": 0.15, "vol_lookback": 20, "trend_lookback": 20,
        "trend_exposure": 0.70, "dd_soft": 0.02, "dd_hard": 0.03,
        "soft_exposure": 0.80, "hard_exposure": 0.30,
    },
    "moderate": {
        "target_vol": 0.18, "vol_lookback": 20, "trend_lookback": 20,
        "trend_exposure": 0.65, "dd_soft": 0.025, "dd_hard": 0.04,
        "soft_exposure": 0.70, "hard_exposure": 0.35,
    },
    "none": {"enabled": False},
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="config/models/alpha158_momentum_volume_k6_dd10_overlay.yaml")
    parser.add_argument("--search-name", default="daily_freq_search_v1")
    parser.add_argument("--bundles", default="", help="逗号分隔 bundle 名，空=全部")
    parser.add_argument("--engine", default="qlib")
    parser.add_argument("--topk-values", default="", help="逗号分隔 topk 值，覆盖默认")
    parser.add_argument("--sticky-values", default="", help="逗号分隔 sticky 值，覆盖默认")
    parser.add_argument("--horizon-values", default="", help="逗号分隔 horizon 值，覆盖默认")
    parser.add_argument("--overlays", default="", help="逗号分隔 overlay 名，覆盖默认")
    parser.add_argument("--max-runs", type=int, default=0, help="0=不限")
    parser.add_argument("--min-sharpe", type=float, default=0.0, help="最低夏普过滤")
    parser.add_argument("--skip-train", action="store_true", help="跳过已训练的配置")
    return parser.parse_args()


def search_root(name: str) -> Path:
    return PROJECT_ROOT / "results" / "model_signals" / "daily_search" / name


def build_candidate_cfg(
    base_cfg: dict,
    search_name: str,
    bundle_name: str,
    features: list,
    topk: int,
    sticky: int,
    horizon: int,
    overlay_key: str,
) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["name"] = f"daily_{bundle_name}_k{topk}_s{sticky}_h{horizon}_{overlay_key}"
    cfg["data"] = {
        "source": "alpha158",
        "start_date": str(base_cfg["data"]["start_date"]),
        "end_date": str(base_cfg["data"]["end_date"]),
        "feature_columns": list(features),
        "alpha158": copy.deepcopy(ALPHA158_BASE),
    }
    cfg["label"]["horizon_days"] = horizon
    cfg["selection"]["freq"] = "day"
    cfg["selection"]["topk"] = topk
    cfg["selection"]["sticky"] = sticky
    cfg["selection"]["buffer"] = 2
    cfg["selection"]["score_smoothing_days"] = 3
    cfg["selection"]["min_hold_days"] = 3
    cfg["selection"]["churn_limit"] = 2
    cfg["selection"]["mode"] = "factor_topk"

    overlay = OVERLAY_PRESETS[overlay_key]
    if overlay_key != "none":
        cfg["overlay"]["enabled"] = True
        cfg["overlay"].update(overlay)
    else:
        cfg["overlay"] = overlay

    cfg["output"]["root"] = str(search_root(search_name) / cfg["name"])
    return cfg


def evaluate_candidate(cfg: dict, engine: str, skip_train: bool = False) -> dict:
    try:
        bundle_path = Path(cfg["output"]["root"]) / "model_bundle.pkl"
        if not (skip_train and bundle_path.exists()):
            train_from_config(cfg)
        score_summary = score_from_config(cfg)
        _, backtest_summary = backtest_from_config(cfg, engine=engine)
        return {"ok": True, "score_summary": score_summary, "backtest_summary": backtest_summary}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def main():
    args = parse_args()
    base_cfg = load_predictive_config(args.base_config)
    root = search_root(args.search_name)
    root.mkdir(parents=True, exist_ok=True)

    selected_bundles = (
        [n.strip() for n in args.bundles.split(",") if n.strip()]
        if args.bundles.strip() else list(FEATURE_BUNDLES.keys())
    )
    topk_values = [int(x.strip()) for x in args.topk_values.split(",") if x.strip()] if args.topk_values.strip() else DAILY_GRID["topk"]
    sticky_values = [int(x.strip()) for x in args.sticky_values.split(",") if x.strip()] if args.sticky_values.strip() else DAILY_GRID["sticky"]
    horizon_values = [int(x.strip()) for x in args.horizon_values.split(",") if x.strip()] if args.horizon_values.strip() else DAILY_GRID["horizon_days"]
    overlay_keys = [x.strip() for x in args.overlays.split(",") if x.strip()] if args.overlays.strip() else list(OVERLAY_PRESETS.keys())

    rows = []
    run_count = 0
    total_candidates = (
        len(selected_bundles)
        * len(topk_values)
        * len(sticky_values)
        * len(horizon_values)
        * len(overlay_keys)
    )
    print(f"[INFO] 共 {total_candidates} 个候选配置")

    for bundle_name in selected_bundles:
        features = FEATURE_BUNDLES[bundle_name]
        for topk in topk_values:
            for sticky in sticky_values:
                for horizon in horizon_values:
                    for overlay_key in overlay_keys:
                        run_count += 1
                        if args.max_runs > 0 and run_count > args.max_runs:
                            break
                        cfg = build_candidate_cfg(
                            base_cfg, args.search_name, bundle_name,
                            features, topk, sticky, horizon, overlay_key,
                        )
                        started = time.time()
                        print(f"[{run_count}/{total_candidates}] {cfg['name']}")
                        payload = evaluate_candidate(cfg, args.engine, args.skip_train)
                        elapsed = time.time() - started
                        if payload["ok"]:
                            bt = payload["backtest_summary"]
                            sharpe = bt["sharpe_ratio"]
                            if sharpe >= args.min_sharpe:
                                rows.append({
                                    "name": cfg["name"],
                                    "bundle": bundle_name,
                                    "features": len(features),
                                    "topk": topk,
                                    "sticky": sticky,
                                    "horizon": horizon,
                                    "overlay": overlay_key,
                                    "annual_return": bt["annual_return"],
                                    "max_drawdown": bt["max_drawdown"],
                                    "sharpe_ratio": sharpe,
                                    "fees": bt.get("total_fee_amount", 0),
                                    "elapsed_sec": round(elapsed, 2),
                                })
                                print(f"  ✅ ann={bt['annual_return']:.2%} sharpe={sharpe:.3f} dd={bt['max_drawdown']:.2%}")
                        else:
                            print(f"  ❌ {payload['error'][:80]}")
                        sys.stdout.flush()

    if not rows:
        print("[WARN] 无候选通过过滤条件")

    summary = pd.DataFrame(rows).sort_values(
        ["sharpe_ratio", "annual_return"],
        ascending=[False, False],
    )
    summary_path = root / "summary.csv"
    summary.to_csv(summary_path, index=False)

    best_json = {}
    if not summary.empty:
        best = summary.iloc[0]
        print(f"\n{'='*60}")
        print(f"🏆 最佳日频策略: {best['name']}")
        print(f"   年化: {best['annual_return']:.2%} | 夏普: {best['sharpe_ratio']:.3f} | 回撤: {best['max_drawdown']:.2%}")
        print(f"   因子: {best['bundle']}({best['features']}个) | topk={best['topk']} sticky={best['sticky']}")
        print(f"   horizon={best['horizon']}d | overlay={best['overlay']}")
        print(f"{'='*60}")
        best_json = best.to_dict()

    save_json({
        "search_name": args.search_name,
        "engine": args.engine,
        "total_candidates": total_candidates,
        "passed_filter": len(rows),
        "best": best_json,
    }, root / "summary.json")
    print(f"[OK] 搜索完成: {summary_path}")


if __name__ == "__main__":
    main()
