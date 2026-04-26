#!/usr/bin/env python3
"""搜索 Alpha158 小因子子集（每组 <= 5 因子）。"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.modeling.predictive_signal import (  # noqa: E402
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
            "ROC",
            "MA",
            "STD",
            "BETA",
            "RSQR",
            "RESI",
            "MAX",
            "LOW",
            "RANK",
            "RSV",
            "IMAX",
            "IMIN",
            "IMXD",
            "CORR",
            "CORD",
            "CNTP",
            "CNTD",
            "SUMP",
            "VSTD",
            "WVMA",
            "VSUMD",
        ],
    },
}


FEATURE_BUNDLES = {
    "trend_core": ["MA20", "ROC20", "BETA20", "RSQR20", "RESI20"],
    "momentum_volume": ["ROC20", "RSV20", "RANK20", "CORD20", "VSUMD20"],
    "price_volume": ["CORR20", "CORD20", "WVMA20", "VSUMD20", "VSTD20"],
    "breakout_range": ["MAX20", "MIN20", "RSV20", "IMAX20", "IMIN20"],
    "behavioral": ["CNTP20", "CNTD20", "SUMP20", "VSUMD20", "ROC20"],
    "kbar_flow": ["KMID", "KLEN", "KSFT", "CORD20", "VSUMD20"],
}


TOPK_VARIANTS = [4, 6]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-config",
        default="config/models/qvf_core_plus_fixed80_k6_overlay_fullspan_turnover_soft_pruned.yaml",
        help="作为回测壳的基础配置",
    )
    parser.add_argument(
        "--search-name",
        default="alpha158_small_factor_v1",
        help="结果输出目录名",
    )
    parser.add_argument(
        "--engine",
        choices=["qlib", "pybroker"],
        default="qlib",
        help="回测引擎",
    )
    parser.add_argument(
        "--bundles",
        default="",
        help="逗号分隔的 bundle 名称；为空则跑全部",
    )
    parser.add_argument(
        "--topks",
        default="4,6",
        help="逗号分隔的 topk 列表，默认 4,6",
    )
    return parser.parse_args()


def search_root(search_name: str) -> Path:
    return PROJECT_ROOT / "results" / "model_signals" / "alpha158_search_runs" / search_name


def build_candidate_cfg(base_cfg: dict, search_name: str, bundle_name: str, topk: int) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["name"] = f"alpha158_{bundle_name}_k{topk}"
    cfg["data"] = {
        "source": "alpha158",
        "start_date": str(base_cfg["data"]["start_date"]),
        "end_date": str(base_cfg["data"]["end_date"]),
        "feature_columns": list(FEATURE_BUNDLES[bundle_name]),
        "alpha158": copy.deepcopy(ALPHA158_BASE),
    }
    cfg["selection"]["topk"] = int(topk)
    cfg["output"]["root"] = str(search_root(search_name) / cfg["name"])
    return cfg


def evaluate_candidate(cfg: dict, engine: str) -> dict:
    train_summary = train_from_config(cfg)
    score_summary = score_from_config(cfg)
    _, backtest_summary = backtest_from_config(cfg, engine=engine)
    return {
        "training_summary": train_summary,
        "scoring_summary": score_summary,
        "backtest_summary": backtest_summary,
    }


def main():
    args = parse_args()
    base_cfg = load_predictive_config(args.base_config)
    root = search_root(args.search_name)
    root.mkdir(parents=True, exist_ok=True)
    selected_bundles = (
        [name.strip() for name in str(args.bundles).split(",") if name.strip()]
        if str(args.bundles).strip()
        else list(FEATURE_BUNDLES.keys())
    )
    topk_values = [int(x.strip()) for x in str(args.topks).split(",") if x.strip()]

    rows = []
    for bundle_name in selected_bundles:
        features = FEATURE_BUNDLES[bundle_name]
        for topk in topk_values:
            cfg = build_candidate_cfg(base_cfg, args.search_name, bundle_name, topk)
            started_at = time.time()
            print(f"[RUN] {cfg['name']} features={len(features)} topk={topk}")
            payload = evaluate_candidate(cfg, engine=args.engine)
            elapsed = time.time() - started_at
            metrics = payload["training_summary"].get("metrics", {})
            backtest = payload["backtest_summary"]
            rows.append(
                {
                    "name": cfg["name"],
                    "bundle": bundle_name,
                    "feature_count": len(features),
                    "features": ",".join(features),
                    "topk": topk,
                    "valid_mean_rank_ic": float(metrics.get("valid_mean_rank_ic", 0.0)),
                    "valid_mean_pearson_ic": float(metrics.get("valid_mean_pearson_ic", 0.0)),
                    "annual_return": float(backtest["annual_return"]),
                    "max_drawdown": float(backtest["max_drawdown"]),
                    "sharpe_ratio": float(backtest["sharpe_ratio"]),
                    "elapsed_sec": round(elapsed, 2),
                    "results_file": backtest.get("results_file"),
                }
            )

    summary = pd.DataFrame(rows).sort_values(
        ["annual_return", "sharpe_ratio", "max_drawdown"],
        ascending=[False, False, False],
    )
    summary_path = root / "summary.csv"
    summary.to_csv(summary_path, index=False)
    payload = {
        "base_config": str(Path(args.base_config).resolve()),
        "search_name": args.search_name,
        "engine": args.engine,
        "summary_csv": str(summary_path),
        "best": summary.iloc[0].to_dict() if not summary.empty else None,
    }
    save_json(payload, root / "summary.json")
    print(f"[OK] 搜索完成: {summary_path}")
    if not summary.empty:
        best = summary.iloc[0]
        print(
            "[INFO] 最优候选 "
            f"{best['name']} annual={best['annual_return']:.2%} "
            f"max_dd={best['max_drawdown']:.2%} sharpe={best['sharpe_ratio']:.3f}"
        )


if __name__ == "__main__":
    main()
