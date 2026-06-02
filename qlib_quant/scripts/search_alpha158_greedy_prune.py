#!/usr/bin/env python3
"""逐步压缩 Alpha158 小因子子集。"""

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


DEFAULT_BASE_FEATURES = ["ROC20", "RSV20", "RANK20", "CORD20", "VSUMD20"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-config",
        default="config/models/qvf_core_plus_fixed80_k6_overlay_fullspan_turnover_soft_pruned.yaml",
    )
    parser.add_argument(
        "--search-name",
        default="alpha158_momentum_volume_greedy_prune_v1",
    )
    parser.add_argument(
        "--engine",
        choices=["qlib", "pybroker"],
        default="qlib",
    )
    parser.add_argument(
        "--features",
        default=",".join(DEFAULT_BASE_FEATURES),
        help="起始特征列表，逗号分隔",
    )
    parser.add_argument(
        "--topks",
        default="6",
        help="每轮要测试的 topk，逗号分隔",
    )
    parser.add_argument(
        "--min-features",
        type=int,
        default=3,
        help="最少压到多少个因子",
    )
    return parser.parse_args()


def search_root(search_name: str) -> Path:
    return PROJECT_ROOT / "results" / "model_signals" / "alpha158_prune_runs" / search_name


def candidate_slug(round_idx: int, removed_feature: str, topk: int) -> str:
    return f"alpha158_prune_r{round_idx}_drop_{removed_feature.lower()}_k{topk}"


def build_candidate_cfg(
    base_cfg: dict,
    search_name: str,
    round_idx: int,
    features: list[str],
    removed_feature: str,
    topk: int,
) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["name"] = candidate_slug(round_idx, removed_feature, topk)
    cfg["data"] = {
        "source": "alpha158",
        "start_date": str(base_cfg["data"]["start_date"]),
        "end_date": str(base_cfg["data"]["end_date"]),
        "feature_columns": list(features),
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


def choose_best(rows: list[dict]) -> dict:
    return sorted(
        rows,
        key=lambda row: (
            float(row["annual_return"]),
            float(row["sharpe_ratio"]),
            float(row["max_drawdown"]),
            float(row["valid_mean_rank_ic"]),
        ),
        reverse=True,
    )[0]


def main():
    args = parse_args()
    base_cfg = load_predictive_config(args.base_config)
    root = search_root(args.search_name)
    root.mkdir(parents=True, exist_ok=True)

    current_features = [x.strip() for x in str(args.features).split(",") if x.strip()]
    initial_features = list(current_features)
    topk_values = [int(x.strip()) for x in str(args.topks).split(",") if x.strip()]
    min_features = max(int(args.min_features), 1)

    if len(current_features) <= min_features:
        raise ValueError("起始特征数必须大于 min-features")

    all_rows: list[dict] = []
    accepted_steps: list[dict] = []
    round_idx = 1

    while len(current_features) > min_features:
        round_rows = []
        print(f"[ROUND {round_idx}] current_features={','.join(current_features)}")
        for removed_feature in current_features:
            next_features = [f for f in current_features if f != removed_feature]
            for topk in topk_values:
                cfg = build_candidate_cfg(
                    base_cfg=base_cfg,
                    search_name=args.search_name,
                    round_idx=round_idx,
                    features=next_features,
                    removed_feature=removed_feature,
                    topk=topk,
                )
                started_at = time.time()
                print(
                    f"[RUN] {cfg['name']} remove={removed_feature} "
                    f"feature_count={len(next_features)} topk={topk}"
                )
                payload = evaluate_candidate(cfg, engine=args.engine)
                elapsed = time.time() - started_at
                metrics = payload["training_summary"].get("metrics", {})
                backtest = payload["backtest_summary"]
                row = {
                    "round": round_idx,
                    "removed_feature": removed_feature,
                    "name": cfg["name"],
                    "feature_count": len(next_features),
                    "features": ",".join(next_features),
                    "topk": topk,
                    "valid_mean_rank_ic": float(metrics.get("valid_mean_rank_ic", 0.0)),
                    "valid_mean_pearson_ic": float(metrics.get("valid_mean_pearson_ic", 0.0)),
                    "annual_return": float(backtest["annual_return"]),
                    "max_drawdown": float(backtest["max_drawdown"]),
                    "sharpe_ratio": float(backtest["sharpe_ratio"]),
                    "elapsed_sec": round(elapsed, 2),
                    "results_file": backtest.get("results_file"),
                }
                round_rows.append(row)
                all_rows.append(row)

        best = choose_best(round_rows)
        accepted_steps.append(best)
        current_features = [x for x in str(best["features"]).split(",") if x]
        print(
            f"[ACCEPT] round={round_idx} name={best['name']} "
            f"annual={best['annual_return']:.2%} max_dd={best['max_drawdown']:.2%} "
            f"sharpe={best['sharpe_ratio']:.3f}"
        )
        round_idx += 1

    summary = pd.DataFrame(all_rows).sort_values(
        ["feature_count", "annual_return", "sharpe_ratio", "max_drawdown"],
        ascending=[False, False, False, False],
    )
    summary_path = root / "summary.csv"
    summary.to_csv(summary_path, index=False)
    accepted_path = root / "accepted_steps.csv"
    pd.DataFrame(accepted_steps).to_csv(accepted_path, index=False)

    final_payload = {
        "base_config": str(Path(args.base_config).resolve()),
        "search_name": args.search_name,
        "engine": args.engine,
        "initial_features": initial_features,
        "summary_csv": str(summary_path),
        "accepted_steps_csv": str(accepted_path),
        "accepted_steps": accepted_steps,
        "final_candidate": accepted_steps[-1] if accepted_steps else None,
    }
    save_json(final_payload, root / "summary.json")
    print(f"[OK] 搜索完成: {summary_path}")
    if accepted_steps:
        final = accepted_steps[-1]
        print(
            "[INFO] 最终候选 "
            f"{final['name']} annual={final['annual_return']:.2%} "
            f"max_dd={final['max_drawdown']:.2%} sharpe={final['sharpe_ratio']:.3f}"
        )


if __name__ == "__main__":
    main()
