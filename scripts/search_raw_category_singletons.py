#!/usr/bin/env python3
"""搜索“每类一个原始因子”的 Qlib 风格小因子配置。"""

from __future__ import annotations

import argparse
import copy
import itertools
import math
import sys
import time
from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.modeling.predictive_signal import (  # noqa: E402
    backtest_from_config,
    load_predictive_config,
    save_json,
    score_from_config,
    train_from_config,
)


ALPHA158_TECH = {"ROC20", "CORD20"}


CATEGORY_CANDIDATES = {
    "value_cashflow": ["fcff_to_mv", "ocf_to_ev"],
    "profitability_quality": ["roe_fina"],
    "balance_sheet": ["current_ratio_fina"],
    "growth_cashflow": ["n_cashflow_act", "operate_profit_inc"],
    "tech_confirmation": ["ROC20", "CORD20"],
}


TOPK_DROPOUT_PRESETS = {
    "qlib_k8_d2": {"topk": 8, "n_drop": 2},
    "qlib_k6_d2": {"topk": 6, "n_drop": 2},
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-config",
        default="config/models/hybrid10_cashflow_quality10_qlib_k8_d2.yaml",
        help="小因子搜索基线配置",
    )
    parser.add_argument(
        "--search-name",
        default="raw_category_singletons_v1",
        help="结果输出目录名",
    )
    parser.add_argument(
        "--engine",
        choices=["qlib", "pybroker"],
        default="qlib",
        help="回测引擎",
    )
    parser.add_argument(
        "--presets",
        default="qlib_k8_d2",
        help="逗号分隔的 Qlib 风格选股 preset",
    )
    return parser.parse_args()


def search_root(search_name: str) -> Path:
    return PROJECT_ROOT / "results" / "model_signals" / "search_runs" / search_name


def _annualized_return(total_return: float, start_date: pd.Timestamp, end_date: pd.Timestamp) -> float:
    days = max((end_date - start_date).days, 1)
    gross = 1.0 + float(total_return)
    if gross <= 0:
        return -1.0
    return gross ** (365.0 / float(days)) - 1.0


def evaluate_overlay_window(path: str | Path, start_date: str, end_date: str) -> dict:
    frame = pd.read_csv(path, parse_dates=["date"])
    mask = (frame["date"] >= pd.Timestamp(start_date)) & (frame["date"] <= pd.Timestamp(end_date))
    frame = frame.loc[mask].copy()
    if frame.empty:
        return {
            "total_return": 0.0,
            "annual_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
        }
    first_value = float(frame["portfolio_value"].iloc[0]) / (1.0 + float(frame["overlay_return"].iloc[0]))
    last_value = float(frame["portfolio_value"].iloc[-1])
    total_return = last_value / first_value - 1.0
    annual_return = _annualized_return(
        total_return,
        pd.Timestamp(frame["date"].iloc[0]),
        pd.Timestamp(frame["date"].iloc[-1]),
    )
    max_drawdown = float(frame["drawdown"].min())
    daily_returns = frame["overlay_return"].astype(float)
    daily_std = float(daily_returns.std(ddof=0))
    sharpe = 0.0 if math.isclose(daily_std, 0.0) else float(daily_returns.mean()) / daily_std * math.sqrt(252.0)
    return {
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "max_drawdown": float(max_drawdown),
        "sharpe_ratio": float(sharpe),
    }


def candidate_rows():
    keys = list(CATEGORY_CANDIDATES.keys())
    for values in itertools.product(*(CATEGORY_CANDIDATES[key] for key in keys)):
        row = dict(zip(keys, values))
        features = list(values)
        tag = "_".join(str(value).replace("_to_", "_").replace("_fina", "").replace("_act", "") for value in values)
        yield tag, row, features


def build_candidate_cfg(base_cfg: dict, search_name: str, tag: str, features: list[str], preset_name: str) -> dict:
    preset = dict(TOPK_DROPOUT_PRESETS[preset_name])
    topk = int(preset["topk"])
    n_drop = int(preset["n_drop"])

    cfg = copy.deepcopy(base_cfg)
    cfg["name"] = f"raw5_{tag}_{preset_name}"
    data_cfg = cfg.setdefault("data", {})
    data_cfg["source"] = "hybrid"
    data_cfg["parquet_feature_columns"] = [col for col in features if col not in ALPHA158_TECH]
    data_cfg["alpha158_feature_columns"] = [col for col in features if col in ALPHA158_TECH]

    selection_cfg = cfg.setdefault("selection", {})
    selection_cfg["topk"] = topk
    selection_cfg["sticky"] = max(topk - n_drop, 0)
    selection_cfg["churn_limit"] = n_drop
    selection_cfg["buffer"] = 0
    selection_cfg["threshold"] = 0.0
    selection_cfg["margin_stable"] = False
    selection_cfg["score_smoothing_days"] = 1
    selection_cfg["entry_persist_days"] = 1
    selection_cfg["exit_persist_days"] = 1
    selection_cfg["min_hold_days"] = 0
    selection_cfg["entry_rank"] = None
    selection_cfg["exit_rank"] = None

    cfg.setdefault("output", {})["root"] = str(search_root(search_name) / cfg["name"])
    return cfg


def write_candidate_yaml(cfg: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def evaluate_candidate(cfg: dict, engine: str) -> dict:
    train_summary = train_from_config(cfg)
    score_summary = score_from_config(cfg)
    _, backtest_summary = backtest_from_config(cfg, engine=engine)
    oos_window = evaluate_overlay_window(
        Path(cfg["output"]["root"]) / "overlay_results.csv",
        "2024-01-01",
        str(cfg["data"]["end_date"]),
    )
    return {
        "training_summary": train_summary,
        "scoring_summary": score_summary,
        "backtest_summary": backtest_summary,
        "oos_window": oos_window,
    }


def main():
    args = parse_args()
    base_cfg = load_predictive_config(args.base_config)
    root = search_root(args.search_name)
    root.mkdir(parents=True, exist_ok=True)
    selected_presets = [name.strip() for name in str(args.presets).split(",") if name.strip()]

    rows = []
    for tag, category_map, features in candidate_rows():
        for preset_name in selected_presets:
            cfg = build_candidate_cfg(base_cfg, args.search_name, tag, features, preset_name)
            write_candidate_yaml(cfg, root / "configs" / f"{cfg['name']}.yaml")
            started_at = time.time()
            print(f"[RUN] {cfg['name']} features={','.join(features)}")
            payload = evaluate_candidate(cfg, engine=args.engine)
            elapsed = time.time() - started_at
            metrics = payload["training_summary"].get("metrics", {})
            backtest = payload["backtest_summary"]
            oos = payload["oos_window"]
            rows.append(
                {
                    "name": cfg["name"],
                    "preset": preset_name,
                    "features": ",".join(features),
                    "value_cashflow": category_map["value_cashflow"],
                    "profitability_quality": category_map["profitability_quality"],
                    "balance_sheet": category_map["balance_sheet"],
                    "growth_cashflow": category_map["growth_cashflow"],
                    "tech_confirmation": category_map["tech_confirmation"],
                    "valid_mean_rank_ic": float(metrics.get("valid_mean_rank_ic", 0.0)),
                    "full_annual_return": float(backtest["annual_return"]),
                    "full_max_drawdown": float(backtest["max_drawdown"]),
                    "full_sharpe_ratio": float(backtest["sharpe_ratio"]),
                    "oos_annual_return": float(oos["annual_return"]),
                    "oos_max_drawdown": float(oos["max_drawdown"]),
                    "oos_sharpe_ratio": float(oos["sharpe_ratio"]),
                    "elapsed_sec": round(elapsed, 2),
                }
            )

    summary = pd.DataFrame(rows).sort_values(
        ["oos_annual_return", "oos_sharpe_ratio", "full_annual_return"],
        ascending=[False, False, False],
    )
    summary_path = root / "summary.csv"
    summary.to_csv(summary_path, index=False)
    save_json(
        {
            "base_config": str(Path(args.base_config).resolve()),
            "search_name": args.search_name,
            "engine": args.engine,
            "summary_csv": str(summary_path),
            "best": summary.iloc[0].to_dict() if not summary.empty else None,
        },
        root / "summary.json",
    )
    print(f"[OK] 搜索完成: {summary_path}")
    if not summary.empty:
        best = summary.iloc[0]
        print(
            "[INFO] 最优候选 "
            f"{best['name']} oos_annual={best['oos_annual_return']:.2%} "
            f"full_annual={best['full_annual_return']:.2%}"
        )


if __name__ == "__main__":
    main()
