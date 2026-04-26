#!/usr/bin/env python3
"""搜索 10 因子 hybrid 配置，并按 Qlib TopkDropout 思路做选股。"""

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

from modules.modeling.predictive_signal import (  # noqa: E402
    backtest_from_config,
    load_predictive_config,
    save_json,
    score_from_config,
    train_from_config,
)


ALPHA158_TECH = {"ROC20", "RANK20", "CORD20"}


FEATURE_BUNDLES = {
    "balanced_core10": [
        "ebit_to_mv",
        "roe_fina",
        "current_ratio_fina",
        "operate_profit_inc",
        "ROC20",
        "CORD20",
        "rank_value_profit_core",
        "rank_growth_quality_core",
        "rank_flow_momentum_core",
        "qvf_core_interaction",
    ],
    "balanced_value10": [
        "fcff_to_mv",
        "roa_fina",
        "debt_to_assets_fina",
        "total_revenue_inc",
        "ROC20",
        "RANK20",
        "rank_value_profit_core",
        "rank_growth_quality_core",
        "rank_balance_core",
        "qvf_core_interaction",
    ],
    "balanced_quality10": [
        "ocf_to_ev",
        "roe_dt_fina",
        "current_ratio_fina",
        "operate_profit_inc",
        "ROC20",
        "CORD20",
        "rank_value_profit_core",
        "rank_growth_quality_core",
        "rank_balance_core",
        "qvf_core_alpha",
    ],
    "tech_core10": [
        "ROC20",
        "RANK20",
        "CORD20",
        "ebit_to_mv",
        "roe_fina",
        "operate_profit_inc",
        "rank_value_profit_core",
        "rank_growth_quality_core",
        "rank_flow_momentum_core",
        "qvf_core_interaction",
    ],
    "fundamental_blend10": [
        "ebit_to_mv",
        "fcff_to_mv",
        "roe_fina",
        "roa_fina",
        "current_ratio_fina",
        "operate_profit_inc",
        "total_revenue_inc",
        "ROC20",
        "CORD20",
        "qvf_core_interaction",
    ],
    "cashflow_quality10": [
        "ocf_to_ev",
        "fcff_to_mv",
        "roe_fina",
        "current_ratio_fina",
        "n_cashflow_act",
        "ROC20",
        "CORD20",
        "rank_value_profit_core",
        "rank_balance_core",
        "qvf_core_interaction",
    ],
    "core_anchor10": [
        "ROC20",
        "CORD20",
        "rank_value_profit_core",
        "rank_growth_quality_core",
        "rank_flow_momentum_core",
        "rank_balance_core",
        "ebit_to_mv",
        "roe_fina",
        "operate_profit_inc",
        "current_ratio_fina",
    ],
    "efficient10": [
        "ROC20",
        "rank_value_profit_core",
        "ebit_to_mv",
        "operate_profit_inc",
        "CORD20",
        "fcff_to_mv",
        "rank_growth_quality_core",
        "roe_fina",
        "total_revenue_inc",
        "RANK20",
    ],
}


TOPK_DROPOUT_PRESETS = {
    "qlib_k6_d1": {"topk": 6, "n_drop": 1},
    "qlib_k6_d2": {"topk": 6, "n_drop": 2},
    "qlib_k8_d2": {"topk": 8, "n_drop": 2},
    "qlib_k8_d3": {"topk": 8, "n_drop": 3},
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-config",
        default="config/models/qvf_alpha158_hybrid21_best_k6.yaml",
        help="10 因子搜索基线配置",
    )
    parser.add_argument(
        "--search-name",
        default="hybrid10_qlib_style_v1",
        help="搜索结果目录名",
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
        help="逗号分隔的 bundle 名称；为空则跑默认全部",
    )
    parser.add_argument(
        "--presets",
        default="qlib_k6_d1,qlib_k6_d2,qlib_k8_d2,qlib_k8_d3",
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
    if frame.empty:
        return {
            "start_date": start_date,
            "end_date": end_date,
            "days": 0,
            "total_return": 0.0,
            "annual_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
        }

    mask = (frame["date"] >= pd.Timestamp(start_date)) & (frame["date"] <= pd.Timestamp(end_date))
    frame = frame.loc[mask].copy()
    if frame.empty:
        return {
            "start_date": start_date,
            "end_date": end_date,
            "days": 0,
            "total_return": 0.0,
            "annual_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
        }

    first_value = float(frame["portfolio_value"].iloc[0]) / (1.0 + float(frame["overlay_return"].iloc[0]))
    last_value = float(frame["portfolio_value"].iloc[-1])
    total_return = last_value / first_value - 1.0
    start_ts = pd.Timestamp(frame["date"].iloc[0])
    end_ts = pd.Timestamp(frame["date"].iloc[-1])
    annual_return = _annualized_return(total_return, start_ts, end_ts)
    max_drawdown = float(frame["drawdown"].min())

    daily_returns = frame["overlay_return"].astype(float)
    daily_std = float(daily_returns.std(ddof=0))
    if math.isclose(daily_std, 0.0):
        sharpe = 0.0
    else:
        sharpe = float(daily_returns.mean()) / daily_std * math.sqrt(252.0)

    return {
        "start_date": start_date,
        "end_date": end_date,
        "days": int(len(frame)),
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "max_drawdown": float(max_drawdown),
        "sharpe_ratio": float(sharpe),
    }


def build_candidate_cfg(base_cfg: dict, search_name: str, bundle_name: str, preset_name: str) -> dict:
    features = list(FEATURE_BUNDLES[bundle_name])
    if len(features) != 10:
        raise ValueError(f"{bundle_name} 不是 10 因子配置: {len(features)}")

    preset = dict(TOPK_DROPOUT_PRESETS[preset_name])
    topk = int(preset["topk"])
    n_drop = int(preset["n_drop"])
    sticky = max(topk - n_drop, 0)

    cfg = copy.deepcopy(base_cfg)
    cfg["name"] = f"hybrid10_{bundle_name}_{preset_name}"

    data_cfg = cfg.setdefault("data", {})
    data_cfg["source"] = "hybrid"
    data_cfg["parquet_feature_columns"] = [col for col in features if col not in ALPHA158_TECH]
    data_cfg["alpha158_feature_columns"] = [col for col in features if col in ALPHA158_TECH]

    selection_cfg = cfg.setdefault("selection", {})
    selection_cfg["topk"] = topk
    selection_cfg["sticky"] = sticky
    selection_cfg["churn_limit"] = n_drop
    selection_cfg["buffer"] = 0
    selection_cfg["threshold"] = 0.0
    selection_cfg["margin_stable"] = False
    selection_cfg["score_smoothing_days"] = 1
    selection_cfg["entry_rank"] = None
    selection_cfg["exit_rank"] = None
    selection_cfg["entry_persist_days"] = 1
    selection_cfg["exit_persist_days"] = 1
    selection_cfg["min_hold_days"] = 0

    output_cfg = cfg.setdefault("output", {})
    output_cfg["root"] = str(search_root(search_name) / cfg["name"])
    return cfg


def write_candidate_yaml(cfg: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def evaluate_candidate(cfg: dict, engine: str) -> dict:
    train_summary = train_from_config(cfg)
    score_summary = score_from_config(cfg)
    _, backtest_summary = backtest_from_config(cfg, engine=engine)
    full_window = evaluate_overlay_window(
        Path(cfg["output"]["root"]) / "overlay_results.csv",
        str(cfg["data"]["start_date"]),
        str(cfg["data"]["end_date"]),
    )
    oos_window = evaluate_overlay_window(
        Path(cfg["output"]["root"]) / "overlay_results.csv",
        "2024-01-01",
        str(cfg["data"]["end_date"]),
    )
    return {
        "training_summary": train_summary,
        "scoring_summary": score_summary,
        "backtest_summary": backtest_summary,
        "full_window": full_window,
        "oos_window": oos_window,
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
    selected_presets = [name.strip() for name in str(args.presets).split(",") if name.strip()]

    rows = []
    for bundle_name in selected_bundles:
        for preset_name in selected_presets:
            cfg = build_candidate_cfg(base_cfg, args.search_name, bundle_name, preset_name)
            write_candidate_yaml(cfg, root / "configs" / f"{cfg['name']}.yaml")
            started_at = time.time()
            print(
                f"[RUN] {cfg['name']} factors={len(FEATURE_BUNDLES[bundle_name])} "
                f"topk={cfg['selection']['topk']} n_drop={cfg['selection']['churn_limit']}"
            )
            payload = evaluate_candidate(cfg, engine=args.engine)
            elapsed = time.time() - started_at

            metrics = payload["training_summary"].get("metrics", {})
            backtest = payload["backtest_summary"]
            full_window = payload["full_window"]
            oos_window = payload["oos_window"]
            rows.append(
                {
                    "name": cfg["name"],
                    "bundle": bundle_name,
                    "preset": preset_name,
                    "feature_count": len(FEATURE_BUNDLES[bundle_name]),
                    "features": ",".join(FEATURE_BUNDLES[bundle_name]),
                    "topk": int(cfg["selection"]["topk"]),
                    "n_drop": int(cfg["selection"]["churn_limit"]),
                    "sticky": int(cfg["selection"]["sticky"]),
                    "valid_mean_rank_ic": float(metrics.get("valid_mean_rank_ic", 0.0)),
                    "full_annual_return": float(backtest["annual_return"]),
                    "full_max_drawdown": float(backtest["max_drawdown"]),
                    "full_sharpe_ratio": float(backtest["sharpe_ratio"]),
                    "oos_annual_return": float(oos_window["annual_return"]),
                    "oos_max_drawdown": float(oos_window["max_drawdown"]),
                    "oos_sharpe_ratio": float(oos_window["sharpe_ratio"]),
                    "oos_total_return": float(oos_window["total_return"]),
                    "elapsed_sec": round(elapsed, 2),
                    "results_file": backtest.get("results_file"),
                }
            )

    summary = pd.DataFrame(rows).sort_values(
        ["oos_sharpe_ratio", "oos_annual_return", "full_annual_return", "oos_max_drawdown"],
        ascending=[False, False, False, False],
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
            f"{best['name']} oos_annual={best['oos_annual_return']:.2%} "
            f"oos_max_dd={best['oos_max_drawdown']:.2%} "
            f"full_annual={best['full_annual_return']:.2%}"
        )


if __name__ == "__main__":
    main()
