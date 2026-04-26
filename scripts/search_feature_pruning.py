#!/usr/bin/env python3
"""按类别逐步精简模型因子，并在每一步验证收益/回撤约束。"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.modeling.feature_pruning import (  # noqa: E402
    CATEGORY_ORDER,
    FEATURE_CATEGORY_SPECS,
    category_status_rows,
    extract_feature_importance_map,
    ordered_category_removals,
)
from modules.modeling.predictive_signal import (  # noqa: E402
    backtest_from_config,
    backtest_summary_path,
    load_model_bundle,
    load_predictive_config,
    model_bundle_path,
    save_json,
    score_from_config,
    scoring_summary_path,
    train_from_config,
    training_summary_path,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/models/qvf_core_plus_fixed80_k6_overlay_fullspan_turnover_soft.yaml",
        help="起始配置 YAML",
    )
    parser.add_argument(
        "--search-name",
        default="qvf_core_turnover_soft_feature_prune_v1",
        help="搜索输出目录名",
    )
    parser.add_argument(
        "--engine",
        choices=["qlib", "pybroker"],
        default="qlib",
        help="回测引擎，默认 qlib",
    )
    parser.add_argument(
        "--min-annual",
        type=float,
        default=0.25,
        help="候选最小年化阈值，默认 0.25",
    )
    parser.add_argument(
        "--max-drawdown",
        type=float,
        default=0.15,
        help="候选最大回撤阈值（正数），默认 0.15",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=16,
        help="最多接受多少次删因子，默认 16",
    )
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=3,
        help="每一步每个类别最多尝试多少个最低重要性候选，默认 3",
    )
    return parser.parse_args()


def _clean_cfg_for_yaml(cfg: dict) -> dict:
    payload = copy.deepcopy(cfg)
    payload.pop("_config_path", None)
    return payload


def _write_yaml(cfg: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(_clean_cfg_for_yaml(cfg), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


def _candidate_slug(step: int, category: str, dropped_feature: str) -> str:
    return f"s{step:02d}__{category}__drop_{dropped_feature}"


def _search_root(search_name: str) -> Path:
    return PROJECT_ROOT / "results" / "model_signals" / "feature_pruning_runs" / search_name


def _candidate_output_root(search_name: str, slug: str) -> Path:
    return _search_root(search_name) / "models" / slug


def _candidate_config_path(search_name: str, slug: str) -> Path:
    return _search_root(search_name) / "configs" / f"{slug}.yaml"


def _feature_column_keys(cfg: dict) -> list[str]:
    source = str(cfg.get("data", {}).get("source", "parquet")).lower()
    if source == "hybrid":
        return ["parquet_feature_columns", "alpha158_feature_columns"]
    return ["feature_columns"]


def _config_feature_columns(cfg: dict) -> list[str]:
    data_cfg = cfg["data"]
    columns: list[str] = []
    for key in _feature_column_keys(cfg):
        columns.extend(str(name) for name in data_cfg.get(key, []))
    return columns


def _drop_feature_from_cfg(cfg: dict, dropped_feature: str) -> None:
    data_cfg = cfg["data"]
    for key in _feature_column_keys(cfg):
        if dropped_feature in data_cfg.get(key, []):
            data_cfg[key] = [name for name in data_cfg[key] if name != dropped_feature]
            return
    raise KeyError(f"配置中找不到待删除因子: {dropped_feature}")


def _load_existing_eval(cfg: dict) -> dict | None:
    train_path = training_summary_path(cfg)
    score_path = scoring_summary_path(cfg)
    backtest_path = backtest_summary_path(cfg)
    bundle_path = model_bundle_path(cfg)
    if not (train_path.exists() and score_path.exists() and backtest_path.exists() and bundle_path.exists()):
        return None

    train_summary = pd.read_json(train_path)
    score_summary = pd.read_json(score_path)
    backtest_summary = pd.read_json(backtest_path)
    bundle = load_model_bundle(bundle_path)
    importance_map = extract_feature_importance_map(bundle["model"], bundle["feature_columns"])
    return {
        "training_summary": train_summary.to_dict(),
        "scoring_summary": score_summary.to_dict(),
        "backtest_summary": backtest_summary.to_dict(),
        "importance_map": importance_map,
    }


def evaluate_config(cfg: dict, engine: str) -> dict:
    train_summary = train_from_config(cfg)
    score_summary = score_from_config(cfg)
    _, backtest_summary = backtest_from_config(cfg, engine=engine)
    bundle = load_model_bundle(model_bundle_path(cfg))
    importance_map = extract_feature_importance_map(bundle["model"], bundle["feature_columns"])
    return {
        "training_summary": train_summary,
        "scoring_summary": score_summary,
        "backtest_summary": backtest_summary,
        "importance_map": importance_map,
    }


def eval_to_row(
    cfg: dict,
    eval_payload: dict,
    category: str,
    dropped_feature: str,
    step: int,
    accepted: bool,
    accepted_name: str,
) -> dict:
    train_summary = eval_payload["training_summary"]
    scoring_summary = eval_payload["scoring_summary"]
    backtest_summary = eval_payload["backtest_summary"]
    training_metrics = train_summary.get("metrics", {})
    feature_columns = _config_feature_columns(cfg)
    category_rows = category_status_rows(feature_columns)
    row = {
        "config_name": cfg["name"],
        "step": int(step),
        "category": category,
        "dropped_feature": dropped_feature,
        "accepted": bool(accepted),
        "accepted_name": accepted_name,
        "feature_count": len(feature_columns),
        "annual_return": float(backtest_summary["annual_return"]),
        "max_drawdown": float(backtest_summary["max_drawdown"]),
        "sharpe_ratio": float(backtest_summary["sharpe_ratio"]),
        "valid_mean_rank_ic": float(training_metrics.get("valid_mean_rank_ic", 0.0)),
        "valid_mean_pearson_ic": float(training_metrics.get("valid_mean_pearson_ic", 0.0)),
        "score_dates": int(scoring_summary.get("selection_dates", 0)),
        "output_root": str(cfg["output"]["root"]),
        "feature_columns": ",".join(feature_columns),
    }
    for status in category_rows:
        row[f"{status['category']}_count"] = status["count"]
    return row


def is_valid_candidate(eval_payload: dict, min_annual: float, max_drawdown: float) -> bool:
    summary = eval_payload["backtest_summary"]
    return (
        float(summary["annual_return"]) >= float(min_annual)
        and float(summary["max_drawdown"]) >= -abs(float(max_drawdown))
    )


def candidate_sort_key(row: dict) -> tuple:
    return (
        float(row["annual_return"]),
        float(row["max_drawdown"]),
        float(row["valid_mean_rank_ic"]),
        -int(row["feature_count"]),
    )


def build_candidate_cfg(current_cfg: dict, search_name: str, step: int, category: str, dropped_feature: str) -> dict:
    slug = _candidate_slug(step, category, dropped_feature)
    cfg = copy.deepcopy(current_cfg)
    cfg["name"] = slug
    _drop_feature_from_cfg(cfg, dropped_feature)
    cfg["output"]["root"] = str(_candidate_output_root(search_name, slug))
    return cfg


def main():
    args = parse_args()
    base_cfg = load_predictive_config(args.config)
    search_root = _search_root(args.search_name)
    search_root.mkdir(parents=True, exist_ok=True)

    current_cfg = copy.deepcopy(base_cfg)
    current_cfg["name"] = f"{base_cfg['name']}__baseline"
    current_cfg["output"]["root"] = str(_candidate_output_root(args.search_name, current_cfg["name"]))
    _write_yaml(current_cfg, _candidate_config_path(args.search_name, current_cfg["name"]))

    rows: list[dict] = []
    accepted_steps: list[dict] = []

    print(f"[INFO] 基线配置: {base_cfg['name']}")
    baseline_eval = evaluate_config(current_cfg, engine=args.engine)
    baseline_row = eval_to_row(
        cfg=current_cfg,
        eval_payload=baseline_eval,
        category="baseline",
        dropped_feature="",
        step=0,
        accepted=True,
        accepted_name=current_cfg["name"],
    )
    rows.append(baseline_row)
    accepted_steps.append(baseline_row)
    current_eval = baseline_eval

    accepted_step_count = 0
    for category in CATEGORY_ORDER:
        target_count = int(FEATURE_CATEGORY_SPECS[category]["target_count"])
        while True:
            grouped = {
                row["category"]: row for row in category_status_rows(_config_feature_columns(current_cfg))
            }
            current_count = int(grouped.get(category, {}).get("count", 0))
            if current_count <= target_count or accepted_step_count >= int(args.max_steps):
                break

            removal_order = ordered_category_removals(
                feature_columns=_config_feature_columns(current_cfg),
                importance_map=current_eval["importance_map"],
                category=category,
            )
            removal_order = removal_order[: max(int(args.candidate_limit), 1)]
            print(
                f"[INFO] 类别 {category} 当前 {current_count} 个，目标 {target_count} 个，"
                f"尝试顺序: {', '.join(removal_order)}"
            )

            candidate_rows: list[dict] = []
            candidate_payloads: dict[str, tuple[dict, dict]] = {}
            for dropped_feature in removal_order:
                candidate_cfg = build_candidate_cfg(
                    current_cfg=current_cfg,
                    search_name=args.search_name,
                    step=accepted_step_count + 1,
                    category=category,
                    dropped_feature=dropped_feature,
                )
                _write_yaml(
                    candidate_cfg,
                    _candidate_config_path(args.search_name, candidate_cfg["name"]),
                )
                started_at = time.time()
                print(f"[RUN] step={accepted_step_count + 1} category={category} drop={dropped_feature}")
                eval_payload = evaluate_config(candidate_cfg, engine=args.engine)
                elapsed = time.time() - started_at
                row = eval_to_row(
                    cfg=candidate_cfg,
                    eval_payload=eval_payload,
                    category=category,
                    dropped_feature=dropped_feature,
                    step=accepted_step_count + 1,
                    accepted=False,
                    accepted_name="",
                )
                row["elapsed_sec"] = round(elapsed, 2)
                row["valid_guardrail"] = is_valid_candidate(
                    eval_payload,
                    min_annual=args.min_annual,
                    max_drawdown=args.max_drawdown,
                )
                candidate_rows.append(row)
                candidate_payloads[candidate_cfg["name"]] = (candidate_cfg, eval_payload)
                rows.append(row)

            valid_rows = [row for row in candidate_rows if row["valid_guardrail"]]
            if not valid_rows:
                print(f"[STOP] 类别 {category} 再删会突破约束，停止该类别。")
                break

            best_row = sorted(valid_rows, key=candidate_sort_key, reverse=True)[0]
            best_cfg, best_eval = candidate_payloads[best_row["config_name"]]
            best_row["accepted"] = True
            best_row["accepted_name"] = best_row["config_name"]
            accepted_steps.append(best_row)
            current_cfg = best_cfg
            current_eval = best_eval
            accepted_step_count += 1
            print(
                "[OK] 接受候选 "
                f"{best_row['config_name']} "
                f"annual={best_row['annual_return']:.2%} "
                f"max_dd={best_row['max_drawdown']:.2%} "
                f"feature_count={best_row['feature_count']}"
            )

    summary_df = pd.DataFrame(rows)
    summary_path = search_root / "summary.csv"
    summary_df.to_csv(summary_path, index=False)

    final_cfg_path = search_root / "final_config.yaml"
    _write_yaml(current_cfg, final_cfg_path)
    final_status = category_status_rows(_config_feature_columns(current_cfg))
    final_payload = {
        "base_config": str(Path(args.config).resolve()),
        "search_name": args.search_name,
        "engine": args.engine,
        "min_annual": args.min_annual,
        "max_drawdown": args.max_drawdown,
        "accepted_steps": accepted_steps,
        "final_config_name": current_cfg["name"],
        "final_config_path": str(final_cfg_path),
        "final_feature_columns": _config_feature_columns(current_cfg),
        "final_category_status": final_status,
        "final_backtest_summary": current_eval["backtest_summary"],
        "summary_csv": str(summary_path),
    }
    save_json(final_payload, search_root / "final_summary.json")

    print(f"[OK] 搜索完成: {summary_path}")
    print(
        "[INFO] 最终结果 "
        f"annual={current_eval['backtest_summary']['annual_return']:.2%} "
        f"max_dd={current_eval['backtest_summary']['max_drawdown']:.2%} "
        f"features={len(_config_feature_columns(current_cfg))}"
    )


if __name__ == "__main__":
    main()
