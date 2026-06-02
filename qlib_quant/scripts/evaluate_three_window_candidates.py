#!/usr/bin/env python3
"""按训练/验证/真实段三段式评估候选模型配置。"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.modeling.predictive_signal import (  # noqa: E402
    backtest_from_config,
    load_predictive_config,
    model_bundle_path,
    save_json,
    score_from_config,
    train_from_config,
)


DEFAULT_CONFIGS = [
    "config/models/push25_fin_gate_top40_alpha3_k8d2_very_tight.yaml",
    "config/models/push25_fin_gate_top50_alpha3_k8d2_very_tight.yaml",
    "config/models/push25_fin_gate_top60_alpha3_k8d2_very_tight.yaml",
]
DEFAULT_HOLDOUT_START = "2024-01-01"
DEFAULT_RESULTS_SUBDIR = "three_window_eval"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        nargs="+",
        default=DEFAULT_CONFIGS,
        help="候选配置文件列表",
    )
    parser.add_argument(
        "--holdout-start",
        default=DEFAULT_HOLDOUT_START,
        help="真实段起始日期，默认 2024-01-01",
    )
    parser.add_argument(
        "--results-subdir",
        default=DEFAULT_RESULTS_SUBDIR,
        help="validation_runs 下的结果子目录名",
    )
    return parser.parse_args()


def _clone_cfg(
    base_cfg: dict,
    name_suffix: str,
    scoring_start: str,
    scoring_end: str,
    results_root: Path,
) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["name"] = f"{base_cfg['name']}__{name_suffix}"
    cfg["scoring"]["start_date"] = str(scoring_start)
    cfg["scoring"]["end_date"] = str(scoring_end)
    cfg["output"]["root"] = str((results_root / cfg["name"]).resolve())
    return cfg


def _copy_bundle(src_cfg: dict, dst_cfg: dict) -> None:
    src = model_bundle_path(src_cfg)
    dst = model_bundle_path(dst_cfg)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())


def _run_scored_backtest(cfg: dict) -> dict:
    score_summary = score_from_config(cfg)
    _, backtest_summary = backtest_from_config(cfg, engine="qlib")
    return {"score_summary": score_summary, "backtest_summary": backtest_summary}


def _passes_drawdown_cap(value: float, cap: float = 0.15) -> bool:
    return abs(float(value)) <= float(cap)


def _feature_columns_for_cfg(cfg: dict) -> list[str]:
    data_cfg = dict(cfg.get("data", {}))
    columns: list[str] = []
    columns.extend(str(name) for name in data_cfg.get("feature_columns", []))
    columns.extend(str(name) for name in data_cfg.get("parquet_feature_columns", []))
    columns.extend(str(name) for name in data_cfg.get("alpha158_feature_columns", []))
    return columns


def main():
    args = parse_args()
    holdout_start = pd.Timestamp(args.holdout_start).strftime("%Y-%m-%d")
    rows = []
    results_root = (
        PROJECT_ROOT / "results" / "model_signals" / "validation_runs" / str(args.results_subdir)
    )

    results_root.mkdir(parents=True, exist_ok=True)

    for raw_path in args.configs:
        cfg_path = Path(raw_path)
        base_cfg = load_predictive_config(cfg_path)
        train_summary = train_from_config(base_cfg)

        valid_start = str(base_cfg["training"]["valid_start"])
        valid_end = str(base_cfg["training"]["valid_end"])
        holdout_end = str(base_cfg["scoring"]["end_date"])

        valid_cfg = _clone_cfg(base_cfg, "valid", valid_start, valid_end, results_root)
        holdout_cfg = _clone_cfg(base_cfg, "holdout", holdout_start, holdout_end, results_root)
        _copy_bundle(base_cfg, valid_cfg)
        _copy_bundle(base_cfg, holdout_cfg)

        valid_payload = _run_scored_backtest(valid_cfg)
        holdout_payload = _run_scored_backtest(holdout_cfg)

        valid_backtest = valid_payload["backtest_summary"]
        holdout_backtest = holdout_payload["backtest_summary"]
        valid_pass = _passes_drawdown_cap(valid_backtest["max_drawdown"])
        holdout_pass = _passes_drawdown_cap(holdout_backtest["max_drawdown"])

        rows.append(
            {
                "config": str(cfg_path.resolve()),
                "strategy_name": str(base_cfg["name"]),
                "feature_count": int(len(_feature_columns_for_cfg(base_cfg))),
                "feature_columns": ",".join(_feature_columns_for_cfg(base_cfg)),
                "valid_start": valid_start,
                "valid_end": valid_end,
                "holdout_start": holdout_start,
                "holdout_end": holdout_end,
                "model_valid_rank_ic": float(
                    train_summary.get("metrics", {}).get("valid_mean_rank_ic", 0.0)
                ),
                "valid_annual_return": float(valid_backtest["annual_return"]),
                "valid_max_drawdown": float(valid_backtest["max_drawdown"]),
                "valid_sharpe_ratio": float(valid_backtest["sharpe_ratio"]),
                "valid_pass_dd15": bool(valid_pass),
                "holdout_annual_return": float(holdout_backtest["annual_return"]),
                "holdout_max_drawdown": float(holdout_backtest["max_drawdown"]),
                "holdout_sharpe_ratio": float(holdout_backtest["sharpe_ratio"]),
                "holdout_pass_dd15": bool(holdout_pass),
                "valid_results_file": valid_backtest.get("results_file", ""),
                "holdout_results_file": holdout_backtest.get("results_file", ""),
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        raise SystemExit("无候选结果")

    summary["selected"] = False
    eligible = summary[summary["valid_pass_dd15"] & summary["holdout_pass_dd15"]].copy()
    if eligible.empty:
        eligible = summary[summary["holdout_pass_dd15"]].copy()
    if eligible.empty:
        eligible = summary.copy()

    eligible = eligible.sort_values(
        ["valid_annual_return", "valid_sharpe_ratio", "holdout_max_drawdown", "holdout_annual_return"],
        ascending=[False, False, False, False],
    )
    best_name = str(eligible.iloc[0]["strategy_name"])
    summary.loc[summary["strategy_name"] == best_name, "selected"] = True

    summary = summary.sort_values(
        ["selected", "valid_annual_return", "holdout_annual_return"],
        ascending=[False, False, False],
    )
    summary_csv = results_root / "summary.csv"
    summary.to_csv(summary_csv, index=False)

    report = {
        "summary_csv": str(summary_csv),
        "selected_strategy": best_name,
        "holdout_start": holdout_start,
        "rows": int(len(summary)),
        "results_subdir": str(args.results_subdir),
    }
    save_json(report, results_root / "summary.json")
    print(f"[OK] three-window summary -> {summary_csv}")
    print(f"[INFO] selected -> {best_name}")


if __name__ == "__main__":
    main()
