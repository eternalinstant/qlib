#!/usr/bin/env python3
"""Ablation for combo factors in push25 cq10 strategy."""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path
from typing import List

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


BASE_CONFIG = PROJECT_ROOT / "config" / "models" / "push25_cq10_k8d2_very_tight.yaml"
RUN_ROOT = PROJECT_ROOT / "results" / "model_signals" / "ablation_runs" / "push25_combo_factors_v1"
OOS_START = pd.Timestamp("2024-01-01")


def evaluate_oos(overlay_csv: Path) -> dict:
    frame = pd.read_csv(overlay_csv, parse_dates=["date"]).sort_values("date")
    frame = frame[frame["date"] >= OOS_START]
    if frame.empty:
        return {"oos_ann": 0.0, "oos_dd": 0.0, "oos_sharpe": 0.0}
    returns = frame["overlay_return"].astype(float)
    total = float((1.0 + returns).prod() - 1.0)
    days = max((frame["date"].iloc[-1] - frame["date"].iloc[0]).days, 1)
    ann = (1.0 + total) ** (365.0 / days) - 1.0 if (1.0 + total) > 0 else -1.0
    dd = float(frame["drawdown"].min())
    std = float(returns.std(ddof=0))
    sharpe = float(returns.mean() / std * math.sqrt(252.0)) if std > 0 else 0.0
    return {"oos_ann": ann, "oos_dd": dd, "oos_sharpe": sharpe}


def main() -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    base_cfg = load_predictive_config(BASE_CONFIG)
    parquet_cols = list(base_cfg["data"]["parquet_feature_columns"])
    alpha_cols = list(base_cfg["data"]["alpha158_feature_columns"])

    combos = [
        "rank_value_profit_core",
        "rank_balance_core",
        "qvf_core_interaction",
    ]
    variants: List[tuple[str, List[str]]] = [
        ("baseline", []),
        ("drop_rank_value_profit_core", ["rank_value_profit_core"]),
        ("drop_rank_balance_core", ["rank_balance_core"]),
        ("drop_qvf_core_interaction", ["qvf_core_interaction"]),
        ("drop_all_combo_factors", combos),
    ]

    rows = []
    for tag, drops in variants:
        cfg = copy.deepcopy(base_cfg)
        cfg["name"] = f"push25_cq10_{tag}"
        cfg["output"]["root"] = str((RUN_ROOT / cfg["name"]).resolve())
        cfg["data"]["parquet_feature_columns"] = [c for c in parquet_cols if c not in set(drops)]
        cfg["data"]["alpha158_feature_columns"] = alpha_cols

        print(
            f"[RUN] {cfg['name']} "
            f"parquet={len(cfg['data']['parquet_feature_columns'])} "
            f"alpha={len(alpha_cols)} "
            f"drops={drops}"
        )
        train_summary = train_from_config(cfg)
        score_summary = score_from_config(cfg)
        _, backtest_summary = backtest_from_config(cfg, engine="qlib")
        oos = evaluate_oos(Path(backtest_summary["results_file"]))

        rows.append(
            {
                "variant": tag,
                "drops": ",".join(drops),
                "parquet_features": len(cfg["data"]["parquet_feature_columns"]),
                "alpha_features": len(alpha_cols),
                "selection_dates": int(score_summary.get("selection_dates", 0)),
                "score_rows": int(score_summary.get("score_rows", 0)),
                "valid_rank_ic": float(train_summary.get("metrics", {}).get("valid_mean_rank_ic", 0.0)),
                "full_ann": float(backtest_summary["annual_return"]),
                "full_dd": float(backtest_summary["max_drawdown"]),
                "full_sharpe": float(backtest_summary["sharpe_ratio"]),
                "oos_ann": float(oos["oos_ann"]),
                "oos_dd": float(oos["oos_dd"]),
                "oos_sharpe": float(oos["oos_sharpe"]),
                "output_root": cfg["output"]["root"],
            }
        )

    summary = pd.DataFrame(rows).sort_values("full_ann", ascending=False)
    summary_csv = RUN_ROOT / "summary.csv"
    summary.to_csv(summary_csv, index=False)
    save_json({"summary_csv": str(summary_csv), "rows": rows}, RUN_ROOT / "summary.json")

    print(f"[OK] summary: {summary_csv}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

