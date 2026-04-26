#!/usr/bin/env python3
"""在固定模型分数上搜索更低换手的选股参数。"""

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
    save_selection_frame,
)
from modules.modeling.predictive_signal import materialize_selections_from_scores as materialize_selections  # noqa: E402


TURNOVER_PRESET_V1 = [
    {
        "name": "turn_buf2_ch2",
        "selection_overrides": {"buffer": 2, "churn_limit": 2},
    },
    {
        "name": "turn_buf3_ch2",
        "selection_overrides": {"buffer": 3, "churn_limit": 2},
    },
    {
        "name": "turn_buf4_ch2",
        "selection_overrides": {"buffer": 4, "churn_limit": 2},
    },
    {
        "name": "turn_buf4_ch1",
        "selection_overrides": {"buffer": 4, "churn_limit": 1},
    },
    {
        "name": "turn_buf6_ch2",
        "selection_overrides": {"buffer": 6, "churn_limit": 2},
    },
    {
        "name": "turn_gate_s3_e6_x10_p2_h10",
        "selection_overrides": {
            "score_smoothing_days": 3,
            "entry_rank": 6,
            "exit_rank": 10,
            "entry_persist_days": 2,
            "exit_persist_days": 2,
            "min_hold_days": 10,
        },
    },
    {
        "name": "turn_gate_s3_e6_x12_p2_h10",
        "selection_overrides": {
            "score_smoothing_days": 3,
            "entry_rank": 6,
            "exit_rank": 12,
            "entry_persist_days": 2,
            "exit_persist_days": 2,
            "min_hold_days": 10,
        },
    },
    {
        "name": "turn_gate_s5_e6_x10_p2_h10",
        "selection_overrides": {
            "score_smoothing_days": 5,
            "entry_rank": 6,
            "exit_rank": 10,
            "entry_persist_days": 2,
            "exit_persist_days": 2,
            "min_hold_days": 10,
        },
    },
    {
        "name": "turn_gate_s3_e6_x10_p2_h20",
        "selection_overrides": {
            "score_smoothing_days": 3,
            "entry_rank": 6,
            "exit_rank": 10,
            "entry_persist_days": 2,
            "exit_persist_days": 2,
            "min_hold_days": 20,
        },
    },
    {
        "name": "turn_gate_s3_e6_x10_p3_h10",
        "selection_overrides": {
            "score_smoothing_days": 3,
            "entry_rank": 6,
            "exit_rank": 10,
            "entry_persist_days": 3,
            "exit_persist_days": 2,
            "min_hold_days": 10,
        },
    },
]


PRESETS = {
    "turnover_v1": TURNOVER_PRESET_V1,
}

TURNOVER_PRESET_V2 = [
    {
        "name": "turn_buf1",
        "selection_overrides": {"buffer": 1},
    },
    {
        "name": "turn_buf1_ch4",
        "selection_overrides": {"buffer": 1, "churn_limit": 4},
    },
    {
        "name": "turn_buf1_ch3",
        "selection_overrides": {"buffer": 1, "churn_limit": 3},
    },
    {
        "name": "turn_buf2_ch4",
        "selection_overrides": {"buffer": 2, "churn_limit": 4},
    },
    {
        "name": "turn_buf2_ch3",
        "selection_overrides": {"buffer": 2, "churn_limit": 3},
    },
    {
        "name": "turn_gate_s2_h5_x8",
        "selection_overrides": {
            "score_smoothing_days": 2,
            "entry_rank": 6,
            "exit_rank": 8,
            "entry_persist_days": 1,
            "exit_persist_days": 1,
            "min_hold_days": 5,
        },
    },
    {
        "name": "turn_gate_s2_h10_x8",
        "selection_overrides": {
            "score_smoothing_days": 2,
            "entry_rank": 6,
            "exit_rank": 8,
            "entry_persist_days": 1,
            "exit_persist_days": 1,
            "min_hold_days": 10,
        },
    },
    {
        "name": "turn_gate_s2_h5_x9",
        "selection_overrides": {
            "score_smoothing_days": 2,
            "entry_rank": 6,
            "exit_rank": 9,
            "entry_persist_days": 1,
            "exit_persist_days": 1,
            "min_hold_days": 5,
        },
    },
]

PRESETS = {
    "turnover_v1": TURNOVER_PRESET_V1,
    "turnover_v2": TURNOVER_PRESET_V2,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-config",
        default="config/models/qvf_core_plus_fixed80_k6_overlay_fullspan_compressed.yaml",
    )
    parser.add_argument(
        "--scores-path",
        default="",
        help="已有 scores.parquet 路径；为空时默认取 base-config 对应产物。",
    )
    parser.add_argument("--search-name", default="turnover_reduction_v1")
    parser.add_argument("--preset", default="turnover_v1", choices=sorted(PRESETS))
    return parser.parse_args()


def load_scores(path: Path) -> pd.Series:
    df = pd.read_parquet(path)
    if df.empty:
        return pd.Series(dtype=float)
    work = df.copy()
    work["datetime"] = pd.to_datetime(work["datetime"])
    return work.set_index(["datetime", "instrument"])["score"].sort_index()


def build_candidate_config(base_cfg: dict, search_root: Path, candidate: dict) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["name"] = f"{base_cfg['name']}__{candidate['name']}"
    cfg["selection"].update(copy.deepcopy(candidate.get("selection_overrides", {})))
    if candidate.get("overlay_overrides"):
        cfg.setdefault("overlay", {})
        cfg["overlay"].update(copy.deepcopy(candidate["overlay_overrides"]))
    cfg["output"]["root"] = str((search_root / candidate["name"]).resolve())
    return cfg


def summarize_selection(selection_df: pd.DataFrame, topk: int) -> dict:
    if selection_df.empty:
        return {
            "selection_dates": 0,
            "avg_buy_count": 0.0,
            "avg_sell_count": 0.0,
            "avg_keep_count": 0.0,
            "full_replace_count": 0,
            "no_change_count": 0,
            "avg_one_way_turnover_ratio": 0.0,
        }

    work = selection_df.copy()
    work["date"] = pd.to_datetime(work["date"])
    rows = []
    prev_symbols: set[str] = set()
    for dt, grp in work.groupby("date", sort=True):
        symbols = set(grp["symbol"].tolist())
        buy_count = len(symbols - prev_symbols)
        sell_count = len(prev_symbols - symbols)
        keep_count = len(symbols & prev_symbols)
        rows.append(
            {
                "date": dt,
                "buy_count": buy_count,
                "sell_count": sell_count,
                "keep_count": keep_count,
            }
        )
        prev_symbols = symbols

    changes = pd.DataFrame(rows)
    avg_buy = float(changes["buy_count"].mean())
    avg_sell = float(changes["sell_count"].mean())
    return {
        "selection_dates": int(len(changes)),
        "avg_buy_count": avg_buy,
        "avg_sell_count": avg_sell,
        "avg_keep_count": float(changes["keep_count"].mean()),
        "full_replace_count": int(((changes["buy_count"] >= topk) & (changes["sell_count"] >= topk)).sum()),
        "no_change_count": int(((changes["buy_count"] + changes["sell_count"]) == 0).sum()),
        "avg_one_way_turnover_ratio": avg_buy / topk if topk > 0 else 0.0,
    }


def summarize_base_backtest(base_results_file: str) -> dict:
    path = Path(base_results_file)
    if not path.exists():
        return {
            "trade_days_with_orders": 0,
            "total_buy_orders": 0,
            "total_sell_orders": 0,
            "total_fee_amount": 0.0,
        }
    df = pd.read_csv(path)
    order_days = (df["buy_count"] + df["sell_count"]) > 0
    return {
        "trade_days_with_orders": int(order_days.sum()),
        "total_buy_orders": int(df["buy_count"].sum()),
        "total_sell_orders": int(df["sell_count"].sum()),
        "total_fee_amount": float(df["fee_amount"].sum()),
    }


def candidate_summary(candidate: dict, cfg: dict, result, backtest_summary: dict, selection_df: pd.DataFrame, elapsed: float) -> dict:
    selection_metrics = summarize_selection(selection_df, int(cfg["selection"]["topk"]))
    base_results_file = result.metadata.get("base_results_file") or result.metadata.get("results_file")
    backtest_metrics = summarize_base_backtest(base_results_file) if base_results_file else {}
    return {
        "name": candidate["name"],
        "selection_overrides": candidate.get("selection_overrides", {}),
        "annual_return": backtest_summary.get("annual_return", 0.0),
        "max_drawdown": backtest_summary.get("max_drawdown", 0.0),
        "sharpe_ratio": backtest_summary.get("sharpe_ratio", 0.0),
        "selection_dates": selection_metrics["selection_dates"],
        "avg_buy_count": selection_metrics["avg_buy_count"],
        "avg_sell_count": selection_metrics["avg_sell_count"],
        "avg_keep_count": selection_metrics["avg_keep_count"],
        "avg_one_way_turnover_ratio": selection_metrics["avg_one_way_turnover_ratio"],
        "full_replace_count": selection_metrics["full_replace_count"],
        "no_change_count": selection_metrics["no_change_count"],
        "trade_days_with_orders": backtest_metrics.get("trade_days_with_orders", 0),
        "total_buy_orders": backtest_metrics.get("total_buy_orders", 0),
        "total_sell_orders": backtest_metrics.get("total_sell_orders", 0),
        "total_fee_amount": backtest_metrics.get("total_fee_amount", 0.0),
        "fee_ratio_to_initial": float(result.metadata.get("fee_ratio_to_initial", 0.0) or 0.0),
        "results_file": backtest_summary.get("results_file"),
        "base_results_file": base_results_file,
        "elapsed_seconds": elapsed,
    }


def main():
    args = parse_args()
    base_cfg = load_predictive_config(args.base_config)
    scores_file = Path(args.scores_path).expanduser()
    if not args.scores_path:
        scores_file = Path(base_cfg["output"]["root"]) / "scores.parquet"
    scores = load_scores(scores_file)
    if scores.empty:
        raise ValueError(f"空 scores 文件: {scores_file}")

    search_root = (
        PROJECT_ROOT / "results" / "model_signals" / "selection_search_runs" / args.search_name
    ).resolve()
    search_root.mkdir(parents=True, exist_ok=True)

    rebalance_dates = pd.DatetimeIndex(sorted(scores.index.get_level_values("datetime").unique()))
    candidates = PRESETS[args.preset]

    rows = []
    for idx, candidate in enumerate(candidates, start=1):
        print(f"[{idx}/{len(candidates)}] {candidate['name']}")
        cfg = build_candidate_config(base_cfg, search_root, candidate)
        selection_df = materialize_selections(scores, rebalance_dates, cfg["selection"])
        save_selection_frame(selection_df, Path(cfg["output"]["root"]) / "selections.csv")
        start = time.perf_counter()
        result, backtest_summary = backtest_from_config(cfg)
        elapsed = time.perf_counter() - start
        row = candidate_summary(candidate, cfg, result, backtest_summary, selection_df, elapsed)
        rows.append(row)
        print(
            "  "
            f"annual={row['annual_return']:.2%} "
            f"dd={row['max_drawdown']:.2%} "
            f"avg_buy={row['avg_buy_count']:.2f} "
            f"fee_ratio={row['fee_ratio_to_initial']:.2%} "
            f"elapsed={row['elapsed_seconds']:.1f}s"
        )

    result_df = pd.DataFrame(rows).sort_values(
        ["avg_buy_count", "annual_return", "max_drawdown"],
        ascending=[True, False, False],
    )
    csv_path = search_root / "summary.csv"
    result_df.to_csv(csv_path, index=False)
    save_json(
        {
            "search_name": args.search_name,
            "preset": args.preset,
            "base_config": str(Path(args.base_config).resolve()),
            "scores_path": str(scores_file.resolve()),
            "summary_csv": str(csv_path),
            "candidates": rows,
        },
        search_root / "summary.json",
    )
    print(f"[OK] 汇总已保存: {csv_path}")


if __name__ == "__main__":
    main()
