#!/usr/bin/env python3
"""Scan results CSV files, compute performance metrics, and cross-check WF status."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results"
ANALYSIS_ROOT = RESULTS_ROOT / "analysis"
OOS_START = pd.Timestamp("2024-01-01")
ANNUAL_RISK_FREE = 0.03
DAILY_RISK_FREE = ANNUAL_RISK_FREE / 252.0

BACKTEST_RE = re.compile(
    r"^backtest_(?P<name>.+?)_(?P<scope>historical_csi300|all_market)_(?P<stamp>\d{8}_\d{6})$"
)
WF_SUFFIX_RE = re.compile(r"__wf_(?:roll\d+|expanding)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default=str(RESULTS_ROOT))
    parser.add_argument("--min-cagr", type=float, default=0.20)
    parser.add_argument("--max-dd", type=float, default=0.12)
    parser.add_argument("--oos-start", default="2024-01-01")
    return parser.parse_args()


def safe_relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def infer_strategy_name(path: Path, return_col: str) -> str:
    stem = path.stem
    match = BACKTEST_RE.match(stem)
    if match:
        return match.group("name")
    if path.name == "overlay_results.csv":
        return path.parent.name
    if return_col.startswith("return__"):
        return return_col[len("return__") :].replace("__", "/")
    if path.name == "selections.csv":
        return path.parent.name
    return stem


def infer_source_group(path: Path, results_root: Path) -> str:
    rel = path.relative_to(results_root)
    parts = rel.parts
    if len(parts) == 1:
        return "top_level_results"
    if path.name == "overlay_results.csv" and len(parts) >= 3:
        return "/".join(parts[:-2])
    if path.name == "selections.csv" and len(parts) >= 3:
        return "/".join(parts[:-2])
    return "/".join(parts[:-1])


def normalize_base_name(strategy_name: str) -> str:
    return WF_SUFFIX_RE.sub("", strategy_name)


def is_walk_forward_name(strategy_name: str, path: Path) -> bool:
    lowered = f"{strategy_name} {path.as_posix()}".lower()
    return "__wf_" in lowered or "walkforward" in lowered or "walk_forward" in lowered


def detect_return_columns(header: Iterable[str]) -> list[str]:
    columns = list(header)
    if "date" not in columns:
        return []
    picked: list[str] = []
    if "overlay_return" in columns:
        picked.append("overlay_return")
    elif "return" in columns:
        picked.append("return")
    picked.extend(col for col in columns if col.startswith("return__"))
    return picked


def read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        return next(reader)


def compute_metrics(frame: pd.DataFrame, return_col: str) -> dict[str, float | str | int]:
    local = frame[["date", return_col]].copy()
    local["date"] = pd.to_datetime(local["date"])
    local[return_col] = pd.to_numeric(local[return_col], errors="coerce")
    local = local.dropna().sort_values("date")
    if len(local) < 2:
        raise ValueError("not enough rows")

    rets = local[return_col].astype(float)
    nav = (1.0 + rets).cumprod()
    rolling_max = nav.cummax()
    drawdown = nav / rolling_max - 1.0
    start_date = pd.Timestamp(local["date"].iloc[0])
    end_date = pd.Timestamp(local["date"].iloc[-1])
    days = max(int((end_date - start_date).days), 1)
    terminal_value = float(nav.iloc[-1])

    cagr = terminal_value ** (365.0 / float(days)) - 1.0 if terminal_value > 0.0 else -1.0
    mean_excess = float((rets - DAILY_RISK_FREE).mean())
    daily_std = float(rets.std(ddof=0))
    sharpe = mean_excess / daily_std * math.sqrt(252.0) if daily_std > 0.0 else 0.0
    max_dd = float(drawdown.min())
    calmar = cagr / abs(max_dd) if max_dd < 0.0 else float("inf")
    total_return = terminal_value - 1.0

    return {
        "rows": int(len(local)),
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "days": days,
        "total_return": total_return,
        "cagr": cagr,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "calmar": calmar,
    }


def maybe_compute_window(frame: pd.DataFrame, return_col: str, start_date: pd.Timestamp) -> dict[str, float | str | int]:
    sliced = frame.loc[pd.to_datetime(frame["date"]) >= start_date].copy()
    if len(sliced) < 2:
        return {
            "rows": 0,
            "start_date": "",
            "end_date": "",
            "days": 0,
            "total_return": np.nan,
            "cagr": np.nan,
            "max_dd": np.nan,
            "sharpe": np.nan,
            "calmar": np.nan,
        }
    return compute_metrics(sliced, return_col)


def load_summary_oos_map(results_root: Path) -> dict[tuple[str, str], dict[str, float]]:
    summary_map: dict[tuple[str, str], dict[str, float]] = {}
    for path in results_root.rglob("*.csv"):
        try:
            header = read_header(path)
        except Exception:
            continue
        if not {"name", "source", "oos_ann", "oos_dd", "oos_sh"} <= set(header):
            continue
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        for _, row in frame.iterrows():
            name = str(row.get("name", "")).strip()
            source = str(row.get("source", "")).strip()
            if not name:
                continue
            summary_map[(name, source)] = {
                "summary_full_cagr": float(row.get("full_ann", np.nan)),
                "summary_full_dd": float(row.get("full_dd", np.nan)),
                "summary_full_sharpe": float(row.get("full_sh", np.nan)),
                "summary_oos_cagr": float(row.get("oos_ann", np.nan)),
                "summary_oos_dd": float(row.get("oos_dd", np.nan)),
                "summary_oos_sharpe": float(row.get("oos_sh", np.nan)),
                "summary_hit_target": str(row.get("hit_target", row.get("meets_oos", ""))),
                "summary_source_file": safe_relative(path, results_root),
            }
    return summary_map


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root).resolve()
    analysis_root = ANALYSIS_ROOT
    analysis_root.mkdir(parents=True, exist_ok=True)
    oos_start = pd.Timestamp(args.oos_start)

    summary_oos_map = load_summary_oos_map(results_root)

    records: list[dict[str, object]] = []
    total_csv = 0
    return_series = 0

    for path in sorted(results_root.rglob("*.csv")):
        total_csv += 1
        try:
            header = read_header(path)
        except Exception as exc:
            records.append(
                {
                    "path": safe_relative(path, results_root),
                    "status": "header_error",
                    "error": str(exc),
                }
            )
            continue

        return_cols = detect_return_columns(header)
        if not return_cols:
            continue

        usecols = ["date", *return_cols]
        try:
            frame = pd.read_csv(path, usecols=usecols)
        except Exception as exc:
            for return_col in return_cols:
                return_series += 1
                records.append(
                    {
                        "path": safe_relative(path, results_root),
                        "return_col": return_col,
                        "status": "read_error",
                        "error": str(exc),
                    }
                )
            continue

        for return_col in return_cols:
            return_series += 1
            strategy_name = infer_strategy_name(path, return_col)
            source_group = infer_source_group(path, results_root)
            summary_key = (strategy_name, source_group.rsplit("/", 1)[-1])
            try:
                full = compute_metrics(frame, return_col)
                oos = maybe_compute_window(frame, return_col, oos_start)
            except Exception as exc:
                records.append(
                    {
                        "path": safe_relative(path, results_root),
                        "return_col": return_col,
                        "strategy_name": strategy_name,
                        "source_group": source_group,
                        "status": "metric_error",
                        "error": str(exc),
                    }
                )
                continue

            record: dict[str, object] = {
                "path": safe_relative(path, results_root),
                "return_col": return_col,
                "status": "ok",
                "series_kind": "overlay" if return_col == "overlay_return" else ("component" if return_col.startswith("return__") else "backtest"),
                "strategy_name": strategy_name,
                "base_strategy_name": normalize_base_name(strategy_name),
                "source_group": source_group,
                "is_wf": is_walk_forward_name(strategy_name, path),
                "mtime": path.stat().st_mtime,
                "file_size": path.stat().st_size,
                **{f"full_{k}": v for k, v in full.items()},
                **{f"oos_{k}": v for k, v in oos.items()},
            }
            if summary_key in summary_oos_map:
                record.update(summary_oos_map[summary_key])
            records.append(record)

    metrics = pd.DataFrame(records)
    metrics_path = analysis_root / "all_return_series_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    usable = metrics.loc[metrics["status"] == "ok"].copy()
    usable["full_abs_dd"] = usable["full_max_dd"].abs()
    usable["oos_abs_dd"] = usable["oos_max_dd"].abs()

    series_priority = {"overlay": 0, "backtest": 1, "component": 2}
    usable["series_priority"] = usable["series_kind"].map(series_priority).fillna(9)
    usable["top_level_penalty"] = np.where(usable["source_group"] == "top_level_results", 1, 0)

    canonical_series = (
        usable.sort_values(
            ["strategy_name", "source_group", "series_priority", "full_days", "mtime"],
            ascending=[True, True, True, False, False],
        )
        .drop_duplicates(subset=["strategy_name", "source_group"], keep="first")
    )

    canonical_by_name = (
        canonical_series.sort_values(
            ["strategy_name", "series_priority", "top_level_penalty", "full_days", "mtime"],
            ascending=[True, True, True, False, False],
        )
        .drop_duplicates(subset=["strategy_name"], keep="first")
    )

    wf_variants = (
        canonical_by_name.loc[canonical_by_name["is_wf"]]
        .sort_values(["base_strategy_name", "full_cagr", "mtime"], ascending=[True, False, False])
        .groupby("base_strategy_name")
        .agg(
            wf_match_count=("strategy_name", "size"),
            wf_variants=("strategy_name", lambda s: " | ".join(dict.fromkeys(str(x) for x in s))),
            wf_best_path=("path", "first"),
            wf_best_cagr=("full_cagr", "first"),
            wf_best_dd=("full_max_dd", "first"),
            wf_best_sharpe=("full_sharpe", "first"),
            wf_best_oos_cagr=("oos_cagr", "first"),
            wf_best_oos_dd=("oos_max_dd", "first"),
        )
        .reset_index()
    )

    strategy_view = canonical_series.merge(
        wf_variants,
        how="left",
        left_on="base_strategy_name",
        right_on="base_strategy_name",
    )
    strategy_view["wf_status"] = np.where(
        strategy_view["is_wf"],
        "self_wf",
        np.where(strategy_view["wf_match_count"].fillna(0) > 0, "has_matching_wf", "no_matching_wf"),
    )
    strategy_view["meets_full_target"] = (
        (strategy_view["full_cagr"] > float(args.min_cagr))
        & (strategy_view["full_abs_dd"] <= float(args.max_dd))
    )
    strategy_view["meets_oos_target"] = (
        strategy_view["oos_cagr"].notna()
        & (strategy_view["oos_cagr"] > float(args.min_cagr))
        & (strategy_view["oos_abs_dd"] <= float(args.max_dd))
    )

    qualifying = strategy_view.loc[strategy_view["meets_full_target"]].copy()
    qualifying = qualifying.sort_values(
        ["is_wf", "full_cagr", "full_sharpe"],
        ascending=[False, False, False],
    )
    qualifying_path = analysis_root / "qualifying_strategies.csv"
    qualifying.to_csv(qualifying_path, index=False)

    wf_qualifying = qualifying.loc[qualifying["is_wf"]].copy()
    wf_qualifying_path = analysis_root / "qualifying_walk_forward_strategies.csv"
    wf_qualifying.to_csv(wf_qualifying_path, index=False)

    need_wf = qualifying.loc[~qualifying["is_wf"] & (qualifying["wf_status"] == "no_matching_wf")].copy()
    need_wf = need_wf.sort_values(["full_cagr", "full_sharpe"], ascending=[False, False])
    need_wf_path = analysis_root / "qualifying_strategies_missing_wf.csv"
    need_wf.to_csv(need_wf_path, index=False)

    summary = {
        "results_root": str(results_root),
        "total_csv_files": int(total_csv),
        "return_series_count": int(return_series),
        "usable_series_count": int(len(usable)),
        "qualifying_strategy_count": int(len(qualifying)),
        "qualifying_wf_strategy_count": int(len(wf_qualifying)),
        "qualifying_missing_wf_count": int(len(need_wf)),
        "metrics_csv": str(metrics_path),
        "qualifying_csv": str(qualifying_path),
        "qualifying_wf_csv": str(wf_qualifying_path),
        "missing_wf_csv": str(need_wf_path),
    }
    pd.Series(summary).to_json(analysis_root / "analysis_summary.json", indent=2, force_ascii=False)

    print("[INFO] total_csv_files=", total_csv)
    print("[INFO] return_series_count=", return_series)
    print("[INFO] usable_series_count=", len(usable))
    print("[INFO] qualifying_strategy_count=", len(qualifying))
    print("[INFO] qualifying_wf_strategy_count=", len(wf_qualifying))
    print("[INFO] qualifying_missing_wf_count=", len(need_wf))
    print(f"[INFO] metrics_csv={metrics_path}")
    print(f"[INFO] qualifying_csv={qualifying_path}")
    print(f"[INFO] qualifying_wf_csv={wf_qualifying_path}")
    print(f"[INFO] missing_wf_csv={need_wf_path}")


if __name__ == "__main__":
    main()
