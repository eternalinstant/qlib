#!/usr/bin/env python3
"""统一生成 push25 候选的正式对照与压力测试结果。"""

from __future__ import annotations

import copy
import math
import shutil
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.modeling.predictive_signal import (
    backtest_from_config,
    backtest_summary_path,
    load_predictive_config,
    model_bundle_path,
    output_root,
    score_from_config,
    save_json,
)

OOS_START = pd.Timestamp("2024-01-01")
VALIDATION_ROOT = PROJECT_ROOT / "results" / "model_signals" / "validation_runs"
VALIDATION_ROOT.mkdir(parents=True, exist_ok=True)


def load_initial_capital() -> float:
    trading_cfg = PROJECT_ROOT / "config" / "trading.yaml"
    if not trading_cfg.exists():
        return 5_000_000.0
    cfg = yaml.safe_load(trading_cfg.read_text(encoding="utf-8")) or {}
    capital = cfg.get("capital", {})
    return float(capital.get("initial", 5_000_000.0))


def annualize_return(total_return: float, start_date: pd.Timestamp, end_date: pd.Timestamp) -> float:
    days = max(int((end_date - start_date).days), 1)
    if 1.0 + total_return <= 0.0:
        return -1.0
    return (1.0 + total_return) ** (365.0 / float(days)) - 1.0


def compute_oos_metrics(overlay_csv: Path) -> Dict[str, float]:
    frame = pd.read_csv(overlay_csv, parse_dates=["date"]).sort_values("date")
    frame = frame.loc[frame["date"] >= OOS_START].copy()
    if frame.empty:
        return {"oos_ann": 0.0, "oos_dd": 0.0, "oos_sharpe": 0.0}
    returns = frame["overlay_return"].astype(float)
    total = float((1.0 + returns).prod() - 1.0)
    ann = annualize_return(total, frame["date"].iloc[0], frame["date"].iloc[-1])
    dd = float(frame["drawdown"].min())
    std = float(returns.std(ddof=0))
    sharpe = float(returns.mean() / std * math.sqrt(252.0)) if std > 0 else 0.0
    return {"oos_ann": ann, "oos_dd": dd, "oos_sharpe": sharpe}


def find_latest_backtest_csv(strategy_name: str) -> Path | None:
    pattern = f"backtest_{strategy_name}_historical_csi300_*.csv"
    files = sorted((PROJECT_ROOT / "results").glob(pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def compute_trading_metrics(backtest_csv: Path, initial_capital: float) -> Dict[str, float]:
    frame = pd.read_csv(backtest_csv)
    fee = frame["fee_amount"].astype(float)
    buy = frame["buy_count"].astype(float)
    sell = frame["sell_count"].astype(float)
    traded = (buy + sell) > 0
    traded_frame = frame.loc[traded]
    avg_trade_count = float((buy[traded] + sell[traded]).mean()) if traded.any() else 0.0
    avg_buy_count = float(buy[buy > 0].mean()) if (buy > 0).any() else 0.0
    avg_sell_count = float(sell[sell > 0].mean()) if (sell > 0).any() else 0.0
    return {
        "fee_total": float(fee.sum()),
        "fee_ratio_initial": float(fee.sum() / initial_capital) if initial_capital > 0 else 0.0,
        "trade_days": int(traded.sum()),
        "avg_trade_count": avg_trade_count,
        "avg_buy_count": avg_buy_count,
        "avg_sell_count": avg_sell_count,
        "rows": int(len(frame)),
        "rebalance_like_days": int(len(traded_frame)),
    }


def load_csi300_close() -> pd.Series:
    idx_path = PROJECT_ROOT / "data" / "tushare" / "index_daily.parquet"
    frame = pd.read_parquet(idx_path, columns=["ts_code", "trade_date", "close"])
    frame = frame.loc[frame["ts_code"] == "000300.SH", ["trade_date", "close"]].copy()
    frame["date"] = pd.to_datetime(frame["trade_date"].astype(str))
    frame = frame.sort_values("date")
    return frame.set_index("date")["close"].astype(float)


def compute_yearly_excess_rows(strategy_name: str, overlay_csv: Path, csi300_close: pd.Series) -> List[dict]:
    overlay = pd.read_csv(overlay_csv, parse_dates=["date"]).sort_values("date")
    if overlay.empty:
        return []
    sret = overlay.set_index("date")["overlay_return"].astype(float)
    dates = sret.index
    bclose = csi300_close.reindex(dates).ffill()
    bret = bclose.pct_change().fillna(0.0)
    rows: List[dict] = []
    for year in sorted(dates.year.unique()):
        mask = dates.year == year
        sr = float((1.0 + sret.loc[mask]).prod() - 1.0)
        br = float((1.0 + bret.loc[mask]).prod() - 1.0)
        rows.append(
            {
                "strategy": strategy_name,
                "year": int(year),
                "strategy_return": sr,
                "csi300_return": br,
                "excess_return": sr - br,
            }
        )
    return rows


def ensure_backtest(cfg: dict, rerun_score: bool = False) -> dict:
    if rerun_score:
        score_from_config(cfg)
    summary_path = backtest_summary_path(cfg)
    if not summary_path.exists():
        _, summary = backtest_from_config(cfg, engine="qlib")
        return summary
    return yaml.safe_load(summary_path.read_text(encoding="utf-8"))


def build_variant_cfg(base_cfg: dict, name: str) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["name"] = name
    cfg["output"]["root"] = str((VALIDATION_ROOT / name).resolve())
    return cfg


def copy_model_bundle(src_cfg: dict, dst_cfg: dict) -> None:
    src = model_bundle_path(src_cfg)
    dst = model_bundle_path(dst_cfg)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def run_variant(base_cfg: dict, name: str, patch_fn) -> dict:
    cfg = build_variant_cfg(base_cfg, name)
    patch_fn(cfg)
    copy_model_bundle(base_cfg, cfg)
    score_from_config(cfg)
    _, summary = backtest_from_config(cfg, engine="qlib")
    summary["config_path"] = "generated"
    return summary


def main() -> None:
    initial_capital = load_initial_capital()
    csi300_close = load_csi300_close()

    baseline_configs = [
        ("main_qvf", PROJECT_ROOT / "config" / "models" / "qvf_core_plus_fixed80_k6_overlay_fullspan_turnover_soft_pruned.yaml"),
        ("push25_cq10", PROJECT_ROOT / "config" / "models" / "push25_cq10_k8d2_very_tight.yaml"),
        ("push25_cq7", PROJECT_ROOT / "config" / "models" / "push25_cq7_k8d2_very_tight.yaml"),
    ]

    rows: List[dict] = []
    yearly_rows: List[dict] = []

    # Baseline summaries
    baseline_cfg_map: Dict[str, dict] = {}
    for label, path in baseline_configs:
        cfg = load_predictive_config(path)
        baseline_cfg_map[label] = cfg
        summary = ensure_backtest(cfg, rerun_score=False)
        overlay_csv = Path(summary["results_file"])
        backtest_csv = find_latest_backtest_csv(str(cfg["name"]))
        trade_metrics = compute_trading_metrics(backtest_csv, initial_capital) if backtest_csv else {}
        oos = compute_oos_metrics(overlay_csv)
        yearly_rows.extend(compute_yearly_excess_rows(str(cfg["name"]), overlay_csv, csi300_close))
        rows.append(
            {
                "group": "baseline",
                "label": label,
                "strategy_name": cfg["name"],
                "annual_return": float(summary["annual_return"]),
                "max_drawdown": float(summary["max_drawdown"]),
                "sharpe_ratio": float(summary["sharpe_ratio"]),
                **oos,
                "overlay_csv": str(overlay_csv),
                "backtest_csv": str(backtest_csv) if backtest_csv else "",
                **trade_metrics,
            }
        )

    # Stress tests on cq10
    cq10_base = baseline_cfg_map["push25_cq10"]

    stress_specs = [
        (
            "cq10_tight_dd",
            lambda c: c["overlay"].update(
                {
                    "target_vol": 0.19,
                    "vol_lookback": 20,
                    "trend_lookback": 20,
                    "trend_exposure": 0.70,
                    "dd_soft": 0.015,
                    "dd_hard": 0.030,
                    "soft_exposure": 0.70,
                    "hard_exposure": 0.35,
                    "exposure_min": 0.0,
                    "exposure_max": 1.0,
                }
            ),
        ),
        (
            "cq10_cost150",
            lambda c: c["trading"].update(
                {
                    "buy_commission_rate": float(c["trading"]["buy_commission_rate"]) * 1.5,
                    "sell_commission_rate": float(c["trading"]["sell_commission_rate"]) * 1.5,
                    "sell_stamp_tax_rate": float(c["trading"]["sell_stamp_tax_rate"]) * 1.5,
                    "min_buy_commission": float(c["trading"]["min_buy_commission"]) * 1.5,
                    "min_sell_commission": float(c["trading"]["min_sell_commission"]) * 1.5,
                    "slippage_bps": float(c["trading"]["slippage_bps"]) * 1.5,
                    "impact_bps": float(c["trading"]["impact_bps"]) * 1.5,
                }
            ),
        ),
        (
            "cq10_slippage10",
            lambda c: c["trading"].update({"slippage_bps": 10.0}),
        ),
        (
            "cq10_topk6",
            lambda c: c["selection"].update({"topk": 6, "sticky": 4, "churn_limit": 2}),
        ),
    ]

    for name, patch_fn in stress_specs:
        summary = run_variant(cq10_base, name, patch_fn=patch_fn)
        overlay_csv = Path(summary["results_file"])
        backtest_csv = find_latest_backtest_csv(name)
        trade_metrics = compute_trading_metrics(backtest_csv, initial_capital) if backtest_csv else {}
        oos = compute_oos_metrics(overlay_csv)
        yearly_rows.extend(compute_yearly_excess_rows(name, overlay_csv, csi300_close))
        rows.append(
            {
                "group": "stress",
                "label": name,
                "strategy_name": name,
                "annual_return": float(summary["annual_return"]),
                "max_drawdown": float(summary["max_drawdown"]),
                "sharpe_ratio": float(summary["sharpe_ratio"]),
                **oos,
                "overlay_csv": str(overlay_csv),
                "backtest_csv": str(backtest_csv) if backtest_csv else "",
                **trade_metrics,
            }
        )

    summary_df = pd.DataFrame(rows).sort_values(["group", "annual_return"], ascending=[True, False])
    yearly_df = pd.DataFrame(yearly_rows).sort_values(["strategy", "year"])

    summary_csv = VALIDATION_ROOT / "strategy_validation_summary.csv"
    yearly_csv = VALIDATION_ROOT / "strategy_validation_yearly_excess.csv"
    summary_df.to_csv(summary_csv, index=False)
    yearly_df.to_csv(yearly_csv, index=False)

    report = {
        "summary_csv": str(summary_csv),
        "yearly_csv": str(yearly_csv),
        "rows": int(len(summary_df)),
        "yearly_rows": int(len(yearly_df)),
    }
    save_json(report, VALIDATION_ROOT / "strategy_validation_report.json")
    print(f"[OK] summary -> {summary_csv}")
    print(f"[OK] yearly  -> {yearly_csv}")


if __name__ == "__main__":
    main()
