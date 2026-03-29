#!/usr/bin/env python3
"""Small parameter scan around the best ma120 mean-reversion baseline."""

from __future__ import annotations

import argparse
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from core.strategy import Strategy
from modules.backtest.qlib_engine import QlibBacktestEngine

BASE_STRATEGY = "sf_ma120dev_rev_daily_buffered_k50"
RESULTS_DIR = Path("results")


def build_candidates(base: Strategy) -> list[tuple[str, int, int, int]]:
    topk_values = [40, 50, 60]
    buffer_values = [8, 12, 16]
    churn_values = [3, 5, 7]

    candidates: list[tuple[str, int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()

    for topk in topk_values:
        item = (topk, base.buffer, base.churn_limit)
        if item not in seen:
            seen.add(item)
            candidates.append((f"topk{topk}", *item))

    for buffer in buffer_values:
        item = (base.topk, buffer, base.churn_limit)
        if item not in seen:
            seen.add(item)
            candidates.append((f"buffer{buffer}", *item))

    for churn in churn_values:
        item = (base.topk, base.buffer, churn)
        if item not in seen:
            seen.add(item)
            candidates.append((f"churn{churn}", *item))

    return candidates


def compute_metrics(results_file: str) -> dict[str, float]:
    df = pd.read_csv(results_file, parse_dates=["date"])
    ret = df["return"].fillna(0.0)
    gross = df["gross_return"].fillna(0.0)

    nav = (1.0 + ret).cumprod()
    gross_nav = (1.0 + gross).cumprod()
    days = (df["date"].iloc[-1] - df["date"].iloc[0]).days

    annual_return = nav.iloc[-1] ** (365 / days) - 1 if days > 0 and nav.iloc[-1] > 0 else np.nan
    gross_annual_return = (
        gross_nav.iloc[-1] ** (365 / days) - 1 if days > 0 and gross_nav.iloc[-1] > 0 else np.nan
    )
    sharpe = ret.mean() / ret.std(ddof=1) * np.sqrt(252) if ret.std(ddof=1) > 0 else np.nan
    max_drawdown = float((nav / nav.cummax() - 1).min())
    trade_days = (df["sell_count"] + df["buy_count"]) > 0

    return {
        "total_return": float(nav.iloc[-1] - 1),
        "annual_return": float(annual_return),
        "gross_total_return": float(gross_nav.iloc[-1] - 1),
        "gross_annual_return": float(gross_annual_return),
        "sharpe": float(sharpe),
        "max_drawdown": max_drawdown,
        "total_fee_amount": float(df["fee_amount"].sum()),
        "sell_count": int(df["sell_count"].sum()),
        "buy_count": int(df["buy_count"].sum()),
        "trade_day_ratio": float(trade_days.mean()),
        "avg_monthly_sells": float(df["sell_count"].sum() / (len(df) / 21)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan single-variable parameter changes around the ma120 reverse buffered baseline."
    )
    return parser.parse_args()


def main() -> None:
    parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    base = Strategy.load(BASE_STRATEGY)
    candidates = build_candidates(base)
    engine = QlibBacktestEngine()

    rows: list[dict[str, object]] = []

    print(f"base={base.name} topk={base.topk} buffer={base.buffer} churn={base.churn_limit}")
    print(f"candidates={len(candidates)}")

    for idx, (label, topk, buffer, churn_limit) in enumerate(candidates, start=1):
        strategy = deepcopy(base)
        strategy.name = (
            f"experimental/turnover/_scan_ma120_rev_k{topk}_b{buffer}_c{churn_limit}"
        )
        strategy.display_name = strategy.name.split("/")[-1]
        strategy.topk = topk
        strategy.buffer = buffer
        strategy.churn_limit = churn_limit
        strategy.entry_rank = topk - 5
        strategy.exit_rank = topk + buffer + 3

        print(
            f"[{idx}/{len(candidates)}] {label}: "
            f"topk={topk} buffer={buffer} churn={churn_limit}"
        )

        t0 = time.perf_counter()
        df_sel = strategy.generate_selections(force=True)
        selection_seconds = time.perf_counter() - t0

        t1 = time.perf_counter()
        result = engine.run(strategy=strategy)
        backtest_seconds = time.perf_counter() - t1
        results_file = result.metadata.get("results_file")
        metrics = compute_metrics(results_file)

        row = {
            "label": label,
            "topk": topk,
            "buffer": buffer,
            "churn_limit": churn_limit,
            "selection_seconds": selection_seconds,
            "selection_rows": len(df_sel),
            "backtest_seconds": backtest_seconds,
            "results_file": results_file,
            **metrics,
        }
        rows.append(row)
        print(
            f"  annual={row['annual_return']:.2%} total={row['total_return']:.2%} "
            f"sharpe={row['sharpe']:.3f} mdd={row['max_drawdown']:.2%}"
        )

    df_out = pd.DataFrame(rows).sort_values(
        ["annual_return", "sharpe", "total_return"], ascending=[False, False, False]
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"ma120_buffer_scan_{timestamp}.csv"
    df_out.to_csv(out_path, index=False)
    print(f"saved={out_path}")
    print(df_out[["label", "topk", "buffer", "churn_limit", "annual_return", "sharpe", "max_drawdown", "total_fee_amount"]].to_string(index=False))


if __name__ == "__main__":
    main()
