#!/usr/bin/env python3
"""把 overlay_results.csv 按年和指数基准做收益/回撤对比。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_strategy_returns(path: Path | str) -> pd.Series:
    frame = pd.read_csv(path, parse_dates=["date"])
    return_col = "overlay_return" if "overlay_return" in frame.columns else "return"
    if return_col not in frame.columns:
        raise ValueError(f"{path} 缺少 overlay_return/return 列")
    returns = frame.sort_values("date").drop_duplicates("date").set_index("date")[return_col]
    return returns.astype(float)


def load_index_returns(
    path: Path | str = PROJECT_ROOT / "data" / "tushare" / "index_daily.parquet",
    index_code: str = "000300.SH",
) -> pd.Series:
    frame = pd.read_parquet(path)
    if "ts_code" in frame.columns:
        frame = frame[frame["ts_code"] == index_code].copy()
    if frame.empty:
        raise ValueError(f"没有找到指数行情: {index_code}")
    if "trade_date" in frame.columns:
        dates = pd.to_datetime(frame["trade_date"].astype(str))
    else:
        dates = pd.to_datetime(frame["date"])
    close = (
        frame.assign(date=dates)
        .sort_values("date")
        .drop_duplicates("date")
        .set_index("date")["close"]
        .astype(float)
    )
    return close.pct_change().fillna(0.0)


def period_metrics(returns: pd.Series) -> dict[str, float]:
    returns = pd.Series(returns, dtype=float).dropna()
    if returns.empty:
        return {"return": 0.0, "max_drawdown": 0.0, "sharpe": 0.0}
    nav = (1.0 + returns).cumprod()
    max_drawdown = float((nav / nav.cummax() - 1.0).min())
    std = float(returns.std(ddof=0))
    sharpe = float(returns.mean() / std * np.sqrt(252)) if std > 0 else 0.0
    return {
        "return": float(nav.iloc[-1] - 1.0),
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
    }


def yearly_compare(strategy_returns: pd.Series, index_returns: pd.Series) -> pd.DataFrame:
    rows = []
    strategy_returns = strategy_returns.sort_index()
    index_returns = index_returns.sort_index()
    for year, strategy_group in strategy_returns.groupby(strategy_returns.index.year):
        benchmark_group = index_returns.reindex(strategy_group.index).fillna(0.0)
        strategy = period_metrics(strategy_group)
        benchmark = period_metrics(benchmark_group)
        rows.append(
            {
                "year": int(year),
                "strategy_return": strategy["return"],
                "strategy_max_drawdown": strategy["max_drawdown"],
                "strategy_sharpe": strategy["sharpe"],
                "benchmark_return": benchmark["return"],
                "benchmark_max_drawdown": benchmark["max_drawdown"],
                "benchmark_sharpe": benchmark["sharpe"],
                "excess_return": strategy["return"] - benchmark["return"],
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy-csv", required=True)
    parser.add_argument("--index-parquet", default="data/tushare/index_daily.parquet")
    parser.add_argument("--index-code", default="000300.SH")
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    strategy_returns = load_strategy_returns(PROJECT_ROOT / args.strategy_csv)
    index_returns = load_index_returns(PROJECT_ROOT / args.index_parquet, args.index_code)
    result = yearly_compare(strategy_returns, index_returns)
    output_csv = PROJECT_ROOT / args.output_csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    print(f"[OK] yearly comparison -> {output_csv}")
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(result.to_string(index=False, formatters=_formatters(result.columns)))


def _formatters(columns: pd.Index) -> dict[str, object]:
    percent_cols = [
        column
        for column in columns
        if column.endswith(("return", "drawdown"))
    ]
    return {column: (lambda value: f"{float(value):.2%}") for column in percent_cols}


if __name__ == "__main__":
    main()
