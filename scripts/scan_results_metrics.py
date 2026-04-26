#!/usr/bin/env python3

from __future__ import annotations

import csv
import math
import statistics
from datetime import date
from pathlib import Path


RESULTS_DIR = Path("~/code/qlib/results").expanduser()
ANNUAL_RF = 0.03
TRADING_DAYS = 252
MIN_CAGR = 0.20
MAX_ABS_DD = 0.12


def load_returns(csv_path: Path) -> tuple[list[date], list[float]]:
    rows: list[tuple[date, float]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if "date" not in (reader.fieldnames or []) or "return" not in (reader.fieldnames or []):
            return [], []

        for row in reader:
            raw_date = (row.get("date") or "").strip()
            raw_return = (row.get("return") or "").strip()
            if not raw_date or not raw_return:
                continue
            try:
                rows.append((date.fromisoformat(raw_date), float(raw_return)))
            except ValueError:
                continue

    rows.sort(key=lambda item: item[0])
    dates = [item[0] for item in rows]
    returns = [item[1] for item in rows]
    return dates, returns


def compute_metrics(dates: list[date], returns: list[float]) -> tuple[float, float, float, float]:
    nav = 1.0
    peak = 1.0
    max_dd = 0.0
    for daily_return in returns:
        nav *= 1.0 + daily_return
        peak = max(peak, nav)
        max_dd = min(max_dd, nav / peak - 1.0)

    span_days = (dates[-1] - dates[0]).days
    if span_days > 0:
        years = span_days / 365.25
        cagr = nav ** (1.0 / years) - 1.0
    else:
        cagr = nav ** (TRADING_DAYS / len(returns)) - 1.0

    rf_daily = (1.0 + ANNUAL_RF) ** (1.0 / TRADING_DAYS) - 1.0
    excess_returns = [daily_return - rf_daily for daily_return in returns]
    if len(excess_returns) > 1:
        daily_std = statistics.stdev(excess_returns)
        sharpe = (
            statistics.mean(excess_returns) / daily_std * math.sqrt(TRADING_DAYS)
            if daily_std > 0
            else float("nan")
        )
    else:
        sharpe = float("nan")

    calmar = cagr / abs(max_dd) if max_dd != 0 else float("inf")
    return cagr, max_dd, sharpe, calmar


def main() -> None:
    matches: list[tuple[Path, float, float, float, float]] = []
    for csv_path in sorted(RESULTS_DIR.rglob("*.csv")):
        dates, returns = load_returns(csv_path)
        if len(returns) < 2:
            continue
        if any(1.0 + daily_return <= 0.0 for daily_return in returns):
            continue

        cagr, max_dd, sharpe, calmar = compute_metrics(dates, returns)
        if cagr > MIN_CAGR and abs(max_dd) <= MAX_ABS_DD:
            matches.append((csv_path, cagr, max_dd, sharpe, calmar))

    matches.sort(key=lambda item: item[1], reverse=True)

    if not matches:
        print("无符合条件的结果")
        return

    print("file,CAGR,MaxDD,Sharpe,Calmar")
    for csv_path, cagr, max_dd, sharpe, calmar in matches:
        print(
            f"{csv_path.name},{cagr:.2%},{abs(max_dd):.2%},"
            f"{sharpe:.2f},{calmar:.2f}"
        )


if __name__ == "__main__":
    main()
