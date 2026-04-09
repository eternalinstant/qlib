#!/usr/bin/env python3
"""
并行重算落后的选股缓存。

只处理：
- CSV 不存在
- CSV 为空
- 最后日期落后于当前本地最新交易日

示例：
    python3 scripts/parallel_regenerate_selections.py --shards 4 --shard 0
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

import pandas as pd

from core.strategy import Strategy
from core.selection import compute_rebalance_dates
from modules.data.updater import DataUpdater


@dataclass
class Task:
    name: str
    force: bool
    update_start_date: str | None


def build_tasks() -> list[Task]:
    updater = DataUpdater()
    expected = pd.Timestamp(updater.get_last_trading_date())
    trade_dates = pd.DatetimeIndex(
        pd.read_parquet(
            updater.tushare_dir / "daily_basic.parquet",
            columns=["trade_date"],
        )["trade_date"].drop_duplicates().sort_values()
    )
    trade_dates = pd.to_datetime(trade_dates.astype(str))
    latest_rebalance = {
        "day": trade_dates[-1],
        "week": compute_rebalance_dates(pd.Series(trade_dates), freq="week")[-1],
        "biweek": compute_rebalance_dates(pd.Series(trade_dates), freq="biweek")[-1],
        "month": compute_rebalance_dates(pd.Series(trade_dates), freq="month")[-1],
    }
    tasks: list[Task] = []

    for name in Strategy.list_available():
        strategy = Strategy.load(name)
        csv_path = strategy.selections_path()
        freq = strategy.rebalance_freq

        if not csv_path.exists():
            tasks.append(Task(name=name, force=True, update_start_date=None))
            continue

        df = pd.read_csv(csv_path, parse_dates=["date"], usecols=["date"])
        if df.empty:
            tasks.append(Task(name=name, force=True, update_start_date=None))
            continue

        last_date = df["date"].max()
        if last_date < latest_rebalance[freq]:
            update_start = last_date + pd.Timedelta(days=1)
            tasks.append(
                Task(
                    name=name,
                    force=False,
                    update_start_date=update_start.strftime("%Y-%m-%d"),
                )
            )

    return tasks


def run_task(idx: int, total: int, task: Task) -> bool:
    strategy = Strategy.load(task.name)
    strategy.validate_data_requirements()

    if task.force:
        print(f"[{idx}/{total}] 全量重算 {task.name}")
        strategy.generate_selections(force=True)
    else:
        print(f"[{idx}/{total}] 增量更新 {task.name}: -> {task.update_start_date}")
        strategy.generate_selections(
            force=False,
            update_start_date=task.update_start_date,
        )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="并行重算落后的选股缓存")
    parser.add_argument("--shards", type=int, default=1, help="总分片数")
    parser.add_argument("--shard", type=int, default=0, help="当前分片编号，从 0 开始")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    tasks = build_tasks()
    selected = [task for i, task in enumerate(tasks) if i % args.shards == args.shard]

    print(f"all_tasks={len(tasks)} shard={args.shard}/{args.shards} selected={len(selected)}")

    ok = 0
    failed = 0
    for i, task in enumerate(selected, 1):
        try:
            run_task(i, len(selected), task)
            ok += 1
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            failed += 1
            print(f"[ERROR] {task.name}: {exc}")

    print(f"done shard={args.shard}/{args.shards} ok={ok} failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
