#!/usr/bin/env python3
"""Profile selection and backtest runtime for a single strategy."""

from __future__ import annotations

import argparse
import time

from core.strategy import Strategy
from modules.backtest.qlib_engine import QlibBacktestEngine


def _format_seconds(value: float) -> str:
    return f"{value:.3f}s"


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile strategy selection/backtest runtime.")
    parser.add_argument("--strategy", "-s", default="top15_core_trend")
    parser.add_argument("--skip-select", action="store_true")
    args = parser.parse_args()

    strategy = Strategy.load(args.strategy)

    print(f"strategy={strategy.name}")
    print(f"selection_path={strategy.selections_path()}")
    print(f"selection_stale_before={strategy.selections_are_stale()}")

    if not args.skip_select:
        t0 = time.perf_counter()
        df_sel = strategy.generate_selections(force=True)
        t1 = time.perf_counter()
        print(f"selection_seconds={_format_seconds(t1 - t0)}")
        print(f"selection_rows={len(df_sel)}")
        print(f"selection_periods={df_sel['date'].nunique() if not df_sel.empty else 0}")
        print(f"selection_stale_after={strategy.selections_are_stale()}")

    t2 = time.perf_counter()
    result = QlibBacktestEngine().run(strategy=strategy)
    t3 = time.perf_counter()
    print(f"backtest_seconds={_format_seconds(t3 - t2)}")
    print(f"backtest_daily_returns={len(result.daily_returns)}")
    print(f"backtest_result_file={result.metadata.get('results_file')}")


if __name__ == "__main__":
    main()
