#!/usr/bin/env python3
"""
批量复核正式日频单因子在 PyBroker 下的结果。

默认口径：
- 样本来源: results/formal_single_factor_results_day.csv
- 过滤: status=success, annual_return > 0
- 排序: 先按 Qlib 年化，再按 Qlib 夏普
- 输出: results/formal_single_factor_pybroker_day.csv / .md
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd

from core.strategy import Strategy
from modules.backtest.composite import run_strategy_backtest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
SOURCE_CSV = RESULTS_DIR / "formal_single_factor_results_day.csv"
OUT_CSV = RESULTS_DIR / "formal_single_factor_pybroker_day.csv"
OUT_MD = RESULTS_DIR / "formal_single_factor_pybroker_day.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-annual", type=float, default=0.0)
    parser.add_argument("--min-sharpe", type=float, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _load_candidates(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(SOURCE_CSV)
    df = df[df["status"] == "success"].copy()
    df = df[df["annual_return"] > args.min_annual]
    if args.min_sharpe is not None:
        df = df[df["sharpe_ratio"] > args.min_sharpe]
    df = df.sort_values(["annual_return", "sharpe_ratio"], ascending=False).reset_index(drop=True)
    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()
    return df


def _pybroker_csv_metrics(path: Path) -> dict:
    df = pd.read_csv(path)
    equity_col = "equity" if "equity" in df.columns else df.columns[0]
    equity = df[equity_col].dropna()
    daily_ret = equity.pct_change().dropna()

    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) > 1 else 0.0
    if len(daily_ret) > 0 and equity.iloc[0] > 0:
        annual_return = float((equity.iloc[-1] / equity.iloc[0]) ** (252 / len(daily_ret)) - 1)
    else:
        annual_return = 0.0
    sharpe_ratio = float(daily_ret.mean() / daily_ret.std() * (252 ** 0.5)) if daily_ret.std() else 0.0
    max_drawdown = float((equity / equity.cummax() - 1).min()) if len(equity) else 0.0
    win_rate = float((daily_ret > 0).mean()) if len(daily_ret) else 0.0
    start_date = str(df["date"].iloc[0]) if not df.empty and "date" in df.columns else ""
    end_date = str(df["date"].iloc[-1]) if not df.empty and "date" in df.columns else ""

    return {
        "pybroker_total_return": total_return,
        "pybroker_annual_return": annual_return,
        "pybroker_sharpe_ratio": sharpe_ratio,
        "pybroker_max_drawdown": max_drawdown,
        "pybroker_win_rate": win_rate,
        "pybroker_start_date": start_date,
        "pybroker_end_date": end_date,
        "pybroker_results_file": str(path),
    }


def _latest_pybroker_file(strategy_name: str) -> Path | None:
    slug = strategy_name.replace("/", "__")
    files = sorted(glob.glob(str(RESULTS_DIR / f"pybroker_{slug}_*.csv")))
    return Path(files[-1]) if files else None


def _run_one(strategy_name: str, force: bool) -> dict:
    latest = None if force else _latest_pybroker_file(strategy_name)
    if latest is not None:
        return _pybroker_csv_metrics(latest)

    strategy = Strategy.load(strategy_name)
    result = run_strategy_backtest(strategy=strategy, engine="pybroker")
    results_file = result.metadata.get("results_file")
    if not results_file:
        raise RuntimeError(f"PyBroker 未返回结果文件: {strategy_name}")
    return _pybroker_csv_metrics(Path(results_file))


def main() -> None:
    args = parse_args()
    candidates = _load_candidates(args)

    print(f"样本数: {len(candidates)}")
    rows = []
    for i, row in candidates.iterrows():
        strategy_name = str(row["strategy_name"])
        print(
            f"[{i+1:02d}/{len(candidates):02d}] {row['factor_name']} {row['direction']} "
            f"Qlib年化:{row['annual_return']:+.2%} 夏普:{row['sharpe_ratio']:.4f}"
        )
        py = _run_one(strategy_name, force=args.force)
        rows.append(
            {
                "factor_name": row["factor_name"],
                "direction": row["direction"],
                "strategy_name": strategy_name,
                "strategy_yaml": row["strategy_yaml"],
                "qlib_annual_return": float(row["annual_return"]),
                "qlib_sharpe_ratio": float(row["sharpe_ratio"]),
                "qlib_max_drawdown": float(row["max_drawdown"]),
                **py,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        out.to_csv(OUT_CSV, index=False)
        OUT_MD.write_text("# Formal Single Factor PyBroker Day\n\nNo rows.\n", encoding="utf-8")
        print(f"已保存: {OUT_CSV}")
        print(f"已保存: {OUT_MD}")
        return

    out["annual_return_delta_py_minus_qlib"] = out["pybroker_annual_return"] - out["qlib_annual_return"]
    out["sharpe_delta_py_minus_qlib"] = out["pybroker_sharpe_ratio"] - out["qlib_sharpe_ratio"]
    out["max_drawdown_delta_py_minus_qlib"] = out["pybroker_max_drawdown"] - out["qlib_max_drawdown"]
    out = out.sort_values(["pybroker_annual_return", "pybroker_sharpe_ratio"], ascending=False).reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False)

    display = out[
        [
            "factor_name",
            "direction",
            "qlib_annual_return",
            "qlib_sharpe_ratio",
            "pybroker_annual_return",
            "pybroker_sharpe_ratio",
            "pybroker_max_drawdown",
            "pybroker_results_file",
        ]
    ]
    OUT_MD.write_text(
        "# Formal Single Factor PyBroker Day\n\n"
        f"Rows: {len(out)}\n\n"
        + display.to_markdown(index=False),
        encoding="utf-8",
    )

    print(display.head(20).to_string(index=False))
    print(f"已保存: {OUT_CSV}")
    print(f"已保存: {OUT_MD}")


if __name__ == "__main__":
    main()
