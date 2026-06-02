#!/usr/bin/env python3
"""
正式单因子零成本回放

口径：
- 复用 results/formal_single_factor_results.csv 中已有的正式单因子策略
- 不重建因子口径，不改选股参数，只把交易成本置零：
  - buy_commission_rate = 0
  - sell_commission_rate = 0
  - sell_stamp_tax_rate = 0
  - min_buy_commission = 0
  - min_sell_commission = 0
  - slippage_bps = 0
  - impact_bps = 0
- 保留涨停/跌停可交易性约束

输出：
- results/formal_single_factor_results_nocost.csv
- results/formal_single_factor_results_nocost_day.csv
- results/formal_single_factor_results_nocost_week.csv
- results/formal_single_factor_results_nocost_summary.md
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.strategy import Strategy
from modules.backtest.composite import run_strategy_backtest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

SOURCE_CSV = RESULTS_DIR / "formal_single_factor_results.csv"
OUT_CSV = RESULTS_DIR / "formal_single_factor_results_nocost.csv"
OUT_DAY_CSV = RESULTS_DIR / "formal_single_factor_results_nocost_day.csv"
OUT_WEEK_CSV = RESULTS_DIR / "formal_single_factor_results_nocost_week.csv"
OUT_SUMMARY = RESULTS_DIR / "formal_single_factor_results_nocost_summary.md"

ZERO_COST = {
    "open_cost": 0.0,
    "close_cost": 0.0,
    "buy_commission_rate": 0.0,
    "sell_commission_rate": 0.0,
    "sell_stamp_tax_rate": 0.0,
    "min_buy_commission": 0.0,
    "min_sell_commission": 0.0,
    "slippage_bps": 0.0,
    "impact_bps": 0.0,
}


def _repo_relative(value: str | Path | None) -> str:
    if value is None:
        return ""
    text = str(value)
    if not text:
        return ""
    path = Path(text)
    if not path.is_absolute():
        return text
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return text


def _run_one(row: pd.Series) -> dict[str, object]:
    strategy_name = str(row["strategy_name"])
    strategy = Strategy.load(strategy_name)
    strategy.generate_selections(force=False)

    trading_cost = dict(getattr(strategy, "trading_cost", {}) or {})
    trading_cost.update(ZERO_COST)
    strategy.trading_cost = trading_cost

    result = run_strategy_backtest(strategy=strategy, engine="qlib")
    daily_returns = result.daily_returns
    return {
        "status": "success",
        "freq": row["freq"],
        "factor_name": row["factor_name"],
        "direction": row["direction"],
        "negate": row["negate"],
        "source": row["source"],
        "expression": row["expression"],
        "expr_hash": row["expr_hash"],
        "window_scale": row.get("window_scale", 1),
        "strategy_name": strategy_name,
        "strategy_yaml": row["strategy_yaml"],
        "selection_file": _repo_relative(strategy.selections_path()),
        "results_file": _repo_relative(result.metadata.get("results_file", "")),
        "annual_return": float(result.annual_return),
        "sharpe_ratio": float(result.sharpe_ratio),
        "max_drawdown": float(result.max_drawdown),
        "total_return": float(result.total_return),
        "win_rate": float((daily_returns > 0).mean()) if not daily_returns.empty else 0.0,
        "n_days": int(len(daily_returns)),
        "start_date": str(daily_returns.index.min().date()) if not daily_returns.empty else "",
        "end_date": str(daily_returns.index.max().date()) if not daily_returns.empty else "",
        "costed_annual_return": float(row["annual_return"]),
        "costed_sharpe_ratio": float(row["sharpe_ratio"]),
        "costed_max_drawdown": float(row["max_drawdown"]),
        "costed_total_return": float(row["total_return"]),
        "annual_return_delta": float(result.annual_return) - float(row["annual_return"]),
        "sharpe_ratio_delta": float(result.sharpe_ratio) - float(row["sharpe_ratio"]),
        "max_drawdown_delta": float(result.max_drawdown) - float(row["max_drawdown"]),
        "total_return_delta": float(result.total_return) - float(row["total_return"]),
    }


def _write_summary(df: pd.DataFrame) -> None:
    lines = [
        "# 正式单因子零成本回放",
        "",
        "口径：",
        "- 基于 `results/formal_single_factor_results.csv` 已有正式策略重跑",
        "- 手续费、印花税、滑点、冲击成本全部为 0",
        "- 保留原有选股和可交易性约束",
        "",
        f"- 策略数：`{len(df)}`",
        "",
    ]

    for freq in ("day", "week"):
        sub = df[df["freq"] == freq].sort_values(
            ["annual_return", "sharpe_ratio"],
            ascending=[False, False],
        ).head(20)
        if sub.empty:
            continue
        lines.append(f"## {freq.title()} Top 20 By Annual Return")
        lines.append("")
        show = sub[
            [
                "factor_name",
                "direction",
                "annual_return",
                "sharpe_ratio",
                "max_drawdown",
                "annual_return_delta",
                "sharpe_ratio_delta",
            ]
        ].copy()
        for col in ("annual_return", "max_drawdown", "annual_return_delta"):
            show[col] = show[col].map(lambda v: f"{v:.4%}")
        for col in ("sharpe_ratio", "sharpe_ratio_delta"):
            show[col] = show[col].map(lambda v: f"{v:.4f}")
        try:
            lines.append(show.to_markdown(index=False))
        except Exception:
            lines.append("```")
            lines.append(show.to_string(index=False))
            lines.append("```")
        lines.append("")

    OUT_SUMMARY.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    src = pd.read_csv(SOURCE_CSV)
    src = src[src["status"] == "success"].copy()
    src = src.drop_duplicates(subset=["freq", "factor_name", "negate", "expr_hash"], keep="last")
    src = src.sort_values(["freq", "strategy_name"]).reset_index(drop=True)

    rows = []
    for item in src.itertuples(index=False):
        rows.append(_run_one(pd.Series(item._asdict())))

    out = pd.DataFrame(rows)
    out = out.sort_values(
        ["freq", "annual_return", "sharpe_ratio"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False)
    out[out["freq"] == "day"].to_csv(OUT_DAY_CSV, index=False)
    out[out["freq"] == "week"].to_csv(OUT_WEEK_CSV, index=False)
    _write_summary(out)
    print(OUT_CSV)
    print(OUT_DAY_CSV)
    print(OUT_WEEK_CSV)
    print(OUT_SUMMARY)


if __name__ == "__main__":
    main()
