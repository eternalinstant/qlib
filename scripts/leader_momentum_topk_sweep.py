#!/usr/bin/env python3
"""
龙头追涨因子持仓数量扫描（Qlib）

统一口径：
- 龙头池：total_mv 前 20%（≈ 前 1000 市值）
- 日频正式单因子执行壳
- 只变 topk；entry_rank / exit_rank 按 topk 同比例推导

输出：
- results/leader_momentum_topk_sweep.csv
- results/leader_momentum_topk_sweep.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from core.strategy import STRATEGIES_DIR, Strategy
from modules.backtest.composite import run_strategy_backtest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_CSV = RESULTS_DIR / "leader_momentum_topk_sweep.csv"
SUMMARY_MD = RESULTS_DIR / "leader_momentum_topk_sweep.md"
STRATEGY_DIR = STRATEGIES_DIR / "research" / "leader_momentum_topk"

FACTOR_SPECS = [
    {
        "name": "bull_power",
        "expression": "$high - EMA($close, 13)",
        "direction": "正向",
        "negate": False,
    },
    {
        "name": "bear_power",
        "expression": "$low - EMA($close, 13)",
        "direction": "正向",
        "negate": False,
    },
    {
        "name": "close_to_high_20d",
        "expression": "$close / Max($close, 20) - 1",
        "direction": "正向",
        "negate": False,
    },
    {
        "name": "mom_20d",
        "expression": "$close / Ref($close, 20) - 1",
        "direction": "正向",
        "negate": False,
    },
]

TOPKS = [5, 10, 15, 20, 30, 50]

COMMON_SELECTION = {
    "mode": "factor_topk",
    "universe": "all",
    "neutralize_industry": True,
    "min_market_cap": 50,
    "exclude_st": True,
    "exclude_new_days": 120,
    "sticky": 5,
    "buffer": 20,
    "score_smoothing_days": 5,
    "entry_persist_days": 3,
    "exit_persist_days": 3,
    "min_hold_days": 10,
    "hard_filter_quantiles": {
        "total_mv": 0.80,
    },
}

COMMON_STABILITY = {
    "churn_limit": 2,
    "margin_stable": True,
}

COMMON_POSITION = {
    "model": "fixed",
    "stock_pct": 0.88,
}

COMMON_TRADING = {
    "buy_commission_rate": 0.0003,
    "sell_commission_rate": 0.0003,
    "sell_stamp_tax_rate": 0.0010,
    "slippage_bps": 5,
    "impact_bps": 5,
    "block_limit_up_buy": True,
    "block_limit_down_sell": True,
}


def _slugify(text: str) -> str:
    buf = []
    for ch in text.lower():
        if ch.isalnum():
            buf.append(ch)
        else:
            buf.append("_")
    slug = "".join(buf).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "factor"


def _entry_rank(topk: int) -> int:
    return max(1, int(round(topk * 0.7)))


def _exit_rank(topk: int) -> int:
    return topk + 15


def _strategy_key(factor_name: str, topk: int) -> str:
    stem = f"day_leader_top1000_{_slugify(factor_name)}_pos_k{topk}"
    return f"research/leader_momentum_topk/{stem}"


def _strategy_yaml_path(strategy_key: str) -> Path:
    return STRATEGIES_DIR / f"{strategy_key}.yaml"


def _build_cfg(spec: dict[str, Any], topk: int, strategy_key: str) -> dict[str, Any]:
    selection = dict(COMMON_SELECTION)
    selection["topk"] = int(topk)
    selection["entry_rank"] = _entry_rank(topk)
    selection["exit_rank"] = _exit_rank(topk)

    return {
        "name": strategy_key.split("/")[-1],
        "description": (
            f"龙头追涨持仓扫描：top1000市值近似龙头，{spec['name']} {spec['direction']}，"
            f"topk={topk}"
        ),
        "weights": {"alpha": 1.0, "risk": 0.0, "enhance": 0.0},
        "factors": {
            "alpha": [
                {
                    "name": spec["name"],
                    "expression": spec["expression"],
                    "source": "qlib",
                    **({"negate": True} if spec.get("negate") else {}),
                }
            ]
        },
        "selection": selection,
        "stability": COMMON_STABILITY,
        "rebalance": {"freq": "day"},
        "position": COMMON_POSITION,
        "trading": COMMON_TRADING,
    }


def _ensure_strategy_yaml(spec: dict[str, Any], topk: int) -> str:
    strategy_key = _strategy_key(spec["name"], topk)
    path = _strategy_yaml_path(strategy_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = _build_cfg(spec, topk, strategy_key)
    path.write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return strategy_key


def _selection_stats(selection_path: Path) -> tuple[int, int, float]:
    df = pd.read_csv(selection_path, usecols=["date", "symbol"])
    if df.empty:
        return 0, 0, 0.0
    n_dates = int(df["date"].nunique())
    n_rows = int(len(df))
    avg_holdings = float(n_rows / n_dates) if n_dates else 0.0
    return n_dates, n_rows, avg_holdings


def _run_one(spec: dict[str, Any], topk: int) -> dict[str, Any]:
    strategy_key = _ensure_strategy_yaml(spec, topk)
    strategy = Strategy.load(strategy_key)
    strategy.generate_selections(force=False)
    result = run_strategy_backtest(strategy=strategy, engine="qlib")
    n_dates, n_rows, avg_holdings = _selection_stats(strategy.selections_path())
    return {
        "strategy_name": strategy.name,
        "factor_name": spec["name"],
        "direction": spec["direction"],
        "leader_definition": "total_mv_top20pct",
        "topk": int(topk),
        "entry_rank": _entry_rank(topk),
        "exit_rank": _exit_rank(topk),
        "annual_return": result.annual_return,
        "sharpe_ratio": result.sharpe_ratio,
        "max_drawdown": result.max_drawdown,
        "total_return": result.total_return,
        "n_rebalance_dates": n_dates,
        "selection_rows": n_rows,
        "avg_holdings": avg_holdings,
        "selection_file": str(strategy.selections_path().relative_to(PROJECT_ROOT)),
        "results_file": result.metadata.get("results_file", ""),
    }


def _write_summary(df: pd.DataFrame) -> None:
    lines: list[str] = []
    lines.append("# 龙头追涨持仓数量扫描")
    lines.append("")
    lines.append("- 龙头池定义: `total_mv` 前 20%（前 1000 市值近似）")
    lines.append("- 因子: `bull_power`、`bear_power`、`close_to_high_20d`、`mom_20d`")
    lines.append("- 只变参数: `topk`")
    lines.append("- 派生参数: `entry_rank = round(topk * 0.7)`，`exit_rank = topk + 15`")
    lines.append("- 固定仓位: `stock_pct = 0.88`")
    lines.append("")

    if df.empty:
        lines.append("无结果。")
        SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")
        return

    for factor_name, sub in df.groupby("factor_name", sort=False):
        lines.append(f"## {factor_name}")
        lines.append("")
        show = sub.sort_values("topk").copy()
        fmt = show[
            ["topk", "entry_rank", "exit_rank", "annual_return", "sharpe_ratio", "max_drawdown"]
        ].copy()
        for col in ("annual_return", "max_drawdown"):
            fmt[col] = fmt[col].map(lambda x: f"{x:.2%}")
        fmt["sharpe_ratio"] = fmt["sharpe_ratio"].map(lambda x: f"{x:.4f}")
        try:
            lines.append(fmt.to_markdown(index=False))
        except Exception:
            lines.append("```")
            lines.append(fmt.to_string(index=False))
            lines.append("```")
        lines.append("")

    best = df.sort_values(["annual_return", "sharpe_ratio"], ascending=[False, False]).head(12).copy()
    lines.append("## Overall Top 12 By Annual Return")
    lines.append("")
    best_fmt = best[["factor_name", "topk", "annual_return", "sharpe_ratio", "max_drawdown"]].copy()
    for col in ("annual_return", "max_drawdown"):
        best_fmt[col] = best_fmt[col].map(lambda x: f"{x:.2%}")
    best_fmt["sharpe_ratio"] = best_fmt["sharpe_ratio"].map(lambda x: f"{x:.4f}")
    try:
        lines.append(best_fmt.to_markdown(index=False))
    except Exception:
        lines.append("```")
        lines.append(best_fmt.to_string(index=False))
        lines.append("```")
    lines.append("")

    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for spec in FACTOR_SPECS:
        for topk in TOPKS:
            rows.append(_run_one(spec, topk))

    df = pd.DataFrame(rows).sort_values(["factor_name", "topk"]).reset_index(drop=True)
    df.to_csv(RESULTS_CSV, index=False)
    _write_summary(df)
    print(f"[OK] saved csv: {RESULTS_CSV}")
    print(f"[OK] saved md : {SUMMARY_MD}")


if __name__ == "__main__":
    main()
