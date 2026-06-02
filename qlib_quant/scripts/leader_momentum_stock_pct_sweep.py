#!/usr/bin/env python3
"""
龙头追涨因子仓位扫描（Qlib）

统一口径：
- 龙头池：total_mv 前 20%（≈ 前 1000 市值）
- 日频正式单因子执行壳：topk=10 + buffer=20 + sticky=5 + persist + churn_limit=2
- 只变 stock_pct，其余参数固定

输出：
- results/leader_momentum_stock_pct_sweep.csv
- results/leader_momentum_stock_pct_sweep.md
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
STRATEGY_DIR = STRATEGIES_DIR / "research" / "leader_momentum_stock_pct"

RESULTS_CSV = RESULTS_DIR / "leader_momentum_stock_pct_sweep.csv"
SUMMARY_MD = RESULTS_DIR / "leader_momentum_stock_pct_sweep.md"
FORMAL_DAY_RESULTS = RESULTS_DIR / "formal_single_factor_results_day.csv"

SELECTION_TEMPLATE = {
    "mode": "factor_topk",
    "topk": 10,
    "universe": "all",
    "neutralize_industry": True,
    "min_market_cap": 50,
    "exclude_st": True,
    "exclude_new_days": 120,
    "sticky": 5,
    "buffer": 20,
    "score_smoothing_days": 5,
    "entry_rank": 7,
    "exit_rank": 25,
    "entry_persist_days": 3,
    "exit_persist_days": 3,
    "min_hold_days": 10,
    "hard_filter_quantiles": {
        "total_mv": 0.80,
    },
}

STABILITY = {
    "churn_limit": 2,
    "margin_stable": True,
}

TRADING = {
    "buy_commission_rate": 0.0003,
    "sell_commission_rate": 0.0003,
    "sell_stamp_tax_rate": 0.0010,
    "slippage_bps": 5,
    "impact_bps": 5,
    "block_limit_up_buy": True,
    "block_limit_down_sell": True,
}

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

STOCK_PCTS = [0.30, 0.50, 0.70, 0.88, 1.00]


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


def _strategy_key(factor_name: str, stock_pct: float) -> str:
    pct = int(round(stock_pct * 100))
    stem = f"day_leader_top1000_{_slugify(factor_name)}_pos_fixed{pct:02d}_k10"
    return f"research/leader_momentum_stock_pct/{stem}"


def _strategy_yaml_path(strategy_key: str) -> Path:
    return STRATEGIES_DIR / f"{strategy_key}.yaml"


def _build_cfg(spec: dict[str, Any], stock_pct: float, strategy_key: str) -> dict[str, Any]:
    stem = strategy_key.split("/")[-1]
    return {
        "name": stem,
        "description": (
            f"龙头追涨仓位扫描：top1000市值近似龙头，{spec['name']} {spec['direction']}，"
            f"fixed {stock_pct:.0%}"
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
        "selection": SELECTION_TEMPLATE,
        "stability": STABILITY,
        "rebalance": {"freq": "day"},
        "position": {"model": "fixed", "stock_pct": stock_pct},
        "trading": TRADING,
    }


def _ensure_strategy_yaml(spec: dict[str, Any], stock_pct: float) -> str:
    strategy_key = _strategy_key(spec["name"], stock_pct)
    path = _strategy_yaml_path(strategy_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = _build_cfg(spec, stock_pct, strategy_key)
    path.write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return strategy_key


def _load_formal_baseline() -> pd.DataFrame:
    df = pd.read_csv(FORMAL_DAY_RESULTS)
    df = df[df["status"] == "success"].copy()
    df = df[df["direction"] == "正向"].copy()
    keep = [
        "factor_name",
        "annual_return",
        "sharpe_ratio",
        "max_drawdown",
        "results_file",
        "selection_file",
    ]
    df = df[keep].rename(
        columns={
            "annual_return": "all_market_annual_return",
            "sharpe_ratio": "all_market_sharpe_ratio",
            "max_drawdown": "all_market_max_drawdown",
            "results_file": "all_market_results_file",
            "selection_file": "all_market_selection_file",
        }
    )
    return df


def _run_one(spec: dict[str, Any], stock_pct: float) -> dict[str, Any]:
    strategy_key = _ensure_strategy_yaml(spec, stock_pct)
    strategy = Strategy.load(strategy_key)
    strategy.generate_selections(force=False)
    result = run_strategy_backtest(strategy=strategy, engine="qlib")
    return {
        "strategy_name": strategy.name,
        "factor_name": spec["name"],
        "direction": spec["direction"],
        "leader_definition": "total_mv_top20pct",
        "stock_pct": stock_pct,
        "annual_return": result.annual_return,
        "sharpe_ratio": result.sharpe_ratio,
        "max_drawdown": result.max_drawdown,
        "total_return": result.total_return,
        "selection_file": str(strategy.selections_path().relative_to(PROJECT_ROOT)),
        "results_file": result.metadata.get("results_file", ""),
    }


def _write_summary(df: pd.DataFrame) -> None:
    lines: list[str] = []
    lines.append("# 龙头追涨仓位扫描")
    lines.append("")
    lines.append("- 龙头池定义: `total_mv` 前 20%（前 1000 市值近似）")
    lines.append("- 因子: `bull_power`、`bear_power`、`close_to_high_20d`、`mom_20d`")
    lines.append("- 只变参数: `position.stock_pct`")
    lines.append("- 其余执行壳固定为正式日频单因子模板")
    lines.append("")

    if df.empty:
        lines.append("无结果。")
        SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")
        return

    for factor_name, sub in df.groupby("factor_name", sort=False):
        lines.append(f"## {factor_name}")
        lines.append("")
        show = sub.sort_values("stock_pct").copy()
        cols = [
            "stock_pct",
            "annual_return",
            "sharpe_ratio",
            "max_drawdown",
            "all_market_annual_return",
            "all_market_sharpe_ratio",
            "all_market_max_drawdown",
        ]
        fmt = show[cols].copy()
        for col in (
            "stock_pct",
            "annual_return",
            "max_drawdown",
            "all_market_annual_return",
            "all_market_max_drawdown",
        ):
            fmt[col] = fmt[col].map(lambda x: f"{x:.2%}")
        for col in ("sharpe_ratio", "all_market_sharpe_ratio"):
            fmt[col] = fmt[col].map(lambda x: f"{x:.4f}")
        try:
            lines.append(fmt.to_markdown(index=False))
        except Exception:
            lines.append("```")
            lines.append(fmt.to_string(index=False))
            lines.append("```")
        lines.append("")

    best = df.sort_values(["annual_return", "sharpe_ratio"], ascending=[False, False]).head(10).copy()
    lines.append("## Overall Top 10 By Annual Return")
    lines.append("")
    best_fmt = best[["factor_name", "stock_pct", "annual_return", "sharpe_ratio", "max_drawdown"]].copy()
    best_fmt["stock_pct"] = best_fmt["stock_pct"].map(lambda x: f"{x:.2%}")
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
    baseline = _load_formal_baseline()

    rows: list[dict[str, Any]] = []
    for spec in FACTOR_SPECS:
        for stock_pct in STOCK_PCTS:
            row = _run_one(spec, stock_pct)
            rows.append(row)

    df = pd.DataFrame(rows)
    df = df.merge(baseline, how="left", left_on="factor_name", right_on="factor_name")
    df = df.sort_values(["factor_name", "stock_pct"]).reset_index(drop=True)
    df.to_csv(RESULTS_CSV, index=False)
    _write_summary(df)
    print(f"[OK] saved csv: {RESULTS_CSV}")
    print(f"[OK] saved md : {SUMMARY_MD}")


if __name__ == "__main__":
    main()
