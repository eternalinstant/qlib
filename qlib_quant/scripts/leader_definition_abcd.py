#!/usr/bin/env python3
"""
龙头定义 A/B/C/D 对照实验（Qlib）

A: 当前基线，total_mv 前 20%
B: 自由流通市值前 20% + 流动性过滤（turnover_rate_f 去掉后 20%）
C: B + 行业内 circ_mv 前 3
D: C + 52 周价格位置确认（price_pos_52w）

追涨因子固定为：
- bull_power，topk=5
- bear_power，topk=15
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
RESULTS_CSV = RESULTS_DIR / "leader_definition_abcd.csv"
SUMMARY_MD = RESULTS_DIR / "leader_definition_abcd.md"

CASE_SPECS = [
    {
        "case_id": "A",
        "case_name": "size_only_total_mv",
        "description": "当前基线：total_mv 前20%",
        "selection": {
            "hard_filter_quantiles": {
                "total_mv": 0.80,
            }
        },
        "extra_factors": [],
    },
    {
        "case_id": "B",
        "case_name": "floatcap_plus_liquidity",
        "description": "自由流通市值前20% + 流动性去掉后20%",
        "selection": {
            "hard_filter_quantiles": {
                "circ_mv": 0.80,
                "turnover_rate_f": 0.20,
            }
        },
        "extra_factors": [],
    },
    {
        "case_id": "C",
        "case_name": "floatcap_liquidity_industry_top3",
        "description": "B + 行业内 circ_mv 前3",
        "selection": {
            "hard_filter_quantiles": {
                "circ_mv": 0.80,
                "turnover_rate_f": 0.20,
            },
            "industry_leader_field": "circ_mv",
            "industry_leader_top_n": 3,
        },
        "extra_factors": [],
    },
    {
        "case_id": "D",
        "case_name": "industry_top3_with_52w_confirmation",
        "description": "C + 52周价格位置确认",
        "selection": {
            "hard_filter_quantiles": {
                "circ_mv": 0.80,
                "turnover_rate_f": 0.20,
            },
            "industry_leader_field": "circ_mv",
            "industry_leader_top_n": 3,
        },
        "extra_factors": [
            {
                "name": "price_pos_52w",
                "expression": "($close - Min($close, 252)) / (Max($close, 252) - Min($close, 252) + 1e-8)",
                "source": "qlib",
            }
        ],
    },
]

BASE_FACTORS = [
    {
        "name": "bull_power",
        "expression": "$high - EMA($close, 13)",
        "source": "qlib",
        "direction": "正向",
        "topk": 5,
        "entry_rank": 4,
        "exit_rank": 20,
    },
    {
        "name": "bear_power",
        "expression": "$low - EMA($close, 13)",
        "source": "qlib",
        "direction": "正向",
        "topk": 15,
        "entry_rank": 10,
        "exit_rank": 30,
    },
]

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


def _strategy_key(base_name: str, case_id: str) -> str:
    stem = f"day_leader_{_slugify(base_name)}_{case_id.lower()}"
    return f"research/leader_definition_abcd/{stem}"


def _strategy_yaml_path(strategy_key: str) -> Path:
    return STRATEGIES_DIR / f"{strategy_key}.yaml"


def _build_cfg(base: dict[str, Any], case: dict[str, Any], strategy_key: str) -> dict[str, Any]:
    selection = dict(COMMON_SELECTION)
    selection["topk"] = base["topk"]
    selection["entry_rank"] = base["entry_rank"]
    selection["exit_rank"] = base["exit_rank"]
    selection.update(case["selection"])

    factor_list = [
        {
            "name": base["name"],
            "expression": base["expression"],
            "source": base["source"],
        }
    ] + list(case.get("extra_factors", []))

    return {
        "name": strategy_key.split("/")[-1],
        "description": f"{base['name']} + 龙头定义{case['case_id']}：{case['description']}",
        "weights": {"alpha": 1.0, "risk": 0.0, "enhance": 0.0},
        "factors": {"alpha": factor_list},
        "selection": selection,
        "stability": COMMON_STABILITY,
        "rebalance": {"freq": "day"},
        "position": COMMON_POSITION,
        "trading": COMMON_TRADING,
    }


def _ensure_strategy_yaml(base: dict[str, Any], case: dict[str, Any]) -> str:
    strategy_key = _strategy_key(base["name"], case["case_id"])
    path = _strategy_yaml_path(strategy_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = _build_cfg(base, case, strategy_key)
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return strategy_key


def _selection_stats(selection_path: Path) -> tuple[int, int, float]:
    df = pd.read_csv(selection_path, usecols=["date", "symbol"])
    if df.empty:
        return 0, 0, 0.0
    n_dates = int(df["date"].nunique())
    n_rows = int(len(df))
    avg_holdings = float(n_rows / n_dates) if n_dates else 0.0
    return n_dates, n_rows, avg_holdings


def _run_one(base: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    strategy_key = _ensure_strategy_yaml(base, case)
    strategy = Strategy.load(strategy_key)
    strategy.generate_selections(force=False)
    result = run_strategy_backtest(strategy=strategy, engine="qlib")
    n_dates, n_rows, avg_holdings = _selection_stats(strategy.selections_path())
    return {
        "strategy_name": strategy.name,
        "factor_name": base["name"],
        "case_id": case["case_id"],
        "case_name": case["case_name"],
        "case_description": case["description"],
        "topk": base["topk"],
        "entry_rank": base["entry_rank"],
        "exit_rank": base["exit_rank"],
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
    lines.append("# 龙头定义 A/B/C/D 对照")
    lines.append("")
    lines.append("- A: `total_mv` 前 20%")
    lines.append("- B: `circ_mv` 前 20% + `turnover_rate_f` 去掉后 20%")
    lines.append("- C: B + 行业内 `circ_mv` 前 3")
    lines.append("- D: C + `price_pos_52w` 确认")
    lines.append("- `bull_power` 固定 `topk=5`")
    lines.append("- `bear_power` 固定 `topk=15`")
    lines.append("")

    for factor_name, sub in df.groupby("factor_name", sort=False):
        lines.append(f"## {factor_name}")
        lines.append("")
        fmt = sub.sort_values("case_id")[
            ["case_id", "case_description", "annual_return", "sharpe_ratio", "max_drawdown"]
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

    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for base in BASE_FACTORS:
        for case in CASE_SPECS:
            rows.append(_run_one(base, case))
    df = pd.DataFrame(rows).sort_values(["factor_name", "case_id"]).reset_index(drop=True)
    df.to_csv(RESULTS_CSV, index=False)
    _write_summary(df)
    print(f"[OK] saved csv: {RESULTS_CSV}")
    print(f"[OK] saved md : {SUMMARY_MD}")


if __name__ == "__main__":
    main()
