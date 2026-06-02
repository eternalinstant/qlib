#!/usr/bin/env python3
"""
比较已有日频好因子在增加迟滞区间前后的表现。

口径：
- 基准集合来自 results/formal_single_factor_results_day.csv
- 只选 `annual_return > 0` 且 `sharpe_ratio > 0.3` 的正式日频单因子
- lag 版本直接使用 formal_single_factor_results_day.csv 中既有正式结果
- raw 版本新建固定仓位、无迟滞的单因子策略并回测
- 只比较迟滞层，不引入 position.model=trend 的防守层

输出：
- results/lag_effect_good_factors_20260331.csv
- results/lag_effect_good_factors_20260331.md
"""

from __future__ import annotations

import math
from pathlib import Path
from statistics import mean

import pandas as pd
import yaml

from core.strategy import STRATEGIES_DIR, Strategy
from modules.backtest.composite import run_strategy_backtest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

FORMAL_DAY_RESULTS = RESULTS_DIR / "formal_single_factor_results_day.csv"
OUT_CSV = RESULTS_DIR / "lag_effect_good_factors_20260331.csv"
OUT_MD = RESULTS_DIR / "lag_effect_good_factors_20260331.md"

RAW_SELECTION = {
    "mode": "factor_topk",
    "topk": 10,
    "universe": "all",
    "neutralize_industry": True,
    "min_market_cap": 50,
    "exclude_st": True,
    "exclude_new_days": 120,
    "sticky": 0,
    "buffer": 0,
    "score_smoothing_days": 1,
    "entry_persist_days": 1,
    "exit_persist_days": 1,
    "min_hold_days": 0,
}

RAW_STABILITY = {
    "churn_limit": 0,
    "margin_stable": False,
}

FIXED_POSITION = {
    "model": "fixed",
    "stock_pct": 0.88,
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


def _strategy_rel_key(row: pd.Series) -> str:
    suffix = "neg" if str(row["direction"]) == "反向" else "pos"
    return (
        f"research/lag_effect/day/"
        f"day_{_slugify(str(row['factor_name']))}_{suffix}_raw_{str(row['expr_hash'])}"
    )


def _strategy_yaml_path(row: pd.Series) -> Path:
    return STRATEGIES_DIR / f"{_strategy_rel_key(row)}.yaml"


def _ensure_raw_strategy(row: pd.Series) -> Path:
    yaml_path = _strategy_yaml_path(row)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = {
        "name": _strategy_rel_key(row).split("/")[-1],
        "description": f"lag effect raw day: {row['factor_name']} {row['direction']}",
        "weights": {"alpha": 1.0, "risk": 0.0, "enhance": 0.0},
        "factors": {
            "alpha": [
                {
                    "name": str(row["factor_name"]),
                    "expression": str(row["expression"]),
                    "source": str(row["source"]),
                    **({"negate": True} if str(row["direction"]) == "反向" else {}),
                }
            ]
        },
        "selection": RAW_SELECTION,
        "stability": RAW_STABILITY,
        "rebalance": {"freq": "day"},
        "position": FIXED_POSITION,
        "trading": TRADING,
    }
    yaml_path.write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return yaml_path


def _repo_relative(path: str | Path | None) -> str:
    if path is None:
        return ""
    text = str(path)
    if not text:
        return ""
    p = Path(text)
    if not p.is_absolute():
        return text
    try:
        return str(p.relative_to(PROJECT_ROOT))
    except ValueError:
        return text


def _calc_stability(selection_file: Path) -> tuple[float, float]:
    df = pd.read_csv(selection_file, usecols=["date", "symbol"])
    grouped = [set(g["symbol"]) for _, g in df.groupby("date", sort=True)]
    if len(grouped) < 2:
        return 0.0, 1.0
    jaccards = []
    retains = []
    for prev_set, curr_set in zip(grouped, grouped[1:]):
        inter = len(prev_set & curr_set)
        union = len(prev_set | curr_set)
        jaccards.append(inter / union if union else 1.0)
        retains.append(inter / len(prev_set) if prev_set else 1.0)
    mean_retain = mean(retains)
    return float(mean(jaccards)), float(1.0 - mean_retain)


def _run_raw(row: pd.Series) -> dict[str, object]:
    yaml_path = _ensure_raw_strategy(row)
    strategy_name = _strategy_rel_key(row)
    strategy = Strategy.load(strategy_name)
    strategy.generate_selections(force=True)
    result = run_strategy_backtest(strategy=strategy, engine="qlib")
    selection_rel = _repo_relative(strategy.selections_path())
    result_rel = _repo_relative(result.metadata.get("results_file", ""))
    mean_jaccard, mean_replace_ratio = _calc_stability(PROJECT_ROOT / selection_rel)
    return {
        "raw_strategy_name": strategy_name,
        "raw_strategy_yaml": _repo_relative(yaml_path),
        "raw_selection_file": selection_rel,
        "raw_results_file": result_rel,
        "raw_annual_return": float(result.annual_return),
        "raw_sharpe_ratio": float(result.sharpe_ratio),
        "raw_max_drawdown": float(result.max_drawdown),
        "raw_total_return": float(result.total_return),
        "raw_mean_jaccard": mean_jaccard,
        "raw_mean_replace_ratio": mean_replace_ratio,
    }


def main() -> None:
    day_df = pd.read_csv(FORMAL_DAY_RESULTS)
    candidates = day_df[
        (day_df["status"] == "success")
        & (day_df["annual_return"] > 0)
        & (day_df["sharpe_ratio"] > 0.3)
    ].copy()
    candidates = candidates.sort_values(
        ["sharpe_ratio", "annual_return"],
        ascending=[False, False],
    ).reset_index(drop=True)

    rows = []
    for item in candidates.itertuples(index=False):
        row = pd.Series(item._asdict())
        raw = _run_raw(row)
        lag_selection_file = PROJECT_ROOT / str(row["selection_file"])
        lag_mean_jaccard, lag_mean_replace_ratio = _calc_stability(lag_selection_file)
        rows.append(
            {
                "factor_name": row["factor_name"],
                "direction": row["direction"],
                "source": row["source"],
                "expression": row["expression"],
                "lag_strategy_name": row["strategy_name"],
                "lag_strategy_yaml": row["strategy_yaml"],
                "lag_selection_file": row["selection_file"],
                "lag_results_file": row["results_file"],
                "lag_annual_return": float(row["annual_return"]),
                "lag_sharpe_ratio": float(row["sharpe_ratio"]),
                "lag_max_drawdown": float(row["max_drawdown"]),
                "lag_total_return": float(row["total_return"]),
                "lag_mean_jaccard": lag_mean_jaccard,
                "lag_mean_replace_ratio": lag_mean_replace_ratio,
                **raw,
            }
        )

    out = pd.DataFrame(rows)
    out["delta_annual_return"] = out["lag_annual_return"] - out["raw_annual_return"]
    out["delta_sharpe_ratio"] = out["lag_sharpe_ratio"] - out["raw_sharpe_ratio"]
    out["delta_max_drawdown"] = out["lag_max_drawdown"] - out["raw_max_drawdown"]
    out["delta_mean_jaccard"] = out["lag_mean_jaccard"] - out["raw_mean_jaccard"]
    out["delta_mean_replace_ratio"] = (
        out["lag_mean_replace_ratio"] - out["raw_mean_replace_ratio"]
    )
    out = out.sort_values(
        ["delta_sharpe_ratio", "delta_annual_return"],
        ascending=[False, False],
    ).reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False)

    show = out[
        [
            "factor_name",
            "direction",
            "raw_annual_return",
            "lag_annual_return",
            "delta_annual_return",
            "raw_sharpe_ratio",
            "lag_sharpe_ratio",
            "delta_sharpe_ratio",
            "raw_max_drawdown",
            "lag_max_drawdown",
            "raw_mean_jaccard",
            "lag_mean_jaccard",
            "raw_mean_replace_ratio",
            "lag_mean_replace_ratio",
        ]
    ].copy()

    for col in [
        "raw_annual_return",
        "lag_annual_return",
        "delta_annual_return",
        "raw_max_drawdown",
        "lag_max_drawdown",
        "raw_mean_jaccard",
        "lag_mean_jaccard",
        "raw_mean_replace_ratio",
        "lag_mean_replace_ratio",
    ]:
        show[col] = show[col].map(lambda v: f"{float(v):.2%}")

    for col in ["raw_sharpe_ratio", "lag_sharpe_ratio", "delta_sharpe_ratio"]:
        show[col] = show[col].map(lambda v: f"{float(v):.4f}")

    lines = [
        "# 好因子增加迟滞区间前后对比",
        "",
        "- 样本：`results/formal_single_factor_results_day.csv` 中 `annual_return > 0` 且 `sharpe_ratio > 0.3` 的日频正式单因子",
        "- 对照：`raw_fixed` vs `lag_fixed`",
        "- `lag` 使用既有正式结果；`raw` 为本次新补的无迟滞固定仓位回测",
        "",
        show.to_markdown(index=False),
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(OUT_CSV)
    print(OUT_MD)
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
