#!/usr/bin/env python3
"""
全量正式单因子回测

统一口径：
- 日频：event-gate k10
- 周频：week + buffered k15
- 每个因子同时跑正向/反向两个版本
- 使用正式策略链路：Strategy.load -> generate_selections -> run_strategy_backtest

输出：
- results/formal_single_factor_catalog.csv
- results/formal_single_factor_results.csv
- results/formal_single_factor_results_day.csv
- results/formal_single_factor_results_week.csv
- results/formal_single_factor_failures.csv
- results/formal_single_factor_summary.md
"""

from __future__ import annotations

import argparse
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import yaml

from core.factors import default_registry
from core.strategy import STRATEGIES_DIR, Strategy
from modules.backtest.composite import run_strategy_backtest
from scripts.factor_scan import get_all_factors, get_weekly_factors

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
STRATEGY_BATCH_DIR = STRATEGIES_DIR / "research" / "formal_single_factor_batch"

CATALOG_CSV = RESULTS_DIR / "formal_single_factor_catalog.csv"
RESULTS_CSV = RESULTS_DIR / "formal_single_factor_results.csv"
RESULTS_DAY_CSV = RESULTS_DIR / "formal_single_factor_results_day.csv"
RESULTS_WEEK_CSV = RESULTS_DIR / "formal_single_factor_results_week.csv"
FAILURES_CSV = RESULTS_DIR / "formal_single_factor_failures.csv"
SUMMARY_MD = RESULTS_DIR / "formal_single_factor_summary.md"

DAY_SELECTION_TEMPLATE = {
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
}

WEEK_SELECTION_TEMPLATE = {
    "mode": "factor_topk",
    "topk": 15,
    "universe": "all",
    "neutralize_industry": True,
    "min_market_cap": 50,
    "exclude_st": True,
    "exclude_new_days": 120,
    "sticky": 5,
    "buffer": 20,
    "score_smoothing_days": 1,
    "entry_persist_days": 1,
    "exit_persist_days": 1,
    "min_hold_days": 0,
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


@dataclass(frozen=True)
class FactorSpec:
    freq: str
    name: str
    expression: str
    source: str
    window_scale: int
    origins: tuple[str, ...]

    @property
    def expr_hash(self) -> str:
        payload = f"{self.freq}|{self.name}|{self.expression}|{self.source}|{self.window_scale}"
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


def _add_spec(
    target: Dict[tuple, Dict[str, object]],
    *,
    freq: str,
    name: str,
    expression: str,
    source: str,
    window_scale: int,
    origin: str,
) -> None:
    key = (freq, name, expression, source, int(window_scale))
    entry = target.setdefault(
        key,
        {
            "freq": freq,
            "name": name,
            "expression": expression,
            "source": source,
            "window_scale": int(window_scale),
            "origins": set(),
        },
    )
    entry["origins"].add(origin)


def collect_factor_specs() -> List[FactorSpec]:
    specs: Dict[tuple, Dict[str, object]] = {}

    for factor in default_registry.all().values():
        _add_spec(
            specs,
            freq="day",
            name=factor.name,
            expression=factor.expression,
            source=factor.source,
            window_scale=1,
            origin="default_registry",
        )

    for name, expr in get_all_factors().items():
        _add_spec(
            specs,
            freq="day",
            name=name,
            expression=expr,
            source="qlib",
            window_scale=1,
            origin="scan_daily",
        )

    for name, expr in get_weekly_factors().items():
        _add_spec(
            specs,
            freq="week",
            name=name,
            expression=expr,
            source="qlib",
            window_scale=1,
            origin="scan_weekly",
        )

    for path in STRATEGIES_DIR.rglob("*.yaml"):
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        factors = cfg.get("factors", {}) or {}
        rebalance = cfg.get("rebalance", {}) or {}
        freq = str(rebalance.get("freq", "day"))
        window_scale = int(cfg.get("factor_window_scale", 1) or 1)
        target_freq = "week" if freq == "week" or window_scale > 1 else "day"
        effective_scale = window_scale if target_freq == "week" else 1

        for factor_list in factors.values():
            if not isinstance(factor_list, list):
                continue
            for item in factor_list:
                if not isinstance(item, dict) or "name" not in item:
                    continue
                name = str(item["name"])
                if "expression" in item:
                    expression = str(item["expression"])
                    source = str(item.get("source", "qlib"))
                else:
                    base = default_registry.get(name)
                    if base is None:
                        continue
                    expression = str(base.expression)
                    source = str(base.source)
                _add_spec(
                    specs,
                    freq=target_freq,
                    name=name,
                    expression=expression,
                    source=source,
                    window_scale=effective_scale,
                    origin=str(path.relative_to(STRATEGIES_DIR)),
                )

    rows = []
    for item in specs.values():
        rows.append(
            FactorSpec(
                freq=str(item["freq"]),
                name=str(item["name"]),
                expression=str(item["expression"]),
                source=str(item["source"]),
                window_scale=int(item["window_scale"]),
                origins=tuple(sorted(item["origins"])),
            )
        )

    return sorted(
        rows,
        key=lambda x: (x.freq, x.source, x.name.lower(), x.expression.lower(), x.window_scale),
    )


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


def _direction_suffix(negate: bool) -> str:
    return "neg" if negate else "pos"


def strategy_rel_key(spec: FactorSpec, negate: bool) -> str:
    stem = (
        f"{spec.freq}_"
        f"{_slugify(spec.name)}_"
        f"{_direction_suffix(negate)}_"
        f"{spec.expr_hash}"
    )
    return f"research/formal_single_factor_batch/{spec.freq}/{stem}"


def strategy_yaml_path(spec: FactorSpec, negate: bool) -> Path:
    return STRATEGIES_DIR / f"{strategy_rel_key(spec, negate)}.yaml"


def build_strategy_config(spec: FactorSpec, negate: bool) -> dict:
    is_week = spec.freq == "week"
    cfg = {
        "name": strategy_rel_key(spec, negate).split("/")[-1],
        "description": (
            f"auto formal single-factor {spec.freq}: {spec.name} "
            f"{'反向' if negate else '正向'}"
        ),
        "weights": {"alpha": 1.0, "risk": 0.0, "enhance": 0.0},
        "factors": {
            "alpha": [
                {
                    "name": spec.name,
                    "expression": spec.expression,
                    "source": spec.source,
                    **({"negate": True} if negate else {}),
                }
            ]
        },
        "selection": WEEK_SELECTION_TEMPLATE if is_week else DAY_SELECTION_TEMPLATE,
        "stability": COMMON_STABILITY,
        "rebalance": {"freq": spec.freq},
        "position": COMMON_POSITION,
        "trading": COMMON_TRADING,
    }
    if is_week and spec.window_scale > 1:
        cfg["factor_window_scale"] = spec.window_scale
    return cfg


def ensure_strategy_yaml(spec: FactorSpec, negate: bool) -> Path:
    path = strategy_yaml_path(spec, negate)
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = build_strategy_config(spec, negate)
    path.write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


def catalog_to_frame(specs: Iterable[FactorSpec]) -> pd.DataFrame:
    rows = []
    for spec in specs:
        rows.append(
            {
                "freq": spec.freq,
                "name": spec.name,
                "expression": spec.expression,
                "source": spec.source,
                "window_scale": spec.window_scale,
                "expr_hash": spec.expr_hash,
                "origin_count": len(spec.origins),
                "origins": " | ".join(spec.origins),
            }
        )
    return pd.DataFrame(rows)


def load_existing_results() -> pd.DataFrame:
    if not RESULTS_CSV.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(RESULTS_CSV)
    except Exception:
        return pd.DataFrame()


def save_results(df: pd.DataFrame) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if df.empty:
        df.to_csv(RESULTS_CSV, index=False)
        pd.DataFrame().to_csv(RESULTS_DAY_CSV, index=False)
        pd.DataFrame().to_csv(RESULTS_WEEK_CSV, index=False)
        pd.DataFrame().to_csv(FAILURES_CSV, index=False)
        return

    ordered = df.copy()
    dedup_cols = ["freq", "factor_name", "negate", "expr_hash"]
    existing_cols = [col for col in dedup_cols if col in ordered.columns]
    if existing_cols:
        ordered = ordered.drop_duplicates(subset=existing_cols, keep="last")

    ordered = ordered.sort_values(
        ["freq", "status", "sharpe_ratio", "annual_return"],
        ascending=[True, True, False, False],
        na_position="last",
    ).reset_index(drop=True)
    ordered.to_csv(RESULTS_CSV, index=False)
    ordered[ordered["freq"] == "day"].to_csv(RESULTS_DAY_CSV, index=False)
    ordered[ordered["freq"] == "week"].to_csv(RESULTS_WEEK_CSV, index=False)
    ordered[ordered["status"] != "success"].to_csv(FAILURES_CSV, index=False)


def write_summary(catalog: pd.DataFrame, results: pd.DataFrame) -> None:
    dedup_cols = ["freq", "factor_name", "negate", "expr_hash"]
    existing_cols = [col for col in dedup_cols if col in results.columns]
    if existing_cols:
        results = results.drop_duplicates(subset=existing_cols, keep="last").copy()

    lines: List[str] = []
    lines.append("# 全量正式单因子回测")
    lines.append("")
    lines.append("## Catalog")
    lines.append("")
    lines.append(f"- 因子定义数: {len(catalog)}")
    if not catalog.empty:
        day_count = int((catalog["freq"] == "day").sum())
        week_count = int((catalog["freq"] == "week").sum())
        lines.append(f"- 日频定义数: {day_count}")
        lines.append(f"- 周频定义数: {week_count}")
        lines.append(f"- 计划回测数(双方向): {len(catalog) * 2}")
    lines.append("")

    if results.empty:
        lines.append("## Status")
        lines.append("")
        lines.append("- 暂无回测结果")
        SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")
        return

    lines.append("## Status")
    lines.append("")
    success = results[results["status"] == "success"]
    failed = results[results["status"] != "success"]
    lines.append(f"- 已完成: {len(results)}")
    lines.append(f"- 成功: {len(success)}")
    lines.append(f"- 失败: {len(failed)}")
    lines.append("")

    for freq in ("day", "week"):
        sub = success[success["freq"] == freq].copy()
        if sub.empty:
            continue
        lines.append(f"## {freq.title()} Top 20 By Sharpe")
        lines.append("")
        show = sub.sort_values(
            ["sharpe_ratio", "annual_return"],
            ascending=[False, False],
        ).head(20)
        cols = [
            "factor_name",
            "direction",
            "annual_return",
            "sharpe_ratio",
            "max_drawdown",
            "strategy_name",
        ]
        fmt = show[cols].copy()
        for col in ("annual_return", "max_drawdown"):
            fmt[col] = fmt[col].map(lambda x: f"{x:.4%}")
        fmt["sharpe_ratio"] = fmt["sharpe_ratio"].map(lambda x: f"{x:.4f}")
        try:
            lines.append(fmt.to_markdown(index=False))
        except Exception:
            lines.append("```")
            lines.append(fmt.to_string(index=False))
            lines.append("```")
        lines.append("")

    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def _repo_relative_text(value: str | Path) -> str:
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


def run_one(spec: FactorSpec, negate: bool, force_selections: bool) -> dict:
    yaml_path = ensure_strategy_yaml(spec, negate)
    strategy_name = strategy_rel_key(spec, negate)

    started_at = time.time()
    try:
        strategy = Strategy.load(strategy_name)
        strategy.generate_selections(force=force_selections)
        result = run_strategy_backtest(strategy=strategy, engine="qlib")
        elapsed = time.time() - started_at
        daily_returns = result.daily_returns
        return {
            "status": "success",
            "freq": spec.freq,
            "factor_name": spec.name,
            "direction": "反向" if negate else "正向",
            "negate": negate,
            "source": spec.source,
            "expression": spec.expression,
            "expr_hash": spec.expr_hash,
            "window_scale": spec.window_scale,
            "strategy_name": strategy_name,
            "strategy_yaml": _repo_relative_text(yaml_path),
            "selection_file": _repo_relative_text(strategy.selections_path()),
            "results_file": _repo_relative_text(result.metadata.get("results_file", "")),
            "annual_return": result.annual_return,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
            "total_return": result.total_return,
            "win_rate": float((daily_returns > 0).mean()) if not daily_returns.empty else 0.0,
            "n_days": int(len(daily_returns)),
            "start_date": str(daily_returns.index.min().date()) if not daily_returns.empty else "",
            "end_date": str(daily_returns.index.max().date()) if not daily_returns.empty else "",
            "elapsed_seconds": elapsed,
            "origin_count": len(spec.origins),
            "origins": " | ".join(spec.origins),
            "error_type": "",
            "error": "",
        }
    except Exception as exc:
        elapsed = time.time() - started_at
        return {
            "status": "failed",
            "freq": spec.freq,
            "factor_name": spec.name,
            "direction": "反向" if negate else "正向",
            "negate": negate,
            "source": spec.source,
            "expression": spec.expression,
            "expr_hash": spec.expr_hash,
            "window_scale": spec.window_scale,
            "strategy_name": strategy_name,
            "strategy_yaml": _repo_relative_text(yaml_path),
            "selection_file": "",
            "results_file": "",
            "annual_return": pd.NA,
            "sharpe_ratio": pd.NA,
            "max_drawdown": pd.NA,
            "total_return": pd.NA,
            "win_rate": pd.NA,
            "n_days": pd.NA,
            "start_date": "",
            "end_date": "",
            "elapsed_seconds": elapsed,
            "origin_count": len(spec.origins),
            "origins": " | ".join(spec.origins),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="全量正式单因子回测")
    parser.add_argument(
        "--freq",
        choices=["all", "day", "week"],
        default="all",
        help="只跑指定频率",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="只跑前 N 个定义（测试用，0=全量）",
    )
    parser.add_argument(
        "--force-selections",
        action="store_true",
        help="强制重算选股缓存",
    )
    parser.add_argument(
        "--rerun-failed",
        action="store_true",
        help="重跑历史失败项",
    )
    parser.add_argument(
        "--rerun-success",
        action="store_true",
        help="连历史成功项也重跑",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    specs = collect_factor_specs()
    catalog = catalog_to_frame(specs)
    catalog.to_csv(CATALOG_CSV, index=False)

    if args.freq != "all":
        specs = [spec for spec in specs if spec.freq == args.freq]
    if args.limit > 0:
        specs = specs[: args.limit]

    existing = load_existing_results()
    done_keys = set()
    if not existing.empty and not args.rerun_success:
        mask = existing["status"] == "success"
        for _, row in existing[mask].iterrows():
            done_keys.add(
                (
                    row.get("freq"),
                    row.get("factor_name"),
                    bool(row.get("negate")),
                    row.get("expr_hash"),
                )
            )
    if not existing.empty and not args.rerun_failed:
        mask = existing["status"] != "success"
        for _, row in existing[mask].iterrows():
            done_keys.add(
                (
                    row.get("freq"),
                    row.get("factor_name"),
                    bool(row.get("negate")),
                    row.get("expr_hash"),
                )
            )

    rows = existing.to_dict("records") if not existing.empty else []
    total_runs = len(specs) * 2
    current = 0

    for spec in specs:
        for negate in (False, True):
            current += 1
            run_key = (spec.freq, spec.name, negate, spec.expr_hash)
            if run_key in done_keys:
                print(
                    f"[SKIP] {current}/{total_runs} "
                    f"{spec.freq} {spec.name} {'反向' if negate else '正向'}"
                )
                continue

            print(
                f"[RUN ] {current}/{total_runs} "
                f"{spec.freq} {spec.name} {'反向' if negate else '正向'}"
            )
            row = run_one(spec, negate, force_selections=args.force_selections)
            rows.append(row)
            save_results(pd.DataFrame(rows))
            status = row["status"]
            if status == "success":
                print(
                    f"[ OK ] {spec.freq} {spec.name} {'反向' if negate else '正向'} "
                    f"ann={row['annual_return']:.2%} sharpe={row['sharpe_ratio']:.4f}"
                )
            else:
                print(
                    f"[FAIL] {spec.freq} {spec.name} {'反向' if negate else '正向'} "
                    f"{row['error_type']}: {row['error']}"
                )

    results = pd.DataFrame(rows)
    save_results(results)
    write_summary(catalog, pd.read_csv(RESULTS_CSV))


if __name__ == "__main__":
    main()
