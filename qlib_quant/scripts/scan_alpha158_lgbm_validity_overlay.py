#!/usr/bin/env python3
"""Scan validity overlays for Alpha158 LGBM formal backtest results.

This script starts from formal Qlib backtest CSVs, so the base returns already
include trading commissions, stamp tax, slippage, market impact, and tradability
constraints. It then applies the existing validity overlay logic to search for
risk rules that can satisfy strict annual-return and drawdown targets.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.validity import ValidityConfig


OUTPUT_COLUMNS = [
    "base_tag",
    "factor_count",
    "factor_names",
    "topk",
    "base_annual",
    "base_max_dd",
    "base_sharpe",
    "lookback_days",
    "min_observations",
    "min_total_return",
    "min_annual_return",
    "min_sharpe",
    "max_drawdown_rule",
    "action",
    "reduce_to",
    "annual_return",
    "max_drawdown",
    "sharpe_ratio",
    "total_return",
    "avg_exposure",
    "min_exposure",
    "active_days",
    "passed",
    "score",
    "results_file",
]


def parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def metrics(returns: pd.Series) -> dict[str, float]:
    rets = pd.Series(returns, copy=False).dropna().astype(float)
    if rets.empty:
        return {"annual": 0.0, "max_dd": 0.0, "sharpe": 0.0, "total": 0.0}
    nav = (1 + rets).cumprod()
    days = (nav.index[-1] - nav.index[0]).days
    terminal = float(nav.iloc[-1])
    annual = terminal ** (365 / days) - 1 if days > 0 and terminal > 0 else -1.0
    max_dd = float((nav / nav.cummax() - 1).min())
    std = float(rets.std())
    sharpe = float(rets.mean() / std * np.sqrt(252)) if std > 0 else 0.0
    return {
        "annual": float(annual),
        "max_dd": max_dd,
        "sharpe": sharpe,
        "total": float(terminal - 1),
    }


def fast_validity_overlay(
    daily_returns: pd.Series,
    config: ValidityConfig,
) -> tuple[pd.Series, pd.Series]:
    """Apply the same validity rule as core.validity, but without per-day pandas rebuilds."""
    rets = pd.Series(daily_returns, copy=False).dropna().astype(float)
    if rets.empty:
        return rets, pd.Series(dtype=float)

    raw = rets.to_numpy(dtype=float)
    adjusted = np.zeros(len(raw), dtype=float)
    exposure = np.ones(len(raw), dtype=float)
    min_obs = int(config.min_observations)
    lookback = int(config.lookback_days)

    for idx, ret in enumerate(raw):
        if idx < min_obs:
            factor = 1.0
        else:
            start = max(0, idx - lookback)
            window = adjusted[start:idx]
            obs = len(window)
            total_return = float(np.prod(1.0 + window) - 1.0)
            annual_return = (
                float((1.0 + total_return) ** (252.0 / obs) - 1.0)
                if obs > 0 and 1.0 + total_return > 0
                else -1.0
            )
            volatility = float(np.std(window, ddof=0))
            if volatility > 0:
                sharpe_ratio = float(np.mean(window) / volatility * np.sqrt(252))
            else:
                sharpe_ratio = float("inf") if float(np.mean(window)) > 0 else 0.0
            nav = np.cumprod(1.0 + window)
            max_drawdown = (
                float(np.min(nav / np.maximum.accumulate(nav) - 1.0))
                if len(nav) > 0
                else 0.0
            )

            breached = (
                obs < min_obs
                or total_return < config.min_total_return
                or annual_return < config.min_annual_return
                or sharpe_ratio < config.min_sharpe
                or max_drawdown < config.max_drawdown
            )
            if not breached:
                factor = 1.0
            elif config.action == "pause":
                factor = 0.0
            elif config.action == "reduce":
                factor = max(min(float(config.reduce_to), 1.0), 0.0)
            else:
                factor = 1.0

        adjusted[idx] = ret * factor
        exposure[idx] = factor

    return (
        pd.Series(adjusted, index=rets.index, dtype=float),
        pd.Series(exposure, index=rets.index, dtype=float),
    )


def score_row(annual: float, max_dd: float, target_annual: float, target_max_dd: float) -> float:
    dd_excess = max(0.0, abs(min(max_dd, 0.0)) - target_max_dd)
    annual_gap = max(0.0, target_annual - annual)
    passed = annual >= target_annual and max_dd >= -target_max_dd
    return float((1.0 if passed else 0.0) + annual - 3.0 * dd_excess - annual_gap)


def load_base_candidates(args) -> pd.DataFrame:
    df = pd.read_csv(args.search_csv)
    df = df[df["status"].eq("ok")].copy()
    df = df[df["topk"] <= args.max_topk].copy()
    df = df[pd.to_numeric(df["factor_count"], errors="coerce") < args.max_factors].copy()
    if args.sort_by not in df.columns:
        args.sort_by = "annual_return"
    df = df.sort_values(args.sort_by, ascending=False)
    if args.limit_base > 0:
        df = df.head(args.limit_base)
    return df


def iter_configs(args):
    lookbacks = parse_ints(args.lookback_days)
    min_total_returns = parse_floats(args.min_total_returns)
    min_annual_returns = parse_floats(args.min_annual_returns)
    min_sharpes = parse_floats(args.min_sharpes)
    max_drawdowns = parse_floats(args.max_drawdown_rules)
    reduce_tos = parse_floats(args.reduce_tos)
    actions = [x.strip() for x in args.actions.split(",") if x.strip()]

    for lookback, min_total, min_annual, min_sharpe, dd_rule, action in itertools.product(
        lookbacks,
        min_total_returns,
        min_annual_returns,
        min_sharpes,
        max_drawdowns,
        actions,
    ):
        if action == "pause":
            reduce_grid = [0.0]
        else:
            reduce_grid = reduce_tos
        for reduce_to in reduce_grid:
            yield ValidityConfig(
                lookback_days=lookback,
                min_observations=max(5, min(args.min_observations, lookback)),
                min_total_return=min_total,
                min_annual_return=min_annual,
                min_sharpe=min_sharpe,
                max_drawdown=dd_rule,
                action=action,
                reduce_to=reduce_to,
                apply_in_backtest=True,
            )


def scan(args) -> pd.DataFrame:
    args.search_csv = Path(args.search_csv)
    args.output = Path(args.output)
    candidates = load_base_candidates(args)
    if candidates.empty:
        raise RuntimeError("No base candidates found.")

    rows = []
    configs = list(iter_configs(args))
    print(f"[INFO] Base candidates: {len(candidates)}")
    print(f"[INFO] Overlay configs: {len(configs)}")

    for _, candidate in candidates.iterrows():
        results_file = Path(str(candidate["results_file"]))
        if not results_file.is_absolute():
            results_file = PROJECT_ROOT / results_file
        if not results_file.exists():
            print(f"[WARN] Missing results file: {results_file}")
            continue

        raw = pd.read_csv(results_file, parse_dates=["date"]).set_index("date")["return"]
        base = metrics(raw)
        best_score = -1e9
        best_preview = None

        for cfg in configs:
            adjusted, exposure = fast_validity_overlay(raw, cfg)
            m = metrics(adjusted)
            passed = bool(
                m["annual"] >= args.target_annual
                and m["max_dd"] >= -args.target_max_drawdown
            )
            score = score_row(
                m["annual"], m["max_dd"], args.target_annual, args.target_max_drawdown
            )
            row = {
                "base_tag": candidate["tag"],
                "factor_count": int(candidate["factor_count"]),
                "factor_names": candidate["factor_names"],
                "topk": int(candidate["topk"]),
                "base_annual": base["annual"],
                "base_max_dd": base["max_dd"],
                "base_sharpe": base["sharpe"],
                "lookback_days": cfg.lookback_days,
                "min_observations": cfg.min_observations,
                "min_total_return": cfg.min_total_return,
                "min_annual_return": cfg.min_annual_return,
                "min_sharpe": cfg.min_sharpe,
                "max_drawdown_rule": cfg.max_drawdown,
                "action": cfg.action,
                "reduce_to": cfg.reduce_to,
                "annual_return": m["annual"],
                "max_drawdown": m["max_dd"],
                "sharpe_ratio": m["sharpe"],
                "total_return": m["total"],
                "avg_exposure": float(exposure.mean()) if not exposure.empty else 0.0,
                "min_exposure": float(exposure.min()) if not exposure.empty else 0.0,
                "active_days": int((exposure > 0).sum()) if not exposure.empty else 0,
                "passed": passed,
                "score": score,
                "results_file": str(results_file),
            }
            rows.append(row)
            if score > best_score:
                best_score = score
                best_preview = row

        if best_preview:
            print(
                f"  {candidate['tag']} best annual={best_preview['annual_return']:.2%} "
                f"dd={best_preview['max_drawdown']:.2%} "
                f"avg_exp={best_preview['avg_exposure']:.1%}"
            )

    out = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    out = out.sort_values(["passed", "score"], ascending=[False, False])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scan validity overlays for formal Alpha158 LGBM results.")
    parser.add_argument("--search-csv", default=PROJECT_ROOT / "results" / "alpha158_lgbm_biweek_search.csv")
    parser.add_argument("--output", default=PROJECT_ROOT / "results" / "alpha158_lgbm_biweek_validity_scan.csv")
    parser.add_argument("--limit-base", type=int, default=20)
    parser.add_argument("--sort-by", default="annual_return")
    parser.add_argument("--max-topk", type=int, default=10)
    parser.add_argument("--max-factors", type=int, default=8)
    parser.add_argument("--target-annual", type=float, default=0.20)
    parser.add_argument("--target-max-drawdown", type=float, default=0.10)
    parser.add_argument("--lookback-days", default="20,40,60,80,120")
    parser.add_argument("--min-observations", type=int, default=20)
    parser.add_argument("--min-total-returns", default="-0.02,-0.03,-0.05,-0.08")
    parser.add_argument("--min-annual-returns", default="-0.10,0.00,0.05")
    parser.add_argument("--min-sharpes", default="-0.5,0.0,0.2")
    parser.add_argument("--max-drawdown-rules", default="-0.03,-0.05,-0.08,-0.10")
    parser.add_argument("--actions", default="reduce,pause")
    parser.add_argument("--reduce-tos", default="0.25,0.50")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    out = scan(args)
    passed = out[out["passed"].astype(bool)]
    print("\n[SUMMARY]")
    print(f"Rows: {len(out)}  Passed: {len(passed)}")
    cols = [
        "base_tag",
        "factor_names",
        "topk",
        "annual_return",
        "max_drawdown",
        "sharpe_ratio",
        "avg_exposure",
        "lookback_days",
        "action",
        "reduce_to",
        "score",
    ]
    print(out[cols].head(20).to_string(index=False))
    print(f"\n[OK] Results saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
