#!/usr/bin/env python3
"""Search Alpha158 LightGBM strategies under formal backtest constraints.

The search is intentionally bounded:
1. Build an Alpha158 candidate pool from the IC/IR scan result.
2. Use beam search to explore factor subsets with fewer than 8 factors.
3. Generate selections for each candidate and run the formal Qlib backtest,
   including commissions, slippage, impact, and tradability constraints.
4. Persist every evaluation so the search can be resumed.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import itertools
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import CONFIG
from core.factors import FactorInfo, FactorRegistry
from core.selection import (
    _load_total_mv_frame,
    extract_topk,
    load_factor_data,
)
from core.strategy import Strategy
from core.lgbm_scorer import predict_with_model, train_lgbm_model
from modules.backtest.composite import run_strategy_backtest


BASE_LGBM_PARAMS = {
    "objective": "regression",
    "metric": "mse",
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbose": -1,
}

OUTPUT_COLUMNS = [
    "tag",
    "status",
    "factor_count",
    "factor_names",
    "topk",
    "rebalance_freq",
    "num_leaves",
    "learning_rate",
    "min_child_samples",
    "n_estimators",
    "forward_days",
    "underfilled_days",
    "annual_return",
    "max_drawdown",
    "sharpe_ratio",
    "total_return",
    "oos_annual",
    "oos_max_dd",
    "oos_sharpe",
    "train_annual",
    "train_max_dd",
    "train_sharpe",
    "passed_full",
    "rank_score",
    "results_file",
    "selection_file",
    "error",
    "elapsed_s",
]


def ensure_openmp_runtime() -> None:
    """Preload bundled OpenMP runtime when LightGBM cannot find libgomp."""
    candidates = [
        PROJECT_ROOT / ".venv" / "lib" / "libgomp.so.1",
        PROJECT_ROOT / ".venv" / "lib" / "libgomp.so",
    ]
    for path in candidates:
        if path.exists():
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
            return


def parse_csv_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_csv_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def get_alpha158_factor_map() -> dict[str, str]:
    from qlib.contrib.data.loader import Alpha158DL

    conf = {
        "kbar": {},
        "price": {
            "windows": [0],
            "feature": ["OPEN", "HIGH", "LOW", "VWAP"],
        },
        "rolling": {},
    }
    fields, names = Alpha158DL.get_feature_config(conf)
    return dict(zip(names, fields))


def make_factor_info(name: str, expression: str, ir: float, negate: bool) -> FactorInfo:
    return FactorInfo(
        name=name,
        expression=expression,
        description=f"Alpha158 {name}",
        category="alpha",
        source="qlib",
        negate=negate,
        ir=abs(float(ir)),
    )


def build_registry(factors: Iterable[FactorInfo]) -> FactorRegistry:
    registry = FactorRegistry()
    for factor in factors:
        registry.register(factor)
    return registry


def load_candidate_factors(args) -> list[FactorInfo]:
    scan = pd.read_csv(args.scan_csv)
    expressions = get_alpha158_factor_map()

    rows = []
    for _, row in scan.iterrows():
        name = str(row["因子"])
        if name not in expressions:
            continue
        abs_ir = float(row["|IR|"])
        stability = float(row.get("稳定性", 0.0))
        if abs_ir < args.min_abs_ir or stability < args.min_stability:
            continue
        direction = str(row.get("方向", ""))
        signed_ir = float(row.get("IR", row.get("IC均值", 0.0)))
        negate = ("反向" in direction) or signed_ir < 0
        rows.append(
            {
                "name": name,
                "expression": expressions[name],
                "ir": signed_ir,
                "abs_ir": abs_ir,
                "stability": stability,
                "negate": negate,
            }
        )

    rows = sorted(rows, key=lambda x: (x["abs_ir"], x["stability"]), reverse=True)
    rows = rows[: args.candidate_pool_size]
    return [
        make_factor_info(r["name"], r["expression"], r["ir"], r["negate"])
        for r in rows
    ]


def candidate_hash(factor_names: tuple[str, ...], params: dict, topk: int) -> str:
    payload = json.dumps(
        {"factors": factor_names, "params": params, "topk": topk},
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def period_metrics(daily_returns: pd.Series, start=None, end=None) -> dict[str, float]:
    returns = daily_returns.copy()
    if start:
        returns = returns[returns.index >= pd.Timestamp(start)]
    if end:
        returns = returns[returns.index <= pd.Timestamp(end)]
    if returns.empty:
        return {"annual": 0.0, "max_dd": 0.0, "sharpe": 0.0, "total": 0.0}

    pv = (1 + returns).cumprod()
    days = (pv.index[-1] - pv.index[0]).days
    terminal = float(pv.iloc[-1])
    annual = terminal ** (365 / days) - 1 if days > 0 and terminal > 0 else -1.0
    max_dd = float((pv / pv.cummax() - 1).min())
    std = returns.std()
    sharpe = float(returns.mean() / std * np.sqrt(252)) if std > 0 else 0.0
    return {
        "annual": float(annual),
        "max_dd": max_dd,
        "sharpe": sharpe,
        "total": float(terminal - 1.0),
    }


def rank_score(row: dict, target_annual: float, target_max_dd: float) -> float:
    dd_excess = max(0.0, abs(min(row["max_drawdown"], 0.0)) - target_max_dd)
    factor_penalty = 0.005 * max(int(row["factor_count"]) - 1, 0)
    pass_bonus = 1.0 if row["passed_full"] else 0.0
    return float(row["annual_return"] - 2.0 * dd_excess - factor_penalty + pass_bonus)


def prepare_market_cap_series(monthly_df: pd.DataFrame, rebalance_dates, args):
    if args.min_market_cap <= 0:
        return 0.0, None

    instruments = monthly_df.index.get_level_values("instrument").unique().tolist()
    total_mv_frame = _load_total_mv_frame(
        instruments=instruments,
        start_date=args.data_start,
        end_date=args.data_end,
    )
    mv_df = total_mv_frame[total_mv_frame["datetime"].isin(rebalance_dates)]
    mv_series = mv_df.set_index(["datetime", "symbol"])["total_mv"]
    return float(args.min_market_cap) * 10000, mv_series


def build_strategy(base: Strategy, tag: str, registry: FactorRegistry, topk: int, args) -> Strategy:
    trading_cost = {
        "open_cost": args.buy_commission_rate,
        "close_cost": args.sell_commission_rate + args.sell_stamp_tax_rate,
        "buy_commission_rate": args.buy_commission_rate,
        "sell_commission_rate": args.sell_commission_rate,
        "sell_stamp_tax_rate": args.sell_stamp_tax_rate,
        "min_buy_commission": args.min_buy_commission,
        "min_sell_commission": args.min_sell_commission,
        "slippage_bps": args.slippage_bps,
        "impact_bps": args.impact_bps,
        "block_limit_up_buy": args.block_limit_up_buy,
        "block_limit_down_sell": args.block_limit_down_sell,
    }
    return replace(
        base,
        name=f"alpha158_lgbm_biweek__{tag}",
        display_name=f"alpha158_lgbm_biweek_{tag}",
        description="Alpha158 LGBM biweekly search candidate",
        registry=registry,
        weights={"alpha": 1.0},
        topk=topk,
        neutralize_industry=True,
        universe=args.universe,
        min_market_cap=args.min_market_cap,
        exclude_st=True,
        exclude_new_days=args.exclude_new_days,
        sticky=args.sticky,
        buffer=args.buffer,
        churn_limit=args.churn_limit,
        margin_stable=args.margin_stable,
        position_model="fixed",
        position_params={"stock_pct": args.stock_pct},
        rebalance_freq="biweek",
        trading_cost=trading_cost,
        scorer="lgbm",
        lgbm_train_start=args.train_start,
        lgbm_train_end=args.train_end,
    )


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([{col: row.get(col, "") for col in OUTPUT_COLUMNS}])
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def load_done(output_path: Path, force: bool) -> set[str]:
    if force or not output_path.exists() or output_path.stat().st_size == 0:
        return set()
    df = pd.read_csv(output_path, usecols=["tag", "status"])
    return set(df.loc[df["status"] == "ok", "tag"].astype(str))


def load_existing_results(output_path: Path, force: bool) -> pd.DataFrame:
    if force or not output_path.exists() or output_path.stat().st_size == 0:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    return pd.read_csv(output_path)


def existing_combo_score(
    existing: pd.DataFrame,
    combo: tuple[str, ...],
    params: dict,
    topk_options: list[int],
) -> float:
    if existing.empty or "factor_names" not in existing.columns:
        return -1e9
    mask = existing["status"].eq("ok")
    mask &= existing["factor_names"].astype(str).eq("|".join(combo))
    mask &= existing["topk"].isin(topk_options)
    for key, value in params.items():
        if key in existing.columns:
            mask &= existing[key].eq(value)
    matched = existing.loc[mask]
    if matched.empty or "rank_score" not in matched.columns:
        return -1e9
    return float(pd.to_numeric(matched["rank_score"], errors="coerce").max())


def evaluate_candidate(
    factor_names: tuple[str, ...],
    factor_map: dict[str, FactorInfo],
    monthly_df_all: pd.DataFrame,
    rebalance_dates,
    mv_floor: float,
    mv_series,
    base_strategy: Strategy,
    params: dict,
    topk: int,
    done_tags: set[str],
    args,
) -> dict | None:
    factor_names = tuple(sorted(factor_names))
    tag_hash = candidate_hash(factor_names, params, topk)
    tag = f"f{len(factor_names)}_k{topk}_{tag_hash}"
    if tag in done_tags:
        return None

    t0 = time.perf_counter()
    selected_factors = [factor_map[name] for name in factor_names]
    registry = build_registry(selected_factors)
    columns = [f"alpha_{name}" for name in factor_names]
    monthly_df = monthly_df_all.loc[:, columns].copy()

    row = {
        "tag": tag,
        "status": "ok",
        "factor_count": len(factor_names),
        "factor_names": "|".join(factor_names),
        "topk": topk,
        "rebalance_freq": "biweek",
        "num_leaves": params["num_leaves"],
        "learning_rate": params["learning_rate"],
        "min_child_samples": params["min_child_samples"],
        "n_estimators": params["n_estimators"],
        "forward_days": params["forward_days"],
    }

    try:
        ensure_openmp_runtime()
        lgbm_params = {
            **BASE_LGBM_PARAMS,
            "num_leaves": params["num_leaves"],
            "learning_rate": params["learning_rate"],
            "min_child_samples": params["min_child_samples"],
            "n_estimators": params["n_estimators"],
        }
        model, feature_cols, df_neutralized = train_lgbm_model(
            monthly_df,
            train_start=args.train_start,
            train_end=args.train_end,
            forward_days=params["forward_days"],
            lgbm_params=lgbm_params,
            neutralize_industry=True,
        )
        signal = predict_with_model(model, df_neutralized, feature_cols)
        df_sel = extract_topk(
            signal,
            rebalance_dates,
            topk=topk,
            mv_floor=mv_floor,
            mv_series=mv_series,
            sticky=args.sticky,
            churn_limit=args.churn_limit,
            margin_stable=args.margin_stable,
            buffer=args.buffer,
            exclude_new_days=args.exclude_new_days,
            exclude_st=True,
            universe=args.universe,
        )
        if df_sel.empty:
            row.update({"status": "empty_selection"})
            append_row(args.output, row)
            return row

        counts = df_sel.groupby("date").size()
        underfilled = int((counts < topk).sum())
        row["underfilled_days"] = underfilled
        if underfilled and not args.allow_underfilled:
            row.update({"status": "underfilled"})
            append_row(args.output, row)
            return row

        strategy = build_strategy(base_strategy, tag, registry, topk, args)
        selection_path = strategy.selections_path()
        selection_path.parent.mkdir(parents=True, exist_ok=True)
        df_sel.to_csv(selection_path, index=False)
        strategy._write_selection_cache_metadata()

        bt = run_strategy_backtest(strategy, engine="qlib")
        oos = period_metrics(bt.daily_returns, start=args.oos_start)
        train = period_metrics(bt.daily_returns, end=args.train_end)
        row.update(
            {
                "annual_return": bt.annual_return,
                "max_drawdown": bt.max_drawdown,
                "sharpe_ratio": bt.sharpe_ratio,
                "total_return": bt.total_return,
                "oos_annual": oos["annual"],
                "oos_max_dd": oos["max_dd"],
                "oos_sharpe": oos["sharpe"],
                "train_annual": train["annual"],
                "train_max_dd": train["max_dd"],
                "train_sharpe": train["sharpe"],
                "passed_full": bool(
                    bt.annual_return >= args.target_annual
                    and bt.max_drawdown >= -args.target_max_drawdown
                    and topk <= 10
                    and len(factor_names) < 8
                ),
                "results_file": bt.metadata.get("results_file", ""),
                "selection_file": str(selection_path),
            }
        )
        row["rank_score"] = rank_score(row, args.target_annual, args.target_max_drawdown)
    except Exception as exc:
        row.update({"status": "failed", "error": str(exc)[:500]})
    finally:
        row["elapsed_s"] = round(time.perf_counter() - t0, 1)
        append_row(args.output, row)

    return row


def fixed_params(args) -> dict:
    return {
        "num_leaves": args.stage1_num_leaves,
        "learning_rate": args.stage1_learning_rate,
        "min_child_samples": args.stage1_min_child_samples,
        "n_estimators": args.stage1_n_estimators,
        "forward_days": args.stage1_forward_days,
    }


def stage2_param_grid(args) -> list[dict]:
    keys = [
        "num_leaves",
        "learning_rate",
        "min_child_samples",
        "n_estimators",
        "forward_days",
    ]
    values = [
        parse_csv_ints(args.stage2_num_leaves),
        parse_csv_floats(args.stage2_learning_rates),
        parse_csv_ints(args.stage2_min_child_samples),
        parse_csv_ints(args.stage2_n_estimators),
        parse_csv_ints(args.stage2_forward_days),
    ]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def search(args) -> pd.DataFrame:
    args.output = Path(args.output)
    args.scan_csv = Path(args.scan_csv)
    args.data_end = args.data_end or CONFIG.get("end_date", "2026-02-26")
    args.oos_start = args.oos_start or (
        pd.Timestamp(args.train_end) + pd.Timedelta(days=1)
    ).strftime("%Y-%m-%d")

    factors = load_candidate_factors(args)
    if not factors:
        raise RuntimeError("No Alpha158 candidate factors after filtering.")

    print(f"[INFO] Candidate factors: {len(factors)}")
    print("[INFO] " + ", ".join(f.name for f in factors))

    factor_map = {f.name: f for f in factors}
    all_registry = build_registry(factors)
    monthly_df_all, rebalance_dates = load_factor_data(
        registry=all_registry,
        start_date=args.data_start,
        end_date=args.data_end,
        rebalance_freq="biweek",
        universe=args.universe,
    )
    mv_floor, mv_series = prepare_market_cap_series(monthly_df_all, rebalance_dates, args)

    base_strategy = Strategy.load("experimental/alpha158/alpha158_csi300")
    done_tags = load_done(args.output, args.force)
    existing_results = load_existing_results(args.output, args.force)
    eval_count = 0
    rows: list[dict] = []
    topk_options = [k for k in parse_csv_ints(args.topk_options) if k <= 10]
    stage1_params = fixed_params(args)

    seen_factor_combos: set[tuple[str, ...]] = set()
    frontier = [(f.name,) for f in factors[: args.seed_count]]

    for size in range(1, args.max_factors + 1):
        if not frontier or eval_count >= args.max_evals:
            break
        print(f"\n[STAGE1] factor_count={size}, combos={len(frontier)}")

        scored: dict[tuple[str, ...], float] = {}
        for combo in frontier[: args.max_expansions_per_level]:
            combo = tuple(sorted(combo))
            if combo in seen_factor_combos:
                continue
            seen_factor_combos.add(combo)
            best_score = existing_combo_score(existing_results, combo, stage1_params, topk_options)
            for topk in topk_options:
                if eval_count >= args.max_evals:
                    break
                row = evaluate_candidate(
                    combo,
                    factor_map,
                    monthly_df_all,
                    rebalance_dates,
                    mv_floor,
                    mv_series,
                    base_strategy,
                    stage1_params,
                    topk,
                    done_tags,
                    args,
                )
                if row is None:
                    continue
                rows.append(row)
                eval_count += 1
                if row.get("status") == "ok":
                    best_score = max(best_score, float(row.get("rank_score", -1e9)))
                    print(
                        f"  {row['tag']} annual={row['annual_return']:.2%} "
                        f"dd={row['max_drawdown']:.2%} oos={row['oos_annual']:.2%}"
                    )
                else:
                    print(f"  {row['tag']} status={row.get('status')}")
            scored[combo] = best_score

        leaders = [
            combo
            for combo, _ in sorted(scored.items(), key=lambda x: x[1], reverse=True)[
                : args.beam_width
            ]
        ]
        next_frontier = set()
        for combo in leaders:
            for factor in factors:
                if factor.name in combo:
                    continue
                expanded = tuple(sorted(set(combo) | {factor.name}))
                if len(expanded) == size + 1:
                    next_frontier.add(expanded)
        frontier = sorted(next_frontier)

    if rows:
        df_rows = pd.DataFrame(rows)
    elif args.output.exists():
        df_rows = pd.read_csv(args.output)
    else:
        df_rows = pd.DataFrame()

    ok_rows = df_rows[df_rows.get("status", pd.Series(dtype=str)) == "ok"].copy()
    if args.stage2_top_combos > 0 and not ok_rows.empty and eval_count < args.max_evals:
        ok_rows = ok_rows.sort_values("rank_score", ascending=False)
        top_combos = []
        for _, row in ok_rows.iterrows():
            combo = tuple(sorted(str(row["factor_names"]).split("|")))
            if combo not in top_combos:
                top_combos.append(combo)
            if len(top_combos) >= args.stage2_top_combos:
                break

        print(f"\n[STAGE2] Tuning {len(top_combos)} factor combos")
        for combo in top_combos:
            for params in stage2_param_grid(args):
                for topk in topk_options:
                    if eval_count >= args.max_evals:
                        break
                    row = evaluate_candidate(
                        combo,
                        factor_map,
                        monthly_df_all,
                        rebalance_dates,
                        mv_floor,
                        mv_series,
                        base_strategy,
                        params,
                        topk,
                        done_tags,
                        args,
                    )
                    if row is None:
                        continue
                    rows.append(row)
                    eval_count += 1
                    if row.get("status") == "ok":
                        print(
                            f"  {row['tag']} annual={row['annual_return']:.2%} "
                            f"dd={row['max_drawdown']:.2%} oos={row['oos_annual']:.2%}"
                        )
                    else:
                        print(f"  {row['tag']} status={row.get('status')}")

    if args.output.exists():
        final = pd.read_csv(args.output)
        if "rank_score" in final.columns:
            final = final.sort_values(["passed_full", "rank_score"], ascending=[False, False])
        final.to_csv(args.output, index=False)
        return final
    return pd.DataFrame(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search formal Alpha158 LGBM biweekly top-k strategies."
    )
    parser.add_argument("--scan-csv", default=PROJECT_ROOT / "results" / "alpha158_scan.csv")
    parser.add_argument("--output", default=PROJECT_ROOT / "results" / "alpha158_lgbm_biweek_search.csv")
    parser.add_argument("--candidate-pool-size", type=int, default=18)
    parser.add_argument("--min-abs-ir", type=float, default=0.12)
    parser.add_argument("--min-stability", type=float, default=0.70)
    parser.add_argument("--seed-count", type=int, default=10)
    parser.add_argument("--max-factors", type=int, default=7)
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--max-expansions-per-level", type=int, default=25)
    parser.add_argument("--max-evals", type=int, default=80)
    parser.add_argument("--stage2-top-combos", type=int, default=4)
    parser.add_argument("--topk-options", default="5,8,10")
    parser.add_argument("--universe", default="csi300")
    parser.add_argument("--data-start", default="2018-12-01")
    parser.add_argument("--data-end", default="")
    parser.add_argument("--train-start", default="2019-01-01")
    parser.add_argument("--train-end", default="2021-12-31")
    parser.add_argument("--oos-start", default="")
    parser.add_argument("--target-annual", type=float, default=0.20)
    parser.add_argument("--target-max-drawdown", type=float, default=0.10)
    parser.add_argument("--stock-pct", type=float, default=0.88)
    parser.add_argument("--min-market-cap", type=float, default=50.0)
    parser.add_argument("--exclude-new-days", type=int, default=120)
    parser.add_argument("--sticky", type=int, default=5)
    parser.add_argument("--buffer", type=int, default=20)
    parser.add_argument("--churn-limit", type=int, default=3)
    parser.add_argument("--margin-stable", action="store_true", default=True)
    parser.add_argument("--allow-underfilled", action="store_true")
    parser.add_argument("--buy-commission-rate", type=float, default=0.0003)
    parser.add_argument("--sell-commission-rate", type=float, default=0.0003)
    parser.add_argument("--sell-stamp-tax-rate", type=float, default=0.0010)
    parser.add_argument("--min-buy-commission", type=float, default=5.0)
    parser.add_argument("--min-sell-commission", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--impact-bps", type=float, default=5.0)
    parser.add_argument("--block-limit-up-buy", action="store_true", default=True)
    parser.add_argument("--block-limit-down-sell", action="store_true", default=True)
    parser.add_argument("--stage1-num-leaves", type=int, default=31)
    parser.add_argument("--stage1-learning-rate", type=float, default=0.05)
    parser.add_argument("--stage1-min-child-samples", type=int, default=200)
    parser.add_argument("--stage1-n-estimators", type=int, default=200)
    parser.add_argument("--stage1-forward-days", type=int, default=5)
    parser.add_argument("--stage2-num-leaves", default="15,31,63")
    parser.add_argument("--stage2-learning-rates", default="0.03,0.05")
    parser.add_argument("--stage2-min-child-samples", default="100,200,500")
    parser.add_argument("--stage2-n-estimators", default="100,200")
    parser.add_argument("--stage2-forward-days", default="5,10")
    parser.add_argument("--force", action="store_true", help="Ignore existing completed rows.")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    final = search(args)
    if final.empty:
        print("[WARN] No evaluations completed.")
        return 1

    ok = final[final["status"] == "ok"].copy()
    passed = ok[ok["passed_full"].astype(bool)] if "passed_full" in ok.columns else pd.DataFrame()
    print("\n[SUMMARY]")
    print(f"Evaluations: {len(final)}  OK: {len(ok)}  Passed full target: {len(passed)}")
    cols = [
        "tag",
        "factor_count",
        "factor_names",
        "topk",
        "annual_return",
        "max_drawdown",
        "sharpe_ratio",
        "oos_annual",
        "oos_max_dd",
        "rank_score",
    ]
    if ok.empty:
        print("No successful evaluations yet.")
    else:
        sort_col = "rank_score" if "rank_score" in ok.columns else "annual_return"
        show = ok.sort_values(sort_col, ascending=False).head(10)
        print(show[[c for c in cols if c in show.columns]].to_string(index=False))
    print(f"\n[OK] Results saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
