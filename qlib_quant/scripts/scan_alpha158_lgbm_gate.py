#!/usr/bin/env python3
"""Run formal market-gate backtests for Alpha158 LGBM selection files."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.strategy import Strategy
from modules.backtest.composite import run_strategy_backtest
from modules.data.tushare_to_qlib import write_dense_bin_file
from scripts.search_alpha158_lgbm_biweek import period_metrics


OUTPUT_COLUMNS = [
    "tag",
    "status",
    "base_tag",
    "factor_count",
    "factor_names",
    "topk",
    "ma_window",
    "strong_stock_pct",
    "mixed_stock_pct",
    "weak_stock_pct",
    "annual_return",
    "max_drawdown",
    "sharpe_ratio",
    "total_return",
    "oos_annual",
    "oos_max_dd",
    "oos_sharpe",
    "passed",
    "score",
    "results_file",
    "selection_file",
    "error",
    "elapsed_s",
]


def parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_triplets(text: str) -> list[tuple[float, float, float]]:
    triplets = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        parts = [float(x.strip()) for x in item.split("/") if x.strip()]
        if len(parts) != 3:
            raise ValueError(f"Invalid pct triplet: {item}")
        strong, mixed, weak = parts
        if not (0.0 <= weak <= mixed <= strong <= 1.0):
            raise ValueError(f"Invalid pct order: {item}")
        triplets.append((strong, mixed, weak))
    return triplets


def metrics_score(annual: float, max_dd: float, target_annual: float, target_max_dd: float) -> float:
    dd_excess = max(0.0, abs(min(max_dd, 0.0)) - target_max_dd)
    annual_gap = max(0.0, target_annual - annual)
    passed = annual >= target_annual and max_dd >= -target_max_dd
    return float((1.0 if passed else 0.0) + annual - 3.0 * dd_excess - annual_gap)


def ensure_sh000300_close_bin(args) -> Path:
    """Build sh000300 close.day.bin from local Tushare index parquet if missing."""
    qlib_dir = Path(args.qlib_dir)
    out_dir = qlib_dir / "features" / args.csi300_symbol
    out_file = out_dir / "close.day.bin"
    if out_file.exists() and not args.force_index_bin:
        return out_file

    parquet = Path(args.index_parquet)
    if not parquet.exists():
        raise FileNotFoundError(f"Index parquet not found: {parquet}")

    cal = pd.read_csv(
        qlib_dir / "calendars" / "day.txt",
        header=None,
        names=["date"],
        parse_dates=["date"],
    )
    cal["date"] = cal["date"].dt.normalize()
    date_to_idx = {date: idx for idx, date in enumerate(cal["date"])}

    df = pd.read_parquet(parquet)
    df = df[df["ts_code"].eq(args.csi300_ts_code)].copy()
    if df.empty:
        raise RuntimeError(f"No rows for {args.csi300_ts_code} in {parquet}")
    df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d").dt.normalize()
    df["cal_idx"] = df["date"].map(date_to_idx)
    df = df.dropna(subset=["cal_idx", "close"]).copy()
    df["cal_idx"] = df["cal_idx"].astype(int)
    df = df.drop_duplicates(subset=["cal_idx"], keep="last").sort_values("cal_idx")
    if df.empty:
        raise RuntimeError(f"No {args.csi300_ts_code} rows matched Qlib calendar")

    out_dir.mkdir(parents=True, exist_ok=True)
    ok = write_dense_bin_file(out_file, df["cal_idx"], pd.to_numeric(df["close"], errors="coerce"))
    if not ok:
        raise RuntimeError(f"Failed writing {out_file}")
    return out_file


def load_base_candidates(args) -> pd.DataFrame:
    df = pd.read_csv(args.search_csv)
    df = df[df["status"].eq("ok")].copy()
    df = df[pd.to_numeric(df["topk"], errors="coerce") <= args.max_topk].copy()
    df = df[pd.to_numeric(df["factor_count"], errors="coerce") < args.max_factors].copy()

    frames = []
    for sort_col in ["rank_score", "annual_return"]:
        if sort_col in df.columns:
            frames.append(df.sort_values(sort_col, ascending=False).head(args.limit_base))
    if frames:
        return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["tag"])
    if args.limit_base > 0:
        df = df.head(args.limit_base)
    return df


def gate_hash(base_tag: str, ma_window: int, strong: float, mixed: float, weak: float) -> str:
    payload = json.dumps(
        {
            "base_tag": base_tag,
            "ma": ma_window,
            "strong": strong,
            "mixed": mixed,
            "weak": weak,
        },
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{col: row.get(col, "") for col in OUTPUT_COLUMNS}]).to_csv(
        path,
        mode="a",
        header=not path.exists(),
        index=False,
    )


def done_tags(path: Path, force: bool) -> set[str]:
    if force or not path.exists() or path.stat().st_size == 0:
        return set()
    df = pd.read_csv(path, usecols=["tag", "status"])
    return set(df.loc[df["status"].eq("ok"), "tag"].astype(str))


def build_strategy_for_gate(base_strategy: Strategy, row: pd.Series, tag: str, args, params: dict) -> Strategy:
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
        base_strategy,
        name=f"alpha158_lgbm_gate__{tag}",
        display_name=f"alpha158_lgbm_gate_{tag}",
        description=f"Gate retest for {row['tag']}",
        topk=int(row["topk"]),
        neutralize_industry=True,
        universe=args.universe,
        min_market_cap=args.min_market_cap,
        exclude_st=True,
        exclude_new_days=args.exclude_new_days,
        sticky=args.sticky,
        buffer=args.buffer,
        churn_limit=args.churn_limit,
        margin_stable=True,
        position_model="gate",
        position_params=params,
        rebalance_freq="biweek",
        trading_cost=trading_cost,
        scorer="lgbm",
        lgbm_train_start=args.train_start,
        lgbm_train_end=args.train_end,
    )


def scan(args) -> pd.DataFrame:
    args.output = Path(args.output)
    ensure_sh000300_close_bin(args)
    candidates = load_base_candidates(args)
    if candidates.empty:
        raise RuntimeError("No base candidates found.")

    ma_windows = parse_ints(args.ma_windows)
    pct_triplets = parse_triplets(args.stock_pct_triplets)
    print(f"[INFO] Base candidates: {len(candidates)}")
    print(f"[INFO] Gate configs: {len(ma_windows) * len(pct_triplets)}")

    completed = done_tags(args.output, args.force)
    base_strategy = Strategy.load("experimental/alpha158/alpha158_csi300")
    rows = []

    for _, candidate in candidates.iterrows():
        base_selection = Path(str(candidate["selection_file"]))
        if not base_selection.is_absolute():
            base_selection = PROJECT_ROOT / base_selection
        if not base_selection.exists():
            print(f"[WARN] Missing selection file: {base_selection}")
            continue

        best = None
        for ma_window in ma_windows:
            for strong, mixed, weak in pct_triplets:
                h = gate_hash(str(candidate["tag"]), ma_window, strong, mixed, weak)
                tag = f"{candidate['tag']}_ma{ma_window}_{h}"
                if tag in completed:
                    continue

                params = {
                    "csi300_symbol": args.csi300_symbol,
                    "csi500_symbol": args.csi500_symbol,
                    "ma_window": ma_window,
                    "strong_stock_pct": strong,
                    "mixed_stock_pct": mixed,
                    "weak_stock_pct": weak,
                    "bond_annual_return": args.bond_annual_return,
                    "qlib_data_path": args.qlib_dir,
                }
                row = {
                    "tag": tag,
                    "status": "ok",
                    "base_tag": candidate["tag"],
                    "factor_count": int(candidate["factor_count"]),
                    "factor_names": candidate["factor_names"],
                    "topk": int(candidate["topk"]),
                    "ma_window": ma_window,
                    "strong_stock_pct": strong,
                    "mixed_stock_pct": mixed,
                    "weak_stock_pct": weak,
                }
                t0 = time.perf_counter()
                try:
                    strategy = build_strategy_for_gate(base_strategy, candidate, tag, args, params)
                    selection_path = strategy.selections_path()
                    selection_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(base_selection, selection_path)
                    strategy._write_selection_cache_metadata()

                    bt = run_strategy_backtest(strategy, engine="qlib")
                    oos = period_metrics(bt.daily_returns, start=args.oos_start)
                    passed = bool(
                        bt.annual_return >= args.target_annual
                        and bt.max_drawdown >= -args.target_max_drawdown
                        and int(candidate["topk"]) <= 10
                        and int(candidate["factor_count"]) < 8
                    )
                    row.update(
                        {
                            "annual_return": bt.annual_return,
                            "max_drawdown": bt.max_drawdown,
                            "sharpe_ratio": bt.sharpe_ratio,
                            "total_return": bt.total_return,
                            "oos_annual": oos["annual"],
                            "oos_max_dd": oos["max_dd"],
                            "oos_sharpe": oos["sharpe"],
                            "passed": passed,
                            "score": metrics_score(
                                bt.annual_return,
                                bt.max_drawdown,
                                args.target_annual,
                                args.target_max_drawdown,
                            ),
                            "results_file": bt.metadata.get("results_file", ""),
                            "selection_file": str(selection_path),
                        }
                    )
                except Exception as exc:
                    row.update({"status": "failed", "error": str(exc)[:500]})
                finally:
                    row["elapsed_s"] = round(time.perf_counter() - t0, 1)
                    append_row(args.output, row)
                rows.append(row)
                if row.get("status") == "ok" and (
                    best is None or float(row.get("score", -1e9)) > float(best.get("score", -1e9))
                ):
                    best = row

        if best:
            print(
                f"  {candidate['tag']} best annual={best['annual_return']:.2%} "
                f"dd={best['max_drawdown']:.2%} "
                f"gate={best['ma_window']}/{best['strong_stock_pct']:.0%}/"
                f"{best['mixed_stock_pct']:.0%}/{best['weak_stock_pct']:.0%}"
            )

    if args.output.exists():
        out = pd.read_csv(args.output)
        out = out.sort_values(["passed", "score"], ascending=[False, False])
        out.to_csv(args.output, index=False)
        return out
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scan market-gate formal backtests for Alpha158 LGBM.")
    parser.add_argument("--search-csv", default=PROJECT_ROOT / "results" / "alpha158_lgbm_biweek_search.csv")
    parser.add_argument("--output", default=PROJECT_ROOT / "results" / "alpha158_lgbm_biweek_gate_scan.csv")
    parser.add_argument("--limit-base", type=int, default=10)
    parser.add_argument("--max-topk", type=int, default=10)
    parser.add_argument("--max-factors", type=int, default=8)
    parser.add_argument("--universe", default="csi300")
    parser.add_argument("--train-start", default="2019-01-01")
    parser.add_argument("--train-end", default="2021-12-31")
    parser.add_argument("--oos-start", default="2022-01-01")
    parser.add_argument("--target-annual", type=float, default=0.20)
    parser.add_argument("--target-max-drawdown", type=float, default=0.10)
    parser.add_argument("--min-market-cap", type=float, default=50.0)
    parser.add_argument("--exclude-new-days", type=int, default=120)
    parser.add_argument("--sticky", type=int, default=5)
    parser.add_argument("--buffer", type=int, default=20)
    parser.add_argument("--churn-limit", type=int, default=3)
    parser.add_argument("--ma-windows", default="20,40,60,80,120")
    parser.add_argument(
        "--stock-pct-triplets",
        default="0.90/0.60/0.30,0.95/0.50/0.10,1.00/0.50/0.00,0.90/0.40/0.00,0.88/0.30/0.00",
        help="Comma-separated strong/mixed/weak stock pct triplets.",
    )
    parser.add_argument("--bond-annual-return", type=float, default=0.03)
    parser.add_argument("--qlib-dir", default=PROJECT_ROOT / "data" / "qlib_data" / "cn_data")
    parser.add_argument("--index-parquet", default=PROJECT_ROOT / "data" / "tushare" / "index_daily.parquet")
    parser.add_argument("--csi300-ts-code", default="000300.SH")
    parser.add_argument("--csi300-symbol", default="sh000300")
    parser.add_argument("--csi500-symbol", default="sz000905")
    parser.add_argument("--force-index-bin", action="store_true")
    parser.add_argument("--buy-commission-rate", type=float, default=0.0003)
    parser.add_argument("--sell-commission-rate", type=float, default=0.0003)
    parser.add_argument("--sell-stamp-tax-rate", type=float, default=0.0010)
    parser.add_argument("--min-buy-commission", type=float, default=5.0)
    parser.add_argument("--min-sell-commission", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--impact-bps", type=float, default=5.0)
    parser.add_argument("--block-limit-up-buy", action="store_true", default=True)
    parser.add_argument("--block-limit-down-sell", action="store_true", default=True)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    out = scan(args)
    ok = out[out["status"].eq("ok")].copy() if not out.empty else pd.DataFrame()
    passed = ok[ok["passed"].astype(bool)] if "passed" in ok.columns else pd.DataFrame()
    print("\n[SUMMARY]")
    print(f"Rows: {len(out)}  OK: {len(ok)}  Passed: {len(passed)}")
    cols = [
        "base_tag",
        "factor_names",
        "topk",
        "ma_window",
        "strong_stock_pct",
        "mixed_stock_pct",
        "weak_stock_pct",
        "annual_return",
        "max_drawdown",
        "sharpe_ratio",
        "oos_annual",
        "score",
    ]
    if not ok.empty:
        print(ok[cols].head(20).to_string(index=False))
    print(f"\n[OK] Results saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
