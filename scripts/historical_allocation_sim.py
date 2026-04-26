#!/usr/bin/env python3
"""Historical simulation for dynamic main/backup strategy allocation."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Metrics:
    total_return: float
    annual_return: float
    max_drawdown: float
    sharpe_ratio: float
    volatility: float


def _pick_return_col(frame: pd.DataFrame) -> str:
    for col in ("overlay_return", "return"):
        if col in frame.columns:
            return col
    raise ValueError("CSV must contain `overlay_return` or `return` column")


def load_returns(path: Path | str) -> pd.Series:
    frame = pd.read_csv(path, parse_dates=["date"])
    col = _pick_return_col(frame)
    series = (
        frame.sort_values("date")
        .drop_duplicates("date")
        .set_index("date")[col]
        .astype(float)
    )
    series.index = pd.to_datetime(series.index)
    return series


def load_index_returns(
    path: Path | str = PROJECT_ROOT / "data" / "tushare" / "index_daily.parquet",
    index_code: str = "000300.SH",
) -> pd.Series:
    frame = pd.read_parquet(path)
    if "ts_code" in frame.columns:
        frame = frame[frame["ts_code"] == index_code].copy()
    if frame.empty:
        raise ValueError(f"Index not found: {index_code}")

    if "trade_date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["trade_date"].astype(str))
    elif "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"])
    else:
        raise ValueError("Index parquet must contain `trade_date` or `date`")

    close = (
        frame.sort_values("date")
        .drop_duplicates("date")
        .set_index("date")["close"]
        .astype(float)
    )
    return close.pct_change().fillna(0.0)


def calc_metrics(returns: pd.Series) -> Metrics:
    returns = pd.Series(returns, dtype=float).dropna()
    if returns.empty:
        return Metrics(0.0, 0.0, 0.0, 0.0, 0.0)

    nav = (1.0 + returns).cumprod()
    total_return = float(nav.iloc[-1] - 1.0)
    annual_return = float(nav.iloc[-1] ** (252.0 / len(nav)) - 1.0)
    drawdown = nav / nav.cummax() - 1.0
    max_drawdown = float(drawdown.min())
    vol = float(returns.std(ddof=0) * np.sqrt(252.0))
    sharpe = float(returns.mean() / returns.std(ddof=0) * np.sqrt(252.0)) if returns.std(ddof=0) > 0 else 0.0
    return Metrics(total_return, annual_return, max_drawdown, sharpe, vol)


def rolling_drawdown(returns: pd.Series, window: int) -> pd.Series:
    nav = (1.0 + returns).cumprod()
    rolling_peak = nav.rolling(window=window, min_periods=window).max()
    dd = nav / rolling_peak - 1.0
    return dd.fillna(0.0)


def period_metrics(returns: pd.Series) -> dict[str, float]:
    returns = pd.Series(returns, dtype=float).dropna()
    if returns.empty:
        return {"return": 0.0, "max_drawdown": 0.0, "sharpe": 0.0}
    nav = (1.0 + returns).cumprod()
    mdd = float((nav / nav.cummax() - 1.0).min())
    std = float(returns.std(ddof=0))
    sharpe = float(returns.mean() / std * np.sqrt(252.0)) if std > 0 else 0.0
    return {"return": float(nav.iloc[-1] - 1.0), "max_drawdown": mdd, "sharpe": sharpe}


def yearly_compare(strategy_returns: pd.Series, index_returns: pd.Series) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for year, group in strategy_returns.groupby(strategy_returns.index.year):
        bench = index_returns.reindex(group.index).fillna(0.0)
        sm = period_metrics(group)
        bm = period_metrics(bench)
        rows.append(
            {
                "year": int(year),
                "strategy_return": sm["return"],
                "strategy_max_drawdown": sm["max_drawdown"],
                "strategy_sharpe": sm["sharpe"],
                "benchmark_return": bm["return"],
                "benchmark_max_drawdown": bm["max_drawdown"],
                "benchmark_sharpe": bm["sharpe"],
                "excess_return": sm["return"] - bm["return"],
            }
        )
    return pd.DataFrame(rows).sort_values("year")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dynamic historical simulation for main/backup allocation")
    parser.add_argument(
        "--main-csv",
        default=(
            "results/model_signals/alpha158_search_runs/alpha158_small_factor_csi300_v3/"
            "alpha158_momentum_volume_k6_dd10_overlay/overlay_results.csv"
        ),
    )
    parser.add_argument(
        "--backup-csv",
        default="results/model_signals/push25_cq10_v3_vol_norm/overlay_results.csv",
    )
    parser.add_argument("--start-date", default="2019-01-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--index-parquet", default="data/tushare/index_daily.parquet")
    parser.add_argument("--index-code", default="000300.SH")
    parser.add_argument("--lookback-days", type=int, default=60)
    parser.add_argument("--stress-dd", type=float, default=0.08)
    parser.add_argument("--severe-dd", type=float, default=0.12)
    parser.add_argument("--recover-dd", type=float, default=0.06)
    parser.add_argument("--base-main-weight", type=float, default=0.60)
    parser.add_argument("--stress-main-weight", type=float, default=0.50)
    parser.add_argument("--severe-main-weight", type=float, default=0.40)
    parser.add_argument("--out-dir", default="results/analysis/phase1_final_main_backup_60_40")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    main_path = (PROJECT_ROOT / args.main_csv).resolve()
    backup_path = (PROJECT_ROOT / args.backup_csv).resolve()
    index_path = (PROJECT_ROOT / args.index_parquet).resolve()

    main_ret = load_returns(main_path)
    backup_ret = load_returns(backup_path)
    index_ret = load_index_returns(index_path, args.index_code)

    aligned_index = main_ret.index.intersection(backup_ret.index)
    aligned_index = aligned_index[aligned_index >= pd.Timestamp(args.start_date)]
    if args.end_date:
        aligned_index = aligned_index[aligned_index <= pd.Timestamp(args.end_date)]

    if len(aligned_index) == 0:
        raise ValueError("No overlapping dates after filtering")

    main_ret = main_ret.reindex(aligned_index).fillna(0.0)
    backup_ret = backup_ret.reindex(aligned_index).fillna(0.0)
    index_ret = index_ret.reindex(aligned_index).fillna(0.0)

    main_nav = (1.0 + main_ret).cumprod()
    main_dd = main_nav / main_nav.cummax() - 1.0
    main_rolling_dd = rolling_drawdown(main_ret, args.lookback_days)

    current_month = aligned_index[0].to_period("M")
    stress_mode = False
    severe_mode = False
    prev_main_weight: float | None = None
    weight_switches = 0
    rows = []

    for i, dt in enumerate(aligned_index):
        prev_dt = aligned_index[i - 1] if i > 0 else None

        if i == 0 or dt.to_period("M") != current_month:
            current_month = dt.to_period("M")
            if prev_dt is not None:
                stress_mode = bool(main_rolling_dd.loc[prev_dt] <= -abs(args.stress_dd))
            else:
                stress_mode = False

        if prev_dt is not None:
            prev_dd = float(main_dd.loc[prev_dt])
            if severe_mode and prev_dd >= -abs(args.recover_dd):
                severe_mode = False
            if (not severe_mode) and prev_dd <= -abs(args.severe_dd):
                severe_mode = True

        if severe_mode:
            main_w = float(args.severe_main_weight)
            regime = "severe"
        elif stress_mode:
            main_w = float(args.stress_main_weight)
            regime = "stress"
        else:
            main_w = float(args.base_main_weight)
            regime = "base"
        backup_w = 1.0 - main_w

        if prev_main_weight is None:
            prev_main_weight = main_w
        elif abs(main_w - prev_main_weight) > 1e-12:
            weight_switches += 1
            prev_main_weight = main_w

        ret = main_w * float(main_ret.loc[dt]) + backup_w * float(backup_ret.loc[dt])
        rows.append(
            {
                "date": dt,
                "main_return": float(main_ret.loc[dt]),
                "backup_return": float(backup_ret.loc[dt]),
                "main_weight": main_w,
                "backup_weight": backup_w,
                "regime": regime,
                "portfolio_return": ret,
                "main_drawdown": float(main_dd.loc[dt]),
                "main_rolling_drawdown": float(main_rolling_dd.loc[dt]),
            }
        )

    sim = pd.DataFrame(rows).sort_values("date").set_index("date")
    sim["portfolio_value"] = (1.0 + sim["portfolio_return"]).cumprod()
    sim["drawdown"] = sim["portfolio_value"] / sim["portfolio_value"].cummax() - 1.0

    strat_metrics = calc_metrics(sim["portfolio_return"])
    bench_metrics = calc_metrics(index_ret)
    yearly = yearly_compare(sim["portfolio_return"], index_ret)

    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    daily_path = out_dir / "dynamic_sim_daily.csv"
    summary_path = out_dir / "dynamic_sim_summary.json"
    yearly_path = out_dir / "dynamic_sim_yearly_vs_hs300.csv"

    sim.reset_index().to_csv(daily_path, index=False)
    yearly.to_csv(yearly_path, index=False)

    summary = {
        "date_start": str(sim.index.min().date()),
        "date_end": str(sim.index.max().date()),
        "n_days": int(len(sim)),
        "main_csv": str(main_path),
        "backup_csv": str(backup_path),
        "index_code": args.index_code,
        "rules": {
            "lookback_days": int(args.lookback_days),
            "stress_dd": float(args.stress_dd),
            "severe_dd": float(args.severe_dd),
            "recover_dd": float(args.recover_dd),
            "base_main_weight": float(args.base_main_weight),
            "stress_main_weight": float(args.stress_main_weight),
            "severe_main_weight": float(args.severe_main_weight),
        },
        "strategy_metrics": asdict(strat_metrics),
        "benchmark_metrics": asdict(bench_metrics),
        "excess_annual_return": float(strat_metrics.annual_return - bench_metrics.annual_return),
        "weight_switches": int(weight_switches),
        "regime_days": sim["regime"].value_counts().to_dict(),
        "output_files": {
            "daily": str(daily_path),
            "summary": str(summary_path),
            "yearly": str(yearly_path),
        },
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[OK] dynamic daily -> {daily_path}")
    print(f"[OK] yearly vs hs300 -> {yearly_path}")
    print(f"[OK] summary -> {summary_path}")
    print(
        "[SUMMARY] "
        f"ann={strat_metrics.annual_return:.2%} "
        f"mdd={strat_metrics.max_drawdown:.2%} "
        f"sharpe={strat_metrics.sharpe_ratio:.3f} "
        f"weight_switches={weight_switches}"
    )


if __name__ == "__main__":
    main()
