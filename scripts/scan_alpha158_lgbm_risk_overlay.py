#!/usr/bin/env python3
"""Scan risk overlays for costed Alpha158 LGBM backtest returns.

The base return files are produced by the formal Qlib backtest, so they already
include commission, stamp tax, slippage, impact, and tradability constraints.
This script does not retrain models. It scans no-lookahead exposure rules on top
of those costed daily returns and can restrict exposure changes to biweekly
rebalance dates.
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

from core.selection import compute_rebalance_dates


OUTPUT_COLUMNS = [
    "base_tag",
    "factor_count",
    "factor_names",
    "topk",
    "base_annual",
    "base_max_dd",
    "base_sharpe",
    "exposure_freq",
    "soft_dd",
    "hard_dd",
    "soft_exposure",
    "hard_exposure",
    "trend_lookback",
    "min_trend_return",
    "trend_exposure",
    "vol_lookback",
    "target_vol",
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


def metrics(returns: pd.Series | np.ndarray, index=None) -> dict[str, float]:
    if isinstance(returns, pd.Series):
        rets = returns.dropna().astype(float)
    else:
        rets = pd.Series(np.asarray(returns, dtype=float), index=index).dropna()
    if rets.empty:
        return {"annual": 0.0, "max_dd": 0.0, "sharpe": 0.0, "total": 0.0}

    nav = (1.0 + rets).cumprod()
    days = (nav.index[-1] - nav.index[0]).days
    terminal = float(nav.iloc[-1])
    annual = terminal ** (365.0 / days) - 1.0 if days > 0 and terminal > 0 else -1.0
    max_dd = float((nav / nav.cummax() - 1.0).min())
    std = float(rets.std())
    sharpe = float(rets.mean() / std * np.sqrt(252.0)) if std > 0 else 0.0
    return {
        "annual": float(annual),
        "max_dd": max_dd,
        "sharpe": sharpe,
        "total": float(terminal - 1.0),
    }


def score_row(annual: float, max_dd: float, target_annual: float, target_max_dd: float) -> float:
    dd_excess = max(0.0, abs(min(max_dd, 0.0)) - target_max_dd)
    annual_gap = max(0.0, target_annual - annual)
    passed = annual >= target_annual and max_dd >= -target_max_dd
    return float((1.0 if passed else 0.0) + annual - 3.0 * dd_excess - annual_gap)


def trailing_total_returns(raw: np.ndarray, windows: list[int]) -> dict[int, np.ndarray]:
    log_rets = np.log1p(np.clip(raw, -0.999999, None))
    prefix = np.concatenate([[0.0], np.cumsum(log_rets)])
    out = {}
    n = len(raw)
    idx = np.arange(n)
    for window in windows:
        if window <= 0:
            out[window] = np.zeros(n, dtype=float)
            continue
        start = np.maximum(0, idx - window)
        vals = np.exp(prefix[idx] - prefix[start]) - 1.0
        vals[idx == 0] = 0.0
        out[window] = vals
    return out


def trailing_volatility(raw: np.ndarray, windows: list[int]) -> dict[int, np.ndarray]:
    prefix = np.concatenate([[0.0], np.cumsum(raw)])
    prefix2 = np.concatenate([[0.0], np.cumsum(raw * raw)])
    out = {}
    n = len(raw)
    idx = np.arange(n)
    for window in windows:
        if window <= 0:
            out[window] = np.zeros(n, dtype=float)
            continue
        start = np.maximum(0, idx - window)
        count = idx - start
        sums = prefix[idx] - prefix[start]
        sums2 = prefix2[idx] - prefix2[start]
        mean = np.divide(sums, count, out=np.zeros(n, dtype=float), where=count > 0)
        var = np.divide(sums2, count, out=np.zeros(n, dtype=float), where=count > 0) - mean * mean
        out[window] = np.sqrt(np.maximum(var, 0.0))
    return out


def shifted_raw_drawdown(raw: np.ndarray) -> np.ndarray:
    nav = np.cumprod(1.0 + raw)
    peak = np.maximum.accumulate(nav)
    dd = nav / peak - 1.0
    shifted = np.empty_like(dd)
    shifted[0] = 0.0
    shifted[1:] = dd[:-1]
    return shifted


def hold_exposure_on_frequency(
    dates: pd.DatetimeIndex,
    signal: np.ndarray,
    freq: str,
    initial_exposure: float,
) -> np.ndarray:
    if freq == "day":
        return signal
    rebalance_dates = set(compute_rebalance_dates(pd.Series(dates), freq=freq))
    held = np.empty_like(signal)
    current = float(initial_exposure)
    for i, date in enumerate(dates):
        if pd.Timestamp(date) in rebalance_dates:
            current = float(signal[i])
        held[i] = current
    return held


def apply_overlay(
    raw: np.ndarray,
    dates: pd.DatetimeIndex,
    raw_dd_prev: np.ndarray,
    trend_returns: dict[int, np.ndarray],
    vols: dict[int, np.ndarray],
    cfg: dict,
    cash_daily_return: float,
) -> tuple[np.ndarray, np.ndarray]:
    exposure = np.full(len(raw), 1.0, dtype=float)

    exposure = np.where(
        raw_dd_prev <= cfg["hard_dd"],
        cfg["hard_exposure"],
        np.where(raw_dd_prev <= cfg["soft_dd"], np.minimum(exposure, cfg["soft_exposure"]), exposure),
    )

    trend = trend_returns[cfg["trend_lookback"]]
    exposure = np.where(
        trend < cfg["min_trend_return"],
        np.minimum(exposure, cfg["trend_exposure"]),
        exposure,
    )

    vol_lookback = cfg["vol_lookback"]
    target_vol = cfg["target_vol"]
    if vol_lookback > 0 and target_vol > 0:
        daily_target = target_vol / np.sqrt(252.0)
        vol = vols[vol_lookback]
        cap = np.divide(daily_target, vol, out=np.ones_like(vol), where=vol > 0)
        exposure = np.minimum(exposure, np.clip(cap, 0.0, 1.0))

    exposure = np.clip(exposure, 0.0, 1.0)
    exposure = hold_exposure_on_frequency(dates, exposure, cfg["exposure_freq"], 1.0)
    adjusted = raw * exposure + (1.0 - exposure) * cash_daily_return
    return adjusted, exposure


def iter_configs(args):
    soft_dds = parse_floats(args.soft_dds)
    hard_dds = parse_floats(args.hard_dds)
    soft_exposures = parse_floats(args.soft_exposures)
    hard_exposures = parse_floats(args.hard_exposures)
    trend_lookbacks = parse_ints(args.trend_lookbacks)
    min_trend_returns = parse_floats(args.min_trend_returns)
    trend_exposures = parse_floats(args.trend_exposures)
    vol_lookbacks = parse_ints(args.vol_lookbacks)
    target_vols = parse_floats(args.target_vols)
    exposure_freqs = [x.strip() for x in args.exposure_freqs.split(",") if x.strip()]

    for (
        exposure_freq,
        soft_dd,
        hard_dd,
        soft_exposure,
        hard_exposure,
        trend_lookback,
        min_trend_return,
        trend_exposure,
        vol_lookback,
        target_vol,
    ) in itertools.product(
        exposure_freqs,
        soft_dds,
        hard_dds,
        soft_exposures,
        hard_exposures,
        trend_lookbacks,
        min_trend_returns,
        trend_exposures,
        vol_lookbacks,
        target_vols,
    ):
        if not hard_dd < soft_dd < 0:
            continue
        if hard_exposure > soft_exposure:
            continue
        if vol_lookback <= 0 and target_vol > 0:
            continue
        if vol_lookback > 0 and target_vol <= 0:
            continue
        yield {
            "exposure_freq": exposure_freq,
            "soft_dd": soft_dd,
            "hard_dd": hard_dd,
            "soft_exposure": soft_exposure,
            "hard_exposure": hard_exposure,
            "trend_lookback": trend_lookback,
            "min_trend_return": min_trend_return,
            "trend_exposure": trend_exposure,
            "vol_lookback": vol_lookback,
            "target_vol": target_vol,
        }


def load_base_candidates(args) -> pd.DataFrame:
    df = pd.read_csv(args.search_csv)
    df = df[df["status"].eq("ok")].copy()
    df = df[pd.to_numeric(df["topk"], errors="coerce") <= args.max_topk].copy()
    df = df[pd.to_numeric(df["factor_count"], errors="coerce") < args.max_factors].copy()
    if args.base_tags:
        tags = {x.strip() for x in args.base_tags.split(",") if x.strip()}
        df = df[df["tag"].astype(str).isin(tags)].copy()

    frames = []
    for sort_col, ascending in [("annual_return", False), ("rank_score", False), ("max_drawdown", False)]:
        if sort_col in df.columns:
            frames.append(df.sort_values(sort_col, ascending=ascending).head(args.limit_base))
    if frames:
        df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["tag"])
    if args.limit_base > 0:
        df = df.head(args.limit_base)
    return df


def scan(args) -> pd.DataFrame:
    args.search_csv = Path(args.search_csv)
    args.output = Path(args.output)
    candidates = load_base_candidates(args)
    configs = list(iter_configs(args))
    if candidates.empty:
        raise RuntimeError("No base candidates found.")
    if not configs:
        raise RuntimeError("No overlay configs generated.")

    trend_windows = sorted(set(parse_ints(args.trend_lookbacks)))
    vol_windows = sorted(set(x for x in parse_ints(args.vol_lookbacks) if x > 0))
    print(f"[INFO] Base candidates: {len(candidates)}")
    print(f"[INFO] Risk overlay configs: {len(configs)}")

    rows = []
    for _, candidate in candidates.iterrows():
        results_file = Path(str(candidate["results_file"]))
        if not results_file.is_absolute():
            results_file = PROJECT_ROOT / results_file
        if not results_file.exists():
            print(f"[WARN] Missing results file: {results_file}")
            continue

        raw_series = pd.read_csv(results_file, parse_dates=["date"]).set_index("date")["return"]
        raw_series = raw_series.dropna().astype(float)
        raw = raw_series.to_numpy(dtype=float)
        dates = pd.DatetimeIndex(raw_series.index)
        base = metrics(raw_series)
        raw_dd_prev = shifted_raw_drawdown(raw)
        trend_returns = trailing_total_returns(raw, trend_windows)
        vols = trailing_volatility(raw, vol_windows) if vol_windows else {}
        best_preview = None
        best_score = -1e9

        for cfg in configs:
            adjusted, exposure = apply_overlay(
                raw=raw,
                dates=dates,
                raw_dd_prev=raw_dd_prev,
                trend_returns=trend_returns,
                vols=vols,
                cfg=cfg,
                cash_daily_return=args.cash_annual_return / 252.0,
            )
            m = metrics(adjusted, index=dates)
            passed = bool(
                m["annual"] >= args.target_annual
                and m["max_dd"] >= -args.target_max_drawdown
            )
            score = score_row(m["annual"], m["max_dd"], args.target_annual, args.target_max_drawdown)
            row = {
                "base_tag": candidate["tag"],
                "factor_count": int(candidate["factor_count"]),
                "factor_names": candidate["factor_names"],
                "topk": int(candidate["topk"]),
                "base_annual": base["annual"],
                "base_max_dd": base["max_dd"],
                "base_sharpe": base["sharpe"],
                **cfg,
                "annual_return": m["annual"],
                "max_drawdown": m["max_dd"],
                "sharpe_ratio": m["sharpe"],
                "total_return": m["total"],
                "avg_exposure": float(np.mean(exposure)),
                "min_exposure": float(np.min(exposure)),
                "active_days": int(np.sum(exposure > 0)),
                "passed": passed,
                "score": score,
                "results_file": str(results_file.relative_to(PROJECT_ROOT))
                if results_file.is_relative_to(PROJECT_ROOT)
                else str(results_file),
            }
            rows.append(row)
            if score > best_score:
                best_score = score
                best_preview = row

        if best_preview:
            print(
                f"  {candidate['tag']} best annual={best_preview['annual_return']:.2%} "
                f"dd={best_preview['max_drawdown']:.2%} avg_exp={best_preview['avg_exposure']:.1%} "
                f"freq={best_preview['exposure_freq']}"
            )

    out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not out.empty:
        out = out.sort_values(["passed", "score", "annual_return"], ascending=[False, False, False])
    out = out.reindex(columns=OUTPUT_COLUMNS)
    out.to_csv(args.output, index=False)

    print("\n[SUMMARY]")
    print(f"Rows: {len(out)}  Passed: {int(out['passed'].sum()) if not out.empty else 0}")
    if not out.empty:
        show_cols = [
            "base_tag",
            "factor_names",
            "topk",
            "annual_return",
            "max_drawdown",
            "sharpe_ratio",
            "avg_exposure",
            "exposure_freq",
            "soft_dd",
            "hard_dd",
            "soft_exposure",
            "hard_exposure",
            "trend_lookback",
            "min_trend_return",
            "trend_exposure",
            "vol_lookback",
            "target_vol",
        ]
        print(out[show_cols].head(args.show_top).to_string(index=False))
    print(f"\n[OK] Results saved: {args.output}")
    return out


def main():
    parser = argparse.ArgumentParser(description="Scan risk overlays for Alpha158 LGBM formal returns.")
    parser.add_argument("--search-csv", default=PROJECT_ROOT / "results" / "alpha158_lgbm_biweek_search.csv")
    parser.add_argument("--output", default=PROJECT_ROOT / "results" / "alpha158_lgbm_biweek_risk_overlay_scan.csv")
    parser.add_argument("--limit-base", type=int, default=40)
    parser.add_argument("--base-tags", default="")
    parser.add_argument("--max-topk", type=int, default=10)
    parser.add_argument("--max-factors", type=int, default=8)
    parser.add_argument("--target-annual", type=float, default=0.20)
    parser.add_argument("--target-max-drawdown", type=float, default=0.10)
    parser.add_argument("--cash-annual-return", type=float, default=0.03)
    parser.add_argument("--exposure-freqs", default="biweek")
    parser.add_argument("--soft-dds", default="-0.03,-0.05,-0.07")
    parser.add_argument("--hard-dds", default="-0.08,-0.10,-0.12,-0.15")
    parser.add_argument("--soft-exposures", default="0.50,0.70,0.85")
    parser.add_argument("--hard-exposures", default="0.00,0.15,0.30,0.50")
    parser.add_argument("--trend-lookbacks", default="10,20,40")
    parser.add_argument("--min-trend-returns", default="-0.02,0.00,0.02")
    parser.add_argument("--trend-exposures", default="0.25,0.50,0.75")
    parser.add_argument("--vol-lookbacks", default="0")
    parser.add_argument("--target-vols", default="0")
    parser.add_argument("--show-top", type=int, default=20)
    args = parser.parse_args()
    scan(args)


if __name__ == "__main__":
    main()
