#!/usr/bin/env python3
"""在现有策略收益上叠加轻量 CSI300 对冲。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.position import MarketPositionController


BASELINES = [
    (
        "qvf_biweek_k3_gate_hotter_h20_mv120",
        "results/backtest_qvf_biweek_k3_gate_hotter_h20_mv120_historical_csi300_20260417_015620.csv",
    ),
    (
        "qvf_biweek_k3_fixed84_h20_mv120",
        "results/backtest_qvf_biweek_k3_fixed84_h20_mv120_historical_csi300_20260417_015809.csv",
    ),
    (
        "qvf_biweek_k3_fixed86_h20_mv120",
        "results/backtest_qvf_biweek_k3_fixed86_h20_mv120_historical_csi300_20260417_015742.csv",
    ),
]


HEDGE_PRESETS = [
    ("hedge_05_10_15", 0.05, 0.10, 0.15),
    ("hedge_08_12_18", 0.08, 0.12, 0.18),
    ("hedge_10_15_20", 0.10, 0.15, 0.20),
    ("hedge_12_18_25", 0.12, 0.18, 0.25),
]


def load_strategy_returns(path: str) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    df.index = pd.to_datetime(df.index)
    return df["return"].astype(float).sort_index()


def load_market_frame() -> pd.DataFrame:
    ctrl = MarketPositionController()
    ctrl.load_market_data()
    close = ctrl.csi300_close.astype(float).sort_index()
    df = pd.DataFrame({"close": close})
    df["market_ret"] = df["close"].pct_change().fillna(0.0)
    df["ma20"] = df["close"].rolling(20, min_periods=1).mean()
    df["ma60"] = df["close"].rolling(60, min_periods=1).mean()
    peak120 = df["close"].rolling(120, min_periods=1).max()
    df["dd120"] = df["close"] / peak120 - 1.0
    return df


def compute_hedged_frame(base_returns: pd.Series, market: pd.DataFrame, weak_hedge: float, bear_hedge: float, dd_hedge: float) -> pd.DataFrame:
    aligned = market.reindex(base_returns.index).copy()
    aligned["base_return"] = base_returns
    aligned = aligned.ffill().fillna(0.0)

    hedge = []
    overlay_returns = []
    nav = 1.0
    navs = []
    peaks = []

    for i, (dt, row) in enumerate(aligned.iterrows()):
        if i == 0:
            current_hedge = 0.0
        else:
            prev = aligned.iloc[i - 1]
            current_hedge = 0.0
            if prev["close"] < prev["ma20"]:
                current_hedge = max(current_hedge, weak_hedge)
            if prev["close"] < prev["ma60"]:
                current_hedge = max(current_hedge, bear_hedge)
            if prev["dd120"] <= -0.10:
                current_hedge = max(current_hedge, dd_hedge)

        ret = float(row["base_return"]) - current_hedge * float(row["market_ret"])
        nav *= 1.0 + ret
        overlay_returns.append(ret)
        hedge.append(current_hedge)
        navs.append(nav)
        peaks.append(max(nav if not peaks else peaks[-1], nav))
    frame = aligned.copy()
    frame["hedge_pct"] = hedge
    frame["overlay_return"] = overlay_returns
    frame["portfolio_value"] = navs
    frame["drawdown"] = pd.Series(navs, index=aligned.index) / pd.Series(peaks, index=aligned.index) - 1.0
    return frame


def summarize(frame: pd.DataFrame) -> dict:
    nav = frame["portfolio_value"]
    ret = frame["overlay_return"]
    days = int((nav.index[-1] - nav.index[0]).days)
    annual_return = float(nav.iloc[-1]) ** (365 / days) - 1 if days > 0 and float(nav.iloc[-1]) > 0 else 0.0
    sharpe = float(ret.mean() / ret.std(ddof=0) * np.sqrt(252)) if ret.std(ddof=0) > 0 else 0.0
    max_drawdown = float((nav / nav.cummax() - 1.0).min())
    return {
        "annual_return": annual_return,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe,
        "avg_hedge": float(frame["hedge_pct"].mean()),
    }


def main():
    out_dir = (PROJECT_ROOT / "results" / "market_hedge_overlay_search").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    market = load_market_frame()
    rows = []
    for baseline_name, path in BASELINES:
        base_returns = load_strategy_returns(path)
        for overlay_name, weak_hedge, bear_hedge, dd_hedge in HEDGE_PRESETS:
            frame = compute_hedged_frame(base_returns, market, weak_hedge, bear_hedge, dd_hedge)
            summary = summarize(frame)
            summary.update(
                {
                    "baseline": baseline_name,
                    "overlay": overlay_name,
                    "weak_hedge": weak_hedge,
                    "bear_hedge": bear_hedge,
                    "dd_hedge": dd_hedge,
                    "source_file": path,
                }
            )
            rows.append(summary)

    df = pd.DataFrame(rows).sort_values(["annual_return", "max_drawdown"], ascending=[False, False])
    csv_path = out_dir / "summary.csv"
    df.to_csv(csv_path, index=False)
    (out_dir / "summary.json").write_text(
        json.dumps(
            {"summary_csv": str(csv_path), "best": df.iloc[0].to_dict() if not df.empty else {}},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[OK] Market hedge search saved: {csv_path}")


if __name__ == "__main__":
    main()
