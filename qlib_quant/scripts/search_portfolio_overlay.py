#!/usr/bin/env python3
"""在现有回测结果上搜索组合级 overlay 参数。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.backtest.qlib_engine import _load_bond_etf_returns
from modules.modeling.portfolio_overlay import OverlayConfig, compute_overlay_frame, summarize_overlay


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
        "qvf_biweek_k3_gate_active_h20_mv120",
        "results/backtest_qvf_biweek_k3_gate_active_h20_mv120_historical_csi300_20260417_003325.csv",
    ),
    (
        "qvf_biweek_k3_gate_hot_h20_mv120",
        "results/backtest_qvf_biweek_k3_gate_hot_h20_mv120_historical_csi300_20260417_003338.csv",
    ),
]


OVERLAYS = {
    "baseline": OverlayConfig(),
    "dd_soft_4_8": OverlayConfig(dd_soft=0.04, dd_hard=0.08, soft_exposure=0.95, hard_exposure=0.85),
    "dd_soft_5_10": OverlayConfig(dd_soft=0.05, dd_hard=0.10, soft_exposure=0.95, hard_exposure=0.80),
    "dd_med_6_10": OverlayConfig(dd_soft=0.06, dd_hard=0.10, soft_exposure=0.90, hard_exposure=0.75),
    "vol22": OverlayConfig(target_vol=0.22, vol_lookback=20),
    "vol24": OverlayConfig(target_vol=0.24, vol_lookback=20),
    "vol26": OverlayConfig(target_vol=0.26, vol_lookback=20),
    "vol24_dd_light": OverlayConfig(target_vol=0.24, vol_lookback=20, dd_soft=0.04, dd_hard=0.08, soft_exposure=0.95, hard_exposure=0.85),
    "vol24_dd_med": OverlayConfig(target_vol=0.24, vol_lookback=20, dd_soft=0.05, dd_hard=0.10, soft_exposure=0.90, hard_exposure=0.80),
    "trend20_90": OverlayConfig(trend_lookback=20, trend_exposure=0.90),
    "trend20_85": OverlayConfig(trend_lookback=20, trend_exposure=0.85),
    "trend20_dd": OverlayConfig(trend_lookback=20, trend_exposure=0.90, dd_soft=0.04, dd_hard=0.08, soft_exposure=0.95, hard_exposure=0.85),
    "vol24_trend20": OverlayConfig(target_vol=0.24, vol_lookback=20, trend_lookback=20, trend_exposure=0.90),
    "vol24_trend20_dd": OverlayConfig(target_vol=0.24, vol_lookback=20, trend_lookback=20, trend_exposure=0.90, dd_soft=0.04, dd_hard=0.08, soft_exposure=0.95, hard_exposure=0.85),
    "trend20_90_lev105": OverlayConfig(trend_lookback=20, trend_exposure=0.90, exposure_max=1.05),
    "trend20_90_lev110": OverlayConfig(trend_lookback=20, trend_exposure=0.90, exposure_max=1.10),
    "trend20_dd_lev105": OverlayConfig(trend_lookback=20, trend_exposure=0.90, dd_soft=0.04, dd_hard=0.08, soft_exposure=0.95, hard_exposure=0.85, exposure_max=1.05),
    "dd_soft_4_8_lev105": OverlayConfig(dd_soft=0.04, dd_hard=0.08, soft_exposure=0.95, hard_exposure=0.85, exposure_max=1.05),
    "vol28_lev110": OverlayConfig(target_vol=0.28, vol_lookback=20, exposure_max=1.10),
    "vol28_trend20_lev110": OverlayConfig(target_vol=0.28, vol_lookback=20, trend_lookback=20, trend_exposure=0.90, exposure_max=1.10),
    "vol30_trend20_dd_lev110": OverlayConfig(target_vol=0.30, vol_lookback=20, trend_lookback=20, trend_exposure=0.90, dd_soft=0.04, dd_hard=0.08, soft_exposure=0.95, hard_exposure=0.85, exposure_max=1.10),
}


def load_returns(path: str) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    df.index = pd.to_datetime(df.index)
    return df["return"].astype(float).sort_index()


def main():
    out_dir = (PROJECT_ROOT / "results" / "portfolio_overlay_search").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    bond = _load_bond_etf_returns()
    bond_returns = bond if bond is not None else 0.03 / 252

    rows = []
    for baseline_name, path in BASELINES:
        base_returns = load_returns(path)
        for overlay_name, config in OVERLAYS.items():
            frame = compute_overlay_frame(base_returns, bond_returns, config)
            summary = summarize_overlay(frame)
            summary.update(
                {
                    "baseline": baseline_name,
                    "overlay": overlay_name,
                    "source_file": path,
                }
            )
            rows.append(summary)

    result_df = pd.DataFrame(rows).sort_values(
        ["annual_return", "max_drawdown", "sharpe_ratio"],
        ascending=[False, False, False],
    )
    csv_path = out_dir / "summary.csv"
    result_df.to_csv(csv_path, index=False)
    payload = {
        "summary_csv": str(csv_path),
        "best_by_annual_return": result_df.iloc[0].to_dict() if not result_df.empty else {},
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Overlay search saved: {csv_path}")


if __name__ == "__main__":
    main()
