#!/usr/bin/env python3
"""扫描市场趋势释放参数，复用已有 base_return，不重新训练模型。"""

from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.modeling.portfolio_overlay import (  # noqa: E402
    OverlayConfig,
    compute_overlay_frame,
    summarize_overlay,
)


@dataclass(frozen=True)
class OverlayReturns:
    base_returns: pd.Series
    bond_returns: pd.Series


def load_overlay_returns(path: Path | str) -> OverlayReturns:
    """从已有 overlay_results.csv 读取底层组合收益和债券收益。"""
    csv_path = Path(path)
    frame = pd.read_csv(csv_path, parse_dates=["date"])
    if "base_return" not in frame.columns:
        raise ValueError(f"{csv_path} 缺少 base_return 列")

    frame = frame.sort_values("date").drop_duplicates("date").set_index("date")
    base_returns = frame["base_return"].astype(float)
    if "bond_return" in frame.columns:
        bond_returns = frame["bond_return"].astype(float).reindex(base_returns.index).fillna(0.0)
    else:
        bond_returns = pd.Series(0.03 / 252, index=base_returns.index, dtype=float)
    return OverlayReturns(base_returns=base_returns, bond_returns=bond_returns)


def load_market_returns(
    parquet_path: Path | str = PROJECT_ROOT / "data" / "tushare" / "index_daily.parquet",
    index_code: str = "000300.SH",
) -> pd.Series:
    frame = pd.read_parquet(parquet_path)
    if "ts_code" in frame.columns:
        frame = frame[frame["ts_code"] == index_code].copy()
    if frame.empty:
        raise ValueError(f"没有找到指数行情: {index_code}")

    if "trade_date" in frame.columns:
        dates = pd.to_datetime(frame["trade_date"].astype(str))
    elif "date" in frame.columns:
        dates = pd.to_datetime(frame["date"])
    else:
        raise ValueError("指数行情缺少 trade_date/date 列")
    if "close" not in frame.columns:
        raise ValueError("指数行情缺少 close 列")

    close = (
        frame.assign(date=dates)
        .sort_values("date")
        .drop_duplicates("date")
        .set_index("date")["close"]
        .astype(float)
    )
    return close.pct_change().fillna(0.0)


def _overlay_config(base_overlay: dict, market_params: dict) -> OverlayConfig:
    return OverlayConfig(
        target_vol=base_overlay.get("target_vol"),
        vol_lookback=int(base_overlay.get("vol_lookback", 20)),
        dd_soft=base_overlay.get("dd_soft"),
        dd_hard=base_overlay.get("dd_hard"),
        soft_exposure=float(base_overlay.get("soft_exposure", 0.95)),
        hard_exposure=float(base_overlay.get("hard_exposure", 0.80)),
        trend_lookback=int(base_overlay.get("trend_lookback", 0)),
        trend_exposure=float(base_overlay.get("trend_exposure", 0.90)),
        market_trend_lookback=int(market_params.get("market_trend_lookback", 0)),
        market_trend_min_return=float(market_params.get("market_trend_min_return", 0.0)),
        market_trend_exposure_floor=float(market_params.get("market_trend_exposure_floor", 0.0)),
        market_trend_max_strategy_drawdown=market_params.get("market_trend_max_strategy_drawdown"),
        exposure_min=float(base_overlay.get("exposure_min", 0.0)),
        exposure_max=float(base_overlay.get("exposure_max", 1.0)),
    )


def _evaluate_segment(
    segment: OverlayReturns,
    config: OverlayConfig,
    market_returns: pd.Series,
) -> dict:
    frame = compute_overlay_frame(
        segment.base_returns,
        segment.bond_returns,
        config,
        market_returns=market_returns,
    )
    metrics = summarize_overlay(frame)
    metrics["market_risk_on_days"] = int(frame["market_risk_on"].sum())
    metrics["market_risk_on_ratio"] = float(frame["market_risk_on"].mean())
    return metrics


def evaluate_candidate(
    tag: str,
    base_overlay: dict,
    market_params: dict,
    segments: dict[str, OverlayReturns],
    market_returns: pd.Series,
) -> dict:
    config = _overlay_config(base_overlay, market_params)
    row = {"tag": tag, **market_params}
    for name, segment in segments.items():
        metrics = _evaluate_segment(segment, config, market_returns)
        for key, value in metrics.items():
            row[f"{name}_{key}"] = value
    row["score"] = score_candidate(row)
    return row


def score_candidate(row: dict) -> float:
    valid_ann = float(row.get("valid_annual_return", 0.0))
    holdout_ann = float(row.get("holdout_annual_return", 0.0))
    full_ann = float(row.get("full_annual_return", 0.0))
    valid_sharpe = float(row.get("valid_sharpe_ratio", 0.0))
    holdout_sharpe = float(row.get("holdout_sharpe_ratio", 0.0))
    full_dd = float(row.get("full_max_drawdown", 0.0))
    valid_dd = float(row.get("valid_max_drawdown", 0.0))
    holdout_dd = float(row.get("holdout_max_drawdown", 0.0))

    score = valid_ann * 1.20 + holdout_ann * 0.80 + full_ann * 0.35
    score += valid_sharpe * 0.08 + holdout_sharpe * 0.08
    score -= _drawdown_penalty(valid_dd, 0.10)
    score -= _drawdown_penalty(holdout_dd, 0.10)
    score -= _drawdown_penalty(full_dd, 0.12)
    return score


def _drawdown_penalty(drawdown: float, cap: float) -> float:
    excess = abs(min(float(drawdown), 0.0)) - float(cap)
    return max(excess, 0.0) * 4.0


def generate_market_param_grid(
    lookbacks: Iterable[int],
    min_returns: Iterable[float],
    exposure_floors: Iterable[float],
    max_drawdowns: Iterable[float],
    include_baseline: bool = True,
) -> list[dict]:
    params = []
    if include_baseline:
        params.append(
            {
                "market_trend_lookback": 0,
                "market_trend_min_return": 0.0,
                "market_trend_exposure_floor": 0.0,
                "market_trend_max_strategy_drawdown": None,
            }
        )
    for lookback, min_return, floor, max_dd in itertools.product(
        lookbacks,
        min_returns,
        exposure_floors,
        max_drawdowns,
    ):
        params.append(
            {
                "market_trend_lookback": int(lookback),
                "market_trend_min_return": float(min_return),
                "market_trend_exposure_floor": float(floor),
                "market_trend_max_strategy_drawdown": float(max_dd),
            }
        )
    return params


def load_base_overlay(config_path: Path | str) -> dict:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    return dict(cfg.get("overlay", {}))


def _parse_float_list(values: list[str]) -> list[float]:
    return [float(value) for value in values]


def _parse_int_list(values: list[str]) -> list[int]:
    return [int(value) for value in values]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="config/models/push25_combo_only_k8d2_very_tight.yaml")
    parser.add_argument("--full-overlay-csv", default="results/model_signals/push25_combo_only_k8d2_very_tight/overlay_results.csv")
    parser.add_argument(
        "--valid-overlay-csv",
        default=(
            "results/model_signals/validation_runs/three_window_eval_bull_release_20260424/"
            "push25_combo_only_k8d2_very_tight__valid/overlay_results.csv"
        ),
    )
    parser.add_argument(
        "--holdout-overlay-csv",
        default=(
            "results/model_signals/validation_runs/three_window_eval_bull_release_20260424/"
            "push25_combo_only_k8d2_very_tight__holdout/overlay_results.csv"
        ),
    )
    parser.add_argument("--market-parquet", default="data/tushare/index_daily.parquet")
    parser.add_argument("--market-index-code", default="000300.SH")
    parser.add_argument("--lookbacks", nargs="+", default=["10", "20", "40", "60"])
    parser.add_argument("--min-returns", nargs="+", default=["0.01", "0.02", "0.03", "0.04"])
    parser.add_argument("--exposure-floors", nargs="+", default=["0.40", "0.45", "0.50", "0.55", "0.60"])
    parser.add_argument("--max-drawdowns", nargs="+", default=["0.05", "0.06", "0.08", "0.10"])
    parser.add_argument(
        "--output-csv",
        default=(
            "results/model_signals/validation_runs/three_window_eval_bull_release_20260424/"
            "market_trend_release_sweep.csv"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_overlay = load_base_overlay(PROJECT_ROOT / args.base_config)
    segments = {
        "full": load_overlay_returns(PROJECT_ROOT / args.full_overlay_csv),
        "valid": load_overlay_returns(PROJECT_ROOT / args.valid_overlay_csv),
        "holdout": load_overlay_returns(PROJECT_ROOT / args.holdout_overlay_csv),
    }
    market_returns = load_market_returns(PROJECT_ROOT / args.market_parquet, args.market_index_code)
    param_grid = generate_market_param_grid(
        _parse_int_list(args.lookbacks),
        _parse_float_list(args.min_returns),
        _parse_float_list(args.exposure_floors),
        _parse_float_list(args.max_drawdowns),
    )

    rows = [
        evaluate_candidate(
            tag=f"market_release_{idx:04d}",
            base_overlay=base_overlay,
            market_params=params,
            segments=segments,
            market_returns=market_returns,
        )
        for idx, params in enumerate(param_grid)
    ]
    result = pd.DataFrame(rows).sort_values(
        ["score", "valid_annual_return", "holdout_annual_return"],
        ascending=[False, False, False],
    )
    output_csv = PROJECT_ROOT / args.output_csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)

    print(f"[OK] market trend release sweep -> {output_csv}")
    print(f"[INFO] candidates: {len(result)}")
    columns = [
        "tag",
        "score",
        "market_trend_lookback",
        "market_trend_min_return",
        "market_trend_exposure_floor",
        "market_trend_max_strategy_drawdown",
        "valid_annual_return",
        "valid_max_drawdown",
        "holdout_annual_return",
        "holdout_max_drawdown",
        "full_annual_return",
        "full_max_drawdown",
    ]
    with pd.option_context("display.max_columns", None, "display.width", 180):
        print(result[columns].head(20).to_string(index=False, formatters=_formatters(columns)))


def _formatters(columns: list[str]) -> dict[str, object]:
    percent_cols = [column for column in columns if column.endswith(("return", "drawdown"))]
    formatters: dict[str, object] = {column: (lambda value: f"{float(value):.2%}") for column in percent_cols}
    if "score" in columns:
        formatters["score"] = lambda value: f"{float(value):.4f}"
    return formatters


if __name__ == "__main__":
    main()
