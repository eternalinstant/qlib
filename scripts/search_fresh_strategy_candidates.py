#!/usr/bin/env python3
"""
Search fresh strategy candidates on the current local dataset.

Goal:
- ignore the repo's existing strategy names/configs
- search a small but purposeful grid around quality/value + trend/flow + low-vol
- report candidates with high annual return and controlled max drawdown

This search uses an approximate but realistic local backtest loop for speed.
The final winner should still be validated with the formal strategy engine.
"""

from __future__ import annotations

import argparse
import warnings

warnings.filterwarnings("ignore")

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from core.compute import compute_layer_score
from core.factors import FactorInfo, FactorRegistry
from core.position import MarketGatePositionController
from core.qlib_init import init_qlib, load_features_safe
from core.selection import (
    _fill_cross_sectional,
    _load_parquet_factors,
    compute_rebalance_dates,
    extract_topk,
)
from core.strategy import _FixedPositionController
from core.universe import filter_instruments
from modules.backtest.qlib_engine import _load_bond_etf_returns, _sum_symbol_returns


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_START_DATE = "2019-01-01"
DEFAULT_END_DATE = "2026-04-15"
INITIAL_CAPITAL = 1_000_000.0


FACTOR_SPECS = [
    FactorInfo("roa", "roa_fina", "ROA", "alpha", source="parquet"),
    FactorInfo("ocf_to_ev", "ocf_to_ev", "OCF/EV", "alpha", source="parquet"),
    FactorInfo("ebit_to_mv", "ebit_to_mv", "EBIT/MV", "alpha", source="parquet"),
    FactorInfo("retained_earnings", "retained_earnings", "retained earnings", "alpha", source="parquet"),
    FactorInfo("book_to_price", "book_to_market", "book to price", "alpha", source="parquet"),
    FactorInfo("total_mv_raw", "$total_mv", "raw mv", "alpha", source="qlib"),
    FactorInfo("total_mv_log", "Log($total_mv + 1)", "log mv", "alpha", source="qlib"),
    FactorInfo(
        "vol_std_20d",
        "Std(($close - Ref($close, 1)) / Ref($close, 1), 20)",
        "20d return vol",
        "risk",
        source="qlib",
        negate=True,
    ),
    FactorInfo("turnover_rate_f", "turnover_rate_f", "turnover rate", "risk", source="parquet", negate=True),
    FactorInfo(
        "vol_std_10d",
        "Std($volume, 10) / (Mean($volume, 10) + 1)",
        "volume std 10d",
        "risk",
        source="qlib",
        negate=True,
    ),
    FactorInfo(
        "amt_std_6d",
        "Std($amount, 6) / (Mean($amount, 6) + 1)",
        "amount std 6d",
        "risk",
        source="qlib",
        negate=True,
    ),
    FactorInfo(
        "vol_ema26_ratio",
        "$volume / (EMA($volume, 26) + 1)",
        "volume ema ratio",
        "risk",
        source="qlib",
        negate=True,
    ),
    FactorInfo("mom_20d", "$close / Ref($close, 20) - 1", "20d momentum", "enhance", source="qlib"),
    FactorInfo(
        "price_pos_52w",
        "($close - Min($close, 252)) / (Max($close, 252) - Min($close, 252) + 1e-8)",
        "52w price position",
        "enhance",
        source="qlib",
    ),
    FactorInfo(
        "bbi_momentum",
        "$close / ((Mean($close, 3) + Mean($close, 6) + Mean($close, 12) + Mean($close, 24)) / 4) - 1",
        "BBI momentum",
        "enhance",
        source="qlib",
    ),
    FactorInfo("net_mf_5d", "net_mf_amount_5d", "5d money flow", "enhance", source="parquet"),
    FactorInfo("net_mf_20d", "net_mf_amount_20d", "20d money flow", "enhance", source="parquet"),
    FactorInfo("smart_ratio", "smart_ratio_5d", "smart money ratio", "enhance", source="parquet"),
    FactorInfo(
        "close_to_high_60d",
        "$close / Max($close, 60) - 1",
        "close to 60d high",
        "enhance",
        source="qlib",
    ),
    FactorInfo(
        "sharpe_120d",
        "Mean(($close - Ref($close, 1)) / Ref($close, 1), 120) / "
        "(Std(($close - Ref($close, 1)) / Ref($close, 1), 120) + 1e-8)",
        "120d sharpe",
        "enhance",
        source="qlib",
    ),
    FactorInfo("ema20_dev", "$close / EMA($close, 20) - 1", "ema20 dev", "enhance", source="qlib"),
]


BUNDLES = {
    "quality_flow": {
        "alpha": ["roa", "ocf_to_ev", "ebit_to_mv"],
        "risk": ["vol_std_20d", "turnover_rate_f"],
        "enhance": ["mom_20d", "net_mf_5d", "smart_ratio"],
        "weight_options": [
            {"alpha": 0.25, "risk": 0.20, "enhance": 0.55},
            {"alpha": 0.30, "risk": 0.20, "enhance": 0.50},
        ],
    },
    "quality_trend": {
        "alpha": ["roa", "ocf_to_ev", "retained_earnings"],
        "risk": ["vol_std_20d", "turnover_rate_f"],
        "enhance": ["mom_20d", "price_pos_52w", "bbi_momentum"],
        "weight_options": [
            {"alpha": 0.30, "risk": 0.20, "enhance": 0.50},
            {"alpha": 0.25, "risk": 0.25, "enhance": 0.50},
        ],
    },
    "value_flow": {
        "alpha": ["book_to_price", "ocf_to_ev", "ebit_to_mv"],
        "risk": ["vol_std_20d", "turnover_rate_f"],
        "enhance": ["price_pos_52w", "net_mf_5d", "smart_ratio"],
        "weight_options": [
            {"alpha": 0.35, "risk": 0.20, "enhance": 0.45},
            {"alpha": 0.30, "risk": 0.25, "enhance": 0.45},
        ],
    },
    "closeonly_breakout": {
        "alpha": ["total_mv_log"],
        "risk": ["vol_std_10d", "amt_std_6d", "vol_ema26_ratio"],
        "enhance": ["close_to_high_60d", "sharpe_120d", "ema20_dev"],
        "weight_options": [
            {"alpha": 0.05, "risk": 0.25, "enhance": 0.70},
            {"alpha": 0.10, "risk": 0.20, "enhance": 0.70},
        ],
    },
}


POSITION_MODELS = {
    "fixed_88": {"model": "fixed", "stock_pct": 0.88},
    "gate_90_60_30": {
        "model": "gate",
        "strong_stock_pct": 0.90,
        "mixed_stock_pct": 0.60,
        "weak_stock_pct": 0.30,
        "ma_window": 60,
    },
    "gate_95_65_20": {
        "model": "gate",
        "strong_stock_pct": 0.95,
        "mixed_stock_pct": 0.65,
        "weak_stock_pct": 0.20,
        "ma_window": 60,
    },
    "gate_90_50_10": {
        "model": "gate",
        "strong_stock_pct": 0.90,
        "mixed_stock_pct": 0.50,
        "weak_stock_pct": 0.10,
        "ma_window": 60,
    },
    "gate_85_40_00": {
        "model": "gate",
        "strong_stock_pct": 0.85,
        "mixed_stock_pct": 0.40,
        "weak_stock_pct": 0.00,
        "ma_window": 60,
    },
    "gate_80_25_00": {
        "model": "gate",
        "strong_stock_pct": 0.80,
        "mixed_stock_pct": 0.25,
        "weak_stock_pct": 0.00,
        "ma_window": 60,
    },
}


@dataclass
class Candidate:
    bundle: str
    weights: dict[str, float]
    rebalance_freq: str
    topk: int
    min_market_cap: int
    position_key: str
    buffer: int
    sticky: int
    score_smoothing_days: int
    entry_rank: int
    exit_rank: int
    entry_persist_days: int
    exit_persist_days: int
    min_hold_days: int

    def name(self) -> str:
        return (
            f"{self.bundle}_{self.rebalance_freq}_k{self.topk}_mv{self.min_market_cap}_"
            f"{self.position_key}_a{int(self.weights['alpha']*100)}"
            f"_r{int(self.weights['risk']*100)}_e{int(self.weights['enhance']*100)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=DEFAULT_START_DATE)
    parser.add_argument("--end", default=DEFAULT_END_DATE)
    parser.add_argument("--bundles", default="")
    parser.add_argument("--freqs", default="week,biweek")
    parser.add_argument("--topks", default="8,10")
    parser.add_argument("--min-market-caps", default="50,80")
    parser.add_argument("--positions", default="fixed_88,gate_90_60_30,gate_95_65_20")
    parser.add_argument("--output-tag", default="")
    parser.add_argument("--report-periods", default="")
    return parser.parse_args()


def annual_return_from_series(ret: pd.Series) -> float:
    if ret.empty:
        return 0.0
    nav = (1 + ret).cumprod()
    days = (nav.index[-1] - nav.index[0]).days
    if days <= 0:
        return 0.0
    terminal_value = float(nav.iloc[-1])
    if terminal_value <= 0:
        return -1.0
    return terminal_value ** (365 / days) - 1


def max_drawdown_from_series(ret: pd.Series) -> float:
    if ret.empty:
        return 0.0
    nav = (1 + ret).cumprod()
    return float((nav / nav.cummax() - 1).min())


def sharpe_from_series(ret: pd.Series) -> float:
    if ret.empty or ret.std() == 0:
        return 0.0
    return float(ret.mean() / ret.std() * np.sqrt(252))


def period_metrics(ret: pd.Series, start_date: str, end_date: str) -> dict[str, float]:
    segment = ret[(ret.index >= pd.Timestamp(start_date)) & (ret.index <= pd.Timestamp(end_date))]
    if segment.empty:
        return {"annual_return": 0.0, "max_drawdown": 0.0, "sharpe_ratio": 0.0}
    return {
        "annual_return": annual_return_from_series(segment),
        "max_drawdown": max_drawdown_from_series(segment),
        "sharpe_ratio": sharpe_from_series(segment),
    }


def build_registry() -> FactorRegistry:
    registry = FactorRegistry()
    for spec in FACTOR_SPECS:
        registry.register(spec)
    return registry


def load_factor_frame(
    registry: FactorRegistry,
    instruments: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    def normalize_index(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        if not isinstance(frame.index, pd.MultiIndex):
            return frame

        names = list(frame.index.names)
        if "datetime" in names and "instrument" in names:
            return frame.reorder_levels(["datetime", "instrument"]).sort_index()

        if len(names) == 2:
            frame.index = frame.index.set_names(["instrument", "datetime"])
            return frame.reorder_levels(["datetime", "instrument"]).sort_index()
        return frame

    qlib_factors = registry.get_by_source("qlib")
    parquet_frame = _load_parquet_factors(
        instruments=instruments,
        start_date=start_date,
        end_date=end_date,
        registry=registry,
    )

    qlib_frame = pd.DataFrame()
    if qlib_factors:
        exprs = [factor.expression for factor in qlib_factors]
        qlib_frame = load_features_safe(
            instruments,
            exprs,
            start_time=start_date,
            end_time=end_date,
            freq="day",
        )
        qlib_frame.columns = [f"{factor.category}_{factor.name}" for factor in qlib_factors]
        for factor in qlib_factors:
            col = f"{factor.category}_{factor.name}"
            if factor.negate and col in qlib_frame.columns:
                qlib_frame[col] = -qlib_frame[col]
        qlib_frame = normalize_index(qlib_frame)

    parquet_frame = normalize_index(parquet_frame)

    frames = [frame for frame in (qlib_frame, parquet_frame) if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return _fill_cross_sectional(pd.concat(frames, axis=1).sort_index())


def build_position_controller(position_key: str):
    cfg = POSITION_MODELS[position_key]
    if cfg["model"] == "fixed":
        return _FixedPositionController(cfg["stock_pct"])
    controller = MarketGatePositionController()
    controller.config.strong_stock_pct = cfg["strong_stock_pct"]
    controller.config.mixed_stock_pct = cfg["mixed_stock_pct"]
    controller.config.weak_stock_pct = cfg["weak_stock_pct"]
    controller.config.ma_window = cfg["ma_window"]
    controller.load_market_data()
    return controller


def compute_signal(df_factors: pd.DataFrame, candidate: Candidate) -> pd.Series:
    bundle = BUNDLES[candidate.bundle]
    signal = pd.Series(0.0, index=df_factors.index)
    for category in ("alpha", "risk", "enhance"):
        cols = [f"{category}_{name}" for name in bundle[category]]
        weight = candidate.weights.get(category, 0.0)
        if weight <= 0:
            continue
        signal = signal + weight * compute_layer_score(df_factors, cols)
    return signal


def _compute_fee(
    pre_fee_value: float,
    topk: int,
    stock_pct: float,
    buy_count: int,
    sell_count: int,
) -> float:
    if topk <= 0 or pre_fee_value <= 0:
        return 0.0
    per_position_value = pre_fee_value * stock_pct / topk
    if per_position_value <= 0:
        return 0.0

    buy_fee = max(per_position_value * 0.0003, 5.0) if buy_count > 0 else 0.0
    sell_commission = max(per_position_value * 0.0003, 5.0) if sell_count > 0 else 0.0
    sell_stamp_tax = per_position_value * 0.001 if sell_count > 0 else 0.0
    execution_cost = per_position_value * 0.001
    return (
        buy_count * buy_fee
        + sell_count * (sell_commission + sell_stamp_tax)
        + (buy_count + sell_count) * execution_cost
    )


def backtest_signal(
    signal: pd.Series,
    candidate: Candidate,
    rebalance_dates: pd.DatetimeIndex,
    all_dates: pd.DatetimeIndex,
    df_ret: pd.DataFrame,
    mv_series: pd.Series,
    bond_etf_returns: Optional[pd.Series],
    report_periods: list[tuple[str, str, str]] | None = None,
) -> Optional[dict[str, Any]]:
    df_sel = extract_topk(
        signal,
        rebalance_dates,
        topk=candidate.topk,
        mv_floor=candidate.min_market_cap * 10000,
        mv_series=mv_series,
        sticky=candidate.sticky,
        threshold=0.0,
        churn_limit=2,
        margin_stable=True,
        buffer=candidate.buffer,
        exclude_new_days=120,
        exclude_st=True,
        universe="all",
        score_smoothing_days=candidate.score_smoothing_days,
        entry_rank=candidate.entry_rank,
        exit_rank=candidate.exit_rank,
        entry_persist_days=candidate.entry_persist_days,
        exit_persist_days=candidate.exit_persist_days,
        min_hold_days=candidate.min_hold_days,
    )
    if df_sel.empty:
        return None

    date_to_symbols = {
        pd.Timestamp(dt): set(grp["symbol"].astype(str).tolist())
        for dt, grp in df_sel.groupby("date")
    }
    sorted_rebal_dates = sorted(date_to_symbols.keys())
    if len(sorted_rebal_dates) < 20:
        return None

    controller = build_position_controller(candidate.position_key)
    default_bond_daily_ret = 0.03 / 252
    current_value = INITIAL_CAPITAL
    current_held_symbols: set[str] = set()
    current_cash_slot_count = candidate.topk
    portfolio_returns = []
    turnover_list = []
    penalized_missing: set[str] = set()

    for i, rebal_date in enumerate(sorted_rebal_dates[:-1]):
        selected = date_to_symbols.get(rebal_date, set())
        next_date = sorted_rebal_dates[i + 1]
        holding_dates = all_dates[(all_dates > rebal_date) & (all_dates <= next_date)]

        for j, hd in enumerate(holding_dates):
            try:
                day_px = df_ret.xs(hd, level="datetime")
            except KeyError:
                continue

            held_sum, _, held_missing = _sum_symbol_returns(
                day_px,
                current_held_symbols,
                "daily_ret",
                penalized_missing=penalized_missing,
            )
            stock_slot_return = held_sum / candidate.topk if candidate.topk > 0 else 0.0
            missing_count = len(held_missing)

            if bond_etf_returns is not None and hd in bond_etf_returns.index:
                bond_daily_ret = float(bond_etf_returns.loc[hd])
            else:
                bond_daily_ret = default_bond_daily_ret

            is_rebalance_day = j == 0
            alloc = controller.get_allocation(hd, is_rebalance_day=is_rebalance_day)
            stock_leg_ret = stock_slot_return + (
                current_cash_slot_count / candidate.topk * bond_daily_ret if candidate.topk > 0 else 0.0
            )
            gross_port_ret = alloc.stock_pct * stock_leg_ret + alloc.cash_pct * bond_daily_ret
            pre_fee_value = current_value * (1 + gross_port_ret)

            fee_amount = 0.0
            buy_count = 0
            sell_count = 0
            next_held_symbols = current_held_symbols.copy()
            next_cash_slot_count = current_cash_slot_count

            if is_rebalance_day:
                buy_count = len(selected - current_held_symbols)
                sell_count = len(current_held_symbols - selected)
                next_held_symbols = set(selected)
                next_cash_slot_count = max(candidate.topk - len(next_held_symbols), 0)
                turnover_list.append(
                    (buy_count + sell_count) / (2 * candidate.topk) if candidate.topk > 0 else 0.0
                )
                fee_amount = _compute_fee(
                    pre_fee_value=pre_fee_value,
                    topk=candidate.topk,
                    stock_pct=alloc.stock_pct,
                    buy_count=buy_count,
                    sell_count=sell_count,
                )

            end_value = pre_fee_value - fee_amount
            port_ret = end_value / current_value - 1 if current_value > 0 else 0.0
            portfolio_returns.append(
                {
                    "date": hd,
                    "return": port_ret,
                    "gross_return": gross_port_ret,
                    "missing_count": missing_count,
                }
            )
            current_value = end_value

            if is_rebalance_day:
                current_held_symbols = next_held_symbols.copy()
                current_cash_slot_count = next_cash_slot_count

    if not portfolio_returns:
        return None

    df_result = pd.DataFrame(portfolio_returns).set_index("date")
    daily_returns = df_result["return"]
    nav = (1 + daily_returns).cumprod()

    yearly = {}
    for year in sorted(daily_returns.index.year.unique()):
        yr = daily_returns[daily_returns.index.year == year]
        if len(yr) > 20:
            yearly[year] = float((1 + yr).prod() - 1)

    return {
        "annual_return": annual_return_from_series(daily_returns),
        "sharpe_ratio": sharpe_from_series(daily_returns),
        "max_drawdown": max_drawdown_from_series(daily_returns),
        "total_return": float(nav.iloc[-1] - 1) if not nav.empty else 0.0,
        "avg_turnover": float(np.mean(turnover_list)) if turnover_list else 0.0,
        "selection_dates": len(sorted_rebal_dates),
        "yearly": yearly,
        "period_metrics": {
            label: period_metrics(daily_returns, period_start, period_end)
            for label, period_start, period_end in (report_periods or [])
        },
    }


def build_candidates(
    bundle_names: list[str],
    freqs: list[str],
    topks: list[int],
    min_market_caps: list[int],
    position_keys: list[str],
) -> list[Candidate]:
    candidates: list[Candidate] = []
    for bundle_name in bundle_names:
        bundle = BUNDLES[bundle_name]
        for weights in bundle["weight_options"]:
            for freq in freqs:
                for topk in topks:
                    for min_market_cap in min_market_caps:
                        for position_key in position_keys:
                            candidates.append(
                                Candidate(
                                    bundle=bundle_name,
                                    weights=weights,
                                    rebalance_freq=freq,
                                    topk=topk,
                                    min_market_cap=min_market_cap,
                                    position_key=position_key,
                                    buffer=8 if freq != "day" else 10,
                                    sticky=4 if freq != "day" else 5,
                                    score_smoothing_days=5,
                                    entry_rank=max(1, int(round(topk * 0.7))),
                                    exit_rank=topk + 8,
                                    entry_persist_days=2 if freq == "week" else 3,
                                    exit_persist_days=2 if freq == "week" else 3,
                                    min_hold_days=10,
                                )
                            )
    return candidates


def main() -> None:
    args = parse_args()
    start_date = str(args.start)
    end_date = str(args.end)
    bundle_names = [x.strip() for x in str(args.bundles).split(",") if x.strip()] or list(BUNDLES.keys())
    freqs = [x.strip() for x in str(args.freqs).split(",") if x.strip()]
    topks = [int(x.strip()) for x in str(args.topks).split(",") if x.strip()]
    min_market_caps = [int(x.strip()) for x in str(args.min_market_caps).split(",") if x.strip()]
    position_keys = [x.strip() for x in str(args.positions).split(",") if x.strip()]
    report_periods = []
    if str(args.report_periods).strip():
        for idx, item in enumerate(str(args.report_periods).split(","), 1):
            item = item.strip()
            if not item:
                continue
            period_start, period_end = item.split(":")
            label = f"p{idx}_{period_start.replace('-', '')}_{period_end.replace('-', '')}"
            report_periods.append((label, period_start, period_end))

    print("=" * 88)
    print("Fresh Strategy Candidate Search")
    print("=" * 88)
    print(f"Range: {start_date} ~ {end_date}")

    init_qlib()
    from qlib.data import D

    registry = build_registry()

    print("[1/4] Loading instruments...")
    instruments = D.instruments(market="all")
    df_close = load_features_safe(
        instruments,
        ["$close"],
        start_time=start_date,
        end_time=end_date,
        freq="day",
    )
    if isinstance(df_close.index, pd.MultiIndex) and "datetime" in df_close.index.names:
        df_close = df_close.reorder_levels(["datetime", "instrument"]).sort_index()
    df_close.columns = ["close"]
    valid = filter_instruments(
        df_close.index.get_level_values("instrument").unique().tolist(),
        exclude_st=False,
    )
    all_dates = df_close.index.get_level_values("datetime").unique().sort_values()
    print(f"  instruments={len(valid)} trade_dates={len(all_dates)}")

    print("[2/4] Loading factor frame...")
    df_factors = load_factor_frame(registry, valid, start_date, end_date)
    print(f"  factor_shape={df_factors.shape}")

    print("[3/4] Loading return data...")
    df_ret = df_close.loc[df_close.index.get_level_values("instrument").isin(valid)].copy()
    df_ret["daily_ret"] = df_ret.groupby(level="instrument")["close"].pct_change()
    df_ret = df_ret[["daily_ret"]].dropna()
    mv_series = df_factors["alpha_total_mv_raw"] if "alpha_total_mv_raw" in df_factors.columns else None
    bond_etf_returns = _load_bond_etf_returns()

    print("[4/4] Running candidate search...")
    candidates = build_candidates(bundle_names, freqs, topks, min_market_caps, position_keys)
    rows = []
    for idx, candidate in enumerate(candidates, 1):
        rebalance_dates = compute_rebalance_dates(pd.Series(all_dates), freq=candidate.rebalance_freq)
        signal = compute_signal(df_factors, candidate)
        result = backtest_signal(
            signal=signal,
            candidate=candidate,
            rebalance_dates=rebalance_dates,
            all_dates=all_dates,
            df_ret=df_ret,
            mv_series=mv_series,
            bond_etf_returns=bond_etf_returns,
            report_periods=report_periods,
        )
        if result is None:
            print(f"  [{idx:03d}/{len(candidates)}] {candidate.name():60s} SKIP")
            continue

        row = {
            "candidate": candidate.name(),
            "bundle": candidate.bundle,
            "rebalance_freq": candidate.rebalance_freq,
            "topk": candidate.topk,
            "min_market_cap": candidate.min_market_cap,
            "position": candidate.position_key,
            "alpha_weight": candidate.weights["alpha"],
            "risk_weight": candidate.weights["risk"],
            "enhance_weight": candidate.weights["enhance"],
            "annual_return": result["annual_return"],
            "sharpe_ratio": result["sharpe_ratio"],
            "max_drawdown": result["max_drawdown"],
            "total_return": result["total_return"],
            "avg_turnover": result["avg_turnover"],
            "selection_dates": result["selection_dates"],
        }
        for year, value in result["yearly"].items():
            row[f"ret_{year}"] = value
        for label, metrics in result["period_metrics"].items():
            row[f"{label}_annual_return"] = metrics["annual_return"]
            row[f"{label}_max_drawdown"] = metrics["max_drawdown"]
            row[f"{label}_sharpe_ratio"] = metrics["sharpe_ratio"]
        rows.append(row)
        print(
            f"  [{idx:03d}/{len(candidates)}] {candidate.name():60s} "
            f"ann={result['annual_return']:+.1%} sharpe={result['sharpe_ratio']:.2f} "
            f"mdd={result['max_drawdown']:.1%}"
        )

    df_results = pd.DataFrame(rows)
    if df_results.empty:
        print("No candidate produced results.")
        return

    df_results = df_results.sort_values(
        ["annual_return", "sharpe_ratio", "max_drawdown"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.output_tag.strip()}" if str(args.output_tag).strip() else ""
    out_path = RESULTS_DIR / (
        f"fresh_strategy_candidate_search_{start_date.replace('-', '')}_{end_date.replace('-', '')}{tag}.csv"
    )
    df_results.to_csv(out_path, index=False, float_format="%.6f")

    print("\nTop candidates by annual return:")
    show = df_results[
        [
            "candidate",
            "annual_return",
            "sharpe_ratio",
            "max_drawdown",
            "avg_turnover",
            "position",
        ]
    ].head(20)
    print(show.to_string(index=False))

    filtered = df_results[
        (df_results["annual_return"] > 0.25) & (df_results["max_drawdown"] > -0.15)
    ]
    print("\nCandidates meeting target (>25% annual, <15% drawdown):")
    if filtered.empty:
        print("  none")
    else:
        print(
            filtered[
                ["candidate", "annual_return", "sharpe_ratio", "max_drawdown", "avg_turnover"]
            ].to_string(index=False)
        )

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
