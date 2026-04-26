#!/usr/bin/env python3
"""
Targeted search around value-flow signals with stoploss-replace selection.

Purpose:
- test whether execution logic, not factor structure, is the missing piece
- keep the search narrow and reproducible
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from core.compute import compute_layer_score
from core.factors import FactorInfo, FactorRegistry
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
START_DATE = "2021-01-01"
END_DATE = "2026-04-15"
INITIAL_CAPITAL = 1_000_000.0


def build_registry() -> FactorRegistry:
    registry = FactorRegistry()
    for spec in [
        FactorInfo("book_to_price", "book_to_market", "", "alpha", source="parquet"),
        FactorInfo("ocf_to_ev", "ocf_to_ev", "", "alpha", source="parquet"),
        FactorInfo("ebit_to_mv", "ebit_to_mv", "", "alpha", source="parquet"),
        FactorInfo("turnover_rate_f", "turnover_rate_f", "", "risk", source="parquet", negate=True),
        FactorInfo("debt_to_assets", "debt_to_assets_fina", "", "risk", source="parquet", negate=True),
        FactorInfo("net_mf_5d", "net_mf_amount_5d", "", "enhance", source="parquet"),
        FactorInfo("smart_ratio", "smart_ratio_5d", "", "enhance", source="parquet"),
        FactorInfo("total_mv", "total_mv", "", "alpha", source="parquet"),
    ]:
        registry.register(spec)
    return registry


WEIGHT_OPTIONS = [
    {"alpha": 0.35, "risk": 0.15, "enhance": 0.50},
    {"alpha": 0.30, "risk": 0.20, "enhance": 0.50},
]


@dataclass
class Candidate:
    weights: dict[str, float]
    freq: str
    topk: int
    stock_pct: float
    stoploss_lookback_days: int
    stoploss_drawdown: float
    replacement_pool_size: int
    min_market_cap: int = 80
    buffer: int = 8
    sticky: int = 4
    score_smoothing_days: int = 5
    entry_persist_days: int = 2
    exit_persist_days: int = 2
    min_hold_days: int = 10

    def name(self) -> str:
        return (
            f"value_flow_stoploss_{self.freq}_k{self.topk}_pct{int(self.stock_pct*100)}_"
            f"lb{self.stoploss_lookback_days}_dd{int(self.stoploss_drawdown*100)}_"
            f"pool{self.replacement_pool_size}_a{int(self.weights['alpha']*100)}"
            f"_r{int(self.weights['risk']*100)}_e{int(self.weights['enhance']*100)}"
        )


def annual_return(ret: pd.Series) -> float:
    nav = (1 + ret).cumprod()
    if nav.empty:
        return 0.0
    days = (nav.index[-1] - nav.index[0]).days
    if days <= 0 or nav.iloc[-1] <= 0:
        return 0.0
    return float(nav.iloc[-1] ** (365 / days) - 1)


def max_drawdown(ret: pd.Series) -> float:
    nav = (1 + ret).cumprod()
    if nav.empty:
        return 0.0
    return float((nav / nav.cummax() - 1).min())


def sharpe_ratio(ret: pd.Series) -> float:
    if ret.empty or ret.std() == 0:
        return 0.0
    return float(ret.mean() / ret.std() * np.sqrt(252))


def period_metrics(ret: pd.Series, start_date: str, end_date: str) -> dict[str, float]:
    segment = ret[(ret.index >= pd.Timestamp(start_date)) & (ret.index <= pd.Timestamp(end_date))]
    if segment.empty:
        return {"annual_return": 0.0, "max_drawdown": 0.0, "sharpe_ratio": 0.0}
    return {
        "annual_return": annual_return(segment),
        "max_drawdown": max_drawdown(segment),
        "sharpe_ratio": sharpe_ratio(segment),
    }


def compute_fee(pre_fee_value: float, topk: int, stock_pct: float, buy_count: int, sell_count: int) -> float:
    if topk <= 0 or pre_fee_value <= 0:
        return 0.0
    position_value = pre_fee_value * stock_pct / topk
    if position_value <= 0:
        return 0.0
    buy_fee = max(position_value * 0.0003, 5.0) if buy_count > 0 else 0.0
    sell_commission = max(position_value * 0.0003, 5.0) if sell_count > 0 else 0.0
    sell_stamp_tax = position_value * 0.001 if sell_count > 0 else 0.0
    execution_cost = position_value * 0.001
    return (
        buy_count * buy_fee
        + sell_count * (sell_commission + sell_stamp_tax)
        + (buy_count + sell_count) * execution_cost
    )


def build_signal(df_factors: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    signal = pd.Series(0.0, index=df_factors.index)
    signal = signal + weights["alpha"] * compute_layer_score(
        df_factors,
        ["alpha_book_to_price", "alpha_ocf_to_ev", "alpha_ebit_to_mv"],
    )
    signal = signal + weights["risk"] * compute_layer_score(
        df_factors,
        ["risk_turnover_rate_f", "risk_debt_to_assets"],
    )
    signal = signal + weights["enhance"] * compute_layer_score(
        df_factors,
        ["enhance_net_mf_5d", "enhance_smart_ratio"],
    )
    return signal


def backtest_candidate(
    signal: pd.Series,
    candidate: Candidate,
    rebalance_dates: pd.DatetimeIndex,
    all_dates: pd.DatetimeIndex,
    df_ret: pd.DataFrame,
    mv_series: pd.Series | None,
    bond_returns: pd.Series | None,
    close_series: pd.Series,
) -> dict | None:
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
        selection_mode="stoploss_replace",
        score_smoothing_days=candidate.score_smoothing_days,
        entry_rank=max(1, round(candidate.topk * 0.7)),
        exit_rank=candidate.topk + 8,
        entry_persist_days=candidate.entry_persist_days,
        exit_persist_days=candidate.exit_persist_days,
        min_hold_days=candidate.min_hold_days,
        close_series=close_series,
        stoploss_lookback_days=candidate.stoploss_lookback_days,
        stoploss_drawdown=candidate.stoploss_drawdown,
        replacement_pool_size=candidate.replacement_pool_size,
    )
    if df_sel.empty:
        return None

    date_to_symbols = {
        pd.Timestamp(dt): set(grp["symbol"].astype(str).tolist())
        for dt, grp in df_sel.groupby("date")
    }
    ordered_rebalance_dates = sorted(date_to_symbols)
    if len(ordered_rebalance_dates) < 20:
        return None

    controller = _FixedPositionController(candidate.stock_pct)
    current_value = INITIAL_CAPITAL
    current_held_symbols: set[str] = set()
    current_cash_slots = candidate.topk
    penalized_missing: set[str] = set()
    returns = []
    turnovers = []
    default_bond_daily_ret = 0.03 / 252

    for i, rebalance_date in enumerate(ordered_rebalance_dates[:-1]):
        selected = date_to_symbols[rebalance_date]
        next_rebalance_date = ordered_rebalance_dates[i + 1]
        holding_dates = all_dates[(all_dates > rebalance_date) & (all_dates <= next_rebalance_date)]

        for j, holding_date in enumerate(holding_dates):
            try:
                day_px = df_ret.xs(holding_date, level="datetime")
            except KeyError:
                continue

            held_sum, _, _ = _sum_symbol_returns(
                day_px,
                current_held_symbols,
                "daily_ret",
                penalized_missing=penalized_missing,
            )
            stock_slot_return = held_sum / candidate.topk if candidate.topk > 0 else 0.0
            if bond_returns is not None and holding_date in bond_returns.index:
                bond_daily_ret = float(bond_returns.loc[holding_date])
            else:
                bond_daily_ret = default_bond_daily_ret

            is_rebalance_day = j == 0
            allocation = controller.get_allocation(holding_date, is_rebalance_day=is_rebalance_day)
            stock_leg_return = stock_slot_return + (
                current_cash_slots / candidate.topk * bond_daily_ret if candidate.topk > 0 else 0.0
            )
            gross_return = allocation.stock_pct * stock_leg_return + allocation.cash_pct * bond_daily_ret
            pre_fee_value = current_value * (1 + gross_return)

            fee_amount = 0.0
            if is_rebalance_day:
                buy_count = len(selected - current_held_symbols)
                sell_count = len(current_held_symbols - selected)
                turnovers.append((buy_count + sell_count) / (2 * candidate.topk))
                fee_amount = compute_fee(
                    pre_fee_value=pre_fee_value,
                    topk=candidate.topk,
                    stock_pct=allocation.stock_pct,
                    buy_count=buy_count,
                    sell_count=sell_count,
                )
                current_held_symbols = set(selected)
                current_cash_slots = max(candidate.topk - len(current_held_symbols), 0)

            end_value = pre_fee_value - fee_amount
            returns.append((holding_date, end_value / current_value - 1))
            current_value = end_value

    if not returns:
        return None

    ret = pd.Series({dt: value for dt, value in returns}).sort_index()
    return {
        "annual_return": annual_return(ret),
        "sharpe_ratio": sharpe_ratio(ret),
        "max_drawdown": max_drawdown(ret),
        "total_return": float((1 + ret).prod() - 1),
        "avg_turnover": float(np.mean(turnovers)) if turnovers else 0.0,
        "p_2021_2023": period_metrics(ret, "2021-01-01", "2023-12-31"),
        "p_2024_2026": period_metrics(ret, "2024-01-01", "2026-04-15"),
    }


def main() -> None:
    print("=" * 88)
    print("Value Flow Stoploss Search")
    print("=" * 88)

    print("[1/3] Loading close and returns...")
    init_qlib()
    from qlib.data import D

    instruments = D.instruments(market="all")
    df_close = load_features_safe(
        instruments,
        ["$close"],
        start_time=START_DATE,
        end_time=END_DATE,
        freq="day",
    )
    df_close = df_close.reorder_levels(["datetime", "instrument"]).sort_index()
    df_close.columns = ["close"]
    close_series = df_close["close"]
    valid = filter_instruments(
        df_close.index.get_level_values("instrument").unique().tolist(),
        exclude_st=False,
    )
    all_dates = df_close.index.get_level_values("datetime").unique().sort_values()
    df_ret = df_close.loc[df_close.index.get_level_values("instrument").isin(valid)].copy()
    df_ret["daily_ret"] = df_ret.groupby(level="instrument")["close"].pct_change()
    df_ret = df_ret[["daily_ret"]].dropna()

    print("[2/3] Loading parquet factors...")
    registry = build_registry()
    df_factors = _fill_cross_sectional(
        _load_parquet_factors(
            instruments=valid,
            start_date=START_DATE,
            end_date=END_DATE,
            registry=registry,
        )
    )
    mv_series = df_factors["alpha_total_mv"] if "alpha_total_mv" in df_factors.columns else None
    bond_returns = _load_bond_etf_returns()
    print(f"  factor_shape={df_factors.shape} ret_shape={df_ret.shape}")

    print("[3/3] Running stoploss grid...")
    candidates = []
    for weights in WEIGHT_OPTIONS:
        for freq in ("week", "biweek"):
            for topk in (8, 10):
                for stock_pct in (0.88, 0.95, 1.00):
                    for lookback in (20, 40):
                        for drawdown in (0.08, 0.10, 0.12):
                            for pool in (20,):
                                candidates.append(
                                    Candidate(
                                        weights=weights,
                                        freq=freq,
                                        topk=topk,
                                        stock_pct=stock_pct,
                                        stoploss_lookback_days=lookback,
                                        stoploss_drawdown=drawdown,
                                        replacement_pool_size=pool,
                                    )
                                )

    rows = []
    for idx, candidate in enumerate(candidates, 1):
        signal = build_signal(df_factors, candidate.weights)
        rebalance_dates = compute_rebalance_dates(pd.Series(all_dates), freq=candidate.freq)
        result = backtest_candidate(
            signal=signal,
            candidate=candidate,
            rebalance_dates=rebalance_dates,
            all_dates=all_dates,
            df_ret=df_ret,
            mv_series=mv_series,
            bond_returns=bond_returns,
            close_series=close_series,
        )
        if result is None:
            print(f"  [{idx:03d}/{len(candidates)}] {candidate.name()} SKIP")
            continue

        row = {
            "name": candidate.name(),
            "freq": candidate.freq,
            "topk": candidate.topk,
            "stock_pct": candidate.stock_pct,
            "stoploss_lookback_days": candidate.stoploss_lookback_days,
            "stoploss_drawdown": candidate.stoploss_drawdown,
            "replacement_pool_size": candidate.replacement_pool_size,
            "annual_return": result["annual_return"],
            "sharpe_ratio": result["sharpe_ratio"],
            "max_drawdown": result["max_drawdown"],
            "total_return": result["total_return"],
            "avg_turnover": result["avg_turnover"],
            "p1_annual_return": result["p_2021_2023"]["annual_return"],
            "p1_max_drawdown": result["p_2021_2023"]["max_drawdown"],
            "p2_annual_return": result["p_2024_2026"]["annual_return"],
            "p2_max_drawdown": result["p_2024_2026"]["max_drawdown"],
        }
        rows.append(row)
        print(
            f"  [{idx:03d}/{len(candidates)}] {candidate.name()} "
            f"ann={result['annual_return']:+.1%} sharpe={result['sharpe_ratio']:.2f} "
            f"mdd={result['max_drawdown']:.1%} "
            f"p1={row['p1_annual_return']:+.1%}/{row['p1_max_drawdown']:.1%} "
            f"p2={row['p2_annual_return']:+.1%}/{row['p2_max_drawdown']:.1%}"
        )

    df_results = pd.DataFrame(rows).sort_values(
        ["annual_return", "sharpe_ratio"],
        ascending=[False, False],
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "search_value_flow_stoploss_20210101_20260415.csv"
    df_results.to_csv(out_path, index=False, float_format="%.6f")

    print("\nTop 20 candidates:")
    print(
        df_results[
            [
                "name",
                "annual_return",
                "sharpe_ratio",
                "max_drawdown",
                "p1_annual_return",
                "p1_max_drawdown",
                "p2_annual_return",
                "p2_max_drawdown",
            ]
        ].head(20).to_string(index=False)
    )

    qualified = df_results[
        (df_results["annual_return"] > 0.25) & (df_results["max_drawdown"] > -0.15)
    ]
    print("\nCandidates meeting target (>25% annual, <15% drawdown):")
    if qualified.empty:
        print("  none")
    else:
        print(
            qualified[
                ["name", "annual_return", "sharpe_ratio", "max_drawdown", "p1_annual_return", "p2_annual_return"]
            ].to_string(index=False)
        )

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
