#!/usr/bin/env python3
"""
Fast candidate search using only factor_data.parquet plus market gate/fixed position models.

This avoids the stdin/spawn issue from ad-hoc heredoc execution on macOS and keeps the
search focused on a smaller, higher-probability space:
- quality / value / cash-flow alpha
- low turnover / low leverage risk
- money-flow enhancement
"""

from __future__ import annotations

import argparse
import warnings

warnings.filterwarnings("ignore")

from dataclasses import dataclass
from pathlib import Path

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


def build_registry() -> FactorRegistry:
    registry = FactorRegistry()
    for spec in [
        FactorInfo("roa", "roa_fina", "", "alpha", source="parquet"),
        FactorInfo("ocf_to_ev", "ocf_to_ev", "", "alpha", source="parquet"),
        FactorInfo("ebit_to_mv", "ebit_to_mv", "", "alpha", source="parquet"),
        FactorInfo("retained_earnings", "retained_earnings", "", "alpha", source="parquet"),
        FactorInfo("book_to_price", "book_to_market", "", "alpha", source="parquet"),
        FactorInfo("turnover_rate_f", "turnover_rate_f", "", "risk", source="parquet", negate=True),
        FactorInfo("debt_to_assets", "debt_to_assets_fina", "", "risk", source="parquet", negate=True),
        FactorInfo("net_mf_5d", "net_mf_amount_5d", "", "enhance", source="parquet"),
        FactorInfo("net_mf_20d", "net_mf_amount_20d", "", "enhance", source="parquet"),
        FactorInfo("smart_ratio", "smart_ratio_5d", "", "enhance", source="parquet"),
        FactorInfo("total_mv", "total_mv", "", "alpha", source="parquet"),
    ]:
        registry.register(spec)
    return registry


FAMILIES = {
    "quality_flow_core": {
        "alpha": ["roa", "ocf_to_ev", "ebit_to_mv"],
        "risk": ["turnover_rate_f", "debt_to_assets"],
        "enhance": ["net_mf_5d", "smart_ratio"],
        "weights_list": [
            {"alpha": 0.30, "risk": 0.20, "enhance": 0.50},
            {"alpha": 0.25, "risk": 0.25, "enhance": 0.50},
        ],
    },
    "quality_flow_wide": {
        "alpha": ["roa", "ocf_to_ev", "retained_earnings"],
        "risk": ["turnover_rate_f", "debt_to_assets"],
        "enhance": ["net_mf_5d", "net_mf_20d", "smart_ratio"],
        "weights_list": [
            {"alpha": 0.25, "risk": 0.15, "enhance": 0.60},
            {"alpha": 0.30, "risk": 0.20, "enhance": 0.50},
        ],
    },
    "value_flow": {
        "alpha": ["book_to_price", "ocf_to_ev", "ebit_to_mv"],
        "risk": ["turnover_rate_f", "debt_to_assets"],
        "enhance": ["net_mf_5d", "smart_ratio"],
        "weights_list": [
            {"alpha": 0.35, "risk": 0.15, "enhance": 0.50},
            {"alpha": 0.30, "risk": 0.20, "enhance": 0.50},
        ],
    },
}


@dataclass
class Candidate:
    family: str
    weights: dict[str, float]
    freq: str
    topk: int
    position: str
    min_market_cap: int = 80
    buffer: int = 8
    sticky: int = 4
    score_smoothing_days: int = 5
    entry_persist_days: int = 2
    exit_persist_days: int = 2
    min_hold_days: int = 10

    def name(self) -> str:
        return (
            f"{self.family}_{self.freq}_k{self.topk}_{self.position}_"
            f"a{int(self.weights['alpha']*100)}_r{int(self.weights['risk']*100)}"
            f"_e{int(self.weights['enhance']*100)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=DEFAULT_START_DATE)
    parser.add_argument("--end", default=DEFAULT_END_DATE)
    parser.add_argument("--families", default="")
    parser.add_argument("--freqs", default="week,biweek")
    parser.add_argument("--topks", default="8,10")
    parser.add_argument("--positions", default="gate,fixed")
    parser.add_argument("--min-market-cap", type=int, default=80)
    parser.add_argument("--output-tag", default="")
    parser.add_argument("--report-periods", default="")
    return parser.parse_args()


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


def make_controller(position: str):
    if position == "fixed":
        return _FixedPositionController(0.88)
    ctl = MarketGatePositionController()
    if position == "gate":
        ctl.config.strong_stock_pct = 0.95
        ctl.config.mixed_stock_pct = 0.65
        ctl.config.weak_stock_pct = 0.20
    elif position == "gate_balanced":
        ctl.config.strong_stock_pct = 0.90
        ctl.config.mixed_stock_pct = 0.50
        ctl.config.weak_stock_pct = 0.10
    elif position == "gate_defensive":
        ctl.config.strong_stock_pct = 0.85
        ctl.config.mixed_stock_pct = 0.40
        ctl.config.weak_stock_pct = 0.00
    elif position == "gate_hard":
        ctl.config.strong_stock_pct = 0.80
        ctl.config.mixed_stock_pct = 0.25
        ctl.config.weak_stock_pct = 0.00
    else:
        raise ValueError(f"Unknown position model: {position}")
    ctl.config.ma_window = 60
    ctl.load_market_data()
    return ctl


def compute_fee(
    pre_fee_value: float,
    topk: int,
    stock_pct: float,
    buy_count: int,
    sell_count: int,
) -> float:
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


def backtest_candidate(
    signal: pd.Series,
    candidate: Candidate,
    rebalance_dates: pd.DatetimeIndex,
    all_dates: pd.DatetimeIndex,
    df_ret: pd.DataFrame,
    mv_series: pd.Series | None,
    bond_returns: pd.Series | None,
    report_periods: list[tuple[str, str, str]] | None = None,
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
        score_smoothing_days=candidate.score_smoothing_days,
        entry_rank=max(1, round(candidate.topk * 0.7)),
        exit_rank=candidate.topk + 8,
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
    ordered_rebalance_dates = sorted(date_to_symbols)
    if len(ordered_rebalance_dates) < 20:
        return None

    controller = make_controller(candidate.position)
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
    yearly = {}
    for year in sorted(ret.index.year.unique()):
        year_ret = ret[ret.index.year == year]
        if len(year_ret) > 20:
            yearly[year] = float((1 + year_ret).prod() - 1)

    return {
        "annual_return": annual_return(ret),
        "sharpe_ratio": sharpe_ratio(ret),
        "max_drawdown": max_drawdown(ret),
        "total_return": float((1 + ret).prod() - 1),
        "avg_turnover": float(np.mean(turnovers)) if turnovers else 0.0,
        "yearly": yearly,
        "period_metrics": {
            label: period_metrics(ret, period_start, period_end)
            for label, period_start, period_end in (report_periods or [])
        },
    }


def main() -> None:
    args = parse_args()
    start_date = str(args.start)
    end_date = str(args.end)
    family_filter = {x.strip() for x in str(args.families).split(",") if x.strip()}
    freqs = [x.strip() for x in str(args.freqs).split(",") if x.strip()]
    topks = [int(x.strip()) for x in str(args.topks).split(",") if x.strip()]
    positions = [x.strip() for x in str(args.positions).split(",") if x.strip()]
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
    print("Parquet Flow Candidate Search")
    print("=" * 88)
    print(f"Range: {start_date} ~ {end_date}")

    print("[1/3] Loading close and returns...")
    init_qlib()
    from qlib.data import D

    instruments = D.instruments(market="all")
    df_close = load_features_safe(
        instruments,
        ["$close"],
        start_time=start_date,
        end_time=end_date,
        freq="day",
    )
    df_close = df_close.reorder_levels(["datetime", "instrument"]).sort_index()
    df_close.columns = ["close"]
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
            start_date=start_date,
            end_date=end_date,
            registry=registry,
        )
    )
    mv_series = df_factors["alpha_total_mv"] if "alpha_total_mv" in df_factors.columns else None
    bond_returns = _load_bond_etf_returns()
    print(f"  factor_shape={df_factors.shape} ret_shape={df_ret.shape}")

    print("[3/3] Running candidate grid...")
    candidates = []
    for family, bundle in FAMILIES.items():
        if family_filter and family not in family_filter:
            continue
        for weights in bundle["weights_list"]:
            for freq in freqs:
                for topk in topks:
                    for position in positions:
                        candidates.append(
                            Candidate(
                                family,
                                weights,
                                freq,
                                topk,
                                position,
                                min_market_cap=int(args.min_market_cap),
                            )
                        )

    rows = []
    for idx, candidate in enumerate(candidates, 1):
        bundle = FAMILIES[candidate.family]
        signal = pd.Series(0.0, index=df_factors.index)
        for category in ("alpha", "risk", "enhance"):
            factor_cols = [f"{category}_{name}" for name in bundle[category]]
            signal = signal + candidate.weights[category] * compute_layer_score(df_factors, factor_cols)

        rebalance_dates = compute_rebalance_dates(pd.Series(all_dates), freq=candidate.freq)
        result = backtest_candidate(
            signal=signal,
            candidate=candidate,
            rebalance_dates=rebalance_dates,
            all_dates=all_dates,
            df_ret=df_ret,
            mv_series=mv_series,
            bond_returns=bond_returns,
            report_periods=report_periods,
        )
        if result is None:
            print(f"  [{idx:02d}/{len(candidates)}] {candidate.name()} SKIP")
            continue

        row = {
            "name": candidate.name(),
            "family": candidate.family,
            "freq": candidate.freq,
            "topk": candidate.topk,
            "position": candidate.position,
            "annual_return": result["annual_return"],
            "sharpe_ratio": result["sharpe_ratio"],
            "max_drawdown": result["max_drawdown"],
            "total_return": result["total_return"],
            "avg_turnover": result["avg_turnover"],
        }
        for year, value in result["yearly"].items():
            row[f"ret_{year}"] = value
        for label, metrics in result["period_metrics"].items():
            row[f"{label}_annual_return"] = metrics["annual_return"]
            row[f"{label}_max_drawdown"] = metrics["max_drawdown"]
            row[f"{label}_sharpe_ratio"] = metrics["sharpe_ratio"]
        rows.append(row)
        print(
            f"  [{idx:02d}/{len(candidates)}] {candidate.name()} "
            f"ann={result['annual_return']:+.1%} "
            f"sharpe={result['sharpe_ratio']:.2f} "
            f"mdd={result['max_drawdown']:.1%}"
        )

    df_results = pd.DataFrame(rows).sort_values(
        ["annual_return", "sharpe_ratio"],
        ascending=[False, False],
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.output_tag.strip()}" if str(args.output_tag).strip() else ""
    out_path = RESULTS_DIR / (
        f"search_parquet_flow_candidates_{start_date.replace('-', '')}_{end_date.replace('-', '')}{tag}.csv"
    )
    df_results.to_csv(out_path, index=False, float_format="%.6f")

    print("\nTop 15 candidates:")
    print(
        df_results[
            ["name", "annual_return", "sharpe_ratio", "max_drawdown", "avg_turnover", "position"]
        ].head(15).to_string(index=False)
    )

    print("\nCandidates meeting target (>25% annual, <15% drawdown):")
    qualified = df_results[
        (df_results["annual_return"] > 0.25) & (df_results["max_drawdown"] > -0.15)
    ]
    if qualified.empty:
        print("  none")
    else:
        print(
            qualified[
                ["name", "annual_return", "sharpe_ratio", "max_drawdown", "avg_turnover"]
            ].to_string(index=False)
        )

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
