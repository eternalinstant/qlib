#!/usr/bin/env python3
"""
搜索更稳健的 close-only 日频策略。

目标：
1. 不依赖 $high/$low
2. 优先保证 2025-2026 表现
3. 同时检查 2019-2024 是否明显断层，避免只看单段样本
"""
import warnings
warnings.filterwarnings("ignore")

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.compute import compute_layer_score
from core.factors import FactorInfo, FactorRegistry
from core.position import MarketPositionController
from core.qlib_init import init_qlib, load_features_safe
from core.selection import extract_topk
from core.strategy import Strategy
from core.universe import filter_instruments
from modules.backtest.base import BacktestResult
from modules.backtest.qlib_engine import QlibBacktestEngine
from scripts.factor_scan import get_all_factors

START_DATE = "2019-01-01"
END_DATE = "2026-03-20"
INITIAL_CAPITAL = 500000.0

BUY_COMMISSION_RATE = 0.0003
SELL_COMMISSION_RATE = 0.0003
SELL_STAMP_TAX_RATE = 0.001
MIN_BUY_COMMISSION = 5.0
MIN_SELL_COMMISSION = 5.0

ALPHA_FACTORS = ["total_mv_log"]
RISK_FACTORS = ["vol_std_10d", "amt_std_6d", "vol_ema26_ratio"]

ENHANCE_BUNDLES = {
    "short_trend_core": ["ema5_dev", "ema10_dev", "ema12_dev"],
    "short_trend_bbi": ["ema5_dev", "ema10_dev", "bbi_momentum"],
    "band_trend_core": ["bb_upper_dist", "ema5_dev", "bb_position"],
    "band_sharpe_core": ["bb_upper_dist", "bb_position", "sharpe_120d"],
    "quality_trend": ["ema10_dev", "ema12_dev", "sharpe_120d"],
    "position_quality": ["bb_position", "sharpe_120d", "price_pos_52w"],
    "mid_trend_quality": ["ema12_dev", "ema20_dev", "sharpe_120d"],
    "breakout_core": ["ema5_dev", "close_to_high_60d", "price_pos_52w"],
    "breakout_band": ["bb_upper_dist", "close_to_high_60d", "price_pos_52w"],
    "slope_quality": ["ema5_dev", "slope_6d", "sharpe_20d"],
    "short_pure": ["ema5_dev", "ema10_dev"],
    "band_pure": ["bb_upper_dist", "bb_position"],
}

COARSE_TOPK = [10, 12, 15]
COARSE_BUFFER = [10, 15]
COARSE_MIN_CAP = [50, 80]
COARSE_EXCLUDE_NEW = 120
COARSE_WEIGHTS = {"alpha": 0.05, "risk": 0.25, "enhance": 0.70}

REFINE_ALPHA = [0.0, 0.05]
REFINE_RISK = [0.15, 0.25, 0.35]
REFINE_EXCLUDE_NEW = [60, 120]
REFINE_POSITION = ["trend", "fixed"]

PHASES = {
    "train_2019_2022": ("2019-01-01", "2022-12-31"),
    "valid_2023_2024": ("2023-01-01", "2024-12-31"),
    "test_2025_2026": ("2025-01-01", "2026-12-31"),
}


@dataclass
class Candidate:
    bundle_name: str
    enhance_factors: List[str]
    topk: int
    buffer: int
    min_cap: int
    exclude_new_days: int
    alpha_weight: float
    risk_weight: float
    enhance_weight: float
    position_model: str

    def name(self) -> str:
        return (
            f"{self.bundle_name}_k{self.topk}_b{self.buffer}_mv{self.min_cap}"
            f"_new{self.exclude_new_days}_a{int(self.alpha_weight*100)}"
            f"_r{int(self.risk_weight*100)}_e{int(self.enhance_weight*100)}"
            f"_{self.position_model}"
        )


def annual_return_from_series(ret: pd.Series) -> float:
    if ret.empty:
        return 0.0
    nav = (1 + ret).cumprod()
    total = nav.iloc[-1] - 1
    days = (nav.index[-1] - nav.index[0]).days
    if days <= 0:
        return 0.0
    return float((1 + total) ** (365 / days) - 1)


def max_drawdown_from_series(ret: pd.Series) -> float:
    if ret.empty:
        return 0.0
    nav = (1 + ret).cumprod()
    return float((nav / nav.cummax() - 1).min())


def sharpe_from_series(ret: pd.Series) -> float:
    if ret.empty or ret.std() == 0:
        return 0.0
    return float(ret.mean() / ret.std() * np.sqrt(252))


def total_return_from_series(ret: pd.Series) -> float:
    if ret.empty:
        return 0.0
    return float((1 + ret).prod() - 1)


def extract_period_metrics(ret: pd.Series, start: str, end: str) -> Dict[str, float]:
    seg = ret[(ret.index >= start) & (ret.index <= end)]
    return {
        "total": total_return_from_series(seg),
        "annual": annual_return_from_series(seg),
        "sharpe": sharpe_from_series(seg),
        "mdd": max_drawdown_from_series(seg),
    }


def candidate_score(metrics: Dict[str, float]) -> float:
    train_ann = metrics["train_annual"]
    valid_ann = metrics["valid_annual"]
    test_ann = metrics["test_annual"]
    year_2025 = metrics["ret_2025"]
    year_2026 = metrics["ret_2026"]
    full_sharpe = metrics["full_sharpe"]
    full_mdd = abs(metrics["full_mdd"])

    # 先压制明显断层，再奖励 2025-2026 与跨期一致性
    phase_floor = min(train_ann, valid_ann, test_ann)
    phase_avg = (train_ann + valid_ann + test_ann) / 3
    dd_penalty = max(0.0, full_mdd - 0.35) * 1.2

    return (
        1.5 * phase_floor +
        0.8 * phase_avg +
        0.7 * year_2025 +
        0.5 * year_2026 +
        0.2 * full_sharpe -
        dd_penalty
    )


def load_sign_map() -> Dict[str, bool]:
    df = pd.read_csv(PROJECT_ROOT / "results" / "single_factor_topk_2019_2026.csv")
    return {
        row["因子"]: ("反向" in str(row["方向"]))
        for _, row in df.iterrows()
    }


def load_base_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DatetimeIndex, MarketPositionController]:
    init_qlib()
    from qlib.data import D

    factor_exprs = get_all_factors()
    factor_names = sorted(
        set(ALPHA_FACTORS + RISK_FACTORS + [f for fs in ENHANCE_BUNDLES.values() for f in fs])
    )
    missing = [f for f in factor_names if f not in factor_exprs]
    if missing:
        raise ValueError(f"缺少因子定义: {missing}")

    print("[1/4] 加载股票池...")
    instruments = D.instruments(market="all")
    df_close = load_features_safe(
        instruments, ["$close"],
        start_time=START_DATE, end_time=END_DATE, freq="day"
    )
    valid = filter_instruments(
        df_close.index.get_level_values("instrument").unique().tolist(),
        exclude_st=False,
    )
    trade_dates = df_close.index.get_level_values("datetime").unique().sort_values()
    print(f"  股票数: {len(valid)}, 交易日: {len(trade_dates)}")

    print("[2/4] 加载候选因子...")
    exprs = [factor_exprs[name] for name in factor_names]
    df_factors = load_features_safe(
        valid, exprs,
        start_time=START_DATE, end_time=END_DATE, freq="day"
    )
    df_factors.columns = factor_names
    medians = df_factors.groupby(level="datetime").transform("median")
    df_factors = df_factors.fillna(medians).fillna(0)

    print("[3/4] 加载收益与市值...")
    df_ret = load_features_safe(
        valid, ["$close / Ref($close, 1) - 1"],
        start_time=START_DATE, end_time=END_DATE, freq="day"
    )
    df_ret.columns = ["daily_ret"]
    df_mv = load_features_safe(
        valid, ["$total_mv"],
        start_time=START_DATE, end_time=END_DATE, freq="day"
    )
    mv_series = df_mv.iloc[:, 0]

    print("[4/4] 加载仓位控制器...")
    controller = MarketPositionController()
    controller.load_market_data()

    return df_factors, df_ret, mv_series, trade_dates, controller


def apply_directions(df_factors: pd.DataFrame, sign_map: Dict[str, bool]) -> pd.DataFrame:
    df = df_factors.copy()
    for name, negate in sign_map.items():
        if negate and name in df.columns:
            df[name] = -df[name]
    return df


def build_signal(
    ranked_df: pd.DataFrame,
    candidate: Candidate,
) -> pd.Series:
    signal = pd.Series(0.0, index=ranked_df.index)

    if candidate.alpha_weight > 0:
        signal = signal + candidate.alpha_weight * ranked_df[ALPHA_FACTORS].mean(axis=1)
    if candidate.risk_weight > 0:
        signal = signal + candidate.risk_weight * ranked_df[RISK_FACTORS].mean(axis=1)
    signal = signal + candidate.enhance_weight * ranked_df[candidate.enhance_factors].mean(axis=1)

    return signal


def backtest_candidate(
    signal: pd.Series,
    trade_dates: pd.DatetimeIndex,
    df_ret: pd.DataFrame,
    mv_series: pd.Series,
    controller: MarketPositionController,
    candidate: Candidate,
) -> BacktestResult:
    rebalance_dates = trade_dates
    df_sel = extract_topk(
        signal=signal,
        rebalance_dates=rebalance_dates,
        topk=candidate.topk,
        mv_floor=candidate.min_cap * 10000,
        mv_series=mv_series,
        buffer=candidate.buffer,
        exclude_new_days=candidate.exclude_new_days,
    )

    date_to_symbols: Dict[pd.Timestamp, set] = {}
    for dt, grp in df_sel.groupby("date"):
        date_to_symbols[pd.Timestamp(dt)] = set(grp["symbol"].tolist())

    if len(date_to_symbols) < 2:
        return BacktestResult(pd.Series(dtype=float), pd.Series(dtype=float))

    all_ret_dates = df_ret.index.get_level_values("datetime").unique().sort_values()
    portfolio_returns = []
    prev_selected = set()
    current_value = INITIAL_CAPITAL

    monthly_dates_list = sorted(date_to_symbols.keys())
    for i, rebal_date in enumerate(monthly_dates_list[:-1]):
        selected = date_to_symbols.get(rebal_date, set())
        if len(selected) < candidate.topk:
            continue

        next_date = monthly_dates_list[i + 1]
        holding_dates = all_ret_dates[(all_ret_dates > rebal_date) & (all_ret_dates <= next_date)]
        penalized_missing = set()

        for j, hd in enumerate(holding_dates):
            try:
                daily_ret = df_ret.xs(hd, level="datetime")
            except KeyError:
                continue

            available_symbols = set(daily_ret.index)
            selected_available = available_symbols & selected
            selected_missing = selected - selected_available
            observed_returns = daily_ret.loc[
                daily_ret.index.isin(selected_available), "daily_ret"
            ]

            newly_missing = selected_missing - penalized_missing
            if newly_missing:
                penalty_returns = pd.Series([-1.0] * len(newly_missing), dtype=float)
                stock_ret = pd.concat(
                    [observed_returns.reset_index(drop=True), penalty_returns],
                    ignore_index=True,
                ).mean()
                penalized_missing.update(newly_missing)
            else:
                stock_ret = observed_returns.mean()

            if np.isnan(stock_ret):
                stock_ret = 0.0

            is_rebal = (j == 0)
            if candidate.position_model == "trend":
                alloc = controller.get_allocation(hd, is_rebalance_day=is_rebal)
                stock_pct = alloc.stock_pct
                bond_daily_ret = alloc.cash_pct * controller.get_bond_daily_return()
            else:
                stock_pct = 0.88
                bond_daily_ret = (1 - stock_pct) * (0.03 / 252)

            cost_deduction = 0.0
            if is_rebal:
                if prev_selected:
                    sell_count = len(prev_selected - selected)
                    buy_count = len(selected - prev_selected)
                else:
                    sell_count = 0
                    buy_count = len(selected)

                per_position_value = current_value * stock_pct / candidate.topk if candidate.topk > 0 else 0.0
                buy_fee_per_order = max(per_position_value * BUY_COMMISSION_RATE, MIN_BUY_COMMISSION) if buy_count > 0 else 0.0
                sell_commission_per_order = max(per_position_value * SELL_COMMISSION_RATE, MIN_SELL_COMMISSION) if sell_count > 0 else 0.0
                sell_stamp_tax_per_order = per_position_value * SELL_STAMP_TAX_RATE if sell_count > 0 else 0.0
                fee_amount = (
                    buy_count * buy_fee_per_order +
                    sell_count * (sell_commission_per_order + sell_stamp_tax_per_order)
                )
                cost_deduction = fee_amount / current_value if current_value > 0 else 0.0

            port_ret = stock_pct * stock_ret + bond_daily_ret - cost_deduction
            portfolio_returns.append({"date": hd, "return": port_ret})
            current_value *= (1 + port_ret)

            if is_rebal:
                prev_selected = selected.copy()

    if not portfolio_returns:
        return BacktestResult(pd.Series(dtype=float), pd.Series(dtype=float))

    df_result = pd.DataFrame(portfolio_returns).set_index("date")
    df_result.index = pd.to_datetime(df_result.index)
    daily_returns = df_result["return"]
    portfolio_value = (1 + daily_returns).cumprod()
    return BacktestResult(daily_returns=daily_returns, portfolio_value=portfolio_value)


def evaluate_candidate(
    ranked_df: pd.DataFrame,
    trade_dates: pd.DatetimeIndex,
    df_ret: pd.DataFrame,
    mv_series: pd.Series,
    controller: MarketPositionController,
    candidate: Candidate,
) -> Optional[Dict[str, object]]:
    signal = build_signal(ranked_df, candidate)
    result = backtest_candidate(signal, trade_dates, df_ret, mv_series, controller, candidate)
    if result.portfolio_value.empty:
        return None

    ret = result.daily_returns.copy()
    year_2025 = extract_period_metrics(ret, "2025-01-01", "2025-12-31")
    year_2026 = extract_period_metrics(ret, "2026-01-01", "2026-12-31")
    train = extract_period_metrics(ret, *PHASES["train_2019_2022"])
    valid = extract_period_metrics(ret, *PHASES["valid_2023_2024"])
    test = extract_period_metrics(ret, *PHASES["test_2025_2026"])

    row = {
        "name": candidate.name(),
        "bundle": candidate.bundle_name,
        "enhance_factors": ",".join(candidate.enhance_factors),
        "topk": candidate.topk,
        "buffer": candidate.buffer,
        "min_cap": candidate.min_cap,
        "exclude_new_days": candidate.exclude_new_days,
        "alpha_weight": candidate.alpha_weight,
        "risk_weight": candidate.risk_weight,
        "enhance_weight": candidate.enhance_weight,
        "position_model": candidate.position_model,
        "full_total": result.total_return,
        "full_annual": result.annual_return,
        "full_sharpe": result.sharpe_ratio,
        "full_mdd": result.max_drawdown,
        "train_annual": train["annual"],
        "valid_annual": valid["annual"],
        "test_annual": test["annual"],
        "ret_2025": year_2025["total"],
        "ret_2026": year_2026["total"],
        "sharpe_2025": year_2025["sharpe"],
        "sharpe_2026": year_2026["sharpe"],
    }
    row["score"] = candidate_score(row)
    return row


def coarse_candidates() -> List[Candidate]:
    out = []
    for bundle_name, enhance_factors in ENHANCE_BUNDLES.items():
        for topk in COARSE_TOPK:
            for buffer in COARSE_BUFFER:
                for min_cap in COARSE_MIN_CAP:
                    out.append(
                        Candidate(
                            bundle_name=bundle_name,
                            enhance_factors=enhance_factors,
                            topk=topk,
                            buffer=buffer,
                            min_cap=min_cap,
                            exclude_new_days=COARSE_EXCLUDE_NEW,
                            alpha_weight=COARSE_WEIGHTS["alpha"],
                            risk_weight=COARSE_WEIGHTS["risk"],
                            enhance_weight=COARSE_WEIGHTS["enhance"],
                            position_model="trend",
                        )
                    )
    return out


def refine_candidates(top_rows: pd.DataFrame) -> List[Candidate]:
    out = []
    for _, row in top_rows.iterrows():
        bundle_name = row["bundle"]
        enhance_factors = ENHANCE_BUNDLES[bundle_name]
        topk = int(row["topk"])
        buffer = int(row["buffer"])
        min_cap = int(row["min_cap"])
        for alpha_weight in REFINE_ALPHA:
            for risk_weight in REFINE_RISK:
                enhance_weight = 1.0 - alpha_weight - risk_weight
                if enhance_weight <= 0:
                    continue
                for exclude_new_days in REFINE_EXCLUDE_NEW:
                    for position_model in REFINE_POSITION:
                        out.append(
                            Candidate(
                                bundle_name=bundle_name,
                                enhance_factors=enhance_factors,
                                topk=topk,
                                buffer=buffer,
                                min_cap=min_cap,
                                exclude_new_days=exclude_new_days,
                                alpha_weight=alpha_weight,
                                risk_weight=risk_weight,
                                enhance_weight=enhance_weight,
                                position_model=position_model,
                            )
                        )
    return out


def search_candidates(
    candidates: List[Candidate],
    ranked_df: pd.DataFrame,
    trade_dates: pd.DatetimeIndex,
    df_ret: pd.DataFrame,
    mv_series: pd.Series,
    controller: MarketPositionController,
    label: str,
) -> pd.DataFrame:
    rows = []
    total = len(candidates)
    t0 = time.time()
    for i, cand in enumerate(candidates, 1):
        row = evaluate_candidate(ranked_df, trade_dates, df_ret, mv_series, controller, cand)
        if row is not None:
            rows.append(row)
        if i % 10 == 0 or i == total:
            elapsed = time.time() - t0
            print(f"[{label}] {i}/{total} elapsed={elapsed:.1f}s")

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["score", "test_annual", "ret_2025"], ascending=False).reset_index(drop=True)
    return df


def print_top(df: pd.DataFrame, title: str, n: int = 15) -> None:
    if df.empty:
        print(f"{title}: 无结果")
        return
    cols = [
        "name", "score", "full_annual", "full_sharpe", "full_mdd",
        "train_annual", "valid_annual", "test_annual", "ret_2025", "ret_2026"
    ]
    print("\n" + "=" * 140)
    print(title)
    print("=" * 140)
    print(df[cols].head(n).to_string(index=False))


def build_registry_from_candidate(candidate: Candidate, sign_map: Dict[str, bool]) -> FactorRegistry:
    factor_exprs = get_all_factors()
    reg = FactorRegistry()

    for name in ALPHA_FACTORS:
        reg.register(FactorInfo(name, factor_exprs[name], "", "alpha", source="qlib", negate=sign_map.get(name, False)))
    for name in RISK_FACTORS:
        reg.register(FactorInfo(name, factor_exprs[name], "", "risk", source="qlib", negate=sign_map.get(name, False)))
    for name in candidate.enhance_factors:
        reg.register(FactorInfo(name, factor_exprs[name], "", "enhance", source="qlib", negate=sign_map.get(name, False)))

    return reg


def official_validate(top_df: pd.DataFrame, sign_map: Dict[str, bool], top_n: int = 3) -> pd.DataFrame:
    rows = []
    engine = QlibBacktestEngine()
    for i, row in top_df.head(top_n).iterrows():
        candidate = Candidate(
            bundle_name=row["bundle"],
            enhance_factors=ENHANCE_BUNDLES[row["bundle"]],
            topk=int(row["topk"]),
            buffer=int(row["buffer"]),
            min_cap=int(row["min_cap"]),
            exclude_new_days=int(row["exclude_new_days"]),
            alpha_weight=float(row["alpha_weight"]),
            risk_weight=float(row["risk_weight"]),
            enhance_weight=float(row["enhance_weight"]),
            position_model=row["position_model"],
        )
        reg = build_registry_from_candidate(candidate, sign_map)
        strategy = Strategy(
            name=f"research_verify_{i+1}",
            description=candidate.name(),
            registry=reg,
            weights={
                "alpha": candidate.alpha_weight,
                "risk": candidate.risk_weight,
                "enhance": candidate.enhance_weight,
            },
            topk=candidate.topk,
            neutralize_industry=False,
            min_market_cap=float(candidate.min_cap),
            exclude_st=True,
            exclude_new_days=candidate.exclude_new_days,
            sticky=0,
            buffer=candidate.buffer,
            position_model=candidate.position_model,
            position_params={"stock_pct": 0.88} if candidate.position_model == "fixed" else {},
            rebalance_freq="day",
            trading_cost={
                "open_cost": BUY_COMMISSION_RATE,
                "close_cost": SELL_COMMISSION_RATE + SELL_STAMP_TAX_RATE,
                "buy_commission_rate": BUY_COMMISSION_RATE,
                "sell_commission_rate": SELL_COMMISSION_RATE,
                "sell_stamp_tax_rate": SELL_STAMP_TAX_RATE,
                "min_buy_commission": MIN_BUY_COMMISSION,
                "min_sell_commission": MIN_SELL_COMMISSION,
            },
        )
        result = engine.run(strategy=strategy)
        ret = result.daily_returns
        rows.append({
            "candidate": candidate.name(),
            "bundle": candidate.bundle_name,
            "full_total": result.total_return,
            "full_annual": result.annual_return,
            "full_sharpe": result.sharpe_ratio,
            "full_mdd": result.max_drawdown,
            "ret_2025": total_return_from_series(ret[(ret.index >= "2025-01-01") & (ret.index <= "2025-12-31")]),
            "ret_2026": total_return_from_series(ret[(ret.index >= "2026-01-01") & (ret.index <= "2026-12-31")]),
            "results_file": result.metadata.get("results_file", ""),
        })
    return pd.DataFrame(rows)


def main() -> None:
    t0 = time.time()
    print("=" * 80)
    print("  Robust Close-Only Strategy Search")
    print("=" * 80)

    sign_map = load_sign_map()
    df_factors, df_ret, mv_series, trade_dates, controller = load_base_data()
    signed_df = apply_directions(df_factors, sign_map)
    ranked_df = signed_df.groupby(level="datetime").rank(pct=True)

    coarse = search_candidates(
        coarse_candidates(),
        ranked_df=ranked_df,
        trade_dates=trade_dates,
        df_ret=df_ret,
        mv_series=mv_series,
        controller=controller,
        label="coarse",
    )
    coarse_path = PROJECT_ROOT / "results" / "robust_close_only_coarse_20260322.csv"
    coarse.to_csv(coarse_path, index=False)
    print_top(coarse, "Coarse Top 15")

    shortlist = coarse[
        (coarse["valid_annual"] > 0)
        & (coarse["test_annual"] > 0)
        & (coarse["ret_2025"] > 0.10)
        & (coarse["ret_2026"] > 0)
        & (coarse["full_mdd"] > -0.50)
    ].head(10)
    if shortlist.empty:
        shortlist = coarse.head(8)

    refined = search_candidates(
        refine_candidates(shortlist),
        ranked_df=ranked_df,
        trade_dates=trade_dates,
        df_ret=df_ret,
        mv_series=mv_series,
        controller=controller,
        label="refine",
    )
    refined_path = PROJECT_ROOT / "results" / "robust_close_only_refine_20260322.csv"
    refined.to_csv(refined_path, index=False)
    print_top(refined, "Refine Top 20", n=20)

    verified = official_validate(refined, sign_map=sign_map, top_n=3)
    verified_path = PROJECT_ROOT / "results" / "robust_close_only_verified_20260322.csv"
    verified.to_csv(verified_path, index=False)

    print("\n" + "=" * 140)
    print("Official Validation Top 3")
    print("=" * 140)
    if verified.empty:
        print("无验证结果")
    else:
        print(verified.to_string(index=False))

    elapsed = time.time() - t0
    print(f"\n耗时: {elapsed:.1f}s")
    print(f"Coarse:  {coarse_path}")
    print(f"Refine:  {refined_path}")
    print(f"Verify:  {verified_path}")


if __name__ == "__main__":
    main()
