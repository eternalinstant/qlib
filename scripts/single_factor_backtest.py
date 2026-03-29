#!/usr/bin/env python3
"""
单因子 Top-K 真实口径扫描器

按当前正式研究口径逐一评估单因子：
  因子值排序 -> Top-K 选股 -> T+1 close 执行回测

默认基线对齐当前日频低换手实验：
  - freq=day
  - topk=15
  - buffer=20
  - churn_limit=2
  - 可选事件驱动执行参数
  - stock_pct=0.88
  - exclude_st / exclude_new_days / min_market_cap 生效
  - 交易成本按正式策略口径扣除
"""
import warnings
warnings.filterwarnings("ignore")

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import CONFIG
from core.qlib_init import init_qlib, load_features_safe
from core.selection import (
    _load_total_mv_frame,
    compute_rebalance_dates,
    extract_topk,
    FACTOR_PARQUET,
)
from core.universe import filter_instruments
from modules.backtest.qlib_engine import _load_bond_etf_returns, _sum_symbol_returns
from scripts.factor_scan import get_all_factors, get_weekly_factors


def _resolve_end_date(value: str) -> str:
    if not value or value == "auto":
        return pd.Timestamp.today().strftime("%Y-%m-%d")
    return value


DEFAULT_START = "2019-01-01"
DEFAULT_END = _resolve_end_date(CONFIG.get("end_date", "auto"))


@dataclass
class ScanConfig:
    start_date: str
    end_date: str
    freq: str
    topk: int
    buffer: int
    sticky: int
    churn_limit: int
    margin_stable: bool
    score_smoothing_days: int
    entry_rank: Optional[int]
    exit_rank: Optional[int]
    entry_persist_days: int
    exit_persist_days: int
    min_hold_days: int
    stock_pct: float
    initial_capital: float
    universe: str
    min_market_cap: float
    exclude_st: bool
    exclude_new_days: int
    buy_commission_rate: float
    sell_commission_rate: float
    sell_stamp_tax_rate: float
    slippage_bps: float
    impact_bps: float
    factor_names: Optional[list]
    output_tag: str

    @property
    def min_buy_commission(self) -> float:
        return 5.0

    @property
    def min_sell_commission(self) -> float:
        return 5.0

    @property
    def execution_rate(self) -> float:
        return (self.slippage_bps + self.impact_bps) / 10000.0


def parse_args() -> ScanConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--freq", default="day", choices=["day", "week", "biweek", "month"])
    parser.add_argument("--topk", type=int, default=15)
    parser.add_argument("--buffer", type=int, default=20)
    parser.add_argument("--sticky", type=int, default=5)
    parser.add_argument("--churn-limit", type=int, default=2)
    parser.add_argument("--score-smoothing-days", type=int, default=1)
    parser.add_argument("--entry-rank", type=int)
    parser.add_argument("--exit-rank", type=int)
    parser.add_argument("--entry-persist-days", type=int, default=1)
    parser.add_argument("--exit-persist-days", type=int, default=1)
    parser.add_argument("--min-hold-days", type=int, default=0)
    parser.add_argument("--stock-pct", type=float, default=0.88)
    parser.add_argument("--initial-capital", type=float, default=1_000_000.0)
    parser.add_argument("--min-market-cap", type=float, default=50.0)
    parser.add_argument("--exclude-new-days", type=int, default=120)
    parser.add_argument("--universe", default="all", choices=["all", "csi300"])
    parser.add_argument("--buy-commission-rate", type=float, default=0.0003)
    parser.add_argument("--sell-commission-rate", type=float, default=0.0003)
    parser.add_argument("--sell-stamp-tax-rate", type=float, default=0.0010)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--impact-bps", type=float, default=5.0)
    parser.add_argument("--disable-margin-stable", action="store_true")
    parser.add_argument("--keep-st", action="store_true")
    parser.add_argument("--factors", default="")
    parser.add_argument("--output-tag", default="")
    args = parser.parse_args()

    factor_names = [x.strip() for x in args.factors.split(",") if x.strip()] or None

    return ScanConfig(
        start_date=args.start,
        end_date=_resolve_end_date(args.end),
        freq=args.freq,
        topk=args.topk,
        buffer=args.buffer,
        sticky=args.sticky,
        churn_limit=args.churn_limit,
        margin_stable=not args.disable_margin_stable,
        score_smoothing_days=args.score_smoothing_days,
        entry_rank=args.entry_rank,
        exit_rank=args.exit_rank,
        entry_persist_days=args.entry_persist_days,
        exit_persist_days=args.exit_persist_days,
        min_hold_days=args.min_hold_days,
        stock_pct=args.stock_pct,
        initial_capital=args.initial_capital,
        universe=args.universe,
        min_market_cap=args.min_market_cap,
        exclude_st=not args.keep_st,
        exclude_new_days=args.exclude_new_days,
        buy_commission_rate=args.buy_commission_rate,
        sell_commission_rate=args.sell_commission_rate,
        sell_stamp_tax_rate=args.sell_stamp_tax_rate,
        slippage_bps=args.slippage_bps,
        impact_bps=args.impact_bps,
        factor_names=factor_names,
        output_tag=args.output_tag.strip(),
    )


def load_base_data(cfg: ScanConfig):
    init_qlib()
    from qlib.data import D

    print("[1/4] 加载股票列表...")
    instruments = D.instruments(market="all")
    df_close = load_features_safe(
        instruments,
        ["$close"],
        start_time=cfg.start_date,
        end_time=cfg.end_date,
        freq="day",
    )
    valid_instruments = filter_instruments(
        df_close.index.get_level_values("instrument").unique().tolist(),
        exclude_st=False,
    )
    print(f"  股票: {len(valid_instruments)}")

    print("[2/4] 加载收益率...")
    df_ret = load_features_safe(
        valid_instruments,
        ["$close / Ref($close, 1) - 1"],
        start_time=cfg.start_date,
        end_time=cfg.end_date,
        freq="day",
    )
    df_ret.columns = ["daily_ret"]
    all_dates = df_ret.index.get_level_values("datetime").unique().sort_values()
    rebalance_dates = compute_rebalance_dates(pd.Series(all_dates), freq=cfg.freq)
    print(f"  交易日: {len(all_dates)}, 调仓日: {len(rebalance_dates)}")

    print("[3/4] 加载债券替代收益...")
    bond_etf_returns = _load_bond_etf_returns()
    if bond_etf_returns is not None:
        print(f"  国债ETF收益: {len(bond_etf_returns)} 个交易日")
    else:
        print("  未找到国债ETF数据，回退到固定日收益")

    mv_series = None
    mv_floor = 0.0
    if cfg.min_market_cap > 0 and FACTOR_PARQUET.exists():
        print("[4/4] 加载市值过滤数据...")
        total_mv_frame = _load_total_mv_frame(
            instruments=valid_instruments,
            start_date=cfg.start_date,
            end_date=cfg.end_date,
        )
        mv_df = total_mv_frame[total_mv_frame["datetime"].isin(rebalance_dates)]
        mv_series = mv_df.set_index(["datetime", "symbol"])["total_mv"]
        mv_floor = cfg.min_market_cap * 10000
        print(f"  市值快照: {len(mv_df)} 行")
    else:
        print("[4/4] 跳过市值过滤数据")

    return valid_instruments, df_ret, all_dates, rebalance_dates, bond_etf_returns, mv_series, mv_floor


def load_factor_batch(valid, factor_dict: Dict[str, str], cfg: ScanConfig):
    items = list(factor_dict.items())
    batch_size = 5
    df_parts = []
    failed = []

    for b in range(0, len(items), batch_size):
        batch = items[b:b + batch_size]
        names = [n for n, _ in batch]
        exprs = [e for _, e in batch]
        try:
            df_part = load_features_safe(
                valid,
                exprs,
                start_time=cfg.start_date,
                end_time=cfg.end_date,
                freq="day",
            )
            df_part.columns = names
            df_parts.append(df_part)
            done = min(b + batch_size, len(items))
            print(f"  [{done}/{len(items)}] OK")
        except Exception:
            for name, expr in batch:
                try:
                    df_single = load_features_safe(
                        valid,
                        [expr],
                        start_time=cfg.start_date,
                        end_time=cfg.end_date,
                        freq="day",
                    )
                    df_single.columns = [name]
                    df_parts.append(df_single)
                except Exception as e:
                    failed.append((name, str(e)[:80]))
                    print(f"  FAIL: {name} - {str(e)[:80]}")

    df_all = pd.concat(df_parts, axis=1) if df_parts else pd.DataFrame()
    print(f"  成功加载: {len(df_all.columns)} 个因子")
    if failed:
        print(f"  失败因子: {len(failed)}")
    return df_all, failed


def _compute_fee(pre_fee_value: float, cfg: ScanConfig, buy_count: int, sell_count: int) -> float:
    if cfg.topk <= 0 or pre_fee_value <= 0:
        return 0.0
    per_position_value = pre_fee_value * cfg.stock_pct / cfg.topk
    if per_position_value <= 0:
        return 0.0

    buy_fee_per_order = max(per_position_value * cfg.buy_commission_rate, cfg.min_buy_commission) if buy_count > 0 else 0.0
    sell_commission_per_order = max(per_position_value * cfg.sell_commission_rate, cfg.min_sell_commission) if sell_count > 0 else 0.0
    sell_stamp_tax_per_order = per_position_value * cfg.sell_stamp_tax_rate if sell_count > 0 else 0.0
    execution_cost_per_order = per_position_value * cfg.execution_rate
    return (
        buy_count * buy_fee_per_order
        + sell_count * (sell_commission_per_order + sell_stamp_tax_per_order)
        + (buy_count + sell_count) * execution_cost_per_order
    )


def single_factor_backtest(
    factor_values: pd.Series,
    df_ret: pd.DataFrame,
    all_dates: pd.DatetimeIndex,
    rebalance_dates: pd.DatetimeIndex,
    cfg: ScanConfig,
    bond_etf_returns: Optional[pd.Series],
    mv_series: Optional[pd.Series],
    mv_floor: float,
    negate: bool = False,
):
    signal = factor_values.replace([np.inf, -np.inf], np.nan).dropna()
    if negate:
        signal = -signal
    if signal.empty:
        return None

    df_sel = extract_topk(
        signal,
        rebalance_dates,
        topk=cfg.topk,
        mv_floor=mv_floor,
        mv_series=mv_series,
        sticky=cfg.sticky,
        threshold=0.0,
        churn_limit=cfg.churn_limit,
        margin_stable=cfg.margin_stable,
        buffer=cfg.buffer,
        score_smoothing_days=cfg.score_smoothing_days,
        entry_rank=cfg.entry_rank,
        exit_rank=cfg.exit_rank,
        entry_persist_days=cfg.entry_persist_days,
        exit_persist_days=cfg.exit_persist_days,
        min_hold_days=cfg.min_hold_days,
        exclude_new_days=cfg.exclude_new_days,
        exclude_st=cfg.exclude_st,
        universe=cfg.universe,
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

    current_value = cfg.initial_capital
    current_held_symbols = set()
    current_cash_slot_count = cfg.topk
    current_stock_pct = 0.0
    default_bond_daily_ret = 0.03 / 252
    portfolio_returns = []
    turnover_list = []
    total_fee_amount = 0.0

    for i, rebal_date in enumerate(sorted_rebal_dates[:-1]):
        selected = date_to_symbols.get(rebal_date, set())
        next_date = sorted_rebal_dates[i + 1]
        holding_dates = all_dates[(all_dates > rebal_date) & (all_dates <= next_date)]
        penalized_missing = set()

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
            stock_slot_return = held_sum / cfg.topk if cfg.topk > 0 else 0.0
            missing_count = len(held_missing)

            if bond_etf_returns is not None and hd in bond_etf_returns.index:
                bond_daily_ret = float(bond_etf_returns.loc[hd])
            else:
                bond_daily_ret = default_bond_daily_ret

            stock_return_component = stock_slot_return + (
                current_cash_slot_count / cfg.topk * bond_daily_ret if cfg.topk > 0 else 0.0
            )
            gross_port_ret = current_stock_pct * stock_return_component + (1 - current_stock_pct) * bond_daily_ret
            pre_fee_value = current_value * (1 + gross_port_ret)

            fee_amount = 0.0
            buy_count = 0
            sell_count = 0
            next_held_symbols = current_held_symbols.copy()
            next_cash_slot_count = current_cash_slot_count

            if j == 0:
                buy_count = len(selected - current_held_symbols)
                sell_count = len(current_held_symbols - selected)
                next_held_symbols = set(selected)
                next_cash_slot_count = max(cfg.topk - len(next_held_symbols), 0)
                turnover_list.append((buy_count + sell_count) / (2 * cfg.topk) if cfg.topk > 0 else 0.0)
                fee_amount = _compute_fee(pre_fee_value, cfg, buy_count, sell_count)

            end_value = pre_fee_value - fee_amount
            port_ret = end_value / current_value - 1 if current_value > 0 else 0.0
            portfolio_returns.append(
                {
                    "date": hd,
                    "return": port_ret,
                    "gross_return": gross_port_ret,
                    "fee_amount": fee_amount,
                    "buy_count": buy_count if j == 0 else 0,
                    "sell_count": sell_count if j == 0 else 0,
                    "missing_count": missing_count,
                }
            )
            total_fee_amount += fee_amount
            current_value = end_value

            if j == 0:
                current_held_symbols = next_held_symbols.copy()
                current_cash_slot_count = next_cash_slot_count
                current_stock_pct = cfg.stock_pct

    if not portfolio_returns:
        return None

    df_result = pd.DataFrame(portfolio_returns).set_index("date")
    daily_returns = df_result["return"]
    gross_returns = df_result["gross_return"]
    nav = (1 + daily_returns).cumprod()
    gross_nav = (1 + gross_returns).cumprod()

    total_ret = float(nav.iloc[-1] - 1)
    gross_total_ret = float(gross_nav.iloc[-1] - 1)
    years = len(daily_returns) / 252
    ending_nav = float(nav.iloc[-1])
    annual_ret = (ending_nav ** (1 / years) - 1) if years > 0 and ending_nav > 0 else -1.0
    sharpe = float(daily_returns.mean() / (daily_returns.std() + 1e-10) * np.sqrt(252))
    max_dd = float((nav / nav.cummax() - 1).min())
    avg_turnover = float(np.mean(turnover_list)) if turnover_list else 0.0
    win_rate = float((daily_returns > 0).mean())

    yearly = {}
    all_years = sorted(daily_returns.index.year.unique())
    for year in all_years:
        yr_ret = daily_returns[daily_returns.index.year == year]
        if len(yr_ret) > 20:
            yearly[year] = {
                "ret": float((1 + yr_ret).prod() - 1),
                "sharpe": float(yr_ret.mean() / (yr_ret.std() + 1e-10) * np.sqrt(252)),
            }

    return {
        "total_return": total_ret,
        "gross_total_return": gross_total_ret,
        "annual_return": annual_ret,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "avg_turnover": avg_turnover,
        "win_rate": win_rate,
        "total_fee_amount": total_fee_amount,
        "sum_turns": int(df_result["buy_count"].sum() + df_result["sell_count"].sum()),
        "active_rebalance_days": int(((df_result["buy_count"] + df_result["sell_count"]) > 0).sum()),
        "yearly": yearly,
    }


def main():
    cfg = parse_args()
    t0 = time.time()

    print("=" * 90)
    print("  单因子 Top-K 真实口径扫描")
    print("=" * 90)
    print(
        f"  区间={cfg.start_date} ~ {cfg.end_date} | freq={cfg.freq} | topk={cfg.topk} | "
        f"buffer={cfg.buffer} | sticky={cfg.sticky} | churn_limit={cfg.churn_limit} | "
        f"stock_pct={cfg.stock_pct:.2f} | initial_capital={cfg.initial_capital:,.0f}"
    )
    if any(
        [
            cfg.score_smoothing_days > 1,
            cfg.entry_rank is not None,
            cfg.exit_rank is not None,
            cfg.entry_persist_days > 1,
            cfg.exit_persist_days > 1,
            cfg.min_hold_days > 0,
        ]
    ):
        print(
            "  事件驱动参数="
            f"smooth:{cfg.score_smoothing_days} "
            f"entry:{cfg.entry_rank or cfg.topk}/{cfg.entry_persist_days}d "
            f"exit:{cfg.exit_rank or (cfg.topk + cfg.buffer)}/{cfg.exit_persist_days}d "
            f"min_hold:{cfg.min_hold_days}d"
        )

    valid, df_ret, all_dates, rebalance_dates, bond_etf_returns, mv_series, mv_floor = load_base_data(cfg)

    print("\n[加载因子数据]")
    if cfg.freq == "week":
        factor_dict = get_weekly_factors()
        print("  使用周频专用因子（周期×5）")
    else:
        factor_dict = get_all_factors()
    if cfg.factor_names:
        factor_dict = {k: v for k, v in factor_dict.items() if k in set(cfg.factor_names)}
        print(f"  过滤后因子数: {len(factor_dict)}")
    df_all, failed = load_factor_batch(valid, factor_dict, cfg)

    print(f"\n[开始回测] {len(df_all.columns)} 个因子")
    print("-" * 90)

    results = []
    for i, col in enumerate(df_all.columns, 1):
        factor_values = df_all[col]

        r_pos = single_factor_backtest(
            factor_values,
            df_ret,
            all_dates,
            rebalance_dates,
            cfg,
            bond_etf_returns,
            mv_series,
            mv_floor,
            negate=False,
        )
        r_neg = single_factor_backtest(
            factor_values,
            df_ret,
            all_dates,
            rebalance_dates,
            cfg,
            bond_etf_returns,
            mv_series,
            mv_floor,
            negate=True,
        )

        if r_pos is None and r_neg is None:
            print(f"  [{i:2d}/{len(df_all.columns)}] {col:30s} SKIP")
            continue

        if r_pos is None:
            best, best_dir = r_neg, "反向"
        elif r_neg is None:
            best, best_dir = r_pos, "正向"
        else:
            best, best_dir = (r_neg, "反向") if r_neg["sharpe"] > r_pos["sharpe"] else (r_pos, "正向")

        row = {
            "因子": col,
            "方向": best_dir,
            "年化收益": best["annual_return"],
            "夏普比率": best["sharpe"],
            "最大回撤": best["max_drawdown"],
            "总收益": best["total_return"],
            "扣费前总收益": best["gross_total_return"],
            "日胜率": best["win_rate"],
            "平均换手": best["avg_turnover"],
            "累计费用": best["total_fee_amount"],
            "总买卖次数": best["sum_turns"],
            "活跃调仓日": best["active_rebalance_days"],
        }
        for year in sorted(best["yearly"].keys()):
            row[f"收益_{year}"] = best["yearly"][year]["ret"]
            row[f"夏普_{year}"] = best["yearly"][year]["sharpe"]
        results.append(row)

        print(
            f"  [{i:2d}/{len(df_all.columns)}] {col:30s} {best_dir} "
            f"年化:{best['annual_return']:+.1%} 夏普:{best['sharpe']:.2f} "
            f"回撤:{best['max_drawdown']:.1%} 费用:{best['total_fee_amount']:.0f}"
        )

    df_results = pd.DataFrame(results).sort_values("夏普比率", ascending=False)

    tag_suffix = f"_{cfg.output_tag}" if cfg.output_tag else ""
    out_path = PROJECT_ROOT / "results" / (
        f"single_factor_topk_realistic_{cfg.start_date[:4]}_{cfg.end_date[:4]}_"
        f"{cfg.freq}_k{cfg.topk}_b{cfg.buffer}_c{cfg.churn_limit}"
        f"_s{cfg.score_smoothing_days}"
        f"_er{cfg.entry_rank if cfg.entry_rank is not None else 'na'}"
        f"_xr{cfg.exit_rank if cfg.exit_rank is not None else 'na'}"
        f"_ep{cfg.entry_persist_days}_xp{cfg.exit_persist_days}_mh{cfg.min_hold_days}"
        f"{tag_suffix}.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(out_path, index=False, encoding="utf-8-sig", float_format="%.6f")

    print("\n" + "=" * 120)
    print("  单因子真实口径排行（按夏普降序）")
    print("=" * 120)
    display_cols = [
        "因子",
        "方向",
        "年化收益",
        "夏普比率",
        "最大回撤",
        "总收益",
        "扣费前总收益",
        "日胜率",
        "平均换手",
        "累计费用",
        "总买卖次数",
    ]
    display_cols = [c for c in display_cols if c in df_results.columns]
    print(df_results[display_cols].head(30).to_string(index=False))

    if failed:
        print(f"\n失败因子 {len(failed)} 个")
    print(f"\n结果已保存: {out_path}")
    print(f"总耗时: {(time.time() - t0)/60:.1f} 分钟")
    return df_results


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
