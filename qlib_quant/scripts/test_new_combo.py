#!/usr/bin/env python3
"""
验证推荐6因子组合（2024-2026）
bb_upper_dist + vol_oscillator + vol_std_10d + sharpe_120d + pe_ttm + dv_ttm
"""
import warnings
warnings.filterwarnings("ignore")

import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.qlib_init import init_qlib, load_features_safe
from core.universe import filter_instruments
from core.position import MarketPositionController
from core.compute import compute_layer_score

START_DATE = "2024-01-01"
END_DATE = "2026-03-15"
TOPK = 15
BUFFER = 10
OPEN_COST = 0.0003
CLOSE_COST = 0.0013
BOND_DAILY_RET = 0.03 / 252

ret = "($close - Ref($close, 1)) / Ref($close, 1)"

# 推荐6因子 — 每个来自独立逻辑维度
FACTORS = {
    # enhance 层（技术反转+量能）
    "enhance_bb_upper_dist": {
        "expr": "((Mean($close, 20) + 2 * Std($close, 20)) - $close) / $close",
        "category": "enhance",
        "negate": True,  # 反向：值大=离上轨远=超跌
    },
    "enhance_vol_oscillator": {
        "expr": "(EMA($volume, 5) - EMA($volume, 10)) / (EMA($volume, 10) + 1)",
        "category": "enhance",
        "negate": True,  # 反向：缩量
    },
    "enhance_sharpe_120d": {
        "expr": f"Mean({ret}, 120) / (Std({ret}, 120) + 1e-8)",
        "category": "enhance",
        "negate": False,  # 正向：高质量动量
    },
    # risk 层（低波动）
    "risk_vol_std_10d": {
        "expr": "Std($volume, 10) / (Mean($volume, 10) + 1)",
        "category": "risk",
        "negate": True,  # 反向：低波动
    },
    # alpha 层（估值）
    "alpha_pe_ttm": {
        "expr": "$pe_ttm",
        "category": "alpha",
        "negate": False,  # 正向（高PE=成长偏好，近两年夏普1.26）
    },
    "alpha_dv_ttm": {
        "expr": "$dv_ttm",
        "category": "alpha",
        "negate": True,  # 反向：高股息
    },
}

# 多组权重方案
WEIGHT_SCHEMES = {
    "均衡": {"alpha": 0.33, "risk": 0.33, "enhance": 0.34},
    "增强重": {"alpha": 0.20, "risk": 0.20, "enhance": 0.60},
    "风控重": {"alpha": 0.20, "risk": 0.50, "enhance": 0.30},
    "价值重": {"alpha": 0.50, "risk": 0.20, "enhance": 0.30},
    "纯增强": {"alpha": 0.00, "risk": 0.00, "enhance": 1.00},
}


def main():
    t0 = time.time()
    print("=" * 70)
    print("  6因子组合回测验证（2024-2026）")
    print("=" * 70)

    init_qlib()
    from qlib.data import D

    # 加载股票列表
    print("[1/4] 加载股票列表...")
    instruments = D.instruments(market="all")
    df_close = load_features_safe(
        instruments, ["$close"],
        start_time=START_DATE, end_time=END_DATE, freq="day"
    )
    valid = filter_instruments(
        df_close.index.get_level_values("instrument").unique().tolist()
    )
    all_dates = df_close.index.get_level_values("datetime").unique().sort_values()
    print(f"  股票: {len(valid)}, 交易日: {len(all_dates)}")

    # 加载因子数据
    print("[2/4] 加载因子数据...")
    exprs = [f["expr"] for f in FACTORS.values()]
    names = list(FACTORS.keys())

    df_factors = load_features_safe(
        valid, exprs,
        start_time=START_DATE, end_time=END_DATE, freq="day"
    )
    df_factors.columns = names

    # 处理 negate
    for name, info in FACTORS.items():
        if info["negate"]:
            df_factors[name] = -df_factors[name]

    # 截面中位数填充
    medians = df_factors.groupby(level="datetime").transform("median")
    df_factors = df_factors.fillna(medians).fillna(0)
    print(f"  因子数据: {df_factors.shape}")

    # 加载收益
    print("[3/4] 加载日收益...")
    df_ret = load_features_safe(
        valid, ["$close / Ref($close, 1) - 1"],
        start_time=START_DATE, end_time=END_DATE, freq="day"
    )
    df_ret.columns = ["daily_ret"]

    # 加载仓位控制器
    controller = MarketPositionController()
    controller.load_market_data()

    # 对每种权重方案回测
    print("[4/4] 回测...")
    print("-" * 70)

    results_all = []

    for scheme_name, weights in WEIGHT_SCHEMES.items():
        # 计算综合信号
        signal = pd.Series(0.0, index=df_factors.index)
        for cat in ["alpha", "risk", "enhance"]:
            cat_cols = [n for n, info in FACTORS.items() if info["category"] == cat]
            w = weights.get(cat, 0.0)
            if w > 0 and cat_cols:
                signal = signal + w * compute_layer_score(df_factors, cat_cols)

        # 日度选股 + 缓冲区 → T+1收益
        dates = signal.index.get_level_values("datetime").unique().sort_values()
        prev_symbols = set()
        portfolio_returns = []
        prev_selected = set()
        turnover_list = []

        for dt in dates:
            try:
                day_scores = signal.xs(dt, level="datetime")
            except KeyError:
                continue
            if len(day_scores) < TOPK:
                continue

            day_sorted = day_scores.sort_values(ascending=False)
            day_index_set = set(day_scores.index)

            if BUFFER > 0 and prev_symbols:
                wide_set = set(day_sorted.head(TOPK + BUFFER).index)
                keep_set = prev_symbols & wide_set & day_index_set
                remaining = TOPK - len(keep_set)
                if remaining > 0:
                    available = day_index_set - keep_set
                    new_scores = day_scores[list(available)].nlargest(min(remaining, len(available)))
                    selected = keep_set | set(new_scores.index)
                elif len(keep_set) > TOPK:
                    keep_scores = day_scores[list(keep_set)].nlargest(TOPK)
                    selected = set(keep_scores.index)
                else:
                    selected = keep_set
            else:
                selected = set(day_sorted.head(TOPK).index)

            selected = selected & day_index_set
            prev_symbols = selected.copy()

            # T+1收益
            dt_idx = all_dates.searchsorted(dt)
            if dt_idx + 1 >= len(all_dates):
                continue
            next_dt = all_dates[dt_idx + 1]

            try:
                daily_ret = df_ret.xs(next_dt, level="datetime")
                stock_ret = daily_ret.loc[daily_ret.index.isin(selected), "daily_ret"].mean()
                if np.isnan(stock_ret):
                    stock_ret = 0.0
            except (KeyError, IndexError):
                continue

            sell_count = 0
            buy_count = 0
            if prev_selected:
                kept = prev_selected & selected
                sell_count = len(prev_selected - kept)
                buy_count = len(selected - kept)
                turnover = (sell_count + buy_count) / (2 * TOPK)
                turnover_list.append(turnover)
                cost = (sell_count * CLOSE_COST + buy_count * OPEN_COST) / len(selected)
            else:
                buy_count = len(selected)
                cost = buy_count * OPEN_COST / len(selected)
                turnover_list.append(1.0)

            is_rebal = (sell_count > 0 or buy_count > 0)
            alloc = controller.get_allocation(next_dt, is_rebalance_day=is_rebal)
            port_ret = alloc.stock_pct * stock_ret + alloc.cash_pct * BOND_DAILY_RET - cost

            portfolio_returns.append({"date": next_dt, "return": port_ret})
            prev_selected = selected.copy()

        if not portfolio_returns:
            print(f"  {scheme_name}: 无数据")
            continue

        df_result = pd.DataFrame(portfolio_returns).set_index("date")
        daily_returns = df_result["return"]
        portfolio_value = (1 + daily_returns).cumprod()

        total_ret = portfolio_value.iloc[-1] - 1
        n_years = len(daily_returns) / 252
        annual_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
        sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
        max_dd = (portfolio_value / portfolio_value.cummax() - 1).min()
        avg_turnover = np.mean(turnover_list) if turnover_list else 0
        win_rate = (daily_returns > 0).mean()

        # 分年
        yr_str_parts = []
        for year in [2024, 2025, 2026]:
            yr_ret = daily_returns[daily_returns.index.year == year]
            if len(yr_ret) > 20:
                yr_total = (1 + yr_ret).prod() - 1
                yr_str_parts.append(f"{year}:{yr_total:+.1%}")

        yr_str = " | ".join(yr_str_parts)

        w_str = "/".join(f"{v:.0%}" for v in weights.values())
        print(f"  {scheme_name:6s} (α/r/e={w_str}) "
              f"年化:{annual_ret:+.1%} 夏普:{sharpe:.2f} "
              f"回撤:{max_dd:.1%} 换手:{avg_turnover:.2f} 胜率:{win_rate:.1%}")
        print(f"         {yr_str}")

        results_all.append({
            "方案": scheme_name,
            "权重": w_str,
            "年化收益": annual_ret,
            "夏普比率": sharpe,
            "最大回撤": max_dd,
            "平均换手": avg_turnover,
            "日胜率": win_rate,
        })

    print("\n" + "=" * 70)
    elapsed = time.time() - t0
    print(f"耗时: {elapsed:.1f}秒")

    # 对比当前策略（原7因子）
    print("\n[对比] 当前策略也跑一遍...")
    from core.factors import default_registry
    from core.selection import compute_signal, _load_parquet_factors, _fill_cross_sectional

    # 加载当前因子
    qlib_factors = default_registry.get_by_source("qlib")
    qlib_fields = [f.expression for f in qlib_factors]
    qlib_names = [f"{f.category}_{f.name}" for f in qlib_factors]

    df_qlib = load_features_safe(
        valid, qlib_fields,
        start_time=START_DATE, end_time=END_DATE, freq="day"
    )
    df_qlib.columns = qlib_names

    df_parquet = _load_parquet_factors(valid, START_DATE, END_DATE, registry=default_registry)
    if not df_parquet.empty:
        df_current = df_qlib.join(df_parquet, how="left")
    else:
        df_current = df_qlib
    df_current = _fill_cross_sectional(df_current)

    current_signal = compute_signal(
        df_current, registry=default_registry,
        weights={"alpha": 0.15, "risk": 0.35, "enhance": 0.50},
        neutralize_industry=True,
    )

    # 同样的回测
    dates = current_signal.index.get_level_values("datetime").unique().sort_values()
    prev_symbols = set()
    portfolio_returns = []
    prev_selected = set()

    for dt in dates:
        try:
            day_scores = current_signal.xs(dt, level="datetime")
        except KeyError:
            continue
        if len(day_scores) < TOPK:
            continue

        day_sorted = day_scores.sort_values(ascending=False)
        day_index_set = set(day_scores.index)

        if BUFFER > 0 and prev_symbols:
            wide_set = set(day_sorted.head(TOPK + BUFFER).index)
            keep_set = prev_symbols & wide_set & day_index_set
            remaining = TOPK - len(keep_set)
            if remaining > 0:
                available = day_index_set - keep_set
                new_scores = day_scores[list(available)].nlargest(min(remaining, len(available)))
                selected = keep_set | set(new_scores.index)
            elif len(keep_set) > TOPK:
                keep_scores = day_scores[list(keep_set)].nlargest(TOPK)
                selected = set(keep_scores.index)
            else:
                selected = keep_set
        else:
            selected = set(day_sorted.head(TOPK).index)

        selected = selected & day_index_set
        prev_symbols = selected.copy()

        dt_idx = all_dates.searchsorted(dt)
        if dt_idx + 1 >= len(all_dates):
            continue
        next_dt = all_dates[dt_idx + 1]

        try:
            daily_ret = df_ret.xs(next_dt, level="datetime")
            stock_ret = daily_ret.loc[daily_ret.index.isin(selected), "daily_ret"].mean()
            if np.isnan(stock_ret):
                stock_ret = 0.0
        except (KeyError, IndexError):
            continue

        sell_count = 0
        buy_count = 0
        if prev_selected:
            kept = prev_selected & selected
            sell_count = len(prev_selected - kept)
            buy_count = len(selected - kept)
            cost = (sell_count * CLOSE_COST + buy_count * OPEN_COST) / len(selected)
        else:
            buy_count = len(selected)
            cost = buy_count * OPEN_COST / len(selected)

        is_rebal = (sell_count > 0 or buy_count > 0)
        alloc = controller.get_allocation(next_dt, is_rebalance_day=is_rebal)
        port_ret = alloc.stock_pct * stock_ret + alloc.cash_pct * BOND_DAILY_RET - cost

        portfolio_returns.append({"date": next_dt, "return": port_ret})
        prev_selected = selected.copy()

    if portfolio_returns:
        df_result = pd.DataFrame(portfolio_returns).set_index("date")
        daily_returns = df_result["return"]
        portfolio_value = (1 + daily_returns).cumprod()
        total_ret = portfolio_value.iloc[-1] - 1
        n_years = len(daily_returns) / 252
        annual_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
        sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
        max_dd = (portfolio_value / portfolio_value.cummax() - 1).min()

        yr_str_parts = []
        for year in [2024, 2025, 2026]:
            yr_ret = daily_returns[daily_returns.index.year == year]
            if len(yr_ret) > 20:
                yr_total = (1 + yr_ret).prod() - 1
                yr_str_parts.append(f"{year}:{yr_total:+.1%}")

        print(f"  当前策略（v3 7因子 α15/r35/e50）"
              f"年化:{annual_ret:+.1%} 夏普:{sharpe:.2f} 回撤:{max_dd:.1%}")
        print(f"         {' | '.join(yr_str_parts)}")


if __name__ == "__main__":
    main()
