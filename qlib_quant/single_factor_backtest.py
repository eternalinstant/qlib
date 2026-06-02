"""
单因子回测：对已注册因子逐一进行 Top-K 选股回测（向量化版本）
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from config.config import CONFIG
from core.factors import default_registry, FactorRegistry
from core.qlib_init import init_qlib, load_features_safe
from core.universe import filter_instruments
from core.selection import (
    _load_parquet_factors, compute_rebalance_dates, FACTOR_PARQUET
)

TOPK = 15
FREQ = "week"
START = CONFIG.get("start_date", "2019-01-01")
END = CONFIG.get("end_date")
BOND_RET = 0.03 / 252
OPEN_COST = 0.0008
CLOSE_COST = 0.0018


def run_single_factor_backtest(factor_name, factor_info, df_ret_unstacked, dates, df_qlib_all, df_parquet_all, valid_set):
    """对单因子运行 Top-K 回测，返回结果 dict"""

    col_name = f"{factor_info.category}_{factor_info.name}"

    if factor_info.source == "qlib":
        if col_name not in df_qlib_all.columns:
            return None
        signal = df_qlib_all[col_name].copy()
        if factor_info.negate:
            signal = -signal
    else:
        if col_name not in df_parquet_all.columns:
            return None
        signal = df_parquet_all[col_name].copy()

    # 去 NaN/Inf
    signal = signal.replace([np.inf, -np.inf], np.nan).dropna()
    if len(signal) == 0:
        return None

    # 截面 rank
    signal = signal.groupby(level="datetime").rank(pct=True)

    # 调仓日
    sig_dates = pd.DatetimeIndex(signal.index.get_level_values("datetime").unique()).sort_values()
    rebal_dates = compute_rebalance_dates(pd.Series(sig_dates), freq=FREQ)
    sig_dt_set = set(sig_dates)

    # 回测
    portfolio_returns = []
    prev_holdings = set()

    for i, rdate in enumerate(rebal_dates):
        if rdate not in sig_dt_set:
            continue

        # 取截面信号
        cross = signal.xs(rdate, level="datetime").sort_values(ascending=False)
        # 只保留有效股票
        cross = cross[cross.index.isin(valid_set)]
        top_stocks = cross.head(TOPK).index.tolist()

        if not top_stocks:
            continue

        # 交易成本
        new_holdings = set(top_stocks)
        turnover = len(new_holdings - prev_holdings) / max(len(new_holdings), 1)
        cost = turnover * (OPEN_COST + CLOSE_COST)
        prev_holdings = new_holdings

        # 持仓期间收益（向量化）
        next_rdate = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else dates[-1]
        hold_mask = (dates > rdate) & (dates <= next_rdate)
        hold_dates = dates[hold_mask]

        if len(hold_dates) == 0:
            continue

        # 从 unstacked df_ret 取出持仓期间 & 持仓股票的收益
        hold_ret = df_ret_unstacked.loc[hold_dates, top_stocks]
        daily_avg = hold_ret.mean(axis=1)  # 等权平均

        # 组合收益 = 股票部分 * 仓位 + 债券部分
        port_ret = daily_avg * 0.8 + BOND_RET * 0.2

        # 第一天扣交易成本
        port_ret.iloc[0] -= cost

        valid = port_ret.dropna()
        portfolio_returns.extend(zip(valid.index, valid.values))

    if not portfolio_returns:
        return None

    df_pnl = pd.DataFrame(portfolio_returns, columns=["date", "ret"]).set_index("date")
    nav = (1 + df_pnl["ret"]).cumprod()

    total_days = len(df_pnl)
    years = total_days / 252
    ann_ret = nav.iloc[-1] ** (1 / years) - 1 if years > 0 else 0
    sharpe = df_pnl["ret"].mean() / (df_pnl["ret"].std() + 1e-10) * np.sqrt(252)
    drawdown = (nav / nav.cummax() - 1).min()
    win_rate = (df_pnl["ret"] > 0).mean()

    return {
        "factor": factor_name,
        "category": factor_info.category,
        "ir_scan": factor_info.ir,
        "negate": factor_info.negate,
        "ann_ret": ann_ret,
        "sharpe": sharpe,
        "max_dd": drawdown,
        "win_rate": win_rate,
        "total_ret": nav.iloc[-1] - 1,
    }


def main():
    print("=" * 70)
    print(f"  单因子回测（Top-{TOPK}，{FREQ}调仓，80%仓位）")
    print("=" * 70)

    init_qlib()
    from qlib.data import D

    # 获取有效股票
    df_tmp = load_features_safe(
        D.instruments(market="all"), ["$close"],
        start_time=START, end_time=END, freq="day"
    )
    all_insts = df_tmp.index.get_level_values("instrument").unique().tolist()
    valid_instruments = filter_instruments(all_insts)
    valid_set = set(valid_instruments)
    print(f"有效股票数: {len(valid_instruments)}")
    del df_tmp

    # 加载 qlib 因子
    print("\n[1/3] 加载 qlib 因子数据...")
    qlib_factors = [f for f in default_registry.all().values() if f.source == "qlib"]
    fields = [f.expression for f in qlib_factors]
    names = [f"{f.category}_{f.name}" for f in qlib_factors]
    df_qlib_all = load_features_safe(valid_instruments, fields, start_time=START, end_time=END, freq="day")
    df_qlib_all.columns = names
    print(f"  qlib 因子: {len(names)} 个, 形状: {df_qlib_all.shape}")

    # 加载 parquet 因子
    print("[2/3] 加载 parquet 因子数据...")
    parquet_factors = default_registry.get_by_source("parquet")
    if parquet_factors:
        tmp_reg = FactorRegistry()
        for f in parquet_factors:
            tmp_reg.register(f)
        df_parquet_all = _load_parquet_factors(valid_instruments, START, END, tmp_reg)
        print(f"  parquet 因子: {len(parquet_factors)} 个, 形状: {df_parquet_all.shape}")
    else:
        df_parquet_all = pd.DataFrame()

    # 加载收益率并 unstack
    print("[3/3] 加载收益率数据...")
    ret_field = ["$close / Ref($close, 1) - 1"]
    df_ret = load_features_safe(valid_instruments, ret_field, start_time=START, end_time=END, freq="day")
    df_ret.columns = ["daily_ret"]
    # unstack 为 (日期 x 股票) 矩阵，加速查询
    df_ret_unstacked = df_ret["daily_ret"].unstack(level="instrument")
    dates = df_ret_unstacked.index.sort_values()
    print(f"  收益率矩阵: {df_ret_unstacked.shape}, 交易日: {len(dates)}")

    # 逐因子回测
    print("\n" + "=" * 70)
    print("  开始逐因子回测...")
    print("=" * 70)

    results = []
    all_factors = list(default_registry.all().values())

    for i, f in enumerate(all_factors, 1):
        print(f"  [{i:2d}/{len(all_factors)}] {f.name:30s} ({f.category}, IR={f.ir:+.2f}) ...", end=" ", flush=True)
        res = run_single_factor_backtest(f.name, f, df_ret_unstacked, dates, df_qlib_all, df_parquet_all, valid_set)
        if res:
            results.append(res)
            print(f"年化={res['ann_ret']:+.1%}  夏普={res['sharpe']:.2f}  回撤={res['max_dd']:.1%}")
        else:
            print("SKIP")

    if not results:
        print("\n[WARN] 没有因子产生有效回测结果")
        return

    # 汇总
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("sharpe", ascending=False)

    print("\n" + "=" * 70)
    print("  单因子回测汇总（按夏普排序）")
    print("=" * 70)
    print(f"\n{'因子':<28s} {'层':>6s} {'扫描IR':>7s} {'年化':>8s} {'夏普':>7s} {'回撤':>8s} {'胜率':>6s} {'总收益':>8s}")
    print("-" * 85)
    for _, r in df_results.iterrows():
        print(f"{r['factor']:<28s} {r['category']:>6s} {r['ir_scan']:>+7.2f} {r['ann_ret']:>+7.1%} {r['sharpe']:>7.2f} {r['max_dd']:>7.1%} {r['win_rate']:>6.1%} {r['total_ret']:>+7.1%}")

    out = Path("results/single_factor_backtest.csv")
    out.parent.mkdir(exist_ok=True)
    df_results.to_csv(out, index=False)
    print(f"\n[OK] 结果已保存: {out}")


if __name__ == "__main__":
    main()
