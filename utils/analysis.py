#!/usr/bin/env python3
"""
策略历年分段性能分析
"""
import warnings
warnings.filterwarnings("ignore")

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import multiprocessing

if sys.platform == "darwin":
    multiprocessing.set_start_method('fork', force=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# qlib配置
qlib_data_path = str(PROJECT_ROOT / "data/qlib_data/cn_data")
os.environ["JOBLIB_START_METHOD"] = "fork"

import qlib
from qlib.config import REG_CN
qlib.init(provider_uri=qlib_data_path, region=REG_CN)

from qlib.data import D
from core.selection import get_full_alpha_expressions
from core.signals import SignalGenerator, SignalConfig

# 定义 load_features_safe 函数（原来自 multifactor_backtest）
def load_features_safe(instruments, fields, start_time, end_time, freq="day"):
    if not isinstance(instruments, list):
        inst_list = D.list_instruments(instruments, start_time=start_time, end_time=end_time)
        inst_list = list(inst_list.keys())
    else:
        inst_list = list(instruments)
    return D.features(inst_list, fields, start_time, end_time, freq)

# 定义 CONFIG（与原 multifactor_backtest 一致）
CONFIG = {
    "start_date": "2019-01-01",
    "end_date": "2026-02-26",
    "topk": 20,
    "initial_capital": 500_000,
    "open_cost": 0.0003,
    "close_cost": 0.0013,
    "qlib_data_path": qlib_data_path,
    "w_alpha": 0.65,
    "w_risk": 0.20,
    "w_enhance": 0.15,
}

print("="*80)
print("  策略历年性能分析（MIN_MARKET_CAP=10亿）")
print("="*80)

# ============================================================================
# 运行策略回测
# ============================================================================

print("\n[1/3] 运行策略回测...")

MIN_MARKET_CAP = 10

fields, names = get_full_alpha_expressions()
instruments = D.instruments(market="all")

df_close = load_features_safe(
    instruments, ["$close"],
    start_time=CONFIG["start_date"],
    end_time=CONFIG["end_date"],
    freq="day"
)

valid_instruments = [
    i for i in df_close.index.get_level_values("instrument").unique()
    if not i.startswith("BJ")
    and not i.startswith("SH43") and not i.startswith("SZ43")
    and not i.startswith("SH83") and not i.startswith("SZ83")
    and not i.startswith("SH87") and not i.startswith("SZ87")
]

df = load_features_safe(
    valid_instruments, fields,
    start_time=CONFIG["start_date"],
    end_time=CONFIG["end_date"],
    freq="day"
)
df.columns = names

# 改进的缺失值处理：按日期分组，用截面中位数填充
def fill_missing_cross_sectional(df):
    """按日期分组，用该日期所有股票的中位数填充缺失值"""
    filled_df = df.copy()
    dates = df.index.get_level_values("datetime").unique()
    
    for dt in dates:
        try:
            cross_section = df.xs(dt, level="datetime")
        except KeyError:
            continue
            
        # 计算每列的中位数（排除NaN）
        medians = cross_section.median(axis=0, skipna=True)
        
        # 填充该日期的缺失值
        for col in cross_section.columns:
            if col in filled_df.columns:
                # 只填充该列在该日期的缺失值
                mask = (filled_df.index.get_level_values("datetime") == dt) & filled_df[col].isna()
                if mask.any():
                    filled_df.loc[mask, col] = medians.get(col, 0)
    
    # 如果还有缺失（可能整个截面都缺失），用0填充
    filled_df = filled_df.fillna(0)
    return filled_df

df = fill_missing_cross_sectional(df)

# 计算信号
signal_config = SignalConfig(
    w_alpha=CONFIG.get("w_alpha", 0.65),
    w_risk=CONFIG.get("w_risk", 0.20),
    w_enhance=CONFIG.get("w_enhance", 0.15)
)
signal_gen = SignalGenerator(signal_config)
dates = df.index.get_level_values("datetime").unique().sort_values()
monthly_dates = pd.DatetimeIndex(
    pd.Series(dates).groupby(pd.Series(dates).dt.to_period("M")).last().values
)
monthly_df = df.loc[df.index.get_level_values("datetime").isin(monthly_dates)]

signal = signal_gen.generate_signal(monthly_df)

# 修复索引
if signal.index.nlevels == 1:
    if isinstance(signal.index[0], tuple):
        signal.index = pd.MultiIndex.from_tuples(signal.index, names=["datetime", "instrument"])

try:
    pd.Timestamp(signal.index[0][0])
except:
    signal.index = signal.index.swaplevel()
    signal.index.names = ["datetime", "instrument"]

pred = signal.to_frame("score").dropna()
stock_counts = pred.index.get_level_values(1).value_counts()
valid_stocks = stock_counts[stock_counts >= 10].index
pred = pred.loc[pred.index.get_level_values(1).isin(valid_stocks)]

# 获取收益率
ret_field = ["Ref($close, -1) / $close - 1"]
df_ret = load_features_safe(
    valid_instruments, ret_field,
    start_time=CONFIG["start_date"],
    end_time=CONFIG["end_date"],
    freq="day"
)
df_ret.columns = ["next_ret"]

# 月度调仓
portfolio_returns = []
dates_list = sorted([d for d in monthly_dates if d >= pd.Timestamp(CONFIG["start_date"])])
BOND_RETURN = 0.03 / 252

for i, rebal_date in enumerate(dates_list[:-1]):
    try:
        cross = pred.xs(rebal_date, level="datetime")
    except (KeyError, TypeError):
        cross = pred.loc[pred.index.get_level_values(0) == rebal_date]

    if len(cross) < CONFIG["topk"]:
        continue

    # 市值过滤
    try:
        date_str = str(rebal_date)[:10]
        all_stocks = cross.index.get_level_values(1).unique().tolist() if cross.index.nlevels == 2 else list(cross.index)

        df_mv = load_features_safe(
            all_stocks, ["$close", "$float_share"],
            start_time=date_str,
            end_time=date_str,
            freq="day"
        )

        if df_mv is not None and len(df_mv) > 0:
            df_mv.columns = ["close", "float_share"]
            df_mv["market_cap"] = df_mv["close"] * df_mv["float_share"] * 10000 / 1e8
            valid_by_mv_full = df_mv[df_mv["market_cap"] >= MIN_MARKET_CAP].index.tolist()

            if len(valid_by_mv_full) > 0 and isinstance(valid_by_mv_full[0], tuple):
                valid_by_mv = [x[0] if isinstance(x, tuple) else x for x in valid_by_mv_full]
            else:
                valid_by_mv = valid_by_mv_full

            if len(valid_by_mv) > 0:
                cross = cross.loc[cross.index.isin(valid_by_mv)]
    except:
        pass

    if len(cross) < 10:
        continue

    topk = min(CONFIG.get("bull_topk", 50), len(cross))
    selected = cross.nlargest(topk, "score").index.tolist()

    if i + 1 < len(dates_list):
        next_date = dates_list[i + 1]
    else:
        next_date = dates[-1]

    holding_mask = (dates > rebal_date) & (dates <= next_date)
    holding_dates = dates[holding_mask]

    portfolio_value = 1.0
    peak_value = 1.0

    for hd_idx, hd in enumerate(holding_dates):
        try:
            try:
                daily_ret = df_ret.xs(hd, level="datetime")
            except (KeyError, TypeError):
                daily_ret = df_ret.loc[df_ret.index.get_level_values(0) == hd]

            stock_ret = daily_ret.loc[daily_ret.index.get_level_values(-1).isin(selected)
                                      if daily_ret.index.nlevels > 1
                                      else daily_ret.index.isin(selected), "next_ret"].mean()

            if np.isnan(stock_ret):
                stock_ret = 0

            if hd_idx == 0:
                current_weights = 0.70

            port_ret = stock_ret * current_weights + BOND_RETURN * (1 - current_weights)

            if hd_idx == 0:
                daily_cost = (CONFIG["open_cost"] + CONFIG["close_cost"]) / len(holding_dates)
            else:
                daily_cost = 0

            net_ret = port_ret - daily_cost
            portfolio_value *= (1 + net_ret)

            if portfolio_value > peak_value:
                peak_value = portfolio_value

            dd = (portfolio_value - peak_value) / peak_value

            portfolio_returns.append({
                "date": hd,
                "return": net_ret,
                "cum_return": portfolio_value,
                "drawdown": dd
            })

        except (KeyError, IndexError):
            continue

strategy_df = pd.DataFrame(portfolio_returns)
strategy_df["date"] = pd.to_datetime(strategy_df["date"])
strategy_df = strategy_df.set_index("date")

print(f"  ✓ 策略回测完成: {len(strategy_df)} 个交易日")

# ============================================================================
# 获取沪深300数据用于对比
# ============================================================================

print("\n[2/3] 获取沪深300指数数据...")

try:
    # 尝试获取沪深300指数
    hs300_data = load_features_safe(
        ["SH000300"], ["$close"],
        start_time=CONFIG["start_date"],
        end_time=CONFIG["end_date"],
        freq="day"
    )

    if hs300_data is not None and len(hs300_data) > 0:
        hs300_data.columns = ["close"]

        # 处理MultiIndex
        if isinstance(hs300_data.index, pd.MultiIndex):
            hs300_data = hs300_data.reset_index()
            hs300_data = hs300_data.set_index("datetime")

        hs300_data.index = pd.to_datetime(hs300_data.index)
        hs300_data["return"] = hs300_data["close"].pct_change()
        hs300_data["cum_return"] = (1 + hs300_data["return"]).cumprod()

        print(f"  ✓ 沪深300: {len(hs300_data)} 个交易日")
        has_hs300 = True
    else:
        print(f"  ✗ 沪深300: 数据为空")
        has_hs300 = False
except Exception as e:
    print(f"  ✗ 沪深300: {type(e).__name__}: {str(e)[:50]}")
    has_hs300 = False

# ============================================================================
# 计算指标
# ============================================================================

def calc_metrics(returns_series):
    """计算关键指标"""
    if len(returns_series) == 0:
        return None

    cum_ret = (1 + returns_series).cumprod()
    total_days = len(returns_series)

    total_return = cum_ret.iloc[-1] - 1
    ann_ret = cum_ret.iloc[-1] ** (365 / total_days) - 1 if total_days > 0 else 0
    max_dd = (cum_ret / cum_ret.cummax() - 1).min()
    sharpe = returns_series.mean() / returns_series.std() * np.sqrt(252) if returns_series.std() > 0 else 0
    win_rate = (returns_series > 0).mean()
    volatility = returns_series.std() * np.sqrt(252)

    return {
        "总收益": total_return,
        "年化收益": ann_ret,
        "最大回撤": max_dd,
        "夏普比率": sharpe,
        "日胜率": win_rate,
        "年化波动": volatility,
        "收益回撤比": abs(ann_ret / max_dd) if max_dd != 0 else 0,
    }

print("\n[3/3] 分析历年性能...")

# ============================================================================
# 历年性能对比
# ============================================================================

print("\n" + "="*80)
print("  整体性能指标")
print("="*80)

strategy_metrics = calc_metrics(strategy_df["return"])
print(f"\n当前策略:")
print(f"  总收益:      {strategy_metrics['总收益']*100:>8.2f}%")
print(f"  年化收益:    {strategy_metrics['年化收益']*100:>8.2f}%")
print(f"  最大回撤:    {strategy_metrics['最大回撤']*100:>8.2f}%")
print(f"  夏普比率:    {strategy_metrics['夏普比率']:>8.4f}")
print(f"  日胜率:      {strategy_metrics['日胜率']*100:>8.2f}%")
print(f"  年化波动:    {strategy_metrics['年化波动']*100:>8.2f}%")
print(f"  收益/回撤:   {strategy_metrics['收益回撤比']:>8.4f}")

if has_hs300:
    hs300_metrics = calc_metrics(hs300_data["return"])
    print(f"\n沪深300指数:")
    print(f"  总收益:      {hs300_metrics['总收益']*100:>8.2f}%")
    print(f"  年化收益:    {hs300_metrics['年化收益']*100:>8.2f}%")
    print(f"  最大回撤:    {hs300_metrics['最大回撤']*100:>8.2f}%")
    print(f"  夏普比率:    {hs300_metrics['夏普比率']:>8.4f}")
    print(f"  日胜率:      {hs300_metrics['日胜率']*100:>8.2f}%")
    print(f"  年化波动:    {hs300_metrics['年化波动']*100:>8.2f}%")
    print(f"  收益/回撤:   {hs300_metrics['收益回撤比']:>8.4f}")

    # 计算超额收益
    common_dates = strategy_df.index.intersection(hs300_data.index)
    if len(common_dates) > 0:
        strategy_excess = strategy_df.loc[common_dates, "return"]
        hs300_excess = hs300_data.loc[common_dates, "return"]
        excess = strategy_excess.values - hs300_excess.values

        print(f"\n相对沪深300超额收益:")
        print(f"  年化超额:    {excess.mean() * 252 * 100:>8.2f}%")
        print(f"  跑赢天数:    {(excess > 0).sum()} / {len(excess)} ({(excess > 0).mean()*100:.1f}%)")

# ============================================================================
# 年度性能
# ============================================================================

print("\n" + "="*80)
print("  年度性能对比")
print("="*80)

strategy_df["year"] = strategy_df.index.year
years = sorted(strategy_df["year"].unique())

print(f"\n{'年份':<8} {'策略收益':>12} {'策略回撤':>12} {'策略夏普':>12}", end="")
if has_hs300:
    hs300_data["year"] = hs300_data.index.year
    print(f" {'HS300收益':>12} {'HS300回撤':>12} {'HS300夏普':>12} {'超额收益':>12}", end="")
print()
print("-" * 100)

for year in years:
    strategy_year = strategy_df[strategy_df["year"] == year]["return"]
    strategy_year_metrics = calc_metrics(strategy_year)

    ret_str = f"{strategy_year_metrics['总收益']*100:>10.2f}%" if strategy_year_metrics else "N/A"
    dd_str = f"{strategy_year_metrics['最大回撤']*100:>10.2f}%" if strategy_year_metrics else "N/A"
    sharpe_str = f"{strategy_year_metrics['夏普比率']:>10.4f}" if strategy_year_metrics else "N/A"

    print(f"{year:<8} {ret_str:>12} {dd_str:>12} {sharpe_str:>12}", end="")

    if has_hs300:
        hs300_year = hs300_data[hs300_data["year"] == year]["return"]
        hs300_year_metrics = calc_metrics(hs300_year)

        hs300_ret_str = f"{hs300_year_metrics['总收益']*100:>10.2f}%" if hs300_year_metrics else "N/A"
        hs300_dd_str = f"{hs300_year_metrics['最大回撤']*100:>10.2f}%" if hs300_year_metrics else "N/A"
        hs300_sharpe_str = f"{hs300_year_metrics['夏普比率']:>10.4f}" if hs300_year_metrics else "N/A"

        if strategy_year_metrics and hs300_year_metrics:
            excess_ret = strategy_year_metrics['总收益'] - hs300_year_metrics['总收益']
            excess_str = f"{excess_ret*100:>10.2f}%"
        else:
            excess_str = "N/A"

        print(f" {hs300_ret_str:>12} {hs300_dd_str:>12} {hs300_sharpe_str:>12} {excess_str:>12}", end="")

    print()

# ============================================================================
# 性能评价
# ============================================================================

print("\n" + "="*80)
print("  性能评价")
print("="*80)

print(f"\n✓ 策略优势:")
print(f"  - 年化收益: {strategy_metrics['年化收益']*100:.2f}%")
print(f"  - 最大回撤: {strategy_metrics['最大回撤']*100:.2f}% (控制良好)")
print(f"  - 夏普比率: {strategy_metrics['夏普比率']:.4f} (风险调整收益)")
print(f"  - 日胜率:   {strategy_metrics['日胜率']*100:.2f}%")

if has_hs300:
    print(f"\n📊 相对沪深300:")
    if strategy_metrics['年化收益'] > hs300_metrics['年化收益']:
        outperform = (strategy_metrics['年化收益'] - hs300_metrics['年化收益']) / hs300_metrics['年化收益'] * 100
        print(f"  - 年化收益超越: {outperform:.1f}%")
    if abs(strategy_metrics['最大回撤']) < abs(hs300_metrics['最大回撤']):
        less_dd = (abs(hs300_metrics['最大回撤']) - abs(strategy_metrics['最大回撤'])) / abs(hs300_metrics['最大回撤']) * 100
        print(f"  - 最大回撤更优: {less_dd:.1f}%")
    if strategy_metrics['夏普比率'] > hs300_metrics['夏普比率']:
        sharpe_better = (strategy_metrics['夏普比率'] - hs300_metrics['夏普比率']) / hs300_metrics['夏普比率'] * 100
        print(f"  - 夏普比率更优: {sharpe_better:.1f}%")

print("\n" + "="*80)
