"""
PyBroker vs Qlib 回测差异诊断
逐项隔离：选股差异、收益计算差异、成本差异、持仓期差异
"""
import warnings
warnings.filterwarnings("ignore")

import os, sys
import numpy as np
import pandas as pd

# ============================================================
# 0. 公共配置 & Qlib 初始化
# ============================================================
CONFIG = {
    "start_date": "2019-01-01",
    "end_date":   "2026-02-26",
    "topk": 20,
    "initial_capital": 500_000,
    "open_cost":  0.0003,
    "close_cost": 0.0013,
    "qlib_data_path": "/Users/sxt/code/qlib/data/qlib_data/cn_data",
    "w_alpha":   0.65,
    "w_risk":    0.20,
    "w_enhance": 0.15,
}

import qlib
from qlib.config import REG_CN
os.environ["JOBLIB_START_METHOD"] = "fork"
qlib.init(provider_uri=CONFIG["qlib_data_path"], region=REG_CN)
from qlib.data import D

def load_features_safe(instruments, fields, start_time, end_time, freq="day"):
    if not isinstance(instruments, list):
        inst_list = list(D.list_instruments(instruments, start_time=start_time, end_time=end_time).keys())
    else:
        inst_list = list(instruments)
    return D.features(inst_list, fields, start_time, end_time, freq)

# ============================================================
# 1. 加载公共数据
# ============================================================
print("=" * 70)
print("  PyBroker vs Qlib 差异诊断")
print("=" * 70)

# 因子表达式（两边完全相同）
from core.selection import get_alpha_expressions, get_full_alpha_expressions

fields, names = get_full_alpha_expressions()

instruments = D.instruments(market="all")
df_close_raw = load_features_safe(instruments, ["$close"], CONFIG["start_date"], CONFIG["end_date"])
valid_instruments = [
    i for i in df_close_raw.index.get_level_values("instrument").unique()
    if not i.startswith("BJ")
    and not any(i.startswith(p) for p in ["SH43","SZ43","SH83","SZ83","SH87","SZ87"])
]
print(f"\n有效股票: {len(valid_instruments)} 只")

df_factors = load_features_safe(valid_instruments, fields, CONFIG["start_date"], CONFIG["end_date"])
df_factors.columns = names
df_factors_raw = df_factors.copy()  # 保留原始（含NaN）

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

df_factors = fill_missing_cross_sectional(df_factors)

# 月末日期
dates = df_factors.index.get_level_values("datetime").unique().sort_values()
monthly_dates = pd.DatetimeIndex(
    pd.Series(dates).groupby(pd.Series(dates).dt.to_period("M")).last().values
)
monthly_df = df_factors.loc[df_factors.index.get_level_values("datetime").isin(monthly_dates)]

print(f"月度截面数: {len(monthly_dates)}")
print(f"因子数据: {df_factors.shape[0]:,} 行")

# ============================================================
# 诊断1: 选股差异 — 两种 generate_signal 选出的 Top20 有多少重叠
# ============================================================
print("\n" + "=" * 70)
print("  诊断1: 选股差异（信号评分逻辑）")
print("=" * 70)

alpha_factors, risk_factors, enhance_factors = get_alpha_expressions()

# --- PyBroker 版评分（单次 rank，无二次排序） ---
def pybroker_signal(monthly_df):
    alpha_cols   = [f"alpha_{k}" for k in alpha_factors]
    risk_cols    = [f"risk_{k}" for k in risk_factors]
    enhance_cols = [f"enhance_{k}" for k in enhance_factors]

    def layer_score(df, cols):
        scores = pd.DataFrame(index=df.index)
        for fname in cols:
            if fname in df.columns:
                scores[fname] = df.groupby(level="datetime")[fname].transform(lambda x: x.rank(pct=True))
        if scores.empty:
            return pd.Series(0.0, index=df.index)
        return scores.mean(axis=1)

    alpha_s = layer_score(monthly_df, alpha_cols)
    risk_s  = layer_score(monthly_df, risk_cols)
    enh_s   = layer_score(monthly_df, enhance_cols)

    # 直接加权组合，不再进行二次排序
    return 0.65 * alpha_s + 0.20 * risk_s + 0.15 * enh_s

# --- Qlib 版评分（逐日截面，无二次排序） ---
def qlib_signal(monthly_df):
    from core.signals import SignalGenerator, SignalConfig
    
    # 从CONFIG字典创建SignalConfig对象
    config = SignalConfig(
        w_alpha=CONFIG.get("w_alpha", 0.65),
        w_risk=CONFIG.get("w_risk", 0.20),
        w_enhance=CONFIG.get("w_enhance", 0.15)
    )
    
    sig_gen = SignalGenerator(config)
    return sig_gen.generate_signal(monthly_df)

pb_score = pybroker_signal(monthly_df)
pb_score.name = "pb_score"
# 确保 pb_score 索引顺序为 (datetime, instrument)
if pb_score.index.names == ['instrument', 'datetime']:
    pb_score = pb_score.swaplevel()
    pb_score = pb_score.sort_index()

ql_score = qlib_signal(monthly_df)
ql_score.name = "ql_score"
# Qlib generate_signal 可能返回 flat tuple index，转为 MultiIndex
if ql_score.index.nlevels == 1:
    tuples = ql_score.index.tolist()
    if len(tuples) > 0 and isinstance(tuples[0], tuple):
        mi = pd.MultiIndex.from_tuples(tuples, names=["datetime", "instrument"])
        ql_score = pd.Series(ql_score.values, index=mi, name="ql_score")
    else:
        print("  [WARN] ql_score 索引格式异常")
# 确保 level 0 是 datetime：检查 level 0 的第一个值是否为 Timestamp
if ql_score.index.nlevels == 2:
    first_l0 = ql_score.index.get_level_values(0)[0]
    if isinstance(first_l0, str):
        # level 0 是 instrument，需要 swap
        ql_score = ql_score.swaplevel()
        ql_score.index.names = ["datetime", "instrument"]
        ql_score = ql_score.sort_index()

# 调试：检查两个 score 的索引结构
print(f"\n  pb_score: type={type(pb_score)}, len={len(pb_score)}")
if hasattr(pb_score, 'index') and hasattr(pb_score.index, 'names'):
    print(f"    index names: {pb_score.index.names}")
    print(f"    index nlevels: {pb_score.index.nlevels}")
    if pb_score.index.nlevels >= 1:
        pb_dates = pb_score.index.get_level_values(0).unique()
        print(f"    unique dates: {len(pb_dates)}, first: {pb_dates[0]}, last: {pb_dates[-1]}")

print(f"  ql_score: type={type(ql_score)}, len={len(ql_score)}")
if hasattr(ql_score, 'index') and hasattr(ql_score.index, 'names'):
    print(f"    index names: {ql_score.index.names}")
    print(f"    index nlevels: {ql_score.index.nlevels}")
    if ql_score.index.nlevels >= 1:
        ql_dates = ql_score.index.get_level_values(0).unique()
        print(f"    unique dates: {len(ql_dates)}, first: {ql_dates[0]}, last: {ql_dates[-1]}")

# 逐月比较 Top20
overlap_stats = []
pb_dates_set = set(pb_score.index.get_level_values(0).unique()) if pb_score.index.nlevels >= 2 else set()
ql_dates_set = set(ql_score.index.get_level_values(0).unique()) if ql_score.index.nlevels >= 2 else set()

for dt in monthly_dates:
    try:
        if dt in pb_dates_set:
            pb_day = pb_score.xs(dt, level=0)
        else:
            continue
        if dt in ql_dates_set:
            ql_day = ql_score.xs(dt, level=0)
        else:
            continue
    except KeyError:
        continue

    if len(pb_day) < CONFIG["topk"] or len(ql_day) < CONFIG["topk"]:
        continue

    pb_top = set(pb_day.nlargest(CONFIG["topk"]).index)
    ql_top = set(ql_day.nlargest(CONFIG["topk"]).index)
    overlap = len(pb_top & ql_top)
    overlap_stats.append({"date": dt, "overlap": overlap, "pb_only": len(pb_top - ql_top), "ql_only": len(ql_top - pb_top)})

df_overlap = pd.DataFrame(overlap_stats)
if len(df_overlap) > 0:
    print(f"\n月度选股对比（共 {len(df_overlap)} 个月）:")
    print(f"  平均重叠股票数:   {df_overlap['overlap'].mean():.1f} / {CONFIG['topk']}")
    print(f"  最小重叠:         {df_overlap['overlap'].min()}")
    print(f"  最大重叠:         {df_overlap['overlap'].max()}")
    print(f"  完全相同月份数:   {(df_overlap['overlap'] == CONFIG['topk']).sum()}")
    print(f"  重叠率 < 50% 月数: {(df_overlap['overlap'] < CONFIG['topk'] * 0.5).sum()}")
else:
    print(f"\n  [WARN] 无重叠数据，pb_dates_set 有 {len(pb_dates_set)} 日, ql_dates_set 有 {len(ql_dates_set)} 日")
    # 看看是否日期类型不匹配
    if pb_dates_set and ql_dates_set:
        pb_sample = list(pb_dates_set)[:3]
        ql_sample = list(ql_dates_set)[:3]
        print(f"    pb sample dates: {pb_sample}")
        print(f"    ql sample dates: {ql_sample}")
        print(f"    monthly sample:  {list(monthly_dates[:3])}")

# ============================================================
# 诊断2: 用相同选股，比较收益计算方式的差异
# ============================================================
print("\n" + "=" * 70)
print("  诊断2: 收益计算差异（用 PyBroker 选股，对比两种收益算法）")
print("=" * 70)

# 加载收益率
ret_field_next = ["Ref($close, -1) / $close - 1"]  # Qlib 用的: close[t+1]/close[t]-1
ret_field_curr = ["$close / Ref($close, 1) - 1"]   # 当日收益: close[t]/close[t-1]-1

df_ret_next = load_features_safe(valid_instruments, ret_field_next, CONFIG["start_date"], CONFIG["end_date"])
df_ret_next.columns = ["next_ret"]

df_ret_curr = load_features_safe(valid_instruments, ret_field_curr, CONFIG["start_date"], CONFIG["end_date"])
df_ret_curr.columns = ["curr_ret"]

# 用 PyBroker 选股结果做两种收益计算
monthly_dates_list = sorted([d for d in monthly_dates if d >= pd.Timestamp(CONFIG["start_date"])])

# 方法A: Qlib 方式 (next_ret, holding_dates > rebal_date)
returns_qlib_style = []
# 方法B: 正确方式 (curr_ret, 从 rebal_date+1 开始)
returns_correct = []
# 方法C: Qlib 方式但不扣成本
returns_qlib_nocost = []

for i, rebal_date in enumerate(monthly_dates_list[:-1]):
    pb_day = pb_score.xs(rebal_date, level="datetime") if rebal_date in pb_score.index.get_level_values("datetime") else None
    if pb_day is None or len(pb_day) < CONFIG["topk"]:
        continue

    selected = set(pb_day.nlargest(CONFIG["topk"]).index)
    next_date = monthly_dates_list[i + 1] if i + 1 < len(monthly_dates_list) else dates[-1]
    holding_dates = dates[(dates > rebal_date) & (dates <= next_date)]

    for hd in holding_dates:
        # 方法A: Qlib 原版（next_ret + 每日扣成本）
        try:
            daily_ret = df_ret_next.xs(hd, level="datetime")
            port_ret = daily_ret.loc[daily_ret.index.isin(selected), "next_ret"].mean()
            if not np.isnan(port_ret):
                daily_cost = (CONFIG["open_cost"] + CONFIG["close_cost"]) / len(holding_dates)
                returns_qlib_style.append({"date": hd, "return": port_ret - daily_cost})
                returns_qlib_nocost.append({"date": hd, "return": port_ret})
        except (KeyError, IndexError):
            pass

        # 方法B: curr_ret（当日收益）
        try:
            daily_ret_c = df_ret_curr.xs(hd, level="datetime")
            port_ret_c = daily_ret_c.loc[daily_ret_c.index.isin(selected), "curr_ret"].mean()
            if not np.isnan(port_ret_c):
                returns_correct.append({"date": hd, "return": port_ret_c})
        except (KeyError, IndexError):
            pass

def summarize(name, rets):
    if not rets:
        print(f"  {name}: 无数据")
        return
    df = pd.DataFrame(rets).set_index("date")
    df.index = pd.to_datetime(df.index)
    cum = (1 + df["return"]).cumprod()
    total_days = (df.index[-1] - df.index[0]).days
    ann_ret = cum.iloc[-1] ** (365 / total_days) - 1 if total_days > 0 else 0
    max_dd = (cum / cum.cummax() - 1).min()
    sharpe = df["return"].mean() / df["return"].std() * np.sqrt(252) if df["return"].std() > 0 else 0
    final = CONFIG["initial_capital"] * cum.iloc[-1]
    print(f"  {name}:")
    print(f"    期末资产: ¥{final:>12,.0f}  年化: {ann_ret:>7.2%}  MaxDD: {max_dd:>7.2%}  Sharpe: {sharpe:.4f}")

print("\n用 PyBroker 相同选股，不同收益计算方式:")
summarize("A. Qlib原版 (next_ret + 每日扣成本)", returns_qlib_style)
summarize("B. Qlib原版 (next_ret, 不扣成本)", returns_qlib_nocost)
summarize("C. 当日收益 (curr_ret, 不扣成本)", returns_correct)

# ============================================================
# 诊断3: 交易成本影响量化
# ============================================================
print("\n" + "=" * 70)
print("  诊断3: 交易成本影响")
print("=" * 70)

if returns_qlib_style and returns_qlib_nocost:
    df_cost = pd.DataFrame(returns_qlib_style).set_index("date")
    df_nocost = pd.DataFrame(returns_qlib_nocost).set_index("date")
    cum_cost = (1 + df_cost["return"]).cumprod()
    cum_nocost = (1 + df_nocost["return"]).cumprod()
    cost_drag = (cum_nocost.iloc[-1] - cum_cost.iloc[-1]) / cum_nocost.iloc[-1]
    total_cost_deducted = (df_nocost["return"] - df_cost["return"]).sum()
    print(f"  Qlib 方式总扣除成本: {total_cost_deducted:.4f} ({total_cost_deducted:.2%})")
    print(f"  成本拖累占总收益比: {cost_drag:.2%}")
    ann_cost = total_cost_deducted / ((df_cost.index[-1] - df_cost.index[0]).days / 365)
    print(f"  年均成本扣除: {ann_cost:.2%}")

# ============================================================
# 诊断4: 持仓期偏移（next_ret vs curr_ret）影响
# ============================================================
print("\n" + "=" * 70)
print("  诊断4: 持仓期偏移影响 (next_ret vs curr_ret)")
print("=" * 70)

if returns_qlib_nocost and returns_correct:
    df_next = pd.DataFrame(returns_qlib_nocost).set_index("date")
    df_curr = pd.DataFrame(returns_correct).set_index("date")
    df_next.index = pd.to_datetime(df_next.index)
    df_curr.index = pd.to_datetime(df_curr.index)

    cum_next = (1 + df_next["return"]).cumprod()
    cum_curr = (1 + df_curr["return"]).cumprod()

    ann_next = cum_next.iloc[-1] ** (365 / (df_next.index[-1] - df_next.index[0]).days) - 1
    ann_curr = cum_curr.iloc[-1] ** (365 / (df_curr.index[-1] - df_curr.index[0]).days) - 1

    print(f"  next_ret 年化: {ann_next:.2%}")
    print(f"  curr_ret 年化: {ann_curr:.2%}")
    print(f"  差异: {ann_next - ann_curr:+.2%}")

# ============================================================
# 诊断5: Qlib 选股 vs PyBroker 选股的纯收益差异（排除引擎差异）
# ============================================================
print("\n" + "=" * 70)
print("  诊断5: 选股差异对收益的影响（用 curr_ret, 不扣成本）")
print("=" * 70)

# 用 Qlib 选股
returns_ql_select = []
for i, rebal_date in enumerate(monthly_dates_list[:-1]):
    try:
        ql_day = ql_score.xs(rebal_date, level="datetime")
    except KeyError:
        continue
    if len(ql_day) < CONFIG["topk"]:
        continue

    selected = set(ql_day.nlargest(CONFIG["topk"]).index)
    next_date = monthly_dates_list[i + 1] if i + 1 < len(monthly_dates_list) else dates[-1]
    holding_dates = dates[(dates > rebal_date) & (dates <= next_date)]

    for hd in holding_dates:
        try:
            daily_ret_c = df_ret_curr.xs(hd, level="datetime")
            port_ret_c = daily_ret_c.loc[daily_ret_c.index.isin(selected), "curr_ret"].mean()
            if not np.isnan(port_ret_c):
                returns_ql_select.append({"date": hd, "return": port_ret_c})
        except (KeyError, IndexError):
            pass

print("\n相同收益计算方式（curr_ret, 无成本），不同选股:")
summarize("PyBroker 选股", returns_correct)
summarize("Qlib 选股", returns_ql_select)

# ============================================================
# 诊断6: Qlib 版 valid_stocks 过滤的影响
# ============================================================
print("\n" + "=" * 70)
print("  诊断6: Qlib 额外股票过滤的影响")
print("=" * 70)

# 模拟 Qlib 的 pred 过滤逻辑
ql_pred = ql_score.to_frame("score").dropna()
ql_pred_full = ql_pred.reindex(df_factors.index).groupby(level="instrument").ffill().dropna()
stock_counts = ql_pred_full.index.get_level_values("instrument").value_counts()
valid_stocks = stock_counts[stock_counts > 100].index
filtered_out = stock_counts[stock_counts <= 100].index

print(f"  Qlib signal 覆盖股票: {len(stock_counts)} 只")
print(f"  过滤后保留: {len(valid_stocks)} 只")
print(f"  被过滤掉: {len(filtered_out)} 只（交易日 < 100）")

# 检查被过滤的股票是否出现在 PyBroker 选股中
pb_all_selected = set()
for dt in monthly_dates:
    try:
        pb_day = pb_score.xs(dt, level="datetime")
    except KeyError:
        continue
    if len(pb_day) >= CONFIG["topk"]:
        pb_all_selected.update(pb_day.nlargest(CONFIG["topk"]).index)

filtered_but_selected = pb_all_selected & set(filtered_out)
print(f"  PyBroker 曾选中但被 Qlib 过滤的股票: {len(filtered_but_selected)} 只")

print("\n" + "=" * 70)
print("  诊断完成")
print("=" * 70)
