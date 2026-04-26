#!/usr/bin/env python3
"""
分位数因子回测 — 复现因子数据的口径

方法论：
  - 每日按因子值排序，分10组（decile）
  - 等权持有每组股票，日频调仓
  - 超额收益 = 分位组合日收益 - 全A等权日收益
  - 换手率 = 日间持仓变化比例
  - IC = 截面 Spearman Rank IC
  - IR = mean(IC) / std(IC)
"""
import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.qlib_init import init_qlib, load_features_safe
from core.universe import filter_instruments
from scripts.factor_scan import get_all_factors

N_QUANTILES = 10
START_DATE = "2016-04-11"  # 近10年
END_DATE = "2026-04-11"


def load_data(universe: str = "default"):
    init_qlib()
    from qlib.data import D

    print("[1/3] 加载股票列表...")
    instruments = D.instruments(market="all")
    df_close = load_features_safe(
        instruments, ["$close"],
        start_time=START_DATE, end_time=END_DATE, freq="day"
    )
    include_all = universe == "all_no_filter"
    valid = filter_instruments(
        df_close.index.get_level_values("instrument").unique().tolist(),
        exclude_st=False,
        include_all=include_all,
    )
    print(f"  股票数: {len(valid)}")

    print("[2/3] 计算日收益率...")
    ret_expr = "Ref($close, -1) / $close - 1"  # 次日收益（T+1）
    df_ret = load_features_safe(
        valid, [ret_expr],
        start_time=START_DATE, end_time=END_DATE, freq="day"
    )
    df_ret.columns = ["ret"]

    print("[3/3] 加载因子表达式...")
    factor_dict = get_all_factors()
    items = list(factor_dict.items())
    df_parts = []
    batch_size = 5
    for b in range(0, len(items), batch_size):
        batch = items[b:b + batch_size]
        names = [n for n, _ in batch]
        exprs = [e for _, e in batch]
        try:
            df_part = load_features_safe(
                valid, exprs,
                start_time=START_DATE, end_time=END_DATE, freq="day"
            )
            df_part.columns = names
            df_parts.append(df_part)
        except Exception:
            for name, expr in batch:
                try:
                    df_s = load_features_safe(
                        valid, [expr],
                        start_time=START_DATE, end_time=END_DATE, freq="day"
                    )
                    df_s.columns = [name]
                    df_parts.append(df_s)
                except Exception:
                    pass
        done = min(b + batch_size, len(items))
        print(f"  [{done}/{len(items)}]")

    df_factors = pd.concat(df_parts, axis=1) if df_parts else pd.DataFrame()
    print(f"  成功: {len(df_factors.columns)} 个因子")
    return df_factors, df_ret


def quantile_backtest(factor_series, ret_series):
    """
    分位数回测：日频调仓，10分位
    返回: 最小/最大分位超额年化, 最小/最大分位换手率, IC均值, IR
    """
    # 合并因子和收益
    df = pd.DataFrame({"factor": factor_series, "ret": ret_series}).dropna()
    if df.empty:
        return None

    # ---- IC/IR ----
    def _rank_ic(g):
        if len(g) < 30:
            return np.nan
        return g["factor"].corr(g["ret"], method="spearman")
    ic_series = df.groupby(level="datetime").apply(_rank_ic).dropna()
    if len(ic_series) < 50:
        return None
    ic_mean = ic_series.mean()
    ic_std = ic_series.std()
    ir = ic_mean / ic_std if ic_std > 0 else 0.0

    # ---- 分位数组合 ----
    dates = sorted(df.index.get_level_values("datetime").unique())

    # 计算全A等权基准收益
    bench_rets = df.groupby(level="datetime")["ret"].mean()

    # 每日分位持仓
    min_q_rets = []  # 最小分位(因子值最低)
    max_q_rets = []  # 最大分位(因子值最高)
    min_q_turnovers = []
    max_q_turnovers = []
    prev_min_set = set()
    prev_max_set = set()

    for dt in dates:
        try:
            day = df.xs(dt, level="datetime")
        except KeyError:
            continue
        if len(day) < 50:
            continue

        ranked = day["factor"].rank(ascending=True)
        n = len(ranked)
        q_size = n // N_QUANTILES

        # 最小分位 (rank 1..q_size)
        min_mask = ranked <= q_size
        min_set = set(day.index[min_mask])
        min_ret = day.loc[list(min_set), "ret"].mean()

        # 最大分位 (rank > n-q_size)
        max_mask = ranked > n - q_size
        max_set = set(day.index[max_mask])
        max_ret = day.loc[list(max_set), "ret"].mean()

        bench_ret = bench_rets.get(dt, 0.0)

        min_q_rets.append({"date": dt, "ret": min_ret, "excess": min_ret - bench_ret})
        max_q_rets.append({"date": dt, "ret": max_ret, "excess": max_ret - bench_ret})

        # 换手率
        if prev_min_set and len(min_set) > 0:
            turnover = 1 - len(min_set & prev_min_set) / len(min_set)
            min_q_turnovers.append(turnover)
        if prev_max_set and len(max_set) > 0:
            turnover = 1 - len(max_set & prev_max_set) / len(max_set)
            max_q_turnovers.append(turnover)

        prev_min_set = min_set
        prev_max_set = max_set

    if len(min_q_rets) < 50:
        return None

    df_min = pd.DataFrame(min_q_rets).set_index("date")
    df_max = pd.DataFrame(max_q_rets).set_index("date")

    # 超额年化 = (1+excess_daily).prod() ^ (252/N) - 1
    n_days = len(df_min)
    years = n_days / 252

    min_excess_annual = (1 + df_min["excess"]).prod() ** (1 / years) - 1
    max_excess_annual = (1 + df_max["excess"]).prod() ** (1 / years) - 1

    min_avg_turnover = np.mean(min_q_turnovers) if min_q_turnovers else 0
    max_avg_turnover = np.mean(max_q_turnovers) if max_q_turnovers else 0

    return {
        "最小分位超额年化": min_excess_annual,
        "最大分位超额年化": max_excess_annual,
        "最小分位换手率": min_avg_turnover,
        "最大分位换手率": max_avg_turnover,
        "IC均值": ic_mean,
        "IR": ir,
        "有效天数": n_days,
    }


def main():
    import argparse
    import time

    parser = argparse.ArgumentParser(description="分位数因子回测")
    parser.add_argument("--universe", type=str, default="default",
                        choices=["default", "all_no_filter"],
                        help="股票池选择：default=过滤北交所/科创板/ST，all_no_filter=全A不过滤")
    args = parser.parse_args()

    t0 = time.time()

    df_factors, df_ret = load_data(universe=args.universe)
    ret = df_ret["ret"]

    # 因子名 -> 因子数据文件中的中文名 映射
    # 对照 docs/factor_data.csv 逐条核实
    name_map = {
        # ── 换手率类 ──
        "turnover_5d": "5日平均换手率",
        "turnover_10d": "10日平均换手率",
        "turnover_20d": "20日平均换手率",
        "turnover_60d": "60日平均换手率",
        "turnover_120d": "120日平均换手率",
        "turnover_5d_120d_ratio": "5日平均换手率与120日平均换手率",
        "turnover_10d_120d_ratio": "10日平均换手率与120日平均换手率之比",
        "turnover_20d_120d_ratio": "20日平均换手率与120日平均换手率之比",  # ⚠ 基准无此行，标记
        "turnover_volatility": "换手率相对波动率",

        # ── 成交量/成交额标准差 ──
        "vol_std_10d": "10日成交量标准差",
        "vol_std_20d_v": "20日成交量标准差",
        "amt_std_6d": "6日成交金额的标准差",
        "amt_std_20d": "20日成交金额的标准差",

        # ── 成交量EMA/MA比值 ──
        # 注意：表达式为 $volume / (EMA($volume, N) + 1)，是"当前量/EMA"的比值，不是原始EMA值
        # 基准中的"成交量的X日指数移动平均"是原始EMA值，不是比值，故用不同名称区分
        "vol_ema5_ratio": "成交量与5日EMA比值",  # ⚠ 基准无此名，用描述性名称
        "vol_ema10_ratio": "成交量与10日EMA比值",  # ⚠ 基准无此名
        "vol_ema26_ratio": "成交量与26日EMA比值",  # ⚠ 基准无此名
        "vol_ma12": "12日成交量的移动平均值",
        "amt_ma6": "6日成交金额的移动平均值",
        "amt_ma20": "20日成交金额的移动平均值",

        # ── 量变动速率 ROC ──
        "vol_roc_6d": "6日量变动速率指标",
        "vol_roc_12d": "12日量变动速率指标",

        # ── 成交量震荡 ──
        "vol_oscillator": "成交量震荡",
        "vol_oscillator2": "计算VMACD因子的中间变量",  # ⚠ 基准中VMACD对应的是vol_oscillator2类似

        # ── 价量交叉 ──
        "pvt_6d": "单日价量趋势6日均值",
        "pvt_12d": "单日价量趋势12均值",
        "vol_ret_product": "vol_ret_product",  # ⚠ 基准中无此名称，用原名
        "vmacd": "计算VMACD因子的中间变量",

        # ── 均线偏离（MA deviation = close/MA - 1）──
        # 基准中有"X日乖离率"名称，与 ma*_dev 对应
        "ma5_dev": "5日乖离率",
        "ma10_dev": "10日乖离率",
        "ma20_dev": "20日乖离率",
        "ma60_dev": "60日乖离率",
        "ma120_dev": "120日乖离率",  # ⚠ 基准无120日乖离率，用描述性名称

        # ── EMA偏离（EMA deviation = close/EMA - 1）──
        # ⚠ 基准中没有"EMA乖离率"名称，用"X日EMA乖离率"区分于MA乖离率
        "ema5_dev": "5日EMA乖离率",  # ⚠ 基准无此名
        "ema10_dev": "10日EMA乖离率",  # ⚠ 基准无此名
        "ema12_dev": "12日EMA乖离率",  # ⚠ 基准无此名
        "ema20_dev": "20日EMA乖离率",  # ⚠ 基准无此名
        "ema26_dev": "26日EMA乖离率",  # ⚠ 基准无此名
        "ema60_dev": "60日EMA乖离率",  # ⚠ 基准无此名
        "ema120_dev": "120日EMA乖离率",  # ⚠ 基准无此名

        # ── 乖离率 BIAS ──
        # 注意：bias_*d 表达式为 ($close - Mean($close, N)) / Mean($close, N)
        # 与 ma*_dev ($close / Mean($close, N) - 1) 数学等价，是重复因子
        # 为区分起见，映射到相同基准名但标注重复
        "bias_5d": "5日乖离率",  # ⚠ 与 ma5_dev 数学等价，重复因子
        "bias_10d": "10日乖离率",  # ⚠ 与 ma10_dev 数学等价，重复因子
        "bias_20d": "20日乖离率",  # ⚠ 与 ma20_dev 数学等价，重复因子
        "bias_60d": "60日乖离率",  # ⚠ 与 ma60_dev 数学等价，重复因子

        # ── 均线原始值（raw MA/EMA 值本身）──
        "ma5_raw": "5日移动均线",
        "ma10_raw": "10日移动均线",
        "ma20_raw": "20日移动均线",
        "ma60_raw": "60日移动均线",
        "ma120_raw": "120日移动均线",
        "ema5_raw": "5日指数移动均线",
        "ema10_raw": "10日指数移动均线",
        "ema12_raw": "12日指数移动均线",
        "ema20_raw": "20日指数移动均线",
        "ema26_raw": "26日指数移动均线",
        "ema60_raw": "60日指数移动均线",
        "ema120_raw": "120日指数移动均线",

        # ── 布林线 ──
        "bb_lower_dist": "下轨线（布林线）指标",
        "bb_upper_dist": "上轨线（布林线）指标",
        "bb_width": "bb_width",  # ⚠ 基准中无此名称
        "bb_position": "bb_position",  # ⚠ 基准中无此名称

        # ── BBI ──
        "bbi_momentum": "BBI 动量",

        # ── 动量 ROC ──
        "mom_5d": "5日变动速率（Price Rate of Change）",
        "mom_10d": "10日变动速率（Price Rate of Change）",
        "mom_20d": "20日变动速率（Price Rate of Change）",
        "mom_60d": "60日变动速率（Price Rate of Change）",
        "mom_120d": "120日变动速率（Price Rate of Change）",
        "roc_6d": "6日变动速率（Price Rate of Change）",
        "roc_12d": "12日变动速率（Price Rate of Change）",
        "roc_20d": "20日变动速率（Price Rate of Change）",  # ⚠ 基准无此行
        "roc_60d": "60日变动速率（Price Rate of Change）",
        "roc_120d": "120日变动速率（Price Rate of Change）",

        # ── 波动率 ──
        "ret_vol_20d": "ret_vol_20d",  # ⚠ 基准中无此名称，用原名
        "ret_vol_60d": "ret_vol_60d",  # ⚠ 基准中无此名称
        "ret_vol_120d": "ret_vol_120d",  # ⚠ 基准中无此名称
        "ret_var_20d": "20日收益方差",
        "ret_var_60d": "60日收益方差",
        # ret_var_120d: 基准中有"120日收益方差"

        # ── 夏普比率 ──
        "sharpe_20d": "20日夏普比率",
        "sharpe_60d": "60日夏普比率",
        "sharpe_120d": "120日夏普比率",

        # ── 52周价格位置 ──
        "price_pos_52w": "当前价格处于过去1年股价的位置",
        "price_pos_60d": "price_pos_60d",  # ⚠ 基准中无此名称
        "close_to_high_20d": "20日最高价位置",  # ⚠ 基准中无此行
        "close_to_high_60d": "60日最高价位置",  # ⚠ 基准中无此行

        # ── 线性回归斜率 ──
        "slope_6d": "6日收盘价格与日期线性回归系数",
        "slope_12d": "12日收盘价格与日期线性回归系数",
        "slope_24d": "24日收盘价格与日期线性回归系数",

        # ── MACD ──
        "macd_dif": "平滑异同移动平均线",
        "macd_signal": "平滑异同移动平均线",  # ⚠ 基准中无signal单独名称

        # ── 量比 ──
        "volume_ratio": "成交量比率（Volume Ratio）",

        # ── 基本面 ──
        "pb_factor": "市净率因子",
        "pe_ttm_factor": "市盈率因子",  # ⚠ 基准中可能叫"市盈率TTM"
        "total_mv_log": "对数总市值",
        "turnover_rate_raw": "换手率",  # ⚠ 基准中可能有不同名称
        "dv_ttm_factor": "dv_ttm",  # ⚠ 基准中不确定
    }

    # 加载因子数据做对照
    import csv
    ref = {}
    with open(PROJECT_ROOT / "docs/factor_data.csv", "r", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["时间段"] == "全a 近10年":
                ref[r["因子名称"]] = r

    results = []
    factors = list(df_factors.columns)
    print(f"\n开始分位数回测: {len(factors)} 个因子, {N_QUANTILES} 分位")

    for i, col in enumerate(factors, 1):
        r = quantile_backtest(df_factors[col], ret)
        if r is None:
            continue

        cn_name = name_map.get(col, col)
        ref_row = ref.get(cn_name)

        row = {
            "因子": col,
            "中文名": cn_name,
            "IC均值": r["IC均值"],
            "IR": r["IR"],
            "最小分位超额年化": r["最小分位超额年化"],
            "最大分位超额年化": r["最大分位超额年化"],
            "最小分位换手率": r["最小分位换手率"],
            "最大分位换手率": r["最大分位换手率"],
        }
        if ref_row:
            row["参考IC"] = float(ref_row["IC均值"]) if ref_row["IC均值"] else None
            row["参考IR"] = float(ref_row["IR值"]) if ref_row["IR值"] else None
            row["参考最小超额"] = ref_row["最小分位数超额年化收益率"]
            row["参考最大超额"] = ref_row["最大分位数超额年化收益率"]
            row["参考最小换手"] = ref_row["最小分位数换手率"]
            row["参考最大换手"] = ref_row["最大分位数换手率"]

        results.append(row)
        if i % 10 == 0:
            print(f"  [{i}/{len(factors)}]")

    df_out = pd.DataFrame(results).sort_values("IR", key=lambda x: abs(x), ascending=False)
    out_path = PROJECT_ROOT / "results" / "quantile_backtest_comparison.csv"
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig", float_format="%.6f")

    # 打印对比表
    print("\n" + "=" * 130)
    print("  分位数回测 vs 因子数据 对照表（按 |IR| 降序）")
    print("=" * 130)
    pd.set_option("display.max_rows", 80)
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")

    display = []
    for _, row in df_out.iterrows():
        line = {
            "因子": row["中文名"][:20],
            "IC": row["IC均值"],
            "参考IC": row.get("参考IC", ""),
            "IR": row["IR"],
            "参考IR": row.get("参考IR", ""),
            "最小超额": f"{row['最小分位超额年化']:.2%}",
            "参考最小超额": row.get("参考最小超额", ""),
            "最大超额": f"{row['最大分位超额年化']:.2%}",
            "参考最大超额": row.get("参考最大超额", ""),
        }
        display.append(line)

    df_display = pd.DataFrame(display)
    print(df_display.to_string(index=False))

    print(f"\n结果已保存: {out_path}")
    print(f"总耗时: {(time.time() - t0)/60:.1f} 分钟")
    return df_out


if __name__ == "__main__":
    main()
