#!/usr/bin/env python3
"""
全因子 IC/IR 扫描 — 70个跨期稳健因子 + 扩展因子
所有因子均用 qlib 表达式，从 bin 文件直接计算
"""
import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.qlib_init import init_qlib, load_features_safe
from core.universe import filter_instruments

START_DATE = "2019-01-01"
END_DATE = "2026-03-08"


def get_all_factors():
    """返回 {因子名: qlib表达式} 字典"""
    ret = "($close - Ref($close, 1)) / Ref($close, 1)"

    factors = {
        # ═══════════════════════════════════════════════
        # 换手率比值类（跨期最强大类，IR 0.5~0.7）
        # ═══════════════════════════════════════════════
        "turnover_5d": "Mean($turnover_rate_f, 5)",
        "turnover_10d": "Mean($turnover_rate_f, 10)",
        "turnover_20d": "Mean($turnover_rate_f, 20)",
        "turnover_60d": "Mean($turnover_rate_f, 60)",
        "turnover_120d": "Mean($turnover_rate_f, 120)",
        "turnover_5d_120d_ratio": "Mean($turnover_rate_f, 5) / (Mean($turnover_rate_f, 120) + 1e-8)",
        "turnover_10d_120d_ratio": "Mean($turnover_rate_f, 10) / (Mean($turnover_rate_f, 120) + 1e-8)",
        "turnover_20d_120d_ratio": "Mean($turnover_rate_f, 20) / (Mean($turnover_rate_f, 120) + 1e-8)",
        "turnover_volatility": "Std($turnover_rate_f, 20) / (Mean($turnover_rate_f, 20) + 1e-8)",

        # ═══════════════════════════════════════════════
        # 成交量/成交额标准差（IR 0.5~0.7）
        # ═══════════════════════════════════════════════
        "vol_std_10d": "Std($volume, 10) / (Mean($volume, 10) + 1)",
        "vol_std_20d_v": "Std($volume, 20) / (Mean($volume, 20) + 1)",
        "amt_std_6d": "Std($amount, 6) / (Mean($amount, 6) + 1)",
        "amt_std_20d": "Std($amount, 20) / (Mean($amount, 20) + 1)",

        # ═══════════════════════════════════════════════
        # 成交量EMA / 移动平均（IR 0.4~0.6）
        # ═══════════════════════════════════════════════
        "vol_ema5_ratio": "$volume / (EMA($volume, 5) + 1)",
        "vol_ema10_ratio": "$volume / (EMA($volume, 10) + 1)",
        "vol_ema26_ratio": "$volume / (EMA($volume, 26) + 1)",
        "vol_ma12": "$volume / (Mean($volume, 12) + 1)",
        "amt_ma6": "$amount / (Mean($amount, 6) + 1)",
        "amt_ma20": "$amount / (Mean($amount, 20) + 1)",

        # ═══════════════════════════════════════════════
        # 量变动速率 ROC (Volume)（IR 0.5~0.6）
        # ═══════════════════════════════════════════════
        "vol_roc_6d": "$volume / (Ref($volume, 6) + 1) - 1",
        "vol_roc_12d": "$volume / (Ref($volume, 12) + 1) - 1",

        # ═══════════════════════════════════════════════
        # 成交量震荡 Volume Oscillator（IR 0.5~0.6）
        # ═══════════════════════════════════════════════
        "vol_oscillator": "(EMA($volume, 5) - EMA($volume, 10)) / (EMA($volume, 10) + 1)",
        "vol_oscillator2": "(EMA($volume, 12) - EMA($volume, 26)) / (EMA($volume, 26) + 1)",

        # ═══════════════════════════════════════════════
        # 价量交叉类（IR 0.4~0.6）
        # ═══════════════════════════════════════════════
        # 单日价量趋势 PVT = sum(ret * volume)，用均值近似
        "pvt_6d": f"Mean(({ret}) * $volume, 6)",
        "pvt_12d": f"Mean(({ret}) * $volume, 12)",
        # 量价乘积
        "vol_ret_product": f"$volume / (Mean($volume, 20) + 1) * Mean({ret}, 20)",
        # VMACD 中间变量
        "vmacd": "(EMA($volume, 12) - EMA($volume, 26)) / (EMA($volume, 26) + 1)",

        # ═══════════════════════════════════════════════
        # 均线 / EMA 偏离（IR 0.4~0.7）
        # ═══════════════════════════════════════════════
        "ma5_dev": "$close / Mean($close, 5) - 1",
        "ma10_dev": "$close / Mean($close, 10) - 1",
        "ma20_dev": "$close / Mean($close, 20) - 1",
        "ma60_dev": "$close / Mean($close, 60) - 1",
        "ma120_dev": "$close / Mean($close, 120) - 1",
        "ema5_dev": "$close / EMA($close, 5) - 1",
        "ema10_dev": "$close / EMA($close, 10) - 1",
        "ema12_dev": "$close / EMA($close, 12) - 1",
        "ema20_dev": "$close / EMA($close, 20) - 1",
        "ema26_dev": "$close / EMA($close, 26) - 1",
        "ema60_dev": "$close / EMA($close, 60) - 1",
        "ema120_dev": "$close / EMA($close, 120) - 1",

        # ═══════════════════════════════════════════════
        # 乖离率 BIAS（IR 0.4~0.6）
        # ═══════════════════════════════════════════════
        "bias_5d": "($close - Mean($close, 5)) / Mean($close, 5)",
        "bias_10d": "($close - Mean($close, 10)) / Mean($close, 10)",
        "bias_20d": "($close - Mean($close, 20)) / Mean($close, 20)",
        "bias_60d": "($close - Mean($close, 60)) / Mean($close, 60)",

        # ═══════════════════════════════════════════════
        # 布林线（IR 0.5~0.8）
        # ═══════════════════════════════════════════════
        "bb_lower_dist": "($close - (Mean($close, 20) - 2 * Std($close, 20))) / $close",
        "bb_upper_dist": "((Mean($close, 20) + 2 * Std($close, 20)) - $close) / $close",
        "bb_width": "Std($close, 20) / (Mean($close, 20) + 1e-8)",
        "bb_position": "($close - (Mean($close, 20) - 2 * Std($close, 20))) / (4 * Std($close, 20) + 1e-8)",

        # ═══════════════════════════════════════════════
        # BBI 动量（IR 0.5~0.6）
        # ═══════════════════════════════════════════════
        "bbi_momentum": "$close / ((Mean($close, 3) + Mean($close, 6) + Mean($close, 12) + Mean($close, 24)) / 4) - 1",

        # ═══════════════════════════════════════════════
        # 动量 / 变动速率 ROC（IR 0.4~0.6）
        # ═══════════════════════════════════════════════
        "mom_5d": "$close / Ref($close, 5) - 1",
        "mom_10d": "$close / Ref($close, 10) - 1",
        "mom_20d": "$close / Ref($close, 20) - 1",
        "mom_60d": "$close / Ref($close, 60) - 1",
        "mom_120d": "$close / Ref($close, 120) - 1",
        "roc_6d": "$close / Ref($close, 6) - 1",
        "roc_12d": "$close / Ref($close, 12) - 1",
        "roc_20d": "$close / Ref($close, 20) - 1",
        "roc_60d": "$close / Ref($close, 60) - 1",
        "roc_120d": "$close / Ref($close, 120) - 1",

        # ═══════════════════════════════════════════════
        # 波动率 / 方差（IR 0.3~0.5）
        # ═══════════════════════════════════════════════
        "ret_vol_20d": f"-1 * Std({ret}, 20)",
        "ret_vol_60d": f"-1 * Std({ret}, 60)",
        "ret_vol_120d": f"-1 * Std({ret}, 120)",
        "ret_var_20d": f"-1 * Std({ret}, 20) * Std({ret}, 20)",
        "ret_var_60d": f"-1 * Std({ret}, 60) * Std({ret}, 60)",

        # ═══════════════════════════════════════════════
        # 偏度 / 峰度（IR 0.4~0.6）
        # ═══════════════════════════════════════════════
        # qlib 没有 Skew/Kurt/Pow 算子，跳过
        # "ret_skew_20d_proxy": ...

        # ═══════════════════════════════════════════════
        # 夏普比率（IR 0.4~0.5）
        # ═══════════════════════════════════════════════
        "sharpe_20d": f"Mean({ret}, 20) / (Std({ret}, 20) + 1e-8)",
        "sharpe_60d": f"Mean({ret}, 60) / (Std({ret}, 60) + 1e-8)",
        "sharpe_120d": f"Mean({ret}, 120) / (Std({ret}, 120) + 1e-8)",

        # ═══════════════════════════════════════════════
        # 52周价格位置（IR 0.3~0.5）
        # ═══════════════════════════════════════════════
        "price_pos_52w": "($close - Min($close, 252)) / (Max($close, 252) - Min($close, 252) + 1e-8)",
        "price_pos_60d": "($close - Min($close, 60)) / (Max($close, 60) - Min($close, 60) + 1e-8)",
        "close_to_high_20d": "$close / Max($close, 20) - 1",
        "close_to_high_60d": "$close / Max($close, 60) - 1",

        # ═══════════════════════════════════════════════
        # 人气/多空力道（IR 0.4~0.6）
        # ═══════════════════════════════════════════════
        # $high/$low 字段可能损坏，跳过以下因子：
        # ar_indicator, br_indicator, bull_power, bear_power

        # ═══════════════════════════════════════════════
        # CR 指标 / ATR / 振幅 / MFI - 跳过（依赖 $high/$low）
        # ═══════════════════════════════════════════════

        # ═══════════════════════════════════════════════
        # 线性回归斜率（IR 0.3~0.5）
        # qlib 有 Slope 算子
        # ═══════════════════════════════════════════════
        "slope_6d": "Slope($close, 6) / $close",
        "slope_12d": "Slope($close, 12) / $close",
        "slope_24d": "Slope($close, 24) / $close",

        # ═══════════════════════════════════════════════
        # MACD 类（IR 0.3~0.5）
        # ═══════════════════════════════════════════════
        "macd_dif": "(EMA($close, 12) - EMA($close, 26)) / $close",
        "macd_signal": "EMA(EMA($close, 12) - EMA($close, 26), 9) / $close",

        # ═══════════════════════════════════════════════
        # 资金流量 MFI - 跳过（依赖 $high/$low）
        # ═══════════════════════════════════════════════

        # ═══════════════════════════════════════════════
        # 量比 Volume Ratio（IR 0.3~0.4）
        # ═══════════════════════════════════════════════
        "volume_ratio": "$volume / (Mean($volume, 20) + 1)",

        # ═══════════════════════════════════════════════
        # 基本面（从 bin 文件读取）
        # ═══════════════════════════════════════════════
        "pb_factor": "$pb",
        "pe_ttm_factor": "$pe_ttm",
        "total_mv_log": "Log($total_mv + 1)",
        "turnover_rate_raw": "$turnover_rate_f",
        "dv_ttm_factor": "$dv_ttm",
    }
    return factors


def get_weekly_factors():
    """周频专用因子：日频周期 × 5，适配周频调仓节奏"""
    ret = "($close - Ref($close, 1)) / Ref($close, 1)"

    factors = {
        # ═══════════════════════════════════════════════
        # 换手率（周频版：25d=5周, 50d=10周, 100d=20周, 150d=30周, 250d≈1年）
        # ═══════════════════════════════════════════════
        "w_turnover_25d": "Mean($turnover_rate_f, 25)",
        "w_turnover_50d": "Mean($turnover_rate_f, 50)",
        "w_turnover_100d": "Mean($turnover_rate_f, 100)",
        "w_turnover_150d": "Mean($turnover_rate_f, 150)",
        "w_turnover_250d": "Mean($turnover_rate_f, 250)",
        "w_turnover_25d_250d_ratio": "Mean($turnover_rate_f, 25) / (Mean($turnover_rate_f, 250) + 1e-8)",
        "w_turnover_50d_250d_ratio": "Mean($turnover_rate_f, 50) / (Mean($turnover_rate_f, 250) + 1e-8)",
        "w_turnover_100d_250d_ratio": "Mean($turnover_rate_f, 100) / (Mean($turnover_rate_f, 250) + 1e-8)",

        # ═══════════════════════════════════════════════
        # 均线偏离（周频版）
        # ═══════════════════════════════════════════════
        "w_ma25_dev": "$close / Mean($close, 25) - 1",
        "w_ma50_dev": "$close / Mean($close, 50) - 1",
        "w_ma100_dev": "$close / Mean($close, 100) - 1",
        "w_ma150_dev": "$close / Mean($close, 150) - 1",
        "w_ma250_dev": "$close / Mean($close, 250) - 1",
        "w_ema25_dev": "$close / EMA($close, 25) - 1",
        "w_ema50_dev": "$close / EMA($close, 50) - 1",
        "w_ema100_dev": "$close / EMA($close, 100) - 1",
        "w_ema150_dev": "$close / EMA($close, 150) - 1",
        "w_ema250_dev": "$close / EMA($close, 250) - 1",

        # ═══════════════════════════════════════════════
        # 动量（周频版）
        # ═══════════════════════════════════════════════
        "w_mom_25d": "$close / Ref($close, 25) - 1",
        "w_mom_50d": "$close / Ref($close, 50) - 1",
        "w_mom_100d": "$close / Ref($close, 100) - 1",
        "w_mom_150d": "$close / Ref($close, 150) - 1",
        "w_mom_250d": "$close / Ref($close, 250) - 1",

        # ═══════════════════════════════════════════════
        # 波动率（周频版）
        # ═══════════════════════════════════════════════
        "w_ret_vol_50d": f"-1 * Std({ret}, 50)",
        "w_ret_vol_100d": f"-1 * Std({ret}, 100)",
        "w_ret_vol_150d": f"-1 * Std({ret}, 150)",
        "w_ret_vol_250d": f"-1 * Std({ret}, 250)",

        # ═══════════════════════════════════════════════
        # 成交量（周频版）
        # ═══════════════════════════════════════════════
        "w_vol_std_25d": "Std($volume, 25) / (Mean($volume, 25) + 1)",
        "w_vol_std_50d": "Std($volume, 50) / (Mean($volume, 50) + 1)",
        "w_vol_ema25_ratio": "$volume / (EMA($volume, 25) + 1)",
        "w_vol_ema50_ratio": "$volume / (EMA($volume, 50) + 1)",
        "w_vol_ema130_ratio": "$volume / (EMA($volume, 130) + 1)",

        # ═══════════════════════════════════════════════
        # 布林线（周频版：100日=20周）
        # ═══════════════════════════════════════════════
        "w_bb_position_100": "($close - (Mean($close, 100) - 2 * Std($close, 100))) / (4 * Std($close, 100) + 1e-8)",
        "w_bb_width_100": "Std($close, 100) / (Mean($close, 100) + 1e-8)",

        # ═══════════════════════════════════════════════
        # 夏普比率（周频版）
        # ═══════════════════════════════════════════════
        "w_sharpe_50d": f"Mean({ret}, 50) / (Std({ret}, 50) + 1e-8)",
        "w_sharpe_100d": f"Mean({ret}, 100) / (Std({ret}, 100) + 1e-8)",
        "w_sharpe_150d": f"Mean({ret}, 150) / (Std({ret}, 150) + 1e-8)",

        # ═══════════════════════════════════════════════
        # 斜率（周频版）
        # ═══════════════════════════════════════════════
        "w_slope_25d": "Slope($close, 25) / $close",
        "w_slope_50d": "Slope($close, 50) / $close",
        "w_slope_100d": "Slope($close, 100) / $close",

        # ═══════════════════════════════════════════════
        # MACD（周频版：60/130≈12周/26周）
        # ═══════════════════════════════════════════════
        "w_macd_dif": "(EMA($close, 60) - EMA($close, 130)) / $close",
        "w_macd_signal": "EMA(EMA($close, 60) - EMA($close, 130), 45) / $close",

        # ═══════════════════════════════════════════════
        # 价格位置（周频版）
        # ═══════════════════════════════════════════════
        "w_close_to_high_50d": "$close / Max($close, 50) - 1",
        "w_close_to_high_100d": "$close / Max($close, 100) - 1",

        # ═══════════════════════════════════════════════
        # 基本面（不变）
        # ═══════════════════════════════════════════════
        "pb_factor": "$pb",
        "pe_ttm_factor": "$pe_ttm",
        "total_mv_log": "Log($total_mv + 1)",
        "turnover_rate_raw": "$turnover_rate_f",
        "dv_ttm_factor": "$dv_ttm",
    }
    return factors


def load_all_data():
    """加载因子和前向收益"""
    init_qlib()
    from qlib.data import D

    print("[1/3] 加载股票列表...")
    instruments = D.instruments(market="all")
    df_close = load_features_safe(
        instruments, ["$close"],
        start_time=START_DATE, end_time=END_DATE, freq="day"
    )
    valid = filter_instruments(
        df_close.index.get_level_values("instrument").unique().tolist()
    )
    print(f"  股票数: {len(valid)}")

    print("[2/3] 计算前向收益...")
    df_ret = load_features_safe(
        valid, ["Ref($close, -1) / $close - 1"],
        start_time=START_DATE, end_time=END_DATE, freq="day"
    )
    df_ret.columns = ["fwd_ret"]

    print("[3/3] 加载因子表达式...")
    all_factors = get_all_factors()

    # 分批加载（每批5个，避免内存和报错问题）
    batch_size = 5
    items = list(all_factors.items())
    df_parts = []
    failed = []

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
            print(f"  [{b + batch_size}/{len(items)}] {names} OK")
        except Exception as e:
            # 逐个重试找出问题因子
            for name, expr in batch:
                try:
                    df_single = load_features_safe(
                        valid, [expr],
                        start_time=START_DATE, end_time=END_DATE, freq="day"
                    )
                    df_single.columns = [name]
                    df_parts.append(df_single)
                except Exception as e2:
                    failed.append((name, str(e2)[:60]))
                    print(f"  FAIL: {name} - {str(e2)[:60]}")

    if failed:
        print(f"\n失败因子: {len(failed)}")
        for name, err in failed:
            print(f"  {name}: {err}")

    df_all = pd.concat(df_parts, axis=1) if df_parts else pd.DataFrame()
    print(f"\n成功加载: {len(df_all.columns)} 个因子")
    return df_all, df_ret, valid


def compute_ic_series(factor_values, fwd_ret):
    """按日期计算截面 Rank IC"""
    df = pd.DataFrame({"factor": factor_values, "ret": fwd_ret}).dropna()
    if df.empty:
        return pd.Series(dtype=float)

    def _rank_ic(group):
        if len(group) < 30:
            return np.nan
        return group["factor"].corr(group["ret"], method="spearman")

    return df.groupby(level="datetime").apply(_rank_ic).dropna()


def scan_all_factors(df_all, df_ret):
    """扫描所有因子的 IC/IR"""
    fwd_ret = df_ret["fwd_ret"]
    results = []
    factors = list(df_all.columns)
    print(f"\n扫描 {len(factors)} 个因子...")

    for i, col in enumerate(factors, 1):
        ic_series = compute_ic_series(df_all[col], fwd_ret)
        if len(ic_series) < 50:
            continue

        ic_mean = ic_series.mean()
        ic_std = ic_series.std()
        ir = ic_mean / ic_std if ic_std > 0 else 0.0
        ic_pos_rate = (ic_series > 0).mean()

        yearly_ir = {}
        for year in range(2019, 2027):
            yr_ic = ic_series[ic_series.index.year == year]
            if len(yr_ic) > 20:
                yearly_ir[year] = yr_ic.mean() / yr_ic.std() if yr_ic.std() > 0 else 0.0

        # 跨期稳定性：各年IR同号的比例
        yr_irs = [v for v in yearly_ir.values() if not np.isnan(v)]
        if yr_irs:
            sign_consistency = sum(1 for v in yr_irs if (v > 0) == (yr_irs[0] > 0)) / len(yr_irs)
        else:
            sign_consistency = 0.0

        direction = "正向" if ic_mean >= 0 else "反向(negate)"

        results.append({
            "因子": col,
            "IC均值": ic_mean,
            "|IC|": abs(ic_mean),
            "IR": ir,
            "|IR|": abs(ir),
            "IC胜率": ic_pos_rate,
            "方向": direction,
            "稳定性": sign_consistency,
            "有效天数": len(ic_series),
            **{f"IR_{y}": yearly_ir.get(y, np.nan) for y in range(2019, 2027)},
        })

        if i % 20 == 0:
            print(f"  [{i}/{len(factors)}]")

    df_results = pd.DataFrame(results).sort_values("|IR|", ascending=False)
    return df_results


def main():
    import time
    start = time.time()

    df_all, df_ret, valid = load_all_data()
    df_results = scan_all_factors(df_all, df_ret)

    out_path = PROJECT_ROOT / "results" / "factor_scan_v2.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(out_path, index=False, encoding="utf-8-sig", float_format="%.4f")
    print(f"\n结果已保存: {out_path}")

    print("\n" + "=" * 150)
    print("  全因子扫描结果（按 |IR| 降序）")
    print("=" * 150)

    pd.set_option("display.max_rows", 120)
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")

    cols = ["因子", "IC均值", "IR", "|IR|", "IC胜率", "方向", "稳定性",
            "IR_2019", "IR_2020", "IR_2021", "IR_2022", "IR_2023", "IR_2024", "IR_2025"]
    print(df_results[cols].to_string(index=False))

    strong = df_results[df_results["|IR|"] >= 0.3]
    moderate = df_results[(df_results["|IR|"] >= 0.2) & (df_results["|IR|"] < 0.3)]
    print(f"\n强因子 (|IR|≥0.3): {len(strong)}")
    print(f"中等因子 (0.2≤|IR|<0.3): {len(moderate)}")
    print(f"弱因子 (|IR|<0.2): {len(df_results) - len(strong) - len(moderate)}")

    # 跨期稳定（所有年份同方向）
    stable = df_results[(df_results["稳定性"] >= 1.0) & (df_results["|IR|"] >= 0.2)]
    print(f"跨期稳定且|IR|≥0.2: {len(stable)}")

    elapsed = time.time() - start
    print(f"\n总耗时: {elapsed:.1f}秒 ({elapsed/60:.1f}分钟)")
    return df_results


if __name__ == "__main__":
    df = main()
