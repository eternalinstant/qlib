#!/usr/bin/env python3
"""
Alpha158 因子 IC/IR 扫描
动态从 qlib.contrib.data.handler.Alpha158 获取 158 个因子表达式，
复用 factor_scan.py 的 compute_ic_series / scan_all_factors 函数。
"""
import warnings
warnings.filterwarnings("ignore")

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.qlib_init import init_qlib, load_features_safe
from core.universe import get_universe_instruments, filter_instruments
from scripts.factor_scan import compute_ic_series, scan_all_factors

START_DATE = "2019-01-01"
END_DATE = "2026-03-08"
UNIVERSE = "csi300"
OUTPUT_PATH = PROJECT_ROOT / "results" / "alpha158_scan.csv"


def get_alpha158_factors():
    """从 Alpha158DL 静态方法获取 158 个因子表达式和名称"""
    from qlib.contrib.data.loader import Alpha158DL

    conf = {
        "kbar": {},
        "price": {
            "windows": [0],
            "feature": ["OPEN", "HIGH", "LOW", "VWAP"],
        },
        "rolling": {},
    }
    fields, names = Alpha158DL.get_feature_config(conf)
    assert len(fields) == len(names) == 158, f"Expected 158 factors, got {len(fields)}"
    return fields, names


def load_alpha158_data(fields, names, instruments):
    """分批加载 158 个因子数据"""
    batch_size = 5
    df_parts = []
    failed = []

    for b in range(0, len(fields), batch_size):
        batch_fields = fields[b : b + batch_size]
        batch_names = names[b : b + batch_size]
        try:
            df_part = load_features_safe(
                instruments,
                batch_fields,
                start_time=START_DATE,
                end_time=END_DATE,
                freq="day",
            )
            df_part.columns = batch_names
            df_parts.append(df_part)
            print(f"  [{b + batch_size}/{len(fields)}] {batch_names} OK")
        except Exception:
            for name, expr in zip(batch_names, batch_fields):
                try:
                    df_single = load_features_safe(
                        instruments,
                        [expr],
                        start_time=START_DATE,
                        end_time=END_DATE,
                        freq="day",
                    )
                    df_single.columns = [name]
                    df_parts.append(df_single)
                except Exception as e2:
                    failed.append((name, str(e2)[:80]))
                    print(f"  FAIL: {name} - {str(e2)[:80]}")

    if failed:
        print(f"\n失败因子: {len(failed)}")
        for name, err in failed:
            print(f"  {name}: {err}")

    df_all = pd.concat(df_parts, axis=1) if df_parts else pd.DataFrame()
    print(f"\n成功加载: {len(df_all.columns)} 个因子")
    return df_all


def main():
    start = time.time()

    # 1. 获取因子列表
    fields, names = get_alpha158_factors()
    print(f"Alpha158 因子数: {len(fields)}")

    # 2. 初始化 qlib + 获取股票池
    init_qlib()

    print(f"\n[1/3] 获取 {UNIVERSE} 股票池...")
    instruments = get_universe_instruments(START_DATE, END_DATE, universe=UNIVERSE)
    instruments = filter_instruments(instruments)
    print(f"  股票数: {len(instruments)}")

    # 3. 计算前向收益
    print("[2/3] 计算前向收益...")
    df_ret = load_features_safe(
        instruments,
        ["Ref($close, -1) / $close - 1"],
        start_time=START_DATE,
        end_time=END_DATE,
        freq="day",
    )
    df_ret.columns = ["fwd_ret"]

    # 4. 加载因子数据
    print("[3/3] 加载 Alpha158 因子...")
    df_all = load_alpha158_data(fields, names, instruments)

    # 5. 扫描 IC/IR
    df_results = scan_all_factors(df_all, df_ret)

    # 6. 保存结果
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig", float_format="%.4f")
    print(f"\n结果已保存: {OUTPUT_PATH}")

    # 7. 打印 top-30 by |IR|
    print("\n" + "=" * 150)
    print("  Alpha158 因子扫描结果 — Top 30 by |IR|")
    print("=" * 150)

    pd.set_option("display.max_rows", 120)
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")

    cols = ["因子", "IC均值", "IR", "|IR|", "IC胜率", "方向", "稳定性"]
    top30 = df_results.head(30)
    print(top30[cols].to_string(index=False))

    # 8. 统计摘要
    strong = df_results[df_results["|IR|"] >= 0.3]
    moderate = df_results[(df_results["|IR|"] >= 0.15) & (df_results["|IR|"] < 0.3)]
    stable_strong = df_results[(df_results["|IR|"] >= 0.15) & (df_results["稳定性"] >= 0.7)]
    print(f"\n强因子 (|IR|>=0.3): {len(strong)}")
    print(f"中等因子 (0.15<=|IR|<0.3): {len(moderate)}")
    print(f"|IR|>=0.15 且 稳定性>=0.7: {len(stable_strong)}")

    elapsed = time.time() - start
    print(f"\n总耗时: {elapsed:.1f}秒 ({elapsed/60:.1f}分钟)")
    return df_results


if __name__ == "__main__":
    df = main()
