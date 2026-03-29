#!/usr/bin/env python3
"""
批量回测：stoploss 策略扫描 + 月频 baseline
直接调用底层回测引擎
"""
import warnings
warnings.filterwarnings("ignore")

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def analyze_result(result, name):
    """分析回测结果"""
    dr = result.daily_returns
    if dr.empty or len(dr) < 50:
        return None

    nav = (1 + dr).cumprod()
    years = len(dr) / 252
    ending = float(nav.iloc[-1])
    ann = (ending ** (1 / years) - 1) if years > 0 and ending > 0 else -1
    sharpe = float(dr.mean() / (dr.std() + 1e-10) * 252 ** 0.5)
    max_dd = float((nav / nav.cummax() - 1).min())
    win_rate = float((dr > 0).mean())

    yearly = {}
    for year in range(2019, 2027):
        yr = dr[dr.index.year == year]
        if len(yr) > 20:
            yearly[year] = float((1 + yr).prod() - 1)

    return {
        "name": name, "ann": ann, "sharpe": sharpe,
        "max_dd": max_dd, "win_rate": win_rate, "yearly": yearly,
    }


def main():
    from core.strategy import Strategy
    from modules.backtest.qlib_engine import QlibBacktestEngine

    print("=" * 80)
    print("  Stoploss 策略扫描 (2019-2026)")
    print("=" * 80)

    # 要测试的策略
    strategies = [
        # stoploss_replace 系列（不同 topk）
        "experimental/value_plan/value_book_ocf_stoploss_replace_15",
        "experimental/value_plan/value_book_ocf_stoploss_replace_5",
        "experimental/value_plan/value_book_ocf_stoploss_replace_3",

        # 单因子 stoploss
        "experimental/value_plan/value_book_stoploss_15",
        "experimental/value_plan/value_ocf_stoploss_15",
        "experimental/value_plan/value_retained_stoploss_15",
        "experimental/value_plan/value_ebit_stoploss_15",
        "experimental/value_plan/value_roa_stoploss_15",
        "experimental/value_plan/value_turnover_stoploss_15",
        "experimental/value_plan/value_mom20_stoploss_15",
        "experimental/value_plan/value_pricepos52w_stoploss_15",

        # 双因子 stoploss
        "experimental/value_plan/value_book_ocf_retained_stoploss_15",
        "experimental/value_plan/value_ocf_ebit_stoploss_15",
        "experimental/value_plan/value_ocf_retained_stoploss_15",

        # 事件驱动
        "experimental/value_plan/value_book_ocf_event_15",
        "experimental/value_plan/value_book_ocf_event_10",

        # 现金流守卫
        "experimental/value_plan/value_book_ocf_cashflow_guard_15",
    ]

    engine = QlibBacktestEngine()
    results = []

    for i, strat_name in enumerate(strategies, 1):
        short_name = strat_name.split("/")[-1]
        print(f"\n[{i}/{len(strategies)}] {short_name}")
        t0 = time.time()
        try:
            strategy = Strategy.load(strat_name)
            result = engine.run(strategy)
            elapsed = time.time() - t0

            res = analyze_result(result, short_name)
            if res:
                print(f"  年化{res['ann']:+.2%} 夏普{res['sharpe']:.2f} 回撤{res['max_dd']:.1%} 耗时{elapsed:.0f}s")
                results.append(res)
            else:
                print(f"  数据不足")
        except Exception as e:
            print(f"  ERROR: {str(e)[:120]}")

    # 汇总
    print("\n" + "=" * 80)
    print("  扫描结果（按夏普排序）")
    print("=" * 80)
    print(f"{'策略':<45} {'年化':>8} {'夏普':>6} {'回撤':>8} {'胜率':>6}")
    print("-" * 80)
    results.sort(key=lambda x: x["sharpe"], reverse=True)
    for r in results:
        print(f"{r['name']:<45} {r['ann']:>+8.2%} {r['sharpe']:>6.2f} {r['max_dd']:>8.1%} {r['win_rate']:>6.1%}")

    # Top 3 分年度对比
    print("\n" + "=" * 80)
    print("  Top 3 分年度对比")
    print("=" * 80)
    for r in results[:3]:
        print(f"\n{r['name']}: 年化{r['ann']:+.2%} 夏普{r['sharpe']:.2f} 回撤{r['max_dd']:.1%}")
        print(f"  {'年份':^6} {'收益':>10}")
        print(f"  {'-'*20}")
        for y in range(2019, 2027):
            if y in r["yearly"]:
                print(f"  {y:^6} {r['yearly'][y]:>+10.1%}")

    # 保存
    out_path = PROJECT_ROOT / "results" / "stoploss_scan_summary.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\n已保存: {out_path}")


if __name__ == "__main__":
    main()
