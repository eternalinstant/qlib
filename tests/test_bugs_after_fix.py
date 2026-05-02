"""
修复后验证脚本 — 确认所有bug已修复
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import inspect

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"


def test_fix1_qlib_negate():
    """验证: Qlib source 因子 negate 现在已生效"""
    print("\n" + "="*70)
    print("Fix#1: Qlib source 因子 negate")
    print("="*70)

    from core.selection import load_factor_data
    source = inspect.getsource(load_factor_data)

    idx = source.find("df_qlib.columns = qlib_names")
    after = source[idx:idx+500]
    has_negate = "negate" in after and "f.negate" in after

    print(f"  df_qlib 重命名后有 negate 处理: {has_negate}")

    if has_negate:
        # 检查具体逻辑
        has_multiply_neg = "-df_qlib" in after or "= -" in after
        print(f"  对 negate=True 列取负: {has_multiply_neg}")
        print(f"  {PASS} Bug#1 已修复")
        return True
    else:
        print(f"  {FAIL} Bug#1 未修复")
        return False


def test_fix2_st_no_lookahead():
    """验证: 选股和回测中关闭ST过滤"""
    print("\n" + "="*70)
    print("Fix#2: ST名单前视偏差")
    print("="*70)

    from core.selection import load_factor_data
    sel_source = inspect.getsource(load_factor_data)
    sel_no_st = "exclude_st=False" in sel_source
    print(f"  selection.py 关闭ST过滤: {sel_no_st}")

    from modules.backtest.qlib_engine import QlibBacktestEngine
    bt_source = inspect.getsource(QlibBacktestEngine.run)
    bt_no_st = "exclude_st=False" in bt_source
    print(f"  qlib_engine.py 关闭ST过滤: {bt_no_st}")

    if sel_no_st and bt_no_st:
        print(f"  {PASS} Bug#2 已修复（回测中不做ST过滤）")
        return True
    else:
        print(f"  {FAIL} Bug#2 未完全修复")
        return False


def test_fix3_position_prev_day():
    """验证: 仓位控制使用前一天数据"""
    print("\n" + "="*70)
    print("Fix#3: 仓位控制前视偏差")
    print("="*70)

    from core.position import MarketPositionController
    regime_src = inspect.getsource(MarketPositionController._get_regime)
    opp_src = inspect.getsource(MarketPositionController._get_opportunity)

    # 应该使用 iloc[-2] 取前一天
    regime_prev = "iloc[-2]" in regime_src
    opp_prev = "iloc[-2]" in opp_src
    print(f"  _get_regime 使用前一天 (iloc[-2]): {regime_prev}")
    print(f"  _get_opportunity 使用前一天 (iloc[-2]): {opp_prev}")

    if regime_prev and opp_prev:
        print(f"  {PASS} Bug#3 已修复")
        return True
    else:
        print(f"  {FAIL} Bug#3 未修复")
        return False


def test_fix5_ann_date_delay():
    """验证: 财报公告日加1天延迟"""
    print("\n" + "="*70)
    print("Fix#5: 财报公告日延迟")
    print("="*70)

    from modules.data.tushare_to_qlib import TushareToQlibConverter
    source = inspect.getsource(TushareToQlibConverter.convert)

    delay_count = source.count("Timedelta(days=1)")
    print(f"  公告日 +1 天延迟出现次数: {delay_count}")
    print(f"  (预期4次: fina/income/cashflow/balance)")

    if delay_count >= 4:
        print(f"  {PASS} Bug#5 已修复")
        return True
    else:
        print(f"  {FAIL} Bug#5 未完全修复 (只有{delay_count}处)")
        return False


def test_fix6_cost_not_scaled():
    """验证: 交易成本不再被仓位缩小"""
    print("\n" + "="*70)
    print("Fix#6: 交易成本计算")
    print("="*70)

    from modules.backtest.qlib_engine import QlibBacktestEngine
    source = inspect.getsource(QlibBacktestEngine.run)

    # 成本应该在 port_ret 层面扣除，而不是从 stock_ret 扣除
    old_pattern = "stock_ret = stock_ret - total_cost_ratio"
    new_pattern = "cost_deduction"

    has_old = old_pattern in source
    has_new = new_pattern in source

    # 检查成本是否在 port_ret 中扣除
    port_ret_lines = [l for l in source.split("\n") if "port_ret" in l and "cost" in l.lower()]
    print(f"  旧模式（从stock_ret扣）: {'存在' if has_old else '已移除'}")
    print(f"  新模式（cost_deduction）: {'存在' if has_new else '不存在'}")
    print(f"  port_ret 中扣除成本:")
    for l in port_ret_lines:
        print(f"    {l.strip()}")

    if not has_old and has_new and port_ret_lines:
        print(f"  {PASS} Bug#6 已修复")
        return True
    else:
        print(f"  {FAIL} Bug#6 未修复")
        return False


def test_fix7_first_rebal_cost():
    """验证: 首次建仓扣买入成本"""
    print("\n" + "="*70)
    print("Fix#7: 首次建仓成本")
    print("="*70)

    from modules.backtest.qlib_engine import QlibBacktestEngine
    source = inspect.getsource(QlibBacktestEngine.run)

    # 应该不再要求 prev_selected 非空
    old_check = "if is_rebal and prev_selected:"
    has_old = old_check in source

    # 新逻辑应处理 prev_selected 为空的情况
    handles_first = "首次建仓" in source or "buy_count = len(selected)" in source
    print(f"  旧条件（要求prev_selected非空）: {'存在' if has_old else '已修改'}")
    print(f"  处理首次建仓: {handles_first}")

    if not has_old and handles_first:
        print(f"  {PASS} Bug#7 已修复")
        return True
    else:
        print(f"  {FAIL} Bug#7 未修复")
        return False


def test_fix9_delist_penalty():
    """验证: 退市股票有惩罚"""
    print("\n" + "="*70)
    print("Fix#9: 退市股票处理")
    print("="*70)

    from modules.backtest.qlib_engine import QlibBacktestEngine
    source = inspect.getsource(QlibBacktestEngine.run)

    has_missing = "missing_count" in source
    has_penalty = "delist_penalty" in source or "penalty" in source

    print(f"  检测缺失股票: {has_missing}")
    print(f"  施加退市惩罚: {has_penalty}")

    if has_missing and has_penalty:
        print(f"  {PASS} Bug#9 已修复")
        return True
    else:
        print(f"  {FAIL} Bug#9 未修复")
        return False


def main():
    results = {}
    tests = [
        ("Fix#1: qlib因子negate", test_fix1_qlib_negate),
        ("Fix#2: ST前视偏差", test_fix2_st_no_lookahead),
        ("Fix#3: 仓位控制前视偏差", test_fix3_position_prev_day),
        ("Fix#5: 公告日延迟", test_fix5_ann_date_delay),
        ("Fix#6: 交易成本计算", test_fix6_cost_not_scaled),
        ("Fix#7: 首次建仓成本", test_fix7_first_rebal_cost),
        ("Fix#9: 退市股票处理", test_fix9_delist_penalty),
    ]

    for name, func in tests:
        try:
            results[name] = func()
        except Exception as e:
            print(f"\n  {FAIL} 异常: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    # 汇总
    print("\n" + "="*70)
    print("修复验证汇总")
    print("="*70)
    for name, ok in results.items():
        status = PASS if ok else FAIL
        print(f"  {status} {name}")

    fixed = sum(1 for v in results.values() if v)
    print(f"\n  已修复: {fixed}/{len(results)}")


if __name__ == "__main__":
    main()
