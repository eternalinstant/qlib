"""
策略审计验证脚本 — 逐项确认已发现的bug
运行: cd /Users/sxt/code/qlib && python tests/test_bugs.py
"""
import sys
sys.path.insert(0, "/Users/sxt/code/qlib")

import pandas as pd
import numpy as np

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"


def check_bug1_qlib_negate_not_applied():
    """Bug#1: Qlib source 因子的 negate 未在 load_factor_data 中生效"""
    print("\n" + "="*70)
    print("Bug#1: Qlib source 因子 negate 未生效")
    print("="*70)

    from core.factors import FactorRegistry, FactorInfo

    # 构造一个有 negate=True 的 qlib 因子的 registry
    registry = FactorRegistry()
    registry.register(FactorInfo(
        "test_negate", "$close / Mean($close, 10) - 1",
        "test", "enhance", source="qlib", negate=True, ir=0.5
    ))

    # 检查 selection.py 中 load_factor_data 的代码逻辑
    import inspect
    from core.selection import load_factor_data
    source = inspect.getsource(load_factor_data)

    # 在 load_factor_data 中搜索对 qlib 因子 negate 的处理
    has_qlib_negate = "negate" in source and "qlib" in source

    # 对比: _load_parquet_factors 有 negate 处理
    from core.selection import _load_parquet_factors
    parquet_source = inspect.getsource(_load_parquet_factors)
    parquet_has_negate = "f.negate" in parquet_source or "negate" in parquet_source

    print(f"  _load_parquet_factors 中有 negate 处理: {parquet_has_negate}")
    print(f"  load_factor_data 中对 qlib 因子有 negate 处理: {has_qlib_negate}")

    # 精确检查: load_factor_data 在 df_qlib.columns = qlib_names 之后是否有 negate 逻辑
    # 找到 "df_qlib.columns = qlib_names" 之后的代码
    idx = source.find("df_qlib.columns = qlib_names")
    if idx >= 0:
        after_rename = source[idx:idx+300]
        has_negate_after = "negate" in after_rename
        print(f"  df_qlib 重命名后有 negate 处理: {has_negate_after}")
    else:
        has_negate_after = False
        print(f"  未找到 df_qlib.columns = qlib_names 行")

    # 列出所有受影响的因子
    from core.factors import default_registry
    affected = [f for f in default_registry.get_by_source("qlib") if f.negate]
    print(f"\n  受影响因子（qlib source + negate=True）:")
    for f in affected:
        print(f"    - {f.name} (IR={f.ir}, category={f.category})")

    if not has_negate_after and affected:
        print(f"\n  {FAIL} 确认Bug: {len(affected)} 个 qlib 因子的 negate 未生效!")
        return False
    else:
        print(f"\n  {PASS} negate 已正确处理")
        return True


def check_bug2_st_lookahead():
    """Bug#2: ST名单前视偏差 — 用当前名单过滤历史"""
    print("\n" + "="*70)
    print("Bug#2: ST名单前视偏差")
    print("="*70)

    import inspect
    from core.universe import filter_instruments, _load_st_set

    source = inspect.getsource(_load_st_set)

    # 检查是否有日期参数
    has_date_param = "date" in inspect.signature(_load_st_set).parameters
    print(f"  _load_st_set 有 date 参数: {has_date_param}")

    # 检查是否使用了静态缓存
    has_static_cache = "_st_instruments" in source and "global" in source
    print(f"  使用全局静态缓存: {has_static_cache}")

    # 检查数据源
    uses_stock_basic = "stock_basic" in source
    print(f"  从 stock_basic.csv（当前快照）加载: {uses_stock_basic}")

    # 检查是否有时间维度
    has_temporal = "ann_date" in source or "list_date" in source or "delist_date" in source
    print(f"  有时间维度处理: {has_temporal}")

    if has_static_cache and uses_stock_basic and not has_date_param:
        print(f"\n  {FAIL} 确认Bug: ST名单无时间维度，存在前视偏差!")
        # 量化影响
        from pathlib import Path
        csv = Path("/Users/sxt/code/qlib/data/tushare/stock_basic.csv")
        if csv.exists():
            df = pd.read_csv(csv, dtype=str)
            st_count = df[df["name"].str.contains("ST", na=False)].shape[0]
            total = df.shape[0]
            print(f"  当前ST股票数: {st_count}/{total} ({st_count/total:.1%})")
        return False
    else:
        print(f"\n  {PASS} ST名单有时间维度")
        return True


def check_bug3_position_lookahead():
    """Bug#3: 仓位控制用当天收盘价决定当天仓位"""
    print("\n" + "="*70)
    print("Bug#3: 仓位控制使用当天收盘价（前视偏差）")
    print("="*70)

    import inspect
    from core.position import MarketPositionController

    source = inspect.getsource(MarketPositionController._get_regime)
    print(f"  _get_regime 代码:")
    print(f"    {source.strip().split(chr(10))[1].strip()}")

    # 关键: loc[:date] 包含当天
    uses_current_day = "loc[:date]" in source or "loc[:ts]" in source
    print(f"  使用 loc[:date] (包含当天): {uses_current_day}")

    # 检查 _get_opportunity 同样的问题
    opp_source = inspect.getsource(MarketPositionController._get_opportunity)
    opp_uses_current = "loc[:date]" in opp_source or "loc[:ts]" in opp_source
    print(f"  _get_opportunity 同样使用当天数据: {opp_uses_current}")

    if uses_current_day:
        print(f"\n  {FAIL} 确认Bug: MA/drawdown 使用当天收盘价，应使用前一天!")
        return False
    else:
        print(f"\n  {PASS} 已使用前一天数据")
        return True


def check_bug4_report_type_not_filtered():
    """Bug#4: 财务数据未过滤 report_type"""
    print("\n" + "="*70)
    print("Bug#4: 财务数据未过滤 report_type")
    print("="*70)

    # 检查实际数据中是否有重复
    from pathlib import Path
    fina_path = Path("/Users/sxt/code/qlib/data/tushare/fina_indicator.parquet")
    if not fina_path.exists():
        print(f"  {WARN} fina_indicator.parquet 不存在，跳过")
        return None

    fina = pd.read_parquet(fina_path)
    if "report_type" in fina.columns:
        print(f"  report_type 分布:")
        for rt, cnt in fina["report_type"].value_counts().items():
            print(f"    {rt}: {cnt:,}")

        # 检查同一股票同一公告日是否有多条记录
        if "ann_date" in fina.columns:
            dupes = fina.groupby(["ts_code", "ann_date"]).size()
            multi = dupes[dupes > 1]
            print(f"\n  同一(股票,公告日)出现多条记录的数量: {len(multi):,}")
            if len(multi) > 0:
                print(f"  示例:")
                for (ts, ann), cnt in multi.head(3).items():
                    print(f"    {ts} {ann}: {cnt} 条")
                print(f"\n  {FAIL} 确认Bug: 存在 {len(multi):,} 个重复(股票,公告日)，未过滤 report_type!")
                return False
    else:
        print(f"  数据中无 report_type 列")

    print(f"\n  {PASS} 无重复数据")
    return True


def check_bug5_ann_date_no_delay():
    """Bug#5: 财报公告日当天即可用，缺少1天延迟"""
    print("\n" + "="*70)
    print("Bug#5: 财报公告日无延迟")
    print("="*70)

    import inspect
    from modules.data.tushare_to_qlib import TushareToQlibConverter

    source = inspect.getsource(TushareToQlibConverter.convert)

    # 检查 ann_date 是否直接用作 datetime
    uses_ann_date_directly = "ann_date" in source and "datetime" in source
    print(f"  ann_date 直接转为 datetime: {uses_ann_date_directly}")

    # 检查是否有 +1 天延迟
    has_delay = "timedelta" in source or "DateOffset" in source or "+ 1" in source
    print(f"  有延迟处理: {has_delay}")

    if uses_ann_date_directly and not has_delay:
        print(f"\n  {FAIL} 确认Bug: ann_date 当天即可用，盘后公告无法在当天使用!")
        return False
    else:
        print(f"\n  {PASS} 已有延迟处理")
        return True


def check_bug6_cost_scaled_by_position():
    """Bug#6: 交易成本被仓位比例缩小"""
    print("\n" + "="*70)
    print("Bug#6: 交易成本被仓位比例缩小")
    print("="*70)

    import inspect
    from modules.backtest.qlib_engine import QlibBacktestEngine

    source = inspect.getsource(QlibBacktestEngine.run)

    # 找到成本计算和仓位应用的顺序
    cost_line = None
    alloc_line = None
    for i, line in enumerate(source.split("\n")):
        if "total_cost_ratio" in line and "stock_ret" in line and "=" in line:
            cost_line = i
        if "stock_ret - total_cost_ratio" in line:
            cost_line = i
        if "alloc.stock_pct * stock_ret" in line:
            alloc_line = i

    print(f"  成本从 stock_ret 扣除的行: {cost_line}")
    print(f"  stock_ret 乘以 stock_pct 的行: {alloc_line}")

    if cost_line and alloc_line and cost_line < alloc_line:
        print(f"  顺序: 先从 stock_ret 扣成本 → 再乘 stock_pct")
        print(f"  问题: port_ret = stock_pct * (stock_ret - cost)")
        print(f"       正确: port_ret = stock_pct * stock_ret - cost")
        print(f"\n  {FAIL} 确认Bug: 当 stock_pct<1 时，成本被等比缩小!")

        # 量化影响
        # 假设 biweek 调仓，每年约26次，每次换手50%，stock_pct平均0.8
        avg_turnover = 0.5
        open_cost, close_cost = 0.0003, 0.0013
        annual_rebal = 26
        cost_per_rebal = avg_turnover * (open_cost + close_cost)
        correct_annual_cost = cost_per_rebal * annual_rebal
        actual_annual_cost = correct_annual_cost * 0.8  # 被缩小到80%
        diff = correct_annual_cost - actual_annual_cost
        print(f"  估算年化成本差异: {diff:.4%} (高估收益)")
        return False
    else:
        print(f"\n  {PASS} 成本计算正确")
        return True


def check_bug7_first_rebal_no_cost():
    """Bug#7: 首次建仓不扣买入成本"""
    print("\n" + "="*70)
    print("Bug#7: 首次建仓不扣买入成本")
    print("="*70)

    import inspect
    from modules.backtest.qlib_engine import QlibBacktestEngine
    source = inspect.getsource(QlibBacktestEngine.run)

    # 检查 "if is_rebal and prev_selected:" 条件
    has_prev_check = "prev_selected" in source and "is_rebal" in source
    # prev_selected 初始为 set()，第一次 is_rebal 时 prev_selected 为空
    init_empty = "prev_selected = set()" in source

    print(f"  调仓成本条件检查 prev_selected 非空: {has_prev_check}")
    print(f"  prev_selected 初始为空集: {init_empty}")

    if has_prev_check and init_empty:
        print(f"  首次调仓时 prev_selected 为空 → 条件不成立 → 不扣成本")
        open_cost = 0.0003
        print(f"  漏扣成本: {open_cost:.4%} (全部买入)")
        print(f"\n  {FAIL} 确认Bug: 首次建仓未扣买入成本")
        return False
    else:
        print(f"\n  {PASS} 首次建仓已扣成本")
        return True


def check_bug8_biweek_crossyear():
    """Bug#8: biweek 分组键跨年不连续"""
    print("\n" + "="*70)
    print("Bug#8: biweek 分组跨年问题")
    print("="*70)

    from core.selection import compute_rebalance_dates

    # 构造跨年交易日序列
    dates = pd.bdate_range("2023-12-01", "2024-01-31")
    dates_series = pd.Series(dates)

    result = compute_rebalance_dates(dates_series, freq="biweek")
    print(f"  2023-12 ~ 2024-01 的 biweek 调仓日:")
    for d in result:
        iso = d.isocalendar()
        print(f"    {d.strftime('%Y-%m-%d')} (ISO year={iso[0]}, week={iso[1]})")

    # 检查是否有间隔异常短的
    gaps = [(result[i+1] - result[i]).days for i in range(len(result)-1)]
    short_gaps = [g for g in gaps if g < 7]
    print(f"\n  调仓间隔(天): {gaps}")
    if short_gaps:
        print(f"  {FAIL} 确认Bug: 存在间隔<7天的调仓: {short_gaps}")
        return False
    else:
        print(f"  {PASS} 调仓间隔正常")
        return True


def check_bug9_delisted_stocks():
    """Bug#9: 退市股票被静默排除"""
    print("\n" + "="*70)
    print("Bug#9: 退市股票静默排除（幸存者偏差）")
    print("="*70)

    import inspect
    from modules.backtest.qlib_engine import QlibBacktestEngine
    source = inspect.getsource(QlibBacktestEngine.run)

    # 检查收益计算中对缺失股票的处理
    has_missing_check = "isin(selected)" in source
    has_delisted_handle = "delist" in source.lower() or "退市" in source
    has_nan_fallback = "isnan" in source or "np.isnan" in source

    print(f"  使用 isin(selected) 过滤: {has_missing_check}")
    print(f"  有退市专门处理: {has_delisted_handle}")
    print(f"  NaN 收益降为 0: {has_nan_fallback}")

    if has_missing_check and not has_delisted_handle:
        print(f"\n  {WARN} 退市股票无专门处理:")
        print(f"    - 退市日后无数据 → isin 匹配不到 → 自动排除")
        print(f"    - 等效于退市资金无损退出，实际通常亏损较大")
        return False
    else:
        print(f"\n  {PASS} 已有退市处理")
        return True


from tests.conftest import make_pytest_wrapper

test_bug1_qlib_negate_not_applied = make_pytest_wrapper(check_bug1_qlib_negate_not_applied)
test_bug2_st_lookahead = make_pytest_wrapper(check_bug2_st_lookahead)
test_bug3_position_lookahead = make_pytest_wrapper(check_bug3_position_lookahead)
test_bug4_report_type_not_filtered = make_pytest_wrapper(check_bug4_report_type_not_filtered)
test_bug5_ann_date_no_delay = make_pytest_wrapper(check_bug5_ann_date_no_delay)
test_bug6_cost_scaled_by_position = make_pytest_wrapper(check_bug6_cost_scaled_by_position)
test_bug7_first_rebal_no_cost = make_pytest_wrapper(check_bug7_first_rebal_no_cost)
test_bug8_biweek_crossyear = make_pytest_wrapper(check_bug8_biweek_crossyear)
test_bug9_delisted_stocks = make_pytest_wrapper(check_bug9_delisted_stocks)


def main():
    results = {}
    tests = [
        ("Bug#1: qlib因子negate未生效", check_bug1_qlib_negate_not_applied),
        ("Bug#2: ST名单前视偏差", check_bug2_st_lookahead),
        ("Bug#3: 仓位控制前视偏差", check_bug3_position_lookahead),
        ("Bug#4: report_type未过滤", check_bug4_report_type_not_filtered),
        ("Bug#5: 公告日无延迟", check_bug5_ann_date_no_delay),
        ("Bug#6: 交易成本被仓位缩小", check_bug6_cost_scaled_by_position),
        ("Bug#7: 首次建仓不扣成本", check_bug7_first_rebal_no_cost),
        ("Bug#8: biweek跨年问题", check_bug8_biweek_crossyear),
        ("Bug#9: 退市股票静默排除", check_bug9_delisted_stocks),
    ]

    for name, func in tests:
        try:
            results[name] = func()
        except Exception as e:
            print(f"\n  {WARN} 异常: {e}")
            results[name] = None

    # 汇总
    print("\n" + "="*70)
    print("汇总")
    print("="*70)
    for name, ok in results.items():
        if ok is True:
            status = PASS
        elif ok is False:
            status = FAIL
        else:
            status = WARN
        print(f"  {status} {name}")

    confirmed = sum(1 for v in results.values() if v is False)
    print(f"\n  确认Bug数: {confirmed}/{len(results)}")


if __name__ == "__main__":
    main()
