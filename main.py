#!/usr/bin/env python3
"""
量化交易系统主入口

Usage:
    python main.py update                          # 更新数据
    python main.py select                          # 生成选股列表（默认策略）
    python main.py select -s value                 # 生成选股列表（指定策略）
    python main.py backtest --list                 # 列出可用策略
    python main.py backtest -e qlib                # Qlib 回测（默认策略）
    python main.py backtest -s value -e qlib       # Qlib 回测（指定策略）
    python main.py backtest -s momentum -e pybroker  # PyBroker 回测（指定策略）
    python main.py plot                            # 绘制净值曲线
    python main.py plot --benchmark                # 绘制基准对比图
    python main.py analyze                         # 因子分析（全类别）
    python main.py analyze -c technical -n 20      # 分析技术类因子 Top20
    python main.py analyze --report                # 生成因子完整报告
    python main.py compare                         # 对比所有策略 + 沪深300
    python main.py compare -s value,momentum       # 只对比指定策略
    python main.py compare -e pybroker             # 指定引擎（默认 qlib）
    python main.py compare --no-benchmark          # 不画沪深300
    python main.py run                             # 全流程（更新+回测）
    python main.py run -s value                    # 全流程（指定策略）
"""
import warnings
warnings.filterwarnings("ignore", message=".*OpenSSL.*")

import argparse
import sys
from pathlib import Path


def _load_strategy(name):
    """加载策略对象，未指定时优先加载 default 策略"""
    from core.strategy import Strategy
    strategy_name = name or "default"
    try:
        return Strategy.load(strategy_name)
    except FileNotFoundError:
        if name is None:
            return None
        raise


def _load_config(config_file: str):
    """加载策略配置文件并更新全局配置"""
    from config.config import ConfigManager, config as global_config
    if config_file and config_file != "strategy.yaml":
        new_config = ConfigManager(strategy_file=config_file)
        # 更新全局配置的内部数据
        global_config._data.update(new_config.get_config()._data)
        return new_config.get_config()
    return global_config


def cmd_backtest(args):
    """运行回测"""
    # 加载策略配置
    config_file = getattr(args, "config", "strategy.yaml")
    if config_file != "strategy.yaml":
        _load_config(config_file)

    # --list: 列出可用策略
    if getattr(args, "list_strategies", False):
        from core.strategy import Strategy
        strategies = Strategy.list_available()
        if not strategies:
            print("[WARN] 没有可用策略 (config/strategies/ 目录为空)")
        else:
            print(f"\n  可用策略 ({len(strategies)}):")
            for layer, names in Strategy.list_grouped().items():
                print(f"\n    [{layer}] ({len(names)})")
                for name in names:
                    meta = Strategy.load_metadata(name)
                    print(f"      {name:<48} - {meta['description']}")
        return

    strategy = _load_strategy(getattr(args, "strategy", None))
    strategy_label = strategy.name if strategy else "default"
    if strategy is not None:
        strategy.validate_data_requirements()

    print(f"\n{'='*60}")
    print(f"  运行回测 - 引擎: {args.engine}, 策略: {strategy_label}, 配置: {config_file}")
    print(f"{'='*60}\n")

    if args.engine == "qlib":
        from config.config import CONFIG
        from modules.backtest.composite import run_strategy_backtest
        from modules.backtest.qlib_engine import QlibBacktestEngine

        result = (
            run_strategy_backtest(strategy=strategy, engine="qlib")
            if strategy is not None
            else QlibBacktestEngine().run(strategy=None)
        )
        initial_capital = CONFIG.get("initial_capital", 500000)
        result.print_summary(initial_capital)
        if result.metadata.get("results_file"):
            print(f"\n  [OK] 结果已保存: {result.metadata['results_file']}")
    elif args.engine == "pybroker":
        from config.config import CONFIG
        from modules.backtest.composite import run_strategy_backtest
        from modules.backtest.pybroker_engine import PyBrokerBacktestEngine

        result = (
            run_strategy_backtest(strategy=strategy, engine="pybroker")
            if strategy is not None
            else PyBrokerBacktestEngine().run(strategy=None)
        )
        initial_capital = CONFIG.get("trading.capital.initial", 500000)
        result.print_summary(initial_capital)
        if result.metadata.get("results_file"):
            print(f"\n  [OK] 结果已保存: {result.metadata['results_file']}")
    else:
        print(f"[ERROR] 不支持的引擎: {args.engine}")
        sys.exit(1)


def cmd_update(args):
    """更新数据"""
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print(f"\n{'='*60}")
    print("  更新数据")
    print(f"{'='*60}\n")

    from modules.data.updater import DataUpdater

    updater = DataUpdater()
    result = updater.update_daily()

    if result["success"]:
        print(f"[OK] {result['message']}")
    else:
        print(f"[WARN] {result['message']}")


def cmd_select(args):
    """生成选股列表"""
    # 加载策略配置
    config_file = getattr(args, "config", "strategy.yaml")
    if config_file != "strategy.yaml":
        _load_config(config_file)

    strategy = _load_strategy(getattr(args, "strategy", None))
    strategy_label = strategy.name if strategy else "default"
    if strategy is not None:
        strategy.validate_data_requirements()

    print(f"\n{'='*60}")
    print(f"  生成选股列表 - 策略: {strategy_label}, 配置: {config_file}")
    print(f"{'='*60}\n")

    if strategy:
        strategy.generate_selections(force=True)
    else:
        from core.selection import generate_selections
        generate_selections(force=True)


def cmd_plot(args):
    """绘制图表"""
    if args.benchmark:
        print(f"\n{'='*60}")
        print("  基准对比图")
        print(f"{'='*60}\n")

        from utils.benchmark_comparison_akshare import main as benchmark_main
        benchmark_main()
    else:
        print(f"\n{'='*60}")
        print("  回测净值对比图")
        print(f"{'='*60}\n")

        from utils.compare_plot import plot_comparison
        plot_comparison()


def cmd_analyze(args):
    """因子分析"""
    print(f"\n{'='*60}")
    print("  因子分析")
    print(f"{'='*60}\n")

    from scripts.factor_mining import FactorMining

    miner = FactorMining()

    if args.report:
        miner.generate_factor_report()
    elif args.category == "all":
        for category in ["technical", "fundamental", "risk", "quality", "sentiment"]:
            miner.mine_factors(category, args.start_date, args.end_date, args.top_n)
    else:
        miner.mine_factors(args.category, args.start_date, args.end_date, args.top_n)

    print(f"\n{'='*60}")
    print("  因子分析完成")
    print(f"{'='*60}")


def cmd_run(args):
    """每日运行（数据更新 + 回测验证）"""
    strategy = _load_strategy(getattr(args, "strategy", None))
    strategy_label = strategy.name if strategy else "default"

    print(f"\n{'='*60}")
    print(f"  每日运行 - 策略: {strategy_label}")
    print(f"{'='*60}\n")

    # 1. 更新数据
    from modules.data.updater import DataUpdater
    updater = DataUpdater()
    updater.update_daily()

    # 2. 快速回测验证
    if strategy is not None:
        strategy.validate_data_requirements()
    from config.config import CONFIG
    from modules.backtest.composite import run_strategy_backtest
    from modules.backtest.qlib_engine import QlibBacktestEngine

    result = (
        run_strategy_backtest(strategy=strategy, engine="qlib")
        if strategy is not None
        else QlibBacktestEngine().run(strategy=None)
    )
    initial_capital = CONFIG.get("initial_capital", 500000)
    result.print_summary(initial_capital)

    print("\n[OK] 每日运行完成")


def cmd_compare(args):
    """多策略对比"""
    from modules.backtest.compare import run_compare
    names = args.strategy.split(",") if args.strategy else None
    run_compare(
        strategy_names=names,
        engine=args.engine,
        benchmark=not args.no_benchmark,
    )


def cmd_report(args):
    """生成 QuantStats 回测报告"""
    from utils.quantstats_report import generate_report, print_summary
    
    print(f"\n{'='*60}")
    print("  QuantStats 回测报告")
    print(f"{'='*60}\n")
    
    if not args.result:
        print("[ERROR] 请提供 --result <回测结果CSV文件>")
        print("   示例: python main.py report -r results/backtest_xxx.csv")
        return
    
    import pandas as pd
    df = pd.read_csv(args.result, index_col=0, parse_dates=True)
    if 'return' in df.columns:
        returns = df['return']
    elif 'daily_return' in df.columns:
        returns = df['daily_return']
    else:
        returns = df.iloc[:, 0]
    
    if hasattr(returns, 'squeeze'):
        returns = returns.squeeze()
    
    benchmark = args.benchmark if args.benchmark else None
    
    if args.summary:
        print_summary(returns=returns, benchmark=benchmark)
    else:
        generate_report(
            returns=returns,
            benchmark=benchmark,
            output_path=args.output,
        )


MENU_ITEMS = [
    ("1", "update",    "更新数据",                   "下载 Tushare → 转换 → 重新选股"),
    ("2", "select",    "生成选股列表",               "基于最新数据运行多因子选股"),
    ("3", "backtest",  "运行回测 (Qlib)",            "使用 Qlib 引擎回测策略表现"),
    ("4", "backtest",  "运行回测 (PyBroker)",        "使用 PyBroker 引擎回测策略表现"),
    ("5", "plot",      "绘制净值对比图",             "Qlib vs PyBroker 净值曲线"),
    ("6", "plot",      "绘制基准对比图",             "策略 vs 沪深300 / 中证500"),
    ("7", "analyze",   "因子分析 (全类别)",          "IC/IR 分析所有因子类别"),
    ("8", "analyze",   "因子分析 (指定类别)",        "选择 technical/fundamental/..."),
    ("9", "run",       "每日全流程",                 "更新数据 + Qlib 回测验证"),
    ("10", "compare",  "多策略对比",                 "跑所有策略并对比沪深300"),
    ("11", "report",   "生成 QuantStats 报告",       "HTML 交互式回测分析报告"),
    ("0", "quit",      "退出",                       ""),
]


def interactive_menu(parser, subparsers, commands):
    """交互式菜单"""
    print(f"\n{'='*60}")
    print("  A股多因子量化选股系统")
    print(f"{'='*60}\n")

    for key, _, label, desc in MENU_ITEMS:
        if key == "0":
            print(f"  {'─'*50}")
        line = f"  [{key}] {label}"
        if desc:
            line += f"  - {desc}"
        print(line)

    print()

    try:
        choice = input("请选择 [0-10]: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return

    if choice == "0" or not choice:
        return

    # 构造 args 并调用对应命令
    if choice == "1":
        sys.argv = ["main.py", "update"]
    elif choice == "2":
        strategy_name = _ask_strategy()
        if not strategy_name:
            return
        sys.argv = ["main.py", "select", "-s", strategy_name]
    elif choice == "3":
        strategy_name = _ask_strategy()
        if not strategy_name:
            return
        sys.argv = ["main.py", "backtest", "-e", "qlib", "-s", strategy_name]
    elif choice == "4":
        strategy_name = _ask_strategy()
        if not strategy_name:
            return
        sys.argv = ["main.py", "backtest", "-e", "pybroker", "-s", strategy_name]
    elif choice == "5":
        sys.argv = ["main.py", "plot"]
    elif choice == "6":
        sys.argv = ["main.py", "plot", "--benchmark"]
    elif choice == "7":
        sys.argv = ["main.py", "analyze"]
    elif choice == "8":
        category = _ask_category()
        if not category:
            return
        top_n = input("显示 Top N 因子 [10]: ").strip() or "10"
        sys.argv = ["main.py", "analyze", "-c", category, "-n", top_n]
    elif choice == "9":
        sys.argv = ["main.py", "run"]
    elif choice == "10":
        sys.argv = ["main.py", "compare"]
    elif choice == "11":
        result_file = input("回测结果CSV路径 [留空则列出results目录]: ").strip()
        if result_file:
            sys.argv = ["main.py", "report", "-r", result_file]
        else:
            import os
            results_dir = "results"
            if os.path.exists(results_dir):
                files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]
                if files:
                    print(f"\n  可用回测结果:")
                    for i, f in enumerate(files, 1):
                        print(f"  [{i}] {f}")
                    print()
                    idx = input(f"选择文件 [1]: ").strip() or "1"
                    try:
                        result_file = files[int(idx) - 1]
                        sys.argv = ["main.py", "report", "-r", f"{results_dir}/{result_file}"]
                    except (ValueError, IndexError):
                        print(f"[ERROR] 无效选择")
                        return
                else:
                    print("[WARN] results 目录为空，请先运行回测")
                    return
            else:
                print("[WARN] results 目录不存在，请先运行回测")
                return
    else:
        print(f"[ERROR] 无效选择: {choice}")
        return

    args = parser.parse_args()
    handler = commands.get(args.command)
    if handler:
        handler(args)


def _ask_strategy():
    """交互选择策略"""
    from core.strategy import Strategy
    strategies = Strategy.list_available()
    if not strategies:
        print("[WARN] 没有可用策略")
        return None

    print(f"\n  可用策略:")
    for i, name in enumerate(strategies, 1):
        meta = Strategy.load_metadata(name)
        print(f"  [{i}] {name:<16} - {meta['description']}")
    print()

    try:
        idx = input(f"选择策略 [1]: ").strip() or "1"
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    try:
        return strategies[int(idx) - 1]
    except (ValueError, IndexError):
        print(f"[ERROR] 无效选择: {idx}")
        return None


def _ask_category():
    """交互选择因子类别"""
    categories = ["technical", "fundamental", "risk", "quality", "sentiment"]
    print()
    for i, cat in enumerate(categories, 1):
        print(f"  [{i}] {cat}")
    print()
    try:
        idx = input("选择因子类别 [1-5]: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    try:
        return categories[int(idx) - 1]
    except (ValueError, IndexError):
        print(f"[ERROR] 无效选择: {idx}")
        return None


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="量化交易系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
命令说明:
  update     更新 Tushare 数据 → 转换 → 重新选股
  select     单独运行选股（不下载数据）
  backtest   运行回测（qlib / pybroker）
  plot       绘制净值曲线或基准对比图
  analyze    因子 IC/IR 分析
  compare    多策略对比（自动跑回测 + 沪深300基准）
  run        全流程（更新 + 回测）
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # 更新命令
    subparsers.add_parser("update", help="更新数据")

    # 选股命令
    select_parser = subparsers.add_parser("select", help="生成选股列表")
    select_parser.add_argument("--strategy", "-s", default=None,
                               help="策略名称 (默认: 使用全局配置)")
    select_parser.add_argument("--config", "-c", default="strategy.yaml",
                               help="策略配置文件 (默认: strategy.yaml)")

    # 回测命令
    backtest_parser = subparsers.add_parser("backtest", help="运行回测")
    backtest_parser.add_argument("--engine", "-e", choices=["qlib", "pybroker"],
                                default="qlib", help="回测引擎 (默认: qlib)")
    backtest_parser.add_argument("--strategy", "-s", default=None,
                                help="策略名称 (默认: 使用全局配置)")
    backtest_parser.add_argument("--config", "-c", default="strategy.yaml",
                                help="策略配置文件 (默认: strategy.yaml)")
    backtest_parser.add_argument("--list", dest="list_strategies", action="store_true",
                                help="列出所有可用策略")

    # 绘图命令
    plot_parser = subparsers.add_parser("plot", help="绘制图表")
    plot_parser.add_argument("--benchmark", "-b", action="store_true",
                             help="绘制基准对比图（vs 沪深300/中证500）")

    # 因子分析命令
    analyze_parser = subparsers.add_parser("analyze", help="因子分析")
    analyze_parser.add_argument("--category", "-c", default="all",
                                choices=["technical", "fundamental", "risk",
                                         "quality", "sentiment", "all"],
                                help="因子类别 (默认: all)")
    analyze_parser.add_argument("--start_date", "-s", default="2019-01-01",
                                help="开始日期")
    analyze_parser.add_argument("--end_date", "-e", default="2026-02-26",
                                help="结束日期")
    analyze_parser.add_argument("--top_n", "-n", type=int, default=10,
                                help="显示 Top N 因子 (默认: 10)")
    analyze_parser.add_argument("--report", "-r", action="store_true",
                                help="生成完整因子报告")

    # 多策略对比命令
    compare_parser = subparsers.add_parser("compare", help="多策略对比")
    compare_parser.add_argument("--strategy", "-s", default=None,
                                help="策略名称，逗号分隔 (默认: 全部)")
    compare_parser.add_argument("--engine", "-e", choices=["qlib", "pybroker"],
                                default="qlib", help="回测引擎 (默认: qlib)")
    compare_parser.add_argument("--no-benchmark", action="store_true",
                                help="不包含沪深300基准")

    # 每日运行命令
    run_parser = subparsers.add_parser("run", help="每日运行（更新+回测）")
    run_parser.add_argument("--strategy", "-s", default=None,
                            help="策略名称 (默认: 使用全局配置)")

    # QuantStats 报告命令
    qs_parser = subparsers.add_parser("report", help="生成 QuantStats 回测报告")
    qs_parser.add_argument("--result", "-r", default=None,
                          help="回测结果 CSV 文件路径")
    qs_parser.add_argument("--strategy", "-s", default=None,
                          help="策略名称（运行回测后生成报告）")
    qs_parser.add_argument("--engine", "-e", choices=["qlib", "pybroker"],
                          default="qlib", help="回测引擎 (默认: qlib)")
    qs_parser.add_argument("--output", "-o", default=None,
                          help="输出 HTML 路径")
    qs_parser.add_argument("--benchmark", "-b", default="^000300",
                          help="基准代码 (默认: ^000300 沪深300)")
    qs_parser.add_argument("--summary", action="store_true",
                          help="仅打印摘要，不生成 HTML")

    args = parser.parse_args()

    commands = {
        "update": cmd_update,
        "select": cmd_select,
        "backtest": cmd_backtest,
        "plot": cmd_plot,
        "analyze": cmd_analyze,
        "compare": cmd_compare,
        "run": cmd_run,
        "report": cmd_report,
    }

    handler = commands.get(args.command)
    if handler:
        handler(args)
    else:
        interactive_menu(parser, subparsers, commands)


if __name__ == "__main__":
    main()
