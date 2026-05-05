"""
策略对比框架
提供多策略回测结果的统一对比和组合功能
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class StrategyMetrics:
    """单策略指标"""
    name: str
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    volatility: float


def compare_strategies(results: Dict[str, "BacktestResult"]) -> pd.DataFrame:
    """
    输出各策略指标对比表

    Parameters
    ----------
    results : Dict[str, BacktestResult]
        策略名 -> BacktestResult 的映射

    Returns
    -------
    pd.DataFrame
        对比表格，index 为策略名
    """
    rows = []
    for name, result in results.items():
        if result.portfolio_value.empty:
            continue

        metrics = calculate_metrics(result)
        metrics.name = name
        rows.append(metrics)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame([{
        "策略名": r.name,
        "总收益率": f"{r.total_return:.2%}",
        "年化收益率": f"{r.annual_return:.2%}",
        "夏普比率": f"{r.sharpe_ratio:.4f}",
        "最大回撤": f"{r.max_drawdown:.2%}",
        "日胜率": f"{r.win_rate:.2%}",
        "波动率": f"{r.volatility:.2%}",
    } for r in rows])

    return df.set_index("策略名")


def comparison_metrics_raw(results: Dict[str, "BacktestResult"]) -> pd.DataFrame:
    """输出未格式化的多策略指标表，便于落盘和研究复核。"""
    rows = []
    for name, result in results.items():
        if result.portfolio_value.empty:
            continue
        metrics = calculate_metrics(result)
        rows.append(
            {
                "strategy": name,
                "total_return": metrics.total_return,
                "annual_return": metrics.annual_return,
                "sharpe_ratio": metrics.sharpe_ratio,
                "max_drawdown": metrics.max_drawdown,
                "win_rate": metrics.win_rate,
                "volatility": metrics.volatility,
            }
        )
    return pd.DataFrame(rows).set_index("strategy") if rows else pd.DataFrame()


def calculate_metrics(result: "BacktestResult") -> StrategyMetrics:
    """计算单个策略的各项指标"""
    if result.portfolio_value.empty:
        return StrategyMetrics(
            name="",
            total_return=0.0,
            annual_return=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            volatility=0.0,
        )

    daily_returns = result.daily_returns
    win_rate = (daily_returns > 0).mean() if not daily_returns.empty else 0.0
    volatility = daily_returns.std() * np.sqrt(252) if not daily_returns.empty else 0.0

    return StrategyMetrics(
        name="",
        total_return=result.total_return,
        annual_return=result.annual_return,
        sharpe_ratio=result.sharpe_ratio,
        max_drawdown=result.max_drawdown,
        win_rate=win_rate,
        volatility=volatility,
    )


def blend_strategies(
    results: Dict[str, "BacktestResult"],
    weights: Optional[Dict[str, float]] = None,
) -> "BacktestResult":
    """
    等权/加权组合多策略净值

    Parameters
    ----------
    results : Dict[str, BacktestResult]
        策略名 -> BacktestResult 的映射
    weights : Dict[str, float], optional
        策略权重，默认等权

    Returns
    -------
    BacktestResult
        组合后的回测结果
    """
    from modules.backtest.base import BacktestResult

    if not results:
        return BacktestResult(
            daily_returns=pd.Series(dtype=float),
            portfolio_value=pd.Series(dtype=float),
        )

    if weights is None:
        weights = {name: 1.0 / len(results) for name in results}

    all_dates = set()
    for result in results.values():
        all_dates.update(result.daily_returns.index)

    all_dates = sorted(all_dates)

    blended_returns = pd.Series(0.0, index=all_dates)

    for name, result in results.items():
        w = weights.get(name, 0.0)
        aligned = result.daily_returns.reindex(all_dates).fillna(0)
        blended_returns += w * aligned

    blended_returns = blended_returns.dropna()
    portfolio_value = (1 + blended_returns).cumprod()

    return BacktestResult(
        daily_returns=blended_returns,
        portfolio_value=portfolio_value,
        metadata={
            "blended_strategies": list(results.keys()),
            "weights": weights,
        },
    )


def print_comparison(results: Dict[str, "BacktestResult"]) -> None:
    """打印策略对比表格"""
    df = compare_strategies(results)
    if df.empty:
        print("无有效回测结果")
        return

    print("\n" + "=" * 80)
    print("策略表现对比")
    print("=" * 80)
    print(df.to_string())
    print()


def yearly_comparison(results: Dict[str, "BacktestResult"]) -> pd.DataFrame:
    """按年计算各策略收益率、夏普、最大回撤，返回多级列 DataFrame"""
    records = []
    for name, result in results.items():
        if result.daily_returns.empty:
            continue
        rets = result.daily_returns.copy()
        rets.index = pd.to_datetime(rets.index)
        for year, grp in rets.groupby(rets.index.year):
            yr_return = (1 + grp).prod() - 1
            yr_sharpe = grp.mean() / grp.std() * np.sqrt(252) if grp.std() > 0 else 0.0
            nav = (1 + grp).cumprod()
            yr_mdd = float((nav / nav.cummax() - 1).min())
            records.append({
                "策略": name,
                "年份": int(year),
                "收益率": yr_return,
                "夏普": yr_sharpe,
                "最大回撤": yr_mdd,
            })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    return df.set_index(["年份", "策略"]).sort_index()


def print_yearly_comparison(results: Dict[str, "BacktestResult"]) -> None:
    """打印按年对比表格"""
    df = yearly_comparison(results)
    if df.empty:
        print("无有效回测结果")
        return

    # 转成 pivot 格式：行=年份，列=策略
    names = list(results.keys())
    years = sorted(df.index.get_level_values("年份").unique())

    print("\n" + "=" * 90)
    print("按年收益率对比")
    print("=" * 90)

    # 表头
    header = f"{'年份':>6}"
    for n in names:
        header += f"  {n:>12}"
    print(header)
    print("-" * 90)

    for year in years:
        row = f"{year:>6}"
        for n in names:
            try:
                val = df.loc[(year, n), "收益率"]
                row += f"  {val:>+11.2%}"
            except KeyError:
                row += f"  {'N/A':>12}"
        print(row)

    print()
    print("=" * 90)
    print("按年夏普比率对比")
    print("=" * 90)
    print(header)
    print("-" * 90)

    for year in years:
        row = f"{year:>6}"
        for n in names:
            try:
                val = df.loc[(year, n), "夏普"]
                row += f"  {val:>11.2f}"
            except KeyError:
                row += f"  {'N/A':>12}"
        print(row)

    print()
    print("=" * 90)
    print("按年最大回撤对比")
    print("=" * 90)
    print(header)
    print("-" * 90)

    for year in years:
        row = f"{year:>6}"
        for n in names:
            try:
                val = df.loc[(year, n), "最大回撤"]
                row += f"  {val:>11.2%}"
            except KeyError:
                row += f"  {'N/A':>12}"
        print(row)

    print()


def plot_yearly_comparison(
    results: Dict[str, "BacktestResult"],
    output_path: Path = None,
):
    """按年收益率柱状图"""
    plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    df = yearly_comparison(results)
    if df.empty:
        return

    names = list(results.keys())
    years = sorted(df.index.get_level_values("年份").unique())

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # --- 子图1: 按年收益率柱状图 ---
    ax1 = axes[0]
    x = np.arange(len(years))
    width = 0.8 / len(names)
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(names), 3)))

    for i, name in enumerate(names):
        vals = []
        for year in years:
            try:
                vals.append(df.loc[(year, name), "收益率"])
            except KeyError:
                vals.append(0)
        bars = ax1.bar(x + i * width, [v * 100 for v in vals], width,
                       label=name, color=colors[i])
        for bar, v in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{v:+.1%}", ha="center", va="bottom", fontsize=7)

    ax1.set_xticks(x + width * (len(names) - 1) / 2)
    ax1.set_xticklabels([str(y) for y in years])
    ax1.set_ylabel("收益率 (%)")
    ax1.set_title("按年收益率对比")
    ax1.legend(fontsize=9)
    ax1.axhline(y=0, color="gray", linewidth=0.8)
    ax1.grid(axis="y", alpha=0.3)

    # --- 子图2: 按年最大回撤柱状图 ---
    ax2 = axes[1]
    for i, name in enumerate(names):
        vals = []
        for year in years:
            try:
                vals.append(df.loc[(year, name), "最大回撤"])
            except KeyError:
                vals.append(0)
        bars = ax2.bar(x + i * width, [v * 100 for v in vals], width,
                       label=name, color=colors[i])
        for bar, v in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{v:.1%}", ha="center", va="top", fontsize=7)

    ax2.set_xticks(x + width * (len(names) - 1) / 2)
    ax2.set_xticklabels([str(y) for y in years])
    ax2.set_ylabel("最大回撤 (%)")
    ax2.set_title("按年最大回撤对比")
    ax2.legend(fontsize=9)
    ax2.axhline(y=0, color="gray", linewidth=0.8)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if output_path is None:
        output_path = Path("results") / "yearly_compare.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] 按年对比图已保存: {output_path}")


def load_benchmark(start_date: str, end_date: str) -> Optional["BacktestResult"]:
    """加载沪深300基准，返回 BacktestResult 格式"""
    from modules.backtest.base import BacktestResult
    from utils.benchmark_comparison_akshare import get_benchmark_data_akshare

    hs300 = get_benchmark_data_akshare("sh000300", start_date, end_date)
    if hs300 is None or hs300.empty:
        print("[WARN] 无法加载沪深300基准数据")
        return None

    returns = hs300["close"].pct_change().fillna(0)
    nav = (1 + returns).cumprod()
    return BacktestResult(
        daily_returns=returns,
        portfolio_value=nav,
        metadata={"name": "沪深300"},
    )


def run_compare(
    strategy_names: list = None,
    engine: str = "qlib",
    benchmark: bool = True,
) -> Dict[str, "BacktestResult"]:
    """跑所有/指定策略，返回 {name: BacktestResult}"""
    from core.strategy import Strategy, is_composite_strategy
    from modules.backtest.composite import run_strategy_backtest
    from config.config import CONFIG

    # 1. 确定策略列表
    if strategy_names:
        names = strategy_names
    else:
        names = Strategy.list_available()

    if not names:
        print("[WARN] 没有可用策略")
        return {}

    print(f"\n{'='*60}")
    print(f"  多策略对比 - 引擎: {engine}")
    print(f"  策略: {', '.join(names)}")
    print(f"{'='*60}\n")

    # 2. 逐个跑回测
    if engine not in {"qlib", "pybroker"}:
        print(f"[ERROR] 不支持的引擎: {engine}")
        return {}

    results: Dict[str, "BacktestResult"] = {}
    strategy_objects = {}
    for name in names:
        print(f"[INFO] 运行策略: {name} ...")
        try:
            strategy = Strategy.load(name)
            strategy.validate_data_requirements()
            strategy_objects[name] = strategy
            result = run_strategy_backtest(strategy=strategy, engine=engine)
            if not result.portfolio_value.empty:
                results[name] = result
                print(f"  [OK] {name}: 总收益 {result.total_return:+.2%}")
            else:
                print(f"  [WARN] {name}: 回测结果为空")
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")

    if not results:
        print("[WARN] 没有有效的回测结果")
        return {}

    # 3. 加载沪深300基准
    benchmark_result = None
    if benchmark:
        # 从回测结果中推断日期范围
        indices = [r.portfolio_value.index for r in results.values() if not r.portfolio_value.empty]
        if indices:
            start_date = min(idx.min() for idx in indices).strftime("%Y-%m-%d")
            end_date = max(idx.max() for idx in indices).strftime("%Y-%m-%d")
            print(f"\n[INFO] 加载沪深300基准 ({start_date} ~ {end_date}) ...")
            benchmark_result = load_benchmark(start_date, end_date)
            if benchmark_result is not None:
                results["沪深300"] = benchmark_result
                print(f"  [OK] 沪深300: 总收益 {benchmark_result.total_return:+.2%}")

    # 4. 输出对比表格
    print_comparison(results)

    # 4.5 按年对比
    print_yearly_comparison(results)

    # 5. 画图
    output_dir = Path(CONFIG.get("results_path", "results")).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    strategy_results = {k: v for k, v in results.items() if k != "沪深300"}
    universe_values = set()
    for name in strategy_results:
        strategy = strategy_objects.get(name)
        if strategy is None:
            continue
        if is_composite_strategy(strategy):
            universe_values.add(strategy.effective_universe())
        else:
            universe_values.add(getattr(strategy, "universe", "all"))
    if universe_values == {"csi300"}:
        scope_tag = "historical_csi300"
    elif universe_values == {"csi800"}:
        scope_tag = "historical_csi800"
    elif universe_values == {"all"}:
        scope_tag = "all_market"
    else:
        scope_tag = "mixed_universe"

    output_path = output_dir / f"multi_strategy_compare_{scope_tag}_{timestamp}.png"

    if benchmark_result is not None:
        plot_multi_strategy(strategy_results, benchmark_result, output_path)
    else:
        plot_multi_strategy(strategy_results, None, output_path)

    # 5.5 按年对比图
    yearly_path = output_dir / f"yearly_compare_{scope_tag}_{timestamp}.png"
    plot_yearly_comparison(results, yearly_path)

    # 5.6 保存对比表
    summary_csv = output_dir / f"strategy_compare_{scope_tag}_{timestamp}.csv"
    yearly_csv = output_dir / f"strategy_yearly_compare_{scope_tag}_{timestamp}.csv"
    raw_summary = comparison_metrics_raw(results)
    if not raw_summary.empty:
        raw_summary.to_csv(summary_csv)
        print(f"[OK] 策略对比表已保存: {summary_csv}")
    yearly_df = yearly_comparison(results)
    if not yearly_df.empty:
        yearly_df.to_csv(yearly_csv)
        print(f"[OK] 按年对比表已保存: {yearly_csv}")

    # 6. 返回结果字典
    return results


def plot_multi_strategy(
    results: Dict[str, "BacktestResult"],
    benchmark: "BacktestResult" = None,
    output_path: Path = None,
):
    """多策略 + 基准净值对比图（3子图）"""
    plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    colors = plt.cm.Set2(np.linspace(0, 1, max(len(results), 3)))
    benchmark_color = "#333333"

    # --- 子图1: 归一化净值叠加 ---
    ax1 = axes[0]
    for i, (name, result) in enumerate(results.items()):
        nav = result.portfolio_value / result.portfolio_value.iloc[0]
        ax1.plot(nav.index, nav.values, label=name, color=colors[i], linewidth=1.5)

    if benchmark is not None and not benchmark.portfolio_value.empty:
        bnav = benchmark.portfolio_value / benchmark.portfolio_value.iloc[0]
        ax1.plot(bnav.index, bnav.values, label="沪深300",
                 color=benchmark_color, linewidth=1.5, linestyle="--")

    ax1.set_ylabel("归一化净值")
    ax1.set_title("多策略净值对比")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # --- 子图2: 超额收益（各策略 vs 沪深300）---
    ax2 = axes[1]
    if benchmark is not None and not benchmark.portfolio_value.empty:
        bnav = benchmark.portfolio_value / benchmark.portfolio_value.iloc[0]
        for i, (name, result) in enumerate(results.items()):
            nav = result.portfolio_value / result.portfolio_value.iloc[0]
            # 对齐日期
            common = nav.index.intersection(bnav.index)
            if len(common) > 0:
                excess = nav.reindex(common) - bnav.reindex(common)
                ax2.plot(common, excess.values, label=name, color=colors[i], linewidth=1.5)
        ax2.axhline(y=0, color="gray", linestyle="-", linewidth=0.8)
    else:
        ax2.text(0.5, 0.5, "无基准数据", transform=ax2.transAxes,
                 ha="center", va="center", fontsize=14, color="gray")

    ax2.set_ylabel("超额收益")
    ax2.set_title("相对沪深300超额收益")
    handles, labels = ax2.get_legend_handles_labels()
    if handles:
        ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.3)

    # --- 子图3: 回撤对比 ---
    ax3 = axes[2]
    for i, (name, result) in enumerate(results.items()):
        nav = result.portfolio_value / result.portfolio_value.iloc[0]
        rolling_max = nav.cummax()
        drawdown = nav / rolling_max - 1
        ax3.fill_between(drawdown.index, drawdown.values, 0,
                         alpha=0.3, color=colors[i], label=name)
        ax3.plot(drawdown.index, drawdown.values, color=colors[i], linewidth=0.8)

    if benchmark is not None and not benchmark.portfolio_value.empty:
        bnav = benchmark.portfolio_value / benchmark.portfolio_value.iloc[0]
        rolling_max = bnav.cummax()
        drawdown = bnav / rolling_max - 1
        ax3.plot(drawdown.index, drawdown.values, label="沪深300",
                 color=benchmark_color, linewidth=1.2, linestyle="--")

    ax3.set_ylabel("回撤")
    ax3.set_title("回撤对比")
    ax3.legend(loc="lower left", fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path is None:
        output_path = Path("results") / "multi_strategy_compare.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[OK] 对比图已保存: {output_path}")
