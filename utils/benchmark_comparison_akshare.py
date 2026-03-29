"""
基准指数对比分析 (使用 AKShare 获取准确数据)
对比策略与沪深300、中证500的表现
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from config.config import CONFIG
import akshare as ak


def get_benchmark_data_akshare(symbol: str, start_date: str, end_date: str):
    """使用 AKShare 获取准确的基准数据"""
    try:
        df = ak.stock_zh_index_daily(symbol=symbol)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()

        # 筛选日期范围
        df = df.loc[start_date:end_date]
        return df
    except Exception as e:
        print(f"[WARN] 获取 {symbol} 失败: {e}")
        return None


def load_strategy_results():
    """加载策略回测结果"""
    results_dir = Path("results")
    initial = 500000

    # 优先使用最新的 PyBroker 结果
    pybroker_files = sorted(results_dir.glob("pybroker_*.csv"), reverse=True)
    if pybroker_files:
        df = pd.read_csv(pybroker_files[0], parse_dates=["date"])
        df = df.set_index("date").sort_index()
        if "equity" in df.columns:
            return df["equity"], "PyBroker", df.index[0], df.index[-1]

    # 备用 Qlib 结果
    qlib_files = sorted(results_dir.glob("backtest_*.csv"), reverse=True)
    if qlib_files:
        df = pd.read_csv(qlib_files[0], parse_dates=["date"])
        df = df.set_index("date").sort_index()
        if "return" in df.columns:
            equity = initial * (1 + df["return"]).cumprod()
            return equity, "Qlib", df.index[0], df.index[-1]

    raise FileNotFoundError("没有找到回测结果")


def calculate_metrics(returns, ann_days=252):
    """计算回测指标"""
    total_ret = (1 + returns).prod() - 1
    days = len(returns)
    ann_ret = (1 + total_ret) ** (ann_days / days) - 1

    cumret = (1 + returns).cumprod()
    max_dd = (cumret / cumret.cummax() - 1).min()

    sharpe = returns.mean() / returns.std() * np.sqrt(ann_days) if returns.std() > 0 else 0
    win_rate = (returns > 0).mean()

    return {
        "total_return": total_ret,
        "annual_return": ann_ret,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "win_rate": win_rate,
    }


def main():
    """主函数"""
    print("=" * 70)
    print("  策略 vs 基准指数对比分析 (AKShare 准确数据)")
    print("=" * 70)

    # 加载策略结果
    print("\n[1/3] 加载策略回测结果...")
    strategy_equity, engine, strat_start, strat_end = load_strategy_results()
    print(f"  使用 {engine} 回测结果")
    print(f"  日期范围: {strat_start.date()} ~ {strat_end.date()}")

    # 加载基准数据 (使用 AKShare)
    print("\n[2/3] 加载基准指数数据 (从 AKShare)...")

    start_str = strat_start.strftime("%Y-%m-%d")
    end_str = strat_end.strftime("%Y-%m-%d")

    hs300_data = get_benchmark_data_akshare("sh000300", start_str, end_str)  # 沪深300
    csi500_data = get_benchmark_data_akshare("sz399500", start_str, end_str)  # 中证500

    if hs300_data is not None:
        print(f"  沪深300: {hs300_data.index[0].date()} ~ {hs300_data.index[-1].date()}")
        print(f"           起始: {hs300_data.iloc[0]['close']:.2f}, 结束: {hs300_data.iloc[-1]['close']:.2f}")

    if csi500_data is not None:
        print(f"  中证500: {csi500_data.index[0].date()} ~ {csi500_data.index[-1].date()}")
        print(f"           起始: {csi500_data.iloc[0]['close']:.2f}, 结束: {csi500_data.iloc[-1]['close']:.2f}")

    # 计算收益率
    print("\n[3/3] 计算对比指标...")

    # 策略收益率
    strategy_returns = strategy_equity.pct_change().fillna(0)
    strategy_metrics = calculate_metrics(strategy_returns)

    # 基准收益率
    hs300_metrics = None
    csi500_metrics = None
    hs300_returns = None
    csi500_returns = None

    if hs300_data is not None:
        hs300_returns = hs300_data["close"].pct_change().fillna(0)
        hs300_metrics = calculate_metrics(hs300_returns)

    if csi500_data is not None:
        csi500_returns = csi500_data["close"].pct_change().fillna(0)
        csi500_metrics = calculate_metrics(csi500_returns)

    # 打印对比结果
    print("\n" + "=" * 70)
    print("  收益率对比 (准确数据)")
    print("=" * 70)

    print(f"\n{'指标':<15} {'策略':>12} {'沪深300':>12} {'中证500':>12}")
    print("-" * 70)

    for metric, name in [
        ("total_return", "总收益率"),
        ("annual_return", "年化收益率"),
        ("max_drawdown", "最大回撤"),
        ("sharpe", "夏普比率"),
        ("win_rate", "日胜率"),
    ]:
        strat_val = strategy_metrics[metric]
        hs300_val = hs300_metrics[metric] if hs300_metrics else None
        csi500_val = csi500_metrics[metric] if csi500_metrics else None

        if metric in ["total_return", "annual_return", "max_drawdown"]:
            fmt = ">11.2%"
        else:
            fmt = ">11.4f"

        print(f"{name:<15} {strat_val:{fmt}}", end="")
        if hs300_val is not None:
            excess_hs = strat_val - hs300_val
            print(f" {hs300_val:{fmt}} ({excess_hs:>+9.2%})", end="")
        if csi500_val is not None:
            excess_csi = strat_val - csi500_val
            print(f" {csi500_val:{fmt}} ({excess_csi:>+9.2%})", end="")
        print()

    # 超额收益分析
    print("\n" + "=" * 70)
    print("  超额收益分析")
    print("=" * 70)

    if hs300_returns is not None:
        common_idx = strategy_returns.index.intersection(hs300_returns.index)
        if len(common_idx) > 0:
            strat_aligned = strategy_returns.loc[common_idx]
            hs300_aligned = hs300_returns.loc[common_idx]
            excess_hs = strat_aligned - hs300_aligned

            cum_excess_hs = (1 + excess_hs).cumprod()

            print(f"\n相对沪深300:")
            print(f"  累计超额收益: {cum_excess_hs.iloc[-1] - 1:.2%}")
            if len(excess_hs) > 0:
                ann_excess_hs = (1 + cum_excess_hs.iloc[-1] - 1) ** (252 / len(excess_hs)) - 1
                print(f"  年化超额收益: {ann_excess_hs:.2%}")
                print(f"  超额收益夏普: {excess_hs.mean() / excess_hs.std() * np.sqrt(252):.4f}")

    if csi500_returns is not None:
        common_idx = strategy_returns.index.intersection(csi500_returns.index)
        if len(common_idx) > 0:
            strat_aligned = strategy_returns.loc[common_idx]
            csi500_aligned = csi500_returns.loc[common_idx]
            excess_csi = strat_aligned - csi500_aligned

            cum_excess_csi = (1 + excess_csi).cumprod()

            print(f"\n相对中证500:")
            print(f"  累计超额收益: {cum_excess_csi.iloc[-1] - 1:.2%}")
            if len(excess_csi) > 0:
                ann_excess_csi = (1 + cum_excess_csi.iloc[-1] - 1) ** (252 / len(excess_csi)) - 1
                print(f"  年化超额收益: {ann_excess_csi:.2%}")
                print(f"  超额收益夏普: {excess_csi.mean() / excess_csi.std() * np.sqrt(252):.4f}")

    # 绘制对比图
    print("\n生成对比图表...")
    plot_comparison(
        strategy_equity,
        hs300_data["close"] if hs300_data is not None else None,
        csi500_data["close"] if csi500_data is not None else None,
        strategy_metrics,
        hs300_metrics,
        csi500_metrics,
        strategy_returns,
        hs300_returns,
        csi500_returns,
    )

    print("\n[OK] 对比分析完成")


def plot_comparison(strategy_equity, hs300_series, csi500_series,
                    strat_metrics, hs300_metrics, csi500_metrics,
                    strategy_returns, hs300_returns, csi500_returns):
    """绘制对比图"""
    fig, axes = plt.subplots(3, 1, figsize=(16, 14), dpi=120)

    # 归一化净值
    initial = 500000
    strat_nav = strategy_equity / initial

    if hs300_series is not None:
        hs300_nav = hs300_series / hs300_series.iloc[0]
    if csi500_series is not None:
        csi500_nav = csi500_series / csi500_series.iloc[0]

    # 子图1: 净值曲线对比
    ax1 = axes[0]
    ax1.plot(strat_nav.index, strat_nav.values,
             label=f"Strategy (Total:{strat_metrics['total_return']:.1%}, Sharpe:{strat_metrics['sharpe']:.2f})",
             color="#2E75B6", linewidth=2)

    if hs300_series is not None:
        ax1.plot(hs300_nav.index, hs300_nav.values,
                 label=f"HS300 (Total:{hs300_metrics['total_return']:.1%}, Sharpe:{hs300_metrics['sharpe']:.2f})",
                 color="#E74C3C", linewidth=1.5, linestyle="--")

    if csi500_series is not None:
        ax1.plot(csi500_nav.index, csi500_nav.values,
                 label=f"CSI500 (Total:{csi500_metrics['total_return']:.1%}, Sharpe:{csi500_metrics['sharpe']:.2f})",
                 color="#27AE60", linewidth=1.5, linestyle="--")

    ax1.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
    ax1.set_title("Strategy vs Benchmark: Net Value (AKShare Data)", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11, loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylabel("Normalized Value")

    # 子图2: 超额收益
    ax2 = axes[1]

    if hs300_series is not None:
        common_idx = strat_nav.index.intersection(hs300_nav.index)
        excess_hs = strat_nav.loc[common_idx].values - hs300_nav.loc[common_idx].values

        ax2.fill_between(common_idx, 0, excess_hs,
                         where=excess_hs >= 0, alpha=0.3, color="#2E7D32", label="Outperform HS300")
        ax2.fill_between(common_idx, 0, excess_hs,
                         where=excess_hs < 0, alpha=0.3, color="#C62828", label="Underperform HS300")
        ax2.plot(common_idx, excess_hs, color="#1B3A5C", linewidth=1)
        ax2.axhline(0, color="black", linewidth=0.5)

    ax2.set_title("Excess Return (vs HS300)", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylabel("Excess Return")

    # 子图3: 回撤对比
    ax3 = axes[2]

    strat_dd = strat_nav / strat_nav.cummax() - 1
    ax3.fill_between(strat_dd.index, 0, strat_dd.values,
                     alpha=0.3, color="#2E75B6", label=f"Strategy (MaxDD:{strat_metrics['max_drawdown']:.1%})")

    if hs300_series is not None:
        hs300_dd = hs300_nav / hs300_nav.cummax() - 1
        ax3.fill_between(hs300_dd.index, 0, hs300_dd.values,
                         alpha=0.3, color="#E74C3C",
                         label=f"HS300 (MaxDD:{hs300_metrics['max_drawdown']:.1%})")

    if csi500_series is not None:
        csi500_dd = csi500_nav / csi500_nav.cummax() - 1
        ax3.fill_between(csi500_dd.index, 0, csi500_dd.values,
                         alpha=0.3, color="#27AE60",
                         label=f"CSI500 (MaxDD:{csi500_metrics['max_drawdown']:.1%})")

    ax3.set_title("Drawdown Comparison", fontsize=14, fontweight="bold")
    ax3.legend(fontsize=11, loc="lower left")
    ax3.grid(True, alpha=0.3)
    ax3.set_ylabel("Drawdown")

    # X轴格式
    from matplotlib.dates import DateFormatter, MonthLocator
    for ax in axes:
        ax.xaxis.set_major_locator(MonthLocator(interval=6))
        ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()

    # 保存
    results_dir = Path("results")
    output_path = results_dir / "benchmark_comparison_akshare.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    print(f"[OK] Chart saved: {output_path}")


if __name__ == "__main__":
    main()
