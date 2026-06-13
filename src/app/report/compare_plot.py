"""
回测引擎对比可视化
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def plot_comparison():
    """绘制对比图"""
    results_dir = Path("results")

    # 加载数据
    qlib_files = sorted(results_dir.glob("backtest_*.csv"), reverse=True)
    pybroker_files = sorted(results_dir.glob("pybroker_*.csv"), reverse=True)

    if not qlib_files or not pybroker_files:
        print("没有找到回测结果文件")
        return

    qlib_df = pd.read_csv(qlib_files[0], parse_dates=["date"])
    pybroker_df = pd.read_csv(pybroker_files[0], parse_dates=["date"])

    # 设置索引
    qlib_df = qlib_df.set_index("date").sort_index()
    pybroker_df = pybroker_df.set_index("date").sort_index()

    # 计算净值
    initial = 500000
    if "return" in qlib_df.columns:
        qlib_nav = initial * (1 + qlib_df["return"]).cumprod()
    else:
        qlib_nav = qlib_df.get("equity", qlib_df.iloc[:, 0])

    if "equity" in pybroker_df.columns:
        pyb_nav = pybroker_df["equity"]
    else:
        pyb_nav = initial * (1 + pybroker_df["return"]).cumprod()

    # 归一化
    qlib_nav_norm = qlib_nav / qlib_nav.iloc[0]
    pyb_nav_norm = pyb_nav / pyb_nav.iloc[0]

    # 计算回测指标
    def calc_metrics(nav):
        returns = nav.pct_change().fillna(0)
        total_ret = nav.iloc[-1] / nav.iloc[0] - 1
        days = (nav.index[-1] - nav.index[0]).days
        ann_ret = (1 + total_ret) ** (365 / days) - 1
        rolling_max = nav.cummax()
        dd = nav / rolling_max - 1
        max_dd = dd.min()
        sharpe = returns.mean() / returns.std() * np.sqrt(252)
        return total_ret, ann_ret, max_dd, sharpe

    qlib_total, qlib_ann, qlib_dd, qlib_sharpe = calc_metrics(qlib_nav)
    pyb_total, pyb_ann, pyb_dd, pyb_sharpe = calc_metrics(pyb_nav)

    # 创建图表
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), dpi=120)

    # 子图1: 净值曲线
    ax1 = axes[0]
    ax1.plot(qlib_nav_norm.index, qlib_nav_norm.values,
             label=f"Qlib (总收益:{qlib_total:.1%}, 夏普:{qlib_sharpe:.2f})",
             color="#2E75B6", linewidth=1.5)
    ax1.plot(pyb_nav_norm.index, pyb_nav_norm.values,
             label=f"PyBroker (总收益:{pyb_total:.1%}, 夏普:{pyb_sharpe:.2f})",
             color="#E74C3C", linewidth=1.5, linestyle="--")
    ax1.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
    ax1.set_title("回测引擎对比: 净值曲线", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11, loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylabel("归一化净值")

    # 子图2: 收益差异
    ax2 = axes[1]
    # 对齐日期范围
    common_start = max(qlib_nav_norm.index[0], pyb_nav_norm.index[0])
    common_end = min(qlib_nav_norm.index[-1], pyb_nav_norm.index[-1])
    qlib_common = qlib_nav_norm.loc[common_start:common_end]
    pyb_common = pyb_nav_norm.loc[common_start:common_end]

    diff = pyb_common.values - qlib_common.values
    ax2.fill_between(qlib_common.index, 0, diff,
                     where=diff >= 0, alpha=0.3, color="#2E7D32", label="PyBroker 更高")
    ax2.fill_between(qlib_common.index, 0, diff,
                     where=diff < 0, alpha=0.3, color="#C62828", label="Qlib 更高")
    ax2.plot(qlib_common.index, diff, color="#1B3A5C", linewidth=0.8)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_title("净值差异 (PyBroker - Qlib)", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylabel("差异")

    # 子图3: 回撤对比
    ax3 = axes[2]
    qlib_dd_series = qlib_nav_norm / qlib_nav_norm.cummax() - 1
    pyb_dd_series = pyb_nav_norm / pyb_nav_norm.cummax() - 1

    ax3.fill_between(qlib_dd_series.index, 0, qlib_dd_series.values,
                     alpha=0.3, color="#2E75B6", label=f"Qlib (最大回撤:{qlib_dd:.1%})")
    ax3.fill_between(pyb_dd_series.index, 0, pyb_dd_series.values,
                     alpha=0.3, color="#E74C3C", label=f"PyBroker (最大回撤:{pyb_dd:.1%})")
    ax3.plot(qlib_dd_series.index, qlib_dd_series.values, color="#2E75B6", linewidth=0.8)
    ax3.plot(pyb_dd_series.index, pyb_dd_series.values, color="#E74C3C", linewidth=0.8, linestyle="--")
    ax3.set_title("回撤对比", fontsize=14, fontweight="bold")
    ax3.legend(fontsize=11, loc="lower left")
    ax3.grid(True, alpha=0.3)
    ax3.set_ylabel("回撤")

    for ax in axes:
        from matplotlib.dates import DateFormatter, MonthLocator
        ax.xaxis.set_major_locator(MonthLocator(interval=6))
        ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()

    # 保存
    output_path = results_dir / "backtest_comparison.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    print(f"[OK] 对比图表已保存: {output_path}")
    print(f"\n  Qlib:    总收益 {qlib_total:.2%}, 年化 {qlib_ann:.2%}, 最大回撤 {qlib_dd:.2%}, 夏普 {qlib_sharpe:.2f}")
    print(f"  PyBroker: 总收益 {pyb_total:.2%}, 年化 {pyb_ann:.2%}, 最大回撤 {pyb_dd:.2%}, 夏普 {pyb_sharpe:.2f}")
    print(f"  差异:    总收益 +{pyb_total-qlib_total:.2%}, 年化 +{pyb_ann-qlib_ann:.2%}")


if __name__ == "__main__":
    plot_comparison()
