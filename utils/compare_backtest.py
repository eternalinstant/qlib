"""
回测引擎对比分析
对比 Qlib 和 PyBroker 两个回测引擎的结果差异
"""
import pandas as pd
import numpy as np
from pathlib import Path


def load_latest_results():
    """加载最新的回测结果"""
    results_dir = Path("results")

    # 找到最新的结果文件
    qlib_files = sorted(results_dir.glob("backtest_*.csv"), reverse=True)
    pybroker_files = sorted(results_dir.glob("pybroker_*.csv"), reverse=True)

    if not qlib_files:
        raise FileNotFoundError("没有找到 Qlib 回测结果")
    if not pybroker_files:
        raise FileNotFoundError("没有找到 PyBroker 回测结果")

    qlib_df = pd.read_csv(qlib_files[0], parse_dates=["date"])
    pybroker_df = pd.read_csv(pybroker_files[0], parse_dates=["date"])

    print(f"Qlib 结果: {qlib_files[0].name}")
    print(f"PyBroker 结果: {pybroker_files[0].name}\n")

    return qlib_df, pybroker_df


def calculate_metrics(df, initial_capital=500000):
    """计算回测指标"""
    df = df.set_index("date").sort_index()

    # 检查数据格式：Qlib 有 return 列，PyBroker 有 equity 列
    if "return" in df.columns:
        # Qlib 格式
        returns = df["return"]
        equity_series = initial_capital * (1 + returns).cumprod()
    elif "equity" in df.columns:
        # PyBroker 格式
        equity_series = df["equity"]
        # 计算日收益率
        returns = equity_series.pct_change().fillna(0)
    else:
        # 尝试第一列
        col = df.columns[0]
        if col == "equity":
            equity_series = df[col]
            returns = equity_series.pct_change().fillna(0)
        else:
            returns = df[col]
            equity_series = initial_capital * (1 + returns).cumprod()

    # 累计收益
    cum_ret = equity_series / equity_series.iloc[0]

    # 总收益率
    total_ret = cum_ret.iloc[-1] - 1

    # 年化收益率
    days = (cum_ret.index[-1] - cum_ret.index[0]).days
    ann_ret = (1 + total_ret) ** (365 / days) - 1

    # 最大回撤
    rolling_max = cum_ret.cummax()
    drawdown = cum_ret / rolling_max - 1
    max_dd = drawdown.min()

    # 夏普比率
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

    # 日胜率
    win_rate = (returns > 0).mean()

    # 期末资产
    final_value = equity_series.iloc[-1]

    return {
        "total_return": total_ret,
        "annual_return": ann_ret,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "win_rate": win_rate,
        "final_value": final_value,
        "cum_ret": cum_ret,
        "returns": returns,
    }


def print_comparison(qlib_metrics, pybroker_metrics):
    """打印对比结果"""
    print("=" * 70)
    print("  回测引擎对比分析")
    print("=" * 70)

    print(f"\n{'指标':<20} {'Qlib':>15} {'PyBroker':>15} {'差异':>15}")
    print("-" * 70)

    # 总收益率
    qlib_total = qlib_metrics["total_return"]
    pyb_total = pybroker_metrics["total_return"]
    diff_total = pyb_total - qlib_total
    print(f"{'总收益率':<20} {qlib_total:>14.2%} {pyb_total:>14.2%} {diff_total:>+14.2%}")

    # 年化收益率
    qlib_ann = qlib_metrics["annual_return"]
    pyb_ann = pybroker_metrics["annual_return"]
    diff_ann = pyb_ann - qlib_ann
    print(f"{'年化收益率':<20} {qlib_ann:>14.2%} {pyb_ann:>14.2%} {diff_ann:>+14.2%}")

    # 最大回撤
    qlib_dd = qlib_metrics["max_drawdown"]
    pyb_dd = pybroker_metrics["max_drawdown"]
    diff_dd = pyb_dd - qlib_dd
    print(f"{'最大回撤':<20} {qlib_dd:>14.2%} {pyb_dd:>14.2%} {diff_dd:>+14.2%}")

    # 夏普比率
    qlib_sharpe = qlib_metrics["sharpe"]
    pyb_sharpe = pybroker_metrics["sharpe"]
    diff_sharpe = pyb_sharpe - qlib_sharpe
    print(f"{'夏普比率':<20} {qlib_sharpe:>14.4f} {pyb_sharpe:>14.4f} {diff_sharpe:>+14.4f}")

    # 日胜率
    qlib_wr = qlib_metrics["win_rate"]
    pyb_wr = pybroker_metrics["win_rate"]
    diff_wr = pyb_wr - qlib_wr
    print(f"{'日胜率':<20} {qlib_wr:>14.2%} {pyb_wr:>14.2%} {diff_wr:>+14.2%}")

    # 期末资产
    qlib_final = qlib_metrics["final_value"]
    pyb_final = pybroker_metrics["final_value"]
    diff_final = pyb_final - qlib_final
    print(f"{'期末资产':<20} {qlib_final:>14,.0f} {pyb_final:>14,.0f} {diff_final:>+14,.0f}")

    print()


def analyze_differences(qlib_df, pybroker_df):
    """分析差异原因"""
    print("=" * 70)
    print("  差异分析")
    print("=" * 70)

    # 检查数据范围
    qlib_df = qlib_df.set_index("date").sort_index()
    pybroker_df = pybroker_df.set_index("date").sort_index()

    # 获取收益率
    if "return" in qlib_df.columns:
        qlib_returns = qlib_df["return"]
    else:
        qlib_equity = qlib_df["equity"] if "equity" in qlib_df.columns else qlib_df.iloc[:, 0]
        qlib_returns = qlib_equity.pct_change().fillna(0)

    if "return" in pybroker_df.columns:
        pyb_returns = pybroker_df["return"]
    else:
        pyb_equity = pybroker_df["equity"] if "equity" in pybroker_df.columns else pybroker_df.iloc[:, 0]
        pyb_returns = pyb_equity.pct_change().fillna(0)

    print(f"\n数据范围:")
    print(f"  Qlib:    {qlib_returns.index[0].date()} ~ {qlib_returns.index[-1].date()} ({len(qlib_returns)} 天)")
    print(f"  PyBroker: {pyb_returns.index[0].date()} ~ {pyb_returns.index[-1].date()} ({len(pyb_returns)} 天)")

    # 找到共同的日期范围
    common_start = max(qlib_returns.index[0], pyb_returns.index[0])
    common_end = min(qlib_returns.index[-1], pyb_returns.index[-1])

    qlib_common = qlib_returns.loc[common_start:common_end]
    pyb_common = pyb_returns.loc[common_start:common_end]

    print(f"\n共同范围:")
    print(f"  {common_start.date()} ~ {common_end.date()} ({len(qlib_common)} 天)")

    # 计算日收益差异
    diff = pyb_common - qlib_common
    print(f"\n日收益率差异:")
    print(f"  平均差异: {diff.mean():+.4f} ({diff.mean()*100:+.2f} bp)")
    print(f"  最大差异: {diff.max():+.4f} ({diff.max()*100:+.2f} bp)")
    print(f"  最小差异: {diff.min():+.4f} ({diff.min()*100:+.2f} bp)")
    print(f"  标准差: {diff.std():.4f}")

    # 差异分布
    large_diff = diff[abs(diff) > 0.001]  # 差异超过 10bp
    print(f"\n大差异天数 (>|10bp|): {len(large_diff)} 天")

    # 相关性
    correlation = qlib_common.corr(pyb_common)
    print(f"日收益率相关性: {correlation:.4f}")

    print()


def main():
    """主函数"""
    qlib_df, pybroker_df = load_latest_results()

    qlib_metrics = calculate_metrics(qlib_df)
    pybroker_metrics = calculate_metrics(pybroker_df)

    print_comparison(qlib_metrics, pybroker_metrics)
    analyze_differences(qlib_df, pybroker_df)

    print("=" * 70)
    print("  结论")
    print("=" * 70)
    print("""
两个引擎的主要差异可能来自:

1. 交易成本计算方式
   - Qlib: 每日按比例扣除
   - PyBroker: 按实际交易发生时扣除

2. T+1 成交假设
   - 两者都使用 buy_delay=1, sell_delay=1
   - 但具体实现可能有细微差异

3. 资金利用率
   - PyBroker 可能有更高的资金利用率
   - 导致最终收益更高

4. 数据处理
   - 两者对停牌、涨跌停的处理可能不同

建议: 使用 PyBroker 结果作为更准确的实盘参考。
    """)


if __name__ == "__main__":
    main()
