#!/usr/bin/env python3
"""
QuantStats 集成
生成专业回测分析报告

Usage:
    from utils.quantstats_report import generate_report, analyze_result
    
    # 从回测结果生成报告
    result = engine.run(strategy=strategy)
    generate_report(result, output_path="results/report.html")
    
    # 直接传入收益率序列
    generate_report(returns=daily_returns, benchmark=benchmark_returns)
"""

import warnings
warnings.filterwarnings("ignore")

import os
from pathlib import Path
from typing import Optional, Union
import pandas as pd
import numpy as np


def _ensure_package():
    """延迟导入 quantstats"""
    try:
        import quantstats as qs
        return qs
    except ImportError:
        raise ImportError("请安装 quantstats: pip install quantstats")


def prepare_returns(result) -> pd.Series:
    """
    从 BacktestResult 提取日收益率序列
    
    Parameters
    ----------
    result : BacktestResult
        回测结果对象，需包含 daily_returns 属性
        
    Returns
    -------
    pd.Series
        日收益率序列，index 为日期
    """
    if hasattr(result, 'daily_returns'):
        returns = result.daily_returns
        if isinstance(returns, pd.DataFrame):
            returns = returns.iloc[:, 0]
        return returns
    elif hasattr(result, 'portfolio_value'):
        pv = result.portfolio_value
        if pv.empty:
            raise ValueError("portfolio_value 为空")
        returns = pv.pct_change().dropna()
        returns.name = "portfolio"
        return returns
    else:
        raise ValueError(f"不支持的 result 类型: {type(result)}")


def prepare_benchmark(symbol: str = "^000300") -> Optional[pd.Series]:
    """
    获取基准收益率序列
    
    Parameters
    ----------
    symbol : str
        基准代码，默认沪深300 (^000300)
        
    Returns
    -------
    pd.Series or None
        日收益率序列
    """
    try:
        import akshare as ak
        if symbol == "^000300":
            df = ak.stock_zh_index_daily(symbol="sh000300")
        elif symbol == "^000905":
            df = ak.stock_zh_index_daily(symbol="sh000905")
        else:
            return None
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        returns = df['close'].pct_change().dropna()
        returns.name = symbol
        return returns
    except Exception:
        return None


def generate_report(
    result=None,
    returns: Optional[pd.Series] = None,
    benchmark: Optional[Union[str, pd.Series]] = "^000300",
    output_path: Optional[str] = None,
    title: str = "策略回测报告",
    rf: float = 0.03,
) -> None:
    """
    生成 QuantStats HTML 报告
    
    Parameters
    ----------
    result : BacktestResult, optional
        回测结果对象
    returns : pd.Series, optional
        日收益率序列，与 result 二选一
    benchmark : str or pd.Series, optional
        基准代码 (str) 或收益率序列，默认沪深300
    output_path : str, optional
        输出 HTML 文件路径
    title : str
        报告标题
    rf : float
        无风险利率，默认 3%
    """
    qs = _ensure_package()
    
    # 获取收益率序列
    if returns is None:
        if result is None:
            raise ValueError("必须提供 result 或 returns")
        returns = prepare_returns(result)
    
    # 设置无风险利率
    qs.stats.RF = rf
    
    # 获取基准
    bench_series = None
    if benchmark is not None:
        if isinstance(benchmark, str):
            bench_series = prepare_benchmark(benchmark)
        elif isinstance(benchmark, pd.Series):
            bench_series = benchmark
        if bench_series is not None:
            # 对齐日期
            common = returns.index.intersection(bench_series.index)
            if len(common) > 0:
                returns = returns.loc[common]
                bench_series = bench_series.loc[common]
    
    # 输出路径
    if output_path is None:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"results/quantstats_{ts}.html"
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # 生成报告
    qs.reports.html(
        returns=returns,
        benchmark=bench_series,
        title=title,
        output=output_path,
        rf=rf,
    )
    print(f"报告已生成: {output_path}")


def print_summary(
    result=None,
    returns: Optional[pd.Series] = None,
    benchmark: Optional[Union[str, pd.Series]] = "^000300",
    rf: float = 0.03,
) -> dict:
    """
    打印关键指标摘要
    
    Parameters
    ----------
    result : BacktestResult, optional
        回测结果对象
    returns : pd.Series, optional
        日收益率序列
    benchmark : str or pd.Series, optional
        基准代码或序列
    rf : float
        无风险利率
        
    Returns
    -------
    dict
        关键指标字典
    """
    qs = _ensure_package()
    
    if returns is None:
        returns = prepare_returns(result)
    
    qs.stats.RF = rf
    
    bench_series = None
    if benchmark is not None:
        if isinstance(benchmark, str):
            bench_series = prepare_benchmark(benchmark)
        elif isinstance(benchmark, pd.Series):
            bench_series = benchmark
    
    print(f"\n{'='*50}")
    print("  QuantStats 策略分析")
    print(f"{'='*50}")
    
    print(f"\n📈 收益指标:")
    print(f"  总收益率:   {qs.stats.comp(returns):.2%}")
    print(f"  年化收益:   {qs.stats.cagr(returns):.2%}")
    monthly = qs.stats.monthly_returns(returns)
    if isinstance(monthly, pd.DataFrame):
        monthly = monthly.values.flatten()
    print(f"  月收益:     {pd.Series(monthly).mean():.2%}")
    
    print(f"\n📉 风险指标:")
    print(f"  波动率:     {qs.stats.volatility(returns):.2%}")
    print(f"  最大回撤:   {qs.stats.max_drawdown(returns):.2%}")
    print(f"  VaR(95%):   {qs.stats.value_at_risk(returns):.2%}")
    
    print(f"\n⚖️ 风险调整收益:")
    print(f"  夏普比率:   {qs.stats.sharpe(returns):.4f}")
    print(f"  索提诺:    {qs.stats.sortino(returns):.4f}")
    print(f"  Calmar:    {qs.stats.calmar(returns):.4f}")
    print(f"  Omega:     {qs.stats.omega(returns):.4f}")
    
    print(f"\n🎯 胜率指标:")
    print(f"  日胜率:     {qs.stats.win_rate(returns):.2%}")
    print(f"  盈亏比:     {qs.stats.profit_factor(returns):.4f}")
    
    if bench_series is not None:
        common = returns.index.intersection(bench_series.index)
        if len(common) > 0:
            r = returns.loc[common]
            b = bench_series.loc[common]
            print(f"\n📊 相对基准:")
            print(f"  超额收益:  {(r.mean() - b.mean()) * 252:.2%}")
            cov = np.cov(r, b)[0][1]
            var_b = np.var(b)
            if var_b > 0:
                print(f"  Beta:      {cov / var_b:.4f}")
    
    print(f"\n{'='*50}\n")
    
    return {
        "cagr": qs.stats.cagr(returns),
        "annual_return": qs.stats.cagr(returns),
        "total_return": qs.stats.comp(returns),
        "volatility": qs.stats.volatility(returns),
        "max_drawdown": qs.stats.max_drawdown(returns),
        "sharpe": qs.stats.sharpe(returns),
        "sortino": qs.stats.sortino(returns),
        "calmar": qs.stats.calmar(returns),
        "win_rate": qs.stats.win_rate(returns),
        "profit_factor": qs.stats.profit_factor(returns),
    }


def plot_returns(
    result=None,
    returns: Optional[pd.Series] = None,
    benchmark: Optional[Union[str, pd.Series]] = "^000300",
    save_path: Optional[str] = None,
) -> None:
    """
    绘制收益曲线对比图
    
    Parameters
    ----------
    result : BacktestResult, optional
        回测结果
    returns : pd.Series, optional
        收益率序列
    benchmark : str or pd.Series, optional
        基准
    save_path : str, optional
        保存路径
    """
    qs = _ensure_package()
    import matplotlib
    matplotlib.use("Agg")
    
    if returns is None:
        returns = prepare_returns(result)
    
    bench_series = None
    if benchmark is not None:
        if isinstance(benchmark, str):
            bench_series = prepare_benchmark(benchmark)
        elif isinstance(benchmark, pd.Series):
            bench_series = benchmark
    
    if save_path is None:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"results/qs_returns_{ts}.png"
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    qs.plots(
        returns,
        benchmark=bench_series,
        savefig=save_path,
        figsize=(12, 8),
    )
    print(f"图表已保存: {save_path}")


def plot_distribution(
    result=None,
    returns: Optional[pd.Series] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    绘制收益分布图
    
    Parameters
    ----------
    result : BacktestResult, optional
        回测结果
    returns : pd.Series, optional
        收益率序列
    save_path : str, optional
        保存路径
    """
    qs = _ensure_package()
    import matplotlib
    matplotlib.use("Agg")
    
    if returns is None:
        returns = prepare_returns(result)
    
    if save_path is None:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"results/qs_dist_{ts}.png"
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    qs.plots(returns, mode="freq", savefig=save_path, figsize=(12, 8))
    print(f"图表已保存: {save_path}")


# 命令行入口
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="生成 QuantStats 回测报告")
    parser.add_argument("-r", "--result", type=str, help="回测结果 CSV 文件路径")
    parser.add_argument("-o", "--output", type=str, help="输出 HTML 路径")
    parser.add_argument("-b", "--benchmark", type=str, default="^000300", help="基准代码")
    parser.add_argument("--summary", action="store_true", help="仅打印摘要")
    parser.add_argument("--plot", action="store_true", help="生成图表")
    
    args = parser.parse_args()
    
    if args.result:
        df = pd.read_csv(args.result, index_col=0, parse_dates=True)
        returns = df['return'] if 'return' in df.columns else df.iloc[:, 0]
    else:
        print("请提供回测结果文件: -r <path>")
        exit(1)
    
    if args.summary:
        print_summary(returns=returns, benchmark=args.benchmark)
    elif args.plot:
        plot_returns(returns=returns, benchmark=args.benchmark, save_path=args.output)
    else:
        generate_report(returns=returns, benchmark=args.benchmark, output_path=args.output)
