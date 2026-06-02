"""
策略对比框架测试
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.backtest.compare import (
    compare_strategies,
    blend_strategies,
    calculate_metrics,
    comparison_metrics_raw,
    StrategyMetrics,
    yearly_comparison,
    print_yearly_comparison,
    plot_yearly_comparison,
    plot_multi_strategy,
    load_benchmark,
    print_comparison,
    run_compare,
)
from modules.backtest.base import BacktestResult


class TestCompareStrtegies:
    """测试策略对比功能"""

    @pytest.fixture
    def sample_results(self):
        """创建样例回测结果"""
        dates = pd.date_range("2024-01-01", "2024-01-10", freq="D")
        
        # 策略1
        returns1 = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01, 0.02, -0.02, 0.01, 0.02, 0.01], index=dates)
        value1 = (1 + returns1).cumprod()
        result1 = BacktestResult(daily_returns=returns1, portfolio_value=value1)
        
        # 策略2
        returns2 = pd.Series([0.015, 0.01, -0.005, 0.02, 0.015, 0.01, -0.01, 0.02, 0.015, 0.01], index=dates)
        value2 = (1 + returns2).cumprod()
        result2 = BacktestResult(daily_returns=returns2, portfolio_value=value2)
        
        return {"策略1": result1, "策略2": result2}

    def test_compare_strategies(self, sample_results):
        """测试策略对比表格生成"""
        df = compare_strategies(sample_results)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "策略名" in df.columns or df.index.name == "策略名" or "策略名" in df.index.name

    def test_compare_empty_results(self):
        """测试空结果"""
        df = compare_strategies({})
        assert df.empty

    def test_blend_strategies_equal_weight(self, sample_results):
        """测试等权组合"""
        blended = blend_strategies(sample_results)
        
        assert isinstance(blended, BacktestResult)
        assert not blended.daily_returns.empty

    def test_blend_strategies_custom_weight(self, sample_results):
        """测试自定义权重组合"""
        weights = {"策略1": 0.7, "策略2": 0.3}
        blended = blend_strategies(sample_results, weights=weights)
        
        assert isinstance(blended, BacktestResult)
        assert blended.metadata["weights"] == weights

    def test_blend_strategies_empty(self):
        """测试空组合"""
        blended = blend_strategies({})
        
        assert isinstance(blended, BacktestResult)
        assert blended.daily_returns.empty

    def test_calculate_metrics(self, sample_results):
        """测试指标计算"""
        metrics = calculate_metrics(sample_results["策略1"])
        
        assert isinstance(metrics, StrategyMetrics)
        assert metrics.total_return > 0
        assert metrics.annual_return > 0
        assert isinstance(metrics.sharpe_ratio, float)
        assert isinstance(metrics.max_drawdown, float)

    def test_print_comparison(self, sample_results, capsys):
        """测试打印对比表格"""
        print_comparison(sample_results)
        captured = capsys.readouterr()
        assert "策略" in captured.out or "对比" in captured.out

    def test_comparison_metrics_raw(self, sample_results):
        """测试原始指标表输出。"""
        df = comparison_metrics_raw(sample_results)

        assert isinstance(df, pd.DataFrame)
        assert "annual_return" in df.columns
        assert "max_drawdown" in df.columns


def test_run_compare_validates_strategy_and_saves_outputs(tmp_path):
    """compare 入口应先做预检，并保存新的对比 CSV。"""
    fake_strategy = Mock()
    fake_strategy.name = "top15_robust_ma5_fixed"
    fake_strategy.universe = "csi300"

    dates = pd.date_range("2026-01-01", periods=3, freq="D")
    returns = pd.Series([0.01, -0.005, 0.02], index=dates)
    result = BacktestResult(daily_returns=returns, portfolio_value=(1 + returns).cumprod())

    fake_engine = Mock()
    fake_engine.run.return_value = result
    fake_config = SimpleNamespace(get=lambda key, default=None: str(tmp_path) if key == "results_path" else default)

    with patch("core.strategy.Strategy.load", return_value=fake_strategy), \
         patch("modules.backtest.compare.print_comparison"), \
         patch("modules.backtest.compare.print_yearly_comparison"), \
         patch("modules.backtest.compare.plot_multi_strategy"), \
         patch("modules.backtest.compare.plot_yearly_comparison"), \
         patch("modules.backtest.qlib_engine.QlibBacktestEngine", return_value=fake_engine), \
         patch("config.config.CONFIG", fake_config):
        run_compare(strategy_names=["top15_robust_ma5_fixed"], engine="qlib", benchmark=False)

    fake_strategy.validate_data_requirements.assert_called_once_with()
    csv_files = sorted(tmp_path.glob("strategy_compare_historical_csi300_*.csv"))
    yearly_files = sorted(tmp_path.glob("strategy_yearly_compare_historical_csi300_*.csv"))
    assert csv_files
    assert yearly_files


def test_yearly_comparison_groups_by_year():
    """按年对比应输出年份/策略双层索引。"""
    dates = pd.to_datetime(
        ["2024-12-30", "2024-12-31", "2025-01-02", "2025-01-03"]
    )
    returns1 = pd.Series([0.01, -0.01, 0.02, 0.01], index=dates)
    returns2 = pd.Series([0.0, 0.01, -0.005, 0.02], index=dates)
    results = {
        "策略1": BacktestResult(returns1, (1 + returns1).cumprod()),
        "策略2": BacktestResult(returns2, (1 + returns2).cumprod()),
    }

    df = yearly_comparison(results)

    assert isinstance(df, pd.DataFrame)
    assert df.index.names == ["年份", "策略"]
    assert (2024, "策略1") in df.index
    assert (2025, "策略2") in df.index
    assert set(df.columns) == {"收益率", "夏普", "最大回撤"}


def test_print_yearly_comparison_outputs_tables(capsys):
    """按年打印应包含收益、夏普和回撤区块。"""
    dates = pd.to_datetime(
        ["2024-12-30", "2024-12-31", "2025-01-02", "2025-01-03"]
    )
    returns = pd.Series([0.01, -0.02, 0.03, 0.01], index=dates)
    results = {"策略1": BacktestResult(returns, (1 + returns).cumprod())}

    print_yearly_comparison(results)
    captured = capsys.readouterr()

    assert "按年收益率对比" in captured.out
    assert "按年夏普比率对比" in captured.out
    assert "按年最大回撤对比" in captured.out


def test_plot_yearly_comparison_saves_file(tmp_path):
    """按年对比图应成功落盘。"""
    dates = pd.to_datetime(
        ["2024-12-30", "2024-12-31", "2025-01-02", "2025-01-03"]
    )
    returns = pd.Series([0.01, -0.01, 0.02, 0.0], index=dates)
    results = {"策略1": BacktestResult(returns, (1 + returns).cumprod())}
    output_path = tmp_path / "yearly_compare.png"

    plot_yearly_comparison(results, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_multi_strategy_saves_file_without_benchmark(tmp_path):
    """无基准时也应能输出多策略图。"""
    dates = pd.date_range("2026-01-01", periods=4, freq="D")
    returns1 = pd.Series([0.01, -0.005, 0.02, 0.0], index=dates)
    returns2 = pd.Series([0.005, 0.01, -0.01, 0.015], index=dates)
    results = {
        "策略1": BacktestResult(returns1, (1 + returns1).cumprod()),
        "策略2": BacktestResult(returns2, (1 + returns2).cumprod()),
    }
    output_path = tmp_path / "multi_compare.png"

    plot_multi_strategy(results, None, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_load_benchmark_returns_backtest_result():
    """基准加载成功时应转换成 BacktestResult。"""
    benchmark_df = pd.DataFrame(
        {"close": [100.0, 101.0, 102.0]},
        index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
    )

    with patch(
        "utils.benchmark_comparison_akshare.get_benchmark_data_akshare",
        return_value=benchmark_df,
    ) as mock_get:
        result = load_benchmark("2026-01-01", "2026-01-03")

    mock_get.assert_called_once_with("sh000300", "2026-01-01", "2026-01-03")
    assert isinstance(result, BacktestResult)
    assert result.metadata["name"] == "沪深300"
    assert not result.portfolio_value.empty


def test_load_benchmark_returns_none_when_data_missing(capsys):
    """基准缺失时应返回 None 并打印警告。"""
    with patch(
        "utils.benchmark_comparison_akshare.get_benchmark_data_akshare",
        return_value=pd.DataFrame(),
    ):
        result = load_benchmark("2026-01-01", "2026-01-03")

    captured = capsys.readouterr()
    assert result is None
    assert "无法加载沪深300基准数据" in captured.out


def test_run_compare_with_benchmark_and_mixed_universe(tmp_path):
    """混合股票池 + benchmark 时应保存 mixed_universe 结果并调用基准加载。"""
    strategy_all = Mock()
    strategy_all.name = "top15_core_day"
    strategy_all.universe = "all"

    strategy_csi300 = Mock()
    strategy_csi300.name = "top15_core_trend"
    strategy_csi300.universe = "csi300"

    dates = pd.date_range("2026-01-01", periods=3, freq="D")
    returns1 = pd.Series([0.01, -0.005, 0.02], index=dates)
    returns2 = pd.Series([0.0, 0.01, -0.01], index=dates)
    benchmark_returns = pd.Series([0.0, 0.002, 0.001], index=dates)
    result1 = BacktestResult(daily_returns=returns1, portfolio_value=(1 + returns1).cumprod())
    result2 = BacktestResult(daily_returns=returns2, portfolio_value=(1 + returns2).cumprod())
    benchmark_result = BacktestResult(
        daily_returns=benchmark_returns,
        portfolio_value=(1 + benchmark_returns).cumprod(),
        metadata={"name": "沪深300"},
    )

    fake_engine = Mock()
    fake_engine.run.side_effect = [result1, result2]
    fake_config = SimpleNamespace(
        get=lambda key, default=None: str(tmp_path) if key == "results_path" else default
    )

    with patch(
        "core.strategy.Strategy.load",
        side_effect=[strategy_all, strategy_csi300],
    ), \
        patch("modules.backtest.compare.print_comparison"), \
        patch("modules.backtest.compare.print_yearly_comparison"), \
        patch("modules.backtest.compare.plot_multi_strategy") as mock_plot_multi, \
        patch("modules.backtest.compare.plot_yearly_comparison") as mock_plot_yearly, \
        patch("modules.backtest.compare.load_benchmark", return_value=benchmark_result) as mock_load_benchmark, \
        patch("modules.backtest.qlib_engine.QlibBacktestEngine", return_value=fake_engine), \
        patch("modules.backtest.compare.datetime") as mock_datetime, \
        patch("config.config.CONFIG", fake_config):
        mock_datetime.now.return_value.strftime.return_value = "20260322_235959"
        results = run_compare(
            strategy_names=["top15_core_day", "top15_core_trend"],
            engine="qlib",
            benchmark=True,
        )

    strategy_all.validate_data_requirements.assert_called_once_with()
    strategy_csi300.validate_data_requirements.assert_called_once_with()
    mock_load_benchmark.assert_called_once_with("2026-01-01", "2026-01-03")
    assert "沪深300" in results
    plot_path = mock_plot_multi.call_args.args[2]
    yearly_path = mock_plot_yearly.call_args.args[1]
    assert "mixed_universe" in str(plot_path)
    assert "mixed_universe" in str(yearly_path)
    csv_files = sorted(tmp_path.glob("strategy_compare_mixed_universe_*.csv"))
    yearly_files = sorted(tmp_path.glob("strategy_yearly_compare_mixed_universe_*.csv"))
    assert csv_files
    assert yearly_files


def test_run_compare_invalid_engine_returns_empty(capsys):
    """非法引擎应直接返回空结果。"""
    results = run_compare(strategy_names=["top15_core_day"], engine="invalid", benchmark=False)

    captured = capsys.readouterr()
    assert results == {}
    assert "不支持的引擎" in captured.out


def test_run_compare_without_available_strategies_returns_empty(capsys):
    """没有可用策略时应返回空结果。"""
    with patch("core.strategy.Strategy.list_available", return_value=[]):
        results = run_compare(strategy_names=None, engine="qlib", benchmark=False)

    captured = capsys.readouterr()
    assert results == {}
    assert "没有可用策略" in captured.out
