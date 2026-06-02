"""
主入口测试
"""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import main


def test_load_strategy_none_uses_default():
    """未传 strategy 时应加载 default 策略文件"""
    fake_strategy = Mock()
    fake_strategy.name = "default"

    with patch("core.strategy.Strategy.load", return_value=fake_strategy) as mock_load:
        strategy = main._load_strategy(None)

    assert strategy is fake_strategy
    mock_load.assert_called_once_with("default")


def test_cmd_select_without_strategy_uses_default_strategy():
    """select 默认应走 Strategy.generate_selections，而不是旧兼容路径"""
    fake_strategy = Mock()
    fake_strategy.name = "default"

    args = SimpleNamespace(strategy=None, config="strategy.yaml")

    with patch("main._load_strategy", return_value=fake_strategy) as mock_load, \
         patch("core.selection.generate_selections") as legacy_generate:
        main.cmd_select(args)

    mock_load.assert_called_once_with(None)
    fake_strategy.validate_data_requirements.assert_called_once_with()
    fake_strategy.generate_selections.assert_called_once_with(force=True)
    legacy_generate.assert_not_called()


def test_cmd_backtest_validates_strategy_data():
    """backtest 前应先做正式数据预检。"""
    fake_strategy = Mock()
    fake_strategy.name = "top15_robust_ma5_fixed"
    fake_result = Mock()
    fake_result.metadata = {"results_file": "results/demo.csv"}

    args = SimpleNamespace(
        strategy="top15_robust_ma5_fixed",
        config="strategy.yaml",
        list_strategies=False,
        engine="qlib",
    )

    with patch("main._load_strategy", return_value=fake_strategy), \
         patch("modules.backtest.composite.run_strategy_backtest", return_value=fake_result) as mock_run, \
         patch("config.config.CONFIG", SimpleNamespace(get=lambda key, default=None: 500000 if key == "initial_capital" else default)):
        main.cmd_backtest(args)

    fake_strategy.validate_data_requirements.assert_called_once_with()
    mock_run.assert_called_once_with(strategy=fake_strategy, engine="qlib")
