"""TDD: modules/backtest/rule_engine.py"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from strategies.base import RuleStrategy, PositionState, StrategySignal
from modules.backtest.base import BacktestResult
from modules.backtest.rule_engine import RuleBasedEngine


# ── 辅助：简单的 Hold 策略 ─────────────────────────────────────────────────────

class HoldStrategy(RuleStrategy):
    """每天什么都不做，用于测试引擎骨架"""
    def on_bar(self, date, universe, positions, data_provider):
        return []


class BuyAndHoldStrategy(RuleStrategy):
    """第一天买入，之后持有不动"""
    def __init__(self, config):
        super().__init__(config)
        self._bought = False

    def on_bar(self, date, universe, positions, data_provider):
        if not self._bought and universe:
            self._bought = True
            instrument = universe[0]
            return [StrategySignal(date=date, instrument=instrument, action="buy", weight=1.0)]
        return []


class BuyThenSellStrategy(RuleStrategy):
    """第一天买，第二天卖"""
    def __init__(self, config):
        super().__init__(config)
        self._bars = 0

    def on_bar(self, date, universe, positions, data_provider):
        self._bars += 1
        if self._bars == 1 and universe:
            return [StrategySignal(date=date, instrument=universe[0], action="buy", weight=1.0)]
        if self._bars == 2 and positions:
            instrument = list(positions.keys())[0]
            return [StrategySignal(date=date, instrument=instrument, action="sell")]
        return []


# ── 引擎基础 ──────────────────────────────────────────────────────────────────

def test_rule_engine_instantiates():
    engine = RuleBasedEngine({"initial_capital": 1_000_000})
    assert engine is not None


def test_rule_engine_run_returns_backtest_result():
    engine = RuleBasedEngine({"initial_capital": 1_000_000})
    strategy = HoldStrategy({"name": "hold", "selection": {"universe": "all"}})

    mock_dp = MagicMock()
    mock_dp.get_universe.return_value = []

    trade_dates = pd.date_range("2024-01-02", periods=5, freq="B")

    with patch("modules.backtest.rule_engine.QlibDataProvider", return_value=mock_dp):
        with patch("modules.backtest.rule_engine.load_trade_calendar", return_value=trade_dates):
            result = engine.run(strategy)

    assert isinstance(result, BacktestResult)


def test_rule_engine_daily_returns_length_matches_calendar():
    engine = RuleBasedEngine({"initial_capital": 1_000_000})
    strategy = HoldStrategy({"name": "hold", "selection": {"universe": "all"}})

    mock_dp = MagicMock()
    mock_dp.get_universe.return_value = []

    trade_dates = pd.date_range("2024-01-02", periods=10, freq="B")

    with patch("modules.backtest.rule_engine.QlibDataProvider", return_value=mock_dp):
        with patch("modules.backtest.rule_engine.load_trade_calendar", return_value=trade_dates):
            result = engine.run(strategy)

    assert len(result.daily_returns) == len(trade_dates)


def test_rule_engine_calls_on_start_and_on_end():
    engine = RuleBasedEngine({"initial_capital": 1_000_000})

    call_log = []

    class LogStrategy(RuleStrategy):
        def on_start(self, dp):
            call_log.append("start")
        def on_bar(self, date, universe, positions, dp):
            call_log.append("bar")
            return []
        def on_end(self):
            call_log.append("end")

    strategy = LogStrategy({"name": "log", "selection": {"universe": "all"}})
    mock_dp = MagicMock()
    mock_dp.get_universe.return_value = []
    trade_dates = pd.date_range("2024-01-02", periods=3, freq="B")

    with patch("modules.backtest.rule_engine.QlibDataProvider", return_value=mock_dp):
        with patch("modules.backtest.rule_engine.load_trade_calendar", return_value=trade_dates):
            engine.run(strategy)

    assert call_log[0] == "start"
    assert call_log[-1] == "end"
    assert call_log.count("bar") == 3


def test_rule_engine_passes_universe_to_on_bar():
    engine = RuleBasedEngine({"initial_capital": 1_000_000})
    received_universes = []

    class RecordStrategy(RuleStrategy):
        def on_bar(self, date, universe, positions, dp):
            received_universes.append(list(universe))
            return []

    strategy = RecordStrategy({"name": "rec", "selection": {"universe": "csi800"}})
    mock_dp = MagicMock()
    mock_dp.get_universe.return_value = ["SH600000", "SZ000001"]
    trade_dates = pd.date_range("2024-01-02", periods=2, freq="B")

    with patch("modules.backtest.rule_engine.QlibDataProvider", return_value=mock_dp):
        with patch("modules.backtest.rule_engine.load_trade_calendar", return_value=trade_dates):
            engine.run(strategy)

    assert received_universes[0] == ["SH600000", "SZ000001"]


# ── 仓位管理 ──────────────────────────────────────────────────────────────────

def test_rule_engine_buy_signal_creates_position():
    engine = RuleBasedEngine({"initial_capital": 1_000_000})
    strategy = BuyAndHoldStrategy({"name": "bah", "selection": {"universe": "all"}})

    mock_dp = MagicMock()
    mock_dp.get_universe.return_value = ["SZ000001"]
    ohlcv = pd.DataFrame(
        {"open": [10.0], "close": [10.0], "prev_close": [9.9]},
        index=pd.MultiIndex.from_tuples(
            [("SZ000001", pd.Timestamp("2024-01-02"))], names=["instrument", "date"]
        ),
    )
    mock_dp.get_ohlcv.return_value = ohlcv

    trade_dates = pd.date_range("2024-01-02", periods=3, freq="B")
    final_positions = {}

    class CaptureBuyStrategy(RuleStrategy):
        def __init__(self, config, capture_ref):
            super().__init__(config)
            self._bought = False
            self._capture = capture_ref
        def on_bar(self, date, universe, positions, dp):
            self._capture.update(positions)
            if not self._bought and universe:
                self._bought = True
                return [StrategySignal(date=date, instrument=universe[0], action="buy", weight=1.0)]
            return []

    strategy2 = CaptureBuyStrategy(
        {"name": "capture", "selection": {"universe": "all"}}, final_positions
    )

    with patch("modules.backtest.rule_engine.QlibDataProvider", return_value=mock_dp):
        with patch("modules.backtest.rule_engine.load_trade_calendar", return_value=trade_dates):
            engine.run(strategy2)

    assert "SZ000001" in final_positions


def test_rule_engine_sell_signal_removes_position():
    engine = RuleBasedEngine({"initial_capital": 1_000_000})

    mock_dp = MagicMock()
    mock_dp.get_universe.return_value = ["SZ000001"]
    ohlcv = pd.DataFrame(
        {"open": [10.0, 10.2, 10.3], "close": [10.0, 10.2, 10.3], "prev_close": [9.9, 10.0, 10.2]},
        index=pd.MultiIndex.from_tuples(
            [
                ("SZ000001", pd.Timestamp("2024-01-02")),
                ("SZ000001", pd.Timestamp("2024-01-03")),
                ("SZ000001", pd.Timestamp("2024-01-04")),
            ],
            names=["instrument", "date"],
        ),
    )
    mock_dp.get_ohlcv.return_value = ohlcv

    final_positions = {}

    class TrackSell(RuleStrategy):
        def __init__(self, config, ref):
            super().__init__(config)
            self._bars = 0
            self._ref = ref
        def on_bar(self, date, universe, positions, dp):
            self._bars += 1
            # 每个 bar 完整替换，而非 merge
            self._ref.clear()
            self._ref.update(positions)
            if self._bars == 1:
                return [StrategySignal(date=date, instrument="SZ000001", action="buy", weight=1.0)]
            if self._bars == 2:
                return [StrategySignal(date=date, instrument="SZ000001", action="sell")]
            return []

    strategy = TrackSell({"name": "track", "selection": {"universe": "all"}}, final_positions)
    trade_dates = pd.date_range("2024-01-02", periods=3, freq="B")

    with patch("modules.backtest.rule_engine.QlibDataProvider", return_value=mock_dp):
        with patch("modules.backtest.rule_engine.load_trade_calendar", return_value=trade_dates):
            engine.run(strategy)

    assert "SZ000001" not in final_positions


# ── 净值计算 ──────────────────────────────────────────────────────────────────

def test_rule_engine_hold_returns_zero_with_no_positions():
    engine = RuleBasedEngine({"initial_capital": 1_000_000})
    strategy = HoldStrategy({"name": "hold", "selection": {"universe": "all"}})

    mock_dp = MagicMock()
    mock_dp.get_universe.return_value = []
    trade_dates = pd.date_range("2024-01-02", periods=5, freq="B")

    with patch("modules.backtest.rule_engine.QlibDataProvider", return_value=mock_dp):
        with patch("modules.backtest.rule_engine.load_trade_calendar", return_value=trade_dates):
            result = engine.run(strategy)

    assert (result.daily_returns == 0.0).all()


def test_rule_engine_result_can_enter_leaderboard():
    """BacktestResult 格式兼容 Leaderboard"""
    engine = RuleBasedEngine({"initial_capital": 1_000_000})
    strategy = HoldStrategy({"name": "hold_test", "selection": {"universe": "all"}})

    mock_dp = MagicMock()
    mock_dp.get_universe.return_value = []
    trade_dates = pd.date_range("2024-01-02", periods=5, freq="B")

    with patch("modules.backtest.rule_engine.QlibDataProvider", return_value=mock_dp):
        with patch("modules.backtest.rule_engine.load_trade_calendar", return_value=trade_dates):
            result = engine.run(strategy)

    assert hasattr(result, "daily_returns")
    assert hasattr(result, "portfolio_value")
    assert isinstance(result.daily_returns, pd.Series)
    assert isinstance(result.portfolio_value, pd.Series)
