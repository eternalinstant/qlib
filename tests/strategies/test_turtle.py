"""TDD: strategies/turtle/turtle_strategy.py"""
import pytest
import pandas as pd
from unittest.mock import MagicMock
from strategies.turtle.turtle_strategy import TurtleStrategy
from strategies.base import RuleStrategy, PositionState, StrategySignal


def _make_dp(close_prices: dict):
    dp = MagicMock()

    def get_ohlcv(instruments, start_date, end_date):
        frames = []
        for inst in instruments:
            for date_str, price in close_prices.get(inst, {}).items():
                frames.append({
                    "instrument": inst,
                    "date": pd.Timestamp(date_str),
                    "open": price, "close": price,
                    "prev_close": price * 0.99,
                })
        if not frames:
            return pd.DataFrame(columns=["open", "close", "prev_close"])
        return pd.DataFrame(frames).set_index(["instrument", "date"])

    dp.get_ohlcv.side_effect = get_ohlcv
    dp.get_atr.return_value = 1.0
    return dp


_DEFAULT_CONFIG = {
    "name": "turtle_test",
    "selection": {"universe": "all"},
    "turtle": {
        "entry_lookback": 4,   # 20 日突破，测试用 4
        "exit_lookback": 2,    # 10 日低点，测试用 2
        "atr_period": 4,
        "stop_atr": 2.0,
        "position_size": 0.02,
    },
}


# ── 基本属性 ──────────────────────────────────────────────────────────────────

def test_turtle_is_rule_strategy():
    s = TurtleStrategy(_DEFAULT_CONFIG)
    assert isinstance(s, RuleStrategy)
    assert s.strategy_type == "rule"


# ── 入场逻辑 ──────────────────────────────────────────────────────────────────

def test_turtle_buy_on_20day_breakout():
    s = TurtleStrategy(_DEFAULT_CONFIG)
    date = pd.Timestamp("2024-01-08")
    dp = _make_dp({"SZ000001": {
        "2024-01-02": 10.0,
        "2024-01-03": 10.5,
        "2024-01-04": 11.0,
        "2024-01-05": 11.5,
        "2024-01-08": 11.6,   # > 11.5 (4日最高)
    }})
    signals = s.on_bar(date, ["SZ000001"], {}, dp)
    buys = [sg for sg in signals if sg.action == "buy"]
    assert len(buys) == 1
    assert buys[0].instrument == "SZ000001"


def test_turtle_no_buy_below_breakout():
    s = TurtleStrategy(_DEFAULT_CONFIG)
    date = pd.Timestamp("2024-01-08")
    dp = _make_dp({"SZ000001": {
        "2024-01-02": 10.0,
        "2024-01-03": 10.5,
        "2024-01-04": 11.0,
        "2024-01-05": 11.5,
        "2024-01-08": 11.3,   # < 11.5，不突破
    }})
    signals = s.on_bar(date, ["SZ000001"], {}, dp)
    buys = [sg for sg in signals if sg.action == "buy"]
    assert len(buys) == 0


# ── 出场逻辑（10 日低点出场）─────────────────────────────────────────────────

def test_turtle_sell_on_10day_low():
    """价格跌破 N 日低点，持仓应卖出"""
    s = TurtleStrategy(_DEFAULT_CONFIG)
    date = pd.Timestamp("2024-01-08")
    dp = _make_dp({"SZ000001": {
        "2024-01-05": 11.0,
        "2024-01-08": 10.4,   # < 10日低点（测试用2日低点=11.0），正常场景
    }})
    # 设置 exit_lookback=2，过去2日最低 close = 11.0，今日 10.4 < 11.0 → 出场
    dp.get_ohlcv.side_effect = None
    def get_ohlcv_exit(instruments, start_date, end_date):
        rows = [
            {"instrument": "SZ000001", "date": pd.Timestamp("2024-01-05"), "open": 11.0, "close": 11.0, "prev_close": 10.9},
            {"instrument": "SZ000001", "date": pd.Timestamp("2024-01-08"), "open": 10.4, "close": 10.4, "prev_close": 11.0},
        ]
        return pd.DataFrame(rows).set_index(["instrument", "date"])
    dp.get_ohlcv.side_effect = get_ohlcv_exit

    positions = {
        "SZ000001": PositionState(
            instrument="SZ000001",
            entry_price=12.0,
            stop_loss=8.0,
            layers=1, layer_prices=[12.0], units=100.0,
        )
    }
    signals = s.on_bar(date, ["SZ000001"], positions, dp)
    sells = [sg for sg in signals if sg.action == "sell"]
    assert len(sells) == 1
    assert sells[0].reason == "exit_low"


def test_turtle_no_sell_above_exit_threshold():
    s = TurtleStrategy(_DEFAULT_CONFIG)
    date = pd.Timestamp("2024-01-08")

    dp = MagicMock()
    dp.get_atr.return_value = 1.0

    def get_ohlcv_hold(instruments, start_date, end_date):
        rows = [
            {"instrument": "SZ000001", "date": pd.Timestamp("2024-01-05"), "open": 11.0, "close": 11.0, "prev_close": 10.9},
            {"instrument": "SZ000001", "date": pd.Timestamp("2024-01-08"), "open": 12.0, "close": 12.0, "prev_close": 11.0},
        ]
        return pd.DataFrame(rows).set_index(["instrument", "date"])
    dp.get_ohlcv.side_effect = get_ohlcv_hold

    positions = {
        "SZ000001": PositionState(
            instrument="SZ000001",
            entry_price=10.0, stop_loss=8.0,
            layers=1, layer_prices=[10.0], units=100.0,
        )
    }
    signals = s.on_bar(date, ["SZ000001"], positions, dp)
    sells = [sg for sg in signals if sg.action == "sell"]
    assert len(sells) == 0


# ── ATR 止损 ──────────────────────────────────────────────────────────────────

def test_turtle_atr_stop_loss_triggers():
    """价格跌破 ATR 止损线应卖出"""
    s = TurtleStrategy(_DEFAULT_CONFIG)
    date = pd.Timestamp("2024-01-08")

    dp = MagicMock()
    dp.get_atr.return_value = 1.0

    def get_ohlcv_stop(instruments, start_date, end_date):
        rows = [
            {"instrument": "SZ000001", "date": pd.Timestamp("2024-01-05"), "open": 10.0, "close": 10.0, "prev_close": 9.9},
            {"instrument": "SZ000001", "date": pd.Timestamp("2024-01-08"), "open": 7.9, "close": 7.9, "prev_close": 10.0},
        ]
        return pd.DataFrame(rows).set_index(["instrument", "date"])
    dp.get_ohlcv.side_effect = get_ohlcv_stop

    positions = {
        "SZ000001": PositionState(
            instrument="SZ000001",
            entry_price=10.0,
            stop_loss=8.0,   # 7.9 < 8.0 → 触发 ATR 止损
            layers=1, layer_prices=[10.0], units=100.0,
        )
    }
    signals = s.on_bar(date, ["SZ000001"], positions, dp)
    sells = [sg for sg in signals if sg.action == "sell"]
    assert len(sells) == 1
    assert sells[0].reason == "atr_stop"
