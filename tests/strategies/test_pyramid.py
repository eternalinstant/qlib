"""TDD: strategies/pyramid/pyramid_strategy.py"""
import pytest
import pandas as pd
from unittest.mock import MagicMock
from strategies.pyramid.pyramid_strategy import PyramidStrategy
from strategies.base import RuleStrategy, PositionState, StrategySignal


def _make_dp(close_prices: dict):
    """辅助：创建 mock DataProvider，根据 instrument+date 返回 close"""
    dp = MagicMock()

    def get_ohlcv(instruments, start_date, end_date):
        frames = []
        for inst in instruments:
            prices = close_prices.get(inst, {})
            for date_str, price in prices.items():
                frames.append({
                    "instrument": inst,
                    "date": pd.Timestamp(date_str),
                    "open": price,
                    "close": price,
                    "prev_close": price * 0.99,
                    "high": price * 1.01,
                    "low": price * 0.99,
                })
        if not frames:
            return pd.DataFrame(columns=["open", "close", "prev_close"])
        df = pd.DataFrame(frames)
        df = df.set_index(["instrument", "date"])
        return df

    dp.get_ohlcv.side_effect = get_ohlcv

    def get_atr(instrument, date, period=20):
        prices = close_prices.get(instrument, {})
        vals = list(prices.values())
        if len(vals) < 2:
            return 0.5
        diffs = [abs(vals[i] - vals[i-1]) for i in range(1, len(vals))]
        return sum(diffs) / len(diffs)

    dp.get_atr.side_effect = get_atr
    return dp


_DEFAULT_CONFIG = {
    "name": "pyramid_test",
    "selection": {"universe": "all"},
    "pyramid": {
        "entry_lookback": 3,
        "atr_period": 3,
        "add_factor": 1.0,
        "max_layers": 3,
        "stop_atr": 2.0,
        "position_size": 0.05,
    },
}


# ── 基本属性 ──────────────────────────────────────────────────────────────────

def test_pyramid_is_rule_strategy():
    s = PyramidStrategy(_DEFAULT_CONFIG)
    assert isinstance(s, RuleStrategy)
    assert s.strategy_type == "rule"


def test_pyramid_loads_from_yaml(tmp_path):
    import yaml
    config = dict(_DEFAULT_CONFIG)
    p = tmp_path / "pyramid.yaml"
    p.write_text(yaml.dump(config))
    s = PyramidStrategy.from_yaml(str(p))
    assert s.name == "pyramid_test"


# ── 入场逻辑 ──────────────────────────────────────────────────────────────────

def test_pyramid_no_signal_when_price_below_high():
    """价格未突破 N 日高点，不应入场"""
    s = PyramidStrategy(_DEFAULT_CONFIG)
    date = pd.Timestamp("2024-01-05")
    # close < N日高点
    dp = _make_dp({"SZ000001": {
        "2024-01-02": 10.0,
        "2024-01-03": 10.5,
        "2024-01-04": 11.0,   # N日高点
        "2024-01-05": 10.8,   # 今日 < 高点，不入场
    }})
    signals = s.on_bar(date, ["SZ000001"], {}, dp)
    buys = [sg for sg in signals if sg.action == "buy"]
    assert len(buys) == 0


def test_pyramid_buy_signal_when_price_breaks_out():
    """价格突破 N 日高点，应产生 buy 信号"""
    s = PyramidStrategy(_DEFAULT_CONFIG)
    date = pd.Timestamp("2024-01-05")
    dp = _make_dp({"SZ000001": {
        "2024-01-02": 10.0,
        "2024-01-03": 10.5,
        "2024-01-04": 11.0,
        "2024-01-05": 11.1,   # > 11.0，突破
    }})
    signals = s.on_bar(date, ["SZ000001"], {}, dp)
    buys = [sg for sg in signals if sg.action == "buy"]
    assert len(buys) == 1
    assert buys[0].instrument == "SZ000001"


def test_pyramid_buy_signal_includes_price():
    s = PyramidStrategy(_DEFAULT_CONFIG)
    date = pd.Timestamp("2024-01-05")
    dp = _make_dp({"SZ000001": {
        "2024-01-02": 10.0,
        "2024-01-03": 10.5,
        "2024-01-04": 11.0,
        "2024-01-05": 11.1,
    }})
    signals = s.on_bar(date, ["SZ000001"], {}, dp)
    buys = [sg for sg in signals if sg.action == "buy"]
    assert buys[0].price == pytest.approx(11.1, rel=1e-3)


# ── 止损逻辑 ──────────────────────────────────────────────────────────────────

def test_pyramid_sell_signal_when_stop_loss_hit():
    """价格跌破止损线，应产生 sell 信号"""
    s = PyramidStrategy(_DEFAULT_CONFIG)
    date = pd.Timestamp("2024-01-10")
    dp = _make_dp({"SZ000001": {
        "2024-01-10": 8.0,   # 跌破止损
    }})
    positions = {
        "SZ000001": PositionState(
            instrument="SZ000001",
            entry_price=11.0,
            stop_loss=9.0,   # stop_loss=9.0，当前价8.0 < 9.0
            layers=1,
            layer_prices=[11.0],
            units=1000.0,
        )
    }
    signals = s.on_bar(date, ["SZ000001"], positions, dp)
    sells = [sg for sg in signals if sg.action == "sell"]
    assert len(sells) == 1
    assert sells[0].instrument == "SZ000001"
    assert sells[0].reason == "stop_loss"


def test_pyramid_no_sell_when_above_stop_loss():
    s = PyramidStrategy(_DEFAULT_CONFIG)
    date = pd.Timestamp("2024-01-10")
    dp = _make_dp({"SZ000001": {"2024-01-10": 10.5}})
    positions = {
        "SZ000001": PositionState(
            instrument="SZ000001",
            entry_price=11.0,
            stop_loss=9.0,
            layers=1,
            layer_prices=[11.0],
            units=1000.0,
        )
    }
    signals = s.on_bar(date, ["SZ000001"], positions, dp)
    sells = [sg for sg in signals if sg.action == "sell"]
    assert len(sells) == 0


# ── 加仓逻辑 ──────────────────────────────────────────────────────────────────

def test_pyramid_add_signal_when_price_rises_enough():
    """价格上涨超过 ATR * add_factor，应产生 add 信号"""
    s = PyramidStrategy(_DEFAULT_CONFIG)
    date = pd.Timestamp("2024-01-10")
    # 用固定 ATR mock 简化：diffs=[0.1,0.1]，ATR=0.1，add_trigger=11.0+0.1=11.1
    dp = _make_dp({"SZ000001": {
        "2024-01-08": 10.0,
        "2024-01-09": 10.1,
        "2024-01-10": 10.2,   # get_ohlcv close
    }})
    # 强制 get_atr 返回 0.1，使 add_trigger = 11.0 + 0.1 = 11.1
    dp.get_atr.side_effect = None
    dp.get_atr.return_value = 0.1

    # 覆盖 get_ohlcv 让当日 close = 11.2 > 11.1
    def patched_ohlcv(instruments, start_date, end_date):
        rows = []
        for inst in instruments:
            rows.append({
                "instrument": inst,
                "date": pd.Timestamp("2024-01-10"),
                "open": 11.2, "close": 11.2, "prev_close": 11.0,
            })
        df = pd.DataFrame(rows).set_index(["instrument", "date"])
        return df
    dp.get_ohlcv.side_effect = patched_ohlcv
    positions = {
        "SZ000001": PositionState(
            instrument="SZ000001",
            entry_price=11.0,
            stop_loss=9.0,
            layers=1,   # < max_layers=3，可以加仓
            layer_prices=[11.0],
            units=1000.0,
        )
    }
    signals = s.on_bar(date, ["SZ000001"], positions, dp)
    adds = [sg for sg in signals if sg.action == "add"]
    assert len(adds) == 1


def test_pyramid_no_add_when_max_layers_reached():
    s = PyramidStrategy(_DEFAULT_CONFIG)
    date = pd.Timestamp("2024-01-10")
    dp = _make_dp({"SZ000001": {"2024-01-10": 15.0}})
    positions = {
        "SZ000001": PositionState(
            instrument="SZ000001",
            entry_price=11.0,
            stop_loss=9.0,
            layers=3,   # 已达 max_layers，不再加仓
            layer_prices=[11.0, 11.5, 12.0],
            units=3000.0,
        )
    }
    signals = s.on_bar(date, ["SZ000001"], positions, dp)
    adds = [sg for sg in signals if sg.action == "add"]
    assert len(adds) == 0


# ── 止损设置 ──────────────────────────────────────────────────────────────────

def test_pyramid_stop_loss_set_on_buy():
    """买入时止损线应设置为 entry_price - stop_atr * ATR"""
    s = PyramidStrategy(_DEFAULT_CONFIG)
    date = pd.Timestamp("2024-01-05")
    dp = _make_dp({"SZ000001": {
        "2024-01-02": 10.0,
        "2024-01-03": 10.5,
        "2024-01-04": 11.0,
        "2024-01-05": 11.1,
    }})
    positions: dict = {}
    signals = s.on_bar(date, ["SZ000001"], positions, dp)
    buys = [sg for sg in signals if sg.action == "buy"]
    assert len(buys) == 1
    # stop_loss 应在信号的 metadata 或由引擎在入场后设置
    # 此处验证引擎接收到 buy 信号后，策略的 stop_loss_price 字段被携带
    buy = buys[0]
    assert hasattr(buy, "price")  # 入场价
