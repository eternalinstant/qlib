"""端到端：RuleBasedEngine + PyramidStrategy / TurtleStrategy mock 验证"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from strategies.pyramid.pyramid_strategy import PyramidStrategy
from strategies.turtle.turtle_strategy import TurtleStrategy
from modules.backtest.rule_engine import RuleBasedEngine
from modules.backtest.base import BacktestResult


# ── Mock DataProvider ────────────────────────────────────────────────────────

def _build_price_series(start_price=10.0, n_days=60, trend=0.002):
    """生成简单趋势价格序列（每日上涨 trend%）"""
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    prices = [start_price * (1 + trend) ** i for i in range(n_days)]
    return dict(zip([str(d.date()) for d in dates], prices))


UNIVERSE = ["SZ000001", "SZ000002", "SZ000003"]

# n_days 需覆盖整个回测区间（2024-01-02 ~ 2024-03-29 约 64 个工作日），
# 否则末尾交易日无价格 → 持仓市值漏算 → NAV 失真
_PRICE_DATA = {
    inst: _build_price_series(start_price=10.0 + i, n_days=80, trend=0.003)
    for i, inst in enumerate(UNIVERSE)
}


def _make_full_dp():
    dp = MagicMock()

    def get_universe(date, universe_name="csi800"):
        return UNIVERSE

    def get_ohlcv(instruments, start_date, end_date):
        rows = []
        for inst in instruments:
            prices = _PRICE_DATA.get(inst, {})
            for date_str, price in prices.items():
                d = pd.Timestamp(date_str)
                if start_date <= str(d.date()) <= end_date:
                    rows.append({
                        "instrument": inst,
                        "date": d,
                        "open": price, "close": price,
                        "prev_close": price * 0.99,
                    })
        if not rows:
            return pd.DataFrame(columns=["open", "close", "prev_close"])
        return pd.DataFrame(rows).set_index(["instrument", "date"])

    def get_atr(instrument, date, period=20):
        prices = list(_PRICE_DATA.get(instrument, {}).values())
        if len(prices) < 2:
            return 0.1
        diffs = [abs(prices[i] - prices[i - 1]) for i in range(1, min(period + 1, len(prices)))]
        return sum(diffs) / len(diffs) if diffs else 0.1

    dp.get_universe.side_effect = get_universe
    dp.get_ohlcv.side_effect = get_ohlcv
    dp.get_atr.side_effect = get_atr
    return dp


# ── 测试 ─────────────────────────────────────────────────────────────────────

_PYRAMID_CONFIG = {
    "name": "e2e_pyramid",
    "selection": {"universe": "csi800"},
    "pyramid": {
        "entry_lookback": 5,
        "atr_period": 5,
        "add_factor": 1.0,
        "max_layers": 3,
        "stop_atr": 2.0,
        "position_size": 0.1,
    },
    "initial_capital": 100_000,
    "start_date": "2024-01-02",
    "end_date": "2024-03-29",
}

_TURTLE_CONFIG = {
    "name": "e2e_turtle",
    "selection": {"universe": "csi800"},
    "turtle": {
        "entry_lookback": 5,
        "exit_lookback": 3,
        "atr_period": 5,
        "stop_atr": 2.0,
        "position_size": 0.1,
    },
    "initial_capital": 100_000,
    "start_date": "2024-01-02",
    "end_date": "2024-03-29",
}


def _run_with_mock_dp(config, strategy_cls):
    """通过真实 RuleBasedEngine.run() 跑端到端，patch 数据源与交易日历为合成数据。"""
    strategy = strategy_cls(config)
    mock_dp = _make_full_dp()
    trade_dates = pd.bdate_range(config["start_date"], config["end_date"])

    engine = RuleBasedEngine(config)
    with patch("modules.backtest.rule_engine.QlibDataProvider", return_value=mock_dp):
        with patch("modules.backtest.rule_engine.load_trade_calendar", return_value=trade_dates):
            result = engine.run(strategy)

    final_value = float(config["initial_capital"]) * float(result.portfolio_value.iloc[-1])
    return result.daily_returns, final_value


def test_pyramid_e2e_runs_without_error():
    returns, final_value = _run_with_mock_dp(_PYRAMID_CONFIG, PyramidStrategy)
    assert len(returns) > 0
    assert final_value > 0
    assert not returns.isna().any()


def test_turtle_e2e_runs_without_error():
    returns, final_value = _run_with_mock_dp(_TURTLE_CONFIG, TurtleStrategy)
    assert len(returns) > 0
    assert final_value > 0
    assert not returns.isna().any()


def test_pyramid_e2e_on_uptrend_gains_value():
    """趋势上涨行情下金字塔策略最终净值 >= 初始资金"""
    returns, final_value = _run_with_mock_dp(_PYRAMID_CONFIG, PyramidStrategy)
    initial = float(_PYRAMID_CONFIG["initial_capital"])
    assert final_value >= initial * 0.9, f"final_value={final_value:.0f} 低于 90% 初始资金"


def test_turtle_e2e_on_uptrend_gains_value():
    """趋势上涨行情下海龟策略最终净值 >= 初始资金"""
    returns, final_value = _run_with_mock_dp(_TURTLE_CONFIG, TurtleStrategy)
    initial = float(_TURTLE_CONFIG["initial_capital"])
    assert final_value >= initial * 0.9, f"final_value={final_value:.0f} 低于 90% 初始资金"


def test_rule_engine_result_is_backtest_result():
    """RuleBasedEngine.run() 返回 BacktestResult"""
    from modules.backtest.common import load_trade_calendar
    with patch("modules.backtest.rule_engine.QlibDataProvider") as MockDP:
        mock_dp = _make_full_dp()
        MockDP.return_value = mock_dp
        with patch("modules.backtest.rule_engine.load_trade_calendar") as mock_cal:
            mock_cal.return_value = pd.bdate_range("2024-01-02", "2024-03-29")
            engine = RuleBasedEngine(config=_PYRAMID_CONFIG)
            strategy = PyramidStrategy(_PYRAMID_CONFIG)
            result = engine.run(strategy)

    assert isinstance(result, BacktestResult)
    assert len(result.daily_returns) > 0
    assert result.metadata["strategy"] == "e2e_pyramid"
