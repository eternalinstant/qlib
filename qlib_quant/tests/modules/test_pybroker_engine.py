"""
PyBroker 回测引擎测试
"""

from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from pybroker.common import FeeInfo, PriceType

from modules.backtest.pybroker_engine import (
    PyBrokerBacktestEngine,
    _build_close_only_price_frame,
    make_exec_fn,
    make_fee_fn,
)


class DummyContext:
    def __init__(self, dt, symbol, target_shares, current_shares=0):
        self.dt = dt
        self.symbol = symbol
        self._target_shares = target_shares
        self._current_shares = current_shares
        self.buy_shares = None
        self.sell_shares = None
        self.buy_fill_price = None
        self.sell_fill_price = None

    def calc_target_shares(self, target_pct):
        return self._target_shares

    def long_pos(self):
        if self._current_shares <= 0:
            return None
        return SimpleNamespace(shares=self._current_shares)


def test_make_exec_fn_sets_close_fill_price_for_buy_and_sell():
    exec_fn = make_exec_fn(
        {pd.Timestamp("2026-03-03"): {"SH600000"}},
        {pd.Timestamp("2026-03-03")},
        controller=None,
    )

    buy_ctx = DummyContext("2026-03-03", "SH600000", target_shares=1000, current_shares=0)
    exec_fn(buy_ctx)
    assert buy_ctx.buy_shares == 1000
    assert buy_ctx.buy_fill_price == PriceType.CLOSE

    sell_ctx = DummyContext("2026-03-03", "SH600001", target_shares=0, current_shares=500)
    exec_fn(sell_ctx)
    assert sell_ctx.sell_shares == 500
    assert sell_ctx.sell_fill_price == PriceType.CLOSE


def test_make_fee_fn_uses_strategy_trading_cost():
    fee_fn = make_fee_fn(
        {
            "buy_commission_rate": 0.0005,
            "sell_commission_rate": 0.0004,
            "sell_stamp_tax_rate": 0.001,
            "min_buy_commission": 5.0,
            "min_sell_commission": 6.0,
            "slippage_bps": 5,
            "impact_bps": 7,
        }
    )

    buy_fee = fee_fn(FeeInfo("SH600000", Decimal("100"), Decimal("10"), "buy"))
    sell_fee = fee_fn(FeeInfo("SH600000", Decimal("100"), Decimal("10"), "sell"))

    assert buy_fee == Decimal("6.2000")
    assert sell_fee == Decimal("8.2000")


def test_build_close_only_price_frame_uses_close_for_ohlc():
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2026-03-03"), "SH600000"),
            (pd.Timestamp("2026-03-04"), "SH600000"),
        ],
        names=["datetime", "instrument"],
    )
    df = pd.DataFrame({"close": [10.0, 10.5], "volume": [1000.0, 1200.0]}, index=idx)

    result = _build_close_only_price_frame(df)

    assert list(result.columns) == ["symbol", "date", "open", "high", "low", "close", "volume"]
    assert (result["open"] == result["close"]).all()
    assert (result["high"] == result["close"]).all()
    assert (result["low"] == result["close"]).all()


def test_build_result_saves_with_explicit_strategy_slug(tmp_path):
    engine = PyBrokerBacktestEngine()
    portfolio = pd.DataFrame(
        {"equity": [100000.0, 101000.0, 102000.0]},
        index=pd.to_datetime(["2026-03-01", "2026-03-02", "2026-03-03"]),
    )
    positions = pd.DataFrame(
        {
            "symbol": ["SH600000"],
            "date": [pd.Timestamp("2026-03-03")],
            "shares": [100],
        }
    ).set_index(["symbol", "date"])
    trades = pd.DataFrame({"symbol": ["SH600000"], "shares": [100], "price": [10.0]})
    result_obj = SimpleNamespace(portfolio=portfolio, positions=positions, trades=trades)

    fake_config = SimpleNamespace(
        get=lambda key, default=None: str(tmp_path) if key == "paths.results" else default
    )

    with patch("modules.backtest.pybroker_engine.CONFIG", fake_config), \
         patch("modules.backtest.pybroker_engine.get_name_map", return_value={"SH600000": "浦发银行"}):
        result = engine._build_result(
            result_obj,
            initial_cash=100000.0,
            strategy_name="demo",
            strategy_slug="fixed__portfolio__demo",
        )

    saved = Path(result.metadata["results_file"])
    assert saved.exists()
    assert "fixed__portfolio__demo" in saved.name
