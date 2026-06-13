"""TDD: core/data_provider.py + core/qlib_data_provider.py"""
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from core.data_provider import DataProvider
from core.qlib_data_provider import QlibDataProvider


# ── DataProvider ABC ──────────────────────────────────────────────────────────

def test_data_provider_is_abstract():
    with pytest.raises(TypeError):
        DataProvider()


def test_data_provider_subclass_must_implement_get_ohlcv():
    class Incomplete(DataProvider):
        def get_factor(self, instruments, factor, start_date, end_date):
            return pd.Series(dtype=float)
        def get_universe(self, date, universe="csi800"):
            return []
    with pytest.raises(TypeError):
        Incomplete()


def test_data_provider_subclass_must_implement_get_factor():
    class Incomplete(DataProvider):
        def get_ohlcv(self, instruments, start_date, end_date):
            return pd.DataFrame()
        def get_universe(self, date, universe="csi800"):
            return []
    with pytest.raises(TypeError):
        Incomplete()


def test_data_provider_subclass_must_implement_get_universe():
    class Incomplete(DataProvider):
        def get_ohlcv(self, instruments, start_date, end_date):
            return pd.DataFrame()
        def get_factor(self, instruments, factor, start_date, end_date):
            return pd.Series(dtype=float)
    with pytest.raises(TypeError):
        Incomplete()


def test_data_provider_concrete_subclass_instantiates():
    class Complete(DataProvider):
        def get_ohlcv(self, instruments, start_date, end_date):
            return pd.DataFrame()
        def get_factor(self, instruments, factor, start_date, end_date):
            return pd.Series(dtype=float)
        def get_universe(self, date, universe="csi800"):
            return []
    dp = Complete()
    assert dp is not None


# ── get_atr 默认实现 ──────────────────────────────────────────────────────────

def test_get_atr_returns_float():
    class FakeDP(DataProvider):
        def get_ohlcv(self, instruments, start_date, end_date):
            dates = pd.date_range("2024-01-01", periods=30)
            idx = pd.MultiIndex.from_product([["SH600000"], dates], names=["instrument", "date"])
            df = pd.DataFrame({
                "open":  [10.0] * 30,
                "high":  [10.5] * 30,
                "low":   [9.5]  * 30,
                "close": [10.0] * 30,
                "volume":[1e6]  * 30,
            }, index=idx)
            return df
        def get_factor(self, instruments, factor, start_date, end_date):
            return pd.Series(dtype=float)
        def get_universe(self, date, universe="csi800"):
            return []

    dp = FakeDP()
    atr = dp.get_atr("SH600000", pd.Timestamp("2024-02-15"), period=20)
    assert isinstance(atr, float)
    assert atr >= 0


def test_get_atr_uses_true_range_with_high_low():
    """有 high/low 时用真 ATR：TR = max(H-L, |H-prevC|, |L-prevC|)，ATR = TR 均值"""
    class FakeDP(DataProvider):
        def get_ohlcv(self, instruments, start_date, end_date):
            dates = pd.date_range("2024-01-01", periods=5)
            idx = pd.MultiIndex.from_product([["X"], dates], names=["instrument", "date"])
            return pd.DataFrame({
                "high":  [10.5, 11.5, 12.5, 11.5, 13.5],
                "low":   [9.5,  10.5, 11.0, 10.5, 12.5],
                "close": [10.0, 11.0, 12.0, 11.0, 13.0],
            }, index=idx)
        def get_factor(self, i, f, s, e):
            return pd.Series(dtype=float)
        def get_universe(self, date, universe="csi800"):
            return []

    dp = FakeDP()
    atr = dp.get_atr("X", pd.Timestamp("2024-01-05"), period=4)
    # TR: d2=1.5, d3=1.5, d4=1.5, d5=2.5 → ATR=1.75（旧 close-based 误给 1.25）
    assert atr == pytest.approx(1.75, rel=1e-9)


def test_get_atr_falls_back_to_close_when_no_high_low():
    """无 high/low 列时退化为 close-based 近似（兼容 mock 数据源）"""
    class FakeDP(DataProvider):
        def get_ohlcv(self, instruments, start_date, end_date):
            dates = pd.date_range("2024-01-01", periods=5)
            idx = pd.MultiIndex.from_product([["X"], dates], names=["instrument", "date"])
            return pd.DataFrame({"close": [10.0, 11.0, 12.0, 11.0, 13.0]}, index=idx)
        def get_factor(self, i, f, s, e):
            return pd.Series(dtype=float)
        def get_universe(self, date, universe="csi800"):
            return []

    dp = FakeDP()
    atr = dp.get_atr("X", pd.Timestamp("2024-01-05"), period=4)
    # close-based: mean(|Δclose|)=mean(1,1,1,2)=1.25
    assert atr == pytest.approx(1.25, rel=1e-9)


# ── QlibDataProvider ──────────────────────────────────────────────────────────

def test_qlib_data_provider_is_data_provider():
    assert issubclass(QlibDataProvider, DataProvider)


def test_qlib_data_provider_get_ohlcv_returns_dataframe():
    dp = QlibDataProvider()
    with patch("modules.backtest.common.load_raw_trade_quotes") as mock_load:
        mock_load.return_value = pd.DataFrame(
            {"open": [10.0], "close": [10.0], "prev_close": [9.9]},
            index=pd.MultiIndex.from_tuples(
                [("SH600000", pd.Timestamp("2024-01-02"))],
                names=["instrument", "date"],
            ),
        )
        result = dp.get_ohlcv(["SH600000"], "2024-01-01", "2024-01-31")
    assert isinstance(result, pd.DataFrame)
    assert "close" in result.columns


def test_qlib_data_provider_get_universe_returns_list():
    dp = QlibDataProvider()
    with patch("core.universe.filter_instruments") as mock_filter:
        mock_filter.return_value = ["SH600000", "SZ000001"]
        result = dp.get_universe(pd.Timestamp("2024-01-02"), universe="csi800")
    assert isinstance(result, list)
    assert len(result) == 2
