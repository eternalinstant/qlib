from abc import ABC, abstractmethod
from typing import List
import pandas as pd


class DataProvider(ABC):

    @abstractmethod
    def get_ohlcv(
        self,
        instruments: List[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """返回 MultiIndex(instrument, date) 的 OHLCV DataFrame"""
        pass

    @abstractmethod
    def get_factor(
        self,
        instruments: List[str],
        factor: str,
        start_date: str,
        end_date: str,
    ) -> pd.Series:
        """返回 MultiIndex(instrument, date) 的因子值 Series"""
        pass

    @abstractmethod
    def get_universe(
        self,
        date: pd.Timestamp,
        universe: str = "csi800",
    ) -> List[str]:
        """返回指定日期的合法股票池"""
        pass

    def get_atr(
        self,
        instrument: str,
        date: pd.Timestamp,
        period: int = 20,
    ) -> float:
        """ATR 计算，基于 OHLCV close-based 简化版（high/low 可能损坏）"""
        lookback_start = date - pd.Timedelta(days=period * 2)
        df = self.get_ohlcv([instrument], str(lookback_start.date()), str(date.date()))
        if df.empty:
            return 0.0
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(instrument, level="instrument") if instrument in df.index.get_level_values("instrument") else df
        close = df["close"].dropna().tail(period + 1)
        if len(close) < 2:
            return 0.0
        tr = close.diff().abs().dropna()
        return float(tr.mean())
