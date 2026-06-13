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
        """ATR（平均真实波幅）。

        优先用真 ATR：TR = max(H-L, |H-prevClose|, |L-prevClose|)，ATR = TR 的 period 均值。
        当数据源不提供 high/low（如 mock 或损坏的 qlib provider）时，退化为
        close-to-close 绝对变动均值的近似。
        """
        lookback_start = date - pd.Timedelta(days=period * 2)
        df = self.get_ohlcv([instrument], str(lookback_start.date()), str(date.date()))
        if df is None or df.empty:
            return 0.0
        if isinstance(df.index, pd.MultiIndex):
            if instrument in df.index.get_level_values("instrument"):
                df = df.xs(instrument, level="instrument")

        # 真 ATR（数据源提供可用 high/low）
        if "high" in df.columns and "low" in df.columns:
            sub = df[["high", "low", "close"]].dropna().tail(period + 1)
            if len(sub) >= 2:
                prev_close = sub["close"].shift(1)
                tr = pd.concat([
                    sub["high"] - sub["low"],
                    (sub["high"] - prev_close).abs(),
                    (sub["low"] - prev_close).abs(),
                ], axis=1).max(axis=1)
                tr = tr.iloc[1:]  # 首行无 prev_close
                if len(tr) > 0:
                    return float(tr.tail(period).mean())

        # 退化：close-based 近似
        close = df["close"].dropna().tail(period + 1)
        if len(close) < 2:
            return 0.0
        tr = close.diff().abs().dropna()
        return float(tr.mean())
