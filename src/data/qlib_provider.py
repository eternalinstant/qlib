from typing import List
import pandas as pd
from data.provider import DataProvider


class QlibDataProvider(DataProvider):
    """基于现有 qlib + parquet 双源的数据提供者"""

    def get_ohlcv(self, instruments: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        from data.quotes import load_raw_trade_quotes
        return load_raw_trade_quotes(instruments, start_date, end_date)

    def get_factor(self, instruments: List[str], factor: str, start_date: str, end_date: str) -> pd.Series:
        from data.qlib_init import load_features_safe
        try:
            df = load_features_safe(instruments, [f"${factor}"], start_date, end_date)
            if df is None or df.empty:
                return pd.Series(dtype=float)
            col = df.columns[0]
            return df[col].rename(factor)
        except Exception:
            return pd.Series(dtype=float)

    def get_universe(self, date: pd.Timestamp, universe: str = "csi800") -> List[str]:
        from data.universe import filter_instruments
        try:
            return list(filter_instruments(universe, date=date))
        except Exception:
            return []
