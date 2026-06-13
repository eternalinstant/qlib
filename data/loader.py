"""
数据加载抽象层
提供统一的数据加载接口，支持多数据源合并
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

import pandas as pd

from config.config import CONFIG
from core.factors import FactorRegistry
from core.universe import filter_instruments


class DataLoader(ABC):
    """数据加载器抽象基类"""

    @abstractmethod
    def load(
        self,
        instruments: List[str],
        fields: List[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """加载数据

        Returns
        -------
        pd.DataFrame
            MultiIndex (datetime, instrument) DataFrame
        """
        ...


class QlibDataLoader(DataLoader):
    """Qlib 数据加载器"""

    def __init__(self, provider_uri: str = None, region: str = "cn"):
        self.provider_uri = provider_uri or CONFIG.get("qlib_data_path")
        self.region = region
        self._initialized = False

    def _ensure_qlib(self):
        if self._initialized:
            return

        import qlib
        from qlib.config import REG_CN

        provider_uri = str(Path(self.provider_uri).expanduser())
        qlib.init(provider_uri=provider_uri, region=REG_CN if self.region == "cn" else REG_CN)
        self._initialized = True

    def load(
        self,
        instruments: List[str],
        fields: List[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        self._ensure_qlib()
        from qlib.data import D

        df = D.features(instruments, fields, start_date, end_date, "day")
        return df


class ParquetDataLoader(DataLoader):
    """Parquet 文件数据加载器"""

    def __init__(self, parquet_path: str = None):
        if parquet_path is None:
            parquet_path = CONFIG.get("paths.qlib_data", "~/code/qlib/data/qlib_data/cn_data")
        self.parquet_path = Path(parquet_path).expanduser() / "factor_data.parquet"

    def load(
        self,
        instruments: List[str],
        fields: List[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        if not self.parquet_path.exists():
            return pd.DataFrame()

        df = pd.read_parquet(self.parquet_path)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df[(df["datetime"] >= start_date) & (df["datetime"] <= end_date)]

        if instruments:
            df = df[df["instrument"].isin(instruments)]

        available_fields = [f for f in fields if f in df.columns]
        if not available_fields:
            return pd.DataFrame()

        df = df[["datetime", "instrument"] + available_fields]
        df = df.set_index(["datetime", "instrument"])
        return df[available_fields]


class CompositeDataLoader(DataLoader):
    """组合数据加载器 - 多源合并 + 日期/instrument 对齐"""

    def __init__(
        self,
        loaders: List[DataLoader],
        join_how: str = "left",
    ):
        self.loaders = loaders
        self.join_how = join_how

    def load(
        self,
        instruments: List[str],
        fields: List[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        dfs = []
        for loader in self.loaders:
            df = loader.load(instruments, fields, start_date, end_date)
            if not df.empty:
                dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        result = dfs[0]
        for df in dfs[1:]:
            result = result.join(df, how=self.join_how)

        return result


def create_data_loader(source: str, **kwargs) -> DataLoader:
    """工厂函数：创建数据加载器

    Parameters
    ----------
    source : str
        数据源: "qlib" | "parquet"
    **kwargs
        传递给加载器的参数

    Returns
    -------
    DataLoader
    """
    if source == "qlib":
        return QlibDataLoader(**kwargs)
    elif source == "parquet":
        return ParquetDataLoader(**kwargs)
    else:
        raise ValueError(f"未知数据源: {source}")
