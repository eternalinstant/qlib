"""交易日历加载。

公共库：从 qlib provider 的 calendars/day.txt 读取交易日并按区间过滤。
原驻 modules/backtest/common.py，五层迁移下沉至此。
"""
from functools import lru_cache

import pandas as pd

from common.config import CONFIG
from common.platform import resolve_path


@lru_cache(maxsize=32)
def load_trade_calendar(start_date: str, end_date: str) -> pd.DatetimeIndex:
    qlib_root = resolve_path(
        CONFIG.get(
            "paths.data.qlib_data",
            CONFIG.get("qlib_data_path", "~/code/qlib/data/qlib_data/cn_data"),
        )
    )
    cal_file = qlib_root / "calendars" / "day.txt"
    cal = pd.read_csv(cal_file, header=None, names=["date"], parse_dates=["date"])["date"]
    mask = (cal >= pd.Timestamp(start_date)) & (cal <= pd.Timestamp(end_date))
    return pd.DatetimeIndex(cal.loc[mask].tolist())
