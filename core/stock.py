"""
股票通用 API

统一三种代码格式的转换，提供名称/行业查询。

格式说明:
    tushare   000001.SZ   Tushare API 原始格式
    qlib      SZ000001    Qlib D.features() 使用的格式
    internal  sz000001    factor_data.parquet 内部存储格式
"""

import functools
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from config.config import CONFIG

# ---------------------------------------------------------------------------
# 格式转换
# ---------------------------------------------------------------------------

def to_tushare(code: str) -> str:
    """任意格式 → Tushare 格式 (000001.SZ)"""
    code = code.strip()

    # 已经是 tushare 格式
    if "." in code:
        parts = code.split(".")
        return f"{parts[0]}.{parts[1].upper()}"

    # qlib / internal: SZ000001 / sz000001
    code_upper = code.upper()
    if code_upper.startswith(("SZ", "SH", "BJ")):
        exchange = code_upper[:2]
        num = code_upper[2:]
        return f"{num}.{exchange}"

    # 纯数字: 按首位判断交易所
    if code.isdigit():
        if code.startswith(("6", "9")):
            return f"{code}.SH"
        elif code.startswith(("0", "2", "3")):
            return f"{code}.SZ"
        elif code.startswith(("4", "8")):
            return f"{code}.BJ"

    return code


def to_qlib(code: str) -> str:
    """任意格式 → Qlib 格式 (SZ000001)"""
    ts = to_tushare(code)
    if "." in ts:
        num, exchange = ts.split(".")
        return f"{exchange.upper()}{num}"
    return code.upper()


def to_internal(code: str) -> str:
    """任意格式 → 内部格式 (sz000001)"""
    return to_qlib(code).lower()


# ---------------------------------------------------------------------------
# 股票信息查询 (带缓存)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _load_stock_basic() -> pd.DataFrame:
    """加载 stock_basic.csv（自动缓存）"""
    candidates = [
        Path(CONFIG.get("paths.data.tushare", "")) / "stock_basic.csv",
        Path("~/code/qlib/data/tushare/stock_basic.csv").expanduser(),
        Path("data/tushare/stock_basic.csv"),
    ]
    for p in candidates:
        p = Path(p).expanduser()
        if p.exists():
            df = pd.read_csv(p, dtype=str)
            df.columns = df.columns.str.strip()
            return df
    return pd.DataFrame(columns=["ts_code", "name", "industry"])


def get_name(code: str) -> Optional[str]:
    """股票代码 → 名称"""
    ts = to_tushare(code)
    df = _load_stock_basic()
    match = df.loc[df["ts_code"] == ts, "name"]
    return match.iloc[0] if len(match) > 0 else None


def get_industry(code: str) -> Optional[str]:
    """股票代码 → 行业"""
    ts = to_tushare(code)
    df = _load_stock_basic()
    match = df.loc[df["ts_code"] == ts, "industry"]
    return match.iloc[0] if len(match) > 0 else None


def search(keyword: str) -> List[Dict[str, str]]:
    """按名称/代码/行业模糊搜索"""
    df = _load_stock_basic()
    if df.empty:
        return []
    mask = (
        df["ts_code"].str.contains(keyword, case=False, na=False)
        | df["name"].str.contains(keyword, na=False)
        | df["industry"].str.contains(keyword, na=False)
    )
    results = df[mask].head(20)
    return results.to_dict("records")


def get_all_codes(fmt: str = "tushare") -> List[str]:
    """获取全部股票代码列表

    Parameters
    ----------
    fmt : 'tushare' | 'qlib' | 'internal'
    """
    df = _load_stock_basic()
    if df.empty:
        return []
    converters = {"tushare": to_tushare, "qlib": to_qlib, "internal": to_internal}
    convert = converters.get(fmt, to_tushare)
    return [convert(c) for c in df["ts_code"]]


def display(code: str) -> str:
    """返回 '名称(代码)' 的可读字符串"""
    name = get_name(code)
    ts = to_tushare(code)
    if name:
        return f"{name}({ts})"
    return ts


def display_list(codes: List[str]) -> List[str]:
    """批量转换为可读字符串"""
    return [display(c) for c in codes]
