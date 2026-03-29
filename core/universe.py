"""
股票池过滤模块
排除指数、北交所、科创板、ST 等不适合策略的标的
"""

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from config.config import CONFIG


EXCLUDED_PREFIXES = (
    "BJ",
    "SH43",
    "SZ43",
    "SH83",
    "SZ83",
    "SH87",
    "SZ87",
    "SH000",  # 上交所指数，例如 SH000300
    "SZ399",  # 深交所指数，例如 SZ399001
    "SH880",  # 常见行业/主题指数
    "SH881",
)
UNIVERSE_ALL = "all"
UNIVERSE_CSI300 = "csi300"
INDEX_CODE_BY_UNIVERSE = {
    UNIVERSE_CSI300: "000300.SH",
}

_st_instruments: set = None
_list_date_map: Dict[str, pd.Timestamp] = None
_list_date_series: pd.Series = None
_historical_st_intervals: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]] = None
_historical_st_loaded: bool = False
_index_weight_df: pd.DataFrame = None
_historical_st_by_date_cache: Dict[str, Tuple[str, ...]] = {}
_newly_listed_by_date_cache: Dict[Tuple[str, int], Tuple[str, ...]] = {}
_index_constituents_as_of_cache: Dict[Tuple[str, str], Tuple[str, ...]] = {}


def _normalize_cache_date(as_of_date) -> str:
    return pd.Timestamp(as_of_date).normalize().strftime("%Y-%m-%d")


def _load_st_set() -> set:
    """从 stock_basic.csv 加载所有 ST / *ST 股票的 instrument 集合（懒加载）"""
    global _st_instruments
    if _st_instruments is not None:
        return _st_instruments

    csv = Path(__file__).parent.parent / "data" / "tushare" / "stock_basic.csv"
    if not csv.exists():
        _st_instruments = set()
        return _st_instruments

    import pandas as pd
    df = pd.read_csv(csv, dtype=str)
    result = set()
    for _, row in df.iterrows():
        name = str(row.get("name", ""))
        if "ST" in name:
            ts_code = str(row["ts_code"])
            if "." in ts_code:
                code, exchange = ts_code.split(".")
                result.add(f"{exchange}{code}")  # SZ300762 格式

    _st_instruments = result
    return _st_instruments


def _load_list_date_map() -> Dict[str, pd.Timestamp]:
    """从 Qlib instruments/all.txt 加载上市日期映射（instrument -> list_date）"""
    global _list_date_map
    if _list_date_map is not None:
        return _list_date_map

    instruments_path = (
        Path(CONFIG.get("paths.qlib_data", "~/code/qlib/data/qlib_data/cn_data"))
        .expanduser()
        / "instruments"
        / "all.txt"
    )
    if not instruments_path.exists():
        _list_date_map = {}
        return _list_date_map

    df = pd.read_csv(
        instruments_path,
        sep="\t",
        header=None,
        names=["instrument", "list_date", "end_date"],
        dtype=str,
    )
    df["instrument"] = df["instrument"].str[:2].str.upper() + df["instrument"].str[2:]
    df["list_date"] = pd.to_datetime(df["list_date"], errors="coerce")
    df = df.dropna(subset=["instrument", "list_date"])

    _list_date_map = dict(zip(df["instrument"], df["list_date"]))
    return _list_date_map


def _load_list_date_series() -> pd.Series:
    """加载上市日期序列（instrument -> list_date），供按日期批量过滤。"""
    global _list_date_series
    if _list_date_series is not None:
        return _list_date_series

    list_date_map = _load_list_date_map()
    if not list_date_map:
        _list_date_series = pd.Series(dtype="datetime64[ns]")
        return _list_date_series

    series = pd.Series(list_date_map, dtype="datetime64[ns]")
    series = series.dropna().sort_index()
    _list_date_series = series
    return _list_date_series


def _iter_namechange_paths() -> List[Path]:
    project_root = Path(__file__).parent.parent
    paths = [
        project_root / "data" / "tushare" / "namechange.parquet",
        project_root / "data" / "tushare" / "namechange.csv",
    ]
    return list(dict.fromkeys(paths))


def _iter_index_weight_paths() -> List[Path]:
    project_root = Path(__file__).parent.parent
    paths = [
        project_root / "data" / "tushare" / "index_weight.parquet",
        project_root / "data" / "tushare" / "index_weight.csv",
    ]
    return list(dict.fromkeys(paths))


def _normalize_ts_code_to_instrument(ts_code: str) -> str:
    ts_code = str(ts_code)
    if "." not in ts_code:
        return ts_code
    code, exchange = ts_code.split(".")
    return f"{exchange.upper()}{code}"


def _load_historical_st_intervals() -> Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]]:
    """加载历史 ST 区间。

    若本地不存在 namechange 历史文件，则返回空映射并自动降级为 no-op。
    支持 Tushare 常见字段: ts_code, name, start_date, end_date。
    """
    global _historical_st_intervals, _historical_st_loaded
    if _historical_st_intervals is not None:
        return _historical_st_intervals

    intervals: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for path in _iter_namechange_paths():
        if not path.exists():
            continue

        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path, dtype=str)

        required = {"ts_code", "name", "start_date"}
        if not required.issubset(df.columns):
            continue

        work = df.copy()
        work["instrument"] = work["ts_code"].map(_normalize_ts_code_to_instrument)
        work["start_date"] = pd.to_datetime(work["start_date"], errors="coerce")
        if "end_date" in work.columns:
            work["end_date"] = pd.to_datetime(work["end_date"], errors="coerce")
        else:
            work["end_date"] = pd.NaT

        work = work[work["name"].astype(str).str.contains("ST", na=False)]
        work = work.dropna(subset=["instrument", "start_date"])
        work["end_date"] = work["end_date"].fillna(pd.Timestamp("2099-12-31"))

        grouped = {}
        for inst, grp in work.groupby("instrument"):
            grouped[inst] = [
                (row.start_date, row.end_date)
                for row in grp.sort_values("start_date").itertuples()
            ]

        intervals = grouped
        _historical_st_loaded = True
        break

    _historical_st_intervals = intervals
    return _historical_st_intervals


def _load_index_weight_table() -> pd.DataFrame:
    """加载指数成分权重快照。

    依赖 Tushare index_weight 数据，核心字段:
    index_code, con_code, trade_date[, weight]
    """
    global _index_weight_df
    if _index_weight_df is not None:
        return _index_weight_df

    for path in _iter_index_weight_paths():
        if not path.exists():
            continue

        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path, dtype=str)

        required = {"index_code", "con_code", "trade_date"}
        if not required.issubset(df.columns):
            continue

        work = df.copy()
        work["index_code"] = work["index_code"].astype(str)
        work["instrument"] = work["con_code"].map(_normalize_ts_code_to_instrument)
        work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce")
        work = work.dropna(subset=["index_code", "instrument", "trade_date"])
        work = work.sort_values(["index_code", "trade_date", "instrument"])
        work = work.drop_duplicates(subset=["index_code", "trade_date", "instrument"])

        keep_cols = ["index_code", "trade_date", "instrument"]
        if "weight" in work.columns:
            keep_cols.append("weight")
        _index_weight_df = work[keep_cols].reset_index(drop=True)
        return _index_weight_df

    _index_weight_df = pd.DataFrame(columns=["index_code", "trade_date", "instrument"])
    return _index_weight_df


def _required_index_weight_path_hint() -> str:
    existing = [str(p) for p in _iter_index_weight_paths()]
    return " / ".join(existing)


def has_historical_universe_data(universe: str) -> bool:
    if universe == UNIVERSE_ALL:
        return True
    if universe not in INDEX_CODE_BY_UNIVERSE:
        return False
    table = _load_index_weight_table()
    if table.empty:
        return False
    return table["index_code"].eq(INDEX_CODE_BY_UNIVERSE[universe]).any()


def get_index_constituents_as_of(index_code: str, as_of_date) -> List[str]:
    """返回指定日期可见的最近一期指数成分股列表。"""
    cache_key = (index_code, _normalize_cache_date(as_of_date))
    cached = _index_constituents_as_of_cache.get(cache_key)
    if cached is not None:
        return list(cached)

    table = _load_index_weight_table()
    if table.empty:
        raise FileNotFoundError(
            "缺少历史指数成分数据，请先准备 data/tushare/index_weight.parquet 或 csv。"
            f" 期望路径: {_required_index_weight_path_hint()}"
        )

    subset = table[table["index_code"] == index_code]
    if subset.empty:
        raise ValueError(f"index_weight 数据中不存在指数 {index_code}")

    as_of_ts = pd.Timestamp(as_of_date)
    subset = subset[subset["trade_date"] <= as_of_ts]
    if subset.empty:
        raise ValueError(f"指数 {index_code} 在 {as_of_ts.date()} 之前没有可用成分快照")

    snapshot_date = subset["trade_date"].max()
    result = tuple(sorted(subset.loc[subset["trade_date"] == snapshot_date, "instrument"].unique().tolist()))
    _index_constituents_as_of_cache[cache_key] = result
    return list(result)


def get_universe_instruments(
    start_date,
    end_date,
    universe: str = UNIVERSE_ALL,
) -> List[str]:
    """返回给定股票池在区间内可能出现过的标的并集。"""
    if universe == UNIVERSE_ALL:
        instruments_path = (
            Path(CONFIG.get("paths.qlib_data", "~/code/qlib/data/qlib_data/cn_data"))
            .expanduser()
            / "instruments"
            / "all.txt"
        )
        if not instruments_path.exists():
            return []
        df = pd.read_csv(
            instruments_path,
            sep="\t",
            header=None,
            names=["instrument", "list_date", "end_date"],
            dtype=str,
        )
        df["instrument"] = df["instrument"].str[:2].str.upper() + df["instrument"].str[2:]
        return sorted(df["instrument"].dropna().unique().tolist())

    index_code = INDEX_CODE_BY_UNIVERSE.get(universe)
    if index_code is None:
        raise ValueError(f"未知股票池: {universe}")

    table = _load_index_weight_table()
    if table.empty:
        raise FileNotFoundError(
            "缺少历史指数成分数据，请先准备 data/tushare/index_weight.parquet 或 csv。"
            f" 期望路径: {_required_index_weight_path_hint()}"
        )

    end_ts = pd.Timestamp(end_date)
    subset = table[(table["index_code"] == index_code) & (table["trade_date"] <= end_ts)]
    if subset.empty:
        raise ValueError(f"指数 {index_code} 在 {pd.Timestamp(start_date).date()} ~ {end_ts.date()} 无可用成分数据")
    return sorted(subset["instrument"].unique().tolist())


def filter_instruments_by_universe(
    instruments: List[str],
    as_of_date,
    universe: str = UNIVERSE_ALL,
) -> List[str]:
    """按历史时点过滤到指定股票池。"""
    if universe == UNIVERSE_ALL:
        return list(instruments)

    index_code = INDEX_CODE_BY_UNIVERSE.get(universe)
    if index_code is None:
        raise ValueError(f"未知股票池: {universe}")

    constituent_set = set(get_index_constituents_as_of(index_code, as_of_date))
    return [inst for inst in instruments if inst in constituent_set]


def filter_instruments(instruments: List[str], exclude_st: bool = True) -> List[str]:
    """过滤股票池：排除指数、北交所、科创板、ST / *ST 股票

    注意：ST 过滤使用当前快照（stock_basic.csv），在回测中存在前视偏差。
    如需回测无偏差，请传入 exclude_st=False。
    """
    st_set = _load_st_set() if exclude_st else set()
    return [
        i for i in instruments
        if not any(i.startswith(p) for p in EXCLUDED_PREFIXES)
        and i not in st_set
    ]


def has_historical_st_data() -> bool:
    """本地是否存在可用的历史 ST 文件。"""
    _load_historical_st_intervals()
    return _historical_st_loaded


def is_st_on_date(instrument: str, as_of_date) -> bool:
    """判断股票在指定日期是否处于 ST 状态。

    若本地缺少历史数据，则返回 False，调用方可安全降级。
    """
    as_of_ts = pd.Timestamp(as_of_date)
    intervals = _load_historical_st_intervals().get(instrument, [])
    for start, end in intervals:
        if start <= as_of_ts <= end:
            return True
    return False


def get_st_instruments_on_date(as_of_date) -> List[str]:
    """返回指定日期处于 ST 的股票列表，并缓存结果。"""
    cache_key = _normalize_cache_date(as_of_date)
    cached = _historical_st_by_date_cache.get(cache_key)
    if cached is not None:
        return list(cached)

    intervals_map = _load_historical_st_intervals()
    if not intervals_map:
        _historical_st_by_date_cache[cache_key] = tuple()
        return []

    as_of_ts = pd.Timestamp(as_of_date)
    blocked = []
    for instrument, intervals in intervals_map.items():
        for start, end in intervals:
            if start <= as_of_ts <= end:
                blocked.append(instrument)
                break

    result = tuple(sorted(blocked))
    _historical_st_by_date_cache[cache_key] = result
    return list(result)


def filter_st_instruments_by_date(
    instruments: List[str],
    as_of_date,
) -> List[str]:
    """按历史时点过滤 ST 股票；无本地历史数据时自动 no-op。"""
    if not has_historical_st_data():
        return list(instruments)

    blocked = set(get_st_instruments_on_date(as_of_date))
    return [inst for inst in instruments if inst not in blocked]


def get_newly_listed_instruments_on_date(
    as_of_date,
    min_days_listed: int = 0,
) -> List[str]:
    """返回指定日期上市天数不足的股票列表，并缓存结果。"""
    if min_days_listed <= 0:
        return []

    cache_key = (_normalize_cache_date(as_of_date), int(min_days_listed))
    cached = _newly_listed_by_date_cache.get(cache_key)
    if cached is not None:
        return list(cached)

    list_dates = _load_list_date_series()
    if list_dates.empty:
        _newly_listed_by_date_cache[cache_key] = tuple()
        return []

    cutoff = pd.Timestamp(as_of_date) - pd.Timedelta(days=min_days_listed)
    result = tuple(sorted(list_dates[list_dates > cutoff].index.tolist()))
    _newly_listed_by_date_cache[cache_key] = result
    return list(result)


def filter_new_listed_instruments(
    instruments: List[str],
    as_of_date,
    min_days_listed: int = 0,
) -> List[str]:
    """按上市天数过滤股票池；缺失上市日期时保留原标的避免误杀。"""
    if min_days_listed <= 0:
        return list(instruments)

    blocked = set(get_newly_listed_instruments_on_date(as_of_date, min_days_listed))
    return [inst for inst in instruments if inst not in blocked]
