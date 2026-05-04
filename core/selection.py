"""
选股模块 — 统一信号计算，生成月度 Top-K 选股列表

因子加载策略
-----------
- source='qlib'    : 通过 D.features() 加载（基于价格的算子表达式）
- source='parquet' : 从 factor_data.parquet 加载（基本面数据）
两者按 (datetime, instrument) 合并后计算综合信号。
"""

import warnings

warnings.filterwarnings("ignore")

import os
import re
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).parent.parent

from config.config import CONFIG
from core.factors import FactorRegistry, default_registry
from core.compute import compute_layer_score, neutralize_by_industry
from core.universe import (
    filter_instruments,
    filter_instruments_by_universe,
    filter_new_listed_instruments,
    filter_st_instruments_by_date,
    get_universe_instruments,
)

SELECTION_CSV = Path(
    CONFIG.get("paths.selections", "~/code/qlib/data/monthly_selections.csv")
).expanduser()
FACTOR_PARQUET = (
    Path(CONFIG.get("paths.qlib_data", "~/code/qlib/data/qlib_data/cn_data")).expanduser()
    / "factor_data.parquet"
)
INDUSTRY_CSV = PROJECT_ROOT / "data" / "tushare" / "stock_industry.csv"
STOCK_BASIC_CSV = PROJECT_ROOT / "data" / "tushare" / "stock_basic.csv"
_trade_calendar_cache: Optional[pd.DatetimeIndex] = None
WINDOW_FUNCS = {"Mean", "Std", "EMA", "Ref", "Slope", "Min", "Max"}
INT_LITERAL_RE = re.compile(r"^\d+$")


# ── 内部工具函数 ──────────────────────────────────────────────────────────────


def compute_rebalance_dates(dates_series: pd.Series, freq: str = "month") -> pd.DatetimeIndex:
    """按指定频率从交易日序列中提取调仓日（每段最后一个交易日）

    Parameters
    ----------
    dates_series : pd.Series
        交易日序列（已排序）
    freq : str
        调仓频率: "day" | "week" | "biweek" | "month"

    Returns
    -------
    pd.DatetimeIndex
    """
    if freq == "day":
        return pd.DatetimeIndex(dates_series.values)
    elif freq == "month":
        result = dates_series.groupby(dates_series.dt.to_period("M")).last()
    elif freq == "biweek":
        iso = dates_series.dt.isocalendar()
        result = dates_series.groupby(iso.week // 2 + iso.year * 100).last()
    elif freq == "week":
        iso = dates_series.dt.isocalendar()
        result = dates_series.groupby(iso.week + iso.year * 100).last()
    else:
        raise ValueError(f"未知调仓频率: {freq}")

    return pd.DatetimeIndex(result.values)


def _get_rebalance_dates(df: pd.DataFrame, freq: str = "month") -> pd.DatetimeIndex:
    """从 MultiIndex DataFrame 中按指定频率提取调仓日"""
    dates = df.index.get_level_values("datetime").unique().sort_values()
    return compute_rebalance_dates(pd.Series(dates), freq=freq)


def _split_top_level_args(text: str) -> list[str]:
    """按顶层逗号切分函数参数，避免拆开嵌套表达式。"""
    parts = []
    depth = 0
    start = 0
    for i, ch in enumerate(text):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(text[start:i].strip())
            start = i + 1
    parts.append(text[start:].strip())
    return parts


def _scale_expression_windows(expr: str, window_scale: int) -> str:
    """放大 qlib 表达式中的窗口参数，用于真周频/双周/月频因子。"""
    if window_scale <= 1:
        return expr

    def walk(text: str) -> str:
        out = []
        i = 0
        n = len(text)
        while i < n:
            ch = text[i]
            if ch.isalpha() or ch == "_":
                j = i + 1
                while j < n and (text[j].isalnum() or text[j] == "_"):
                    j += 1
                name = text[i:j]
                if j < n and text[j] == "(":
                    depth = 1
                    k = j + 1
                    while k < n and depth > 0:
                        if text[k] == "(":
                            depth += 1
                        elif text[k] == ")":
                            depth -= 1
                        k += 1
                    inner = text[j + 1 : k - 1]
                    args = [walk(arg) for arg in _split_top_level_args(inner)]
                    if name in WINDOW_FUNCS and len(args) >= 2 and INT_LITERAL_RE.fullmatch(args[1]):
                        args[1] = str(int(args[1]) * window_scale)
                    out.append(f"{name}(" + ", ".join(args) + ")")
                    i = k
                    continue
            out.append(ch)
            i += 1
        return "".join(out)

    return walk(expr)


def _smooth_signal_over_time(signal: pd.Series, window: int) -> pd.Series:
    """沿时间维对单票得分做滚动均值平滑，降低日频抖动。"""
    if window <= 1 or signal.empty:
        return signal

    work = signal.rename("score").reset_index()
    work = work.sort_values(["instrument", "datetime"])
    work["score"] = (
        work.groupby("instrument", sort=False)["score"]
        .rolling(window=window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    return work.set_index(["datetime", "instrument"])["score"].sort_index()


def _rolling_max_over_time(series: pd.Series, window: int, value_name: str = "value") -> pd.Series:
    """沿时间维对单票数值做滚动最大值。"""
    if window <= 1 or series.empty:
        return series

    work = series.rename(value_name).reset_index()
    work = work.sort_values(["instrument", "datetime"])
    work[value_name] = (
        work.groupby("instrument", sort=False)[value_name]
        .rolling(window=window, min_periods=1)
        .max()
        .reset_index(level=0, drop=True)
    )
    return work.set_index(["datetime", "instrument"])[value_name].sort_index()


_factor_parquet_columns_cache: Optional[set] = None


def _to_provider_instruments(instruments: Optional[list]) -> Optional[list]:
    """将内部 instrument 转成 qlib provider 使用的小写前缀格式。"""
    if not instruments:
        return None

    normalized = []
    for inst in instruments:
        text = str(inst)
        if "." in text:
            code, exchange = text.split(".", 1)
            normalized.append(f"{exchange.lower()}{code}")
        elif len(text) >= 2:
            normalized.append(f"{text[:2].lower()}{text[2:]}")
        else:
            normalized.append(text.lower())
    return sorted(set(normalized))


def _to_parquet_instruments(instruments: Optional[list]) -> Optional[list]:
    """将 qlib instrument 格式转换为 factor_data.parquet 使用的格式。"""
    provider_instruments = _to_provider_instruments(instruments)
    if not provider_instruments:
        return None
    return sorted({inst[2:] + inst[:2] for inst in provider_instruments})


def _to_qlib_instruments(series: pd.Series) -> pd.Series:
    """将 provider/parquet 的 instrument 列转回内部统一的大写前缀格式。"""
    series = series.astype(str)
    dot_mask = series.str.contains(".", regex=False, na=False)
    parquet_mask = (~dot_mask) & series.str.match(r"^\d+[A-Za-z]{2}$", na=False)
    out = series.copy()
    if dot_mask.any():
        parts = series[dot_mask].str.split(".", n=1, expand=True)
        out.loc[dot_mask] = parts[1].str.upper() + parts[0]

    if parquet_mask.any():
        out.loc[parquet_mask] = (
            out.loc[parquet_mask].str[-2:].str.upper() + out.loc[parquet_mask].str[:-2]
        )

    plain_mask = (~dot_mask) & (~parquet_mask)
    if plain_mask.any():
        out.loc[plain_mask] = (
            out.loc[plain_mask].str[:2].str.upper() + out.loc[plain_mask].str[2:]
        )
    return out


def _normalize_multiindex_instruments(df: pd.DataFrame) -> pd.DataFrame:
    """把 MultiIndex 中的 instrument 统一到内部大写前缀格式。"""
    if df.empty or not isinstance(df.index, pd.MultiIndex) or "instrument" not in df.index.names:
        return df

    index_names = list(df.index.names)
    work = df.reset_index()
    work["instrument"] = _to_qlib_instruments(work["instrument"])
    return work.set_index(index_names)


def _get_factor_parquet_columns() -> set:
    """读取 factor_data.parquet 的列名并缓存，避免后续反复探测。"""
    global _factor_parquet_columns_cache
    if _factor_parquet_columns_cache is not None:
        return _factor_parquet_columns_cache

    if not FACTOR_PARQUET.exists():
        _factor_parquet_columns_cache = set()
        return _factor_parquet_columns_cache

    try:
        import pyarrow.parquet as pq

        _factor_parquet_columns_cache = set(pq.ParquetFile(FACTOR_PARQUET).schema.names)
    except Exception:
        _factor_parquet_columns_cache = set(pd.read_parquet(FACTOR_PARQUET).columns.tolist())
    return _factor_parquet_columns_cache


def _qlib_data_root() -> Path:
    """返回 Qlib 数据根目录。"""
    return Path(
        CONFIG.get(
            "paths.data.qlib_data",
            CONFIG.get("qlib_data_path", "~/code/qlib/data/qlib_data/cn_data"),
        )
    ).expanduser()


def _load_trade_calendar(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DatetimeIndex:
    """加载交易日历，避免为取调仓日额外拉取全市场收盘价。"""
    global _trade_calendar_cache
    if _trade_calendar_cache is None:
        cal_path = _qlib_data_root() / "calendars" / "day.txt"
        if not cal_path.exists():
            return pd.DatetimeIndex([])
        calendar = pd.read_csv(cal_path, header=None, names=["date"])
        dates = pd.to_datetime(calendar["date"], errors="coerce").dropna().sort_values().unique()
        _trade_calendar_cache = pd.DatetimeIndex(dates)

    dates = _trade_calendar_cache
    if start_date is not None:
        dates = dates[dates >= pd.Timestamp(start_date)]
    if end_date is not None:
        dates = dates[dates <= pd.Timestamp(end_date)]
    return pd.DatetimeIndex(dates)


def _read_factor_parquet(
    columns: list,
    start_date: str = None,
    end_date: str = None,
    instruments: list = None,
) -> pd.DataFrame:
    """按列读取 factor_data.parquet，并尽量下推日期/标的过滤。"""
    if not FACTOR_PARQUET.exists():
        return pd.DataFrame(columns=columns)

    columns = list(dict.fromkeys(columns))
    parquet_instruments = _to_parquet_instruments(instruments)
    start_ts = pd.Timestamp(start_date) if start_date is not None else None
    end_ts = pd.Timestamp(end_date) if end_date is not None else None

    try:
        import pyarrow.dataset as ds

        dataset = ds.dataset(str(FACTOR_PARQUET), format="parquet")
        filter_expr = None
        if start_ts is not None:
            filter_expr = ds.field("datetime") >= start_ts.to_pydatetime()
        if end_ts is not None:
            end_expr = ds.field("datetime") <= end_ts.to_pydatetime()
            filter_expr = end_expr if filter_expr is None else filter_expr & end_expr
        if parquet_instruments:
            inst_expr = ds.field("instrument").isin(parquet_instruments)
            filter_expr = inst_expr if filter_expr is None else filter_expr & inst_expr

        df = dataset.to_table(columns=columns, filter=filter_expr).to_pandas()
    except Exception:
        df = pd.read_parquet(FACTOR_PARQUET, columns=columns)
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
            if start_ts is not None:
                df = df[df["datetime"] >= start_ts]
            if end_ts is not None:
                df = df[df["datetime"] <= end_ts]
        if parquet_instruments and "instrument" in df.columns:
            df = df[df["instrument"].isin(parquet_instruments)]

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def _load_total_mv_frame(
    instruments: list,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """加载并整理 total_mv，供市值过滤和选股结果富化共用。"""
    available_cols = _get_factor_parquet_columns()
    required_cols = {"datetime", "instrument", "total_mv"}
    if not required_cols.issubset(available_cols):
        return pd.DataFrame(columns=["datetime", "symbol", "total_mv"])

    raw = _read_factor_parquet(
        ["datetime", "instrument", "total_mv"],
        start_date=start_date,
        end_date=end_date,
        instruments=instruments,
    )
    if raw.empty:
        return pd.DataFrame(columns=["datetime", "symbol", "total_mv"])

    raw = raw.dropna(subset=["datetime", "instrument"]).copy()
    raw["symbol"] = _to_qlib_instruments(raw["instrument"])
    raw = raw.sort_values(["symbol", "datetime"])
    raw["total_mv"] = raw.groupby("symbol")["total_mv"].ffill()
    return (
        raw[["datetime", "symbol", "total_mv"]]
        .drop_duplicates(subset=["datetime", "symbol"], keep="last")
        .reset_index(drop=True)
    )


def _load_parquet_factors(
    instruments: list,
    start_date: str,
    end_date: str,
    registry: FactorRegistry = None,
) -> pd.DataFrame:
    """
    从 factor_data.parquet 加载基本面因子。

    返回 MultiIndex (datetime, instrument) 的 DataFrame，
    列名格式为 "{category}_{factor_name}"（与 qlib 因子列名一致）。
    negate=True 的因子会在此处乘以 -1。
    """
    if registry is None:
        registry = default_registry

    parquet_factors = registry.get_by_source("parquet")
    if not parquet_factors:
        return pd.DataFrame()

    if not FACTOR_PARQUET.exists():
        print(f"[WARN] factor_data.parquet 不存在: {FACTOR_PARQUET}")
        return pd.DataFrame()

    available_cols = _get_factor_parquet_columns()
    required_cols = ["datetime", "instrument"]
    for f in parquet_factors:
        if f.expression not in available_cols:
            print(f"[WARN] parquet 缺少列 '{f.expression}'（因子: {f.name}），将跳过")
            continue
        required_cols.append(f.expression)

    if len(required_cols) <= 2:
        return pd.DataFrame()

    df = _read_factor_parquet(
        required_cols,
        start_date=start_date,
        end_date=end_date,
        instruments=instruments,
    )
    if df.empty:
        return pd.DataFrame()

    # 转换 instrument 回 qlib 格式（SZ000001），保持索引与 df_qlib_m 一致
    df["instrument"] = _to_qlib_instruments(df["instrument"])
    df = df.drop_duplicates(subset=["datetime", "instrument"], keep="last")

    df = df.set_index(["datetime", "instrument"])

    # 按注册信息选列、取反、重命名
    col_map = {}
    for f in parquet_factors:
        if f.expression not in df.columns:
            continue
        new_col = f"{f.category}_{f.name}"
        col_map[f.expression] = new_col
        if f.negate:
            df[f.expression] = -df[f.expression]

    if not col_map:
        return pd.DataFrame()

    return df[list(col_map.keys())].rename(columns=col_map)


def _fill_cross_sectional(df: pd.DataFrame) -> pd.DataFrame:
    """按日期截面用中位数填充 NaN，剩余用 0 兜底"""
    medians = df.groupby(level="datetime").transform("median")
    return df.fillna(medians).fillna(0)


def _split_by_datetime(obj):
    """把 MultiIndex(datetime, instrument) 对象切成按日期索引的字典。"""
    if obj is None or len(obj) == 0:
        return {}
    result = {}
    for dt, grp in obj.groupby(level="datetime", sort=False):
        result[pd.Timestamp(dt)] = grp.droplevel("datetime")
    return result


# ── 行业中性化 ─────────────────────────────────────────────────────────────────

_industry_map_cache: Dict[str, str] = None


def _load_industry_map() -> Dict[str, str]:
    """加载股票→行业映射，将 ts_code (000001.SZ) 转为 instrument (SZ000001) 格式"""
    global _industry_map_cache
    if _industry_map_cache is not None:
        return _industry_map_cache

    if not INDUSTRY_CSV.exists():
        print(f"[WARN] 行业数据不存在: {INDUSTRY_CSV}，跳过行业中性化")
        _industry_map_cache = {}
        return _industry_map_cache

    df = pd.read_csv(INDUSTRY_CSV, dtype=str)
    df = df[df["ts_code"].str.contains(".", na=False)]
    parts = df["ts_code"].str.split(".", expand=True)
    df["instrument"] = parts[1] + parts[0]
    result = dict(zip(df["instrument"], df["industry"]))

    _industry_map_cache = result
    print(f"[OK] 行业数据加载: {len(result)} 只股票")
    return result


# ── 元数据富化 ────────────────────────────────────────────────────────────────

_name_map_cache: Dict[str, str] = None


def _load_name_map() -> Dict[str, str]:
    """加载股票名称映射：instrument (SZ300762) → 股票名称"""
    global _name_map_cache
    if _name_map_cache is not None:
        return _name_map_cache

    if not STOCK_BASIC_CSV.exists():
        print(f"[WARN] stock_basic.csv 不存在: {STOCK_BASIC_CSV}")
        _name_map_cache = {}
        return _name_map_cache

    df = pd.read_csv(STOCK_BASIC_CSV, dtype=str)
    df = df[df["ts_code"].str.contains(".", na=False)]
    parts = df["ts_code"].str.split(".", expand=True)
    df["instrument"] = parts[1] + parts[0]
    result = dict(zip(df["instrument"], df["name"]))

    _name_map_cache = result
    return result


get_name_map = _load_name_map  # public alias for external modules


def _enrich_selections(df_sel: pd.DataFrame, total_mv_frame: pd.DataFrame = None) -> pd.DataFrame:
    """为选股 DataFrame 添加 name 和 total_mv 列"""
    df_sel = df_sel.copy()
    for col, dtype in (
        ("date", "datetime64[ns]"),
        ("rank", "int64"),
        ("symbol", "object"),
        ("score", "float64"),
    ):
        if col not in df_sel.columns:
            df_sel[col] = pd.Series(dtype=dtype)

    # 添加股票名称
    name_map = _load_name_map()
    df_sel["name"] = df_sel["symbol"].map(name_map).fillna("")

    # 从 factor_data.parquet 读取 total_mv，按股票前向填充后匹配选股日期
    if total_mv_frame is None:
        if df_sel.empty:
            df_sel["total_mv"] = float("nan")
            return df_sel
        total_mv_frame = _load_total_mv_frame(
            instruments=df_sel["symbol"].unique().tolist(),
            start_date=str(df_sel["date"].min().date()),
            end_date=str(df_sel["date"].max().date()),
        )

    if total_mv_frame is None or total_mv_frame.empty:
        df_sel["total_mv"] = float("nan")
        return df_sel

    # 用 merge_asof 取每个选股日期当天或之前最近的 total_mv
    mv_df = (
        total_mv_frame[["datetime", "symbol", "total_mv"]]
        .rename(columns={"datetime": "date"})
        .drop_duplicates(["date", "symbol"])
        .sort_values("date")
    )

    df_sel = df_sel.sort_values("date")
    df_sel = (
        pd.merge_asof(
            df_sel,
            mv_df,
            on="date",
            by="symbol",
            direction="backward",
        )
        .sort_values(["date", "rank"])
        .reset_index(drop=True)
    )
    return df_sel


# ── 公开接口 ──────────────────────────────────────────────────────────────────


def compute_signal(
    monthly_df: pd.DataFrame,
    registry: FactorRegistry = None,
    weights: Dict[str, float] = None,
    neutralize_industry: bool = True,
) -> pd.Series:
    """
    统一信号计算（唯一实现）

    Parameters
    ----------
    monthly_df : pd.DataFrame
        MultiIndex (datetime, instrument)，包含所有因子列
    registry : FactorRegistry, optional
        因子注册表，默认使用 default_registry
    weights : Dict[str, float], optional
        各 category 的权重，如 {"alpha": 0.55, "risk": 0.20, "enhance": 0.25}
        默认从 CONFIG 读取 w_{category}
    neutralize_industry : bool
        是否做行业中性化（去均值）

    Returns
    -------
    pd.Series
        综合得分，index 同 monthly_df
    """
    if registry is None:
        registry = default_registry

    # 动态获取所有 category
    categories = registry.categories()

    if weights is None:
        weights = {cat: CONFIG.get(f"w_{cat}", 0.0) for cat in categories}

    # 行业中性化：在 rank 之前对每个因子按行业-日期去均值
    df = monthly_df
    if neutralize_industry:
        neutralize_start = time.perf_counter()
        industry_map = _load_industry_map()
        if industry_map:
            df = neutralize_by_industry(monthly_df, industry_map)
        print(f"[INFO] 行业中性化完成: 用时 {time.perf_counter() - neutralize_start:.1f}s")

    # 动态计算各层得分并加权合成
    score_start = time.perf_counter()
    signal = pd.Series(0.0, index=df.index)
    for cat in categories:
        cat_factors = registry.get_by_category(cat)
        cat_cols = [f"{cat}_{f.name}" for f in cat_factors]
        w = weights.get(cat, 0.0)
        if w > 0 and cat_cols:
            layer_start = time.perf_counter()
            ir_weights = {f"{cat}_{f.name}": f.ir for f in cat_factors if f.ir != 0.0}
            signal = signal + w * compute_layer_score(df, cat_cols, ir_weights=ir_weights or None)
            print(
                f"[INFO] {cat} 层得分完成: {len(cat_cols)} 列, 用时 {time.perf_counter() - layer_start:.1f}s"
            )

    print(f"[INFO] 分层得分聚合完成: 用时 {time.perf_counter() - score_start:.1f}s")

    return signal


def load_factor_data(
    registry: FactorRegistry = None,
    start_date: str = None,
    end_date: str = None,
    rebalance_freq: str = "month",
    universe: str = "all",
    factor_window_scale: int = 1,
) -> tuple:
    """
    Step 1: 加载因子数据（可缓存）

    Returns
    -------
    tuple
        (monthly_df, rebalance_dates)
        - monthly_df: MultiIndex (datetime, instrument) DataFrame
        - rebalance_dates: pd.DatetimeIndex
    """
    if registry is None:
        registry = default_registry

    import qlib
    from qlib.config import REG_CN
    from qlib.data import D

    os.environ["JOBLIB_START_METHOD"] = "fork"
    try:
        qlib.init(provider_uri=CONFIG.get("qlib_data_path"), region=REG_CN)
    except Exception:
        pass

    if start_date is None:
        start_date = CONFIG.get("start_date", "2019-01-01")
    if end_date is None:
        end_date = CONFIG.get("end_date", "2026-02-26")

    print(
        f"[INFO] 加载因子数据: universe={universe}, freq={rebalance_freq}, "
        f"window_scale={factor_window_scale}, {start_date} ~ {end_date}"
    )

    if universe == "all":
        instruments = D.instruments(market="all")
        candidate_instruments = list(D.list_instruments(instruments, start_date, end_date).keys())
    else:
        candidate_instruments = get_universe_instruments(start_date, end_date, universe=universe)

    print(f"[INFO] 候选股票数: {len(candidate_instruments)}")

    # 回测场景下禁用基于当前快照的 ST 过滤，避免前视偏差。
    valid_instruments = filter_instruments(candidate_instruments, exclude_st=False)
    trade_dates = _load_trade_calendar(start_date=start_date, end_date=end_date)
    if trade_dates.empty:
        raise FileNotFoundError(f"Qlib 交易日历为空: {_qlib_data_root() / 'calendars' / 'day.txt'}")

    qlib_factors = registry.get_by_source("qlib")
    qlib_fields = [
        _scale_expression_windows(f.expression, int(max(factor_window_scale, 1)))
        for f in qlib_factors
    ]
    qlib_names = [f"{f.category}_{f.name}" for f in qlib_factors]

    qlib_start = time.perf_counter()
    if qlib_fields:
        provider_instruments = _to_provider_instruments(valid_instruments)
        df_qlib = D.features(provider_instruments, qlib_fields, start_date, end_date, "day")
        df_qlib = _normalize_multiindex_instruments(df_qlib)
        df_qlib.columns = qlib_names
        available_instruments = filter_instruments(
            df_qlib.index.get_level_values("instrument").unique().tolist(),
            exclude_st=False,
        )
        df_qlib = df_qlib.loc[
            df_qlib.index.get_level_values("instrument").isin(available_instruments)
        ]
        valid_instruments = available_instruments
        print(
            f"[INFO] Qlib 因子加载完成: {len(valid_instruments)} 只股票, {len(qlib_fields)} 列, 用时 {time.perf_counter() - qlib_start:.1f}s"
        )

        # 处理 qlib 因子的 negate（parquet 在 _load_parquet_factors 中已处理）
        for f in qlib_factors:
            col = f"{f.category}_{f.name}"
            if f.negate and col in df_qlib.columns:
                df_qlib[col] = -df_qlib[col]
    else:
        empty_index = pd.MultiIndex.from_arrays([[], []], names=["datetime", "instrument"])
        df_qlib = pd.DataFrame(index=empty_index)
        print(
            f"[INFO] Qlib 因子加载完成: 0 列（纯 parquet 策略），用时 {time.perf_counter() - qlib_start:.1f}s"
        )

    rebalance_dates = compute_rebalance_dates(pd.Series(trade_dates), freq=rebalance_freq)
    if not df_qlib.empty and len(df_qlib.columns) > 0:
        df_qlib_m = df_qlib.loc[df_qlib.index.get_level_values("datetime").isin(rebalance_dates)]
    else:
        empty_index = pd.MultiIndex.from_arrays([[], []], names=["datetime", "instrument"])
        df_qlib_m = pd.DataFrame(index=empty_index)

    parquet_start = time.perf_counter()
    df_parquet_m = _load_parquet_factors(valid_instruments, start_date, end_date, registry)
    if not df_parquet_m.empty:
        df_parquet_m = df_parquet_m.loc[
            df_parquet_m.index.get_level_values("datetime").isin(rebalance_dates)
        ]
    print(
        f"[INFO] Parquet 因子加载完成: {len(df_parquet_m.columns) if not df_parquet_m.empty else 0} 列, 用时 {time.perf_counter() - parquet_start:.1f}s"
    )

    if df_parquet_m.empty:
        monthly_df = df_qlib_m
    elif df_qlib_m.empty or len(df_qlib_m.columns) == 0:
        monthly_df = df_parquet_m
    else:
        monthly_df = df_qlib_m.join(df_parquet_m, how="left")

    fill_start = time.perf_counter()
    monthly_df = _fill_cross_sectional(monthly_df)
    print(
        f"[INFO] 截面缺失填充完成: {len(monthly_df)} 行, {len(rebalance_dates)} 个调仓日, 用时 {time.perf_counter() - fill_start:.1f}s"
    )
    return monthly_df, rebalance_dates


def _load_close_series(
    instruments: list,
    start_date: str,
    end_date: str,
) -> pd.Series:
    """加载日频收盘价，供动态止损换仓模式使用。"""
    if not instruments:
        return pd.Series(dtype=float)

    from qlib.data import D

    provider_instruments = _to_provider_instruments(instruments)
    df_close = D.features(provider_instruments, ["$close"], start_date, end_date, "day")
    if df_close.empty:
        return pd.Series(dtype=float)

    if list(df_close.index.names) == ["instrument", "datetime"]:
        df_close = df_close.swaplevel().sort_index()
    df_close = _normalize_multiindex_instruments(df_close)

    close_col = df_close.columns[0]
    close_series = df_close[close_col].astype(float).sort_index()
    return close_series


def extract_topk(
    signal: pd.Series,
    rebalance_dates: pd.DatetimeIndex,
    topk: int = 20,
    mv_floor: float = 0.0,
    mv_series: "pd.Series | None" = None,
    sticky: int = 0,
    threshold: float = 0.0,
    churn_limit: int = 0,
    margin_stable: bool = False,
    buffer: int = 0,
    exclude_new_days: int = 0,
    exclude_st: bool = False,
    universe: str = "all",
    selection_mode: str = "factor_topk",
    hard_filters: dict = None,
    hard_filter_quantiles: dict = None,
    industry_leader_field: str = None,
    industry_leader_top_n: int = None,
    hard_filter_data: "pd.DataFrame | None" = None,
    score_smoothing_days: int = 1,
    entry_rank: int = None,
    exit_rank: int = None,
    entry_persist_days: int = 1,
    exit_persist_days: int = 1,
    min_hold_days: int = 0,
    close_series: "pd.Series | None" = None,
    recent_high_series: "pd.Series | None" = None,
    stoploss_lookback_days: int = 20,
    stoploss_drawdown: float = 0.10,
    replacement_pool_size: int = 0,
) -> pd.DataFrame:
    """
    Step 3: 从信号中提取 Top-K 选股（支持排名缓冲区）

    Parameters
    ----------
    signal : pd.Series
        综合得分，MultiIndex (datetime, instrument)
    rebalance_dates : pd.DatetimeIndex
        调仓日期序列
    topk : int
        每期选股数量
    mv_floor : float
        市值下限（万元），0 = 不过滤
    mv_series : pd.Series, optional
        MultiIndex (datetime, instrument) → total_mv（万元）
    sticky : int
        (旧参数，buffer > 0 时忽略) 从上期保留的股票数量
    threshold : float
        得分阈值（0-1），只有得分排名比例超过该阈值的旧持仓才保留
    churn_limit : int
        每次调仓最多换出的股票数量，0=不限制
    margin_stable : bool
        边缘股票稳定性
    buffer : int
        排名缓冲区大小。持仓股排名在 topk+buffer 内就保留，
        新股必须进入 topk 才买入。例如 buffer=10, topk=15:
        持仓股跌出 top-25 才卖，新股必须进入 top-15 才买。
        0 = 不使用缓冲区（退回 sticky 模式）
    hard_filters : dict, optional
        财务因子硬过滤条件，格式: {"roa": 0, "ocf_to_ev": 0}
        键为因子名（对应 parquet 列），值为最小阈值
    hard_filter_quantiles : dict, optional
        分位过滤条件，格式: {"roa_fina": 0.4}
        含义为仅保留该因子值高于当日该分位阈值的股票
    industry_leader_field : str, optional
        行业内龙头筛选字段（来自 parquet 列），例如 circ_mv
    industry_leader_top_n : int, optional
        每个行业仅保留按该字段排序前 N 的股票
    hard_filter_data : pd.DataFrame, optional
        MultiIndex (datetime, instrument) → 各因子值
    selection_mode : str
        选股模式:
        - factor_topk: 原始因子 Top-K / 稳定性 / 事件驱动模式
        - stoploss_replace: 初始按因子建仓，之后仅在个股跌破近期高点阈值时换仓
    close_series : pd.Series, optional
        MultiIndex (datetime, instrument) → 日收盘价
    recent_high_series : pd.Series, optional
        MultiIndex (datetime, instrument) → 近期最高收盘价
    stoploss_lookback_days : int
        动态止损回看窗口（交易日）
    stoploss_drawdown : float
        触发换仓的回撤阈值，正数，如 0.10 表示较近期高点回撤 10%
    replacement_pool_size : int
        备选股票池大小，0 表示不限制

    Returns
    -------
    pd.DataFrame
        columns = [date, rank, symbol, score]
    """
    signal = _smooth_signal_over_time(signal, int(max(score_smoothing_days, 1)))
    signal_by_date = _split_by_datetime(signal)
    mv_by_date = _split_by_datetime(mv_series)
    hard_filter_by_date = _split_by_datetime(hard_filter_data)
    close_by_date = _split_by_datetime(close_series)
    recent_high_by_date = _split_by_datetime(recent_high_series)
    industry_map = _load_industry_map() if industry_leader_field and industry_leader_top_n else {}
    if selection_mode == "stoploss_replace":
        if close_series is None and recent_high_series is None:
            raise ValueError("stoploss_replace 模式需要 close_series 或 recent_high_series")
        if recent_high_series is None:
            recent_high_series = _rolling_max_over_time(
                close_series,
                int(max(stoploss_lookback_days, 1)),
                value_name="recent_high",
            )
            recent_high_by_date = _split_by_datetime(recent_high_series)

    rows = []
    prev_symbols = set()
    prev_top_scores = {}
    entry_streaks = {}
    exit_fail_streaks = {}
    held_since = {}
    use_event_driven_gate = selection_mode == "factor_topk" and any(
        [
            score_smoothing_days > 1,
            entry_rank is not None,
            exit_rank is not None,
            entry_persist_days > 1,
            exit_persist_days > 1,
            min_hold_days > 0,
        ]
    )

    for dt in rebalance_dates:
        dt_key = pd.Timestamp(dt)
        day_scores = signal_by_date.get(dt_key)
        if day_scores is None:
            continue

        if universe != "all":
            eligible = filter_instruments_by_universe(
                day_scores.index.tolist(),
                as_of_date=dt,
                universe=universe,
            )
            day_scores = day_scores[day_scores.index.isin(eligible)]

        if exclude_st:
            eligible = filter_st_instruments_by_date(
                day_scores.index.tolist(),
                as_of_date=dt,
            )
            day_scores = day_scores[day_scores.index.isin(eligible)]

        if exclude_new_days > 0:
            eligible = filter_new_listed_instruments(
                day_scores.index.tolist(),
                as_of_date=dt,
                min_days_listed=exclude_new_days,
            )
            day_scores = day_scores[day_scores.index.isin(eligible)]

        # 市值过滤
        if mv_floor > 0 and mv_series is not None:
            day_mv = mv_by_date.get(dt_key)
            if day_mv is not None:
                valid = day_mv[day_mv >= mv_floor].index
                day_scores = day_scores[day_scores.index.isin(valid)]

        # 财务因子硬过滤
        if hard_filters and hard_filter_data is not None:
            day_factors = hard_filter_by_date.get(dt_key)
            if day_factors is not None:
                for factor_name, min_value in hard_filters.items():
                    if factor_name in day_factors.columns:
                        valid_mask = day_factors[factor_name] >= min_value
                        day_scores = day_scores[day_scores.index.isin(valid_mask[valid_mask].index)]

        if hard_filter_quantiles and hard_filter_data is not None:
            day_factors = hard_filter_by_date.get(dt_key)
            if day_factors is not None:
                for factor_name, quantile in hard_filter_quantiles.items():
                    if factor_name not in day_factors.columns:
                        continue
                    factor_series = day_factors[factor_name].dropna()
                    if factor_series.empty:
                        continue
                    threshold = factor_series.quantile(float(quantile))
                    valid_index = factor_series[factor_series >= threshold].index
                    day_scores = day_scores[day_scores.index.isin(valid_index)]

        if industry_leader_field and industry_leader_top_n and hard_filter_data is not None and industry_map:
            day_factors = hard_filter_by_date.get(dt_key)
            if day_factors is not None and industry_leader_field in day_factors.columns:
                rank_series = day_factors[industry_leader_field].dropna()
                rank_series = rank_series[rank_series.index.isin(day_scores.index)]
                if not rank_series.empty:
                    rank_df = rank_series.rename("leader_value").reset_index()
                    rank_df["industry"] = rank_df["instrument"].map(
                        lambda x: industry_map.get(x, "unknown")
                    )
                    rank_df = rank_df.sort_values(
                        ["industry", "leader_value", "instrument"],
                        ascending=[True, False, True],
                    )
                    keep_index = (
                        rank_df.groupby("industry", sort=False)
                        .head(int(industry_leader_top_n))["instrument"]
                        .astype(str)
                    )
                    day_scores = day_scores[day_scores.index.isin(set(keep_index.tolist()))]

        if len(day_scores) < topk:
            continue

        # 排序获取候选列表
        day_sorted = day_scores.sort_values(ascending=False)
        day_index_set = set(day_scores.index)
        effective_entry_rank = min(max(int(entry_rank or topk), 1), len(day_sorted))
        default_exit_rank = topk + buffer if buffer > 0 else topk
        effective_exit_rank = min(
            max(int(exit_rank or default_exit_rank), effective_entry_rank),
            len(day_sorted),
        )
        entry_pool = set(day_sorted.head(effective_entry_rank).index)
        exit_pool = set(day_sorted.head(effective_exit_rank).index)
        entry_streaks = {sym: entry_streaks.get(sym, 0) + 1 for sym in entry_pool}

        # ── 动态止损换仓模式 ──
        if selection_mode == "stoploss_replace" and prev_symbols:
            kept_symbols = set()
            stopped_symbols = set()

            for sym in prev_symbols:
                if sym not in day_index_set:
                    stopped_symbols.add(sym)
                    continue

                day_close = close_by_date.get(dt_key)
                day_recent_high = recent_high_by_date.get(dt_key)
                if day_close is None or day_recent_high is None:
                    stopped_symbols.add(sym)
                    continue
                current_close = day_close.get(sym)
                recent_high = day_recent_high.get(sym)

                if pd.isna(current_close) or pd.isna(recent_high) or recent_high <= 0:
                    stopped_symbols.add(sym)
                    continue

                drawdown = float(current_close) / float(recent_high) - 1.0
                if drawdown <= -abs(float(stoploss_drawdown)):
                    stopped_symbols.add(sym)
                else:
                    kept_symbols.add(sym)

            selected_symbols = set(kept_symbols)
            remaining = topk - len(selected_symbols)
            available_order = [
                sym
                for sym in day_sorted.index
                if sym not in selected_symbols and sym not in stopped_symbols
            ]
            if replacement_pool_size and replacement_pool_size > 0:
                available_order = available_order[: int(replacement_pool_size)]

            selected_symbols.update(available_order[:remaining])
            remaining = topk - len(selected_symbols)
            if remaining > 0:
                fallback_order = [
                    sym
                    for sym in day_sorted.index
                    if sym not in selected_symbols and sym not in stopped_symbols
                ]
                selected_symbols.update(fallback_order[:remaining])

        # ── 事件驱动低换手模式 ──
        elif use_event_driven_gate and prev_symbols:
            keep_set = set()
            dropped = set()
            next_exit_fail_streaks = {}

            for sym in prev_symbols:
                if sym not in day_index_set:
                    dropped.add(sym)
                    continue

                held_days = (pd.Timestamp(dt) - pd.Timestamp(held_since.get(sym, dt))).days
                if sym in exit_pool:
                    next_exit_fail_streaks[sym] = 0
                    keep_set.add(sym)
                    continue

                fail_streak = exit_fail_streaks.get(sym, 0) + 1
                next_exit_fail_streaks[sym] = fail_streak
                if held_days < min_hold_days or fail_streak < max(int(exit_persist_days), 1):
                    keep_set.add(sym)
                else:
                    dropped.add(sym)

            if churn_limit > 0 and len(dropped) > churn_limit:
                drop_scores = day_scores.reindex(list(dropped)).fillna(float("-inf"))
                actually_drop = set(drop_scores.nsmallest(churn_limit).index)
                keep_set = keep_set | (dropped - actually_drop)
                dropped = actually_drop

            selected_symbols = set(keep_set)
            remaining = topk - len(selected_symbols)
            available_order = [s for s in day_sorted.index if s not in selected_symbols]

            if remaining > 0:
                strict_candidates = [
                    s
                    for s in available_order
                    if s in entry_pool
                    and entry_streaks.get(s, 0) >= max(int(entry_persist_days), 1)
                ]
                strict_add = strict_candidates[:remaining]
                selected_symbols.update(strict_add)
                remaining = topk - len(selected_symbols)

            if remaining > 0:
                relaxed_candidates = [
                    s for s in available_order if s not in selected_symbols and s in entry_pool
                ]
                relaxed_add = relaxed_candidates[:remaining]
                selected_symbols.update(relaxed_add)
                remaining = topk - len(selected_symbols)

            if remaining > 0:
                fallback_candidates = [s for s in available_order if s not in selected_symbols]
                selected_symbols.update(fallback_candidates[:remaining])

            held_since = {sym: held_since.get(sym, dt) for sym in selected_symbols}
            exit_fail_streaks = {
                sym: next_exit_fail_streaks.get(sym, 0) if sym in prev_symbols else 0
                for sym in selected_symbols
            }

        # ── 排名缓冲区模式 ──
        elif buffer > 0 and prev_symbols:
            # 宽松名单：top-(topk+buffer) 内的持仓股保留
            wide_set = set(day_sorted.head(topk + buffer).index)
            keep_set = prev_symbols & wide_set  # 旧持仓仍在宽松名单中

            # 换仓数量限制
            if churn_limit > 0:
                max_sell = churn_limit
                dropped = prev_symbols - keep_set
                if len(dropped) > max_sell:
                    # 只卖掉排名最差的 max_sell 只
                    drop_scores = day_scores.reindex(list(dropped)).dropna()
                    actually_drop = set(drop_scores.nsmallest(max_sell).index)
                    keep_set = keep_set | (dropped - actually_drop)

            # 名额不足：从 top-K 新股中补充
            remaining = topk - len(keep_set)
            if remaining > 0:
                available = day_index_set - keep_set
                new_scores = day_scores[list(available)].nlargest(min(remaining, len(available)))
                selected_symbols = keep_set | set(new_scores.index)
            elif len(keep_set) > topk:
                # 保留太多：淘汰排名最低的
                keep_scores = day_scores[list(keep_set)].nlargest(topk)
                selected_symbols = set(keep_scores.index)
            else:
                selected_symbols = keep_set

        # ── 旧 sticky 模式（向后兼容）──
        elif sticky > 0 and prev_symbols:
            keep_candidates = prev_symbols & day_index_set

            # 计算阈值对应的得分线
            threshold_score = None
            if threshold > 0 and prev_top_scores:
                rank_idx = int(len(day_sorted) * threshold)
                if rank_idx < len(day_sorted):
                    threshold_score = day_sorted.iloc[rank_idx]

            if keep_candidates:
                if threshold_score is not None:
                    keep_candidates = {
                        s for s in keep_candidates if day_scores[s] >= threshold_score
                    }

                if margin_stable and sticky > 0:
                    prev_kept = prev_symbols - set(day_sorted.head(10).index)
                    margin_candidates = prev_kept & day_index_set
                    keep_candidates = keep_candidates | margin_candidates

                if keep_candidates:
                    keep_scores = day_scores[list(keep_candidates)].nlargest(
                        min(sticky, len(keep_candidates))
                    )
                    keep_set = set(keep_scores.index)

                    if churn_limit > 0 and len(prev_symbols) > 0:
                        current_hold = keep_set
                        new_needed = min(churn_limit, topk - len(current_hold))
                        if new_needed > 0:
                            exclude = current_hold | (prev_symbols - keep_set)
                            available = [s for s in day_sorted.index if s not in exclude]
                            new_stocks = day_scores[available].nlargest(new_needed)
                            keep_set = keep_set | set(new_stocks.index)

                    remaining = topk - len(keep_set)
                    if remaining > 0:
                        available_symbols = day_index_set - keep_set
                        new_scores = day_scores[list(available_symbols)].nlargest(
                            min(remaining, len(available_symbols))
                        )
                        selected_symbols = keep_set | set(new_scores.index)
                    else:
                        selected_symbols = keep_set
                else:
                    selected_symbols = set(day_sorted.head(topk).index)
            else:
                selected_symbols = set(day_sorted.head(topk).index)

        else:
            # 首次运行或无缓冲，正常选股
            selected_symbols = set(day_sorted.head(topk).index)
            if use_event_driven_gate:
                held_since = {sym: dt for sym in selected_symbols}
                exit_fail_streaks = {sym: 0 for sym in selected_symbols}

        # 按得分排序输出（过滤掉当天无数据的股票）
        selected_symbols = selected_symbols & day_index_set
        selected_scores = day_scores[list(selected_symbols)].sort_values(ascending=False)
        for rank, (sym, score) in enumerate(selected_scores.items(), 1):
            rows.append({"date": dt, "rank": rank, "symbol": sym, "score": score})

        # 更新上期持仓
        prev_symbols = selected_symbols
        prev_top_scores = dict(day_sorted.head(topk))

    return pd.DataFrame(rows, columns=["date", "rank", "symbol", "score"])


def compute_selections(
    registry: FactorRegistry = None,
    weights: Dict[str, float] = None,
    topk: int = None,
    rebalance_freq: str = "month",
    neutralize_industry: bool = True,
    min_market_cap: float = 0.0,
    sticky: int = 0,
    threshold: float = 0.0,
    churn_limit: int = 0,
    margin_stable: bool = False,
    buffer: int = 0,
    exclude_new_days: int = 0,
    exclude_st: bool = False,
    universe: str = "all",
    selection_mode: str = "factor_topk",
    return_context: bool = False,
    hard_filters: Dict[str, float] = None,
    hard_filter_quantiles: Dict[str, float] = None,
    industry_leader_field: str = None,
    industry_leader_top_n: int = None,
    score_smoothing_days: int = 1,
    entry_rank: int = None,
    exit_rank: int = None,
    entry_persist_days: int = 1,
    exit_persist_days: int = 1,
    min_hold_days: int = 0,
    stoploss_lookback_days: int = 20,
    stoploss_drawdown: float = 0.10,
    replacement_pool_size: int = 0,
    update_start_date: str = None,
    update_lookback_days: int = 60,
    factor_window_scale: int = 1,
    scorer: str = "linear",
    lgbm_train_start: str = None,
    lgbm_train_end: str = None,
) -> pd.DataFrame:
    """
    计算月度 Top-K 选股列表（纯内存，不写 CSV）。

    Parameters
    ----------
    min_market_cap : float
        市值下限（亿元），0 = 不过滤
    sticky : int
        从上期保留的股票数量，0 = 不保留
    threshold : float
        得分阈值（0-1），只有得分排名比例超过该阈值的旧持仓才保留
        例如 threshold=0.5 表示保留上期排名前50%的持仓
    churn_limit : int
        每次调仓最多换出的股票数量，0=不限制
    margin_stable : bool
        边缘股票稳定性：若为True，top10保留但11-20名若在旧持仓中也保留
    buffer : int
        排名缓冲区大小。持仓股在 topk+buffer 内优先保留，新股必须进入更高名次才补入
    hard_filters : Dict[str, float], optional
        财务因子硬过滤条件，格式: {"roa_fina": 0, "ocf_to_ev": 0}
        键为 parquet 列名，值为最小阈值
    hard_filter_quantiles : Dict[str, float], optional
        财务因子分位过滤条件，格式: {"roa_fina": 0.4}
        键为 parquet 列名，值为分位数阈值
    industry_leader_field : str, optional
        行业内龙头筛选字段（来自 parquet 列），例如 circ_mv
    industry_leader_top_n : int, optional
        每个行业仅保留按该字段排序前 N 的股票
    selection_mode : str
        选股模式，默认 factor_topk；stoploss_replace 会启用“跌破近期高点再换仓”

    Returns
    -------
    pd.DataFrame
        columns = [date, rank, symbol, score]
    """
    if registry is None:
        registry = default_registry
    if topk is None:
        topk = CONFIG.get("topk", 20)

    # 读取稳定性配置
    if sticky is None or sticky == 0:
        sticky = CONFIG.get("stability_sticky", 0)
    if threshold is None or threshold == 0.0:
        threshold = CONFIG.get("stability_threshold", 0.0)
    if churn_limit is None or churn_limit == 0:
        churn_limit = CONFIG.get("stability_churn_limit", 0)
    if margin_stable is None:
        margin_stable = CONFIG.get("stability_margin_stable", False)

    output_start_date = (
        update_start_date if update_start_date else CONFIG.get("start_date", "2019-01-01")
    )
    end_date = CONFIG.get("end_date", "2026-02-26")

    if update_start_date and update_lookback_days > 0:
        from pandas import Timestamp

        lookback_date = Timestamp(update_start_date) - pd.Timedelta(days=update_lookback_days)
        data_load_start = lookback_date.strftime("%Y-%m-%d")
        print(f"[INFO] 增量更新: 数据加载从 {data_load_start}, 输出从 {update_start_date}")
    else:
        data_load_start = output_start_date

    overall_start = time.perf_counter()
    monthly_df, rebalance_dates = load_factor_data(
        registry=registry,
        start_date=data_load_start,
        end_date=end_date,
        rebalance_freq=rebalance_freq,
        universe=universe,
        factor_window_scale=factor_window_scale,
    )

    signal_start = time.perf_counter()
    if scorer == "lgbm":
        from core.lgbm_scorer import compute_lgbm_signal

        signal = compute_lgbm_signal(
            monthly_df,
            neutralize_industry=neutralize_industry,
            train_start=lgbm_train_start,
            train_end=lgbm_train_end,
        )
    else:
        signal = compute_signal(
            monthly_df, registry=registry, weights=weights, neutralize_industry=neutralize_industry
        )
    print(
        f"[INFO] 综合信号计算完成 (scorer={scorer}): {len(signal)} 行, "
        f"用时 {time.perf_counter() - signal_start:.1f}s"
    )

    # 加载市值数据用于过滤
    mv_series = None
    total_mv_frame = None
    mv_floor = 0.0
    if min_market_cap > 0 and FACTOR_PARQUET.exists():
        mv_start = time.perf_counter()
        candidate_instruments = monthly_df.index.get_level_values("instrument").unique().tolist()
        total_mv_frame = _load_total_mv_frame(
            instruments=candidate_instruments,
            start_date=data_load_start,
            end_date=end_date,
        )
        mv_df = total_mv_frame[total_mv_frame["datetime"].isin(rebalance_dates)]
        mv_series = mv_df.set_index(["datetime", "symbol"])["total_mv"]
        mv_floor = min_market_cap * 10000  # 亿元 → 万元
        print(
            f"[INFO] 市值过滤数据准备完成: {len(candidate_instruments)} 只股票, 用时 {time.perf_counter() - mv_start:.1f}s"
        )

    # 加载硬过滤因子数据
    hard_filter_data = None
    if (hard_filters or hard_filter_quantiles or industry_leader_field) and FACTOR_PARQUET.exists():
        hf_start = time.perf_counter()
        candidate_instruments = monthly_df.index.get_level_values("instrument").unique().tolist()
        requested_hf_cols = set()
        if hard_filters:
            requested_hf_cols.update(hard_filters.keys())
        if hard_filter_quantiles:
            requested_hf_cols.update(hard_filter_quantiles.keys())
        if industry_leader_field:
            requested_hf_cols.add(industry_leader_field)
        hf_cols = ["datetime", "instrument"] + sorted(requested_hf_cols)
        available_cols = _get_factor_parquet_columns()
        hf_cols = [c for c in hf_cols if c in available_cols]
        if len(hf_cols) > 2:
            hf_raw = _read_factor_parquet(
                hf_cols,
                start_date=data_load_start,
                end_date=end_date,
                instruments=candidate_instruments,
            )
            if not hf_raw.empty:
                hf_raw["instrument"] = _to_qlib_instruments(hf_raw["instrument"])
                hf_raw = hf_raw.drop_duplicates(subset=["datetime", "instrument"], keep="last")
                hf_raw = hf_raw[hf_raw["datetime"].isin(rebalance_dates)]
                hard_filter_data = hf_raw.set_index(["datetime", "instrument"])
                print(
                    f"[INFO] 过滤因子数据加载完成: {sorted(requested_hf_cols)}, 用时 {time.perf_counter() - hf_start:.1f}s"
                )

    close_series = None
    recent_high_series = None
    if selection_mode == "stoploss_replace":
        price_start = time.perf_counter()
        candidate_instruments = monthly_df.index.get_level_values("instrument").unique().tolist()
        close_series = _load_close_series(
            candidate_instruments,
            start_date=data_load_start,
            end_date=end_date,
        )
        recent_high_series = _rolling_max_over_time(
            close_series,
            int(max(stoploss_lookback_days, 1)),
            value_name="recent_high",
        )
        print(
            f"[INFO] 动态止损价格数据准备完成: lookback={stoploss_lookback_days}, 用时 {time.perf_counter() - price_start:.1f}s"
        )

    topk_start = time.perf_counter()
    df_sel = extract_topk(
        signal,
        rebalance_dates,
        topk=topk,
        mv_floor=mv_floor,
        mv_series=mv_series,
        sticky=sticky,
        threshold=threshold,
        churn_limit=churn_limit,
        margin_stable=margin_stable,
        buffer=buffer,
        exclude_new_days=exclude_new_days,
        exclude_st=exclude_st,
        universe=universe,
        selection_mode=selection_mode,
        hard_filters=hard_filters,
        hard_filter_quantiles=hard_filter_quantiles,
        industry_leader_field=industry_leader_field,
        industry_leader_top_n=industry_leader_top_n,
        hard_filter_data=hard_filter_data,
        score_smoothing_days=score_smoothing_days,
        entry_rank=entry_rank,
        exit_rank=exit_rank,
        entry_persist_days=entry_persist_days,
        exit_persist_days=exit_persist_days,
        min_hold_days=min_hold_days,
        close_series=close_series,
        recent_high_series=recent_high_series,
        stoploss_lookback_days=stoploss_lookback_days,
        stoploss_drawdown=stoploss_drawdown,
        replacement_pool_size=replacement_pool_size,
    )
    print(f"[INFO] TopK 提取完成: {len(df_sel)} 行, 用时 {time.perf_counter() - topk_start:.1f}s")
    print(f"[INFO] 选股计算完成，总用时 {time.perf_counter() - overall_start:.1f}s")

    if update_start_date:
        output_start = pd.Timestamp(update_start_date)
        df_sel = df_sel[df_sel["date"] >= output_start].copy()
        print(f"[INFO] 增量输出过滤: {len(df_sel)} 行 (从 {update_start_date} 开始)")

    if return_context:
        return df_sel, {"total_mv_frame": total_mv_frame}
    return df_sel


def generate_selections(force=False, output_path=None, update_start_date=None, **kwargs):
    """
    计算并保存 Top-K 选股列表到 CSV（薄封装）。
    返回 DataFrame: columns = [date, rank, symbol, score]

    Parameters
    ----------
    force : bool
        是否强制重新计算
    output_path : Path or str, optional
        自定义输出路径，默认使用 SELECTION_CSV
    update_start_date : str, optional
        增量更新起始日期，只计算此日期之后的选股（需要CSV已存在）
    **kwargs
        传递给 compute_selections() 的参数
    """
    csv_path = Path(output_path) if output_path else SELECTION_CSV
    existing_df = None
    do_incremental = False

    if not force and csv_path.exists():
        if update_start_date:
            existing_df = pd.read_csv(csv_path, parse_dates=["date"])
            last_date = existing_df["date"].max()
            update_from = pd.Timestamp(update_start_date)
            if last_date >= update_from:
                print(f"[OK] 选股已是最新: {csv_path}, 最后日期: {last_date.strftime('%Y-%m-%d')}")
                return existing_df
            print(f"[INFO] 增量更新: {last_date.strftime('%Y-%m-%d')} -> {update_start_date}")
            kwargs["update_start_date"] = update_start_date
            do_incremental = True
        else:
            print(f"[OK] 选股列表已存在: {csv_path}")
            return existing_df

    df_sel, context = compute_selections(return_context=True, **kwargs)
    df_sel = _enrich_selections(df_sel, total_mv_frame=context.get("total_mv_frame"))

    if do_incremental and existing_df is not None:
        update_from = pd.Timestamp(update_start_date)
        existing_before = existing_df[existing_df["date"] < update_from]
        df_sel = pd.concat([existing_before, df_sel], ignore_index=True)
        print(
            f"[INFO] 合并完成: 保留 {len(existing_before)} 行历史, 新增 {len(df_sel) - len(existing_before)} 行"
        )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_sel.to_csv(csv_path, index=False)
    print(f"[OK] 已保存: {csv_path}  期数: {df_sel['date'].nunique()}  总行: {len(df_sel)}")
    return df_sel


def load_selections(csv_path=None):
    """
    加载预计算的选股列表。

    Parameters
    ----------
    csv_path : Path or str, optional
        自定义选股文件路径，默认使用 SELECTION_CSV

    返回:
      date_to_symbols : {Timestamp: set of symbols}
      rebalance_dates : set of Timestamps
    """
    csv_path = Path(csv_path) if csv_path else SELECTION_CSV
    if not csv_path.exists():
        print("[INFO] 选股列表不存在，正在生成...")
        generate_selections(force=True, output_path=csv_path)

    df = pd.read_csv(csv_path, parse_dates=["date"])
    date_to_symbols = {}
    for dt, grp in df.groupby("date"):
        date_to_symbols[pd.Timestamp(dt)] = set(grp["symbol"].tolist())
    return date_to_symbols, set(date_to_symbols.keys())


if __name__ == "__main__":
    generate_selections(force=True)
