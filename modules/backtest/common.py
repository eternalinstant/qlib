"""
共享交易工具层
从 qlib_engine.py 提取，供 QlibBacktestEngine 和 RuleBasedEngine 共用
"""
import numpy as np
import pandas as pd
from decimal import Decimal, ROUND_HALF_UP
from functools import lru_cache
from pathlib import Path

from config.config import CONFIG
from utils.platform import resolve_path


# 涨跌停规则已下沉至 common/price_limit.py（五层迁移）。re-export 兼容旧 import 路径。
from common.price_limit import (  # noqa: E402,F401
    CHINEXT_REFORM_DATE,
    PRICE_LIMIT_TOL,
    round_limit_price,
    get_price_limit_pct,
    get_limit_prices,
    can_buy_at_open,
    can_sell_at_open,
)


# ── 路径 ──────────────────────────────────────────────────────────────────────
# 已下沉至 common/paths.py（五层迁移）。re-export 兼容旧 import 路径。
# 注：load_raw_trade_quotes（仍驻本模块）调用这两个 re-export 名，
# 故 patch("modules.backtest.common.raw_data_root") 仍能拦截。
from common.paths import raw_data_root, raw_data_path_for_instrument  # noqa: E402,F401


# ── 交易日历 ──────────────────────────────────────────────────────────────────
# 已下沉至 common/calendar.py（五层迁移）。re-export 兼容旧 import 路径。
from common.calendar import load_trade_calendar  # noqa: E402,F401


# ── 涨跌停 ────────────────────────────────────────────────────────────────────
# 已下沉至 common/price_limit.py（见文件顶部 re-export 块）。


# ── 交易成本 ──────────────────────────────────────────────────────────────────
# 已下沉至 common/costs.py（五层迁移）。此处 re-export 兼容旧 import 路径。
from common.costs import compute_trade_cost  # noqa: E402,F401


# ── 原始日线行情 ──────────────────────────────────────────────────────────────

def load_raw_trade_quotes(instruments, start_date: str, end_date: str) -> pd.DataFrame:
    if not instruments:
        return pd.DataFrame(columns=["open", "close", "prev_close"])

    root = raw_data_root()
    lookback_start = pd.Timestamp(start_date) - pd.Timedelta(days=10)
    end_ts = pd.Timestamp(end_date)
    frames = []
    missing_files = []

    for instrument in sorted(set(instruments)):
        path = raw_data_path_for_instrument(instrument)
        if not path.exists():
            missing_files.append(instrument)
            continue

        df = pd.read_parquet(path)
        if df.empty:
            continue

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")
        df = df[(df["date"] >= lookback_start) & (df["date"] <= end_ts)].copy()
        if df.empty:
            continue

        df["instrument"] = instrument
        if "pre_close" in df.columns:
            df["prev_close"] = pd.to_numeric(df["pre_close"], errors="coerce")
        else:
            df["prev_close"] = pd.to_numeric(df["close"], errors="coerce").groupby(
                df["instrument"]
            ).shift(1)
        frames.append(df)

    if missing_files:
        preview = ", ".join(missing_files[:10])
        suffix = " ..." if len(missing_files) > 10 else ""
        print(
            f"[WARN] raw_data 缺少 {len(missing_files)} 个标的文件，按不可买卖处理: {preview}{suffix}"
        )

    if not frames:
        return pd.DataFrame(columns=["open", "close", "prev_close"])

    raw = pd.concat(frames, ignore_index=True)
    raw = raw.sort_values(["instrument", "date"])
    raw = raw.rename(columns={"date": "datetime"})
    raw = raw[raw["datetime"] >= pd.Timestamp(start_date)].copy()
    raw = raw.drop_duplicates(subset=["datetime", "instrument"], keep="last")
    raw_cols = [c for c in ["open", "high", "low", "close", "prev_close", "amount"] if c in raw.columns]
    return raw.set_index(["datetime", "instrument"])[raw_cols].sort_index()
