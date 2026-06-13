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


CHINEXT_REFORM_DATE = pd.Timestamp("2020-08-24")
PRICE_LIMIT_TOL = 1e-6


# ── 路径 ──────────────────────────────────────────────────────────────────────

def raw_data_root() -> Path:
    qlib_root = Path(
        CONFIG.get("paths.qlib_data", "~/code/qlib/data/qlib_data/cn_data")
    ).expanduser()
    return qlib_root.parent / "raw_data"


def raw_data_path_for_instrument(instrument: str) -> Path:
    return raw_data_root() / f"{instrument[:2].lower()}{instrument[2:]}.parquet"


# ── 交易日历 ──────────────────────────────────────────────────────────────────

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


# ── 涨跌停 ────────────────────────────────────────────────────────────────────

def round_limit_price(value: float) -> float:
    if pd.isna(value):
        return np.nan
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def get_price_limit_pct(instrument: str, trade_date, is_st: bool = False) -> float:
    trade_ts = pd.Timestamp(trade_date)
    if is_st:
        return 0.05
    if instrument.startswith("BJ"):
        return 0.30
    if instrument.startswith("SH688"):
        return 0.20
    if instrument.startswith("SZ300") and trade_ts >= CHINEXT_REFORM_DATE:
        return 0.20
    return 0.10


def get_limit_prices(instrument: str, trade_date, prev_close: float, is_st: bool = False):
    if pd.isna(prev_close) or prev_close <= 0:
        return np.nan, np.nan
    pct = get_price_limit_pct(instrument, trade_date, is_st=is_st)
    up_limit = round_limit_price(prev_close * (1 + pct))
    down_limit = round_limit_price(prev_close * (1 - pct))
    return up_limit, down_limit


def can_buy_at_open(
    instrument: str, trade_date, open_price: float, prev_close: float, is_st: bool = False
) -> bool:
    if pd.isna(open_price) or pd.isna(prev_close) or open_price <= 0 or prev_close <= 0:
        return False
    up_limit, _ = get_limit_prices(instrument, trade_date, prev_close, is_st=is_st)
    if pd.isna(up_limit):
        return False
    return float(open_price) < float(up_limit) - PRICE_LIMIT_TOL


def can_sell_at_open(
    instrument: str, trade_date, open_price: float, prev_close: float, is_st: bool = False
) -> bool:
    if pd.isna(open_price) or pd.isna(prev_close) or open_price <= 0 or prev_close <= 0:
        return False
    _, down_limit = get_limit_prices(instrument, trade_date, prev_close, is_st=is_st)
    if pd.isna(down_limit):
        return False
    return float(open_price) > float(down_limit) + PRICE_LIMIT_TOL


# ── 交易成本 ──────────────────────────────────────────────────────────────────

def compute_trade_cost(
    buy_value: float,
    sell_value: float,
    buy_commission_rate: float = 0.0003,
    sell_commission_rate: float = 0.0003,
    sell_stamp_tax_rate: float = 0.001,
    slippage_bps: float = 5,
    impact_bps: float = 2,
) -> float:
    """计算单次换仓的总摩擦成本"""
    slippage_pct = (slippage_bps + impact_bps) / 10000
    buy_cost = buy_value * (buy_commission_rate + slippage_pct)
    sell_cost = sell_value * (sell_commission_rate + sell_stamp_tax_rate + slippage_pct)
    return buy_cost + sell_cost


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
    raw_cols = [c for c in ["open", "close", "prev_close", "amount"] if c in raw.columns]
    return raw.set_index(["datetime", "instrument"])[raw_cols].sort_index()
