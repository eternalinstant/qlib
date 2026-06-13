"""A股涨跌停规则。

公共库：板块涨跌停幅度、涨跌停价、开盘可买/可卖判定。
原驻 modules/backtest/common.py，五层迁移下沉至此。
"""
import numpy as np
import pandas as pd
from decimal import Decimal, ROUND_HALF_UP


CHINEXT_REFORM_DATE = pd.Timestamp("2020-08-24")
PRICE_LIMIT_TOL = 1e-6


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
