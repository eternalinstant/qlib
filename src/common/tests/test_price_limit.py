"""TDD: common/price_limit.py — A股涨跌停（从引擎层 common.py 下沉到公共库）"""
import pytest
import numpy as np
import pandas as pd
from common.price_limit import (
    CHINEXT_REFORM_DATE,
    PRICE_LIMIT_TOL,
    round_limit_price,
    get_price_limit_pct,
    get_limit_prices,
    can_buy_at_open,
    can_sell_at_open,
)


# ── 常量 ──────────────────────────────────────────────────────────────────────

def test_chinext_reform_date_constant():
    assert CHINEXT_REFORM_DATE == pd.Timestamp("2020-08-24")


def test_price_limit_tol_constant():
    assert PRICE_LIMIT_TOL == 1e-6


# ── round_limit_price ─────────────────────────────────────────────────────────

def test_round_half_up():
    assert round_limit_price(10.005) == 10.01
    assert round_limit_price(10.004) == 10.00


def test_round_nan():
    assert np.isnan(round_limit_price(np.nan))


# ── get_price_limit_pct（各板块/状态）─────────────────────────────────────────

@pytest.mark.parametrize("inst,date,is_st,pct", [
    ("SH600000", "2024-01-01", False, 0.10),   # 主板
    ("SH600000", "2024-01-01", True, 0.05),    # ST
    ("SH688001", "2024-01-01", False, 0.20),   # 科创板
    ("SZ300001", "2021-01-01", False, 0.20),   # 创业板改革后
    ("SZ300001", "2020-01-01", False, 0.10),   # 创业板改革前
    ("BJ430001", "2024-01-01", False, 0.30),   # 北交所
])
def test_price_limit_pct(inst, date, is_st, pct):
    assert get_price_limit_pct(inst, date, is_st=is_st) == pct


# ── get_limit_prices ──────────────────────────────────────────────────────────

def test_limit_prices_normal():
    up, down = get_limit_prices("SH600000", "2024-01-01", prev_close=10.0)
    assert up == 11.0 and down == 9.0


def test_limit_prices_invalid_prev_close():
    up, down = get_limit_prices("SH600000", "2024-01-01", prev_close=0.0)
    assert np.isnan(up) and np.isnan(down)


# ── can_buy / can_sell at open ────────────────────────────────────────────────

def test_can_buy_normal():
    assert can_buy_at_open("SH600000", "2024-01-01", open_price=10.5, prev_close=10.0) is True


def test_cannot_buy_at_limit_up():
    assert can_buy_at_open("SH600000", "2024-01-01", open_price=11.0, prev_close=10.0) is False


def test_cannot_buy_nan():
    assert can_buy_at_open("SH600000", "2024-01-01", open_price=np.nan, prev_close=10.0) is False


def test_can_sell_normal():
    assert can_sell_at_open("SH600000", "2024-01-01", open_price=10.0, prev_close=10.0) is True


def test_cannot_sell_at_limit_down():
    assert can_sell_at_open("SH600000", "2024-01-01", open_price=9.0, prev_close=10.0) is False
