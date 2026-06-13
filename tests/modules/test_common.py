"""TDD: modules/backtest/common.py 共享工具层"""
import pytest
import numpy as np
import pandas as pd
from modules.backtest.common import (
    round_limit_price,
    get_price_limit_pct,
    get_limit_prices,
    can_buy_at_open,
    can_sell_at_open,
    compute_trade_cost,
    load_raw_trade_quotes,
)


# ── round_limit_price ─────────────────────────────────────────────────────────

def test_round_limit_price_rounds_half_up():
    assert round_limit_price(10.005) == 10.01
    assert round_limit_price(10.004) == 10.00


def test_round_limit_price_nan_returns_nan():
    assert np.isnan(round_limit_price(np.nan))


# ── get_price_limit_pct ───────────────────────────────────────────────────────

def test_price_limit_pct_normal_stock():
    assert get_price_limit_pct("SH600000", "2024-01-01") == 0.10


def test_price_limit_pct_st_stock():
    assert get_price_limit_pct("SH600000", "2024-01-01", is_st=True) == 0.05


def test_price_limit_pct_star_market():
    assert get_price_limit_pct("SH688001", "2024-01-01") == 0.20


def test_price_limit_pct_chinext_after_reform():
    assert get_price_limit_pct("SZ300001", "2021-01-01") == 0.20


def test_price_limit_pct_chinext_before_reform():
    assert get_price_limit_pct("SZ300001", "2020-01-01") == 0.10


def test_price_limit_pct_beijing_exchange():
    assert get_price_limit_pct("BJ430001", "2024-01-01") == 0.30


# ── get_limit_prices ──────────────────────────────────────────────────────────

def test_get_limit_prices_normal():
    up, down = get_limit_prices("SH600000", "2024-01-01", prev_close=10.0)
    assert up == 11.0
    assert down == 9.0


def test_get_limit_prices_invalid_prev_close():
    up, down = get_limit_prices("SH600000", "2024-01-01", prev_close=0.0)
    assert np.isnan(up)
    assert np.isnan(down)


# ── can_buy_at_open ───────────────────────────────────────────────────────────

def test_can_buy_at_open_normal():
    assert can_buy_at_open("SH600000", "2024-01-01", open_price=10.5, prev_close=10.0) is True


def test_cannot_buy_at_limit_up():
    assert can_buy_at_open("SH600000", "2024-01-01", open_price=11.0, prev_close=10.0) is False


def test_cannot_buy_with_nan():
    assert can_buy_at_open("SH600000", "2024-01-01", open_price=np.nan, prev_close=10.0) is False


# ── can_sell_at_open ──────────────────────────────────────────────────────────

def test_can_sell_at_open_normal():
    assert can_sell_at_open("SH600000", "2024-01-01", open_price=10.0, prev_close=10.0) is True


def test_cannot_sell_at_limit_down():
    assert can_sell_at_open("SH600000", "2024-01-01", open_price=9.0, prev_close=10.0) is False


# ── compute_trade_cost ────────────────────────────────────────────────────────

def test_compute_trade_cost_positive():
    cost = compute_trade_cost(buy_value=100_000, sell_value=100_000)
    assert cost > 0


def test_compute_trade_cost_zero_when_no_trade():
    cost = compute_trade_cost(buy_value=0, sell_value=0)
    assert cost == 0.0


def test_compute_trade_cost_sell_higher_than_buy_due_to_stamp_tax():
    buy_cost = compute_trade_cost(buy_value=100_000, sell_value=0)
    sell_cost = compute_trade_cost(buy_value=0, sell_value=100_000)
    assert sell_cost > buy_cost


# ── load_raw_trade_quotes ─────────────────────────────────────────────────────

def test_load_raw_trade_quotes_empty_instruments():
    df = load_raw_trade_quotes([], "2024-01-01", "2024-01-31")
    assert isinstance(df, pd.DataFrame)
    assert "close" in df.columns
