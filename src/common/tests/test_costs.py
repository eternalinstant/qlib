"""TDD: common/costs.py — 交易摩擦成本计算（从引擎层 common.py 下沉到公共库）"""
import pytest
from common.costs import compute_trade_cost


# ── 基本行为 ──────────────────────────────────────────────────────────────────

def test_compute_trade_cost_positive():
    cost = compute_trade_cost(buy_value=100_000, sell_value=100_000)
    assert cost > 0


def test_compute_trade_cost_zero_when_no_trade():
    assert compute_trade_cost(buy_value=0, sell_value=0) == 0.0


def test_compute_trade_cost_sell_higher_than_buy_due_to_stamp_tax():
    """卖出含印花税，同等金额下卖出成本 > 买入成本"""
    buy_cost = compute_trade_cost(buy_value=100_000, sell_value=0)
    sell_cost = compute_trade_cost(buy_value=0, sell_value=100_000)
    assert sell_cost > buy_cost


# ── 公式精确锁定（防搬运中改动口径）──────────────────────────────────────────

def test_compute_trade_cost_exact_buy_default_rates():
    """买入 = buy_value * (佣金0.0003 + 滑点(5+2)/10000=0.0007) = buy_value * 0.001"""
    cost = compute_trade_cost(buy_value=100_000, sell_value=0)
    assert cost == pytest.approx(100_000 * (0.0003 + 0.0007), rel=1e-12)


def test_compute_trade_cost_exact_sell_default_rates():
    """卖出 = sell_value * (佣金0.0003 + 印花0.001 + 滑点0.0007) = sell_value * 0.002"""
    cost = compute_trade_cost(buy_value=0, sell_value=100_000)
    assert cost == pytest.approx(100_000 * (0.0003 + 0.001 + 0.0007), rel=1e-12)


def test_compute_trade_cost_custom_rates():
    cost = compute_trade_cost(
        buy_value=50_000, sell_value=50_000,
        buy_commission_rate=0.0005, sell_commission_rate=0.0005,
        sell_stamp_tax_rate=0.001, slippage_bps=10, impact_bps=0,
    )
    slip = (10 + 0) / 10000  # 0.001
    expected = 50_000 * (0.0005 + slip) + 50_000 * (0.0005 + 0.001 + slip)
    assert cost == pytest.approx(expected, rel=1e-12)
