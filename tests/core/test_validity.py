"""
策略有效性评估测试
"""

import pandas as pd

from core.validity import (
    ValidityConfig,
    apply_validity_overlay,
    build_validity_config,
    evaluate_strategy_validity,
)


def test_build_validity_config_disabled_returns_none():
    assert build_validity_config({"enabled": False}) is None


def test_build_validity_config_reads_apply_in_backtest():
    cfg = build_validity_config({"action": "reduce", "apply_in_backtest": True})

    assert cfg is not None
    assert cfg.apply_in_backtest is True


def test_evaluate_strategy_validity_active():
    daily_returns = pd.Series([0.002] * 30)
    config = ValidityConfig(
        lookback_days=20,
        min_observations=10,
        min_total_return=0.0,
        min_annual_return=0.0,
        min_sharpe=0.0,
        max_drawdown=-0.10,
        action="reduce",
        reduce_to=0.5,
    )

    result = evaluate_strategy_validity(daily_returns, config)

    assert result.status == "active"
    assert result.action == "keep"
    assert result.reduction_factor == 1.0


def test_evaluate_strategy_validity_reduce():
    daily_returns = pd.Series([-0.01] * 20)
    config = ValidityConfig(
        lookback_days=20,
        min_observations=10,
        min_total_return=-0.05,
        min_annual_return=-0.20,
        min_sharpe=0.0,
        max_drawdown=-0.08,
        action="reduce",
        reduce_to=0.4,
    )

    result = evaluate_strategy_validity(daily_returns, config)

    assert result.status == "degraded"
    assert result.action == "reduce"
    assert result.reduction_factor == 0.4
    assert "区间收益" in result.reason or "最大回撤" in result.reason


def test_apply_validity_overlay_reduces_after_warmup():
    daily_returns = pd.Series([0.01] * 5 + [-0.03] * 6)
    config = ValidityConfig(
        lookback_days=5,
        min_observations=5,
        min_total_return=-0.01,
        min_annual_return=-0.20,
        min_sharpe=0.0,
        max_drawdown=-0.05,
        action="reduce",
        reduce_to=0.5,
        apply_in_backtest=True,
    )

    adjusted, exposure = apply_validity_overlay(daily_returns, config)

    assert len(adjusted) == len(daily_returns)
    assert len(exposure) == len(daily_returns)
    assert (exposure.iloc[:5] == 1.0).all()
    assert exposure.iloc[6:].min() < 1.0
    assert adjusted.abs().sum() < daily_returns.abs().sum()
