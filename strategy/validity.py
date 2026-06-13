"""
策略有效性评估模块

面向实盘监控场景，对策略最近一段时间的收益、夏普和回撤做轻量体检。
默认只给出动作建议，不直接改写历史回测收益。
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


VALID_VALIDITY_ACTIONS = {"review", "reduce", "pause"}


@dataclass
class ValidityConfig:
    """策略有效性规则"""

    lookback_days: int = 60
    min_observations: int = 40
    min_total_return: float = -0.05
    min_annual_return: float = -0.10
    min_sharpe: float = 0.0
    max_drawdown: float = -0.12
    action: str = "review"  # review | reduce | pause
    reduce_to: float = 0.50
    apply_in_backtest: bool = False


@dataclass
class ValidityResult:
    """策略有效性评估结果"""

    status: str
    action: str
    reduction_factor: float
    lookback_days: int
    observations: int
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    reason: str


def build_validity_config(raw_cfg: Optional[dict]) -> Optional[ValidityConfig]:
    """从 YAML validity 段构建配置。"""
    if not raw_cfg:
        return None
    if raw_cfg.get("enabled", True) is False:
        return None

    action = raw_cfg.get("action", "review")
    if action not in VALID_VALIDITY_ACTIONS:
        raise ValueError(
            f"validity.action='{action}' 无效，可选: {sorted(VALID_VALIDITY_ACTIONS)}"
        )

    return ValidityConfig(
        lookback_days=int(raw_cfg.get("lookback_days", 60)),
        min_observations=int(raw_cfg.get("min_observations", 40)),
        min_total_return=float(raw_cfg.get("min_total_return", -0.05)),
        min_annual_return=float(raw_cfg.get("min_annual_return", -0.10)),
        min_sharpe=float(raw_cfg.get("min_sharpe", 0.0)),
        max_drawdown=float(raw_cfg.get("max_drawdown", -0.12)),
        action=action,
        reduce_to=float(raw_cfg.get("reduce_to", 0.50)),
        apply_in_backtest=bool(raw_cfg.get("apply_in_backtest", False)),
    )


def evaluate_strategy_validity(
    daily_returns: pd.Series, config: ValidityConfig
) -> ValidityResult:
    """对最近 lookback_days 的日收益做体检并输出动作建议。"""
    if config is None:
        raise ValueError("缺少有效性配置")

    rets = pd.Series(daily_returns, copy=False).dropna().astype(float)
    if rets.empty:
        return ValidityResult(
            status="review",
            action="review",
            reduction_factor=1.0,
            lookback_days=config.lookback_days,
            observations=0,
            total_return=0.0,
            annual_return=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            reason="样本为空，无法评估",
        )

    window = rets.tail(config.lookback_days)
    obs = len(window)
    total_return = float((1.0 + window).prod() - 1.0)
    annual_return = float((1.0 + total_return) ** (252.0 / obs) - 1.0) if obs > 0 else 0.0

    volatility = float(window.std(ddof=0))
    if volatility > 0:
        sharpe_ratio = float(window.mean() / volatility * np.sqrt(252))
    else:
        sharpe_ratio = float("inf") if window.mean() > 0 else 0.0

    nav = (1.0 + window).cumprod()
    max_drawdown = float((nav / nav.cummax() - 1.0).min()) if not nav.empty else 0.0

    breaches = []
    if obs < config.min_observations:
        breaches.append(f"样本不足({obs}<{config.min_observations})")
    if total_return < config.min_total_return:
        breaches.append(
            f"区间收益 {total_return:+.2%} < {config.min_total_return:+.2%}"
        )
    if annual_return < config.min_annual_return:
        breaches.append(
            f"年化收益 {annual_return:+.2%} < {config.min_annual_return:+.2%}"
        )
    if sharpe_ratio < config.min_sharpe:
        breaches.append(f"夏普 {sharpe_ratio:.2f} < {config.min_sharpe:.2f}")
    if max_drawdown < config.max_drawdown:
        breaches.append(
            f"最大回撤 {max_drawdown:+.2%} < {config.max_drawdown:+.2%}"
        )

    if not breaches:
        return ValidityResult(
            status="active",
            action="keep",
            reduction_factor=1.0,
            lookback_days=config.lookback_days,
            observations=obs,
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            reason="最近窗口满足有效性阈值",
        )

    if config.action == "pause":
        status = "halt"
        reduction_factor = 0.0
    elif config.action == "reduce":
        status = "degraded"
        reduction_factor = max(min(config.reduce_to, 1.0), 0.0)
    else:
        status = "review"
        reduction_factor = 1.0

    return ValidityResult(
        status=status,
        action=config.action,
        reduction_factor=reduction_factor,
        lookback_days=config.lookback_days,
        observations=obs,
        total_return=total_return,
        annual_return=annual_return,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
        reason="; ".join(breaches),
    )


def apply_validity_overlay(
    daily_returns: pd.Series,
    config: ValidityConfig,
) -> tuple[pd.Series, pd.Series]:
    """把 validity 规则真正应用到收益曲线上，输出调整后收益与每日暴露系数。"""
    rets = pd.Series(daily_returns, copy=False).dropna().astype(float)
    if rets.empty:
        return rets, pd.Series(dtype=float)

    adjusted = []
    exposure = []

    for idx, ret in enumerate(rets):
        realized = pd.Series(adjusted, index=rets.index[:idx], dtype=float)
        if len(realized) < config.min_observations:
            factor = 1.0
        else:
            result = evaluate_strategy_validity(realized, config)
            factor = float(result.reduction_factor)
            if result.status == "active":
                factor = 1.0

        adjusted.append(ret * factor)
        exposure.append(factor)

    return (
        pd.Series(adjusted, index=rets.index, dtype=float),
        pd.Series(exposure, index=rets.index, dtype=float),
    )
