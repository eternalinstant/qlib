"""组合级 overlay：在策略收益之上叠一层现金/债券仓位控制。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OverlayConfig:
    target_vol: float | None = None
    vol_lookback: int = 20
    dd_soft: float | None = None
    dd_hard: float | None = None
    soft_exposure: float = 0.95
    hard_exposure: float = 0.80
    trend_lookback: int = 0
    trend_exposure: float = 0.90
    market_trend_lookback: int = 0
    market_trend_min_return: float = 0.0
    market_trend_exposure_floor: float = 0.0
    market_trend_max_strategy_drawdown: float | None = None
    exposure_min: float = 0.0
    exposure_max: float = 1.0


def _align_bond_returns(base_returns: pd.Series, bond_returns: Union[pd.Series, float, int]) -> pd.Series:
    if np.isscalar(bond_returns):
        return pd.Series(float(bond_returns), index=base_returns.index, dtype=float)
    aligned = pd.Series(bond_returns, dtype=float).reindex(base_returns.index)
    return aligned.ffill().fillna(0.0)


def _annualized_vol(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    std = float(returns.std(ddof=0))
    if not np.isfinite(std) or std <= 0:
        return 0.0
    return std * np.sqrt(252)


def compute_overlay_frame(
    base_returns: pd.Series,
    bond_returns: Union[pd.Series, float, int] = 0.03 / 252,
    config: OverlayConfig = OverlayConfig(),
    market_returns: Union[pd.Series, None] = None,
) -> pd.DataFrame:
    base = pd.Series(base_returns, dtype=float).copy()
    base.index = pd.to_datetime(base.index)
    base = base.sort_index()
    bond = _align_bond_returns(base, bond_returns)
    market = None
    if market_returns is not None:
        market = pd.Series(market_returns, dtype=float).copy()
        market.index = pd.to_datetime(market.index)
        market = market.sort_index().reindex(base.index).ffill().fillna(0.0)

    rows = []
    overlay_returns = []
    overlay_nav = 1.0
    overlay_peak = 1.0
    nav_history = []

    for i, dt in enumerate(base.index):
        exposure = float(config.exposure_max)

        if config.target_vol is not None and i >= max(int(config.vol_lookback), 1):
            hist = pd.Series(overlay_returns[-int(config.vol_lookback) :], dtype=float)
            realized_vol = _annualized_vol(hist)
            if realized_vol > 0:
                exposure = min(exposure, float(config.target_vol) / realized_vol)

        current_drawdown = overlay_nav / overlay_peak - 1.0
        if config.dd_hard is not None and current_drawdown <= -abs(float(config.dd_hard)):
            exposure = min(exposure, float(config.hard_exposure))
        elif config.dd_soft is not None and current_drawdown <= -abs(float(config.dd_soft)):
            exposure = min(exposure, float(config.soft_exposure))

        if config.trend_lookback and len(nav_history) >= int(config.trend_lookback):
            ma = float(np.mean(nav_history[-int(config.trend_lookback) :]))
            if overlay_nav < ma:
                exposure = min(exposure, float(config.trend_exposure))

        market_trend_return = np.nan
        market_risk_on = False
        market_lookback = int(config.market_trend_lookback or 0)
        market_floor = float(config.market_trend_exposure_floor or 0.0)
        if market is not None and market_lookback > 0 and market_floor > 0 and i >= market_lookback:
            market_window = market.iloc[i - market_lookback : i]
            market_trend_return = float((1.0 + market_window).prod() - 1.0)
            drawdown_ok = True
            if config.market_trend_max_strategy_drawdown is not None:
                drawdown_ok = current_drawdown >= -abs(float(config.market_trend_max_strategy_drawdown))
            market_risk_on = market_trend_return >= float(config.market_trend_min_return) and drawdown_ok
            if market_risk_on:
                exposure = max(exposure, market_floor)

        exposure = min(max(exposure, float(config.exposure_min)), float(config.exposure_max))

        overlay_ret = exposure * float(base.loc[dt]) + (1.0 - exposure) * float(bond.loc[dt])
        overlay_returns.append(overlay_ret)
        overlay_nav *= 1.0 + overlay_ret
        overlay_peak = max(overlay_peak, overlay_nav)
        nav_history.append(overlay_nav)

        rows.append(
            {
                "date": dt,
                "base_return": float(base.loc[dt]),
                "bond_return": float(bond.loc[dt]),
                "market_return": float(market.loc[dt]) if market is not None else np.nan,
                "market_trend_return": market_trend_return,
                "market_risk_on": bool(market_risk_on),
                "exposure": exposure,
                "overlay_return": overlay_ret,
                "portfolio_value": overlay_nav,
                "drawdown": overlay_nav / overlay_peak - 1.0,
            }
        )

    return pd.DataFrame(rows).set_index("date")


def summarize_overlay(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {
            "annual_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "avg_exposure": 0.0,
        }

    returns = frame["overlay_return"]
    nav = frame["portfolio_value"]
    days = int((nav.index[-1] - nav.index[0]).days)
    if days <= 0 or float(nav.iloc[-1]) <= 0:
        annual_return = 0.0
    else:
        annual_return = float(nav.iloc[-1]) ** (365 / days) - 1
    sharpe = 0.0
    if returns.std(ddof=0) > 0:
        sharpe = float(returns.mean() / returns.std(ddof=0) * np.sqrt(252))
    max_drawdown = float((nav / nav.cummax() - 1.0).min())
    return {
        "annual_return": annual_return,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe,
        "avg_exposure": float(frame["exposure"].mean()),
    }


def compute_inverse_vol_weights(
    volatilities: pd.Series,
    method: str = "inverse",
    cap_max_weight: float = 0.0,
) -> pd.Series:
    """计算 inverse-volatility 归一化权重。"""
    vols = pd.Series(volatilities, dtype=float).replace([np.inf, -np.inf], np.nan)
    if vols.empty:
        return pd.Series(dtype=float)

    valid = vols.notna() & (vols > 0)
    if not valid.any():
        return pd.Series(1.0 / len(vols), index=vols.index, dtype=float)

    method_key = str(method or "inverse").strip().lower()
    if method_key == "inverse":
        raw = 1.0 / vols[valid]
    elif method_key == "inverse_sqrt":
        raw = 1.0 / np.sqrt(vols[valid])
    else:
        raise ValueError(f"未知 inverse-vol 方法: {method}")

    raw = pd.Series(raw, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if raw.empty or float(raw.sum()) <= 0:
        return pd.Series(1.0 / len(vols), index=vols.index, dtype=float)

    weights = raw / float(raw.sum())

    cap = float(cap_max_weight or 0.0)
    if cap <= 0:
        out = pd.Series(0.0, index=vols.index, dtype=float)
        out.loc[weights.index] = weights
        return out
    if cap >= 1:
        cap = 1.0

    n_valid = len(weights)
    if cap * n_valid < 1.0 - 1e-12:
        out = pd.Series(0.0, index=vols.index, dtype=float)
        out.loc[weights.index] = weights
        return out

    capped = weights.copy()
    locked = pd.Series(False, index=weights.index, dtype=bool)
    remaining = 1.0

    # 迭代 water-filling：超上限先锁定，再把剩余权重按比例分配给未锁定标的。
    for _ in range(n_valid):
        active = capped.index[~locked]
        if len(active) == 0:
            break
        active_sum = float(weights.loc[active].sum())
        if active_sum <= 0:
            capped.loc[active] = 0.0
            break
        trial = weights.loc[active] / active_sum * remaining
        over_cap = trial > cap + 1e-12
        if not over_cap.any():
            capped.loc[active] = trial
            remaining = 0.0
            break
        over_idx = trial.index[over_cap]
        capped.loc[over_idx] = cap
        locked.loc[over_idx] = True
        remaining = 1.0 - float(capped.loc[locked].sum())
        if remaining <= 1e-12:
            capped.loc[~locked] = 0.0
            remaining = 0.0
            break

    if remaining > 1e-12 and (~locked).any():
        active = capped.index[~locked]
        active_sum = float(weights.loc[active].sum())
        if active_sum > 0:
            capped.loc[active] = weights.loc[active] / active_sum * remaining

    capped = capped.clip(lower=0.0)
    total = float(capped.sum())
    if total > 0:
        capped /= total

    out = pd.Series(0.0, index=vols.index, dtype=float)
    out.loc[capped.index] = capped
    return out
