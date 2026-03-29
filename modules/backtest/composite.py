"""
组合策略回测入口

支持把多个已有策略按固定权重混合成一个“母策略”回测结果。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import pandas as pd

from config.config import CONFIG
from core.strategy import Strategy, is_composite_strategy
from core.validity import ValidityConfig, apply_validity_overlay
from modules.backtest.base import BacktestResult


def _results_dir() -> Path:
    return Path(
        CONFIG.get("paths.results", CONFIG.get("results_path", "./results"))
    ).expanduser()


def _scope_tag(universe: str) -> str:
    if universe == "csi300":
        return "historical_csi300"
    if universe == "all":
        return "all_market"
    return "mixed_universe"


def _make_engine(engine: str):
    if engine == "qlib":
        from modules.backtest.qlib_engine import QlibBacktestEngine

        return QlibBacktestEngine()
    if engine == "pybroker":
        from modules.backtest.pybroker_engine import PyBrokerBacktestEngine

        return PyBrokerBacktestEngine()
    raise ValueError(f"不支持的引擎: {engine}")


def _blend_results(
    results: Dict[str, BacktestResult],
    weights: Dict[str, float],
) -> BacktestResult:
    if not results:
        return BacktestResult(
            daily_returns=pd.Series(dtype=float),
            portfolio_value=pd.Series(dtype=float),
        )

    all_dates = sorted(
        {
            dt
            for result in results.values()
            for dt in result.daily_returns.index
        }
    )
    blended_returns = pd.Series(0.0, index=all_dates, dtype=float)

    for name, result in results.items():
        w = float(weights.get(name, 0.0))
        aligned = result.daily_returns.reindex(all_dates).fillna(0.0)
        blended_returns += w * aligned

    portfolio_value = (1 + blended_returns).cumprod()
    return BacktestResult(
        daily_returns=blended_returns,
        portfolio_value=portfolio_value,
        metadata={
            "blended_strategies": list(results.keys()),
            "weights": weights,
        },
    )


def _apply_validity_if_needed(
    strategy: Strategy,
    result: BacktestResult,
) -> Tuple[BacktestResult, Optional[pd.Series], Optional[pd.Series]]:
    validity = getattr(strategy, "validity", None)
    if not isinstance(validity, ValidityConfig) or not validity.apply_in_backtest:
        return result, None, None

    raw_returns = pd.Series(result.daily_returns, copy=True).astype(float)
    adjusted_returns, exposure_factor = apply_validity_overlay(raw_returns, validity)
    adjusted_returns = adjusted_returns.reindex(raw_returns.index).fillna(0.0)
    exposure_factor = exposure_factor.reindex(raw_returns.index).fillna(1.0)

    metadata = dict(result.metadata)
    metadata["validity_overlay"] = {
        "applied": True,
        "lookback_days": validity.lookback_days,
        "min_observations": validity.min_observations,
        "action": validity.action,
        "reduce_to": validity.reduce_to,
        "avg_exposure_factor": float(exposure_factor.mean()) if not exposure_factor.empty else 1.0,
        "min_exposure_factor": float(exposure_factor.min()) if not exposure_factor.empty else 1.0,
    }
    return (
        BacktestResult(
            daily_returns=adjusted_returns,
            portfolio_value=(1.0 + adjusted_returns).cumprod(),
            metadata=metadata,
        ),
        raw_returns,
        exposure_factor,
    )


def _save_strategy_result(
    strategy: Strategy,
    result: BacktestResult,
    component_results: Optional[Dict[str, BacktestResult]] = None,
    raw_returns: Optional[pd.Series] = None,
    exposure_factor: Optional[pd.Series] = None,
) -> str:
    output_dir = _results_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_dates = pd.Index(sorted(result.daily_returns.index))
    df = pd.DataFrame(index=all_dates)
    df.index.name = "date"
    df["return"] = result.daily_returns.reindex(all_dates).fillna(0.0)
    if raw_returns is not None:
        df["raw_return"] = raw_returns.reindex(all_dates).fillna(0.0)
    if exposure_factor is not None:
        df["exposure_factor"] = exposure_factor.reindex(all_dates).fillna(1.0)
    df["portfolio_value"] = result.portfolio_value.reindex(all_dates).ffill()

    for name, child_result in (component_results or {}).items():
        slug = name.replace("/", "__")
        df[f"return__{slug}"] = child_result.daily_returns.reindex(all_dates).fillna(0.0)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scope_tag = _scope_tag(strategy.effective_universe())
    results_file = output_dir / f"backtest_{strategy.artifact_slug()}_{scope_tag}_{timestamp}.csv"
    df.to_csv(results_file)
    return str(results_file)


def run_strategy_backtest(
    strategy: Strategy,
    engine: str = "qlib",
    stack: Optional[Set[str]] = None,
) -> BacktestResult:
    """统一执行单策略/组合策略回测。"""
    if not is_composite_strategy(strategy):
        result = _make_engine(engine).run(strategy=strategy)
        result, raw_returns, exposure_factor = _apply_validity_if_needed(strategy, result)
        if raw_returns is not None:
            result.metadata["results_file"] = _save_strategy_result(
                strategy,
                result,
                raw_returns=raw_returns,
                exposure_factor=exposure_factor,
            )
        return result

    stack = stack or set()
    if strategy.name in stack:
        chain = " -> ".join(list(stack) + [strategy.name])
        raise ValueError(f"检测到组合策略循环引用: {chain}")

    next_stack = set(stack)
    next_stack.add(strategy.name)

    component_results: Dict[str, BacktestResult] = {}
    for child, _ in strategy.load_component_strategies():
        component_results[child.name] = run_strategy_backtest(
            child,
            engine=engine,
            stack=next_stack,
        )

    weight_map = strategy.component_weights()
    blended = _blend_results(component_results, weights=weight_map)
    blended, raw_returns, exposure_factor = _apply_validity_if_needed(strategy, blended)
    blended.metadata.update(
        {
            "strategy_name": strategy.name,
            "universe": strategy.effective_universe(),
            "cash_weight": strategy.cash_weight,
            "component_weights": weight_map,
            "component_results_files": {
                name: child_result.metadata.get("results_file")
                for name, child_result in component_results.items()
            },
        }
    )
    blended.metadata["results_file"] = _save_strategy_result(
        strategy,
        blended,
        component_results,
        raw_returns=raw_returns,
        exposure_factor=exposure_factor,
    )
    return blended
