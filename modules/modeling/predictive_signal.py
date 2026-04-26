"""
最小预测式信号 pipeline

目标：
1. 直接复用 factor_data.parquet 作为特征源
2. 训练一个未来 N 日收益预测模型
3. 产出 MultiIndex(datetime, instrument) -> score
4. 复用现有 extract_topk + QlibBacktestEngine
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
import yaml
from qlib.contrib.data.loader import Alpha158DL
from sklearn.ensemble import HistGradientBoostingRegressor

from config.config import CONFIG
from modules.backtest.base import BacktestResult
from core.qlib_init import init_qlib, load_features_safe
from core.selection import (
    FACTOR_PARQUET,
    _get_factor_parquet_columns,
    _load_total_mv_frame,
    _load_trade_calendar,
    _read_factor_parquet,
    _to_qlib_instruments,
    compute_rebalance_dates,
    extract_topk,
)
from core.universe import get_universe_instruments


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = Path(CONFIG.get("paths.results", "./results")).expanduser()
DEFAULT_FEATURE_EXCLUDES = {"datetime", "instrument", "label", "score", "rank", "symbol", "date"}


def _empty_index() -> pd.MultiIndex:
    return pd.MultiIndex(levels=[[], []], codes=[[], []], names=["datetime", "instrument"])


def _maybe_resolve_path(path_value: str, base_dir: Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def default_feature_columns(columns: Sequence[str]) -> list[str]:
    return sorted(str(col) for col in columns if str(col) not in DEFAULT_FEATURE_EXCLUDES)


def _safe_series_div(num: pd.Series, den: pd.Series, floor: float = 1e-6) -> pd.Series:
    den_safe = den.abs().clip(lower=floor)
    return num / den_safe


def _cross_sectional_rank(series: pd.Series) -> pd.Series:
    return series.groupby(level="datetime", sort=False).rank(pct=True)


def _instrument_diff(series: pd.Series, periods: int = 1) -> pd.Series:
    return series.groupby(level="instrument", sort=False).diff(periods=int(periods))


def _mean_features(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    return frame.loc[:, list(columns)].mean(axis=1)


DERIVED_FEATURE_SPECS = {
    "rank_book_to_market": {
        "deps": ["book_to_market"],
        "fn": lambda f: _cross_sectional_rank(f["book_to_market"]),
    },
    "rank_ebitda_to_mv": {
        "deps": ["ebitda_to_mv"],
        "fn": lambda f: _cross_sectional_rank(f["ebitda_to_mv"]),
    },
    "rank_ebit_to_mv": {
        "deps": ["ebit_to_mv"],
        "fn": lambda f: _cross_sectional_rank(f["ebit_to_mv"]),
    },
    "rank_roe_fina": {
        "deps": ["roe_fina"],
        "fn": lambda f: _cross_sectional_rank(f["roe_fina"]),
    },
    "rank_roe_dt_fina": {
        "deps": ["roe_dt_fina"],
        "fn": lambda f: _cross_sectional_rank(f["roe_dt_fina"]),
    },
    "rank_roic_proxy": {
        "deps": ["roic_proxy"],
        "fn": lambda f: _cross_sectional_rank(f["roic_proxy"]),
    },
    "rank_roa_fina": {
        "deps": ["roa_fina"],
        "fn": lambda f: _cross_sectional_rank(f["roa_fina"]),
    },
    "rank_ocf_to_mv": {
        "deps": ["ocf_to_mv"],
        "fn": lambda f: _cross_sectional_rank(f["ocf_to_mv"]),
    },
    "rank_ocf_to_ev": {
        "deps": ["ocf_to_ev"],
        "fn": lambda f: _cross_sectional_rank(f["ocf_to_ev"]),
    },
    "rank_fcff_to_mv": {
        "deps": ["fcff_to_mv"],
        "fn": lambda f: _cross_sectional_rank(f["fcff_to_mv"]),
    },
    "rank_low_debt": {
        "deps": ["debt_to_assets_fina"],
        "fn": lambda f: _cross_sectional_rank(-f["debt_to_assets_fina"]),
    },
    "rank_current_ratio_fina": {
        "deps": ["current_ratio_fina"],
        "fn": lambda f: _cross_sectional_rank(f["current_ratio_fina"]),
    },
    "rank_quick_ratio_fina": {
        "deps": ["quick_ratio_fina"],
        "fn": lambda f: _cross_sectional_rank(f["quick_ratio_fina"]),
    },
    "rank_smart_ratio_5d": {
        "deps": ["smart_ratio_5d"],
        "fn": lambda f: _cross_sectional_rank(f["smart_ratio_5d"]),
    },
    "rank_net_mf_amount_20d": {
        "deps": ["net_mf_amount_20d"],
        "fn": lambda f: _cross_sectional_rank(f["net_mf_amount_20d"]),
    },
    "rank_net_mf_amount_5d": {
        "deps": ["net_mf_amount_5d"],
        "fn": lambda f: _cross_sectional_rank(f["net_mf_amount_5d"]),
    },
    "rank_net_mf_vol_5d": {
        "deps": ["net_mf_vol_5d"],
        "fn": lambda f: _cross_sectional_rank(f["net_mf_vol_5d"]),
    },
    "rank_total_revenue_inc": {
        "deps": ["total_revenue_inc"],
        "fn": lambda f: _cross_sectional_rank(f["total_revenue_inc"]),
    },
    "rank_operate_profit_inc": {
        "deps": ["operate_profit_inc"],
        "fn": lambda f: _cross_sectional_rank(f["operate_profit_inc"]),
    },
    "rank_n_cashflow_act": {
        "deps": ["n_cashflow_act"],
        "fn": lambda f: _cross_sectional_rank(f["n_cashflow_act"]),
    },
    "rank_net_margin": {
        "deps": ["net_margin"],
        "fn": lambda f: _cross_sectional_rank(f["net_margin"]),
    },
    "rank_liquidity_buffer": {
        "deps": ["current_ratio_fina", "quick_ratio_fina"],
        "fn": lambda f: _cross_sectional_rank(f["current_ratio_fina"] + f["quick_ratio_fina"]),
    },
    "rank_balance_strength": {
        "deps": ["money_cap", "total_liab"],
        "fn": lambda f: _cross_sectional_rank(_safe_series_div(f["money_cap"], f["total_liab"])),
    },
    "rank_earnings_accel": {
        "deps": ["operate_profit_inc", "n_income_inc"],
        "fn": lambda f: _cross_sectional_rank(f["operate_profit_inc"] - f["n_income_inc"]),
    },
    "rank_value_delta_1": {
        "deps": ["book_to_market"],
        "fn": lambda f: _cross_sectional_rank(_instrument_diff(f["book_to_market"])),
    },
    "rank_flow_delta_1": {
        "deps": ["net_mf_amount_20d"],
        "fn": lambda f: _cross_sectional_rank(_instrument_diff(f["net_mf_amount_20d"])),
    },
    "rank_smart_delta_1": {
        "deps": ["smart_ratio_5d"],
        "fn": lambda f: _cross_sectional_rank(_instrument_diff(f["smart_ratio_5d"])),
    },
    "qvf_rank_blend": {
        "deps": [
            "rank_book_to_market",
            "rank_roe_fina",
            "rank_ocf_to_ev",
            "rank_low_debt",
            "rank_smart_ratio_5d",
            "rank_net_mf_amount_20d",
        ],
        "fn": lambda f: (
            f["rank_book_to_market"]
            + f["rank_roe_fina"]
            + f["rank_ocf_to_ev"]
            + f["rank_low_debt"]
            + f["rank_smart_ratio_5d"]
            + f["rank_net_mf_amount_20d"]
        )
        / 6.0,
    },
    "rank_value_profit_core": {
        "deps": ["rank_book_to_market", "rank_ebitda_to_mv", "rank_ebit_to_mv", "rank_roa_fina"],
        "fn": lambda f: (
            f["rank_book_to_market"]
            + f["rank_ebitda_to_mv"]
            + f["rank_ebit_to_mv"]
            + f["rank_roa_fina"]
        )
        / 4.0,
    },
    "rank_flow_momentum_core": {
        "deps": ["rank_net_mf_amount_20d", "rank_net_mf_amount_5d", "rank_smart_ratio_5d"],
        "fn": lambda f: (
            f["rank_net_mf_amount_20d"]
            + f["rank_net_mf_amount_5d"]
            + f["rank_smart_ratio_5d"]
        )
        / 3.0,
    },
    "rank_growth_quality_core": {
        "deps": ["rank_total_revenue_inc", "rank_operate_profit_inc", "rank_net_margin"],
        "fn": lambda f: (
            f["rank_total_revenue_inc"]
            + f["rank_operate_profit_inc"]
            + f["rank_net_margin"]
        )
        / 3.0,
    },
    "rank_balance_core": {
        "deps": ["rank_liquidity_buffer", "rank_low_debt", "rank_balance_strength"],
        "fn": lambda f: (
            f["rank_liquidity_buffer"]
            + f["rank_low_debt"]
            + f["rank_balance_strength"]
        )
        / 3.0,
    },
    "rank_value_cashflow_core": {
        "deps": [
            "rank_book_to_market",
            "rank_ebit_to_mv",
            "rank_ebitda_to_mv",
            "rank_ocf_to_mv",
            "rank_ocf_to_ev",
            "rank_fcff_to_mv",
        ],
        "fn": lambda f: _mean_features(
            f,
            [
                "rank_book_to_market",
                "rank_ebit_to_mv",
                "rank_ebitda_to_mv",
                "rank_ocf_to_mv",
                "rank_ocf_to_ev",
                "rank_fcff_to_mv",
            ],
        ),
    },
    "rank_profitability_quality_core": {
        "deps": [
            "rank_roic_proxy",
            "rank_roe_fina",
            "rank_roe_dt_fina",
            "rank_roa_fina",
            "rank_net_margin",
        ],
        "fn": lambda f: _mean_features(
            f,
            [
                "rank_roic_proxy",
                "rank_roe_fina",
                "rank_roe_dt_fina",
                "rank_roa_fina",
                "rank_net_margin",
            ],
        ),
    },
    "rank_balance_sheet_core": {
        "deps": [
            "rank_current_ratio_fina",
            "rank_quick_ratio_fina",
            "rank_low_debt",
            "rank_balance_strength",
        ],
        "fn": lambda f: _mean_features(
            f,
            [
                "rank_current_ratio_fina",
                "rank_quick_ratio_fina",
                "rank_low_debt",
                "rank_balance_strength",
            ],
        ),
    },
    "rank_growth_cashflow_core": {
        "deps": ["rank_total_revenue_inc", "rank_operate_profit_inc", "rank_n_cashflow_act"],
        "fn": lambda f: _mean_features(
            f,
            ["rank_total_revenue_inc", "rank_operate_profit_inc", "rank_n_cashflow_act"],
        ),
    },
    "rank_flow_liquidity_core": {
        "deps": [
            "rank_smart_ratio_5d",
            "rank_net_mf_amount_5d",
            "rank_net_mf_amount_20d",
            "rank_net_mf_vol_5d",
        ],
        "fn": lambda f: _mean_features(
            f,
            [
                "rank_smart_ratio_5d",
                "rank_net_mf_amount_5d",
                "rank_net_mf_amount_20d",
                "rank_net_mf_vol_5d",
            ],
        ),
    },
    "quality_value_interaction": {
        "deps": ["rank_book_to_market", "rank_roe_fina"],
        "fn": lambda f: f["rank_book_to_market"] * f["rank_roe_fina"],
    },
    "cashflow_flow_interaction": {
        "deps": ["rank_ocf_to_ev", "rank_smart_ratio_5d"],
        "fn": lambda f: f["rank_ocf_to_ev"] * f["rank_smart_ratio_5d"],
    },
    "fundamental_momentum_interaction": {
        "deps": ["rank_earnings_accel", "rank_liquidity_buffer"],
        "fn": lambda f: f["rank_earnings_accel"] * f["rank_liquidity_buffer"],
    },
    "qvf_dynamic_blend": {
        "deps": ["qvf_rank_blend", "rank_value_delta_1", "rank_flow_delta_1", "rank_smart_delta_1"],
        "fn": lambda f: (
            f["qvf_rank_blend"]
            + f["rank_value_delta_1"].fillna(0.5)
            + f["rank_flow_delta_1"].fillna(0.5)
            + f["rank_smart_delta_1"].fillna(0.5)
        )
        / 4.0,
    },
    "qvf_core_alpha": {
        "deps": [
            "rank_value_profit_core",
            "rank_flow_momentum_core",
            "rank_growth_quality_core",
            "rank_balance_core",
        ],
        "fn": lambda f: (
            0.35 * f["rank_value_profit_core"]
            + 0.35 * f["rank_flow_momentum_core"]
            + 0.20 * f["rank_growth_quality_core"]
            + 0.10 * f["rank_balance_core"]
        ),
    },
    "qvf_core_interaction": {
        "deps": ["rank_value_profit_core", "rank_flow_momentum_core"],
        "fn": lambda f: f["rank_value_profit_core"] * f["rank_flow_momentum_core"],
    },
    "qvf_core_dynamic": {
        "deps": ["qvf_core_alpha", "rank_value_delta_1", "rank_flow_delta_1"],
        "fn": lambda f: (
            f["qvf_core_alpha"]
            + f["rank_value_delta_1"].fillna(0.5)
            + f["rank_flow_delta_1"].fillna(0.5)
        )
        / 3.0,
    },
    "qvf_group_alpha": {
        "deps": [
            "rank_value_cashflow_core",
            "rank_profitability_quality_core",
            "rank_balance_sheet_core",
            "rank_growth_cashflow_core",
            "rank_flow_liquidity_core",
        ],
        "fn": lambda f: (
            0.26 * f["rank_value_cashflow_core"]
            + 0.22 * f["rank_profitability_quality_core"]
            + 0.12 * f["rank_balance_sheet_core"]
            + 0.16 * f["rank_growth_cashflow_core"]
            + 0.24 * f["rank_flow_liquidity_core"]
        ),
    },
    "qvf_group_interaction": {
        "deps": ["rank_value_cashflow_core", "rank_flow_liquidity_core"],
        "fn": lambda f: f["rank_value_cashflow_core"] * f["rank_flow_liquidity_core"],
    },
    "qvf_group_quality_anchor": {
        "deps": [
            "rank_profitability_quality_core",
            "rank_balance_sheet_core",
            "rank_growth_cashflow_core",
        ],
        "fn": lambda f: _mean_features(
            f,
            [
                "rank_profitability_quality_core",
                "rank_balance_sheet_core",
                "rank_growth_cashflow_core",
            ],
        ),
    },
    "qvf_financial_gate": {
        "deps": [
            "rank_value_cashflow_core",
            "rank_profitability_quality_core",
            "rank_balance_sheet_core",
            "rank_growth_cashflow_core",
        ],
        "fn": lambda f: (
            0.30 * f["rank_value_cashflow_core"]
            + 0.30 * f["rank_profitability_quality_core"]
            + 0.20 * f["rank_balance_sheet_core"]
            + 0.20 * f["rank_growth_cashflow_core"]
        ),
    },
    "qvf_group_blend": {
        "deps": [
            "qvf_group_alpha",
            "qvf_group_interaction",
            "qvf_group_quality_anchor",
            "qvf_financial_gate",
        ],
        "fn": lambda f: (
            0.50 * f["qvf_group_alpha"]
            + 0.20 * f["qvf_group_interaction"]
            + 0.10 * f["qvf_group_quality_anchor"]
            + 0.20 * f["qvf_financial_gate"]
        ),
    },
}


def derived_feature_names() -> set[str]:
    return set(DERIVED_FEATURE_SPECS)


def required_raw_columns(feature_columns: Sequence[str]) -> list[str]:
    required = set()

    def walk(name: str):
        if name in DERIVED_FEATURE_SPECS:
            for dep in DERIVED_FEATURE_SPECS[name]["deps"]:
                walk(dep)
        else:
            required.add(name)

    for col in feature_columns:
        walk(str(col))
    return sorted(required)


def augment_with_derived_features(
    feature_frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    work = feature_frame.copy()

    def ensure(name: str):
        if name in work.columns:
            return
        if name not in DERIVED_FEATURE_SPECS:
            raise KeyError(f"未知特征列: {name}")
        spec = DERIVED_FEATURE_SPECS[name]
        for dep in spec["deps"]:
            ensure(dep)
        work[name] = spec["fn"](work)

    for col in feature_columns:
        ensure(str(col))
    return work


def load_predictive_config(config_path: str | Path) -> dict:
    path = Path(config_path).expanduser().resolve()
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    cfg.setdefault("name", path.stem)
    cfg["_config_path"] = str(path)

    output = cfg.setdefault("output", {})
    output_root = output.get("root")
    if output_root:
        output["root"] = str(_maybe_resolve_path(str(output_root), PROJECT_ROOT))
    else:
        output["root"] = str((DEFAULT_RESULTS_ROOT / "model_signals" / cfg["name"]).resolve())

    data = cfg.setdefault("data", {})
    data.setdefault("start_date", "2019-01-01")
    data.setdefault("end_date", CONFIG.get("end_date", pd.Timestamp.today().strftime("%Y-%m-%d")))

    label = cfg.setdefault("label", {})
    label.setdefault("horizon_days", 10)

    selection = cfg.setdefault("selection", {})
    selection.setdefault("freq", "biweek")
    selection.setdefault("topk", 8)
    selection.setdefault("universe", "csi300")
    selection.setdefault("min_market_cap", 0)
    selection.setdefault("exclude_st", True)
    selection.setdefault("exclude_new_days", 60)
    selection.setdefault("sticky", 0)
    selection.setdefault("threshold", 0.0)
    selection.setdefault("churn_limit", 0)
    selection.setdefault("margin_stable", False)
    selection.setdefault("buffer", 0)
    selection.setdefault("mode", "factor_topk")
    selection.setdefault("score_smoothing_days", 1)
    selection.setdefault("entry_persist_days", 1)
    selection.setdefault("exit_persist_days", 1)
    selection.setdefault("min_hold_days", 0)
    selection.setdefault("stoploss_lookback_days", 20)
    selection.setdefault("stoploss_drawdown", 0.10)
    selection.setdefault("replacement_pool_size", 0)

    training = cfg.setdefault("training", {})
    training.setdefault("train_start", data["start_date"])
    training.setdefault("train_end", "2022-12-31")
    training.setdefault("valid_start", "2023-01-01")
    training.setdefault("valid_end", "2023-12-31")

    scoring = cfg.setdefault("scoring", {})
    scoring.setdefault("start_date", "2024-01-01")
    scoring.setdefault("end_date", data["end_date"])

    model = cfg.setdefault("model", {})
    model.setdefault("preferred_backend", "lightgbm")
    model.setdefault(
        "params",
        {
            "learning_rate": 0.05,
            "n_estimators": 300,
            "max_depth": 4,
            "min_samples_leaf": 32,
            "random_state": 42,
        },
    )

    position = cfg.setdefault("position", {})
    position.setdefault("model", "fixed")
    position.setdefault("params", {"stock_pct": 1.0})

    trading = cfg.setdefault("trading", {})
    trading.setdefault("buy_commission_rate", 0.0003)
    trading.setdefault("sell_commission_rate", 0.0003)
    trading.setdefault("sell_stamp_tax_rate", 0.0010)
    trading.setdefault("min_buy_commission", 5.0)
    trading.setdefault("min_sell_commission", 5.0)
    trading.setdefault("slippage_bps", 5.0)
    trading.setdefault("impact_bps", 5.0)

    overlay = cfg.setdefault("overlay", {})
    overlay.setdefault("enabled", False)
    overlay.setdefault("vol_lookback", 20)
    overlay.setdefault("trend_lookback", 0)
    overlay.setdefault("trend_exposure", 0.90)
    overlay.setdefault("market_trend_lookback", 0)
    overlay.setdefault("market_trend_min_return", 0.0)
    overlay.setdefault("market_trend_exposure_floor", 0.0)
    overlay.setdefault("market_trend_max_strategy_drawdown", None)
    overlay.setdefault("exposure_min", 0.0)
    overlay.setdefault("exposure_max", 1.0)

    walk_forward = cfg.setdefault("walk_forward", {})
    walk_forward.setdefault("enabled", False)
    walk_forward.setdefault("min_train_years", 3)
    walk_forward.setdefault("retrain_months", 12)
    walk_forward.setdefault("train_years", None)

    return cfg


def output_root(cfg: dict) -> Path:
    root = Path(cfg["output"]["root"]).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root


def model_bundle_path(cfg: dict) -> Path:
    return output_root(cfg) / "model_bundle.pkl"


def scores_path(cfg: dict) -> Path:
    return output_root(cfg) / "scores.parquet"


def selection_path(cfg: dict) -> Path:
    return output_root(cfg) / "selections.csv"


def training_summary_path(cfg: dict) -> Path:
    return output_root(cfg) / "training_summary.json"


def scoring_summary_path(cfg: dict) -> Path:
    return output_root(cfg) / "scoring_summary.json"


def backtest_summary_path(cfg: dict) -> Path:
    return output_root(cfg) / "backtest_summary.json"


def overlay_results_path(cfg: dict) -> Path:
    return output_root(cfg) / "overlay_results.csv"


def load_bond_overlay_returns():
    from modules.backtest.qlib_engine import _load_bond_etf_returns

    return _load_bond_etf_returns()


def load_market_overlay_returns(index_code: str = "000300.SH") -> Optional[pd.Series]:
    index_path = PROJECT_ROOT / "data" / "tushare" / "index_daily.parquet"
    if not index_path.exists():
        return None
    frame = pd.read_parquet(index_path, columns=["ts_code", "trade_date", "close"])
    frame = frame.loc[frame["ts_code"].astype(str) == str(index_code), ["trade_date", "close"]].copy()
    if frame.empty:
        return None
    frame["date"] = pd.to_datetime(frame["trade_date"].astype(str))
    close = frame.sort_values("date").drop_duplicates("date").set_index("date")["close"].astype(float)
    return close.pct_change().fillna(0.0)


def apply_overlay_to_backtest_result(result: BacktestResult, cfg: dict) -> tuple[BacktestResult, Optional[pd.DataFrame]]:
    overlay_cfg = dict(cfg.get("overlay", {}))
    if not bool(overlay_cfg.get("enabled", False)):
        return result, None

    from modules.modeling.portfolio_overlay import OverlayConfig, compute_overlay_frame

    bond_returns = load_bond_overlay_returns()
    if bond_returns is None:
        bond_returns = 0.03 / 252

    config = OverlayConfig(
        target_vol=overlay_cfg.get("target_vol"),
        vol_lookback=int(overlay_cfg.get("vol_lookback", 20)),
        dd_soft=overlay_cfg.get("dd_soft"),
        dd_hard=overlay_cfg.get("dd_hard"),
        soft_exposure=float(overlay_cfg.get("soft_exposure", 0.95)),
        hard_exposure=float(overlay_cfg.get("hard_exposure", 0.80)),
        trend_lookback=int(overlay_cfg.get("trend_lookback", 0)),
        trend_exposure=float(overlay_cfg.get("trend_exposure", 0.90)),
        market_trend_lookback=int(overlay_cfg.get("market_trend_lookback", 0)),
        market_trend_min_return=float(overlay_cfg.get("market_trend_min_return", 0.0)),
        market_trend_exposure_floor=float(overlay_cfg.get("market_trend_exposure_floor", 0.0)),
        market_trend_max_strategy_drawdown=overlay_cfg.get("market_trend_max_strategy_drawdown"),
        exposure_min=float(overlay_cfg.get("exposure_min", 0.0)),
        exposure_max=float(overlay_cfg.get("exposure_max", 1.0)),
    )
    market_returns = None
    if int(overlay_cfg.get("market_trend_lookback", 0) or 0) > 0:
        market_returns = load_market_overlay_returns(str(overlay_cfg.get("market_index_code", "000300.SH")))
    frame = compute_overlay_frame(result.daily_returns, bond_returns, config, market_returns=market_returns)
    overlay_result = BacktestResult(
        daily_returns=frame["overlay_return"].astype(float),
        portfolio_value=frame["portfolio_value"].astype(float),
        metadata=dict(result.metadata),
    )
    overlay_result.metadata["base_results_file"] = result.metadata.get("results_file")
    overlay_result.metadata["overlay"] = {"enabled": True, **overlay_cfg}
    return overlay_result, frame


def _alpha158_feature_map(alpha158_cfg: Optional[Dict[str, Any]] = None) -> dict[str, str]:
    fields, names = Alpha158DL.get_feature_config(alpha158_cfg or {})
    return {str(name): str(field) for field, name in zip(fields, names)}


def load_alpha158_feature_frame(
    start_date: str,
    end_date: str,
    rebalance_freq: str,
    feature_columns: Sequence[str],
    alpha158_cfg: Optional[Dict[str, Any]] = None,
    instruments: str = "all",
) -> tuple[pd.DataFrame, pd.DatetimeIndex, list[str]]:
    feature_map = _alpha158_feature_map(alpha158_cfg)
    selected_columns = list(feature_columns)
    missing = sorted(set(selected_columns) - set(feature_map))
    if missing:
        raise ValueError(f"Alpha158 不支持特征列: {missing}")

    calendar = _load_trade_calendar(start_date, end_date)
    rebalance_dates = compute_rebalance_dates(pd.Series(calendar), freq=rebalance_freq)
    qlib_instruments = instruments or "all"
    init_qlib()
    if isinstance(qlib_instruments, str):
        from qlib.data import D

        qlib_instruments = D.instruments(market=qlib_instruments)
    df = load_features_safe(
        qlib_instruments,
        [feature_map[col] for col in selected_columns],
        start_time=start_date,
        end_time=end_date,
        freq="day",
    )
    if df.empty:
        return pd.DataFrame(index=_empty_index(), columns=selected_columns), rebalance_dates, selected_columns

    df.columns = selected_columns
    if list(df.index.names) == ["instrument", "datetime"]:
        df = df.swaplevel().sort_index()
    else:
        df = df.sort_index()
    df = df[df.index.get_level_values("datetime").isin(rebalance_dates)].copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    return df, rebalance_dates, selected_columns


def _resolve_alpha158_instruments(
    start_date: str,
    end_date: str,
    data_cfg: Optional[Dict[str, Any]] = None,
    selection_cfg: Optional[Dict[str, Any]] = None,
):
    data_cfg = dict(data_cfg or {})
    selection_cfg = dict(selection_cfg or {})
    universe = str(data_cfg.get("alpha158_universe", selection_cfg.get("universe", "all")))
    if universe == "all":
        return "all"
    return get_universe_instruments(
        start_date=start_date,
        end_date=end_date,
        universe=universe,
    )


def load_parquet_feature_frame(
    start_date: str,
    end_date: str,
    rebalance_freq: str,
    feature_columns: Optional[Sequence[str]] = None,
) -> tuple[pd.DataFrame, pd.DatetimeIndex, list[str]]:
    if not FACTOR_PARQUET.exists():
        raise FileNotFoundError(f"缺少特征文件: {FACTOR_PARQUET}")

    available_columns = _get_factor_parquet_columns()
    selected_columns = list(feature_columns) if feature_columns else default_feature_columns(available_columns)
    raw_columns = required_raw_columns(selected_columns)
    missing = sorted(set(raw_columns) - set(available_columns))
    if missing:
        raise ValueError(f"factor_data.parquet 缺少特征列: {missing}")

    raw = _read_factor_parquet(
        ["datetime", "instrument"] + raw_columns,
        start_date=start_date,
        end_date=end_date,
    )
    if raw.empty:
        return pd.DataFrame(index=_empty_index(), columns=selected_columns), pd.DatetimeIndex([]), selected_columns

    raw["datetime"] = pd.to_datetime(raw["datetime"])
    raw["instrument"] = _to_qlib_instruments(raw["instrument"])
    raw = raw.drop_duplicates(subset=["datetime", "instrument"], keep="last")

    calendar = _load_trade_calendar(start_date, end_date)
    rebalance_dates = compute_rebalance_dates(pd.Series(calendar), freq=rebalance_freq)
    raw = raw[raw["datetime"].isin(rebalance_dates)].copy()

    if raw.empty:
        return pd.DataFrame(index=_empty_index(), columns=selected_columns), rebalance_dates, selected_columns

    for col in raw_columns:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    frame = raw.set_index(["datetime", "instrument"]).sort_index()[raw_columns]
    frame = frame.replace([np.inf, -np.inf], np.nan)
    if any(col in DERIVED_FEATURE_SPECS for col in selected_columns):
        frame = augment_with_derived_features(frame, selected_columns)
    frame = frame.loc[:, selected_columns]
    return frame, rebalance_dates, selected_columns


def load_feature_frame(
    start_date: str,
    end_date: str,
    rebalance_freq: str,
    feature_columns: Optional[Sequence[str]] = None,
    data_cfg: Optional[Dict[str, Any]] = None,
    selection_cfg: Optional[Dict[str, Any]] = None,
) -> tuple[pd.DataFrame, pd.DatetimeIndex, list[str]]:
    data_cfg = dict(data_cfg or {})
    source = str(data_cfg.get("source", "parquet")).lower()

    if source == "alpha158":
        selected_columns = list(feature_columns or data_cfg.get("feature_columns", []))
        if not selected_columns:
            raise ValueError("Alpha158 模式必须显式提供 feature_columns")
        instruments = _resolve_alpha158_instruments(
            start_date=start_date,
            end_date=end_date,
            data_cfg=data_cfg,
            selection_cfg=selection_cfg,
        )
        return load_alpha158_feature_frame(
            start_date=start_date,
            end_date=end_date,
            rebalance_freq=rebalance_freq,
            feature_columns=selected_columns,
            alpha158_cfg=data_cfg.get("alpha158"),
            instruments=instruments,
        )

    if source == "hybrid":
        parquet_columns = list(data_cfg.get("parquet_feature_columns", []))
        alpha_columns = list(data_cfg.get("alpha158_feature_columns", []))
        if not parquet_columns or not alpha_columns:
            raise ValueError("hybrid 模式必须同时提供 parquet_feature_columns 和 alpha158_feature_columns")
        parquet_frame, parquet_dates, _ = load_parquet_feature_frame(
            start_date=start_date,
            end_date=end_date,
            rebalance_freq=rebalance_freq,
            feature_columns=parquet_columns,
        )
        alpha_frame, alpha_dates, _ = load_alpha158_feature_frame(
            start_date=start_date,
            end_date=end_date,
            rebalance_freq=rebalance_freq,
            feature_columns=alpha_columns,
            alpha158_cfg=data_cfg.get("alpha158"),
            instruments=_resolve_alpha158_instruments(
                start_date=start_date,
                end_date=end_date,
                data_cfg=data_cfg,
                selection_cfg=selection_cfg,
            ),
        )
        selected_columns = parquet_columns + alpha_columns
        frame = parquet_frame.join(alpha_frame, how="inner")
        frame = frame.loc[:, selected_columns] if not frame.empty else pd.DataFrame(
            index=_empty_index(),
            columns=selected_columns,
        )
        return frame, parquet_dates.intersection(alpha_dates), selected_columns

    return load_parquet_feature_frame(
        start_date=start_date,
        end_date=end_date,
        rebalance_freq=rebalance_freq,
        feature_columns=feature_columns,
    )


def _extend_end_date(end_date: str, horizon_days: int) -> str:
    calendar = _load_trade_calendar()
    if len(calendar) == 0:
        return end_date

    ts = pd.Timestamp(end_date)
    pos = int(calendar.searchsorted(ts, side="right") - 1)
    pos = max(pos, 0)
    target = min(pos + max(int(horizon_days), 0), len(calendar) - 1)
    return pd.Timestamp(calendar[target]).strftime("%Y-%m-%d")


def load_close_series(
    instruments: Sequence[str],
    start_date: str,
    end_date: str,
    horizon_days: int,
) -> pd.Series:
    if not instruments:
        return pd.Series(dtype=float, index=_empty_index(), name="close")

    init_qlib()
    close_end_date = _extend_end_date(end_date, horizon_days)
    df_close = load_features_safe(
        list(instruments),
        ["$close"],
        start_time=start_date,
        end_time=close_end_date,
        freq="day",
    )
    if df_close.empty:
        return pd.Series(dtype=float, index=_empty_index(), name="close")

    df_close.columns = ["close"]
    close_series = df_close["close"].astype(float)
    if list(close_series.index.names) == ["instrument", "datetime"]:
        close_series = close_series.swaplevel().sort_index()
    else:
        close_series = close_series.sort_index()
    close_series.name = "close"
    return close_series


def build_forward_return_labels(close_series: pd.Series, horizon_days: int) -> pd.Series:
    if close_series.empty:
        return pd.Series(dtype=float, index=_empty_index(), name="label")

    work = close_series.sort_index()
    future_close = work.groupby(level="instrument").shift(-int(horizon_days))
    labels = future_close / work - 1.0
    labels.name = "label"
    return labels


def assemble_labeled_frame(
    feature_frame: pd.DataFrame,
    close_series: pd.Series,
    horizon_days: int,
) -> pd.DataFrame:
    labels = build_forward_return_labels(close_series, horizon_days)
    combined = feature_frame.copy()
    combined["label"] = labels.reindex(combined.index)
    return combined


def slice_frame_by_date(frame: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    dt_index = frame.index.get_level_values("datetime")
    mask = (dt_index >= pd.Timestamp(start_date)) & (dt_index <= pd.Timestamp(end_date))
    return frame.loc[mask].copy()


def resolve_regressor(
    preferred_backend: str = "lightgbm",
    params: Optional[Dict[str, Any]] = None,
):
    params = dict(params or {})
    backend = str(preferred_backend or "lightgbm").lower()

    if backend in {"lightgbm", "lgbm", "lgb"}:
        try:
            from lightgbm import LGBMRegressor

            lgb_params = dict(params)
            if "min_samples_leaf" in lgb_params and "min_child_samples" not in lgb_params:
                lgb_params["min_child_samples"] = int(lgb_params["min_samples_leaf"])
                lgb_params.pop("min_samples_leaf", None)
            # 避免在受限环境里触发 joblib 对物理核心数的探测告警。
            lgb_params.setdefault("n_jobs", 1)
            model = LGBMRegressor(**lgb_params)
            return model, "lightgbm"
        except Exception:
            pass

    if backend in {"catboost", "cb", "cat"}:
        try:
            from catboost import CatBoostRegressor

            cb_params = {
                "learning_rate": float(params.get("learning_rate", 0.05)),
                "iterations": int(params.get("n_estimators", params.get("iterations", 200))),
                "depth": int(params.get("max_depth", 6)),
                "random_seed": int(params.get("random_state", 42)),
                "verbose": 0,
                "allow_writing_files": False,
            }
            # Map optional params
            for src, dst in [
                ("l2_leaf_reg", "l2_leaf_reg"),
                ("min_child_samples", "min_data_in_leaf"),
            ]:
                if src in params:
                    cb_params[dst] = int(params[src])
            model = CatBoostRegressor(**cb_params)
            return model, "catboost"
        except Exception:
            pass

    hist_params = {
        "learning_rate": float(params.get("learning_rate", 0.05)),
        "max_iter": int(params.get("max_iter", params.get("n_estimators", 300))),
        "max_depth": params.get("max_depth"),
        "max_leaf_nodes": params.get("max_leaf_nodes"),
        "min_samples_leaf": int(params.get("min_samples_leaf", params.get("min_child_samples", 20))),
        "l2_regularization": float(params.get("l2_regularization", 0.0)),
        "random_state": int(params.get("random_state", 42)),
    }
    hist_params = {k: v for k, v in hist_params.items() if v is not None}
    return HistGradientBoostingRegressor(**hist_params), "sklearn_hist_gbm"


def fit_regressor(
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    preferred_backend: str,
    params: Optional[Dict[str, Any]] = None,
) -> tuple[dict, dict]:
    train_clean = train_frame.dropna(subset=["label"])
    if train_clean.empty:
        raise ValueError("训练集为空，无法训练模型")

    model, backend = resolve_regressor(preferred_backend=preferred_backend, params=params)
    x_train = train_clean.loc[:, list(feature_columns)]
    y_train = train_clean["label"].astype(float)
    model.fit(x_train, y_train)

    metrics = {
        "train_rows": int(len(train_clean)),
        "train_dates": int(train_clean.index.get_level_values("datetime").nunique()),
        "backend": backend,
    }

    valid_clean = valid_frame.dropna(subset=["label"])
    if not valid_clean.empty:
        valid_scores = predict_signal_frame(model, valid_clean, feature_columns)
        metrics.update(
            {
                f"valid_{key}": value
                for key, value in evaluate_cross_sectional_predictions(
                    valid_scores, valid_clean["label"]
                ).items()
            }
        )
    else:
        metrics.update({"valid_samples": 0, "valid_dated_groups": 0, "valid_mean_rank_ic": 0.0})

    bundle = {
        "backend": backend,
        "model": model,
        "feature_columns": list(feature_columns),
        "label_horizon_days": None,
        "metrics": metrics,
    }
    return bundle, metrics


def predict_signal_frame(
    model: Any,
    feature_frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> pd.Series:
    if feature_frame.empty:
        return pd.Series(dtype=float, index=_empty_index(), name="score")
    scores = model.predict(feature_frame.loc[:, list(feature_columns)])
    return pd.Series(np.asarray(scores, dtype=float), index=feature_frame.index, name="score").sort_index()


def evaluate_cross_sectional_predictions(scores: pd.Series, labels: pd.Series) -> dict:
    merged = pd.concat([scores.rename("score"), labels.rename("label")], axis=1).dropna()
    if merged.empty:
        return {
            "samples": 0,
            "dated_groups": 0,
            "mean_rank_ic": 0.0,
            "median_rank_ic": 0.0,
            "mean_pearson_ic": 0.0,
        }

    def _date_metrics(group: pd.DataFrame) -> pd.Series:
        if len(group) < 2:
            return pd.Series({"rank_ic": np.nan, "pearson_ic": np.nan})
        return pd.Series(
            {
                "rank_ic": group["score"].corr(group["label"], method="spearman"),
                "pearson_ic": group["score"].corr(group["label"], method="pearson"),
            }
        )

    per_date = merged.groupby(level="datetime", sort=True).apply(_date_metrics)
    valid_rank_ic = per_date["rank_ic"].dropna()
    valid_pearson_ic = per_date["pearson_ic"].dropna()
    return {
        "samples": int(len(merged)),
        "dated_groups": int(len(valid_rank_ic)),
        "mean_rank_ic": float(valid_rank_ic.mean()) if not valid_rank_ic.empty else 0.0,
        "median_rank_ic": float(valid_rank_ic.median()) if not valid_rank_ic.empty else 0.0,
        "mean_pearson_ic": float(valid_pearson_ic.mean()) if not valid_pearson_ic.empty else 0.0,
    }


def save_model_bundle(bundle: dict, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wb") as f:
        pickle.dump(bundle, f)
    return output


def load_model_bundle(path: str | Path) -> dict:
    with open(Path(path), "rb") as f:
        return pickle.load(f)


def save_scores(scores: pd.Series, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    scores.rename("score").reset_index().to_parquet(output, index=False)
    return output


def save_json(payload: dict, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_json_ready(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output


def _selection_filter_columns(selection_cfg: Dict[str, Any]) -> list[str]:
    requested = set()
    hard_filters = selection_cfg.get("hard_filters") or {}
    hard_filter_quantiles = selection_cfg.get("hard_filter_quantiles") or {}
    requested.update(str(col) for col in hard_filters.keys())
    requested.update(str(col) for col in hard_filter_quantiles.keys())
    industry_leader_field = selection_cfg.get("industry_leader_field")
    if industry_leader_field:
        requested.add(str(industry_leader_field))
    return sorted(requested)


def _load_selection_filter_frame(
    candidate_instruments: Sequence[str],
    start_date: str,
    end_date: str,
    rebalance_dates: pd.DatetimeIndex,
    selection_cfg: Dict[str, Any],
) -> Optional[pd.DataFrame]:
    selected_columns = _selection_filter_columns(selection_cfg)
    if not selected_columns:
        return None
    if not FACTOR_PARQUET.exists():
        raise FileNotFoundError(f"缺少特征文件，无法执行选股 gate: {FACTOR_PARQUET}")

    available_columns = _get_factor_parquet_columns()
    raw_columns = required_raw_columns(selected_columns)
    missing = sorted(set(raw_columns) - set(available_columns))
    if missing:
        raise ValueError(f"factor_data.parquet 缺少选股 gate 所需列: {missing}")

    raw = _read_factor_parquet(
        ["datetime", "instrument"] + raw_columns,
        start_date=start_date,
        end_date=end_date,
        instruments=list(candidate_instruments),
    )
    if raw.empty:
        return pd.DataFrame(index=_empty_index(), columns=selected_columns)

    raw["datetime"] = pd.to_datetime(raw["datetime"])
    raw["instrument"] = _to_qlib_instruments(raw["instrument"])
    raw = raw.drop_duplicates(subset=["datetime", "instrument"], keep="last")
    raw = raw[raw["datetime"].isin(rebalance_dates)].copy()
    if raw.empty:
        return pd.DataFrame(index=_empty_index(), columns=selected_columns)

    for col in raw_columns:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    frame = raw.set_index(["datetime", "instrument"]).sort_index()[raw_columns]
    frame = frame.replace([np.inf, -np.inf], np.nan)
    if any(col in DERIVED_FEATURE_SPECS for col in selected_columns):
        frame = augment_with_derived_features(frame, selected_columns)
    return frame.loc[:, selected_columns]


def materialize_selections_from_scores(
    scores: pd.Series,
    rebalance_dates: pd.DatetimeIndex,
    selection_cfg: Dict[str, Any],
) -> pd.DataFrame:
    if scores.empty:
        return pd.DataFrame(columns=["date", "rank", "symbol", "score"])

    mv_series = None
    mv_floor = 0.0
    close_series = None
    hard_filter_data = None
    min_market_cap = float(selection_cfg.get("min_market_cap", 0) or 0)
    if min_market_cap > 0:
        candidate_instruments = sorted(scores.index.get_level_values("instrument").unique().tolist())
        date_index = scores.index.get_level_values("datetime")
        start_date = pd.Timestamp(date_index.min()).strftime("%Y-%m-%d")
        end_date = pd.Timestamp(date_index.max()).strftime("%Y-%m-%d")
        total_mv_frame = _load_total_mv_frame(
            instruments=candidate_instruments,
            start_date=start_date,
            end_date=end_date,
        )
        mv_df = total_mv_frame[total_mv_frame["datetime"].isin(rebalance_dates)]
        if not mv_df.empty:
            mv_series = mv_df.set_index(["datetime", "symbol"])["total_mv"]
            mv_floor = min_market_cap * 10000
    else:
        candidate_instruments = sorted(scores.index.get_level_values("instrument").unique().tolist())
        date_index = scores.index.get_level_values("datetime")
        start_date = pd.Timestamp(date_index.min()).strftime("%Y-%m-%d")
        end_date = pd.Timestamp(date_index.max()).strftime("%Y-%m-%d")

    hard_filter_data = _load_selection_filter_frame(
        candidate_instruments=candidate_instruments,
        start_date=start_date,
        end_date=end_date,
        rebalance_dates=rebalance_dates,
        selection_cfg=selection_cfg,
    )

    selection_mode = str(selection_cfg.get("mode", "factor_topk"))
    if selection_mode == "stoploss_replace":
        close_series = load_close_series(
            instruments=candidate_instruments,
            start_date=start_date,
            end_date=end_date,
            horizon_days=0,
        )

    return extract_topk(
        scores,
        rebalance_dates,
        topk=int(selection_cfg.get("topk", 8)),
        mv_floor=mv_floor,
        mv_series=mv_series,
        sticky=int(selection_cfg.get("sticky", 0)),
        threshold=float(selection_cfg.get("threshold", 0.0)),
        churn_limit=int(selection_cfg.get("churn_limit", 0)),
        margin_stable=bool(selection_cfg.get("margin_stable", False)),
        buffer=int(selection_cfg.get("buffer", 0)),
        exclude_new_days=int(selection_cfg.get("exclude_new_days", 0)),
        exclude_st=bool(selection_cfg.get("exclude_st", False)),
        universe=str(selection_cfg.get("universe", "all")),
        selection_mode=selection_mode,
        hard_filters=selection_cfg.get("hard_filters"),
        hard_filter_quantiles=selection_cfg.get("hard_filter_quantiles"),
        industry_leader_field=selection_cfg.get("industry_leader_field"),
        industry_leader_top_n=selection_cfg.get("industry_leader_top_n"),
        hard_filter_data=hard_filter_data,
        score_smoothing_days=int(selection_cfg.get("score_smoothing_days", 1)),
        entry_rank=selection_cfg.get("entry_rank"),
        exit_rank=selection_cfg.get("exit_rank"),
        entry_persist_days=int(selection_cfg.get("entry_persist_days", 1)),
        exit_persist_days=int(selection_cfg.get("exit_persist_days", 1)),
        min_hold_days=int(selection_cfg.get("min_hold_days", 0)),
        close_series=close_series,
        stoploss_lookback_days=int(selection_cfg.get("stoploss_lookback_days", 20)),
        stoploss_drawdown=float(selection_cfg.get("stoploss_drawdown", 0.10)),
        replacement_pool_size=int(selection_cfg.get("replacement_pool_size", 0)),
    )


def save_selection_frame(selection_df: pd.DataFrame, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    selection_df.to_csv(output, index=False)
    return output


def train_from_config(cfg: dict) -> dict:
    training = cfg["training"]
    label_cfg = cfg["label"]
    selection_cfg = cfg["selection"]
    feature_cfg = cfg.get("data", {})

    train_start = str(training["train_start"])
    valid_end = str(training["valid_end"])
    feature_frame, _, feature_columns = load_feature_frame(
        start_date=train_start,
        end_date=valid_end,
        rebalance_freq=str(selection_cfg["freq"]),
        feature_columns=feature_cfg.get("feature_columns"),
        data_cfg=feature_cfg,
        selection_cfg=selection_cfg,
    )
    close_series = load_close_series(
        instruments=sorted(feature_frame.index.get_level_values("instrument").unique().tolist()),
        start_date=train_start,
        end_date=valid_end,
        horizon_days=int(label_cfg["horizon_days"]),
    )
    labeled_frame = assemble_labeled_frame(feature_frame, close_series, int(label_cfg["horizon_days"]))

    train_frame = slice_frame_by_date(labeled_frame, train_start, str(training["train_end"]))
    valid_frame = slice_frame_by_date(
        labeled_frame,
        str(training["valid_start"]),
        str(training["valid_end"]),
    )

    bundle, metrics = fit_regressor(
        train_frame=train_frame,
        valid_frame=valid_frame,
        feature_columns=feature_columns,
        preferred_backend=str(cfg["model"]["preferred_backend"]),
        params=dict(cfg["model"].get("params", {})),
    )
    bundle["label_horizon_days"] = int(label_cfg["horizon_days"])
    bundle["config_name"] = str(cfg["name"])
    bundle["selection_freq"] = str(selection_cfg["freq"])
    bundle["feature_columns"] = feature_columns

    bundle_output = save_model_bundle(bundle, model_bundle_path(cfg))
    summary = {
        "config_name": cfg["name"],
        "bundle_path": str(bundle_output),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "training": training,
        "label": label_cfg,
        "model": {"backend": bundle["backend"], "params": cfg["model"].get("params", {})},
        "metrics": metrics,
    }
    save_json(summary, training_summary_path(cfg))
    return summary


def score_from_config(cfg: dict) -> dict:
    if bool(cfg.get("walk_forward", {}).get("enabled", False)):
        return score_walk_forward_from_config(cfg)

    bundle = load_model_bundle(model_bundle_path(cfg))
    scoring = cfg["scoring"]
    selection_cfg = cfg["selection"]
    feature_frame, rebalance_dates, _ = load_feature_frame(
        start_date=str(scoring["start_date"]),
        end_date=str(scoring["end_date"]),
        rebalance_freq=str(selection_cfg["freq"]),
        feature_columns=bundle["feature_columns"],
        data_cfg=cfg.get("data", {}),
        selection_cfg=selection_cfg,
    )
    scores = predict_signal_frame(bundle["model"], feature_frame, bundle["feature_columns"])
    score_output = save_scores(scores, scores_path(cfg))

    selection_df = materialize_selections_from_scores(scores, rebalance_dates, selection_cfg)
    selection_output = save_selection_frame(selection_df, selection_path(cfg))
    summary = {
        "config_name": cfg["name"],
        "score_path": str(score_output),
        "selection_path": str(selection_output),
        "score_rows": int(len(scores)),
        "score_dates": int(scores.index.get_level_values("datetime").nunique()) if not scores.empty else 0,
        "selection_rows": int(len(selection_df)),
        "selection_dates": int(selection_df["date"].nunique()) if not selection_df.empty else 0,
    }
    save_json(summary, scoring_summary_path(cfg))
    return summary


def score_walk_forward_from_config(cfg: dict) -> dict:
    training = cfg["training"]
    scoring = cfg["scoring"]
    selection_cfg = cfg["selection"]
    label_cfg = cfg["label"]
    feature_cfg = cfg.get("data", {})
    walk_cfg = cfg.get("walk_forward", {})

    full_start = str(training["train_start"])
    full_end = str(scoring["end_date"])
    feature_frame, _, feature_columns = load_feature_frame(
        start_date=full_start,
        end_date=full_end,
        rebalance_freq=str(selection_cfg["freq"]),
        feature_columns=feature_cfg.get("feature_columns"),
        data_cfg=feature_cfg,
        selection_cfg=selection_cfg,
    )
    close_series = load_close_series(
        instruments=sorted(feature_frame.index.get_level_values("instrument").unique().tolist()),
        start_date=full_start,
        end_date=full_end,
        horizon_days=int(label_cfg["horizon_days"]),
    )
    labeled_frame = assemble_labeled_frame(feature_frame, close_series, int(label_cfg["horizon_days"]))

    windows = build_walk_forward_windows(
        training_start=full_start,
        score_start=str(scoring["start_date"]),
        score_end=full_end,
        min_train_years=int(walk_cfg.get("min_train_years", 3)),
        retrain_months=int(walk_cfg.get("retrain_months", 12)),
        train_years=walk_cfg.get("train_years"),
        purge_days=int(walk_cfg.get("purge_days", 0)),
        embargo_days=int(walk_cfg.get("embargo_days", 0)),
    )

    score_parts: list[pd.Series] = []
    window_rows: list[dict] = []
    purged_kfold_cfg = walk_cfg.get("purged_kfold")
    for window in windows:
        train_frame = slice_frame_by_date(labeled_frame, window.train_start, window.train_end)
        score_frame = slice_frame_by_date(labeled_frame, window.score_start, window.score_end)
        if train_frame.empty or score_frame.empty:
            continue
        bundle, train_metrics = fit_regressor(
            train_frame=train_frame,
            valid_frame=score_frame.iloc[0:0].copy(),
            feature_columns=feature_columns,
            preferred_backend=str(cfg["model"]["preferred_backend"]),
            params=dict(cfg["model"].get("params", {})),
        )
        window_scores = predict_signal_frame(bundle["model"], score_frame, feature_columns)
        score_metrics = evaluate_cross_sectional_predictions(window_scores, score_frame["label"])

        # Optional: purged K-fold CV within training window
        kfold_ic = None
        if purged_kfold_cfg and bool(purged_kfold_cfg.get("enabled", False)):
            kfold_results = purged_kfold_cv(
                train_frame=train_frame,
                feature_columns=feature_columns,
                preferred_backend=str(cfg["model"]["preferred_backend"]),
                params=dict(cfg["model"].get("params", {})),
                n_folds=int(purged_kfold_cfg.get("n_folds", 5)),
                purge_days=int(purged_kfold_cfg.get("purge_days", walk_cfg.get("purge_days", 0))),
                embargo_days=int(purged_kfold_cfg.get("embargo_days", walk_cfg.get("embargo_days", 0))),
                horizon_days=int(label_cfg["horizon_days"]),
            )
            if kfold_results:
                kfold_ic = float(np.mean([f["rank_ic"] for f in kfold_results]))
                logging.info(
                    "Window %s~%s purged_kfold_cv mean_rank_ic=%.4f (%d folds)",
                    window.train_start, window.train_end, kfold_ic, len(kfold_results),
                )

        window_rows.append(
            {
                "train_start": window.train_start,
                "train_end": window.train_end,
                "score_start": window.score_start,
                "score_end": window.score_end,
                "train_rows": int(train_metrics.get("train_rows", 0)),
                "score_rows": int(len(window_scores)),
                "score_dates": int(window_scores.index.get_level_values("datetime").nunique()) if not window_scores.empty else 0,
                "mean_rank_ic": float(score_metrics.get("mean_rank_ic", 0.0)),
                "mean_pearson_ic": float(score_metrics.get("mean_pearson_ic", 0.0)),
                "purged_kfold_mean_rank_ic": kfold_ic,
            }
        )
        score_parts.append(window_scores)

    if score_parts:
        scores = pd.concat(score_parts).sort_index()
        scores = scores[~scores.index.duplicated(keep="last")]
    else:
        scores = pd.Series(dtype=float, index=_empty_index(), name="score")

    score_output = save_scores(scores, scores_path(cfg))
    rebalance_dates = (
        pd.DatetimeIndex(sorted(scores.index.get_level_values("datetime").unique()))
        if not scores.empty
        else pd.DatetimeIndex([])
    )
    selection_df = materialize_selections_from_scores(scores, rebalance_dates, selection_cfg)
    selection_output = save_selection_frame(selection_df, selection_path(cfg))
    window_df = pd.DataFrame(window_rows)
    summary = {
        "config_name": cfg["name"],
        "mode": "walk_forward",
        "score_path": str(score_output),
        "selection_path": str(selection_output),
        "score_rows": int(len(scores)),
        "score_dates": int(scores.index.get_level_values("datetime").nunique()) if not scores.empty else 0,
        "selection_rows": int(len(selection_df)),
        "selection_dates": int(selection_df["date"].nunique()) if not selection_df.empty else 0,
        "window_count": int(len(window_rows)),
        "mean_window_rank_ic": float(window_df["mean_rank_ic"].mean()) if not window_df.empty else 0.0,
        "windows": window_rows,
    }
    save_json(summary, scoring_summary_path(cfg))
    return summary


@dataclass
class _FixedPctController:
    stock_pct: float = 1.0

    def load_market_data(self):
        pass

    def get_allocation(self, date, is_rebalance_day: bool = False):
        from core.position import AllocationResult

        return AllocationResult(
            stock_pct=float(self.stock_pct),
            cash_pct=round(1 - float(self.stock_pct), 4),
            regime="fixed",
            opportunity_level="none",
            market_drawdown=0.0,
            trend_score=0.0,
        )

    def get_bond_daily_return(self) -> float:
        return 0.03 / 252


@dataclass(frozen=True)
class WalkForwardWindow:
    train_start: str
    train_end: str
    score_start: str
    score_end: str


def build_walk_forward_windows(
    training_start: str,
    score_start: str,
    score_end: str,
    min_train_years: int = 3,
    retrain_months: int = 12,
    train_years: Optional[int] = None,
    purge_days: int = 0,
    embargo_days: int = 0,
) -> list[WalkForwardWindow]:
    anchor_start = pd.Timestamp(training_start).normalize()
    current_score_start = pd.Timestamp(score_start).normalize()
    terminal_score_end = pd.Timestamp(score_end).normalize()
    min_score_start = anchor_start + pd.DateOffset(years=int(min_train_years))
    if current_score_start < min_score_start:
        current_score_start = min_score_start

    step = pd.DateOffset(months=int(retrain_months))
    windows: list[WalkForwardWindow] = []
    while current_score_start <= terminal_score_end:
        # Apply embargo: shift scoring start later
        effective_score_start = current_score_start + pd.Timedelta(days=int(embargo_days))
        if effective_score_start > terminal_score_end:
            break

        # Apply purge: shift training end earlier
        train_end = effective_score_start - pd.Timedelta(days=int(purge_days) + 1)
        if train_years is None:
            train_start_ts = anchor_start
        else:
            train_start_ts = max(
                anchor_start,
                effective_score_start - pd.DateOffset(years=int(train_years)),
            )
        score_end_ts = min(current_score_start + step - pd.Timedelta(days=1), terminal_score_end)
        if train_end < train_start_ts:
            current_score_start = current_score_start + step
            continue
        windows.append(
            WalkForwardWindow(
                train_start=train_start_ts.strftime("%Y-%m-%d"),
                train_end=train_end.strftime("%Y-%m-%d"),
                score_start=effective_score_start.strftime("%Y-%m-%d"),
                score_end=score_end_ts.strftime("%Y-%m-%d"),
            )
        )
        current_score_start = current_score_start + step
    return windows


def purged_kfold_cv(
    train_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    preferred_backend: str,
    params: Optional[Dict[str, Any]] = None,
    n_folds: int = 5,
    purge_days: int = 0,
    embargo_days: int = 0,
    horizon_days: int = 20,
) -> list[dict]:
    """Purged K-Fold CV on temporal data within a training window.

    Splits training data into K temporal folds with purge gaps between
    train and validation to prevent label leakage from horizon overlap.
    """
    dates = sorted(train_frame.index.get_level_values("datetime").unique())
    if len(dates) < n_folds + 1:
        return []

    fold_size = len(dates) // n_folds
    fold_results = []
    for k in range(n_folds):
        val_start_idx = k * fold_size
        val_end_idx = min((k + 1) * fold_size - 1, len(dates) - 1)
        val_start = dates[val_start_idx]
        val_end = dates[val_end_idx]

        # Apply embargo to validation start
        effective_val_start = val_start + pd.Timedelta(days=int(embargo_days))
        if effective_val_start > val_end:
            continue

        # Train on everything before validation, minus purge gap
        train_cutoff = effective_val_start - pd.Timedelta(days=int(purge_days) + 1)

        # Also add data after validation (if not last fold)
        train_dates_before = [d for d in dates if d <= train_cutoff]
        train_dates_after = [d for d in dates if d > val_end] if k < n_folds - 1 else []
        all_train_dates = set(train_dates_before + train_dates_after)

        if len(all_train_dates) < 2:
            continue

        train_mask = train_frame.index.get_level_values("datetime").isin(all_train_dates)
        val_mask = (train_frame.index.get_level_values("datetime") >= effective_val_start) & \
                   (train_frame.index.get_level_values("datetime") <= val_end)

        fold_train = train_frame.loc[train_mask]
        fold_val = train_frame.loc[val_mask]

        if fold_train.empty or fold_val.dropna(subset=["label"]).empty:
            continue

        bundle, _ = fit_regressor(
            train_frame=fold_train,
            valid_frame=fold_val,
            feature_columns=feature_columns,
            preferred_backend=preferred_backend,
            params=params,
        )
        val_scores = predict_signal_frame(bundle["model"], fold_val, feature_columns)
        val_metrics = evaluate_cross_sectional_predictions(val_scores, fold_val["label"])
        fold_results.append({
            "fold": k,
            "train_dates": len(all_train_dates),
            "val_dates": int(fold_val.index.get_level_values("datetime").nunique()),
            "rank_ic": float(val_metrics.get("mean_rank_ic", 0.0)),
            "pearson_ic": float(val_metrics.get("mean_pearson_ic", 0.0)),
        })
    return fold_results


@dataclass
class ModelSignalStrategy:
    name: str
    selection_csv: Path
    topk: int
    universe: str = "all"
    position_model: str = "fixed"
    position_params: Dict[str, Any] = field(default_factory=dict)
    trading_cost: Dict[str, Any] = field(default_factory=dict)

    def selections_path(self) -> Path:
        return Path(self.selection_csv)

    def artifact_slug(self) -> str:
        return self.name.replace("/", "__")

    def load_selections(self):
        from core.selection import load_selections

        return load_selections(csv_path=self.selection_csv)

    def build_position_controller(self):
        if self.position_model == "trend":
            from core.position import MarketConfig, MarketPositionController

            params = {
                key: value
                for key, value in self.position_params.items()
                if key in MarketConfig.__dataclass_fields__
            }
            return MarketPositionController(config=MarketConfig(**params) if params else None)
        if self.position_model == "gate":
            from core.position import MarketGateConfig, MarketGatePositionController

            params = {
                key: value
                for key, value in self.position_params.items()
                if key in MarketGateConfig.__dataclass_fields__
            }
            return MarketGatePositionController(
                config=MarketGateConfig(**params) if params else None
            )
        if self.position_model == "fixed":
            return _FixedPctController(stock_pct=float(self.position_params.get("stock_pct", 1.0)))
        if self.position_model == "vol_norm":
            return _FixedPctController(stock_pct=float(self.position_params.get("stock_pct", 1.0)))
        if self.position_model == "full":
            return None
        raise ValueError(f"未知 position.model: {self.position_model}")


def backtest_from_config(cfg: dict, engine: str = "qlib"):
    if engine == "qlib":
        from modules.backtest.qlib_engine import QlibBacktestEngine

        backtest_engine = QlibBacktestEngine()
    elif engine == "pybroker":
        from modules.backtest.pybroker_engine import PyBrokerBacktestEngine

        backtest_engine = PyBrokerBacktestEngine()
    else:
        raise ValueError(f"未知 backtest engine: {engine}")

    strategy = ModelSignalStrategy(
        name=str(cfg["name"]),
        selection_csv=selection_path(cfg),
        topk=int(cfg["selection"]["topk"]),
        universe=str(cfg["selection"].get("universe", "all")),
        position_model=str(cfg["position"]["model"]),
        position_params=dict(cfg["position"].get("params", {})),
        trading_cost=dict(cfg.get("trading", {})),
    )
    result = backtest_engine.run(strategy=strategy)
    overlay_applied = False
    overlay_frame = None
    if bool(cfg.get("overlay", {}).get("enabled", False)):
        result, overlay_frame = apply_overlay_to_backtest_result(result, cfg)
        overlay_applied = overlay_frame is not None
        if overlay_frame is not None:
            overlay_csv = overlay_results_path(cfg)
            overlay_frame.to_csv(overlay_csv)
            result.metadata["results_file"] = str(overlay_csv)
    summary = {
        "config_name": cfg["name"],
        "annual_return": float(result.annual_return),
        "max_drawdown": float(result.max_drawdown),
        "sharpe_ratio": float(result.sharpe_ratio),
        "results_file": result.metadata.get("results_file"),
        "strategy_name": result.metadata.get("strategy_name"),
        "overlay_applied": overlay_applied,
        "engine": engine,
    }
    save_json(summary, backtest_summary_path(cfg))
    return result, summary
