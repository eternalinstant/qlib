#!/usr/bin/env python3
"""批量搜索预测式模型信号配置。"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.modeling.predictive_signal import (
    backtest_from_config,
    load_predictive_config,
    output_root,
    save_json,
    score_from_config,
    train_from_config,
)


FEATURE_BUNDLES = {
    "quality_flow": [
        "roe_fina",
        "roe_dt_fina",
        "roa_fina",
        "current_ratio_fina",
        "quick_ratio_fina",
        "debt_to_assets_fina",
        "net_margin",
        "total_revenue_inc",
        "operate_profit_inc",
        "n_cashflow_act",
        "smart_ratio_5d",
        "net_mf_amount_5d",
        "net_mf_amount_20d",
        "net_mf_vol_5d",
    ],
    "value_flow": [
        "book_to_market",
        "roic_proxy",
        "ebit_to_mv",
        "ebitda_to_mv",
        "ocf_to_mv",
        "ocf_to_ev",
        "fcff_to_mv",
        "smart_ratio_5d",
        "net_mf_amount_5d",
        "net_mf_amount_20d",
        "net_mf_vol_5d",
    ],
    "quality_value_flow": [
        "book_to_market",
        "roic_proxy",
        "ebit_to_mv",
        "ebitda_to_mv",
        "ocf_to_mv",
        "ocf_to_ev",
        "fcff_to_mv",
        "roe_fina",
        "roe_dt_fina",
        "roa_fina",
        "current_ratio_fina",
        "quick_ratio_fina",
        "debt_to_assets_fina",
        "net_margin",
        "total_revenue_inc",
        "operate_profit_inc",
        "n_cashflow_act",
        "smart_ratio_5d",
        "net_mf_amount_5d",
        "net_mf_amount_20d",
        "net_mf_vol_5d",
    ],
    "quality_value_flow_plus": [
        "book_to_market",
        "roic_proxy",
        "ebit_to_mv",
        "ebitda_to_mv",
        "ocf_to_mv",
        "ocf_to_ev",
        "fcff_to_mv",
        "roe_fina",
        "roe_dt_fina",
        "roa_fina",
        "current_ratio_fina",
        "quick_ratio_fina",
        "debt_to_assets_fina",
        "net_margin",
        "total_revenue_inc",
        "operate_profit_inc",
        "n_cashflow_act",
        "smart_ratio_5d",
        "net_mf_amount_5d",
        "net_mf_amount_20d",
        "net_mf_vol_5d",
        "rank_book_to_market",
        "rank_roe_fina",
        "rank_ocf_to_ev",
        "rank_low_debt",
        "rank_smart_ratio_5d",
        "rank_net_mf_amount_20d",
        "rank_liquidity_buffer",
        "rank_balance_strength",
        "rank_earnings_accel",
        "quality_value_interaction",
        "cashflow_flow_interaction",
        "fundamental_momentum_interaction",
        "rank_value_delta_1",
        "rank_flow_delta_1",
        "rank_smart_delta_1",
        "qvf_rank_blend",
        "qvf_dynamic_blend",
    ],
    "quality_value_flow_ranked": [
        "rank_book_to_market",
        "rank_roe_fina",
        "rank_ocf_to_ev",
        "rank_low_debt",
        "rank_smart_ratio_5d",
        "rank_net_mf_amount_20d",
        "rank_liquidity_buffer",
        "rank_balance_strength",
        "rank_earnings_accel",
        "quality_value_interaction",
        "cashflow_flow_interaction",
        "fundamental_momentum_interaction",
        "rank_value_delta_1",
        "rank_flow_delta_1",
        "rank_smart_delta_1",
        "qvf_rank_blend",
        "qvf_dynamic_blend",
    ],
    "quality_value_flow_core_plus": [
        "book_to_market",
        "roic_proxy",
        "ebit_to_mv",
        "ebitda_to_mv",
        "ocf_to_mv",
        "ocf_to_ev",
        "fcff_to_mv",
        "roe_fina",
        "roe_dt_fina",
        "roa_fina",
        "current_ratio_fina",
        "quick_ratio_fina",
        "debt_to_assets_fina",
        "net_margin",
        "total_revenue_inc",
        "operate_profit_inc",
        "n_cashflow_act",
        "smart_ratio_5d",
        "net_mf_amount_5d",
        "net_mf_amount_20d",
        "net_mf_vol_5d",
        "rank_value_profit_core",
        "rank_flow_momentum_core",
        "rank_growth_quality_core",
        "rank_balance_core",
        "qvf_core_alpha",
        "qvf_core_interaction",
        "qvf_core_dynamic",
    ],
    "quality_value_flow_compact": [
        "book_to_market",
        "ebit_to_mv",
        "ebitda_to_mv",
        "ocf_to_ev",
        "fcff_to_mv",
        "roa_fina",
        "current_ratio_fina",
        "net_margin",
        "total_revenue_inc",
        "operate_profit_inc",
        "smart_ratio_5d",
        "net_mf_amount_5d",
        "net_mf_amount_20d",
        "rank_value_profit_core",
        "rank_flow_momentum_core",
        "rank_growth_quality_core",
        "rank_balance_core",
        "qvf_core_alpha",
        "qvf_core_interaction",
        "qvf_core_dynamic",
    ],
}


POSITION_PRESETS = {
    "gate_balanced": {
        "model": "gate",
        "params": {
            "ma_window": 60,
            "strong_stock_pct": 0.95,
            "mixed_stock_pct": 0.65,
            "weak_stock_pct": 0.20,
        },
    },
    "gate_active": {
        "model": "gate",
        "params": {
            "ma_window": 40,
            "strong_stock_pct": 1.00,
            "mixed_stock_pct": 0.80,
            "weak_stock_pct": 0.35,
        },
    },
    "fixed_100": {
        "model": "fixed",
        "params": {"stock_pct": 1.0},
    },
    "fixed_85": {
        "model": "fixed",
        "params": {"stock_pct": 0.85},
    },
    "fixed_80": {
        "model": "fixed",
        "params": {"stock_pct": 0.80},
    },
    "fixed_75": {
        "model": "fixed",
        "params": {"stock_pct": 0.75},
    },
    "fixed_70": {
        "model": "fixed",
        "params": {"stock_pct": 0.70},
    },
    "fixed_65": {
        "model": "fixed",
        "params": {"stock_pct": 0.65},
    },
    "fixed_60": {
        "model": "fixed",
        "params": {"stock_pct": 0.60},
    },
    "fixed_58": {
        "model": "fixed",
        "params": {"stock_pct": 0.58},
    },
    "fixed_62": {
        "model": "fixed",
        "params": {"stock_pct": 0.62},
    },
    "fixed_64": {
        "model": "fixed",
        "params": {"stock_pct": 0.64},
    },
    "fixed_90": {
        "model": "fixed",
        "params": {"stock_pct": 0.90},
    },
    "fixed_95": {
        "model": "fixed",
        "params": {"stock_pct": 0.95},
    },
    "fixed_88": {
        "model": "fixed",
        "params": {"stock_pct": 0.88},
    },
    "fixed_86": {
        "model": "fixed",
        "params": {"stock_pct": 0.86},
    },
    "fixed_84": {
        "model": "fixed",
        "params": {"stock_pct": 0.84},
    },
    "gate_defensive": {
        "model": "gate",
        "params": {
            "ma_window": 60,
            "strong_stock_pct": 0.90,
            "mixed_stock_pct": 0.50,
            "weak_stock_pct": 0.10,
        },
    },
    "gate_hot": {
        "model": "gate",
        "params": {
            "ma_window": 40,
            "strong_stock_pct": 1.00,
            "mixed_stock_pct": 0.90,
            "weak_stock_pct": 0.45,
        },
    },
    "gate_hotter": {
        "model": "gate",
        "params": {
            "ma_window": 40,
            "strong_stock_pct": 1.00,
            "mixed_stock_pct": 0.93,
            "weak_stock_pct": 0.50,
        },
    },
    "gate_hottest": {
        "model": "gate",
        "params": {
            "ma_window": 40,
            "strong_stock_pct": 1.00,
            "mixed_stock_pct": 0.96,
            "weak_stock_pct": 0.55,
        },
    },
}


BATCH_SCAN_V1 = [
    {"name": "qvf_biweek_k8_gate_balanced_h10", "bundle": "quality_value_flow", "freq": "biweek", "topk": 8, "position": "gate_balanced", "horizon_days": 10, "min_market_cap": 80},
    {"name": "qvf_week_k8_gate_balanced_h10", "bundle": "quality_value_flow", "freq": "week", "topk": 8, "position": "gate_balanced", "horizon_days": 10, "min_market_cap": 80},
    {"name": "qvf_biweek_k5_gate_active_h10", "bundle": "quality_value_flow", "freq": "biweek", "topk": 5, "position": "gate_active", "horizon_days": 10, "min_market_cap": 80},
    {"name": "vf_biweek_k8_gate_balanced_h10", "bundle": "value_flow", "freq": "biweek", "topk": 8, "position": "gate_balanced", "horizon_days": 10, "min_market_cap": 80},
    {"name": "vf_week_k8_gate_balanced_h10", "bundle": "value_flow", "freq": "week", "topk": 8, "position": "gate_balanced", "horizon_days": 10, "min_market_cap": 80},
    {"name": "qf_biweek_k8_gate_balanced_h10", "bundle": "quality_flow", "freq": "biweek", "topk": 8, "position": "gate_balanced", "horizon_days": 10, "min_market_cap": 80},
    {"name": "vf_biweek_k8_fixed85_h10", "bundle": "value_flow", "freq": "biweek", "topk": 8, "position": "fixed_85", "horizon_days": 10, "min_market_cap": 80},
    {"name": "vf_biweek_k5_fixed100_h10", "bundle": "value_flow", "freq": "biweek", "topk": 5, "position": "fixed_100", "horizon_days": 10, "min_market_cap": 80},
    {"name": "qvf_biweek_k8_gate_balanced_h5", "bundle": "quality_value_flow", "freq": "biweek", "topk": 8, "position": "gate_balanced", "horizon_days": 5, "min_market_cap": 80},
    {"name": "qvf_biweek_k8_gate_balanced_h20", "bundle": "quality_value_flow", "freq": "biweek", "topk": 8, "position": "gate_balanced", "horizon_days": 20, "min_market_cap": 80},
    {"name": "vf_biweek_k8_gate_active_h5", "bundle": "value_flow", "freq": "biweek", "topk": 8, "position": "gate_active", "horizon_days": 5, "min_market_cap": 50},
    {"name": "qf_week_k8_fixed85_h10", "bundle": "quality_flow", "freq": "week", "topk": 8, "position": "fixed_85", "horizon_days": 10, "min_market_cap": 50},
]

FOCUSED_H20 = [
    {"name": "qvf_biweek_k8_gate_balanced_h20", "bundle": "quality_value_flow", "freq": "biweek", "topk": 8, "position": "gate_balanced", "horizon_days": 20, "min_market_cap": 80},
    {"name": "qvf_biweek_k5_gate_balanced_h20", "bundle": "quality_value_flow", "freq": "biweek", "topk": 5, "position": "gate_balanced", "horizon_days": 20, "min_market_cap": 80},
    {"name": "qvf_biweek_k10_gate_balanced_h20", "bundle": "quality_value_flow", "freq": "biweek", "topk": 10, "position": "gate_balanced", "horizon_days": 20, "min_market_cap": 80},
    {"name": "qvf_biweek_k8_gate_active_h20", "bundle": "quality_value_flow", "freq": "biweek", "topk": 8, "position": "gate_active", "horizon_days": 20, "min_market_cap": 80},
    {"name": "qvf_biweek_k8_fixed85_h20", "bundle": "quality_value_flow", "freq": "biweek", "topk": 8, "position": "fixed_85", "horizon_days": 20, "min_market_cap": 80},
    {"name": "qvf_biweek_k5_fixed100_h20", "bundle": "quality_value_flow", "freq": "biweek", "topk": 5, "position": "fixed_100", "horizon_days": 20, "min_market_cap": 80},
    {"name": "qvf_biweek_k8_gate_balanced_h30", "bundle": "quality_value_flow", "freq": "biweek", "topk": 8, "position": "gate_balanced", "horizon_days": 30, "min_market_cap": 80},
    {"name": "qvf_biweek_k8_gate_balanced_h40", "bundle": "quality_value_flow", "freq": "biweek", "topk": 8, "position": "gate_balanced", "horizon_days": 40, "min_market_cap": 80},
    {"name": "vf_biweek_k8_gate_balanced_h20", "bundle": "value_flow", "freq": "biweek", "topk": 8, "position": "gate_balanced", "horizon_days": 20, "min_market_cap": 80},
    {"name": "qf_biweek_k8_gate_balanced_h20", "bundle": "quality_flow", "freq": "biweek", "topk": 8, "position": "gate_balanced", "horizon_days": 20, "min_market_cap": 80},
]

FOCUSED_H20_STOPLOSS = [
    {
        "name": "qvf_biweek_k5_fixed100_h20_sl08_lb15_rp5",
        "bundle": "quality_value_flow",
        "freq": "biweek",
        "topk": 5,
        "position": "fixed_100",
        "horizon_days": 20,
        "min_market_cap": 80,
        "selection_overrides": {
            "mode": "stoploss_replace",
            "stoploss_lookback_days": 15,
            "stoploss_drawdown": 0.08,
            "replacement_pool_size": 5,
        },
    },
    {
        "name": "qvf_biweek_k5_fixed100_h20_sl10_lb15_rp5",
        "bundle": "quality_value_flow",
        "freq": "biweek",
        "topk": 5,
        "position": "fixed_100",
        "horizon_days": 20,
        "min_market_cap": 80,
        "selection_overrides": {
            "mode": "stoploss_replace",
            "stoploss_lookback_days": 15,
            "stoploss_drawdown": 0.10,
            "replacement_pool_size": 5,
        },
    },
    {
        "name": "qvf_biweek_k5_fixed100_h20_sl12_lb15_rp5",
        "bundle": "quality_value_flow",
        "freq": "biweek",
        "topk": 5,
        "position": "fixed_100",
        "horizon_days": 20,
        "min_market_cap": 80,
        "selection_overrides": {
            "mode": "stoploss_replace",
            "stoploss_lookback_days": 15,
            "stoploss_drawdown": 0.12,
            "replacement_pool_size": 5,
        },
    },
    {
        "name": "qvf_biweek_k5_fixed100_h20_sl10_lb20_rp5",
        "bundle": "quality_value_flow",
        "freq": "biweek",
        "topk": 5,
        "position": "fixed_100",
        "horizon_days": 20,
        "min_market_cap": 80,
        "selection_overrides": {
            "mode": "stoploss_replace",
            "stoploss_lookback_days": 20,
            "stoploss_drawdown": 0.10,
            "replacement_pool_size": 5,
        },
    },
    {
        "name": "qvf_biweek_k5_fixed100_h20_sl10_lb30_rp5",
        "bundle": "quality_value_flow",
        "freq": "biweek",
        "topk": 5,
        "position": "fixed_100",
        "horizon_days": 20,
        "min_market_cap": 80,
        "selection_overrides": {
            "mode": "stoploss_replace",
            "stoploss_lookback_days": 30,
            "stoploss_drawdown": 0.10,
            "replacement_pool_size": 5,
        },
    },
    {
        "name": "qvf_biweek_k5_fixed100_h20_sl08_lb20_rp10",
        "bundle": "quality_value_flow",
        "freq": "biweek",
        "topk": 5,
        "position": "fixed_100",
        "horizon_days": 20,
        "min_market_cap": 80,
        "selection_overrides": {
            "mode": "stoploss_replace",
            "stoploss_lookback_days": 20,
            "stoploss_drawdown": 0.08,
            "replacement_pool_size": 10,
        },
    },
    {
        "name": "qvf_biweek_k8_fixed85_h20_sl08_lb15_rp8",
        "bundle": "quality_value_flow",
        "freq": "biweek",
        "topk": 8,
        "position": "fixed_85",
        "horizon_days": 20,
        "min_market_cap": 80,
        "selection_overrides": {
            "mode": "stoploss_replace",
            "stoploss_lookback_days": 15,
            "stoploss_drawdown": 0.08,
            "replacement_pool_size": 8,
        },
    },
    {
        "name": "qvf_biweek_k8_fixed85_h20_sl10_lb20_rp8",
        "bundle": "quality_value_flow",
        "freq": "biweek",
        "topk": 8,
        "position": "fixed_85",
        "horizon_days": 20,
        "min_market_cap": 80,
        "selection_overrides": {
            "mode": "stoploss_replace",
            "stoploss_lookback_days": 20,
            "stoploss_drawdown": 0.10,
            "replacement_pool_size": 8,
        },
    },
    {
        "name": "qvf_biweek_k10_gate_balanced_h20_sl08_lb15_rp12",
        "bundle": "quality_value_flow",
        "freq": "biweek",
        "topk": 10,
        "position": "gate_balanced",
        "horizon_days": 20,
        "min_market_cap": 80,
        "selection_overrides": {
            "mode": "stoploss_replace",
            "stoploss_lookback_days": 15,
            "stoploss_drawdown": 0.08,
            "replacement_pool_size": 12,
        },
    },
    {
        "name": "qvf_biweek_k10_gate_balanced_h20_sl10_lb20_rp12",
        "bundle": "quality_value_flow",
        "freq": "biweek",
        "topk": 10,
        "position": "gate_balanced",
        "horizon_days": 20,
        "min_market_cap": 80,
        "selection_overrides": {
            "mode": "stoploss_replace",
            "stoploss_lookback_days": 20,
            "stoploss_drawdown": 0.10,
            "replacement_pool_size": 12,
        },
    },
]

FOCUSED_QVF_RISK_RETURN = [
    {"name": "qvf_biweek_k3_fixed100_h20_mv120", "bundle": "quality_value_flow", "freq": "biweek", "topk": 3, "position": "fixed_100", "horizon_days": 20, "min_market_cap": 120},
    {"name": "qvf_biweek_k3_fixed95_h20_mv120", "bundle": "quality_value_flow", "freq": "biweek", "topk": 3, "position": "fixed_95", "horizon_days": 20, "min_market_cap": 120},
    {"name": "qvf_biweek_k5_fixed95_h20_mv120", "bundle": "quality_value_flow", "freq": "biweek", "topk": 5, "position": "fixed_95", "horizon_days": 20, "min_market_cap": 120},
    {"name": "qvf_biweek_k5_fixed90_h20_mv80", "bundle": "quality_value_flow", "freq": "biweek", "topk": 5, "position": "fixed_90", "horizon_days": 20, "min_market_cap": 80},
    {"name": "qvf_biweek_k5_gate_active_h20_mv80", "bundle": "quality_value_flow", "freq": "biweek", "topk": 5, "position": "gate_active", "horizon_days": 20, "min_market_cap": 80},
    {"name": "qvf_biweek_k5_gate_active_h20_mv120", "bundle": "quality_value_flow", "freq": "biweek", "topk": 5, "position": "gate_active", "horizon_days": 20, "min_market_cap": 120},
    {"name": "qvf_biweek_k5_gate_balanced_h20_mv120", "bundle": "quality_value_flow", "freq": "biweek", "topk": 5, "position": "gate_balanced", "horizon_days": 20, "min_market_cap": 120},
    {"name": "qvf_biweek_k5_gate_defensive_h20_mv80", "bundle": "quality_value_flow", "freq": "biweek", "topk": 5, "position": "gate_defensive", "horizon_days": 20, "min_market_cap": 80},
    {"name": "qvf_biweek_k6_gate_active_h20_mv80", "bundle": "quality_value_flow", "freq": "biweek", "topk": 6, "position": "gate_active", "horizon_days": 20, "min_market_cap": 80},
    {"name": "qvf_biweek_k8_gate_active_h20_mv50", "bundle": "quality_value_flow", "freq": "biweek", "topk": 8, "position": "gate_active", "horizon_days": 20, "min_market_cap": 50},
    {"name": "qvf_biweek_k3_gate_active_h20_mv80", "bundle": "quality_value_flow", "freq": "biweek", "topk": 3, "position": "gate_active", "horizon_days": 20, "min_market_cap": 80},
    {"name": "qvf_biweek_k3_gate_defensive_h20_mv80", "bundle": "quality_value_flow", "freq": "biweek", "topk": 3, "position": "gate_defensive", "horizon_days": 20, "min_market_cap": 80},
    {"name": "qvf_month_k5_fixed100_h20_mv80", "bundle": "quality_value_flow", "freq": "month", "topk": 5, "position": "fixed_100", "horizon_days": 20, "min_market_cap": 80},
    {"name": "qvf_month_k5_gate_active_h20_mv80", "bundle": "quality_value_flow", "freq": "month", "topk": 5, "position": "gate_active", "horizon_days": 20, "min_market_cap": 80},
]

EDGE_QVF_FRONTIER = [
    {"name": "qvf_biweek_k3_gate_active_h20_mv120", "bundle": "quality_value_flow", "freq": "biweek", "topk": 3, "position": "gate_active", "horizon_days": 20, "min_market_cap": 120},
    {"name": "qvf_biweek_k3_gate_hot_h20_mv120", "bundle": "quality_value_flow", "freq": "biweek", "topk": 3, "position": "gate_hot", "horizon_days": 20, "min_market_cap": 120},
    {"name": "qvf_biweek_k3_gate_hot_h20_mv80", "bundle": "quality_value_flow", "freq": "biweek", "topk": 3, "position": "gate_hot", "horizon_days": 20, "min_market_cap": 80},
    {"name": "qvf_biweek_k4_gate_hot_h20_mv120", "bundle": "quality_value_flow", "freq": "biweek", "topk": 4, "position": "gate_hot", "horizon_days": 20, "min_market_cap": 120},
    {"name": "qvf_biweek_k4_gate_active_h20_mv120", "bundle": "quality_value_flow", "freq": "biweek", "topk": 4, "position": "gate_active", "horizon_days": 20, "min_market_cap": 120},
    {"name": "qvf_biweek_k3_fixed90_h20_mv120", "bundle": "quality_value_flow", "freq": "biweek", "topk": 3, "position": "fixed_90", "horizon_days": 20, "min_market_cap": 120},
    {"name": "qvf_biweek_k3_fixed85_h20_mv120", "bundle": "quality_value_flow", "freq": "biweek", "topk": 3, "position": "fixed_85", "horizon_days": 20, "min_market_cap": 120},
    {"name": "qvf_biweek_k4_fixed95_h20_mv120", "bundle": "quality_value_flow", "freq": "biweek", "topk": 4, "position": "fixed_95", "horizon_days": 20, "min_market_cap": 120},
]

EDGE_QVF_FINAL = [
    {"name": "qvf_biweek_k3_gate_hot_h20_mv130", "bundle": "quality_value_flow", "freq": "biweek", "topk": 3, "position": "gate_hot", "horizon_days": 20, "min_market_cap": 130},
    {"name": "qvf_biweek_k3_gate_hot_h20_mv140", "bundle": "quality_value_flow", "freq": "biweek", "topk": 3, "position": "gate_hot", "horizon_days": 20, "min_market_cap": 140},
    {"name": "qvf_biweek_k3_gate_hot_h20_mv160", "bundle": "quality_value_flow", "freq": "biweek", "topk": 3, "position": "gate_hot", "horizon_days": 20, "min_market_cap": 160},
    {"name": "qvf_biweek_k3_gate_hotter_h20_mv120", "bundle": "quality_value_flow", "freq": "biweek", "topk": 3, "position": "gate_hotter", "horizon_days": 20, "min_market_cap": 120},
    {"name": "qvf_biweek_k3_gate_hotter_h20_mv140", "bundle": "quality_value_flow", "freq": "biweek", "topk": 3, "position": "gate_hotter", "horizon_days": 20, "min_market_cap": 140},
    {"name": "qvf_biweek_k3_gate_hotter_h20_mv160", "bundle": "quality_value_flow", "freq": "biweek", "topk": 3, "position": "gate_hotter", "horizon_days": 20, "min_market_cap": 160},
    {"name": "qvf_biweek_k3_gate_hottest_h20_mv140", "bundle": "quality_value_flow", "freq": "biweek", "topk": 3, "position": "gate_hottest", "horizon_days": 20, "min_market_cap": 140},
    {"name": "qvf_biweek_k3_fixed88_h20_mv120", "bundle": "quality_value_flow", "freq": "biweek", "topk": 3, "position": "fixed_88", "horizon_days": 20, "min_market_cap": 120},
    {"name": "qvf_biweek_k3_fixed88_h20_mv140", "bundle": "quality_value_flow", "freq": "biweek", "topk": 3, "position": "fixed_88", "horizon_days": 20, "min_market_cap": 140},
    {"name": "qvf_biweek_k3_fixed86_h20_mv120", "bundle": "quality_value_flow", "freq": "biweek", "topk": 3, "position": "fixed_86", "horizon_days": 20, "min_market_cap": 120},
    {"name": "qvf_biweek_k3_fixed86_h20_mv140", "bundle": "quality_value_flow", "freq": "biweek", "topk": 3, "position": "fixed_86", "horizon_days": 20, "min_market_cap": 140},
    {"name": "qvf_biweek_k3_fixed84_h20_mv120", "bundle": "quality_value_flow", "freq": "biweek", "topk": 3, "position": "fixed_84", "horizon_days": 20, "min_market_cap": 120},
    {"name": "qvf_biweek_k3_fixed84_h20_mv140", "bundle": "quality_value_flow", "freq": "biweek", "topk": 3, "position": "fixed_84", "horizon_days": 20, "min_market_cap": 140},
]

EDGE_QVF_HORIZON = [
    {"name": "qvf_biweek_k3_gate_hot_h16_mv120", "bundle": "quality_value_flow", "freq": "biweek", "topk": 3, "position": "gate_hot", "horizon_days": 16, "min_market_cap": 120},
    {"name": "qvf_biweek_k3_gate_hot_h18_mv120", "bundle": "quality_value_flow", "freq": "biweek", "topk": 3, "position": "gate_hot", "horizon_days": 18, "min_market_cap": 120},
    {"name": "qvf_biweek_k3_gate_hot_h22_mv120", "bundle": "quality_value_flow", "freq": "biweek", "topk": 3, "position": "gate_hot", "horizon_days": 22, "min_market_cap": 120},
    {"name": "qvf_biweek_k3_gate_hot_h24_mv120", "bundle": "quality_value_flow", "freq": "biweek", "topk": 3, "position": "gate_hot", "horizon_days": 24, "min_market_cap": 120},
    {"name": "qvf_biweek_k3_gate_hotter_h16_mv120", "bundle": "quality_value_flow", "freq": "biweek", "topk": 3, "position": "gate_hotter", "horizon_days": 16, "min_market_cap": 120},
    {"name": "qvf_biweek_k3_gate_hotter_h18_mv120", "bundle": "quality_value_flow", "freq": "biweek", "topk": 3, "position": "gate_hotter", "horizon_days": 18, "min_market_cap": 120},
    {"name": "qvf_biweek_k3_gate_hotter_h22_mv120", "bundle": "quality_value_flow", "freq": "biweek", "topk": 3, "position": "gate_hotter", "horizon_days": 22, "min_market_cap": 120},
    {"name": "qvf_biweek_k3_gate_hotter_h24_mv120", "bundle": "quality_value_flow", "freq": "biweek", "topk": 3, "position": "gate_hotter", "horizon_days": 24, "min_market_cap": 120},
    {"name": "qvf_biweek_k3_fixed84_h16_mv120", "bundle": "quality_value_flow", "freq": "biweek", "topk": 3, "position": "fixed_84", "horizon_days": 16, "min_market_cap": 120},
    {"name": "qvf_biweek_k3_fixed84_h18_mv120", "bundle": "quality_value_flow", "freq": "biweek", "topk": 3, "position": "fixed_84", "horizon_days": 18, "min_market_cap": 120},
    {"name": "qvf_biweek_k3_fixed84_h22_mv120", "bundle": "quality_value_flow", "freq": "biweek", "topk": 3, "position": "fixed_84", "horizon_days": 22, "min_market_cap": 120},
    {"name": "qvf_biweek_k3_fixed84_h24_mv120", "bundle": "quality_value_flow", "freq": "biweek", "topk": 3, "position": "fixed_84", "horizon_days": 24, "min_market_cap": 120},
]

NEW_ALPHA_QVF = [
    {"name": "qvfplus_biweek_k3_gate_hotter_h20_mv120", "bundle": "quality_value_flow_plus", "freq": "biweek", "topk": 3, "position": "gate_hotter", "horizon_days": 20, "min_market_cap": 120},
    {"name": "qvfplus_biweek_k3_gate_hot_h20_mv120", "bundle": "quality_value_flow_plus", "freq": "biweek", "topk": 3, "position": "gate_hot", "horizon_days": 20, "min_market_cap": 120},
    {"name": "qvfplus_biweek_k3_fixed84_h20_mv120", "bundle": "quality_value_flow_plus", "freq": "biweek", "topk": 3, "position": "fixed_84", "horizon_days": 20, "min_market_cap": 120},
    {"name": "qvfplus_biweek_k3_gate_active_h20_mv120", "bundle": "quality_value_flow_plus", "freq": "biweek", "topk": 3, "position": "gate_active", "horizon_days": 20, "min_market_cap": 120},
    {"name": "qvfranked_biweek_k3_gate_hotter_h20_mv120", "bundle": "quality_value_flow_ranked", "freq": "biweek", "topk": 3, "position": "gate_hotter", "horizon_days": 20, "min_market_cap": 120},
    {"name": "qvfranked_biweek_k3_gate_hot_h20_mv120", "bundle": "quality_value_flow_ranked", "freq": "biweek", "topk": 3, "position": "gate_hot", "horizon_days": 20, "min_market_cap": 120},
    {"name": "qvfranked_biweek_k3_fixed84_h20_mv120", "bundle": "quality_value_flow_ranked", "freq": "biweek", "topk": 3, "position": "fixed_84", "horizon_days": 20, "min_market_cap": 120},
    {"name": "qvfranked_biweek_k3_gate_active_h20_mv120", "bundle": "quality_value_flow_ranked", "freq": "biweek", "topk": 3, "position": "gate_active", "horizon_days": 20, "min_market_cap": 120},
]

NEW_ALPHA_QVF_CORE = [
    {"name": "qvfcoreplus_biweek_k3_gate_hotter_h20_mv120", "bundle": "quality_value_flow_core_plus", "freq": "biweek", "topk": 3, "position": "gate_hotter", "horizon_days": 20, "min_market_cap": 120},
    {"name": "qvfcoreplus_biweek_k3_gate_hottest_h20_mv140", "bundle": "quality_value_flow_core_plus", "freq": "biweek", "topk": 3, "position": "gate_hottest", "horizon_days": 20, "min_market_cap": 140},
    {"name": "qvfcoreplus_biweek_k3_fixed86_h20_mv120", "bundle": "quality_value_flow_core_plus", "freq": "biweek", "topk": 3, "position": "fixed_86", "horizon_days": 20, "min_market_cap": 120},
    {"name": "qvfcoreplus_biweek_k3_fixed88_h20_mv120", "bundle": "quality_value_flow_core_plus", "freq": "biweek", "topk": 3, "position": "fixed_88", "horizon_days": 20, "min_market_cap": 120},
    {"name": "qvfcompact_biweek_k3_gate_hotter_h20_mv120", "bundle": "quality_value_flow_compact", "freq": "biweek", "topk": 3, "position": "gate_hotter", "horizon_days": 20, "min_market_cap": 120},
    {"name": "qvfcompact_biweek_k3_gate_hottest_h20_mv140", "bundle": "quality_value_flow_compact", "freq": "biweek", "topk": 3, "position": "gate_hottest", "horizon_days": 20, "min_market_cap": 140},
    {"name": "qvfcompact_biweek_k3_fixed86_h20_mv120", "bundle": "quality_value_flow_compact", "freq": "biweek", "topk": 3, "position": "fixed_86", "horizon_days": 20, "min_market_cap": 120},
    {"name": "qvfcompact_biweek_k3_fixed88_h20_mv120", "bundle": "quality_value_flow_compact", "freq": "biweek", "topk": 3, "position": "fixed_88", "horizon_days": 20, "min_market_cap": 120},
]

NEW_ALPHA_QVF_CORE_RISK = [
    {"name": "qvfcoreplus_biweek_k3_fixed60_h20_mv120", "bundle": "quality_value_flow_core_plus", "freq": "biweek", "topk": 3, "position": "fixed_60", "horizon_days": 20, "min_market_cap": 120},
    {"name": "qvfcoreplus_biweek_k3_fixed65_h20_mv120", "bundle": "quality_value_flow_core_plus", "freq": "biweek", "topk": 3, "position": "fixed_65", "horizon_days": 20, "min_market_cap": 120},
    {"name": "qvfcoreplus_biweek_k3_fixed70_h20_mv120", "bundle": "quality_value_flow_core_plus", "freq": "biweek", "topk": 3, "position": "fixed_70", "horizon_days": 20, "min_market_cap": 120},
    {"name": "qvfcoreplus_biweek_k3_fixed75_h20_mv120", "bundle": "quality_value_flow_core_plus", "freq": "biweek", "topk": 3, "position": "fixed_75", "horizon_days": 20, "min_market_cap": 120},
    {"name": "qvfcoreplus_biweek_k3_fixed80_h20_mv120", "bundle": "quality_value_flow_core_plus", "freq": "biweek", "topk": 3, "position": "fixed_80", "horizon_days": 20, "min_market_cap": 120},
    {"name": "qvfcoreplus_biweek_k3_fixed84_h20_mv120", "bundle": "quality_value_flow_core_plus", "freq": "biweek", "topk": 3, "position": "fixed_84", "horizon_days": 20, "min_market_cap": 120},
    {"name": "qvfcoreplus_biweek_k3_gate_balanced_h20_mv120", "bundle": "quality_value_flow_core_plus", "freq": "biweek", "topk": 3, "position": "gate_balanced", "horizon_days": 20, "min_market_cap": 120},
    {"name": "qvfcoreplus_biweek_k3_gate_defensive_h20_mv120", "bundle": "quality_value_flow_core_plus", "freq": "biweek", "topk": 3, "position": "gate_defensive", "horizon_days": 20, "min_market_cap": 120},
]

NEW_ALPHA_QVF_CORE_RISK_FINE = [
    {"name": "qvfcoreplus_biweek_k3_fixed58_h20_mv120", "bundle": "quality_value_flow_core_plus", "freq": "biweek", "topk": 3, "position": "fixed_58", "horizon_days": 20, "min_market_cap": 120},
    {"name": "qvfcoreplus_biweek_k3_fixed60_h20_mv120", "bundle": "quality_value_flow_core_plus", "freq": "biweek", "topk": 3, "position": "fixed_60", "horizon_days": 20, "min_market_cap": 120},
    {"name": "qvfcoreplus_biweek_k3_fixed62_h20_mv120", "bundle": "quality_value_flow_core_plus", "freq": "biweek", "topk": 3, "position": "fixed_62", "horizon_days": 20, "min_market_cap": 120},
    {"name": "qvfcoreplus_biweek_k3_fixed64_h20_mv120", "bundle": "quality_value_flow_core_plus", "freq": "biweek", "topk": 3, "position": "fixed_64", "horizon_days": 20, "min_market_cap": 120},
]

QVF_CORE_OVERLAY_LONG = [
    {"name": "qvfcoreplus_k3_fixed70_ovA", "bundle": "quality_value_flow_core_plus", "freq": "biweek", "topk": 3, "position": "fixed_70", "horizon_days": 20, "min_market_cap": 120, "overlay_overrides": {"enabled": True, "target_vol": 0.18, "vol_lookback": 20, "trend_lookback": 20, "trend_exposure": 0.75, "dd_soft": 0.04, "dd_hard": 0.06, "soft_exposure": 0.85, "hard_exposure": 0.55, "exposure_min": 0.0, "exposure_max": 1.0}},
    {"name": "qvfcoreplus_k3_fixed75_ovA", "bundle": "quality_value_flow_core_plus", "freq": "biweek", "topk": 3, "position": "fixed_75", "horizon_days": 20, "min_market_cap": 120, "overlay_overrides": {"enabled": True, "target_vol": 0.18, "vol_lookback": 20, "trend_lookback": 20, "trend_exposure": 0.75, "dd_soft": 0.04, "dd_hard": 0.06, "soft_exposure": 0.85, "hard_exposure": 0.55, "exposure_min": 0.0, "exposure_max": 1.0}},
    {"name": "qvfcoreplus_k4_fixed70_ovA", "bundle": "quality_value_flow_core_plus", "freq": "biweek", "topk": 4, "position": "fixed_70", "horizon_days": 20, "min_market_cap": 120, "overlay_overrides": {"enabled": True, "target_vol": 0.18, "vol_lookback": 20, "trend_lookback": 20, "trend_exposure": 0.75, "dd_soft": 0.04, "dd_hard": 0.06, "soft_exposure": 0.85, "hard_exposure": 0.55, "exposure_min": 0.0, "exposure_max": 1.0}},
    {"name": "qvfcoreplus_k4_fixed75_ovA", "bundle": "quality_value_flow_core_plus", "freq": "biweek", "topk": 4, "position": "fixed_75", "horizon_days": 20, "min_market_cap": 120, "overlay_overrides": {"enabled": True, "target_vol": 0.18, "vol_lookback": 20, "trend_lookback": 20, "trend_exposure": 0.75, "dd_soft": 0.04, "dd_hard": 0.06, "soft_exposure": 0.85, "hard_exposure": 0.55, "exposure_min": 0.0, "exposure_max": 1.0}},
    {"name": "qvfcoreplus_k5_fixed75_ovA", "bundle": "quality_value_flow_core_plus", "freq": "biweek", "topk": 5, "position": "fixed_75", "horizon_days": 20, "min_market_cap": 120, "overlay_overrides": {"enabled": True, "target_vol": 0.18, "vol_lookback": 20, "trend_lookback": 20, "trend_exposure": 0.75, "dd_soft": 0.04, "dd_hard": 0.06, "soft_exposure": 0.85, "hard_exposure": 0.55, "exposure_min": 0.0, "exposure_max": 1.0}},
    {"name": "qvfcoreplus_k5_fixed80_ovA", "bundle": "quality_value_flow_core_plus", "freq": "biweek", "topk": 5, "position": "fixed_80", "horizon_days": 20, "min_market_cap": 120, "overlay_overrides": {"enabled": True, "target_vol": 0.18, "vol_lookback": 20, "trend_lookback": 20, "trend_exposure": 0.75, "dd_soft": 0.04, "dd_hard": 0.06, "soft_exposure": 0.85, "hard_exposure": 0.55, "exposure_min": 0.0, "exposure_max": 1.0}},
    {"name": "qvfcoreplus_k4_fixed75_ovB", "bundle": "quality_value_flow_core_plus", "freq": "biweek", "topk": 4, "position": "fixed_75", "horizon_days": 20, "min_market_cap": 120, "overlay_overrides": {"enabled": True, "target_vol": 0.20, "vol_lookback": 20, "trend_lookback": 20, "trend_exposure": 0.80, "dd_soft": 0.03, "dd_hard": 0.06, "soft_exposure": 0.85, "hard_exposure": 0.55, "exposure_min": 0.0, "exposure_max": 1.0}},
    {"name": "qvfcoreplus_k5_fixed75_ovB", "bundle": "quality_value_flow_core_plus", "freq": "biweek", "topk": 5, "position": "fixed_75", "horizon_days": 20, "min_market_cap": 120, "overlay_overrides": {"enabled": True, "target_vol": 0.20, "vol_lookback": 20, "trend_lookback": 20, "trend_exposure": 0.80, "dd_soft": 0.03, "dd_hard": 0.06, "soft_exposure": 0.85, "hard_exposure": 0.55, "exposure_min": 0.0, "exposure_max": 1.0}},
    {"name": "qvfcoreplus_k5_fixed80_ovB", "bundle": "quality_value_flow_core_plus", "freq": "biweek", "topk": 5, "position": "fixed_80", "horizon_days": 20, "min_market_cap": 120, "overlay_overrides": {"enabled": True, "target_vol": 0.20, "vol_lookback": 20, "trend_lookback": 20, "trend_exposure": 0.80, "dd_soft": 0.03, "dd_hard": 0.06, "soft_exposure": 0.85, "hard_exposure": 0.55, "exposure_min": 0.0, "exposure_max": 1.0}},
    {"name": "qvfcoreplus_k6_fixed80_ovB", "bundle": "quality_value_flow_core_plus", "freq": "biweek", "topk": 6, "position": "fixed_80", "horizon_days": 20, "min_market_cap": 120, "overlay_overrides": {"enabled": True, "target_vol": 0.20, "vol_lookback": 20, "trend_lookback": 20, "trend_exposure": 0.80, "dd_soft": 0.03, "dd_hard": 0.06, "soft_exposure": 0.85, "hard_exposure": 0.55, "exposure_min": 0.0, "exposure_max": 1.0}},
]

QVF_CORE_OVERLAY_COMPRESS = [
    {"name": "qvfcoreplus_k6_fixed80_ovC", "bundle": "quality_value_flow_core_plus", "freq": "biweek", "topk": 6, "position": "fixed_80", "horizon_days": 20, "min_market_cap": 120, "overlay_overrides": {"enabled": True, "target_vol": 0.19, "vol_lookback": 20, "trend_lookback": 20, "trend_exposure": 0.78, "dd_soft": 0.03, "dd_hard": 0.055, "soft_exposure": 0.83, "hard_exposure": 0.50, "exposure_min": 0.0, "exposure_max": 1.0}},
    {"name": "qvfcoreplus_k6_fixed80_ovD", "bundle": "quality_value_flow_core_plus", "freq": "biweek", "topk": 6, "position": "fixed_80", "horizon_days": 20, "min_market_cap": 120, "overlay_overrides": {"enabled": True, "target_vol": 0.18, "vol_lookback": 20, "trend_lookback": 20, "trend_exposure": 0.78, "dd_soft": 0.03, "dd_hard": 0.055, "soft_exposure": 0.82, "hard_exposure": 0.50, "exposure_min": 0.0, "exposure_max": 1.0}},
    {"name": "qvfcoreplus_k6_fixed80_ovE", "bundle": "quality_value_flow_core_plus", "freq": "biweek", "topk": 6, "position": "fixed_80", "horizon_days": 20, "min_market_cap": 120, "overlay_overrides": {"enabled": True, "target_vol": 0.18, "vol_lookback": 20, "trend_lookback": 20, "trend_exposure": 0.75, "dd_soft": 0.025, "dd_hard": 0.05, "soft_exposure": 0.80, "hard_exposure": 0.45, "exposure_min": 0.0, "exposure_max": 1.0}},
    {"name": "qvfcoreplus_k6_fixed80_ovF", "bundle": "quality_value_flow_core_plus", "freq": "biweek", "topk": 6, "position": "fixed_80", "horizon_days": 20, "min_market_cap": 120, "overlay_overrides": {"enabled": True, "target_vol": 0.19, "vol_lookback": 20, "trend_lookback": 20, "trend_exposure": 0.75, "dd_soft": 0.025, "dd_hard": 0.05, "soft_exposure": 0.82, "hard_exposure": 0.48, "exposure_min": 0.0, "exposure_max": 1.0}},
    {"name": "qvfcoreplus_k7_fixed80_ovC", "bundle": "quality_value_flow_core_plus", "freq": "biweek", "topk": 7, "position": "fixed_80", "horizon_days": 20, "min_market_cap": 120, "overlay_overrides": {"enabled": True, "target_vol": 0.19, "vol_lookback": 20, "trend_lookback": 20, "trend_exposure": 0.78, "dd_soft": 0.03, "dd_hard": 0.055, "soft_exposure": 0.83, "hard_exposure": 0.50, "exposure_min": 0.0, "exposure_max": 1.0}},
    {"name": "qvfcoreplus_k7_fixed80_ovD", "bundle": "quality_value_flow_core_plus", "freq": "biweek", "topk": 7, "position": "fixed_80", "horizon_days": 20, "min_market_cap": 120, "overlay_overrides": {"enabled": True, "target_vol": 0.18, "vol_lookback": 20, "trend_lookback": 20, "trend_exposure": 0.78, "dd_soft": 0.03, "dd_hard": 0.055, "soft_exposure": 0.82, "hard_exposure": 0.50, "exposure_min": 0.0, "exposure_max": 1.0}},
    {"name": "qvfcoreplus_k7_fixed80_ovE", "bundle": "quality_value_flow_core_plus", "freq": "biweek", "topk": 7, "position": "fixed_80", "horizon_days": 20, "min_market_cap": 120, "overlay_overrides": {"enabled": True, "target_vol": 0.18, "vol_lookback": 20, "trend_lookback": 20, "trend_exposure": 0.75, "dd_soft": 0.025, "dd_hard": 0.05, "soft_exposure": 0.80, "hard_exposure": 0.45, "exposure_min": 0.0, "exposure_max": 1.0}},
    {"name": "qvfcoreplus_k6_fixed75_ovC", "bundle": "quality_value_flow_core_plus", "freq": "biweek", "topk": 6, "position": "fixed_75", "horizon_days": 20, "min_market_cap": 120, "overlay_overrides": {"enabled": True, "target_vol": 0.19, "vol_lookback": 20, "trend_lookback": 20, "trend_exposure": 0.78, "dd_soft": 0.03, "dd_hard": 0.055, "soft_exposure": 0.83, "hard_exposure": 0.50, "exposure_min": 0.0, "exposure_max": 1.0}},
    {"name": "qvfcoreplus_k6_fixed75_ovD", "bundle": "quality_value_flow_core_plus", "freq": "biweek", "topk": 6, "position": "fixed_75", "horizon_days": 20, "min_market_cap": 120, "overlay_overrides": {"enabled": True, "target_vol": 0.18, "vol_lookback": 20, "trend_lookback": 20, "trend_exposure": 0.78, "dd_soft": 0.03, "dd_hard": 0.055, "soft_exposure": 0.82, "hard_exposure": 0.50, "exposure_min": 0.0, "exposure_max": 1.0}},
    {"name": "qvfcoreplus_k7_fixed75_ovC", "bundle": "quality_value_flow_core_plus", "freq": "biweek", "topk": 7, "position": "fixed_75", "horizon_days": 20, "min_market_cap": 120, "overlay_overrides": {"enabled": True, "target_vol": 0.19, "vol_lookback": 20, "trend_lookback": 20, "trend_exposure": 0.78, "dd_soft": 0.03, "dd_hard": 0.055, "soft_exposure": 0.83, "hard_exposure": 0.50, "exposure_min": 0.0, "exposure_max": 1.0}},
    {"name": "qvfcoreplus_k5_fixed80_ovC", "bundle": "quality_value_flow_core_plus", "freq": "biweek", "topk": 5, "position": "fixed_80", "horizon_days": 20, "min_market_cap": 120, "overlay_overrides": {"enabled": True, "target_vol": 0.19, "vol_lookback": 20, "trend_lookback": 20, "trend_exposure": 0.78, "dd_soft": 0.03, "dd_hard": 0.055, "soft_exposure": 0.83, "hard_exposure": 0.50, "exposure_min": 0.0, "exposure_max": 1.0}},
]

SCAN_PRESETS = {
    "batch_scan_v1": BATCH_SCAN_V1,
    "focused_h20": FOCUSED_H20,
    "focused_h20_stoploss": FOCUSED_H20_STOPLOSS,
    "focused_qvf_risk_return": FOCUSED_QVF_RISK_RETURN,
    "edge_qvf_frontier": EDGE_QVF_FRONTIER,
    "edge_qvf_final": EDGE_QVF_FINAL,
    "edge_qvf_horizon": EDGE_QVF_HORIZON,
    "new_alpha_qvf": NEW_ALPHA_QVF,
    "new_alpha_qvf_core": NEW_ALPHA_QVF_CORE,
    "new_alpha_qvf_core_risk": NEW_ALPHA_QVF_CORE_RISK,
    "new_alpha_qvf_core_risk_fine": NEW_ALPHA_QVF_CORE_RISK_FINE,
    "qvf_core_overlay_long": QVF_CORE_OVERLAY_LONG,
    "qvf_core_overlay_compress": QVF_CORE_OVERLAY_COMPRESS,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="config/models/lgb_10d.yaml")
    parser.add_argument("--search-name", default="batch_scan_v1")
    parser.add_argument("--preset", default="batch_scan_v1", choices=sorted(SCAN_PRESETS))
    return parser.parse_args()


def build_candidate_config(base_cfg: dict, args, candidate: dict) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["name"] = candidate["name"]
    cfg["data"]["feature_columns"] = FEATURE_BUNDLES[candidate["bundle"]]
    cfg["label"]["horizon_days"] = int(candidate["horizon_days"])
    cfg["selection"]["freq"] = candidate["freq"]
    cfg["selection"]["topk"] = int(candidate["topk"])
    cfg["selection"]["min_market_cap"] = float(candidate["min_market_cap"])
    if candidate.get("selection_overrides"):
        cfg["selection"].update(copy.deepcopy(candidate["selection_overrides"]))
    if candidate.get("overlay_overrides"):
        cfg.setdefault("overlay", {})
        cfg["overlay"].update(copy.deepcopy(candidate["overlay_overrides"]))
    cfg["position"] = copy.deepcopy(POSITION_PRESETS[candidate["position"]])
    root = (
        PROJECT_ROOT
        / "results"
        / "model_signals"
        / "search_runs"
        / args.search_name
        / candidate["name"]
    ).resolve()
    cfg["output"]["root"] = str(root)
    cfg["model"]["params"]["verbosity"] = -1
    return cfg


def summarize_candidate(candidate: dict, train_summary: dict, score_summary: dict, backtest_summary: dict, elapsed: float) -> dict:
    metrics = train_summary.get("metrics", {})
    return {
        "name": candidate["name"],
        "bundle": candidate["bundle"],
        "freq": candidate["freq"],
        "topk": candidate["topk"],
        "position": candidate["position"],
        "horizon_days": candidate["horizon_days"],
        "min_market_cap": candidate["min_market_cap"],
        "selection_mode": candidate.get("selection_overrides", {}).get("mode", "factor_topk"),
        "stoploss_drawdown": candidate.get("selection_overrides", {}).get("stoploss_drawdown"),
        "stoploss_lookback_days": candidate.get("selection_overrides", {}).get("stoploss_lookback_days"),
        "replacement_pool_size": candidate.get("selection_overrides", {}).get("replacement_pool_size"),
        "backend": metrics.get("backend"),
        "valid_mean_rank_ic": metrics.get("valid_mean_rank_ic", 0.0),
        "annual_return": backtest_summary.get("annual_return", 0.0),
        "max_drawdown": backtest_summary.get("max_drawdown", 0.0),
        "sharpe_ratio": backtest_summary.get("sharpe_ratio", 0.0),
        "selection_dates": score_summary.get("selection_dates", 0),
        "elapsed_seconds": elapsed,
        "results_file": backtest_summary.get("results_file"),
    }


def main():
    args = parse_args()
    base_cfg = load_predictive_config(args.base_config)
    search_root = (
        PROJECT_ROOT / "results" / "model_signals" / "search_runs" / args.search_name
    ).resolve()
    search_root.mkdir(parents=True, exist_ok=True)
    candidates = SCAN_PRESETS[args.preset]

    rows = []
    for idx, candidate in enumerate(candidates, start=1):
        print(f"[{idx}/{len(candidates)}] {candidate['name']}")
        cfg = build_candidate_config(base_cfg, args, candidate)
        start = time.perf_counter()
        train_summary = train_from_config(cfg)
        score_summary = score_from_config(cfg)
        _, backtest_summary = backtest_from_config(cfg)
        elapsed = time.perf_counter() - start
        row = summarize_candidate(candidate, train_summary, score_summary, backtest_summary, elapsed)
        rows.append(row)
        print(
            "  "
            f"ic={row['valid_mean_rank_ic']:.4f} "
            f"annual={row['annual_return']:.2%} "
            f"dd={row['max_drawdown']:.2%} "
            f"sharpe={row['sharpe_ratio']:.3f} "
            f"elapsed={row['elapsed_seconds']:.1f}s"
        )

    result_df = pd.DataFrame(rows).sort_values(
        ["annual_return", "max_drawdown", "sharpe_ratio"],
        ascending=[False, False, False],
    )
    csv_path = search_root / "summary.csv"
    result_df.to_csv(csv_path, index=False)

    payload = {
        "search_name": args.search_name,
        "preset": args.preset,
        "base_config": str(Path(args.base_config).resolve()),
        "summary_csv": str(csv_path),
        "best_by_annual_return": result_df.iloc[0].to_dict() if not result_df.empty else {},
        "candidates": rows,
    }
    save_json(payload, search_root / "summary.json")
    print(f"[OK] 汇总已保存: {csv_path}")


if __name__ == "__main__":
    main()
