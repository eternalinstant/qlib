"""因子分组与逐步精简辅助工具。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from typing import Iterable


CATEGORY_ORDER = [
    "value_cashflow",
    "profitability_quality",
    "balance_sheet",
    "growth_cashflow",
    "flow_momentum",
    "alpha158_tech",
    "meta_blends",
    "core_pillars",
]


FEATURE_CATEGORY_SPECS = {
    "value_cashflow": {
        "label": "估值现金流",
        "target_count": 3,
        "features": [
            "book_to_market",
            "ebit_to_mv",
            "ebitda_to_mv",
            "ocf_to_mv",
            "ocf_to_ev",
            "fcff_to_mv",
        ],
    },
    "profitability_quality": {
        "label": "盈利质量",
        "target_count": 3,
        "features": [
            "roic_proxy",
            "roe_fina",
            "roe_dt_fina",
            "roa_fina",
            "net_margin",
        ],
    },
    "balance_sheet": {
        "label": "资产负债表",
        "target_count": 2,
        "features": [
            "current_ratio_fina",
            "quick_ratio_fina",
            "debt_to_assets_fina",
        ],
    },
    "growth_cashflow": {
        "label": "成长现金流",
        "target_count": 2,
        "features": [
            "total_revenue_inc",
            "operate_profit_inc",
            "n_cashflow_act",
        ],
    },
    "flow_momentum": {
        "label": "资金流动量",
        "target_count": 2,
        "features": [
            "smart_ratio_5d",
            "net_mf_amount_5d",
            "net_mf_amount_20d",
            "net_mf_vol_5d",
        ],
    },
    "alpha158_tech": {
        "label": "Alpha158技术面",
        "target_count": 2,
        "features": [
            "ROC20",
            "RSV20",
            "RANK20",
            "CORD20",
            "VSUMD20",
        ],
    },
    "meta_blends": {
        "label": "高阶组合",
        "target_count": 2,
        "features": [
            "qvf_core_alpha",
            "qvf_core_interaction",
            "qvf_core_dynamic",
        ],
    },
    "core_pillars": {
        "label": "核心支柱",
        "target_count": 4,
        "features": [
            "rank_value_profit_core",
            "rank_flow_momentum_core",
            "rank_growth_quality_core",
            "rank_balance_core",
        ],
    },
}


def classify_feature_columns(feature_columns: Iterable[str]) -> dict[str, list[str]]:
    feature_set = [str(col) for col in feature_columns]
    grouped: dict[str, list[str]] = {}
    assigned = set()
    for category in CATEGORY_ORDER:
        spec = FEATURE_CATEGORY_SPECS[category]
        present = [name for name in spec["features"] if name in feature_set]
        if present:
            grouped[category] = present
            assigned.update(present)
    unknown = [name for name in feature_set if name not in assigned]
    if unknown:
        grouped["uncategorized"] = unknown
    return grouped


def category_status_rows(feature_columns: Iterable[str]) -> list[dict]:
    grouped = classify_feature_columns(feature_columns)
    rows: list[dict] = []
    for category in CATEGORY_ORDER:
        spec = FEATURE_CATEGORY_SPECS[category]
        present = grouped.get(category, [])
        count = len(present)
        target = int(spec["target_count"])
        rows.append(
            {
                "category": category,
                "label": spec["label"],
                "count": count,
                "target_count": target,
                "excess_count": max(count - target, 0),
                "features": present,
            }
        )
    if "uncategorized" in grouped:
        rows.append(
            {
                "category": "uncategorized",
                "label": "未分类",
                "count": len(grouped["uncategorized"]),
                "target_count": len(grouped["uncategorized"]),
                "excess_count": 0,
                "features": grouped["uncategorized"],
            }
        )
    return rows


def ordered_category_removals(
    feature_columns: Iterable[str],
    importance_map: dict[str, float],
    category: str,
) -> list[str]:
    grouped = classify_feature_columns(feature_columns)
    present = list(grouped.get(category, []))
    return sorted(
        present,
        key=lambda name: (float(importance_map.get(name, 0.0)), name),
    )


def extract_feature_importance_map(model, feature_columns: Iterable[str]) -> dict[str, float]:
    feature_list = [str(col) for col in feature_columns]
    values = None

    if hasattr(model, "feature_importances_"):
        values = np.asarray(getattr(model, "feature_importances_"), dtype=float)
    elif hasattr(model, "coef_"):
        values = np.abs(np.asarray(getattr(model, "coef_"), dtype=float)).reshape(-1)

    if values is None or len(values) != len(feature_list):
        values = np.zeros(len(feature_list), dtype=float)

    return {name: float(value) for name, value in zip(feature_list, values)}


def feature_importance_frame(model, feature_columns: Iterable[str]) -> pd.DataFrame:
    importance_map = extract_feature_importance_map(model, feature_columns)
    frame = pd.DataFrame(
        [{"feature": name, "importance": value} for name, value in importance_map.items()]
    )
    if frame.empty:
        return frame
    total = float(frame["importance"].sum())
    frame["importance_ratio"] = frame["importance"] / total if total > 0 else 0.0
    return frame.sort_values(["importance", "feature"], ascending=[False, True]).reset_index(drop=True)
