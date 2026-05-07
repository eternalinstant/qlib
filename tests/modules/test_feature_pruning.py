from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.modeling.feature_pruning import (  # noqa: E402
    category_status_rows,
    classify_feature_columns,
    extract_feature_importance_map,
    ordered_category_removals,
)


def test_classify_feature_columns_groups_current_strategy_features():
    feature_columns = [
        "book_to_market",
        "ebit_to_mv",
        "ebitda_to_mv",
        "ocf_to_ev",
        "fcff_to_mv",
        "roic_proxy",
        "roe_fina",
        "roa_fina",
        "current_ratio_fina",
        "debt_to_assets_fina",
        "total_revenue_inc",
        "operate_profit_inc",
        "smart_ratio_5d",
        "net_mf_amount_20d",
        "ROC20",
        "CORD20",
        "rank_value_profit_core",
        "rank_flow_momentum_core",
        "rank_growth_quality_core",
        "rank_balance_core",
        "qvf_core_alpha",
        "qvf_core_dynamic",
    ]

    grouped = classify_feature_columns(feature_columns)

    assert grouped["value_cashflow"] == [
        "book_to_market",
        "ebit_to_mv",
        "ebitda_to_mv",
        "ocf_to_ev",
        "fcff_to_mv",
    ]
    assert grouped["profitability_quality"] == ["roic_proxy", "roe_fina", "roa_fina"]
    assert grouped["balance_sheet"] == ["current_ratio_fina", "debt_to_assets_fina"]
    assert grouped["growth_cashflow"] == ["total_revenue_inc", "operate_profit_inc"]
    assert grouped["flow_momentum"] == ["smart_ratio_5d", "net_mf_amount_20d"]
    assert grouped["alpha158_tech"] == ["ROC20", "CORD20"]
    assert grouped["core_pillars"] == [
        "rank_value_profit_core",
        "rank_flow_momentum_core",
        "rank_growth_quality_core",
        "rank_balance_core",
    ]
    assert grouped["meta_blends"] == ["qvf_core_alpha", "qvf_core_dynamic"]


def test_ordered_category_removals_prefers_low_importance_features_within_category():
    feature_columns = [
        "book_to_market",
        "ebit_to_mv",
        "ebitda_to_mv",
        "ocf_to_ev",
        "fcff_to_mv",
    ]
    importance_map = {
        "book_to_market": 22.0,
        "ebit_to_mv": 18.0,
        "ebitda_to_mv": 5.0,
        "ocf_to_ev": 8.0,
        "fcff_to_mv": 3.0,
    }

    ordered = ordered_category_removals(
        feature_columns=feature_columns,
        importance_map=importance_map,
        category="value_cashflow",
    )

    assert ordered == ["fcff_to_mv", "ebitda_to_mv", "ocf_to_ev", "ebit_to_mv", "book_to_market"]


def test_category_status_rows_report_targets_and_excess_counts():
    feature_columns = [
        "book_to_market",
        "ebit_to_mv",
        "ebitda_to_mv",
        "ocf_to_ev",
        "fcff_to_mv",
        "ROC20",
        "RANK20",
        "CORD20",
        "qvf_core_alpha",
        "qvf_core_interaction",
        "qvf_core_dynamic",
    ]

    rows = {row["category"]: row for row in category_status_rows(feature_columns)}

    assert rows["value_cashflow"]["count"] == 5
    assert rows["value_cashflow"]["target_count"] == 3
    assert rows["value_cashflow"]["excess_count"] == 2
    assert rows["alpha158_tech"]["count"] == 3
    assert rows["alpha158_tech"]["target_count"] == 2
    assert rows["alpha158_tech"]["excess_count"] == 1
    assert rows["meta_blends"]["count"] == 3
    assert rows["meta_blends"]["target_count"] == 2
    assert rows["meta_blends"]["excess_count"] == 1


def test_extract_feature_importance_map_reads_model_feature_importances():
    class DummyModel:
        feature_importances_ = [5.0, 1.5, 0.0]

    importance_map = extract_feature_importance_map(
        DummyModel(),
        ["book_to_market", "roe_fina", "smart_ratio_5d"],
    )

    assert importance_map == {
        "book_to_market": 5.0,
        "roe_fina": 1.5,
        "smart_ratio_5d": 0.0,
    }
