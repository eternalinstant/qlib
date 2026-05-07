import pickle
import sys
from pathlib import Path

import pandas as pd
import pytest

from modules.backtest.base import BacktestResult

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.modeling.predictive_signal import (
    ModelSignalStrategy,
    _alpha158_feature_map,
    backtest_from_config,
    build_walk_forward_windows,
    augment_with_derived_features,
    build_forward_return_labels,
    evaluate_cross_sectional_predictions,
    load_feature_frame,
    load_predictive_config,
    materialize_selections_from_scores,
    required_raw_columns,
    resolve_regressor,
)


class TestForwardReturnLabels:
    def test_build_forward_return_labels_uses_instrumentwise_shift(self):
        index = pd.MultiIndex.from_product(
            [
                pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
                ["SZ000001", "SZ000002"],
            ],
            names=["datetime", "instrument"],
        )
        close = pd.Series(
            [10.0, 20.0, 11.0, 18.0, 12.0, 21.0],
            index=index,
            name="close",
        )

        labels = build_forward_return_labels(close, horizon_days=1)

        assert labels.loc[(pd.Timestamp("2024-01-02"), "SZ000001")] == pytest.approx(0.1)
        assert labels.loc[(pd.Timestamp("2024-01-03"), "SZ000001")] == pytest.approx(
            12.0 / 11.0 - 1
        )
        assert labels.loc[(pd.Timestamp("2024-01-02"), "SZ000002")] == pytest.approx(-0.1)
        assert pd.isna(labels.loc[(pd.Timestamp("2024-01-04"), "SZ000001")])


class TestRegressorFallback:
    def test_resolve_regressor_falls_back_when_lightgbm_is_unavailable(self, monkeypatch):
        real_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "lightgbm":
                raise OSError("missing libomp")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", fake_import)

        model, backend = resolve_regressor(
            preferred_backend="lightgbm",
            params={"learning_rate": 0.05, "n_estimators": 64, "max_depth": 4},
        )

        assert backend == "sklearn_hist_gbm"
        assert model.__class__.__name__ == "HistGradientBoostingRegressor"

    def test_resolve_regressor_maps_min_samples_leaf_for_lightgbm(self):
        lightgbm = pytest.importorskip("lightgbm")
        model, backend = resolve_regressor(
            preferred_backend="lightgbm",
            params={"learning_rate": 0.05, "n_estimators": 16, "min_samples_leaf": 31},
        )

        assert backend == "lightgbm"
        assert isinstance(model, lightgbm.LGBMRegressor)
        assert model.get_params()["min_child_samples"] == 31


class TestCrossSectionalEvaluation:
    def test_evaluate_cross_sectional_predictions_reports_mean_ic(self):
        index = pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2024-01-02"), "SZ000001"),
                (pd.Timestamp("2024-01-02"), "SZ000002"),
                (pd.Timestamp("2024-01-03"), "SZ000001"),
                (pd.Timestamp("2024-01-03"), "SZ000002"),
            ],
            names=["datetime", "instrument"],
        )
        scores = pd.Series([2.0, 1.0, 1.0, 2.0], index=index, name="score")
        labels = pd.Series([0.2, 0.1, 0.1, 0.2], index=index, name="label")

        metrics = evaluate_cross_sectional_predictions(scores, labels)

        assert metrics["samples"] == 4
        assert metrics["dated_groups"] == 2
        assert metrics["mean_rank_ic"] == pytest.approx(1.0)


class TestDerivedFeatures:
    def test_alpha158_feature_map_contains_standard_names(self):
        feature_map = _alpha158_feature_map(
            {
                "price": {"windows": [0], "feature": ["OPEN"]},
                "rolling": {"windows": [20], "include": ["MA", "STD", "RSV", "VSUMD", "CORD"]},
            }
        )

        assert "OPEN0" in feature_map
        assert "MA20" in feature_map
        assert "STD20" in feature_map
        assert "RSV20" in feature_map
        assert "VSUMD20" in feature_map
        assert "CORD20" in feature_map

    def test_required_raw_columns_expand_recursive_dependencies(self):
        cols = required_raw_columns(["qvf_core_dynamic", "rank_balance_strength"])

        assert "book_to_market" in cols
        assert "ebitda_to_mv" in cols
        assert "net_mf_amount_5d" in cols
        assert "net_mf_amount_20d" in cols
        assert "smart_ratio_5d" in cols
        assert "money_cap" in cols
        assert "total_revenue_inc" in cols
        assert "total_liab" in cols

    def test_required_raw_columns_expand_grouped_aggregate_dependencies(self):
        cols = required_raw_columns(["qvf_group_blend"])

        assert "book_to_market" in cols
        assert "ocf_to_mv" in cols
        assert "fcff_to_mv" in cols
        assert "roic_proxy" in cols
        assert "roe_dt_fina" in cols
        assert "current_ratio_fina" in cols
        assert "quick_ratio_fina" in cols
        assert "total_revenue_inc" in cols
        assert "n_cashflow_act" in cols
        assert "net_mf_vol_5d" in cols
        assert "money_cap" in cols
        assert "total_liab" in cols

    def test_augment_with_derived_features_builds_rank_interaction_and_core_columns(self):
        index = pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2024-01-02"), "SZ000001"),
                (pd.Timestamp("2024-01-02"), "SZ000002"),
                (pd.Timestamp("2024-01-09"), "SZ000001"),
                (pd.Timestamp("2024-01-09"), "SZ000002"),
            ],
            names=["datetime", "instrument"],
        )
        frame = pd.DataFrame(
            {
                "book_to_market": [1.0, 2.0, 1.5, 2.5],
                "ebitda_to_mv": [0.12, 0.30, 0.16, 0.33],
                "ebit_to_mv": [0.10, 0.25, 0.14, 0.27],
                "roe_fina": [0.10, 0.20, 0.15, 0.25],
                "roa_fina": [0.04, 0.07, 0.05, 0.08],
                "ocf_to_ev": [0.2, 0.1, 0.3, 0.2],
                "debt_to_assets_fina": [0.5, 0.3, 0.4, 0.2],
                "current_ratio_fina": [1.0, 1.3, 1.1, 1.4],
                "quick_ratio_fina": [0.8, 1.1, 0.9, 1.2],
                "money_cap": [40.0, 70.0, 50.0, 80.0],
                "total_liab": [100.0, 80.0, 90.0, 70.0],
                "smart_ratio_5d": [1.0, 2.0, 1.5, 2.5],
                "net_mf_amount_5d": [4.0, 9.0, 5.0, 10.0],
                "net_mf_amount_20d": [10.0, 20.0, 11.0, 21.0],
                "total_revenue_inc": [0.05, 0.12, 0.07, 0.14],
                "operate_profit_inc": [0.04, 0.11, 0.06, 0.13],
                "net_margin": [0.08, 0.15, 0.09, 0.16],
            },
            index=index,
        )

        out = augment_with_derived_features(
            frame,
            [
                "qvf_rank_blend",
                "quality_value_interaction",
                "rank_value_delta_1",
                "qvf_core_alpha",
                "qvf_core_interaction",
                "qvf_core_dynamic",
            ],
        )

        assert out.loc[(pd.Timestamp("2024-01-02"), "SZ000002"), "quality_value_interaction"] > out.loc[
            (pd.Timestamp("2024-01-02"), "SZ000001"), "quality_value_interaction"
        ]
        assert pd.isna(out.loc[(pd.Timestamp("2024-01-02"), "SZ000001"), "rank_value_delta_1"])
        assert out.loc[(pd.Timestamp("2024-01-09"), "SZ000002"), "qvf_rank_blend"] > out.loc[
            (pd.Timestamp("2024-01-09"), "SZ000001"), "qvf_rank_blend"
        ]
        assert out.loc[(pd.Timestamp("2024-01-09"), "SZ000002"), "qvf_core_alpha"] > out.loc[
            (pd.Timestamp("2024-01-09"), "SZ000001"), "qvf_core_alpha"
        ]
        assert out.loc[(pd.Timestamp("2024-01-09"), "SZ000002"), "qvf_core_interaction"] > out.loc[
            (pd.Timestamp("2024-01-09"), "SZ000001"), "qvf_core_interaction"
        ]
        assert out["qvf_core_dynamic"].notna().sum() >= 2

    def test_augment_with_derived_features_builds_grouped_aggregate_columns(self):
        index = pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2024-01-02"), "SZ000001"),
                (pd.Timestamp("2024-01-02"), "SZ000002"),
                (pd.Timestamp("2024-01-09"), "SZ000001"),
                (pd.Timestamp("2024-01-09"), "SZ000002"),
            ],
            names=["datetime", "instrument"],
        )
        frame = pd.DataFrame(
            {
                "book_to_market": [1.0, 2.0, 1.5, 2.5],
                "roic_proxy": [0.08, 0.16, 0.10, 0.18],
                "ebitda_to_mv": [0.12, 0.30, 0.16, 0.33],
                "ebit_to_mv": [0.10, 0.25, 0.14, 0.27],
                "ocf_to_mv": [0.09, 0.18, 0.11, 0.20],
                "ocf_to_ev": [0.2, 0.1, 0.3, 0.2],
                "fcff_to_mv": [0.08, 0.16, 0.10, 0.18],
                "roe_fina": [0.10, 0.20, 0.15, 0.25],
                "roe_dt_fina": [0.09, 0.18, 0.14, 0.23],
                "roa_fina": [0.04, 0.07, 0.05, 0.08],
                "debt_to_assets_fina": [0.5, 0.3, 0.4, 0.2],
                "current_ratio_fina": [1.0, 1.3, 1.1, 1.4],
                "quick_ratio_fina": [0.8, 1.1, 0.9, 1.2],
                "money_cap": [40.0, 70.0, 50.0, 80.0],
                "total_liab": [100.0, 80.0, 90.0, 70.0],
                "total_revenue_inc": [0.05, 0.12, 0.07, 0.14],
                "operate_profit_inc": [0.04, 0.11, 0.06, 0.13],
                "n_cashflow_act": [5.0, 9.0, 6.0, 11.0],
                "net_margin": [0.08, 0.15, 0.09, 0.16],
                "smart_ratio_5d": [1.0, 2.0, 1.5, 2.5],
                "net_mf_amount_5d": [4.0, 9.0, 5.0, 10.0],
                "net_mf_amount_20d": [10.0, 20.0, 11.0, 21.0],
                "net_mf_vol_5d": [100.0, 200.0, 110.0, 220.0],
            },
            index=index,
        )

        out = augment_with_derived_features(
            frame,
            [
                "rank_value_cashflow_core",
                "rank_profitability_quality_core",
                "rank_balance_sheet_core",
                "rank_growth_cashflow_core",
                "rank_flow_liquidity_core",
                "qvf_group_alpha",
                "qvf_group_interaction",
                "qvf_group_quality_anchor",
                "qvf_financial_gate",
                "qvf_group_blend",
            ],
        )

        assert out.loc[(pd.Timestamp("2024-01-09"), "SZ000002"), "rank_value_cashflow_core"] > out.loc[
            (pd.Timestamp("2024-01-09"), "SZ000001"), "rank_value_cashflow_core"
        ]
        assert out.loc[(pd.Timestamp("2024-01-09"), "SZ000002"), "rank_flow_liquidity_core"] > out.loc[
            (pd.Timestamp("2024-01-09"), "SZ000001"), "rank_flow_liquidity_core"
        ]
        assert out.loc[(pd.Timestamp("2024-01-09"), "SZ000002"), "qvf_group_alpha"] > out.loc[
            (pd.Timestamp("2024-01-09"), "SZ000001"), "qvf_group_alpha"
        ]
        assert out.loc[(pd.Timestamp("2024-01-09"), "SZ000002"), "qvf_financial_gate"] > out.loc[
            (pd.Timestamp("2024-01-09"), "SZ000001"), "qvf_financial_gate"
        ]
        assert out.loc[(pd.Timestamp("2024-01-09"), "SZ000002"), "qvf_group_blend"] > out.loc[
            (pd.Timestamp("2024-01-09"), "SZ000001"), "qvf_group_blend"
        ]


class TestModelSignalStrategy:
    def test_model_signal_strategy_loads_custom_selection_csv(self, tmp_path):
        selection_csv = tmp_path / "selections.csv"
        pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-09"]),
                "rank": [1, 2, 1],
                "symbol": ["SZ000001", "SZ000002", "SZ000003"],
                "score": [0.9, 0.8, 0.7],
            }
        ).to_csv(selection_csv, index=False)

        strategy = ModelSignalStrategy(
            name="lgb_signal_demo",
            selection_csv=selection_csv,
            topk=2,
            universe="csi300",
            position_model="fixed",
            position_params={"stock_pct": 0.85},
            trading_cost={"buy_commission_rate": 0.0003},
        )

        date_to_symbols, rebalance_dates = strategy.load_selections()
        controller = strategy.build_position_controller()

        assert strategy.artifact_slug() == "lgb_signal_demo"
        assert date_to_symbols[pd.Timestamp("2024-01-02")] == {"SZ000001", "SZ000002"}
        assert pd.Timestamp("2024-01-09") in rebalance_dates
        assert controller.get_allocation(pd.Timestamp("2024-01-02")).stock_pct == pytest.approx(0.85)

    def test_model_signal_strategy_roundtrips_pickled_bundle_path(self, tmp_path):
        bundle_path = tmp_path / "model.pkl"
        payload = {"backend": "sklearn_hist_gbm", "feature_columns": ["roa_fina"]}
        bundle_path.write_bytes(pickle.dumps(payload))

        strategy = ModelSignalStrategy(
            name="demo",
            selection_csv=tmp_path / "sel.csv",
            topk=8,
        )

        assert strategy.selection_csv.name == "sel.csv"
        assert pickle.loads(bundle_path.read_bytes())["backend"] == "sklearn_hist_gbm"

    def test_model_signal_strategy_supports_vol_norm_position_model(self, tmp_path):
        strategy = ModelSignalStrategy(
            name="demo_vol_norm",
            selection_csv=tmp_path / "sel.csv",
            topk=8,
            position_model="vol_norm",
            position_params={"stock_pct": 0.8, "vol_method": "inverse_sqrt"},
        )

        controller = strategy.build_position_controller()

        assert controller.get_allocation(pd.Timestamp("2024-01-02")).stock_pct == pytest.approx(0.8)


class TestPredictiveConfig:
    def test_relative_output_root_is_resolved_from_project_root(self, tmp_path):
        cfg_path = tmp_path / "demo.yaml"
        cfg_path.write_text(
            "name: demo_model\noutput:\n  root: results/model_signals/demo_model\n",
            encoding="utf-8",
        )

        cfg = load_predictive_config(cfg_path)

        assert Path(cfg["output"]["root"]).as_posix().endswith(
            "/Users/sxt/code/qlib/results/model_signals/demo_model"
        )

    def test_selection_defaults_include_stoploss_fields(self, tmp_path):
        cfg_path = tmp_path / "demo.yaml"
        cfg_path.write_text("name: demo_model\n", encoding="utf-8")

        cfg = load_predictive_config(cfg_path)

        assert cfg["selection"]["mode"] == "factor_topk"
        assert cfg["selection"]["stoploss_lookback_days"] == 20
        assert cfg["selection"]["stoploss_drawdown"] == pytest.approx(0.10)
        assert cfg["selection"]["replacement_pool_size"] == 0

    def test_overlay_defaults_include_disabled_overlay_block(self, tmp_path):
        cfg_path = tmp_path / "demo.yaml"
        cfg_path.write_text("name: demo_model\n", encoding="utf-8")

        cfg = load_predictive_config(cfg_path)

        assert cfg["overlay"]["enabled"] is False
        assert cfg["overlay"]["vol_lookback"] == 20
        assert cfg["overlay"]["exposure_max"] == pytest.approx(1.0)

    def test_walk_forward_defaults_include_disabled_block(self, tmp_path):
        cfg_path = tmp_path / "demo.yaml"
        cfg_path.write_text("name: demo_model\n", encoding="utf-8")

        cfg = load_predictive_config(cfg_path)

        assert cfg["walk_forward"]["enabled"] is False
        assert cfg["walk_forward"]["min_train_years"] == 3
        assert cfg["walk_forward"]["retrain_months"] == 12
        assert cfg["walk_forward"]["train_years"] is None


class TestWalkForwardWindows:
    def test_build_walk_forward_windows_expands_training_window(self):
        windows = build_walk_forward_windows(
            training_start="2019-01-01",
            score_start="2022-01-01",
            score_end="2024-12-31",
            min_train_years=3,
            retrain_months=12,
            train_years=None,
        )

        assert [(w.train_start, w.train_end, w.score_start, w.score_end) for w in windows] == [
            ("2019-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
            ("2019-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
            ("2019-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
        ]

    def test_build_walk_forward_windows_supports_rolling_train_length(self):
        windows = build_walk_forward_windows(
            training_start="2019-01-01",
            score_start="2022-01-01",
            score_end="2024-12-31",
            min_train_years=3,
            retrain_months=12,
            train_years=2,
        )

        assert [(w.train_start, w.train_end, w.score_start, w.score_end) for w in windows] == [
            ("2020-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
            ("2021-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
            ("2022-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
        ]


class TestSelectionMaterialization:
    def test_model_signal_selection_applies_quantile_gate_before_topk(self, monkeypatch):
        index = pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2024-01-02"), "SZ000001"),
                (pd.Timestamp("2024-01-02"), "SZ000002"),
                (pd.Timestamp("2024-01-02"), "SZ000003"),
            ],
            names=["datetime", "instrument"],
        )
        scores = pd.Series([0.99, 0.80, 0.70], index=index, name="score")
        gate_frame = pd.DataFrame({"qvf_quality_gate": [0.10, 0.80, 0.90]}, index=index)

        monkeypatch.setattr(
            "modules.modeling.predictive_signal._load_selection_filter_frame",
            lambda candidate_instruments, start_date, end_date, rebalance_dates, selection_cfg: gate_frame,
            raising=False,
        )

        selection_df = materialize_selections_from_scores(
            scores,
            pd.DatetimeIndex([pd.Timestamp("2024-01-02")]),
            {
                "topk": 2,
                "mode": "factor_topk",
                "hard_filter_quantiles": {"qvf_quality_gate": 0.50},
            },
        )

        assert selection_df["symbol"].tolist() == ["SZ000002", "SZ000003"]

    def test_stoploss_mode_passes_price_inputs_to_extract_topk(self, monkeypatch):
        index = pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2024-01-02"), "SZ000001"),
                (pd.Timestamp("2024-01-02"), "SZ000002"),
            ],
            names=["datetime", "instrument"],
        )
        scores = pd.Series([0.3, 0.2], index=index, name="score")
        rebalance_dates = pd.DatetimeIndex([pd.Timestamp("2024-01-02")])
        close_series = pd.Series([10.0, 11.0], index=index, name="close")
        captured = {}

        monkeypatch.setattr(
            "modules.modeling.predictive_signal.load_close_series",
            lambda instruments, start_date, end_date, horizon_days: close_series,
        )

        def fake_extract_topk(signal, dates, **kwargs):
            captured["kwargs"] = kwargs
            return pd.DataFrame(
                {"date": [pd.Timestamp("2024-01-02")], "rank": [1], "symbol": ["SZ000001"], "score": [0.3]}
            )

        monkeypatch.setattr("modules.modeling.predictive_signal.extract_topk", fake_extract_topk)

        selection_df = materialize_selections_from_scores(
            scores,
            rebalance_dates,
            {
                "topk": 1,
                "mode": "stoploss_replace",
                "stoploss_lookback_days": 15,
                "stoploss_drawdown": 0.08,
                "replacement_pool_size": 4,
            },
        )

        assert not selection_df.empty
        assert captured["kwargs"]["selection_mode"] == "stoploss_replace"
        assert captured["kwargs"]["close_series"].equals(close_series)
        assert captured["kwargs"]["stoploss_lookback_days"] == 15
        assert captured["kwargs"]["stoploss_drawdown"] == pytest.approx(0.08)
        assert captured["kwargs"]["replacement_pool_size"] == 4


class TestBacktestOverlay:
    def test_backtest_from_config_applies_overlay_and_writes_overlay_results(self, monkeypatch, tmp_path):
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        base_returns = pd.Series([0.0, -0.10, -0.10, 0.02], index=dates, dtype=float)
        base_nav = (1.0 + base_returns).cumprod()
        raw_results = tmp_path / "raw_results.csv"
        raw_results.write_text("date,return\n", encoding="utf-8")
        base_result = BacktestResult(
            daily_returns=base_returns,
            portfolio_value=base_nav,
            metadata={"results_file": str(raw_results), "strategy_name": "demo"},
        )

        class FakeEngine:
            def run(self, strategy=None):
                return base_result

        monkeypatch.setattr("modules.backtest.qlib_engine.QlibBacktestEngine", lambda: FakeEngine())
        monkeypatch.setattr("modules.modeling.predictive_signal.selection_path", lambda cfg: tmp_path / "selections.csv")
        monkeypatch.setattr(
            "modules.modeling.predictive_signal.load_bond_overlay_returns",
            lambda: pd.Series([0.0] * len(dates), index=dates, dtype=float),
        )

        cfg = {
            "name": "demo_overlay",
            "selection": {"topk": 3, "universe": "csi300"},
            "position": {"model": "fixed", "params": {"stock_pct": 0.62}},
            "trading": {},
            "output": {"root": str(tmp_path / "out")},
            "overlay": {
                "enabled": True,
                "dd_hard": 0.08,
                "hard_exposure": 0.5,
                "vol_lookback": 20,
                "exposure_max": 1.0,
            },
        }

        result, summary = backtest_from_config(cfg)

        assert summary["overlay_applied"] is True
        assert summary["results_file"].endswith("overlay_results.csv")
        assert result.metadata["base_results_file"].endswith("raw_results.csv")
        assert result.metadata["overlay"]["enabled"] is True
        assert result.daily_returns.iloc[2] == pytest.approx(-0.05)
        assert Path(summary["results_file"]).exists()

    def test_backtest_overlay_can_use_market_trend_release(self, monkeypatch, tmp_path):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        base_returns = pd.Series([0.0, -0.03, 0.01, 0.01, 0.01], index=dates, dtype=float)
        base_nav = (1.0 + base_returns).cumprod()
        raw_results = tmp_path / "raw_results.csv"
        raw_results.write_text("date,return\n", encoding="utf-8")
        base_result = BacktestResult(
            daily_returns=base_returns,
            portfolio_value=base_nav,
            metadata={"results_file": str(raw_results), "strategy_name": "demo"},
        )

        class FakeEngine:
            def run(self, strategy=None):
                return base_result

        monkeypatch.setattr("modules.backtest.qlib_engine.QlibBacktestEngine", lambda: FakeEngine())
        monkeypatch.setattr("modules.modeling.predictive_signal.selection_path", lambda cfg: tmp_path / "selections.csv")
        monkeypatch.setattr(
            "modules.modeling.predictive_signal.load_bond_overlay_returns",
            lambda: pd.Series([0.0] * len(dates), index=dates, dtype=float),
        )
        monkeypatch.setattr(
            "modules.modeling.predictive_signal.load_market_overlay_returns",
            lambda index_code="000300.SH": pd.Series([0.02] * len(dates), index=dates, dtype=float),
        )

        cfg = {
            "name": "demo_overlay_market_release",
            "selection": {"topk": 3, "universe": "csi300"},
            "position": {"model": "fixed", "params": {"stock_pct": 0.62}},
            "trading": {},
            "output": {"root": str(tmp_path / "out_market")},
            "overlay": {
                "enabled": True,
                "dd_hard": 0.02,
                "hard_exposure": 0.25,
                "market_trend_lookback": 2,
                "market_trend_min_return": 0.01,
                "market_trend_exposure_floor": 0.60,
                "market_trend_max_strategy_drawdown": 0.05,
            },
        }

        _, summary = backtest_from_config(cfg)
        frame = pd.read_csv(summary["results_file"], parse_dates=["date"]).set_index("date")

        assert bool(frame.loc[dates[2], "market_risk_on"]) is True
        assert frame.loc[dates[2], "exposure"] == pytest.approx(0.60)

    def test_backtest_from_config_supports_pybroker_engine(self, monkeypatch, tmp_path):
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        base_returns = pd.Series([0.0, 0.01, -0.02, 0.03], index=dates, dtype=float)
        base_nav = (1.0 + base_returns).cumprod()
        raw_results = tmp_path / "pybroker_results.csv"
        raw_results.write_text("date,return\n", encoding="utf-8")
        base_result = BacktestResult(
            daily_returns=base_returns,
            portfolio_value=base_nav,
            metadata={"results_file": str(raw_results), "strategy_name": "demo_pybroker"},
        )

        class FakeEngine:
            def run(self, strategy=None):
                return base_result

        monkeypatch.setattr("modules.backtest.pybroker_engine.PyBrokerBacktestEngine", lambda: FakeEngine())
        monkeypatch.setattr("modules.modeling.predictive_signal.selection_path", lambda cfg: tmp_path / "selections.csv")
        monkeypatch.setattr(
            "modules.modeling.predictive_signal.load_bond_overlay_returns",
            lambda: pd.Series([0.0] * len(dates), index=dates, dtype=float),
        )

        cfg = {
            "name": "demo_pybroker_overlay",
            "selection": {"topk": 3, "universe": "csi300"},
            "position": {"model": "fixed", "params": {"stock_pct": 0.62}},
            "trading": {},
            "output": {"root": str(tmp_path / "out")},
            "overlay": {"enabled": True, "vol_lookback": 20},
        }

        result, summary = backtest_from_config(cfg, engine="pybroker")

        assert summary["engine"] == "pybroker"
        assert summary["overlay_applied"] is True
        assert summary["results_file"].endswith("overlay_results.csv")
        assert result.metadata["base_results_file"].endswith("pybroker_results.csv")


class TestAlpha158Source:
    def test_load_feature_frame_limits_alpha158_to_selection_universe(self, monkeypatch):
        captured = {}
        index = pd.MultiIndex.from_tuples(
            [(pd.Timestamp("2024-01-02"), "SZ000001")],
            names=["datetime", "instrument"],
        )
        feature_frame = pd.DataFrame({"MA20": [1.0]}, index=index)
        rebalance_dates = pd.DatetimeIndex([pd.Timestamp("2024-01-02")])

        monkeypatch.setattr(
            "modules.modeling.predictive_signal.get_universe_instruments",
            lambda start_date, end_date, universe: ["SZ000001", "SZ000002"],
        )

        def fake_load_alpha158_feature_frame(
            start_date,
            end_date,
            rebalance_freq,
            feature_columns,
            alpha158_cfg=None,
            instruments="all",
        ):
            captured["instruments"] = instruments
            return feature_frame, rebalance_dates, list(feature_columns)

        monkeypatch.setattr(
            "modules.modeling.predictive_signal.load_alpha158_feature_frame",
            fake_load_alpha158_feature_frame,
        )

        frame, dates, columns = load_feature_frame(
            start_date="2024-01-01",
            end_date="2024-01-31",
            rebalance_freq="biweek",
            feature_columns=["MA20"],
            data_cfg={"source": "alpha158", "feature_columns": ["MA20"], "alpha158": {}},
            selection_cfg={"universe": "csi300"},
        )

        assert captured["instruments"] == ["SZ000001", "SZ000002"]
        assert frame.equals(feature_frame)
        assert dates.equals(rebalance_dates)
        assert columns == ["MA20"]

    def test_load_feature_frame_hybrid_joins_parquet_and_alpha158_columns(self, monkeypatch):
        parquet_index = pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2024-01-02"), "SZ000001"),
                (pd.Timestamp("2024-01-02"), "SZ000002"),
            ],
            names=["datetime", "instrument"],
        )
        alpha_index = pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2024-01-02"), "SZ000001"),
                (pd.Timestamp("2024-01-02"), "SZ000003"),
            ],
            names=["datetime", "instrument"],
        )
        parquet_frame = pd.DataFrame({"book_to_market": [1.1, 2.2]}, index=parquet_index)
        alpha_frame = pd.DataFrame({"ROC20": [0.3, 0.4]}, index=alpha_index)
        rebalance_dates = pd.DatetimeIndex([pd.Timestamp("2024-01-02")])

        monkeypatch.setattr(
            "modules.modeling.predictive_signal.load_parquet_feature_frame",
            lambda start_date, end_date, rebalance_freq, feature_columns=None: (
                parquet_frame,
                rebalance_dates,
                ["book_to_market"],
            ),
        )
        monkeypatch.setattr(
            "modules.modeling.predictive_signal.load_alpha158_feature_frame",
            lambda start_date, end_date, rebalance_freq, feature_columns, alpha158_cfg=None, instruments="all": (
                alpha_frame,
                rebalance_dates,
                ["ROC20"],
            ),
        )

        frame, dates, columns = load_feature_frame(
            start_date="2024-01-01",
            end_date="2024-01-31",
            rebalance_freq="biweek",
            data_cfg={
                "source": "hybrid",
                "parquet_feature_columns": ["book_to_market"],
                "alpha158_feature_columns": ["ROC20"],
                "alpha158": {},
            },
            selection_cfg={"universe": "csi300"},
        )

        assert list(frame.columns) == ["book_to_market", "ROC20"]
        assert list(frame.index) == [(pd.Timestamp("2024-01-02"), "SZ000001")]
        assert dates.equals(rebalance_dates)
        assert columns == ["book_to_market", "ROC20"]

    def test_train_from_config_uses_alpha158_source_loader(self, monkeypatch, tmp_path):
        from modules.modeling.predictive_signal import train_from_config

        dates = pd.to_datetime(["2024-01-02", "2024-01-09"])
        index = pd.MultiIndex.from_tuples(
            [
                (dates[0], "SZ000001"),
                (dates[0], "SZ000002"),
                (dates[1], "SZ000001"),
                (dates[1], "SZ000002"),
            ],
            names=["datetime", "instrument"],
        )
        feature_frame = pd.DataFrame(
            {
                "MA20": [0.9, 1.1, 1.0, 1.2],
                "STD20": [0.1, 0.2, 0.1, 0.2],
            },
            index=index,
        )
        close_series = pd.Series([10.0, 11.0, 10.5, 11.5], index=index, name="close")

        monkeypatch.setattr(
            "modules.modeling.predictive_signal.load_feature_frame",
            lambda start_date, end_date, rebalance_freq, feature_columns=None, data_cfg=None, selection_cfg=None: (
                feature_frame,
                dates,
                ["MA20", "STD20"],
            ),
        )
        monkeypatch.setattr(
            "modules.modeling.predictive_signal.load_close_series",
            lambda instruments, start_date, end_date, horizon_days: close_series,
        )

        cfg = {
            "name": "alpha158_demo",
            "data": {
                "source": "alpha158",
                "feature_columns": ["MA20", "STD20"],
                "alpha158": {"rolling": {"windows": [20], "include": ["MA", "STD"]}},
            },
            "training": {
                "train_start": "2024-01-02",
                "train_end": "2024-01-02",
                "valid_start": "2024-01-09",
                "valid_end": "2024-01-09",
            },
            "label": {"horizon_days": 1},
            "selection": {"freq": "biweek"},
            "model": {"preferred_backend": "lightgbm", "params": {"n_estimators": 8, "random_state": 42}},
            "output": {"root": str(tmp_path / "out")},
        }

        summary = train_from_config(cfg)

        assert summary["feature_count"] == 2
        assert summary["feature_columns"] == ["MA20", "STD20"]
