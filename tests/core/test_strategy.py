"""
策略模块测试
"""

import json
import pytest
import yaml
import sys
import pandas as pd
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.strategy import Strategy, _FixedPositionController, STRATEGIES_DIR
from core.factors import FactorRegistry


# ── Fixtures ──────────────────────────────────────────────

@pytest.fixture
def tmp_strategies_dir(tmp_path):
    """临时策略目录"""
    d = tmp_path / "strategies"
    d.mkdir()
    return d


@pytest.fixture
def sample_strategy_yaml(tmp_strategies_dir):
    """创建一个样本策略 YAML"""
    cfg = {
        "name": "test_strategy",
        "description": "测试策略",
        "factors": {
            "alpha": [
                {"name": "roa", "expression": "roa_fina", "source": "parquet", "ir": 0.3},
                {"name": "book_to_price", "expression": "book_to_market", "source": "parquet"},
            ],
            "risk": [
                {
                    "name": "vol_std_20d",
                    "expression": "-1 * Std(($close - Ref($close, 1)) / Ref($close, 1), 20)",
                    "source": "qlib",
                },
            ],
        },
        "weights": {"alpha": 0.70, "risk": 0.30},
        "selection": {"topk": 15, "neutralize_industry": False},
        "position": {"model": "fixed", "stock_pct": 0.8},
        "rebalance": {"freq": "biweek"},
    }
    yaml_path = tmp_strategies_dir / "test_strategy.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f)
    return yaml_path


# ── Strategy.load 测试 ────────────────────────────────────

class TestStrategyLoad:

    def test_load_from_yaml(self, sample_strategy_yaml, tmp_strategies_dir):
        with patch("core.strategy.STRATEGIES_DIR", tmp_strategies_dir), \
             patch("core.strategy._load_strategy_defaults", return_value={}):
            s = Strategy.load("test_strategy")

        assert s.name == "test_strategy"
        assert s.description == "测试策略"
        assert s.topk == 15
        assert s.neutralize_industry is False
        assert s.universe == "all"
        assert s.exclude_st is False
        assert s.exclude_new_days == 0
        assert s.position_model == "fixed"
        assert s.position_params == {"stock_pct": 0.8}
        assert s.rebalance_freq == "biweek"
        assert s.threshold == 0.0
        assert s.churn_limit == 0
        assert s.margin_stable is False
        assert s.weights == {"alpha": 0.70, "risk": 0.30}

    def test_load_builds_registry(self, sample_strategy_yaml, tmp_strategies_dir):
        with patch("core.strategy.STRATEGIES_DIR", tmp_strategies_dir):
            s = Strategy.load("test_strategy")

        reg = s.registry
        assert isinstance(reg, FactorRegistry)
        alpha = reg.get_by_category("alpha")
        risk = reg.get_by_category("risk")
        assert len(alpha) == 2
        assert len(risk) == 1
        assert alpha[0].name == "roa"
        assert alpha[0].source == "parquet"
        assert risk[0].source == "qlib"

    def test_load_nonexistent_raises(self, tmp_strategies_dir):
        with patch("core.strategy.STRATEGIES_DIR", tmp_strategies_dir):
            with pytest.raises(FileNotFoundError):
                Strategy.load("nonexistent")

    def test_load_nested_strategy_by_full_path(self, tmp_strategies_dir):
        nested_dir = tmp_strategies_dir / "experimental" / "safety"
        nested_dir.mkdir(parents=True)
        cfg = {"name": "demo", "factors": {}}
        with open(nested_dir / "demo.yaml", "w") as f:
            yaml.dump(cfg, f)

        with patch("core.strategy.STRATEGIES_DIR", tmp_strategies_dir), \
             patch("core.strategy._load_strategy_defaults", return_value={}):
            s = Strategy.load("experimental/safety/demo")

        assert s.name == "experimental/safety/demo"
        assert s.display_name == "demo"

    def test_load_nested_strategy_by_unique_basename(self, tmp_strategies_dir):
        nested_dir = tmp_strategies_dir / "experimental" / "regime"
        nested_dir.mkdir(parents=True)
        cfg = {"name": "unique_demo", "factors": {}}
        with open(nested_dir / "unique_demo.yaml", "w") as f:
            yaml.dump(cfg, f)

        with patch("core.strategy.STRATEGIES_DIR", tmp_strategies_dir), \
             patch("core.strategy._load_strategy_defaults", return_value={}):
            s = Strategy.load("unique_demo")

        assert s.name == "experimental/regime/unique_demo"

    def test_load_ambiguous_basename_raises(self, tmp_strategies_dir):
        left = tmp_strategies_dir / "fixed" / "balanced"
        right = tmp_strategies_dir / "experimental" / "safety"
        left.mkdir(parents=True)
        right.mkdir(parents=True)
        for directory in (left, right):
            with open(directory / "shared.yaml", "w") as f:
                yaml.dump({"name": "shared", "factors": {}}, f)

        with patch("core.strategy.STRATEGIES_DIR", tmp_strategies_dir), \
             patch("core.strategy._load_strategy_defaults", return_value={}):
            with pytest.raises(ValueError, match="不唯一"):
                Strategy.load("shared")

    def test_load_defaults(self, tmp_strategies_dir):
        """缺省字段使用默认值"""
        cfg = {"name": "minimal", "factors": {}}
        yaml_path = tmp_strategies_dir / "minimal.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(cfg, f)

        with patch("core.strategy.STRATEGIES_DIR", tmp_strategies_dir), \
             patch("core.strategy._load_strategy_defaults", return_value={}):
            s = Strategy.load("minimal")

        assert s.topk == 20
        assert s.neutralize_industry is True
        assert s.universe == "all"
        assert s.exclude_st is False
        assert s.exclude_new_days == 0
        assert s.position_model == "trend"
        assert s.rebalance_freq == "month"
        assert s.threshold == 0.0
        assert s.churn_limit == 0
        assert s.margin_stable is False

    def test_load_inherits_global_defaults(self, tmp_strategies_dir):
        """策略 YAML 可继承全局默认配置"""
        cfg = {"name": "inherit_test", "factors": {}}
        yaml_path = tmp_strategies_dir / "inherit_test.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(cfg, f)

        defaults = {
            "weights": {"alpha": 0.20, "risk": 0.55, "enhance": 0.25},
            "selection": {
                "topk": 15,
                "universe": "csi300",
                "min_market_cap": 50,
                "exclude_st": True,
                "exclude_new_days": 60,
            },
            "stability": {"sticky": 5, "threshold": 0.3, "churn_limit": 5, "margin_stable": True},
            "rebalance": {"freq": "week"},
        }

        with patch("core.strategy.STRATEGIES_DIR", tmp_strategies_dir), \
             patch("core.strategy._load_strategy_defaults", return_value=defaults):
            s = Strategy.load("inherit_test")

        assert s.topk == 15
        assert s.universe == "csi300"
        assert s.min_market_cap == 50
        assert s.exclude_st is True
        assert s.exclude_new_days == 60
        assert s.sticky == 5
        assert s.rebalance_freq == "week"
        assert s.threshold == 0.3
        assert s.churn_limit == 5
        assert s.margin_stable is True
        assert s.weights == {"alpha": 0.20, "risk": 0.55, "enhance": 0.25}

    def test_load_strategy_stability_override(self, tmp_strategies_dir):
        cfg = {
            "name": "stability_override",
            "factors": {},
            "selection": {"sticky": 2, "buffer": 7},
            "stability": {"threshold": 0.45, "churn_limit": 3, "margin_stable": False},
        }
        yaml_path = tmp_strategies_dir / "stability_override.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(cfg, f)

        defaults = {
            "selection": {"sticky": 5, "buffer": 10},
            "stability": {"threshold": 0.3, "churn_limit": 5, "margin_stable": True},
        }

        with patch("core.strategy.STRATEGIES_DIR", tmp_strategies_dir), \
             patch("core.strategy._load_strategy_defaults", return_value=defaults):
            s = Strategy.load("stability_override")

        assert s.sticky == 2
        assert s.buffer == 7
        assert s.threshold == 0.45
        assert s.churn_limit == 3
        assert s.margin_stable is False

    def test_generate_selections_forwards_stability_options(self, tmp_strategies_dir):
        cfg = {
            "name": "forward_stability",
            "factors": {},
            "selection": {"sticky": 2, "buffer": 7},
            "stability": {"threshold": 0.45, "churn_limit": 3, "margin_stable": True},
        }
        yaml_path = tmp_strategies_dir / "forward_stability.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(cfg, f)

        with patch("core.strategy.STRATEGIES_DIR", tmp_strategies_dir), \
             patch("core.strategy._load_strategy_defaults", return_value={}), \
             patch("core.selection.generate_selections", return_value=pd.DataFrame()) as mock_generate:
            s = Strategy.load("forward_stability")
            s.generate_selections(force=True)

        kwargs = mock_generate.call_args.kwargs
        assert kwargs["sticky"] == 2
        assert kwargs["buffer"] == 7
        assert kwargs["threshold"] == 0.45
        assert kwargs["churn_limit"] == 3
        assert kwargs["margin_stable"] is True

    def test_load_strategy_selection_stability_fallback(self, tmp_strategies_dir):
        cfg = {
            "name": "selection_stability_fallback",
            "factors": {},
            "selection": {
                "sticky": 2,
                "buffer": 12,
                "threshold": 0.25,
                "churn_limit": 2,
                "margin_stable": True,
            },
        }
        yaml_path = tmp_strategies_dir / "selection_stability_fallback.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(cfg, f)

        with patch("core.strategy.STRATEGIES_DIR", tmp_strategies_dir), \
             patch("core.strategy._load_strategy_defaults", return_value={}):
            s = Strategy.load("selection_stability_fallback")

        assert s.sticky == 2
        assert s.buffer == 12
        assert s.threshold == 0.25
        assert s.churn_limit == 2
        assert s.margin_stable is True

    def test_selection_stability_fallback_overrides_global_defaults(self, tmp_strategies_dir):
        cfg = {
            "name": "selection_override_defaults",
            "factors": {},
            "selection": {
                "threshold": 0.25,
                "churn_limit": 2,
                "margin_stable": True,
            },
        }
        yaml_path = tmp_strategies_dir / "selection_override_defaults.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(cfg, f)

        defaults = {
            "stability": {"threshold": 0.4, "churn_limit": 5, "margin_stable": False},
        }

        with patch("core.strategy.STRATEGIES_DIR", tmp_strategies_dir), \
             patch("core.strategy._load_strategy_defaults", return_value=defaults):
            s = Strategy.load("selection_override_defaults")

        assert s.threshold == 0.25
        assert s.churn_limit == 2
        assert s.margin_stable is True

    def test_load_and_forward_hard_filter_quantiles(self, tmp_strategies_dir):
        cfg = {
            "name": "quantile_filter_test",
            "factors": {},
            "selection": {"hard_filter_quantiles": {"roa_fina": 0.4, "ocf_to_ev": 0.3}},
        }
        yaml_path = tmp_strategies_dir / "quantile_filter_test.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(cfg, f)

        with patch("core.strategy.STRATEGIES_DIR", tmp_strategies_dir), \
             patch("core.strategy._load_strategy_defaults", return_value={}), \
             patch("core.selection.generate_selections", return_value=pd.DataFrame()) as mock_generate:
            s = Strategy.load("quantile_filter_test")
            s.generate_selections(force=True)

        assert s.hard_filter_quantiles == {"roa_fina": 0.4, "ocf_to_ev": 0.3}
        kwargs = mock_generate.call_args.kwargs
        assert kwargs["hard_filter_quantiles"] == {"roa_fina": 0.4, "ocf_to_ev": 0.3}

    def test_load_and_forward_low_turnover_selection_options(self, tmp_strategies_dir):
        cfg = {
            "name": "low_turnover_gate",
            "factors": {},
            "selection": {
                "score_smoothing_days": 5,
                "entry_rank": 10,
                "exit_rank": 30,
                "entry_persist_days": 3,
                "exit_persist_days": 4,
                "min_hold_days": 15,
            },
        }
        yaml_path = tmp_strategies_dir / "low_turnover_gate.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(cfg, f)

        with patch("core.strategy.STRATEGIES_DIR", tmp_strategies_dir), \
             patch("core.strategy._load_strategy_defaults", return_value={}), \
             patch("core.selection.generate_selections", return_value=pd.DataFrame()) as mock_generate:
            s = Strategy.load("low_turnover_gate")
            s.generate_selections(force=True)

        assert s.score_smoothing_days == 5
        assert s.entry_rank == 10
        assert s.exit_rank == 30
        assert s.entry_persist_days == 3
        assert s.exit_persist_days == 4
        assert s.min_hold_days == 15

        kwargs = mock_generate.call_args.kwargs
        assert kwargs["score_smoothing_days"] == 5
        assert kwargs["entry_rank"] == 10
        assert kwargs["exit_rank"] == 30
        assert kwargs["entry_persist_days"] == 3
        assert kwargs["exit_persist_days"] == 4
        assert kwargs["min_hold_days"] == 15

    def test_load_and_forward_stoploss_replace_selection_options(self, tmp_strategies_dir):
        cfg = {
            "name": "stoploss_replace_plan",
            "factors": {},
            "selection": {
                "mode": "stoploss_replace",
                "topk": 15,
                "stoploss_lookback_days": 20,
                "stoploss_drawdown": 0.10,
                "replacement_pool_size": 30,
            },
        }
        yaml_path = tmp_strategies_dir / "stoploss_replace_plan.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(cfg, f)

        with patch("core.strategy.STRATEGIES_DIR", tmp_strategies_dir), \
             patch("core.strategy._load_strategy_defaults", return_value={}), \
             patch("core.selection.generate_selections", return_value=pd.DataFrame()) as mock_generate:
            s = Strategy.load("stoploss_replace_plan")
            s.generate_selections(force=True)

        assert s.selection_mode == "stoploss_replace"
        assert s.stoploss_lookback_days == 20
        assert s.stoploss_drawdown == pytest.approx(0.10)
        assert s.replacement_pool_size == 30

        kwargs = mock_generate.call_args.kwargs
        assert kwargs["selection_mode"] == "stoploss_replace"
        assert kwargs["stoploss_lookback_days"] == 20
        assert kwargs["stoploss_drawdown"] == pytest.approx(0.10)
        assert kwargs["replacement_pool_size"] == 30

    def test_generate_selections_writes_cache_metadata(self, tmp_strategies_dir, tmp_path):
        cfg = {
            "name": "cache_meta_plan",
            "factors": {},
            "selection": {"mode": "stoploss_replace"},
        }
        yaml_path = tmp_strategies_dir / "cache_meta_plan.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(cfg, f)

        selections_dir = tmp_path / "selections"
        with patch("core.strategy.STRATEGIES_DIR", tmp_strategies_dir), \
             patch("core.strategy.SELECTIONS_DIR", selections_dir), \
             patch("core.strategy._load_strategy_defaults", return_value={}), \
             patch("core.selection.generate_selections", return_value=pd.DataFrame()):
            s = Strategy.load("cache_meta_plan")
            s.generate_selections(force=True)
            meta = json.loads(s.selections_meta_path().read_text(encoding="utf-8"))

        assert meta["strategy_name"] == "cache_meta_plan"
        assert meta["selection_mode"] == "stoploss_replace"
        assert "cache_version" in meta

    def test_selections_are_stale_when_metadata_missing(self, tmp_strategies_dir, tmp_path):
        cfg = {"name": "stale_meta_missing", "factors": {}}
        yaml_path = tmp_strategies_dir / "stale_meta_missing.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(cfg, f)

        selections_dir = tmp_path / "selections"
        with patch("core.strategy.STRATEGIES_DIR", tmp_strategies_dir), \
             patch("core.strategy.SELECTIONS_DIR", selections_dir), \
             patch("core.strategy._load_strategy_defaults", return_value={}):
            s = Strategy.load("stale_meta_missing")
            s.selections_path().parent.mkdir(parents=True, exist_ok=True)
            s.selections_path().write_text("date,rank,symbol,score\n", encoding="utf-8")
            assert s.selections_are_stale() is True

    def test_selections_are_stale_when_metadata_mismatches(self, tmp_strategies_dir, tmp_path):
        cfg = {
            "name": "stale_meta_mismatch",
            "factors": {},
            "selection": {"mode": "stoploss_replace"},
        }
        yaml_path = tmp_strategies_dir / "stale_meta_mismatch.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(cfg, f)

        selections_dir = tmp_path / "selections"
        with patch("core.strategy.STRATEGIES_DIR", tmp_strategies_dir), \
             patch("core.strategy.SELECTIONS_DIR", selections_dir), \
             patch("core.strategy._load_strategy_defaults", return_value={}):
            s = Strategy.load("stale_meta_mismatch")
            s.selections_path().parent.mkdir(parents=True, exist_ok=True)
            s.selections_path().write_text("date,rank,symbol,score\n", encoding="utf-8")
            s.selections_meta_path().write_text(
                json.dumps(
                    {
                        "cache_version": 1,
                        "strategy_name": "stale_meta_mismatch",
                        "selection_mode": "factor_topk",
                    }
                ),
                encoding="utf-8",
            )
            assert s.selections_are_stale() is True

    def test_load_composition_strategy(self, tmp_strategies_dir):
        for name in ("base_a", "base_b"):
            with open(tmp_strategies_dir / f"{name}.yaml", "w") as f:
                yaml.dump({"name": name, "factors": {}}, f)

        cfg = {
            "name": "combo",
            "description": "组合策略",
            "composition": {
                "components": [
                    {"strategy": "base_a", "weight": 0.5},
                    {"strategy": "base_b", "weight": 0.2},
                ],
                "cash_weight": 0.3,
            },
        }
        with open(tmp_strategies_dir / "combo.yaml", "w") as f:
            yaml.dump(cfg, f)

        with patch("core.strategy.STRATEGIES_DIR", tmp_strategies_dir), \
             patch("core.strategy._load_strategy_defaults", return_value={}):
            s = Strategy.load("combo")
            assert s.is_composite is True
            assert s.component_weights() == {"base_a": 0.5, "base_b": 0.2}
            assert s.cash_weight == 0.3
            assert s.effective_universe() == "all"

    def test_validate_composition_cash_mismatch_raises(self, tmp_strategies_dir):
        for name in ("base_a", "base_b"):
            with open(tmp_strategies_dir / f"{name}.yaml", "w") as f:
                yaml.dump({"name": name, "factors": {}}, f)

        cfg = {
            "name": "bad_combo",
            "composition": {
                "components": [
                    {"strategy": "base_a", "weight": 0.5},
                    {"strategy": "base_b", "weight": 0.2},
                ],
                "cash_weight": 0.1,
            },
        }
        with open(tmp_strategies_dir / "bad_combo.yaml", "w") as f:
            yaml.dump(cfg, f)

        with patch("core.strategy.STRATEGIES_DIR", tmp_strategies_dir), \
             patch("core.strategy._load_strategy_defaults", return_value={}):
            with pytest.raises(ValueError, match="cash_weight"):
                Strategy.load("bad_combo")

    def test_composition_validate_data_requirements_recurses(self, tmp_strategies_dir):
        for name, universe in (("base_a", "all"), ("base_b", "csi300")):
            with open(tmp_strategies_dir / f"{name}.yaml", "w") as f:
                yaml.dump({"name": name, "factors": {}, "selection": {"universe": universe}}, f)

        cfg = {
            "name": "combo",
            "composition": {
                "components": [
                    {"strategy": "base_a", "weight": 0.5},
                    {"strategy": "base_b", "weight": 0.3},
                ],
                "cash_weight": 0.2,
            },
        }
        with open(tmp_strategies_dir / "combo.yaml", "w") as f:
            yaml.dump(cfg, f)

        with patch("core.strategy.STRATEGIES_DIR", tmp_strategies_dir), \
             patch("core.strategy._load_strategy_defaults", return_value={}), \
             patch("modules.data.precheck.ensure_strategy_data_ready", return_value=True) as mock_ready:
            s = Strategy.load("combo")
            result = s.validate_data_requirements()
            assert s.effective_universe() == "mixed"

        assert mock_ready.call_count == 2
        assert len(result) == 2


# ── Strategy.list_available 测试 ──────────────────────────

class TestListAvailable:

    def test_list_available(self, tmp_strategies_dir, sample_strategy_yaml):
        # 添加第二个策略
        with open(tmp_strategies_dir / "another.yaml", "w") as f:
            yaml.dump({"name": "another"}, f)

        with patch("core.strategy.STRATEGIES_DIR", tmp_strategies_dir):
            names = Strategy.list_available()

        assert "test_strategy" in names
        assert "another" in names
        assert names == sorted(names)

    def test_list_empty(self, tmp_strategies_dir):
        with patch("core.strategy.STRATEGIES_DIR", tmp_strategies_dir):
            names = Strategy.list_available()
        assert names == []

    def test_list_nonexistent_dir(self, tmp_path):
        with patch("core.strategy.STRATEGIES_DIR", tmp_path / "nope"):
            names = Strategy.list_available()
        assert names == []

    def test_list_available_recurses_and_groups(self, tmp_strategies_dir):
        fixed_dir = tmp_strategies_dir / "fixed" / "balanced"
        exp_dir = tmp_strategies_dir / "experimental" / "safety"
        fixed_dir.mkdir(parents=True)
        exp_dir.mkdir(parents=True)

        with open(tmp_strategies_dir / "root.yaml", "w") as f:
            yaml.dump({"name": "root", "factors": {}}, f)
        with open(fixed_dir / "stable.yaml", "w") as f:
            yaml.dump({"name": "stable", "factors": {}}, f)
        with open(exp_dir / "trial.yaml", "w") as f:
            yaml.dump({"name": "trial", "factors": {}}, f)

        with patch("core.strategy.STRATEGIES_DIR", tmp_strategies_dir):
            names = Strategy.list_available()
            grouped = Strategy.list_grouped()

        assert "root" in names
        assert "fixed/balanced/stable" in names
        assert "experimental/safety/trial" in names
        assert grouped["winners"] == ["root"]
        assert grouped["fixed"] == ["fixed/balanced/stable"]
        assert grouped["experimental"] == ["experimental/safety/trial"]


# ── build_registry 测试 ───────────────────────────────────

class TestBuildRegistry:

    def test_negate_flag(self, tmp_strategies_dir):
        cfg = {
            "name": "negate_test",
            "factors": {
                "risk": [
                    {"name": "turnover", "expression": "turnover_rate_f",
                     "source": "parquet", "negate": True},
                ],
            },
        }
        with open(tmp_strategies_dir / "negate_test.yaml", "w") as f:
            yaml.dump(cfg, f)

        with patch("core.strategy.STRATEGIES_DIR", tmp_strategies_dir):
            s = Strategy.load("negate_test")

        risk = s.registry.get_by_category("risk")
        assert risk[0].negate is True


# ── build_position_controller 测试 ────────────────────────

class TestBuildPositionController:

    def test_trend_model(self, sample_strategy_yaml, tmp_strategies_dir):
        cfg = {"name": "trend_test", "factors": {}, "position": {"model": "trend"}}
        with open(tmp_strategies_dir / "trend_test.yaml", "w") as f:
            yaml.dump(cfg, f)

        with patch("core.strategy.STRATEGIES_DIR", tmp_strategies_dir):
            s = Strategy.load("trend_test")
        ctrl = s.build_position_controller()

        from core.position import MarketPositionController
        assert isinstance(ctrl, MarketPositionController)

    def test_fixed_model(self, sample_strategy_yaml, tmp_strategies_dir):
        with patch("core.strategy.STRATEGIES_DIR", tmp_strategies_dir):
            s = Strategy.load("test_strategy")  # model=fixed, stock_pct=0.8
        ctrl = s.build_position_controller()

        assert isinstance(ctrl, _FixedPositionController)
        assert ctrl.stock_pct == 0.8

        alloc = ctrl.get_allocation("2025-01-01")
        assert alloc.stock_pct == 0.8
        assert alloc.regime == "fixed"

    def test_full_model(self, tmp_strategies_dir):
        cfg = {"name": "full_test", "factors": {}, "position": {"model": "full"}}
        with open(tmp_strategies_dir / "full_test.yaml", "w") as f:
            yaml.dump(cfg, f)

        with patch("core.strategy.STRATEGIES_DIR", tmp_strategies_dir):
            s = Strategy.load("full_test")
        ctrl = s.build_position_controller()
        assert ctrl is None

    def test_unknown_model_raises(self, tmp_strategies_dir):
        cfg = {"name": "bad", "factors": {}, "position": {"model": "unknown"}}
        with open(tmp_strategies_dir / "bad.yaml", "w") as f:
            yaml.dump(cfg, f)

        with patch("core.strategy.STRATEGIES_DIR", tmp_strategies_dir):
            with pytest.raises(ValueError, match="position.model"):
                Strategy.load("bad")


# ── get_rebalance_dates 测试 ──────────────────────────────

class TestGetRebalanceDates:

    @pytest.fixture
    def trade_dates(self):
        """模拟交易日历：2024年1-3月每工作日"""
        return pd.bdate_range("2024-01-01", "2024-03-31")

    def test_month_freq(self, trade_dates, tmp_strategies_dir):
        cfg = {"name": "m", "factors": {}, "rebalance": {"freq": "month"}}
        with open(tmp_strategies_dir / "m.yaml", "w") as f:
            yaml.dump(cfg, f)
        with patch("core.strategy.STRATEGIES_DIR", tmp_strategies_dir):
            s = Strategy.load("m")

        dates = s.get_rebalance_dates(trade_dates)
        # 应该有3个月末日
        assert len(dates) == 3

    def test_week_freq(self, trade_dates, tmp_strategies_dir):
        cfg = {"name": "w", "factors": {}, "rebalance": {"freq": "week"}}
        with open(tmp_strategies_dir / "w.yaml", "w") as f:
            yaml.dump(cfg, f)
        with patch("core.strategy.STRATEGIES_DIR", tmp_strategies_dir):
            s = Strategy.load("w")

        dates = s.get_rebalance_dates(trade_dates)
        # 大约13周
        assert len(dates) >= 12

    def test_biweek_freq(self, trade_dates, tmp_strategies_dir):
        cfg = {"name": "bw", "factors": {}, "rebalance": {"freq": "biweek"}}
        with open(tmp_strategies_dir / "bw.yaml", "w") as f:
            yaml.dump(cfg, f)
        with patch("core.strategy.STRATEGIES_DIR", tmp_strategies_dir):
            s = Strategy.load("bw")

        dates = s.get_rebalance_dates(trade_dates)
        # biweek should be roughly half of week count
        assert len(dates) >= 5

    def test_unknown_freq_raises(self, trade_dates, tmp_strategies_dir):
        cfg = {"name": "bad", "factors": {}, "rebalance": {"freq": "daily"}}
        with open(tmp_strategies_dir / "bad.yaml", "w") as f:
            yaml.dump(cfg, f)
        with patch("core.strategy.STRATEGIES_DIR", tmp_strategies_dir):
            with pytest.raises(ValueError, match="rebalance.freq"):
                Strategy.load("bad")


# ── selections_path 测试 ──────────────────────────────────

class TestSelectionsPath:

    def test_path_contains_name(self, sample_strategy_yaml, tmp_strategies_dir):
        with patch("core.strategy.STRATEGIES_DIR", tmp_strategies_dir):
            s = Strategy.load("test_strategy")

        path = s.selections_path()
        assert "test_strategy.csv" in str(path)
        assert "selections" in str(path)

    def test_nested_strategy_path_is_layered(self, tmp_path, tmp_strategies_dir):
        nested_dir = tmp_strategies_dir / "experimental" / "validity"
        nested_dir.mkdir(parents=True)
        with open(nested_dir / "gated.yaml", "w") as f:
            yaml.dump({"name": "gated", "factors": {}}, f)

        selections_dir = tmp_path / "selections"
        with patch("core.strategy.STRATEGIES_DIR", tmp_strategies_dir), \
             patch("core.strategy.SELECTIONS_DIR", selections_dir), \
             patch("core.strategy._load_strategy_defaults", return_value={}):
            s = Strategy.load("experimental/validity/gated")
            assert s.selections_path() == selections_dir / "experimental" / "validity" / "gated.csv"

    def test_stale_selections_triggers_regeneration(self, tmp_path, tmp_strategies_dir):
        cfg = {"name": "stale_test", "factors": {}}
        with open(tmp_strategies_dir / "stale_test.yaml", "w") as f:
            yaml.dump(cfg, f)

        selections_dir = tmp_path / "selections"
        selections_dir.mkdir()
        csv_path = selections_dir / "stale_test.csv"
        csv_path.write_text("date,rank,symbol,score\n", encoding="utf-8")

        dep = tmp_path / "dep.py"
        dep.write_text("# newer\n", encoding="utf-8")

        with patch("core.strategy.STRATEGIES_DIR", tmp_strategies_dir), \
             patch("core.strategy.SELECTIONS_DIR", selections_dir), \
             patch("core.strategy._load_strategy_defaults", return_value={}), \
             patch.object(Strategy, "selection_dependency_paths", return_value=[dep]), \
             patch.object(Strategy, "generate_selections") as mock_generate, \
             patch("core.selection.load_selections", return_value=({}, set())) as mock_load:
            s = Strategy.load("stale_test")
            dep.touch()
            s.load_selections()

        mock_generate.assert_called_once_with(force=True)
        mock_load.assert_called_once_with(csv_path=csv_path)


# ── _FixedPositionController 测试 ─────────────────────────

class TestFixedPositionController:

    def test_default_pct(self):
        ctrl = _FixedPositionController()
        assert ctrl.stock_pct == 0.8

    def test_custom_pct(self):
        ctrl = _FixedPositionController(0.6)
        alloc = ctrl.get_allocation("2025-01-01")
        assert alloc.stock_pct == 0.6
        assert alloc.cash_pct == 0.4

    def test_load_market_data_noop(self):
        ctrl = _FixedPositionController()
        ctrl.load_market_data()  # 不应抛异常

    def test_bond_return(self):
        ctrl = _FixedPositionController()
        assert abs(ctrl.get_bond_daily_return() - 0.03 / 252) < 1e-10


# ── 实际策略文件加载测试 ──────────────────────────────────

class TestRealStrategyFiles:
    """测试 config/strategies/ 下的实际策略文件"""

    def test_default_strategy_exists(self):
        names = Strategy.list_available()
        assert "fixed/reference/default" in names

    def test_load_default_strategy(self):
        s = Strategy.load("default")
        assert s.name == "fixed/reference/default"
        assert len(s.registry.all()) >= 1
        assert len(s.registry.get_by_category("enhance")) >= 1
        assert s.topk > 0

    def test_load_all_strategies(self):
        for name in Strategy.list_available():
            s = Strategy.load(name)
            assert s.name == name
            assert isinstance(s.registry, FactorRegistry)


# ── YAML 验证测试 ─────────────────────────────────────────

class TestStrategyValidation:
    """测试策略 YAML 配置验证"""

    def test_invalid_position_model(self, tmp_strategies_dir):
        cfg = {
            "name": "bad_model",
            "factors": {},
            "position": {"model": "trends"},  # 错误：多了一个 s
        }
        with open(tmp_strategies_dir / "bad_model.yaml", "w") as f:
            yaml.dump(cfg, f)

        with patch("core.strategy.STRATEGIES_DIR", tmp_strategies_dir):
            with pytest.raises(ValueError, match="position.model='trends' 无效"):
                Strategy.load("bad_model")

    def test_invalid_rebalance_freq(self, tmp_strategies_dir):
        cfg = {
            "name": "bad_freq",
            "factors": {},
            "rebalance": {"freq": "daily"},  # 错误
        }
        with open(tmp_strategies_dir / "bad_freq.yaml", "w") as f:
            yaml.dump(cfg, f)

        with patch("core.strategy.STRATEGIES_DIR", tmp_strategies_dir):
            with pytest.raises(ValueError, match="rebalance.freq='daily' 无效"):
                Strategy.load("bad_freq")

    def test_invalid_factor_source(self, tmp_strategies_dir):
        cfg = {
            "name": "bad_source",
            "factors": {
                "alpha": [
                    {"name": "test", "expression": "test_field", "source": "invalid"},
                ],
            },
        }
        with open(tmp_strategies_dir / "bad_source.yaml", "w") as f:
            yaml.dump(cfg, f)

        with patch("core.strategy.STRATEGIES_DIR", tmp_strategies_dir):
            with pytest.raises(ValueError, match="source='invalid' 无效"):
                Strategy.load("bad_source")

    def test_invalid_hard_filter_quantile_raises(self, tmp_strategies_dir):
        cfg = {
            "name": "bad_quantile",
            "factors": {},
            "selection": {"hard_filter_quantiles": {"roa_fina": 1.2}},
        }
        with open(tmp_strategies_dir / "bad_quantile.yaml", "w") as f:
            yaml.dump(cfg, f)

        with patch("core.strategy.STRATEGIES_DIR", tmp_strategies_dir):
            with pytest.raises(ValueError, match="hard_filter_quantiles.roa_fina=1.2 超出范围"):
                Strategy.load("bad_quantile")

    def test_invalid_selection_mode_raises(self, tmp_strategies_dir):
        cfg = {
            "name": "bad_mode",
            "factors": {},
            "selection": {"mode": "unknown_mode"},
        }
        with open(tmp_strategies_dir / "bad_mode.yaml", "w") as f:
            yaml.dump(cfg, f)

        with patch("core.strategy.STRATEGIES_DIR", tmp_strategies_dir):
            with pytest.raises(ValueError, match="selection.mode='unknown_mode' 无效"):
                Strategy.load("bad_mode")

    def test_invalid_stoploss_drawdown_raises(self, tmp_strategies_dir):
        cfg = {
            "name": "bad_stoploss_drawdown",
            "factors": {},
            "selection": {
                "mode": "stoploss_replace",
                "stoploss_drawdown": 1.2,
            },
        }
        with open(tmp_strategies_dir / "bad_stoploss_drawdown.yaml", "w") as f:
            yaml.dump(cfg, f)

        with patch("core.strategy.STRATEGIES_DIR", tmp_strategies_dir):
            with pytest.raises(ValueError, match="selection.stoploss_drawdown 必须在 \\(0, 1\\) 之间"):
                Strategy.load("bad_stoploss_drawdown")

    def test_missing_factor_name(self, tmp_strategies_dir):
        cfg = {
            "name": "missing_name",
            "factors": {
                "alpha": [
                    {"expression": "roa_fina"},  # 缺少 name
                ],
            },
        }
        with open(tmp_strategies_dir / "missing_name.yaml", "w") as f:
            yaml.dump(cfg, f)

        with patch("core.strategy.STRATEGIES_DIR", tmp_strategies_dir):
            with pytest.raises(ValueError, match="缺少 'name' 字段"):
                Strategy.load("missing_name")

    def test_missing_factor_expression(self, tmp_strategies_dir):
        cfg = {
            "name": "missing_expr",
            "factors": {
                "alpha": [
                    {"name": "not_in_registry"},  # 缺少 expression
                ],
            },
        }
        with open(tmp_strategies_dir / "missing_expr.yaml", "w") as f:
            yaml.dump(cfg, f)

        with patch("core.strategy.STRATEGIES_DIR", tmp_strategies_dir):
            with pytest.raises(ValueError, match="缺少 'expression' 字段"):
                Strategy.load("missing_expr")


# ── 交易成本参数化测试 ──────────────────────────────────

class TestTradingCost:
    """测试策略交易成本参数化"""

    def test_trading_cost_in_strategy(self, tmp_strategies_dir):
        cfg = {
            "name": "with_cost",
            "factors": {},
            "trading": {
                "open_cost": 0.0001,
                "close_cost": 0.0011,
                "slippage_bps": 8,
            },
        }
        with open(tmp_strategies_dir / "with_cost.yaml", "w") as f:
            yaml.dump(cfg, f)

        with patch("core.strategy.STRATEGIES_DIR", tmp_strategies_dir):
            s = Strategy.load("with_cost")

        assert s.trading_cost["open_cost"] == 0.0001
        assert s.trading_cost["close_cost"] == 0.0011
        assert s.trading_cost["slippage_bps"] == 8

    def test_trading_cost_default(self, tmp_strategies_dir):
        cfg = {"name": "no_cost", "factors": {}}
        with open(tmp_strategies_dir / "no_cost.yaml", "w") as f:
            yaml.dump(cfg, f)

        with patch("core.strategy.STRATEGIES_DIR", tmp_strategies_dir):
            s = Strategy.load("no_cost")

        assert s.trading_cost["open_cost"] == 0.0003
        assert s.trading_cost["close_cost"] == 0.0013


class TestValidityConfig:
    """测试 validity 配置解析和评估入口"""

    def test_strategy_loads_validity_config(self, tmp_strategies_dir):
        cfg = {
            "name": "with_validity",
            "factors": {},
            "validity": {
                "lookback_days": 40,
                "min_total_return": -0.03,
                "min_sharpe": 0.1,
                "max_drawdown": -0.08,
                "action": "reduce",
                "reduce_to": 0.4,
            },
        }
        with open(tmp_strategies_dir / "with_validity.yaml", "w") as f:
            yaml.dump(cfg, f)

        with patch("core.strategy.STRATEGIES_DIR", tmp_strategies_dir), \
             patch("core.strategy._load_strategy_defaults", return_value={}):
            s = Strategy.load("with_validity")

        assert s.validity is not None
        assert s.validity.lookback_days == 40
        assert s.validity.action == "reduce"
        assert s.validity.reduce_to == 0.4

    def test_strategy_evaluate_validity(self, tmp_strategies_dir):
        cfg = {
            "name": "gated",
            "factors": {},
            "validity": {
                "lookback_days": 20,
                "min_observations": 10,
                "min_total_return": -0.01,
                "min_annual_return": -0.05,
                "min_sharpe": 0.0,
                "max_drawdown": -0.05,
                "action": "pause",
            },
        }
        with open(tmp_strategies_dir / "gated.yaml", "w") as f:
            yaml.dump(cfg, f)

        daily_returns = pd.Series([-0.01] * 20)

        with patch("core.strategy.STRATEGIES_DIR", tmp_strategies_dir), \
             patch("core.strategy._load_strategy_defaults", return_value={}):
            s = Strategy.load("gated")

        result = s.evaluate_validity(daily_returns)
        assert result is not None
        assert result.status == "halt"
        assert result.action == "pause"
