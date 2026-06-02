"""
组合策略回测测试
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import yaml

from core.strategy import Strategy
from modules.backtest.base import BacktestResult
from modules.backtest.composite import run_strategy_backtest


def test_run_strategy_backtest_blends_component_results(tmp_path):
    strategies_dir = tmp_path / "strategies"
    strategies_dir.mkdir()

    for name in ("base_a", "base_b"):
        with open(strategies_dir / f"{name}.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump({"name": name, "factors": {}}, f, allow_unicode=True)

    combo_cfg = {
        "name": "combo",
        "composition": {
            "components": [
                {"strategy": "base_a", "weight": 0.6},
                {"strategy": "base_b", "weight": 0.2},
            ],
            "cash_weight": 0.2,
        },
    }
    with open(strategies_dir / "combo.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(combo_cfg, f, allow_unicode=True)

    dates = pd.date_range("2026-01-01", periods=3, freq="D")
    result_a = BacktestResult(
        daily_returns=pd.Series([0.01, 0.02, -0.01], index=dates),
        portfolio_value=(1 + pd.Series([0.01, 0.02, -0.01], index=dates)).cumprod(),
        metadata={"results_file": "a.csv"},
    )
    result_b = BacktestResult(
        daily_returns=pd.Series([0.0, 0.01, 0.03], index=dates),
        portfolio_value=(1 + pd.Series([0.0, 0.01, 0.03], index=dates)).cumprod(),
        metadata={"results_file": "b.csv"},
    )

    class FakeEngine:
        def run(self, strategy=None):
            return {"base_a": result_a, "base_b": result_b}[strategy.name]

    fake_config = SimpleNamespace(
        get=lambda key, default=None: str(tmp_path) if key in {"paths.results", "results_path"} else default
    )

    with patch("core.strategy.STRATEGIES_DIR", strategies_dir), \
         patch("core.strategy._load_strategy_defaults", return_value={}), \
         patch("modules.backtest.composite._make_engine", return_value=FakeEngine()), \
         patch("modules.backtest.composite.CONFIG", fake_config):
        combo = Strategy.load("combo")
        result = run_strategy_backtest(combo, engine="qlib")

    expected = pd.Series([0.006, 0.014, 0.0], index=dates)
    pd.testing.assert_series_equal(result.daily_returns, expected, check_names=False, check_freq=False)
    assert result.metadata["cash_weight"] == 0.2
    assert result.metadata["component_weights"] == {"base_a": 0.6, "base_b": 0.2}
    assert "results_file" in result.metadata
    assert Path(result.metadata["results_file"]).exists()


def test_run_strategy_backtest_applies_validity_overlay_for_composite(tmp_path):
    strategies_dir = tmp_path / "strategies"
    strategies_dir.mkdir()

    for name in ("base_a", "base_b"):
        with open(strategies_dir / f"{name}.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump({"name": name, "factors": {}}, f, allow_unicode=True)

    combo_cfg = {
        "name": "combo_dynamic",
        "composition": {
            "components": [
                {"strategy": "base_a", "weight": 0.4},
                {"strategy": "base_b", "weight": 0.4},
            ],
            "cash_weight": 0.2,
        },
        "validity": {
            "lookback_days": 3,
            "min_observations": 3,
            "min_total_return": -0.01,
            "min_annual_return": -0.10,
            "min_sharpe": 0.0,
            "max_drawdown": -0.03,
            "action": "reduce",
            "reduce_to": 0.5,
            "apply_in_backtest": True,
        },
    }
    with open(strategies_dir / "combo_dynamic.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(combo_cfg, f, allow_unicode=True)

    dates = pd.date_range("2026-01-01", periods=6, freq="D")
    child_returns = pd.Series([-0.02] * 6, index=dates)
    child_result = BacktestResult(
        daily_returns=child_returns,
        portfolio_value=(1 + child_returns).cumprod(),
        metadata={"results_file": "child.csv"},
    )

    class FakeEngine:
        def run(self, strategy=None):
            return child_result

    fake_config = SimpleNamespace(
        get=lambda key, default=None: str(tmp_path) if key in {"paths.results", "results_path"} else default
    )

    with patch("core.strategy.STRATEGIES_DIR", strategies_dir), \
         patch("core.strategy._load_strategy_defaults", return_value={}), \
         patch("modules.backtest.composite._make_engine", return_value=FakeEngine()), \
         patch("modules.backtest.composite.CONFIG", fake_config):
        combo = Strategy.load("combo_dynamic")
        result = run_strategy_backtest(combo, engine="qlib")

    assert result.metadata["validity_overlay"]["applied"] is True
    assert result.daily_returns.iloc[3] > -0.016
    assert result.metadata["validity_overlay"]["avg_exposure_factor"] < 1.0

    saved = pd.read_csv(result.metadata["results_file"])
    assert "raw_return" in saved.columns
    assert "exposure_factor" in saved.columns
    assert saved.loc[3, "raw_return"] == -0.016
    assert saved.loc[3, "exposure_factor"] < 1.0
