"""TDD: strategies/base.py — RED 阶段，所有测试应当失败"""
import pytest
import pandas as pd
from strategies.base import (
    PositionState,
    StrategySignal,
    BaseStrategy,
    SignalStrategy,
    RuleStrategy,
)


# ── PositionState ─────────────────────────────────────────────────────────────

def test_position_state_defaults():
    pos = PositionState(instrument="SH600000", entry_price=10.0)
    assert pos.instrument == "SH600000"
    assert pos.entry_price == 10.0
    assert pos.layers == 1
    assert pos.layer_prices == []
    assert pos.stop_loss == 0.0
    assert pos.units == 0.0


def test_position_state_custom_fields():
    pos = PositionState(
        instrument="SZ000001",
        entry_price=20.0,
        layers=2,
        layer_prices=[20.0, 21.0],
        stop_loss=18.0,
        units=5000.0,
    )
    assert pos.layers == 2
    assert pos.layer_prices == [20.0, 21.0]
    assert pos.stop_loss == 18.0


# ── StrategySignal ────────────────────────────────────────────────────────────

def test_strategy_signal_defaults():
    date = pd.Timestamp("2024-01-01")
    sig = StrategySignal(date=date, instrument="SH600000", action="buy")
    assert sig.date == date
    assert sig.instrument == "SH600000"
    assert sig.action == "buy"
    assert sig.weight == 0.0
    assert sig.price is None
    assert sig.reason == ""


def test_strategy_signal_valid_actions():
    date = pd.Timestamp("2024-01-01")
    for action in ("buy", "sell", "add", "hold"):
        sig = StrategySignal(date=date, instrument="SH600000", action=action)
        assert sig.action == action


# ── BaseStrategy ──────────────────────────────────────────────────────────────

def test_base_strategy_loads_name_from_config():
    class ConcreteStrategy(BaseStrategy):
        @property
        def strategy_type(self):
            return "signal"

    s = ConcreteStrategy({"name": "my_strategy"})
    assert s.name == "my_strategy"


def test_base_strategy_uses_class_name_when_no_name():
    class ConcreteStrategy(BaseStrategy):
        @property
        def strategy_type(self):
            return "signal"

    s = ConcreteStrategy({})
    assert s.name == "ConcreteStrategy"


def test_base_strategy_from_yaml(tmp_path):
    import yaml

    class ConcreteStrategy(BaseStrategy):
        @property
        def strategy_type(self):
            return "signal"

    config = {"name": "yaml_strategy", "param": 42}
    p = tmp_path / "strategy.yaml"
    p.write_text(yaml.dump(config))

    s = ConcreteStrategy.from_yaml(str(p))
    assert s.name == "yaml_strategy"
    assert s.config["param"] == 42


# ── SignalStrategy ────────────────────────────────────────────────────────────

def test_signal_strategy_type_is_signal():
    class MySignal(SignalStrategy):
        def compute_signals(self, date, data_provider):
            return {}

    s = MySignal({})
    assert s.strategy_type == "signal"


def test_signal_strategy_compute_signals_is_abstract():
    with pytest.raises(TypeError):
        SignalStrategy({})


# ── RuleStrategy ─────────────────────────────────────────────────────────────

def test_rule_strategy_type_is_rule():
    class MyRule(RuleStrategy):
        def on_bar(self, date, universe, positions, data_provider):
            return []

    s = MyRule({})
    assert s.strategy_type == "rule"


def test_rule_strategy_on_bar_is_abstract():
    with pytest.raises(TypeError):
        RuleStrategy({})


def test_rule_strategy_on_start_and_end_are_noop():
    class MyRule(RuleStrategy):
        def on_bar(self, date, universe, positions, data_provider):
            return []

    s = MyRule({})
    s.on_start(None)  # 不应抛出
    s.on_end()        # 不应抛出
