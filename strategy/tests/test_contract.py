"""TDD: strategy/contract.py — 统一信号契约 TargetPortfolio / ProducerKind"""
import pytest
import pandas as pd
from strategy.contract import TargetPortfolio, ProducerKind


def test_producer_kind_values():
    assert ProducerKind.TARGET_PORTFOLIO.value == "target_portfolio"
    assert ProducerKind.EVENT_DRIVEN.value == "event_driven"


def test_target_portfolio_construction_defaults():
    d = pd.Timestamp("2024-01-02")
    tp = TargetPortfolio(weights={d: {"X": 0.5}}, rebalance_dates={d}, topk=10)
    assert tp.topk == 10
    assert tp.name == ""
    assert tp.universe == "all"
    assert tp.sizing is None
    assert tp.ranked == {}
    assert tp.metadata == {}


def test_target_portfolio_from_equal_weight():
    d = pd.Timestamp("2024-01-02")
    tp = TargetPortfolio.from_equal_weight(
        {d: ["A", "B", "C", "D"]}, {d}, topk=4, name="x", universe="csi300"
    )
    w = tp.weights[d]
    assert sum(w.values()) == pytest.approx(1.0)
    assert all(v == pytest.approx(0.25) for v in w.values())
    assert tp.name == "x"
    assert tp.universe == "csi300"
    assert tp.rebalance_dates == {d}


def test_from_equal_weight_empty_day_yields_empty_weights():
    d = pd.Timestamp("2024-01-02")
    tp = TargetPortfolio.from_equal_weight({d: []}, {d}, topk=4)
    assert tp.weights[d] == {}
