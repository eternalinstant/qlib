"""TDD: strategies/pe_timing/pe_timing_strategy.py"""
import pytest
import pandas as pd
from unittest.mock import MagicMock
from strategies.pe_timing.pe_timing_strategy import PETimingStrategy
from strategies.base import SignalStrategy


def _make_dp(pe_map: dict):
    """辅助：创建 mock DataProvider，根据 instrument+date 返回 PE 值"""
    dp = MagicMock()

    dp.get_universe.return_value = list(pe_map.keys())

    def get_factor(instruments, factor, start_date, end_date):
        if factor != "pe_ttm":
            return pd.Series(dtype=float)
        records = []
        for inst in instruments:
            if inst in pe_map:
                records.append({"instrument": inst, "value": pe_map[inst]})
        if not records:
            return pd.Series(dtype=float)
        df = pd.DataFrame(records).set_index("instrument")
        return df["value"]

    dp.get_factor.side_effect = get_factor
    return dp


_DEFAULT_CONFIG = {
    "name": "pe_timing_test",
    "selection": {"universe": "all", "top_k": 20},
    "pe_timing": {
        "pe_low_pct": 30,    # PE 处于历史 30 百分位以下才入场
        "pe_high_pct": 70,   # PE 处于历史 70 百分位以上减仓
        "factor": "pe_ttm",
    },
}


# ── 基本属性 ──────────────────────────────────────────────────────────────────

def test_pe_timing_is_signal_strategy():
    s = PETimingStrategy(_DEFAULT_CONFIG)
    assert isinstance(s, SignalStrategy)
    assert s.strategy_type == "signal"


def test_pe_timing_loads_from_yaml(tmp_path):
    import yaml
    p = tmp_path / "pe.yaml"
    p.write_text(yaml.dump(_DEFAULT_CONFIG))
    s = PETimingStrategy.from_yaml(str(p))
    assert s.name == "pe_timing_test"


# ── 信号计算 ──────────────────────────────────────────────────────────────────

def test_pe_timing_low_pe_gets_positive_weight():
    """低 PE 股票应获得正权重"""
    s = PETimingStrategy(_DEFAULT_CONFIG)
    date = pd.Timestamp("2024-01-05")
    # SZ000001 PE=8（低），SZ000002 PE=50（高）
    dp = _make_dp({"SZ000001": 8.0, "SZ000002": 50.0})
    signals = s.compute_signals(date, dp)
    assert "SZ000001" in signals
    assert signals["SZ000001"] > 0


def test_pe_timing_high_pe_gets_zero_or_negative_weight():
    """高 PE 股票应获得零或负权重"""
    s = PETimingStrategy(_DEFAULT_CONFIG)
    date = pd.Timestamp("2024-01-05")
    dp = _make_dp({"SZ000001": 8.0, "SZ000002": 50.0})
    signals = s.compute_signals(date, dp)
    # 高 PE 股票不在 signals 或权重 <= 0
    assert signals.get("SZ000002", 0) <= 0


def test_pe_timing_empty_universe_returns_empty():
    """无股票时应返回空字典"""
    s = PETimingStrategy(_DEFAULT_CONFIG)
    date = pd.Timestamp("2024-01-05")
    dp = _make_dp({})
    signals = s.compute_signals(date, dp)
    assert isinstance(signals, dict)
    assert len(signals) == 0


def test_pe_timing_signals_sum_to_one_or_less():
    """权重归一化：所有正权重之和 <= 1"""
    s = PETimingStrategy(_DEFAULT_CONFIG)
    date = pd.Timestamp("2024-01-05")
    dp = _make_dp({
        "SZ000001": 8.0,
        "SZ000002": 10.0,
        "SZ000003": 12.0,
        "SZ000004": 50.0,
    })
    signals = s.compute_signals(date, dp)
    total = sum(w for w in signals.values() if w > 0)
    assert total <= 1.0 + 1e-9
