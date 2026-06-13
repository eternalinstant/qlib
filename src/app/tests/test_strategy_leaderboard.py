"""
Unit tests for scripts/strategy_leaderboard.py
"""
import math
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.strategy_leaderboard import (
    _parse_slug,
    apply_filters,
    build_leaderboard,
    compute_metrics_from_daily,
    render_markdown_table,
)

# ── slug parsing ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("fname,expected_slug,expected_uni", [
    ("backtest_push25_cq10_k8d2_very_tight_historical_csi300_20260508_013650.csv",
     "push25_cq10_k8d2_very_tight", "historical_csi300"),
    ("backtest_alpha158_momentum_volume_k6_historical_csi300_20260509_010753.csv",
     "alpha158_momentum_volume_k6", "historical_csi300"),
    ("backtest_my_strategy_all_20260101_120000.csv",
     "my_strategy", "all"),
    ("backtest_my_strategy_csi500_20260101_120000.csv",
     "my_strategy", "csi500"),
    ("backtest_no_universe_tag_20260101_120000.csv",
     "no_universe_tag", None),
])
def test_parse_slug(fname, expected_slug, expected_uni):
    slug, uni = _parse_slug(fname)
    assert slug == expected_slug
    assert uni == expected_uni


def test_parse_slug_non_backtest():
    slug, uni = _parse_slug("comparison_result_20260101.csv")
    assert slug is None
    assert uni is None


# ── metric calculation ────────────────────────────────────────────────────────

def _make_daily_csv(tmp_path: Path, returns, buy_counts=None, sell_counts=None,
                    cost_per_day=None) -> Path:
    n = len(returns)
    dates = pd.date_range("2022-01-03", periods=n, freq="B")
    data = {"date": dates, "return": returns}
    if buy_counts is not None:
        data["buy_count"] = buy_counts
    if sell_counts is not None:
        data["sell_count"] = sell_counts
    data["gross_return"] = [r + 0.001 for r in returns]
    data["cost"] = [c if cost_per_day else 0.0 for c in (cost_per_day or [0.0] * n)]
    df = pd.DataFrame(data)
    p = tmp_path / "backtest_test.csv"
    df.to_csv(p, index=False)
    return p


def _make_dated_series(rets):
    dates = pd.date_range("2022-01-03", periods=len(rets), freq="B")
    return pd.Series(rets, index=dates)


def test_cagr_matches_backtest_result(tmp_path):
    """CAGR computed here must match BacktestResult.annual_return."""
    from engine.base import BacktestResult

    np.random.seed(42)
    rets = np.random.normal(0.001, 0.015, 500)
    p = _make_daily_csv(tmp_path, rets.tolist())
    m = compute_metrics_from_daily(p)

    s = _make_dated_series(rets)
    pv = (1 + s).cumprod()
    br = BacktestResult(daily_returns=s, portfolio_value=pv)

    assert abs(m["full_cagr"] - br.annual_return) < 0.005


def test_sharpe_matches_backtest_result(tmp_path):
    from engine.base import BacktestResult

    np.random.seed(7)
    rets = np.random.normal(0.0008, 0.012, 600)
    p = _make_daily_csv(tmp_path, rets.tolist())
    m = compute_metrics_from_daily(p)

    s = _make_dated_series(rets)
    pv = (1 + s).cumprod()
    br = BacktestResult(daily_returns=s, portfolio_value=pv)

    assert abs(m["full_sharpe"] - br.sharpe_ratio) < 0.01


def test_max_dd_matches_backtest_result(tmp_path):
    from engine.base import BacktestResult

    np.random.seed(13)
    rets = np.random.normal(0.0005, 0.018, 400)
    p = _make_daily_csv(tmp_path, rets.tolist())
    m = compute_metrics_from_daily(p)

    s = _make_dated_series(rets)
    pv = (1 + s).cumprod()
    br = BacktestResult(daily_returns=s, portfolio_value=pv)

    assert abs(m["full_max_dd"] - br.max_drawdown) < 0.001


def test_missing_return_column(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"date": ["2022-01-03"], "gross_return": [0.001]}).to_csv(p, index=False)
    m = compute_metrics_from_daily(p)
    assert m == {}


def test_turnover_computed(tmp_path):
    rets = [0.001] * 252
    buys  = [4] * 252
    sells = [4] * 252
    p = _make_daily_csv(tmp_path, rets, buy_counts=buys, sell_counts=sells)
    m = compute_metrics_from_daily(p)
    # 252 days × (4+4) trades / 252 days * 252 = 8 * 252
    assert abs(m["turnover"] - 8.0 * 252) < 1.0


# ── filtering ─────────────────────────────────────────────────────────────────

def _make_df():
    return pd.DataFrame([
        {"strategy": "a", "sharpe": 2.0, "max_dd": -0.05, "oos_decay": 0.8,
         "last_run": (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d"), "stale": False},
        {"strategy": "b", "sharpe": 0.5, "max_dd": -0.20, "oos_decay": 0.3,
         "last_run": (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d"), "stale": False},
        {"strategy": "c_old", "sharpe": 1.8, "max_dd": -0.08, "oos_decay": 0.9,
         "last_run": (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d"), "stale": True},
    ])


def test_filter_min_sharpe():
    df = apply_filters(_make_df(), min_sharpe=1.5)
    assert set(df["strategy"]) == {"a", "c_old"}


def test_filter_max_dd():
    df = apply_filters(_make_df(), max_dd=-0.10)
    assert set(df["strategy"]) == {"a", "c_old"}


def test_filter_not_stale():
    df = apply_filters(_make_df(), not_stale_days=30)
    assert "c_old" not in df["strategy"].values


def test_filter_grep():
    df = apply_filters(_make_df(), grep="old")
    assert list(df["strategy"]) == ["c_old"]


def test_filter_combined():
    df = apply_filters(_make_df(), min_sharpe=1.5, not_stale_days=30)
    assert list(df["strategy"]) == ["a"]


# ── fallback from aggregated table ───────────────────────────────────────────

def test_fallback_from_aggregated(tmp_path):
    """Strategy with empty OOS in daily CSV gets OOS from aggregated table."""
    # daily CSV with only IS data (before OOS_START 2024-01-01)
    dates = pd.date_range("2019-01-02", "2023-12-29", freq="B")
    rets = np.random.default_rng(0).normal(0.001, 0.015, len(dates))
    daily_df = pd.DataFrame({"date": dates, "return": rets,
                              "gross_return": rets + 0.001, "cost": 0.0})
    csv_path = tmp_path / "backtest_fallback_strat_historical_csi300_20260101_000000.csv"
    daily_df.to_csv(csv_path, index=False)

    # aggregated table with OOS data for that strategy.
    # series_kind/status/mtime 是真实 all_return_series_metrics.csv 必有的列：
    # build_leaderboard 先按 series_kind=="backtest" & status=="ok" 过滤，
    # 再按 mtime（float epoch，与文件 mtime 同型）排序去重保留最新一行，
    # 下游还 float(mtime)。旧 fixture 漏造这三列，补全以贴合真实数据形态与类型。
    agg_df = pd.DataFrame([{
        "strategy_name": "fallback_strat",
        "series_kind": "backtest", "status": "ok",
        "mtime": pd.Timestamp("2026-06-08").timestamp(),
        "full_cagr": 0.15, "full_max_dd": -0.12, "full_sharpe": 1.2, "full_calmar": 1.25,
        "oos_cagr": 0.22, "oos_max_dd": -0.08, "oos_sharpe": 1.8, "oos_calmar": 2.75,
    }])
    agg_path = tmp_path / "agg.csv"
    agg_df.to_csv(agg_path, index=False)

    df = build_leaderboard(
        results_dir=tmp_path,
        agg_metrics_path=agg_path,
        config_dirs=[],
    )
    assert not df.empty
    row = df[df["strategy"] == "fallback_strat"].iloc[0]
    assert abs(row["oos_sharpe"] - 1.8) < 0.01


# ── markdown rendering ────────────────────────────────────────────────────────

def test_render_markdown_has_header():
    df = _make_df()
    md = render_markdown_table(df)
    # render_markdown_table 有意输出中文表头（与整张榜单中文一致）；
    # 旧断言期望英文 "| Strategy |" 是过时的，对齐实际产品行为。
    assert "| 策略名称 |" in md
    assert "| a |" in md or "a" in md
