import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.search_market_trend_release as search_market_trend_release


def _write_overlay_csv(path: Path, returns: list[float]) -> None:
    dates = pd.date_range("2024-01-02", periods=len(returns), freq="B")
    pd.DataFrame(
        {
            "date": dates,
            "base_return": returns,
            "bond_return": [0.0] * len(returns),
        }
    ).to_csv(path, index=False)


def test_load_overlay_returns_uses_base_and_bond_columns(tmp_path):
    csv_path = tmp_path / "overlay_results.csv"
    _write_overlay_csv(csv_path, [0.01, -0.02, 0.03])

    segment = search_market_trend_release.load_overlay_returns(csv_path)

    assert segment.base_returns.tolist() == pytest.approx([0.01, -0.02, 0.03])
    assert segment.bond_returns.tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert list(segment.base_returns.index) == list(pd.date_range("2024-01-02", periods=3, freq="B"))


def test_evaluate_candidate_reports_market_release_across_segments(tmp_path):
    full_path = tmp_path / "full.csv"
    valid_path = tmp_path / "valid.csv"
    holdout_path = tmp_path / "holdout.csv"
    returns = [0.01] * 8
    _write_overlay_csv(full_path, returns)
    _write_overlay_csv(valid_path, returns[:5])
    _write_overlay_csv(holdout_path, returns[3:])

    market_returns = pd.Series(
        [0.02] * 8,
        index=pd.date_range("2024-01-02", periods=8, freq="B"),
        dtype=float,
    )
    segments = {
        "full": search_market_trend_release.load_overlay_returns(full_path),
        "valid": search_market_trend_release.load_overlay_returns(valid_path),
        "holdout": search_market_trend_release.load_overlay_returns(holdout_path),
    }
    base_overlay = {
        "target_vol": None,
        "vol_lookback": 20,
        "dd_soft": 0.001,
        "dd_hard": 0.002,
        "soft_exposure": 0.20,
        "hard_exposure": 0.10,
        "trend_lookback": 0,
        "trend_exposure": 0.20,
        "exposure_min": 0.0,
        "exposure_max": 1.0,
    }

    row = search_market_trend_release.evaluate_candidate(
        tag="candidate",
        base_overlay=base_overlay,
        market_params={
            "market_trend_lookback": 2,
            "market_trend_min_return": 0.01,
            "market_trend_exposure_floor": 0.60,
            "market_trend_max_strategy_drawdown": 0.05,
        },
        segments=segments,
        market_returns=market_returns,
    )

    assert row["tag"] == "candidate"
    assert row["full_avg_exposure"] > 0.60
    assert row["valid_annual_return"] > 0
    assert row["holdout_max_drawdown"] == pytest.approx(0.0)
    assert row["market_trend_exposure_floor"] == pytest.approx(0.60)
