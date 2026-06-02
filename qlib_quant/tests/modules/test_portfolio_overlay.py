import sys
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.modeling.portfolio_overlay import (
    OverlayConfig,
    compute_inverse_vol_weights,
    compute_overlay_frame,
)


class TestPortfolioOverlay:
    def test_identity_overlay_matches_base_when_exposure_is_full(self):
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        base = pd.Series([0.01, -0.02, 0.03, 0.0], index=dates)
        bond = pd.Series([0.0001] * 4, index=dates)

        frame = compute_overlay_frame(base, bond, OverlayConfig())

        assert frame["exposure"].eq(1.0).all()
        assert frame["overlay_return"].equals(base.rename("overlay_return"))

    def test_drawdown_caps_apply_from_next_day(self):
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        base = pd.Series([0.0, -0.10, -0.10, 0.0], index=dates)
        bond = pd.Series([0.0] * 4, index=dates)

        frame = compute_overlay_frame(
            base,
            bond,
            OverlayConfig(dd_hard=0.08, hard_exposure=0.5),
        )

        assert frame.loc[dates[0], "exposure"] == pytest.approx(1.0)
        assert frame.loc[dates[1], "exposure"] == pytest.approx(1.0)
        assert frame.loc[dates[2], "exposure"] == pytest.approx(0.5)
        assert frame.loc[dates[2], "overlay_return"] == pytest.approx(-0.05)

    def test_target_vol_scales_down_high_volatility(self):
        dates = pd.date_range("2024-01-01", periods=6, freq="D")
        base = pd.Series([0.05, -0.05, 0.05, -0.05, 0.02, 0.01], index=dates)
        bond = pd.Series([0.0] * 6, index=dates)

        frame = compute_overlay_frame(
            base,
            bond,
            OverlayConfig(target_vol=0.20, vol_lookback=3, exposure_max=1.0),
        )

        assert frame.loc[dates[0], "exposure"] == pytest.approx(1.0)
        assert frame.loc[dates[1], "exposure"] == pytest.approx(1.0)
        assert frame.loc[dates[4], "exposure"] < 1.0

    def test_market_trend_floor_releases_exposure_when_strategy_drawdown_is_controlled(self):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        base = pd.Series([0.0, -0.03, 0.01, 0.01, 0.01], index=dates)
        market = pd.Series([0.02, 0.02, 0.02, 0.02, 0.02], index=dates)
        bond = pd.Series([0.0] * 5, index=dates)

        frame = compute_overlay_frame(
            base,
            bond,
            OverlayConfig(
                dd_hard=0.02,
                hard_exposure=0.25,
                market_trend_lookback=2,
                market_trend_min_return=0.01,
                market_trend_exposure_floor=0.60,
                market_trend_max_strategy_drawdown=0.05,
            ),
            market_returns=market,
        )

        assert bool(frame.loc[dates[2], "market_risk_on"]) is True
        assert frame.loc[dates[2], "exposure"] == pytest.approx(0.60)

    def test_market_trend_floor_does_not_release_when_strategy_drawdown_is_too_deep(self):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        base = pd.Series([0.0, -0.10, 0.01, 0.01, 0.01], index=dates)
        market = pd.Series([0.02, 0.02, 0.02, 0.02, 0.02], index=dates)
        bond = pd.Series([0.0] * 5, index=dates)

        frame = compute_overlay_frame(
            base,
            bond,
            OverlayConfig(
                dd_hard=0.02,
                hard_exposure=0.25,
                market_trend_lookback=2,
                market_trend_min_return=0.01,
                market_trend_exposure_floor=0.60,
                market_trend_max_strategy_drawdown=0.05,
            ),
            market_returns=market,
        )

        assert bool(frame.loc[dates[2], "market_risk_on"]) is False
        assert frame.loc[dates[2], "exposure"] == pytest.approx(0.25)

    def test_inverse_vol_weights_inverse_sqrt_biases_toward_low_vol(self):
        vols = pd.Series(
            [0.10, 0.40, 0.25],
            index=["SZ000001", "SZ000002", "SZ000003"],
            dtype=float,
        )

        weights = compute_inverse_vol_weights(vols, method="inverse_sqrt")

        assert weights.sum() == pytest.approx(1.0)
        assert weights["SZ000001"] > weights["SZ000003"] > weights["SZ000002"]

    def test_inverse_vol_weights_fallback_to_equal_when_all_invalid(self):
        vols = pd.Series([0.0, float("nan"), -1.0], index=["A", "B", "C"], dtype=float)

        weights = compute_inverse_vol_weights(vols, method="inverse")

        assert weights["A"] == pytest.approx(1 / 3)
        assert weights["B"] == pytest.approx(1 / 3)
        assert weights["C"] == pytest.approx(1 / 3)

    def test_inverse_vol_weights_respects_cap_when_feasible(self):
        vols = pd.Series([0.01, 0.04, 0.09, 0.16], index=list("ABCD"), dtype=float)

        weights = compute_inverse_vol_weights(vols, method="inverse", cap_max_weight=0.40)

        assert weights.sum() == pytest.approx(1.0)
        assert float(weights.max()) <= 0.40 + 1e-8
