"""
仓位控制模块测试
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.position import (
    MarketConfig,
    AllocationResult,
    MarketPositionController,
)


class TestMarketConfig:
    """MarketConfig 测试"""

    def test_market_config_defaults(self):
        """测试默认配置"""
        config = MarketConfig()

        assert config.ma_fast == 20
        assert config.ma_slow == 60
        assert config.peak_lookback == 120
        assert config.strong_bull_threshold == 0.05
        assert config.mild_bull_threshold == 0.02
        assert config.mild_bear_threshold == -0.02
        assert config.strong_bear_threshold == -0.05
        assert config.smoothing_factor == 0.7
        assert config.bond_annual_return == 0.03

    def test_market_config_regime_allocations(self):
        """测试趋势仓位配置"""
        config = MarketConfig()

        assert "strong_bull" in config.regime_allocations
        assert "strong_bear" in config.regime_allocations
        assert config.regime_allocations["strong_bull"] == 1.0
        assert config.regime_allocations["strong_bear"] == 0.5

    def test_market_config_opportunity_thresholds(self):
        """测试机会捕捉阈值"""
        config = MarketConfig()

        assert len(config.opportunity_thresholds) == 3
        # 检查阈值格式: (回撤阈值, 仓位覆盖, 标签)
        for threshold in config.opportunity_thresholds:
            assert len(threshold) == 3
            assert isinstance(threshold[0], (int, float))
            assert isinstance(threshold[1], (int, float))
            assert isinstance(threshold[2], str)


class TestAllocationResult:
    """AllocationResult 测试"""

    def test_allocation_result_creation(self):
        """测试分配结果创建"""
        result = AllocationResult(
            stock_pct=0.8,
            cash_pct=0.2,
            regime="neutral",
            opportunity_level="none",
            market_drawdown=-0.05,
            trend_score=0.02
        )

        assert result.stock_pct == 0.8
        assert result.cash_pct == 0.2
        assert result.regime == "neutral"
        assert result.opportunity_level == "none"
        assert result.market_drawdown == -0.05
        assert result.trend_score == 0.02


class TestMarketPositionController:
    """MarketPositionController 测试"""

    def test_get_regime_strong_bull(self, mock_position_controller):
        """测试强牛市判断"""
        # 策略使用前一日数据避免前视偏差，因此修改倒数第二个值
        mock_position_controller.ma_ratio.iloc[-2] = 0.06

        regime, allocation = mock_position_controller._get_regime(
            mock_position_controller.csi300_close.index[-1]
        )

        assert regime == "strong_bull"
        assert allocation == 1.0

    def test_get_regime_strong_bear(self, mock_position_controller):
        """测试强熊市判断"""
        mock_position_controller.ma_ratio.iloc[-2] = -0.06

        regime, allocation = mock_position_controller._get_regime(
            mock_position_controller.csi300_close.index[-1]
        )

        assert regime == "strong_bear"
        assert allocation == 0.5

    def test_get_regime_neutral(self, mock_position_controller):
        """测试中性市场判断"""
        mock_position_controller.ma_ratio.iloc[-2] = 0.0

        regime, allocation = mock_position_controller._get_regime(
            mock_position_controller.csi300_close.index[-1]
        )

        assert regime == "neutral"
        assert allocation == 0.8

    def test_get_opportunity_high_drawdown(self, mock_position_controller):
        """测试高回撤机会捕捉"""
        mock_position_controller.drawdown.iloc[-2] = -0.20

        level, override = mock_position_controller._get_opportunity(
            mock_position_controller.csi300_close.index[-1]
        )

        assert level == "heavy"
        assert override > 0

    def test_get_opportunity_no_drawdown(self, mock_position_controller):
        """测试无回撤时的机会捕捉"""
        mock_position_controller.drawdown.iloc[-2] = -0.02

        level, override = mock_position_controller._get_opportunity(
            mock_position_controller.csi300_close.index[-1]
        )

        assert level == "none"
        assert override == 0.0

    def test_get_allocation(self, mock_position_controller):
        """测试仓位分配"""
        result = mock_position_controller.get_allocation(
            mock_position_controller.csi300_close.index[-1],
            is_rebalance_day=True
        )

        assert isinstance(result, AllocationResult)
        assert 0 <= result.stock_pct <= 1
        assert 0 <= result.cash_pct <= 1
        assert result.stock_pct + result.cash_pct == pytest.approx(1.0, abs=0.01)

    def test_get_allocation_out_of_range(self, mock_position_controller):
        """测试日期超出范围时的仓位分配"""
        # 使用一个超出范围的日期
        future_date = mock_position_controller.csi300_close.index[-1] + pd.Timedelta(days=365)

        result = mock_position_controller.get_allocation(future_date)

        # 应该返回默认值
        assert result.stock_pct == 0.8
        assert result.cash_pct == 0.2
        assert result.regime == "neutral"

    def test_get_bond_daily_return(self, mock_position_controller):
        """测试债券日收益"""
        daily_return = mock_position_controller.get_bond_daily_return()

        expected = 0.03 / 252
        assert abs(daily_return - expected) < 1e-10

    def test_get_allocation_smoothing(self, mock_position_controller):
        """测试仓位平滑"""
        # 第一次调用
        result1 = mock_position_controller.get_allocation(
            mock_position_controller.csi300_close.index[-1],
            is_rebalance_day=True
        )

        # 第二次调用（非调仓日，应该应用平滑）
        result2 = mock_position_controller.get_allocation(
            mock_position_controller.csi300_close.index[-1],
            is_rebalance_day=False
        )

        # 平滑后的仓位应该与前一次有关
        assert isinstance(result2, AllocationResult)


class TestMarketPositionControllerIndicators:
    """技术指标计算测试"""

    def test_compute_indicators(self, mock_position_controller):
        """测试指标计算"""
        # 检查指标已计算
        assert mock_position_controller.ma_fast is not None
        assert mock_position_controller.ma_slow is not None
        assert mock_position_controller.ma_ratio is not None
        assert mock_position_controller.peak is not None
        assert mock_position_controller.drawdown is not None

    def test_ma_calculation(self, mock_position_controller):
        """测试 MA 计算"""
        close = mock_position_controller.csi300_close
        ma_fast = mock_position_controller.ma_fast

        # MA20 应该是前 20 天的均值（简化验证）
        assert len(ma_fast) == len(close)

    def test_drawdown_calculation(self, mock_position_controller):
        """测试回撤计算"""
        drawdown = mock_position_controller.drawdown

        # 回撤应该 <= 0
        assert (drawdown <= 0).all() or drawdown.isna().all()
