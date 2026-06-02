"""
风控模块测试
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock


class TestStopLossConfig:
    """止损配置测试"""

    def test_default_config(self):
        """测试默认配置"""
        from modules.risk.stoploss import StopLossConfig, StopLossType
        
        config = StopLossConfig()
        
        assert config.enabled is False
        assert config.type == StopLossType.FIXED_PCT
        assert config.threshold == -0.08

    def test_custom_config(self):
        """测试自定义配置"""
        from modules.risk.stoploss import StopLossConfig, StopLossType
        
        config = StopLossConfig(
            enabled=True,
            type=StopLossType.TRAILING_PCT,
            threshold=-0.10,
            trailing_high_pct=0.08
        )
        
        assert config.enabled is True
        assert config.type == StopLossType.TRAILING_PCT
        assert config.threshold == -0.10
        assert config.trailing_high_pct == 0.08


class TestIndividualStopLoss:
    """个股止损测试"""

    def test_on_buy_records_entry_price(self):
        """测试买入时记录入场价"""
        from modules.risk.stoploss import IndividualStopLoss
        
        stoploss = IndividualStopLoss()
        stoploss.on_buy("SZ000001", 10.5)
        
        assert "SZ000001" in stoploss.entry_prices
        assert stoploss.entry_prices["SZ000001"] == 10.5

    def test_on_sell_clears_record(self):
        """测试卖出时清除记录"""
        from modules.risk.stoploss import IndividualStopLoss
        
        stoploss = IndividualStopLoss()
        stoploss.on_buy("SZ000001", 10.5)
        stoploss.on_sell("SZ000001")
        
        assert "SZ000001" not in stoploss.entry_prices

    def test_fixed_pct_stop_loss_trigger(self):
        """测试固定比例止损触发"""
        from modules.risk.stoploss import IndividualStopLoss, StopLossConfig, StopLossType
        
        config = StopLossConfig(
            enabled=True,
            type=StopLossType.FIXED_PCT,
            threshold=-0.10  # 亏损10%止损
        )
        stoploss = IndividualStopLoss(config)
        stoploss.on_buy("SZ000001", 10.0)
        
        # 亏损 11%，触发止损
        triggered = stoploss.check("SZ000001", 8.9)
        assert triggered is True

    def test_fixed_pct_stop_loss_not_trigger(self):
        """测试固定比例止损未触发"""
        from modules.risk.stoploss import IndividualStopLoss, StopLossConfig, StopLossType
        
        config = StopLossConfig(
            enabled=True,
            type=StopLossType.FIXED_PCT,
            threshold=-0.10
        )
        stoploss = IndividualStopLoss(config)
        stoploss.on_buy("SZ000001", 10.0)
        
        # 盈利 5%，不触发止损
        triggered = stoploss.check("SZ000001", 10.5)
        assert triggered is False

    def test_trailing_stop_loss(self):
        """测试移动止损"""
        from modules.risk.stoploss import IndividualStopLoss, StopLossConfig, StopLossType
        
        config = StopLossConfig(
            enabled=True,
            type=StopLossType.TRAILING_PCT,
            threshold=-0.05,  # 5%回撤触发
            trailing_high_pct=0.05
        )
        stoploss = IndividualStopLoss(config)
        stoploss.on_buy("SZ000001", 10.0)
        
        # 涨到 11.0，然后跌到 10.5 (回撤 4.5%，未触发)
        stoploss.check("SZ000001", 11.0)
        triggered = stoploss.check("SZ000001", 10.5)
        assert triggered is False
        
        # 继续跌到 10.45 (距高点 5%，触发)
        triggered = stoploss.check("SZ000001", 10.45)
        assert triggered is True


class TestPortfolioStopLoss:
    """组合止损测试"""

    def test_portfolio_max_drawdown(self):
        """测试组合最大回撤止损"""
        from modules.risk.stoploss import PortfolioStopLoss, PortfolioStopLossConfig
        
        config = PortfolioStopLossConfig(
            enabled=True,
            max_drawdown_pct=-0.15  # 组合回撤15%止损
        )
        portfolio = PortfolioStopLoss(config)
        
        # 模拟组合净值变化：1.0 -> 1.1 -> 1.05 -> 0.88
        # 回撤 = (1.1 - 0.88) / 1.1 = 20% > 15%，触发
        portfolio.update_peak(1.0)
        portfolio.update_peak(1.1)
        portfolio.update_peak(1.05)
        
        should_stop, reason = portfolio.check(0.88)
        assert should_stop is True
        assert "回撤" in reason

    def test_portfolio_no_drawdown(self):
        """测试组合无回撤不触发"""
        from modules.risk.stoploss import PortfolioStopLoss, PortfolioStopLossConfig
        
        config = PortfolioStopLossConfig(
            enabled=True,
            max_drawdown_pct=-0.15
        )
        portfolio = PortfolioStopLoss(config)
        
        # 组合净值持续上涨
        portfolio.update_peak(1.0)
        portfolio.update_peak(1.05)
        portfolio.update_peak(1.10)
        
        should_stop, reason = portfolio.check(1.12)
        assert should_stop is False
        assert reason == ""

    def test_daily_loss_limit(self):
        """测试单日亏损限制"""
        from modules.risk.stoploss import PortfolioStopLoss, PortfolioStopLossConfig
        
        config = PortfolioStopLossConfig(
            enabled=True,
            daily_loss_limit_pct=-0.05  # 单日亏损5%止损
        )
        portfolio = PortfolioStopLoss(config)
        portfolio.update_peak(1.0)
        
        # 单日亏损 6%，触发
        should_stop, reason = portfolio.check(0.94, daily_return=-0.06)
        assert should_stop is True
        assert "单日亏损" in reason

    def test_daily_loss_limit_not_trigger(self):
        """测试单日亏损未触发"""
        from modules.risk.stoploss import PortfolioStopLoss, PortfolioStopLossConfig
        
        config = PortfolioStopLossConfig(
            enabled=True,
            daily_loss_limit_pct=-0.05
        )
        portfolio = PortfolioStopLoss(config)
        portfolio.update_peak(1.0)
        
        # 单日盈利 2%
        should_stop, reason = portfolio.check(1.02, daily_return=0.02)
        assert should_stop is False


class TestRiskModuleIntegration:
    """风控模块集成测试"""

    def test_stop_loss_disabled(self):
        """测试禁用止损"""
        from modules.risk.stoploss import IndividualStopLoss, StopLossConfig
        
        config = StopLossConfig(enabled=False)
        stoploss = IndividualStopLoss(config)
        stoploss.on_buy("SZ000001", 10.0)
        
        # 即使亏损50%，也不触发（已禁用）
        triggered = stoploss.check("SZ000001", 5.0)
        assert triggered is False

    def test_portfolio_stop_loss_disabled(self):
        """测试组合止损禁用"""
        from modules.risk.stoploss import PortfolioStopLoss, PortfolioStopLossConfig
        
        config = PortfolioStopLossConfig(enabled=False)
        portfolio = PortfolioStopLoss(config)
        
        # 无论回撤多少，都不触发
        portfolio.update_peak(1.0)
        should_stop, reason = portfolio.check(0.5)
        assert should_stop is False

    def test_check_unknown_symbol(self):
        """测试检查未知股票"""
        from modules.risk.stoploss import IndividualStopLoss
        
        stoploss = IndividualStopLoss()
        # 未买入的股票
        triggered = stoploss.check("SZ999999", 5.0)
        assert triggered is False
