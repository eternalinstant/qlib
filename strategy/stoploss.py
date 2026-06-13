"""
止损模块
个股止损、组合止损
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from enum import Enum


class StopLossType(Enum):
    """止损类型"""
    FIXED_AMOUNT = "fixed_amount"      # 固定金额止损
    FIXED_PCT = "fixed_pct"            # 固定比例止损
    TRAILING_PCT = "trailing_pct"      # 移动比例止损
    MAX_DRAWDOWN = "max_drawdown"      # 最大回撤止损


@dataclass
class StopLossConfig:
    """止损配置"""
    enabled: bool = False
    type: StopLossType = StopLossType.FIXED_PCT
    threshold: float = -0.08           # 止损阈值 -8%
    trailing_high_pct: float = 0.05    # 移动止损回撤阈值


class IndividualStopLoss:
    """个股止损"""

    def __init__(self, config: StopLossConfig = None):
        self.config = config or StopLossConfig()
        self.entry_prices = {}         # {symbol: entry_price}
        self.trailing_highs = {}       # {symbol: trailing_high}

    def on_buy(self, symbol: str, price: float):
        """买入时记录入场价格"""
        self.entry_prices[symbol] = price
        self.trailing_highs[symbol] = price

    def on_sell(self, symbol: str):
        """卖出时清除记录"""
        self.entry_prices.pop(symbol, None)
        self.trailing_highs.pop(symbol, None)

    def check(self, symbol: str, current_price: float) -> bool:
        """
        检查是否触发止损

        Returns
        -------
        bool
            True 表示应该卖出
        """
        if not self.config.enabled:
            return False

        if symbol not in self.entry_prices:
            return False

        entry = self.entry_prices[symbol]
        pnl_pct = (current_price - entry) / entry

        if self.config.type == StopLossType.FIXED_PCT:
            return pnl_pct <= self.config.threshold

        elif self.config.type == StopLossType.TRAILING_PCT:
            # 更新移动高点
            if current_price > self.trailing_highs[symbol]:
                self.trailing_highs[symbol] = current_price

            trailing_dd = (current_price - self.trailing_highs[symbol]) / self.trailing_highs[symbol]
            return trailing_dd <= self.config.threshold

        return False


@dataclass
class PortfolioStopLossConfig:
    """组合止损配置"""
    enabled: bool = False
    max_drawdown_pct: float = -0.15    # 最大回撤 -15%
    daily_loss_limit_pct: float = -0.05  # 单日亏损限制 -5%
    peak_equity: float = None          # 历史最高权益


class PortfolioStopLoss:
    """组合止损"""

    def __init__(self, config: PortfolioStopLossConfig = None):
        self.config = config or PortfolioStopLossConfig()
        self.config.peak_equity = None

    def update_peak(self, current_equity: float):
        """更新历史最高权益"""
        if self.config.peak_equity is None or current_equity > self.config.peak_equity:
            self.config.peak_equity = current_equity

    def check(self, current_equity: float, daily_return: float = None) -> tuple[bool, str]:
        """
        检查组合止损

        Returns
        -------
        (should_stop, reason)
        """
        if not self.config.enabled:
            return False, ""

        # 最大回撤止损
        if self.config.peak_equity:
            dd = (current_equity - self.config.peak_equity) / self.config.peak_equity
            if dd <= self.config.max_drawdown_pct:
                return True, f"组合回撤 {dd:.2%} 超过阈值 {self.config.max_drawdown_pct:.2%}"

        # 单日亏损限制
        if daily_return is not None and daily_return <= self.config.daily_loss_limit_pct:
            return True, f"单日亏损 {daily_return:.2%} 超过阈值 {self.config.daily_loss_limit_pct:.2%}"

        return False, ""
