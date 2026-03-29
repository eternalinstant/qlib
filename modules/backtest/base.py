"""
回测引擎基类与统一结果数据结构
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, Set

import numpy as np
import pandas as pd

from config.config import CONFIG
from core.selection import load_selections
from core.position import MarketPositionController


@dataclass
class BacktestResult:
    """统一回测结果"""
    daily_returns: pd.Series
    portfolio_value: pd.Series
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_return(self) -> float:
        if self.portfolio_value.empty:
            return 0.0
        # portfolio_value 由 (1 + daily_returns).cumprod() 构造，首值已包含首日收益。
        return float(self.portfolio_value.iloc[-1] - 1.0)

    @property
    def annual_return(self) -> float:
        if self.portfolio_value.empty:
            return 0.0
        days = (self.portfolio_value.index[-1] - self.portfolio_value.index[0]).days
        if days <= 0:
            return 0.0
        terminal_value = float(self.portfolio_value.iloc[-1])
        if terminal_value <= 0:
            return -1.0
        return terminal_value ** (365 / days) - 1

    @property
    def sharpe_ratio(self) -> float:
        if self.daily_returns.empty or self.daily_returns.std() == 0:
            return 0.0
        return self.daily_returns.mean() / self.daily_returns.std() * np.sqrt(252)

    @property
    def max_drawdown(self) -> float:
        if self.portfolio_value.empty:
            return 0.0
        rolling_max = self.portfolio_value.cummax()
        drawdowns = self.portfolio_value / rolling_max - 1
        return float(drawdowns.min())

    def print_summary(self, initial_capital: float):
        """打印回测结果概览"""
        final_value = initial_capital * (
            self.portfolio_value.iloc[-1] if not self.portfolio_value.empty else 1.0
        )
        profit = final_value - initial_capital
        win_rate = (self.daily_returns > 0).mean() if not self.daily_returns.empty else 0

        print(f"\n{'='*60}")
        print("  资金损益概览")
        print(f"{'='*60}")
        print(f"  初始资金:       ¥{initial_capital:>12,.0f}")
        print(f"  期末资产:       ¥{final_value:>12,.0f}")
        print(f"  总盈亏:         ¥{profit:>+12,.0f}  ({self.total_return:+.2%})")
        print(f"  年化收益率:     {self.annual_return:>10.2%}")
        print(f"  夏普比率:       {self.sharpe_ratio:>10.4f}")
        print(f"  最大回撤:       {self.max_drawdown:>10.2%}")
        print(f"  日胜率:         {win_rate:>10.2%}")


class BacktestEngine(ABC):
    """回测引擎抽象基类"""

    def __init__(self, config=None):
        self.config = config

    def _prepare(self, strategy=None) -> Tuple[Dict, Set, Any, Any]:
        """公共准备步骤：加载选股列表和仓位控制器

        Returns
        -------
        Tuple[Dict, Set, Any, Any]
            (date_to_symbols, rebalance_dates, controller, topk)
        """
        if strategy:
            date_to_symbols, rebalance_dates = strategy.load_selections()
            controller = strategy.build_position_controller()
            topk = strategy.topk
        else:
            date_to_symbols, rebalance_dates = load_selections()
            controller = MarketPositionController()
            topk = CONFIG.get("topk", 20)

        if controller is not None and hasattr(controller, 'load_market_data'):
            controller.load_market_data()

        return date_to_symbols, rebalance_dates, controller, topk

    @abstractmethod
    def run(self, strategy=None) -> BacktestResult:
        """执行回测，返回统一结果

        Parameters
        ----------
        strategy : core.strategy.Strategy, optional
            策略对象。传入时使用策略的选股列表和仓位控制器；
            为 None 时使用默认行为（向后兼容）。
        """
        ...
