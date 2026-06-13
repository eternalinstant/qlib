from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pandas as pd


@dataclass
class PositionState:
    """单票仓位状态，规则驱动策略使用"""
    instrument: str
    entry_price: float
    layers: int = 1
    layer_prices: List[float] = field(default_factory=list)
    stop_loss: float = 0.0
    units: float = 0.0


@dataclass
class StrategySignal:
    """策略信号，两种引擎统一输出格式"""
    date: pd.Timestamp
    instrument: str
    action: str           # buy / sell / add / hold
    weight: float = 0.0
    price: Optional[float] = None
    reason: str = ""


class BaseStrategy(ABC):
    def __init__(self, config: dict):
        self.config = config
        self.name = config.get("name", self.__class__.__name__)

    @property
    @abstractmethod
    def strategy_type(self) -> str:
        pass

    @classmethod
    def from_yaml(cls, path: str) -> "BaseStrategy":
        import yaml
        with open(path) as f:
            config = yaml.safe_load(f)
        return cls(config)


class SignalStrategy(BaseStrategy):
    """因子选股类策略，对接现有 selection.py"""

    @property
    def strategy_type(self) -> str:
        return "signal"

    @abstractmethod
    def compute_signals(self, date: pd.Timestamp, data_provider) -> Dict[str, float]:
        pass


class RuleStrategy(BaseStrategy):
    """逐日事件驱动类策略（海龟、金字塔等）"""

    @property
    def strategy_type(self) -> str:
        return "rule"

    def on_start(self, data_provider) -> None:
        pass

    @abstractmethod
    def on_bar(
        self,
        date: pd.Timestamp,
        universe: List[str],
        positions: Dict[str, PositionState],
        data_provider,
    ) -> List[StrategySignal]:
        pass

    def on_end(self) -> None:
        pass
