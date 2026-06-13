"""统一信号契约。

策略层对引擎层的规范产出物。两类 producer：
- TARGET_PORTFOLIO（因子/模型/PE）：产出 TargetPortfolio（每日目标组合 date→{instrument:weight}）
- EVENT_DRIVEN（海龟/金字塔）：产出 on_bar 事件，由规则引擎内部归约

最终敞口 = sizing(date).stock_pct × weights[date]（per-instrument 权重与组合级仓位意图正交）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Set, Optional, List, Any

import pandas as pd


class ProducerKind(Enum):
    TARGET_PORTFOLIO = "target_portfolio"   # 因子/模型/PE：吃权重
    EVENT_DRIVEN = "event_driven"           # 海龟/金字塔：吃 on_bar 事件


@dataclass
class TargetPortfolio:
    """策略层产出的规范形态：调仓日 → {instrument: weight}。

    weights[date] 内权重和 <= 1.0（余下为现金）；空 dict 表示当日空仓。
    sizing 是可选的组合级仓位意图（= 旧 MarketPositionController），与 per-instrument
    权重正交。ranked 保留排名序，供涨停阻买时的递补。
    """
    weights: Dict[pd.Timestamp, Dict[str, float]]
    rebalance_dates: Set[pd.Timestamp]
    topk: int
    sizing: Optional[Any] = None
    name: str = ""
    universe: str = "all"
    ranked: Dict[pd.Timestamp, List[str]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_equal_weight(
        cls,
        date_to_symbols: Dict[pd.Timestamp, List[str]],
        rebalance_dates,
        topk: int,
        sizing: Optional[Any] = None,
        name: str = "",
        universe: str = "all",
        ranked: Optional[Dict[pd.Timestamp, List[str]]] = None,
    ) -> "TargetPortfolio":
        """从「日期→选股名单」按等权构造（保持 topk 等权口径）。"""
        weights: Dict[pd.Timestamp, Dict[str, float]] = {}
        for d, syms in date_to_symbols.items():
            syms = list(syms)
            n = len(syms)
            weights[d] = {s: 1.0 / n for s in syms} if n else {}
        return cls(
            weights=weights,
            rebalance_dates=set(rebalance_dates),
            topk=topk,
            sizing=sizing,
            name=name,
            universe=universe,
            ranked=ranked or {},
        )
