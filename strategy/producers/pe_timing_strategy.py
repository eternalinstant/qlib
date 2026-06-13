"""
PE 估值择时策略
- 基于截面 PE 百分位选股：低 PE（< pe_low_pct 百分位）入场，高 PE 不持有
- 实现 SignalStrategy 接口，权重按 1/PE 归一化
"""
from typing import Dict
import pandas as pd

from strategies.base import SignalStrategy


class PETimingStrategy(SignalStrategy):

    def compute_signals(self, date: pd.Timestamp, data_provider) -> Dict[str, float]:
        cfg = self.config.get("pe_timing", {})
        pe_low_pct = float(cfg.get("pe_low_pct", 30))
        factor = cfg.get("factor", "pe_ttm")
        sel_cfg = self.config.get("selection", {})
        top_k = int(sel_cfg.get("top_k", 20))

        universe_name = sel_cfg.get("universe", "csi800")
        universe = data_provider.get_universe(date, universe_name)
        if not universe:
            return {}

        pe_series = data_provider.get_factor(
            list(universe), factor,
            str(date.date()), str(date.date())
        )

        if pe_series is None or pe_series.empty:
            return {}

        # 只保留正 PE（负 PE 无意义）
        pe_series = pe_series[pe_series > 0].dropna()
        if pe_series.empty:
            return {}

        # 低 PE 百分位阈值
        threshold = pe_series.quantile(pe_low_pct / 100.0)
        low_pe = pe_series[pe_series <= threshold]

        if low_pe.empty:
            return {}

        # 按 1/PE 赋权，归一化到总和 <= 1
        weights = (1.0 / low_pe).nlargest(top_k)
        total = weights.sum()
        if total <= 0:
            return {}
        weights = weights / total

        return {inst: float(w) for inst, w in weights.items()}
