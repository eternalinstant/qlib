"""
因子定义模块
定义所有因子的表达式和计算逻辑

数据来源说明
-----------
source='qlib'    : 通过 D.features() 加载，expression 为 Qlib 算子表达式（基于 $close 等价格字段）
source='parquet' : 从 factor_data.parquet 加载，expression 为该文件的列名

默认因子池使用更稳的均衡组合：
  - Alpha: 基本面质量 + 价值 + 现金流
  - Risk: 低波动 + 低换手
  - Enhance: 中期趋势增强
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class FactorInfo:
    """因子信息"""
    name: str
    expression: str   # Qlib 算子表达式（source='qlib'）或 parquet 列名（source='parquet'）
    description: str
    category: str     # alpha / risk / enhance
    source: str = 'qlib'    # 'qlib' | 'parquet'
    negate: bool = False    # True：加载后乘以 -1（"越低越好"的因子）
    ir: float = 0.0         # Information Ratio（v2扫描实测值）


class FactorRegistry:
    """因子注册表（实例化设计，支持多注册表并行）"""

    def __init__(self):
        self._factors: Dict[str, FactorInfo] = {}

    def register(self, factor: FactorInfo):
        self._factors[factor.name] = factor

    def get(self, name: str) -> FactorInfo:
        return self._factors.get(name)

    def get_by_category(self, category: str) -> List[FactorInfo]:
        return [f for f in self._factors.values() if f.category == category]

    def get_by_source(self, source: str) -> List[FactorInfo]:
        return [f for f in self._factors.values() if f.source == source]

    def all(self) -> Dict[str, FactorInfo]:
        return self._factors.copy()

    def clear(self):
        self._factors.clear()

    def categories(self) -> set:
        """返回所有已注册因子的 category 集合"""
        return set(f.category for f in self._factors.values())


def init_default_factors(registry: 'FactorRegistry' = None):
    """初始化默认因子到指定注册表"""
    if registry is None:
        registry = default_registry

    ret = "($close - Ref($close, 1)) / Ref($close, 1)"

    alpha_factors = [
        FactorInfo("roa",
                   "roa_fina",
                   "资产收益率",
                   "alpha", source="parquet", ir=0.30),
        FactorInfo("book_to_price",
                   "book_to_market",
                   "账面市值比",
                   "alpha", source="parquet", ir=0.14),
        FactorInfo("ebit_to_mv",
                   "ebit_to_mv",
                   "EBIT/市值",
                   "alpha", source="parquet", ir=0.33),
        FactorInfo("ocf_to_ev",
                   "ocf_to_ev",
                   "经营现金流/企业价值",
                   "alpha", source="parquet", ir=0.33),
        FactorInfo("retained_earnings",
                   "retained_earnings",
                   "留存收益",
                   "alpha", source="parquet", ir=0.33),
    ]

    risk_factors = [
        FactorInfo("vol_std_20d",
                   f"Std({ret}, 20)",
                   "20日收益波动率（越低越好）",
                   "risk", source="qlib", negate=True, ir=0.31),
        FactorInfo("turnover_rate_f",
                   "turnover_rate_f",
                   "自由流通换手率（越低越好）",
                   "risk", source="parquet", negate=True, ir=0.42),
    ]

    enhance_factors = [
        FactorInfo("bbi_momentum",
                   "$close / ((Mean($close, 3) + Mean($close, 6) + Mean($close, 12) + Mean($close, 24)) / 4) - 1",
                   "BBI 动量",
                   "enhance", source="qlib", ir=0.26),
        FactorInfo("price_pos_52w",
                   "($close - Min($close, 252)) / (Max($close, 252) - Min($close, 252) + 1e-8)",
                   "52周价格位置",
                   "enhance", source="qlib", ir=0.24),
        FactorInfo("mom_20d",
                   "$close / Ref($close, 20) - 1",
                   "20日动量",
                   "enhance", source="qlib", ir=0.38),
    ]

    for f in alpha_factors + risk_factors + enhance_factors:
        registry.register(f)


def create_default_registry() -> FactorRegistry:
    """创建并初始化默认因子注册表"""
    registry = FactorRegistry()
    init_default_factors(registry)
    return registry


# 模块级默认注册表
default_registry = create_default_registry()



# ── 便捷查询函数（供 signals.py 等模块使用）──

def get_alpha_expressions() -> Dict[str, str]:
    """Alpha 层因子：{name: expression/parquet_col}"""
    return {f.name: f.expression for f in default_registry.get_by_category("alpha")}


def get_risk_expressions() -> Dict[str, str]:
    """风控层因子：{name: expression/parquet_col}"""
    return {f.name: f.expression for f in default_registry.get_by_category("risk")}


def get_enhance_expressions() -> Dict[str, str]:
    """增强层因子：{name: expression/parquet_col}"""
    return {f.name: f.expression for f in default_registry.get_by_category("enhance")}


def get_all_expressions():
    """获取所有 qlib source 因子的表达式和名称（用于 D.features()）"""
    all_fields, all_names = [], []
    for cat in ("alpha", "risk", "enhance"):
        for f in default_registry.get_by_category(cat):
            if f.source == "qlib":
                all_fields.append(f.expression)
                all_names.append(f"{cat}_{f.name}")
    return all_fields, all_names
