"""
因子模块测试
"""

import pytest
import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.factors import (
    FactorInfo,
    FactorRegistry,
    default_registry,
    init_default_factors,
    get_alpha_expressions,
    get_risk_expressions,
    get_enhance_expressions,
    get_all_expressions,
)


class TestFactorInfo:
    """FactorInfo 数据类测试"""

    def test_factor_info_creation(self):
        """测试因子信息创建"""
        factor = FactorInfo(
            name="test_factor",
            expression="$close",
            description="测试因子",
            category="alpha",
            ir=0.5
        )
        assert factor.name == "test_factor"
        assert factor.expression == "$close"
        assert factor.description == "测试因子"
        assert factor.category == "alpha"
        assert factor.ir == 0.5

    def test_factor_info_default_ir(self):
        """测试默认 IR 值"""
        factor = FactorInfo(
            name="test",
            expression="$close",
            description="test",
            category="alpha"
        )
        assert factor.ir == 0.0


class TestFactorRegistry:
    """FactorRegistry 测试"""

    def test_register_and_get(self, reset_factor_registry):
        """测试注册和获取因子"""
        factor = FactorInfo(
            name="test_factor",
            expression="$close",
            description="测试因子",
            category="alpha"
        )
        default_registry.register(factor)
        result = default_registry.get("test_factor")
        assert result is factor

    def test_get_nonexistent(self, reset_factor_registry):
        """测试获取不存在的因子"""
        result = default_registry.get("nonexistent")
        assert result is None

    def test_get_by_category(self, reset_factor_registry):
        """测试按类别获取因子"""
        alpha_factor = FactorInfo("a1", "$close", "a1", "alpha")
        risk_factor = FactorInfo("r1", "$volume", "r1", "risk")

        default_registry.register(alpha_factor)
        default_registry.register(risk_factor)

        alpha_list = default_registry.get_by_category("alpha")
        risk_list = default_registry.get_by_category("risk")

        assert len(alpha_list) == 1
        assert alpha_list[0].name == "a1"
        assert len(risk_list) == 1
        assert risk_list[0].name == "r1"

    def test_all(self, reset_factor_registry):
        """测试获取所有因子"""
        f1 = FactorInfo("f1", "$close", "f1", "alpha")
        f2 = FactorInfo("f2", "$volume", "f2", "risk")

        default_registry.register(f1)
        default_registry.register(f2)

        all_factors = default_registry.all()
        assert "f1" in all_factors
        assert "f2" in all_factors
        assert len(all_factors) == 2

    def test_instance_isolation(self):
        """测试不同实例之间隔离"""
        reg1 = FactorRegistry()
        reg2 = FactorRegistry()

        f1 = FactorInfo("f1", "$close", "f1", "alpha")
        reg1.register(f1)

        assert reg1.get("f1") is f1
        assert reg2.get("f1") is None

    def test_clear(self, reset_factor_registry):
        """测试清空注册表"""
        default_registry.register(FactorInfo("x", "$close", "x", "alpha"))
        assert len(default_registry.all()) == 1
        default_registry.clear()
        assert len(default_registry.all()) == 0

    def test_categories(self, reset_factor_registry):
        """测试获取所有 category"""
        init_default_factors()
        cats = default_registry.categories()
        assert cats == {"alpha", "risk", "enhance"}


class TestInitDefaultFactors:
    """默认因子初始化测试"""

    def test_init_default_factors(self, reset_factor_registry):
        """测试默认因子初始化"""
        init_default_factors()

        alpha_factors = default_registry.get_by_category("alpha")
        assert len(alpha_factors) == 5
        alpha_names = [f.name for f in alpha_factors]
        assert "roa" in alpha_names
        assert "book_to_price" in alpha_names
        assert "ebit_to_mv" in alpha_names
        assert "ocf_to_ev" in alpha_names
        assert "retained_earnings" in alpha_names

        risk_factors = default_registry.get_by_category("risk")
        assert len(risk_factors) == 2
        risk_names = [f.name for f in risk_factors]
        assert "vol_std_20d" in risk_names
        assert "turnover_rate_f" in risk_names

        enhance_factors = default_registry.get_by_category("enhance")
        assert len(enhance_factors) == 3
        enhance_names = [f.name for f in enhance_factors]
        assert "bbi_momentum" in enhance_names
        assert "price_pos_52w" in enhance_names
        assert "mom_20d" in enhance_names

    def test_init_to_custom_registry(self):
        """测试向自定义注册表初始化"""
        reg = FactorRegistry()
        init_default_factors(reg)
        assert len(reg.all()) == 10


class TestGetExpressions:
    """获取表达式测试"""

    def test_get_alpha_expressions(self, reset_factor_registry):
        """测试获取 Alpha 表达式"""
        init_default_factors()
        expressions = get_alpha_expressions()

        assert isinstance(expressions, dict)
        assert len(expressions) == 5
        assert "roa" in expressions
        assert "book_to_price" in expressions

    def test_get_risk_expressions(self, reset_factor_registry):
        """测试获取风控表达式"""
        init_default_factors()
        expressions = get_risk_expressions()

        assert isinstance(expressions, dict)
        assert len(expressions) == 2
        assert "vol_std_20d" in expressions
        assert "turnover_rate_f" in expressions

    def test_get_enhance_expressions(self, reset_factor_registry):
        """测试获取增强表达式"""
        init_default_factors()
        expressions = get_enhance_expressions()

        assert isinstance(expressions, dict)
        assert len(expressions) == 3
        assert "bbi_momentum" in expressions
        assert "price_pos_52w" in expressions
        assert "mom_20d" in expressions

    def test_get_all_expressions(self, reset_factor_registry):
        """测试获取所有 qlib source 表达式"""
        init_default_factors()
        fields, names = get_all_expressions()

        assert isinstance(fields, list)
        assert isinstance(names, list)
        assert len(fields) == 4
        assert len(names) == 4

        assert "risk_vol_std_20d" in names
        assert "enhance_bbi_momentum" in names
        assert "enhance_price_pos_52w" in names
        assert "enhance_mom_20d" in names
