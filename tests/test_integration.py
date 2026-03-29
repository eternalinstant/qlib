"""
集成测试：确保原有代码仍能正常工作
"""
import sys
sys.path.insert(0, "/Users/sxt/code/qlib")

def test_factors_import():
    """测试因子模块导入"""
    from core.factors import (
        FactorRegistry,
        get_alpha_expressions,
        get_risk_expressions,
        get_enhance_expressions,
    )

    alpha = get_alpha_expressions()
    risk = get_risk_expressions()
    enhance = get_enhance_expressions()

    assert isinstance(alpha, dict)
    assert isinstance(risk, dict)
    assert len(enhance) > 0


def test_selection_import():
    """测试选股模块导入"""
    from core.selection import compute_signal

    assert callable(compute_signal)


def test_position_import():
    """测试仓位模块导入"""
    from core.position import MarketPositionController, MarketConfig

    config = MarketConfig()
    assert config.ma_fast > 0
    assert config.ma_slow > 0


def test_config_backward_compat():
    """测试配置向后兼容性"""
    from config.config import CONFIG

    old_style_access = {
        "w_alpha": CONFIG.get("w_alpha"),
        "w_risk": CONFIG.get("w_risk"),
        "w_enhance": CONFIG.get("w_enhance"),
        "topk": CONFIG.get("topk"),
        "start_date": CONFIG.get("start_date"),
        "end_date": CONFIG.get("end_date"),
        "initial_capital": CONFIG.get("initial_capital"),
    }

    for k, v in old_style_access.items():
        assert v is not None, f"{k} should not be None"
