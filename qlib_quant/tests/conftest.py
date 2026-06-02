"""
共享测试 fixtures
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests.fixtures.sample_data import (
    create_sample_factor_data,
    create_sample_market_data,
    create_sample_selections,
)


# ==================================
# 因子测试 fixtures
# ==================================

@pytest.fixture
def sample_factor_data():
    """样本因子数据"""
    return create_sample_factor_data()


@pytest.fixture
def empty_factor_data():
    """空因子数据"""
    return pd.DataFrame()


# ==================================
# 仓位测试 fixtures
# ==================================

@pytest.fixture
def sample_market_config():
    """样本市场配置"""
    from core.position import MarketConfig
    return MarketConfig()


@pytest.fixture
def mock_position_controller(sample_market_config):
    """Mock 的仓位控制器（不加载真实数据）"""
    from core.position import MarketPositionController

    controller = MarketPositionController.__new__(MarketPositionController)
    controller.config = sample_market_config

    # 使用样本市场数据
    close = create_sample_market_data(200)
    controller.csi300_close = close
    controller._compute_indicators()
    controller._prev_allocation = None

    return controller


# ==================================
# 选股测试 fixtures
# ==================================

@pytest.fixture
def sample_monthly_df():
    """样本月度数据"""
    return create_sample_factor_data(n_dates=3, n_stocks=50)


@pytest.fixture
def mock_selection_csv(tmp_path):
    """Mock 选股 CSV 文件"""
    df = create_sample_selections()
    csv_path = tmp_path / "monthly_selections.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


# ==================================
# 通用 fixtures
# ==================================

@pytest.fixture
def reset_factor_registry():
    """重置因子注册表"""
    from core.factors import default_registry, init_default_factors
    original = default_registry.all()
    default_registry.clear()
    yield
    default_registry.clear()
    for f in original.values():
        default_registry.register(f)


def make_pytest_wrapper(func):
    """将 check_* 函数包装为 test_* 供 pytest 发现，同时保留 CLI main() 调用能力。"""
    def _wrapper():
        func()

    _wrapper.__name__ = func.__name__.replace("check_", "test_", 1)
    _wrapper.__doc__ = func.__doc__
    return _wrapper
