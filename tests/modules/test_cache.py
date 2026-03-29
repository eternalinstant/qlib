"""
回测缓存模块测试
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import pickle

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.backtest.cache import BacktestCache, get_cache


class TestBacktestCache:
    """测试回测结果缓存"""

    @pytest.fixture
    def cache_dir(self, tmp_path):
        """创建临时缓存目录"""
        return tmp_path / "cache"

    @pytest.fixture
    def cache(self, cache_dir):
        """创建缓存实例"""
        return BacktestCache(cache_dir=str(cache_dir))

    def test_cache_key_generation(self, cache):
        """测试缓存键生成"""
        key = cache._get_cache_key(
            strategy_name="test_strategy",
            engine="qlib",
            start_date="2024-01-01",
            end_date="2024-12-31",
            yaml_content="test content"
        )
        
        assert isinstance(key, str)
        assert "test_strategy" in key
        assert "qlib" in key

    def test_cache_save_and_get(self, cache):
        """测试缓存保存和获取"""
        result_data = {
            "daily_returns": [0.01, 0.02, -0.01],
            "portfolio_value": [1.0, 1.01, 1.0],
        }
        
        # 保存缓存
        cache.set(
            strategy_name="test",
            engine="qlib",
            start_date="2024-01-01",
            end_date="2024-12-31",
            yaml_content="test",
            result=result_data,
        )
        
        # 获取缓存
        cached = cache.get(
            strategy_name="test",
            engine="qlib",
            start_date="2024-01-01",
            end_date="2024-12-31",
            yaml_content="test",
        )
        
        assert cached is not None
        assert cached == result_data

    def test_cache_miss(self, cache):
        """测试缓存未命中"""
        result = cache.get(
            strategy_name="nonexistent",
            engine="qlib",
            start_date="2024-01-01",
            end_date="2024-12-31",
            yaml_content="test",
        )
        
        assert result is None

    def test_cache_different_yaml_different_key(self, cache):
        """测试不同 YAML 内容生成不同缓存键"""
        key1 = cache._get_cache_key("test", "qlib", "2024-01-01", "2024-12-31", "content1")
        key2 = cache._get_cache_key("test", "qlib", "2024-01-01", "2024-12-31", "content2")
        
        assert key1 != key2

    def test_cache_clear_all(self, cache, cache_dir):
        """测试清除所有缓存"""
        # 保存一些缓存
        cache.set("test1", "qlib", "2024-01-01", "2024-12-31", "content", {"data": 1})
        cache.set("test2", "qlib", "2024-01-01", "2024-12-31", "content", {"data": 2})
        
        assert len(list(cache_dir.glob("*.pkl"))) == 2
        
        # 清除所有
        cache.clear()
        
        assert len(list(cache_dir.glob("*.pkl"))) == 0

    def test_cache_clear_specific(self, cache, cache_dir):
        """测试清除指定策略的缓存"""
        # 保存一些缓存
        cache.set("test1", "qlib", "2024-01-01", "2024-12-31", "content", {"data": 1})
        cache.set("test2", "qlib", "2024-01-01", "2024-12-31", "content", {"data": 2})
        
        # 只清除 test1
        cache.clear("test1")
        
        # test1 应该被删除，test2 应该保留
        files = list(cache_dir.glob("*.pkl"))
        assert len(files) == 1

    def test_get_cache_singleton(self):
        """测试 get_cache 返回单例"""
        cache1 = get_cache()
        cache2 = get_cache()
        
        assert cache1 is cache2
