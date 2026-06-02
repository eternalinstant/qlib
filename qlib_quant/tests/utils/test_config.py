"""
配置模块测试
"""

import pytest
import os
import sys
from datetime import date
from pathlib import Path
from unittest.mock import patch, mock_open

from config.config import ConfigManager, AppConfig, load_yaml, save_yaml, get_config_manager, CONFIG


class TestLoadYaml:
    """YAML 加载测试"""

    def test_load_yaml_success(self, tmp_path):
        """测试成功加载 YAML"""
        yaml_content = "key: value\nnumber: 42"
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(yaml_content)

        with patch("config.config.CONFIG_DIR", tmp_path):
            result = load_yaml("test.yaml")
            assert result == {"key": "value", "number": 42}

    def test_load_yaml_nonexistent(self, tmp_path):
        """测试加载不存在的 YAML"""
        with patch("config.config.CONFIG_DIR", tmp_path):
            result = load_yaml("nonexistent.yaml")
            assert result == {}


class TestConfigManager:
    """ConfigManager 类测试"""

    def test_config_manager_creation(self):
        """测试创建 ConfigManager"""
        manager = ConfigManager()
        assert manager is not None

    def test_config_manager_get(self):
        """测试获取配置"""
        manager = ConfigManager()
        w_alpha = manager.get("w_alpha")
        assert w_alpha == 0.20

    def test_config_manager_get_nested(self):
        """测试嵌套键获取"""
        manager = ConfigManager()
        alpha = manager.get("strategy.weights.alpha")
        assert alpha == 0.20

    def test_config_manager_get_default(self):
        """测试默认值"""
        manager = ConfigManager()
        result = manager.get("nonexistent.key", default="default_value")
        assert result == "default_value"

    def test_config_manager_get_config(self):
        """测试获取 AppConfig"""
        manager = ConfigManager()
        config = manager.get_config()
        assert isinstance(config, AppConfig)

    def test_config_manager_reload(self):
        """测试重新加载"""
        manager = ConfigManager()
        manager.reload()
        assert manager.get("w_alpha") == 0.20


class TestAppConfig:
    """AppConfig 类测试"""

    def test_app_config_creation(self):
        """测试创建 AppConfig"""
        config = AppConfig({"key": "value"})
        assert config.get("key") == "value"

    def test_app_config_get(self):
        """测试获取配置"""
        config = AppConfig({"w_alpha": 0.6})
        assert config.get("w_alpha") == 0.6

    def test_app_config_get_nested(self):
        """测试嵌套键获取"""
        config = AppConfig({"strategy": {"weights": {"alpha": 0.7}}})
        assert config.get("strategy.weights.alpha") == 0.7

    def test_app_config_get_default(self):
        """测试默认值"""
        config = AppConfig({})
        result = config.get("nonexistent", default="default")
        assert result == "default"

    def test_app_config_strategy_property(self):
        """测试 strategy 属性"""
        config = AppConfig({"strategy": {"key": "value"}})
        assert config.strategy == {"key": "value"}


class TestGlobalConfig:
    """全局配置测试"""

    def test_global_config_exists(self):
        """测试全局配置存在"""
        assert CONFIG is not None

    def test_global_config_get(self):
        """测试全局配置获取"""
        assert CONFIG.get("w_alpha") == 0.20
        assert CONFIG.get("topk") == 15

    def test_get_config_manager_singleton(self):
        """测试全局单例"""
        manager1 = get_config_manager()
        manager2 = get_config_manager()
        assert manager1 is manager2


class TestConfigBackwardCompat:
    """配置向后兼容性测试"""

    def test_weights_accessible(self):
        """测试权重配置可访问"""
        assert CONFIG.get("w_alpha") == 0.20
        assert CONFIG.get("w_risk") == 0.55
        assert CONFIG.get("w_enhance") == 0.25

    def test_selection_accessible(self):
        """测试选股配置可访问"""
        assert CONFIG.get("topk") == 15

    def test_period_accessible(self):
        """测试回测期间配置可访问"""
        assert CONFIG.get("start_date") == "2019-01-01"
        assert CONFIG.get("end_date") == date.today().isoformat()

    def test_trading_accessible(self):
        """测试交易配置可访问"""
        assert CONFIG.get("initial_capital") == 500000
        assert CONFIG.get("open_cost") == 0.0003
        assert CONFIG.get("close_cost") == 0.0013
