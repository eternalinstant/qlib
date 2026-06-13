"""TDD: common/config.py — 全局配置（从 config/config.py 下沉到公共库）"""
from pathlib import Path
from unittest.mock import patch
from common.config import (
    ConfigManager,
    AppConfig,
    load_yaml,
    save_yaml,
    get_config_manager,
    load_config,
    CONFIG,
    CONFIG_DIR,
    PROJECT_ROOT,
)


def test_config_dir_points_to_common_configs():
    """层内聚后 CONFIG_DIR 指向 <repo>/common/configs（共享 YAML：paths/trading/strategy）"""
    assert CONFIG_DIR.name == "configs"
    assert CONFIG_DIR.parent.name == "common"
    assert CONFIG_DIR.exists()  # src-layout: PROJECT_ROOT=src/，configs 在 src/common/configs


def test_load_yaml_reads_from_config_dir(tmp_path):
    (tmp_path / "test.yaml").write_text("key: value\nnumber: 42")
    with patch("common.config.CONFIG_DIR", tmp_path):
        assert load_yaml("test.yaml") == {"key": "value", "number": 42}


def test_load_yaml_missing_returns_empty(tmp_path):
    with patch("common.config.CONFIG_DIR", tmp_path):
        assert load_yaml("nope.yaml") == {}


def test_appconfig_nested_get():
    cfg = AppConfig({"a": {"b": {"c": 7}}})
    assert cfg.get("a.b.c") == 7
    assert cfg.get("a.b.x", "default") == "default"


def test_config_manager_get_config_returns_appconfig():
    mgr = ConfigManager()
    assert isinstance(mgr.get_config(), AppConfig)


def test_global_config_singleton_is_appconfig():
    assert isinstance(CONFIG, AppConfig)
    assert get_config_manager() is get_config_manager()
