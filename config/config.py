"""Backward-compat shim → common/config.py（五层迁移）。

实体已下沉至 common/config.py。此处 re-export 兼容旧 import 路径
（`from config.config import CONFIG` 等，全仓 38 处）。CONFIG 为同一全局单例对象，
对它的 mutate（如 main._load_config）仍共享。脚本迁移完成后可删除。

注意：patch 实现细节请针对 common.config（如 `patch("common.config.CONFIG_DIR")`），
因为 load_yaml/save_yaml 实体在 common.config，从该命名空间解析 CONFIG_DIR。
"""
from common.config import (  # noqa: F401
    PROJECT_ROOT,
    CONFIG_DIR,
    load_yaml,
    save_yaml,
    AppConfig,
    ConfigManager,
    get_config_manager,
    load_config,
    config,
    CONFIG,
)
