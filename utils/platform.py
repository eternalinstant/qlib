"""Backward-compat shim → common/platform.py（五层迁移）。

实体已下沉至 common/platform.py。此处 re-export 兼容旧 import 路径
（`from utils.platform import resolve_path` 等），脚本迁移完成后可删除。
"""
from common.platform import (  # noqa: F401
    system_name,
    machine_name,
    is_macos,
    is_linux,
    is_windows,
    project_root,
    resolve_path,
    temp_dir,
    safe_cpu_count,
    runtime_profile,
)
