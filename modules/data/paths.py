"""Backward-compat shim → common/paths.py（五层迁移）。

实体已下沉至 common/paths.py。此处 re-export 兼容旧 import 路径
（`from modules.data.paths import get_qlib_root` 等）。脚本迁移完成后可删除。
"""
from common.paths import (  # noqa: F401
    get_qlib_root,
    get_data_root,
    get_tushare_root,
    get_raw_root,
    get_selection_csv_path,
    get_selections_dir,
    get_cache_dir,
    get_results_dir,
    raw_data_root,
    raw_data_path_for_instrument,
)
