"""
Qlib 初始化与安全数据加载
共享模块，供 qlib_engine / pybroker_engine / selection 等使用
"""

import os
import sys

from common.platform import is_macos, is_windows


_QLIB_PROVIDER_URI = None


def _qlib_runtime_overrides():
    """返回当前平台需要传给 qlib.init 的运行时覆盖项。

    macOS/Windows 的 multiprocessing 默认使用 spawn（而非 fork），
    joblib 的 loky 后端在 spawn 模式下子进程会触发 "bootstrapping phase" RuntimeError。
    解决方法：强制使用 threading 后端 + 单进程内核。
    Linux 默认使用 fork，joblib multiprocessing 正常，无需覆盖。
    """
    if is_macos():
        # macOS spawn 模式下 joblib multiprocessing 会崩溃，退化为 threading + 单核
        return {
            "kernels": 1,
            "joblib_backend": "threading",
            "maxtasksperchild": None,
        }

    if is_windows():
        if os.environ.get("JOBLIB_START_METHOD") == "fork":
            os.environ["JOBLIB_START_METHOD"] = "spawn"
        return {
            "kernels": 1,
            "joblib_backend": "threading",
            "maxtasksperchild": None,
        }

    # Linux: fork 默认正常，不覆盖，保留 qlib 的多核 multiprocessing
    return {}


def _apply_runtime_overrides(overrides):
    if not overrides:
        return

    try:
        from qlib.config import C

        for key, value in overrides.items():
            C[key] = value
    except Exception:
        pass


def init_qlib():
    """初始化 Qlib"""
    global _QLIB_PROVIDER_URI

    import qlib
    from qlib.config import REG_CN

    # 单一真源：provider 路径统一由 modules/data/paths 推导（项目根相对，跨平台）
    from common.paths import get_qlib_root

    provider_uri = get_qlib_root()
    if not provider_uri.exists():
        print(f"[ERROR] Qlib 数据目录不存在: {provider_uri}")
        sys.exit(1)

    provider_uri_str = str(provider_uri)
    runtime_overrides = _qlib_runtime_overrides()
    if _QLIB_PROVIDER_URI == provider_uri_str:
        _apply_runtime_overrides(runtime_overrides)
        return

    qlib.init(provider_uri=provider_uri_str, region=REG_CN, **runtime_overrides)

    _QLIB_PROVIDER_URI = provider_uri_str
    print("[OK] Qlib 初始化成功")


def load_features_safe(instruments, fields, start_time, end_time, freq="day"):
    """安全加载因子数据"""
    from qlib.data import D

    if not isinstance(instruments, list):
        inst_list = D.list_instruments(instruments, start_time=start_time, end_time=end_time)
        inst_list = list(inst_list.keys())
    else:
        inst_list = list(instruments)

    return D.features(inst_list, fields, start_time, end_time, freq)
