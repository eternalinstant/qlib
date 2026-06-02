"""
Qlib 初始化与安全数据加载
共享模块，供 qlib_engine / pybroker_engine / selection 等使用
"""

import os
import platform
import sys
from pathlib import Path

from config.config import CONFIG


_QLIB_PROVIDER_URI = None


def _qlib_runtime_overrides():
    """返回当前平台需要传给 qlib.init 的运行时覆盖项。"""
    if platform.system() != "Windows":
        return {}

    if os.environ.get("JOBLIB_START_METHOD") == "fork":
        os.environ["JOBLIB_START_METHOD"] = "spawn"

    return {
        "kernels": 1,
        "joblib_backend": "threading",
        "maxtasksperchild": None,
    }


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

    provider_uri = Path(CONFIG.get("paths.qlib_data", "~/Documents/qlib_quant/data/qlib_data/cn_data")).expanduser()
    if not provider_uri.exists():
        print(f"[ERROR] Qlib 数据目录不存在: {provider_uri}")
        sys.exit(1)

    provider_uri_str = str(provider_uri)
    runtime_overrides = _qlib_runtime_overrides()
    if _QLIB_PROVIDER_URI == provider_uri_str:
        _apply_runtime_overrides(runtime_overrides)
        return

    qlib.init(provider_uri=provider_uri_str, region=REG_CN, **runtime_overrides)

    try:
        from qlib.config import C

        if platform.system() == "Windows":
            C["kernels"] = 1
            C["joblib_backend"] = "threading"
            C["maxtasksperchild"] = None
    except Exception:
        pass

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
