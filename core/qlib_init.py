"""
Qlib 初始化与安全数据加载
共享模块，供 qlib_engine / pybroker_engine / selection 等使用
"""

import os
import sys
from pathlib import Path

from config.config import CONFIG


_QLIB_PROVIDER_URI = None


def init_qlib():
    """初始化 Qlib"""
    global _QLIB_PROVIDER_URI

    import qlib
    from qlib.config import REG_CN

    os.environ["JOBLIB_START_METHOD"] = "fork"

    provider_uri = Path(CONFIG.get("paths.qlib_data", "~/code/qlib/data/qlib_data/cn_data")).expanduser()
    if not provider_uri.exists():
        print(f"[ERROR] Qlib 数据目录不存在: {provider_uri}")
        sys.exit(1)

    provider_uri_str = str(provider_uri)
    if _QLIB_PROVIDER_URI == provider_uri_str:
        return

    qlib.init(provider_uri=provider_uri_str, region=REG_CN)

    try:
        from qlib.config import C
        C.n_jobs = 1
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
