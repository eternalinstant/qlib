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
    """安全加载因子数据。

    默认不改变输入股票池；如需临时跳过坏标的，显式设置
    QLIB_FILTER_BAD_INSTRUMENTS=1。
    """
    from qlib.data import D

    if not isinstance(instruments, list):
        inst_list = D.list_instruments(instruments, start_time=start_time, end_time=end_time)
        inst_list = list(inst_list.keys())
    else:
        inst_list = list(instruments)

    try:
        return D.features(inst_list, fields, start_time, end_time, freq)
    except ValueError as exc:
        if "broadcast" not in str(exc):
            raise

        if os.environ.get("QLIB_FILTER_BAD_INSTRUMENTS") != "1":
            raise ValueError(
                "Qlib D.features broadcast error. This usually means one or more "
                "instruments have inconsistent feature lengths. Refusing to change "
                "the training universe implicitly; fix the data or rerun with "
                "QLIB_FILTER_BAD_INSTRUMENTS=1 to explicitly drop bad instruments."
            ) from exc

        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"D.features broadcast error, filtering bad instruments: {exc}")

        clean = []
        bad = []
        for inst in inst_list:
            try:
                D.features([inst], fields, start_time, end_time, freq)
                clean.append(inst)
            except Exception:
                bad.append(inst)

        if bad:
            logger.warning(f"Filtered {len(bad)} bad instruments: {bad}")
        if not clean:
            raise

        return D.features(clean, fields, start_time, end_time, freq)
