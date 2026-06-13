"""
统一路径解析层。

所有数据路径从此模块的 lazy 函数推导，不在 import 时冻结。
配置键与 common/configs/paths.yaml 保持一致。

公共库：合并自
- modules/data/paths.py（get_* 系列）
- modules/backtest/common.py（raw_data_root / raw_data_path_for_instrument）
两处历史用不同 config 键（paths.data.qlib_data vs paths.qlib_data），故保持各自独立。
"""
from pathlib import Path

from common.config import CONFIG
from common.platform import project_root, resolve_path


# ── qlib provider 路径族（原 modules/data/paths.py）──────────────────────────

def get_qlib_root() -> Path:
    """Qlib provider 根目录 → .../data/qlib_data/cn_data"""
    raw = CONFIG.get("paths.data.qlib_data", CONFIG.get("qlib_data_path", ""))
    if raw:
        return resolve_path(raw)
    return project_root() / "data" / "qlib_data" / "cn_data"


def get_data_root() -> Path:
    """数据总目录 → .../data"""
    return get_qlib_root().parent.parent


def get_tushare_root() -> Path:
    """Tushare 原始数据目录 → .../data/tushare"""
    return get_data_root() / "tushare"


def get_raw_root() -> Path:
    """Raw data 目录 → .../data/qlib_data/raw_data"""
    return get_qlib_root().parent / "raw_data"


def get_selection_csv_path() -> Path:
    """月度选股 CSV 文件路径（paths.data.selections）"""
    raw = CONFIG.get("paths.data.selections", "")
    if raw:
        return resolve_path(raw)
    return get_data_root() / "monthly_selections.csv"


def get_selections_dir() -> Path:
    """分层选股结果目录 → .../data/selections"""
    return get_data_root() / "selections"


def get_cache_dir() -> Path:
    """缓存目录（paths.data.cache）"""
    raw = CONFIG.get("paths.data.cache", "")
    if raw:
        return resolve_path(raw)
    return get_data_root() / "cache"


def get_results_dir() -> Path:
    """结果输出目录（output.results）"""
    raw = CONFIG.get("output.results", "")
    if raw:
        return resolve_path(raw)
    return project_root() / "results"


# ── 原始日线 raw_data 路径（原 modules/backtest/common.py）────────────────────

def raw_data_root() -> Path:
    qlib_root = Path(
        CONFIG.get("paths.qlib_data", "~/code/qlib/data/qlib_data/cn_data")
    ).expanduser()
    return qlib_root.parent / "raw_data"


def raw_data_path_for_instrument(instrument: str) -> Path:
    return raw_data_root() / f"{instrument[:2].lower()}{instrument[2:]}.parquet"
