"""
统一路径解析层

所有数据路径从此模块的 lazy 函数推导，不在 import 时冻结。
配置键与 config/paths.yaml 保持一致。
"""

from pathlib import Path

from config.config import CONFIG


def get_qlib_root() -> Path:
    """Qlib provider 根目录 → .../data/qlib_data/cn_data"""
    raw = CONFIG.get("paths.data.qlib_data", CONFIG.get("qlib_data_path", ""))
    if raw:
        return Path(raw).expanduser()
    return Path(__file__).resolve().parent.parent.parent / "data" / "qlib_data" / "cn_data"


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
        return Path(raw).expanduser()
    return get_data_root() / "monthly_selections.csv"


def get_selections_dir() -> Path:
    """分层选股结果目录 → .../data/selections"""
    return get_data_root() / "selections"


def get_cache_dir() -> Path:
    """缓存目录（paths.data.cache）"""
    raw = CONFIG.get("paths.data.cache", "")
    if raw:
        return Path(raw).expanduser()
    return get_data_root() / "cache"


def get_results_dir() -> Path:
    """结果输出目录（output.results）"""
    raw = CONFIG.get("output.results", "")
    if raw:
        return Path(raw).expanduser()
    return Path(__file__).resolve().parent.parent.parent / "results"
