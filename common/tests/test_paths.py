"""TDD: common/paths.py — 统一路径解析（合并 modules/data/paths.py + common.py 的 raw_data_*）"""
from pathlib import Path
from unittest.mock import patch
from common.paths import (
    raw_data_root,
    raw_data_path_for_instrument,
    get_qlib_root,
    get_raw_root,
    get_data_root,
    get_tushare_root,
)


class _FakeCfg:
    """返回固定 qlib_data 路径的假 CONFIG"""
    def get(self, key, default=None):
        if "qlib_data" in key:
            return "/data/qlib_data/cn_data"
        return default


# ── raw_data_*（原 common.py）────────────────────────────────────────────────

def test_raw_data_path_for_instrument_lowercase_prefix():
    with patch("common.paths.raw_data_root", return_value=Path("/x/raw_data")):
        p = raw_data_path_for_instrument("SZ000001")
    assert p == Path("/x/raw_data/sz000001.parquet")


def test_raw_data_root_is_sibling_of_qlib_root():
    with patch("common.paths.CONFIG", _FakeCfg()):
        assert raw_data_root() == Path("/data/qlib_data/raw_data")


# ── get_*（原 modules/data/paths.py）─────────────────────────────────────────

def test_get_qlib_root_from_config():
    with patch("common.paths.CONFIG", _FakeCfg()):
        assert get_qlib_root() == Path("/data/qlib_data/cn_data")


def test_get_raw_root_sibling_of_qlib():
    with patch("common.paths.CONFIG", _FakeCfg()):
        assert get_raw_root() == Path("/data/qlib_data/raw_data")


def test_get_data_root_two_levels_up():
    with patch("common.paths.CONFIG", _FakeCfg()):
        assert get_data_root() == Path("/data")


def test_get_tushare_root():
    with patch("common.paths.CONFIG", _FakeCfg()):
        assert get_tushare_root() == Path("/data/tushare")
