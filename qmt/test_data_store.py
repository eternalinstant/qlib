"""data_store 离线单测 — 不连网络，验证去重合并/缺口/覆盖率逻辑

用法: C:\\Python312\\python.exe test_data_store.py
"""

import os
import tempfile
import shutil

import pandas as pd

import data_config
import data_store


def _df(dates, close):
    """构造 xtquant 风格日线：index=日期串"""
    return pd.DataFrame({"open": close, "close": close}, index=[str(d) for d in dates])


def test_write_daily_dedup_merge():
    rows = data_store.write_daily("TEST.SH", _df(["20260101", "20260102"], [10, 11]))
    assert rows == 2, rows
    # 重叠日 20260102 带订正 + 新日 20260103
    rows = data_store.write_daily("TEST.SH", _df(["20260102", "20260103"], [99, 12]))
    assert rows == 3, rows                      # 去重后只增 1 行
    df = data_store.read_daily("TEST.SH")
    assert df["date"].tolist() == ["20260101", "20260102", "20260103"]
    # keep='last'：20260102 取订正值 99
    val = float(df[df["date"] == "20260102"]["close"].iloc[0])
    assert val == 99, val
    assert data_store.last_date("TEST.SH") == "20260103"
    assert data_store.first_date("TEST.SH") == "20260101"
    print("[OK] write_daily 去重合并 + 订正保留")


def test_write_daily_empty():
    assert data_store.write_daily("EMPTY.SH", None) == 0
    assert data_store.read_daily("EMPTY.SH") is None
    print("[OK] write_daily 空输入")


def test_find_gaps():
    cal = ["20260101", "20260102", "20260103", "20260104", "20260105"]
    stored = ["20260101", "20260103", "20260105"]
    assert data_store.find_gaps(stored, cal) == ["20260102", "20260104"]
    assert data_store.find_gaps([], cal) == []          # 无数据不报缺口
    # 区间外的日历日不算缺口
    assert data_store.find_gaps(["20260102", "20260103"], cal) == []
    print("[OK] find_gaps")


def test_coverage_summary():
    manifest = {
        "generated_at": "2026-05-27 10:00:00",
        "daily": {
            "600000.SH": {"first": "20200101", "last": "20260527", "rows": 100},
            "000001.SZ": {"first": "20200101", "last": "20260520", "rows": 90},
        },
    }
    universe = ["600000.SH", "000001.SZ", "600001.SH"]   # 第三只缺失
    s = data_store.coverage_summary(manifest, universe, latest_trading_day="20260527")
    assert s["universe_size"] == 3
    assert s["stored"] == 2
    assert s["missing"] == ["600001.SH"], s["missing"]
    assert s["stale"] == ["000001.SZ"], s["stale"]        # last 20260520 < 20260527
    assert s["total_rows"] == 190
    print("[OK] coverage_summary")


def main():
    tmp = tempfile.mkdtemp(prefix="qmt_ds_test_")
    orig = data_config.DATA_DIR
    data_config.DATA_DIR = tmp
    try:
        data_config.ensure_dirs()
        test_write_daily_dedup_merge()
        test_write_daily_empty()
        test_find_gaps()
        test_coverage_summary()
        print("\n全部通过")
    finally:
        data_config.DATA_DIR = orig
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
