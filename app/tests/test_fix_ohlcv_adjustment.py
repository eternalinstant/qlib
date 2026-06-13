import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.fix_ohlcv_adjustment as fix_ohlcv_adjustment


def test_fix_all_preserves_existing_history_outside_raw_overlap(tmp_path, monkeypatch):
    qlib_dir = tmp_path / "qlib_data" / "cn_data"
    raw_dir = tmp_path / "qlib_data" / "raw_data"
    tushare_dir = tmp_path / "tushare"
    features_dir = qlib_dir / "features" / "sz000001"
    cal_dir = qlib_dir / "calendars"

    features_dir.mkdir(parents=True)
    cal_dir.mkdir(parents=True)
    raw_dir.mkdir(parents=True)
    tushare_dir.mkdir(parents=True)

    (cal_dir / "day.txt").write_text("2024-01-02\n2024-01-03\n2024-01-04\n", encoding="utf-8")
    pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ", "000001.SZ"],
            "trade_date": ["20240102", "20240103", "20240104"],
            "close": [10.0, 11.0, 12.0],
        }
    ).to_parquet(tushare_dir / "daily_basic.parquet", index=False)
    pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-04")],
            "open": [10.5, 11.5],
            "high": [11.2, 12.2],
            "low": [10.1, 11.1],
            "close": [11.0, 12.0],
            "volume": [1100.0, 1200.0],
            "amount": [11000.0, 12000.0],
        }
    ).to_parquet(raw_dir / "sz000001.parquet", index=False)

    np.array([0.0, 9.5, 9.8, 10.1], dtype="<f4").tofile(features_dir / "close.day.bin")
    np.array([0.0, 9.0, 9.3, 9.6], dtype="<f4").tofile(features_dir / "open.day.bin")
    np.array([0.0, 10.2, 10.4, 10.6], dtype="<f4").tofile(features_dir / "high.day.bin")
    np.array([0.0, 8.8, 9.1, 9.4], dtype="<f4").tofile(features_dir / "low.day.bin")
    np.array([0.0, 1000.0, 1001.0, 1002.0], dtype="<f4").tofile(features_dir / "volume.day.bin")
    np.array([0.0, 10000.0, 10010.0, 10020.0], dtype="<f4").tofile(features_dir / "amount.day.bin")

    monkeypatch.setattr(fix_ohlcv_adjustment, "QLIB_DIR", qlib_dir)
    monkeypatch.setattr(fix_ohlcv_adjustment, "RAW_DIR", raw_dir)
    monkeypatch.setattr(fix_ohlcv_adjustment, "TUSHARE_DIR", tushare_dir)

    rebuilt = fix_ohlcv_adjustment.fix_all()

    assert rebuilt == 1

    close_bin = np.fromfile(features_dir / "close.day.bin", dtype="<f4")
    open_bin = np.fromfile(features_dir / "open.day.bin", dtype="<f4")
    volume_bin = np.fromfile(features_dir / "volume.day.bin", dtype="<f4")

    assert int(close_bin[0]) == 0
    assert close_bin[1:] == pytest.approx([9.5, 11.0, 12.0])
    assert int(open_bin[0]) == 0
    assert open_bin[1:] == pytest.approx([9.0, 10.5, 11.5])
    assert int(volume_bin[0]) == 0
    assert volume_bin[1:] == pytest.approx([1000.0, 1100.0, 1200.0])
