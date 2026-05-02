"""
正式数据预检测试
"""

from pathlib import Path

import numpy as np
import pandas as pd


def _prepare_qlib_root(root: Path):
    (root / "calendars").mkdir(parents=True, exist_ok=True)
    (root / "instruments").mkdir(parents=True, exist_ok=True)
    (root / "calendars" / "day.txt").write_text("2019-01-02\n2026-03-20\n", encoding="utf-8")
    (root / "instruments" / "all.txt").write_text("sz000001\t2010-01-01\t2099-12-31\n", encoding="utf-8")
    pd.DataFrame(
        {
            "datetime": ["2026-03-20"],
            "instrument": ["sz000001"],
            "close": [10.0],
        }
    ).to_parquet(root / "factor_data.parquet", index=False)


def _prepare_tushare_root(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ts_code": ["000001.SZ"], "trade_date": ["20260320"], "close": [10.0]}).to_parquet(
        root / "daily_basic.parquet",
        index=False,
    )
    for name in ["income.parquet", "balancesheet.parquet", "cashflow.parquet", "fina_indicator.parquet"]:
        pd.DataFrame({"ts_code": ["000001.SZ"], "end_date": ["20251231"]}).to_parquet(root / name, index=False)
    pd.DataFrame({"ts_code": ["000300.SH"], "trade_date": ["20260320"], "close": [4000.0]}).to_parquet(
        root / "index_daily.parquet",
        index=False,
    )
    pd.DataFrame({"ts_code": ["000001.SZ"], "name": ["平安银行"], "industry": ["银行"]}).to_csv(
        root / "stock_basic.csv",
        index=False,
    )
    pd.DataFrame({"ts_code": ["000001.SZ"], "industry": ["银行"]}).to_csv(
        root / "stock_industry.csv",
        index=False,
    )


def test_run_data_precheck_requires_index_weight_and_namechange(monkeypatch, tmp_path):
    from modules.data import precheck

    qlib_root = tmp_path / "qlib"
    tushare_root = tmp_path / "tushare"
    _prepare_qlib_root(qlib_root)
    _prepare_tushare_root(tushare_root)
    (qlib_root / "calendars" / "day.txt").write_text(
        "2019-01-02\n2026-03-18\n2026-03-19\n2026-03-20\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(precheck, "_qlib_root", lambda: qlib_root)
    monkeypatch.setattr(precheck, "_tushare_root", lambda: tushare_root)
    monkeypatch.setattr(precheck, "_iter_index_weight_paths", lambda: [tushare_root / "index_weight.parquet"])
    monkeypatch.setattr(precheck, "_iter_namechange_paths", lambda: [tushare_root / "namechange.parquet"])
    monkeypatch.setattr(
        precheck,
        "_backtest_period",
        lambda: (pd.Timestamp("2019-01-01"), pd.Timestamp("2026-03-20")),
    )

    result = precheck.run_data_precheck(universe="csi300", require_st_history=True)

    assert result.ok is False
    assert any("index_weight" in err for err in result.errors)
    assert any("namechange" in err for err in result.errors)


def test_run_data_precheck_passes_with_history_files(monkeypatch, tmp_path):
    from modules.data import precheck

    qlib_root = tmp_path / "qlib"
    tushare_root = tmp_path / "tushare"
    _prepare_qlib_root(qlib_root)
    _prepare_tushare_root(tushare_root)

    pd.DataFrame(
        {
            "index_code": ["000300.SH", "000300.SH"],
            "con_code": ["000001.SZ", "000002.SZ"],
            "trade_date": ["20260302", "20260302"],
            "weight": [1.0, 1.2],
        }
    ).to_parquet(tushare_root / "index_weight.parquet", index=False)
    pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "name": ["ST平安"],
            "start_date": ["20180101"],
            "end_date": ["20190101"],
        }
    ).to_parquet(tushare_root / "namechange.parquet", index=False)

    monkeypatch.setattr(precheck, "_qlib_root", lambda: qlib_root)
    monkeypatch.setattr(precheck, "_tushare_root", lambda: tushare_root)
    monkeypatch.setattr(precheck, "_iter_index_weight_paths", lambda: [tushare_root / "index_weight.parquet"])
    monkeypatch.setattr(precheck, "_iter_namechange_paths", lambda: [tushare_root / "namechange.parquet"])
    monkeypatch.setattr(
        precheck,
        "_backtest_period",
        lambda: (pd.Timestamp("2019-01-01"), pd.Timestamp("2026-03-20")),
    )

    result = precheck.run_data_precheck(universe="csi300", require_st_history=True)

    assert result.ok is True
    assert result.errors == []
    assert "index_weight" in result.resolved_paths
    assert "namechange" in result.resolved_paths


def test_run_data_precheck_detects_provider_inconsistency(monkeypatch, tmp_path):
    from modules.data import precheck

    qlib_root = tmp_path / "qlib"
    tushare_root = tmp_path / "tushare"
    _prepare_qlib_root(qlib_root)
    _prepare_tushare_root(tushare_root)

    pd.DataFrame(
        {
            "index_code": ["000300.SH"],
            "con_code": ["000001.SZ"],
            "trade_date": ["20260320"],
        }
    ).to_parquet(tushare_root / "index_weight.parquet", index=False)
    pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "name": ["ST平安"],
            "start_date": ["20180101"],
            "end_date": ["20190101"],
        }
    ).to_parquet(tushare_root / "namechange.parquet", index=False)

    inst_dir = qlib_root / "features" / "sz000001"
    inst_dir.mkdir(parents=True, exist_ok=True)
    # calendar 只有 2 个交易日，但 close 写了 3 个值，end_idx 超界
    np.array([0.0, 10.0, 11.0, 12.0], dtype="<f4").tofile(inst_dir / "close.day.bin")
    np.array([0.0, 9.8, 10.8], dtype="<f4").tofile(inst_dir / "open.day.bin")
    np.array([0.0, 10.2, 11.2], dtype="<f4").tofile(inst_dir / "high.day.bin")
    np.array([0.0, 9.7, 10.7], dtype="<f4").tofile(inst_dir / "low.day.bin")
    np.array([0.0, 1000.0, 1200.0], dtype="<f4").tofile(inst_dir / "volume.day.bin")
    np.array([0.0, 10000.0, 12000.0], dtype="<f4").tofile(inst_dir / "amount.day.bin")

    monkeypatch.setattr(precheck, "_qlib_root", lambda: qlib_root)
    monkeypatch.setattr(precheck, "_tushare_root", lambda: tushare_root)
    monkeypatch.setattr(precheck, "_iter_index_weight_paths", lambda: [tushare_root / "index_weight.parquet"])
    monkeypatch.setattr(precheck, "_iter_namechange_paths", lambda: [tushare_root / "namechange.parquet"])
    monkeypatch.setattr(
        precheck,
        "_backtest_period",
        lambda: (pd.Timestamp("2019-01-01"), pd.Timestamp("2026-03-20")),
    )

    result = precheck.run_data_precheck(universe="csi300", require_st_history=True)

    assert result.ok is False
    assert any("Qlib provider 字段不一致" in err for err in result.errors)


def test_run_data_precheck_detects_compressed_close_coverage(monkeypatch, tmp_path):
    from modules.data import precheck

    qlib_root = tmp_path / "qlib"
    tushare_root = tmp_path / "tushare"
    _prepare_qlib_root(qlib_root)
    _prepare_tushare_root(tushare_root)

    pd.DataFrame(
        {
            "index_code": ["000300.SH"],
            "con_code": ["000001.SZ"],
            "trade_date": ["20260320"],
        }
    ).to_parquet(tushare_root / "index_weight.parquet", index=False)
    pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "name": ["ST平安"],
            "start_date": ["20180101"],
            "end_date": ["20190101"],
        }
    ).to_parquet(tushare_root / "namechange.parquet", index=False)

    raw_dir = tmp_path / "raw_data"
    raw_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-03-18"), pd.Timestamp("2026-03-20")],
            "open": [9.8, 10.8],
            "high": [10.2, 11.2],
            "low": [9.7, 10.7],
            "close": [10.0, 11.0],
            "volume": [1000.0, 1200.0],
            "amount": [10000.0, 12000.0],
        }
    ).to_parquet(raw_dir / "sz000001.parquet", index=False)

    inst_dir = qlib_root / "features" / "sz000001"
    inst_dir.mkdir(parents=True, exist_ok=True)
    np.array([1.0, 10.0, 11.0], dtype="<f4").tofile(inst_dir / "close.day.bin")
    np.array([1.0, 9.8, 10.8], dtype="<f4").tofile(inst_dir / "open.day.bin")
    np.array([1.0, 10.2, 11.2], dtype="<f4").tofile(inst_dir / "high.day.bin")
    np.array([1.0, 9.7, 10.7], dtype="<f4").tofile(inst_dir / "low.day.bin")
    np.array([1.0, 1000.0, 1200.0], dtype="<f4").tofile(inst_dir / "volume.day.bin")
    np.array([1.0, 10000.0, 12000.0], dtype="<f4").tofile(inst_dir / "amount.day.bin")

    monkeypatch.setattr(precheck, "_qlib_root", lambda: qlib_root)
    monkeypatch.setattr(precheck, "_tushare_root", lambda: tushare_root)
    monkeypatch.setattr(precheck, "_iter_index_weight_paths", lambda: [tushare_root / "index_weight.parquet"])
    monkeypatch.setattr(precheck, "_iter_namechange_paths", lambda: [tushare_root / "namechange.parquet"])
    monkeypatch.setattr(
        precheck,
        "_backtest_period",
        lambda: (pd.Timestamp("2019-01-01"), pd.Timestamp("2026-03-20")),
    )

    result = precheck.run_data_precheck(universe="csi300", require_st_history=True)

    assert result.ok is False
    assert any("Qlib provider 字段不一致" in err for err in result.errors)


def test_run_data_precheck_ignores_orphan_feature_dirs(monkeypatch, tmp_path):
    from modules.data import precheck

    qlib_root = tmp_path / "qlib"
    tushare_root = tmp_path / "tushare"
    _prepare_qlib_root(qlib_root)
    _prepare_tushare_root(tushare_root)

    pd.DataFrame(
        {
            "index_code": ["000300.SH"],
            "con_code": ["000001.SZ"],
            "trade_date": ["20260320"],
        }
    ).to_parquet(tushare_root / "index_weight.parquet", index=False)
    pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "name": ["ST平安"],
            "start_date": ["20180101"],
            "end_date": ["20190101"],
        }
    ).to_parquet(tushare_root / "namechange.parquet", index=False)

    valid_dir = qlib_root / "features" / "sz000001"
    valid_dir.mkdir(parents=True, exist_ok=True)
    np.array([0.0, 10.0, 11.0], dtype="<f4").tofile(valid_dir / "close.day.bin")
    np.array([0.0, 9.8, 10.8], dtype="<f4").tofile(valid_dir / "open.day.bin")
    np.array([0.0, 10.2, 11.2], dtype="<f4").tofile(valid_dir / "high.day.bin")
    np.array([0.0, 9.7, 10.7], dtype="<f4").tofile(valid_dir / "low.day.bin")
    np.array([0.0, 1000.0, 1200.0], dtype="<f4").tofile(valid_dir / "volume.day.bin")
    np.array([0.0, 10000.0, 12000.0], dtype="<f4").tofile(valid_dir / "amount.day.bin")

    orphan_dir = qlib_root / "features" / "sh600253"
    orphan_dir.mkdir(parents=True, exist_ok=True)
    np.array([2615.0, 6.0, 6.1], dtype="<f4").tofile(orphan_dir / "close.day.bin")
    np.array([0.0, 5.8, 5.9], dtype="<f4").tofile(orphan_dir / "open.day.bin")

    monkeypatch.setattr(precheck, "_qlib_root", lambda: qlib_root)
    monkeypatch.setattr(precheck, "_tushare_root", lambda: tushare_root)
    monkeypatch.setattr(precheck, "_iter_index_weight_paths", lambda: [tushare_root / "index_weight.parquet"])
    monkeypatch.setattr(precheck, "_iter_namechange_paths", lambda: [tushare_root / "namechange.parquet"])
    monkeypatch.setattr(
        precheck,
        "_backtest_period",
        lambda: (pd.Timestamp("2019-01-01"), pd.Timestamp("2026-03-20")),
    )

    result = precheck.run_data_precheck(universe="csi300", require_st_history=True)

    assert result.ok is True
    assert result.errors == []


def test_run_data_precheck_missing_calendar_does_not_crash(monkeypatch, tmp_path):
    from modules.data import precheck

    qlib_root = tmp_path / "qlib"
    tushare_root = tmp_path / "tushare"
    _prepare_tushare_root(tushare_root)

    (qlib_root / "instruments").mkdir(parents=True, exist_ok=True)
    (qlib_root / "features" / "sz000001").mkdir(parents=True, exist_ok=True)
    (qlib_root / "instruments" / "all.txt").write_text(
        "sz000001\t2010-01-01\t2099-12-31\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "datetime": ["2026-03-20"],
            "instrument": ["sz000001"],
            "close": [10.0],
        }
    ).to_parquet(qlib_root / "factor_data.parquet", index=False)
    np.array([0.0, 10.0], dtype="<f4").tofile(qlib_root / "features" / "sz000001" / "close.day.bin")

    monkeypatch.setattr(precheck, "_qlib_root", lambda: qlib_root)
    monkeypatch.setattr(precheck, "_tushare_root", lambda: tushare_root)
    monkeypatch.setattr(precheck, "_iter_index_weight_paths", lambda: [tushare_root / "index_weight.parquet"])
    monkeypatch.setattr(precheck, "_iter_namechange_paths", lambda: [tushare_root / "namechange.parquet"])
    monkeypatch.setattr(
        precheck,
        "_backtest_period",
        lambda: (pd.Timestamp("2019-01-01"), pd.Timestamp("2026-03-20")),
    )

    result = precheck.run_data_precheck(universe="all", require_st_history=False)

    assert result.ok is False
    assert any("calendars/day.txt" in err for err in result.errors)
