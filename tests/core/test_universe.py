"""
股票池过滤测试
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import core.universe as universe


def test_filter_st_instruments_by_date_with_history(tmp_path, monkeypatch):
    csv_path = tmp_path / "namechange.csv"
    pd.DataFrame(
        [
            {"ts_code": "000001.SZ", "name": "ST平安", "start_date": "2024-01-01", "end_date": "2024-06-30"},
            {"ts_code": "000002.SZ", "name": "万科A", "start_date": "2024-01-01", "end_date": "2024-12-31"},
        ]
    ).to_csv(csv_path, index=False)

    monkeypatch.setattr(universe, "_historical_st_intervals", None)
    monkeypatch.setattr(universe, "_historical_st_loaded", False)
    monkeypatch.setattr(universe, "_historical_st_by_date_cache", {})
    monkeypatch.setattr(universe, "_iter_namechange_paths", lambda: [csv_path])

    assert universe.is_st_on_date("SZ000001", "2024-03-01") is True
    assert universe.is_st_on_date("SZ000001", "2024-07-01") is False
    assert universe.filter_st_instruments_by_date(["SZ000001", "SZ000002"], "2024-03-01") == ["SZ000002"]


def test_filter_instruments_by_universe_with_history(tmp_path, monkeypatch):
    csv_path = tmp_path / "index_weight.csv"
    pd.DataFrame(
        [
            {"index_code": "000300.SH", "con_code": "000001.SZ", "trade_date": "20240131", "weight": 1.0},
            {"index_code": "000300.SH", "con_code": "000002.SZ", "trade_date": "20240131", "weight": 1.0},
            {"index_code": "000300.SH", "con_code": "000002.SZ", "trade_date": "20240229", "weight": 1.0},
            {"index_code": "000300.SH", "con_code": "000003.SZ", "trade_date": "20240229", "weight": 1.0},
        ]
    ).to_csv(csv_path, index=False)

    monkeypatch.setattr(universe, "_index_weight_df", None)
    monkeypatch.setattr(universe, "_index_constituents_as_of_cache", {})
    monkeypatch.setattr(universe, "_iter_index_weight_paths", lambda: [csv_path])

    assert universe.get_index_constituents_as_of("000300.SH", "2024-02-01") == ["SZ000001", "SZ000002"]
    assert universe.filter_instruments_by_universe(
        ["SZ000001", "SZ000002", "SZ000003"],
        "2024-03-01",
        universe="csi300",
    ) == ["SZ000002", "SZ000003"]


def test_filter_instruments_by_universe_accepts_lowercase_inputs(tmp_path, monkeypatch):
    csv_path = tmp_path / "index_weight.csv"
    pd.DataFrame(
        [
            {"index_code": "000300.SH", "con_code": "000001.SZ", "trade_date": "20240229", "weight": 1.0},
            {"index_code": "000300.SH", "con_code": "000002.SZ", "trade_date": "20240229", "weight": 1.0},
        ]
    ).to_csv(csv_path, index=False)

    monkeypatch.setattr(universe, "_index_weight_df", None)
    monkeypatch.setattr(universe, "_index_constituents_as_of_cache", {})
    monkeypatch.setattr(universe, "_iter_index_weight_paths", lambda: [csv_path])

    assert universe.filter_instruments_by_universe(
        ["sz000001", "sz000002", "sz000003"],
        "2024-03-01",
        universe="csi300",
    ) == ["sz000001", "sz000002"]


def test_filter_new_listed_instruments_uses_cached_blocked_set(monkeypatch):
    monkeypatch.setattr(universe, "_list_date_map", {
        "SZ000001": pd.Timestamp("2023-01-01"),
        "SZ000002": pd.Timestamp("2024-01-20"),
    })
    monkeypatch.setattr(universe, "_list_date_series", None)
    monkeypatch.setattr(universe, "_newly_listed_by_date_cache", {})

    assert universe.filter_new_listed_instruments(
        ["SZ000001", "SZ000002", "SZ000003"],
        "2024-02-01",
        min_days_listed=60,
    ) == ["SZ000001", "SZ000003"]


def test_filter_new_listed_instruments_accepts_lowercase_inputs(monkeypatch):
    monkeypatch.setattr(universe, "_list_date_map", {
        "SZ000001": pd.Timestamp("2023-01-01"),
        "SZ000002": pd.Timestamp("2024-01-20"),
    })
    monkeypatch.setattr(universe, "_list_date_series", None)
    monkeypatch.setattr(universe, "_newly_listed_by_date_cache", {})

    assert universe.filter_new_listed_instruments(
        ["sz000001", "sz000002", "sz000003"],
        "2024-02-01",
        min_days_listed=60,
    ) == ["sz000001", "sz000003"]


def test_filter_instruments_excludes_index_like_codes(monkeypatch):
    monkeypatch.setattr(universe, "_st_instruments", set())

    filtered = universe.filter_instruments(
        ["SH000300", "SZ399001", "SH880001", "SH600000", "SZ000001"],
        exclude_st=False,
    )

    assert filtered == ["SH600000", "SZ000001"]


def test_filter_instruments_excludes_lowercase_index_like_codes(monkeypatch):
    monkeypatch.setattr(universe, "_st_instruments", set())

    filtered = universe.filter_instruments(
        ["sh000300", "sz399001", "sh880001", "sh600000", "sz000001"],
        exclude_st=False,
    )

    assert filtered == ["sh600000", "sz000001"]

