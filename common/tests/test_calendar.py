"""TDD: common/calendar.py — 交易日历加载（从引擎层 common.py 下沉到公共库）"""
import pandas as pd
from unittest.mock import patch
from common.calendar import load_trade_calendar


def test_load_trade_calendar_filters_by_range(tmp_path):
    cal_dir = tmp_path / "calendars"
    cal_dir.mkdir()
    (cal_dir / "day.txt").write_text(
        "\n".join(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"])
    )
    # resolve_path 指向 tmp_path 作为 qlib_root；用唯一日期避免 lru_cache 撞已有缓存
    with patch("common.calendar.resolve_path", return_value=tmp_path):
        cal = load_trade_calendar("2024-01-03", "2024-01-05")
    assert isinstance(cal, pd.DatetimeIndex)
    assert list(cal) == [
        pd.Timestamp("2024-01-03"),
        pd.Timestamp("2024-01-04"),
        pd.Timestamp("2024-01-05"),
    ]


def test_load_trade_calendar_empty_when_out_of_range(tmp_path):
    cal_dir = tmp_path / "calendars"
    cal_dir.mkdir()
    (cal_dir / "day.txt").write_text("2024-02-01\n2024-02-02")
    with patch("common.calendar.resolve_path", return_value=tmp_path):
        cal = load_trade_calendar("2030-01-01", "2030-12-31")
    assert len(cal) == 0
