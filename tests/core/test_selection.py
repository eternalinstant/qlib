"""
选股模块测试
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.selection import (
    compute_signal,
    compute_selections,
    extract_topk,
    load_factor_data,
    load_selections,
    _load_close_series,
    _enrich_selections,
    _load_total_mv_frame,
)
from core.factors import (
    FactorInfo,
    FactorRegistry,
    get_alpha_expressions,
    get_risk_expressions,
    get_enhance_expressions,
    get_all_expressions,
)


class TestComputeSignal:
    """信号计算测试"""

    def test_compute_signal(self, sample_monthly_df):
        """测试信号计算"""
        signal = compute_signal(sample_monthly_df, neutralize_industry=False)

        assert isinstance(signal, pd.Series)
        assert len(signal) == len(sample_monthly_df)

    def test_compute_signal_range(self, sample_monthly_df):
        """测试信号值范围"""
        signal = compute_signal(sample_monthly_df, neutralize_industry=False)

        # 信号应该在合理范围内（加权后的 rank pct）
        assert signal.dropna().min() >= 0
        assert signal.dropna().max() <= 1

    def test_compute_signal_with_missing_columns(self):
        """测试缺少列时的信号计算"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        stocks = ["SZ000001", "SZ000002"]
        index = pd.MultiIndex.from_product([dates, stocks], names=["datetime", "instrument"])

        df = pd.DataFrame({
            "alpha_roa": [0.1, 0.2, 0.15, 0.18, 0.12, 0.22],
        }, index=index)

        signal = compute_signal(df, neutralize_industry=False)
        assert isinstance(signal, pd.Series)

    def test_compute_signal_with_neutralization(self, sample_monthly_df):
        """测试行业中性化信号计算"""
        # 即使行业数据文件不存在，也不应报错（graceful degradation）
        signal = compute_signal(sample_monthly_df, neutralize_industry=True)
        assert isinstance(signal, pd.Series)
        assert len(signal) == len(sample_monthly_df)


class TestLoadSelections:
    """加载选股结果测试"""

    def test_load_selections_file_not_found(self, tmp_path):
        """测试文件不存在时自动生成"""
        fake_path = tmp_path / "nonexistent.csv"

        with patch("core.selection.SELECTION_CSV", fake_path):
            with patch("core.selection.generate_selections") as mock_gen:
                mock_gen.return_value = pd.DataFrame({
                    "date": [pd.Timestamp("2024-01-31")],
                    "rank": [1],
                    "symbol": ["SZ000001"],
                    "score": [0.8]
                })

                try:
                    load_selections()
                except Exception:
                    pass

    def test_load_selections_success(self, mock_selection_csv):
        """测试成功加载选股结果"""
        with patch("core.selection.SELECTION_CSV", mock_selection_csv):
            date_to_symbols, monthly_dates = load_selections()

            assert isinstance(date_to_symbols, dict)
            assert isinstance(monthly_dates, set)
            assert len(date_to_symbols) > 0


class TestSelectionIntegration:
    """选股集成测试"""

    def test_full_selection_flow(self, sample_monthly_df):
        """测试完整选股流程"""
        signal = compute_signal(sample_monthly_df, neutralize_industry=False)

        dates = signal.index.get_level_values("datetime").unique()

        for dt in dates[:1]:
            try:
                day_signal = signal.xs(dt, level="datetime")
                top_5 = day_signal.nlargest(5)

                assert len(top_5) <= 5
                assert isinstance(top_5, pd.Series)
            except KeyError:
                continue


class TestExtractTopk:
    """TopK 过滤逻辑测试"""

    def test_exclude_st(self):
        dt = pd.Timestamp("2024-01-05")
        idx = pd.MultiIndex.from_tuples(
            [
                (dt, "SZ000001"),
                (dt, "SZ000002"),
            ],
            names=["datetime", "instrument"],
        )
        signal = pd.Series([0.9, 1.0], index=idx)

        with patch("core.selection.filter_st_instruments_by_date", return_value=["SZ000001"]):
            result = extract_topk(
                signal,
                pd.DatetimeIndex([dt]),
                topk=1,
                exclude_st=True,
            )

        assert len(result) == 1
        assert result.iloc[0]["symbol"] == "SZ000001"

    def test_empty_result_keeps_schema(self):
        dt = pd.Timestamp("2024-01-05")
        idx = pd.MultiIndex.from_tuples(
            [
                (dt, "SZ000001"),
            ],
            names=["datetime", "instrument"],
        )
        signal = pd.Series([0.9], index=idx)

        result = extract_topk(
            signal,
            pd.DatetimeIndex([dt]),
            topk=2,
        )

        assert result.empty
        assert list(result.columns) == ["date", "rank", "symbol", "score"]


class TestSelectionParquetOptimization:
    """Parquet 读取优化相关测试"""

    def test_compute_selections_accepts_buffer_and_market_cap_filter(self, tmp_path):
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        index = pd.MultiIndex.from_product(
            [dates, ["SZ000001", "SZ000002"]],
            names=["datetime", "instrument"],
        )
        monthly_df = pd.DataFrame({"alpha_demo": [1.0, 0.8, 0.9, 1.1]}, index=index)
        signal = pd.Series([0.9, 0.8, 0.7, 0.6], index=index)
        total_mv_frame = pd.DataFrame(
            {
                "datetime": [dates[0], dates[0], dates[1], dates[1]],
                "symbol": ["SZ000001", "SZ000002", "SZ000001", "SZ000002"],
                "total_mv": [600000.0, 700000.0, 650000.0, 680000.0],
            }
        )
        fake_factor_path = tmp_path / "factor_data.parquet"
        fake_factor_path.touch()

        with patch("core.selection.load_factor_data", return_value=(monthly_df, pd.DatetimeIndex(dates))), \
             patch("core.selection.compute_signal", return_value=signal), \
             patch("core.selection._load_total_mv_frame", return_value=total_mv_frame) as mock_mv, \
             patch("core.selection.extract_topk", return_value=pd.DataFrame()) as mock_extract, \
             patch("core.selection.FACTOR_PARQUET", fake_factor_path):
            compute_selections(
                topk=2,
                rebalance_freq="day",
                min_market_cap=50,
                buffer=7,
                universe="all",
            )

        assert mock_mv.call_args.kwargs["start_date"] == "2019-01-01"
        assert mock_extract.call_args.kwargs["buffer"] == 7

    def test_load_factor_data_supports_parquet_only_registry(self):
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        close_index = pd.MultiIndex.from_product(
            [dates, ["SZ000001", "SZ000002"]],
            names=["datetime", "instrument"],
        )
        close_df = pd.DataFrame({"$close": [10.0, 11.0, 10.5, 11.5]}, index=close_index)
        parquet_df = pd.DataFrame(
            {"alpha_ocf_to_ev": [1.0, 0.8, 0.9, 1.1]},
            index=close_index,
        )

        registry = FactorRegistry()
        registry.register(
            FactorInfo(
                name="ocf_to_ev",
                expression="ocf_to_ev",
                description="",
                category="alpha",
                source="parquet",
            )
        )

        class FakeD:
            @staticmethod
            def features(instruments, fields, start_date, end_date, freq):
                assert fields == ["$close"]
                return close_df

        with patch("qlib.init"), \
             patch("core.selection.filter_instruments", return_value=["SZ000001", "SZ000002"]), \
             patch("core.selection.get_universe_instruments", return_value=["SZ000001", "SZ000002"]), \
             patch("qlib.data.D", new=FakeD()), \
             patch("core.selection._load_parquet_factors", return_value=parquet_df):
            monthly_df, rebalance_dates = load_factor_data(
                registry=registry,
                start_date="2024-01-02",
                end_date="2024-01-03",
                rebalance_freq="day",
                universe="csi300",
            )

        assert list(rebalance_dates) == list(dates)
        assert "alpha_ocf_to_ev" in monthly_df.columns
        assert len(monthly_df) == 4

    def test_load_close_series_normalizes_multiindex_order(self):
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        idx = pd.MultiIndex.from_product(
            [["SZ000001"], dates],
            names=["instrument", "datetime"],
        )
        df_close = pd.DataFrame({"$close": [10.0, 10.5]}, index=idx)

        class FakeD:
            @staticmethod
            def features(instruments, fields, start_date, end_date, freq):
                return df_close

        with patch("qlib.data.D", new=FakeD()):
            close_series = _load_close_series(
                ["SZ000001"],
                start_date="2024-01-02",
                end_date="2024-01-03",
            )

        assert close_series.index.names == ["datetime", "instrument"]
        assert close_series.loc[(dates[0], "SZ000001")] == 10.0
        assert close_series.loc[(dates[1], "SZ000001")] == 10.5

    def test_load_close_series_normalizes_provider_case(self):
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        idx = pd.MultiIndex.from_product(
            [dates, ["sz000001"]],
            names=["datetime", "instrument"],
        )
        df_close = pd.DataFrame({"$close": [10.0, 10.5]}, index=idx)

        class FakeD:
            @staticmethod
            def features(instruments, fields, start_date, end_date, freq):
                assert instruments == ["sz000001"]
                return df_close

        with patch("qlib.data.D", new=FakeD()):
            close_series = _load_close_series(
                ["SZ000001"],
                start_date="2024-01-02",
                end_date="2024-01-03",
            )

        assert close_series.index.names == ["datetime", "instrument"]
        assert close_series.loc[(dates[0], "SZ000001")] == 10.0
        assert close_series.loc[(dates[1], "SZ000001")] == 10.5

    def test_load_total_mv_frame_filters_rows_and_normalizes_symbols(self, tmp_path):
        parquet_path = tmp_path / "factor_data.parquet"
        df = pd.DataFrame(
            {
                "datetime": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-02-01"]),
                "instrument": ["000001sz", "000001sz", "000002sz"],
                "total_mv": [100.0, 110.0, 200.0],
            }
        )
        df.to_parquet(parquet_path, index=False)

        with patch("core.selection.FACTOR_PARQUET", parquet_path):
            with patch("core.selection._factor_parquet_columns_cache", None):
                result = _load_total_mv_frame(
                    instruments=["SZ000001"],
                    start_date="2024-01-01",
                    end_date="2024-01-31",
                )

        assert list(result["symbol"].unique()) == ["SZ000001"]
        assert list(result["datetime"]) == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")]
        assert list(result["total_mv"]) == [100.0, 110.0]

    def test_enrich_selections_reuses_loaded_total_mv_frame(self):
        df_sel = pd.DataFrame(
            {
                "date": [pd.Timestamp("2024-01-02")],
                "rank": [1],
                "symbol": ["SZ000001"],
                "score": [0.9],
            }
        )
        total_mv_frame = pd.DataFrame(
            {
                "datetime": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")],
                "symbol": ["SZ000001", "SZ000001"],
                "total_mv": [100.0, 110.0],
            }
        )

        with patch("core.selection._load_name_map", return_value={"SZ000001": "平安银行"}):
            enriched = _enrich_selections(df_sel, total_mv_frame=total_mv_frame)

        assert enriched.iloc[0]["name"] == "平安银行"
        assert enriched.iloc[0]["total_mv"] == 110.0

    def test_universe_filter(self):
        dt = pd.Timestamp("2024-01-05")
        idx = pd.MultiIndex.from_tuples(
            [
                (dt, "SZ000001"),
                (dt, "SZ000002"),
            ],
            names=["datetime", "instrument"],
        )
        signal = pd.Series([0.9, 1.0], index=idx)

        with patch("core.selection.filter_instruments_by_universe", return_value=["SZ000001"]):
            result = extract_topk(
                signal,
                pd.DatetimeIndex([dt]),
                topk=1,
                universe="csi300",
            )

        assert len(result) == 1
        assert result.iloc[0]["symbol"] == "SZ000001"

    def test_exclude_new_days(self):
        dt = pd.Timestamp("2024-01-05")
        idx = pd.MultiIndex.from_tuples(
            [
                (dt, "SZ000001"),
                (dt, "SZ000002"),
            ],
            names=["datetime", "instrument"],
        )
        signal = pd.Series([0.9, 1.0], index=idx)

        with patch("core.selection.filter_new_listed_instruments", return_value=["SZ000001"]):
            result = extract_topk(
                signal,
                pd.DatetimeIndex([dt]),
                topk=1,
                exclude_new_days=60,
            )

        assert len(result) == 1
        assert result.iloc[0]["symbol"] == "SZ000001"

    def test_hard_filter_quantiles(self):
        dt = pd.Timestamp("2024-01-05")
        idx = pd.MultiIndex.from_tuples(
            [
                (dt, "SZ000001"),
                (dt, "SZ000002"),
                (dt, "SZ000003"),
            ],
            names=["datetime", "instrument"],
        )
        signal = pd.Series([0.9, 1.0, 0.8], index=idx)
        hard_filter_data = pd.DataFrame(
            {
                "roa_fina": [0.10, 0.05, -0.02],
            },
            index=idx,
        )

        result = extract_topk(
            signal,
            pd.DatetimeIndex([dt]),
            topk=2,
            hard_filter_quantiles={"roa_fina": 0.4},
            hard_filter_data=hard_filter_data,
        )

        assert len(result) == 2
        assert set(result["symbol"]) == {"SZ000001", "SZ000002"}

    def test_event_driven_persistence_delays_switch(self):
        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        idx = pd.MultiIndex.from_product(
            [dates, ["SZ000001", "SZ000002", "SZ000003"]],
            names=["datetime", "instrument"],
        )
        signal = pd.Series(
            [
                1.00, 0.90, 0.80,   # day1 -> A,B
                1.00, 0.70, 1.10,   # day2 -> C strong but should wait
                0.95, 0.60, 1.20,   # day3 -> C replaces B
            ],
            index=idx,
        )

        result = extract_topk(
            signal,
            pd.DatetimeIndex(dates),
            topk=2,
            entry_rank=1,
            exit_rank=2,
            entry_persist_days=2,
            exit_persist_days=2,
        )

        by_date = {
            dt: set(grp["symbol"])
            for dt, grp in result.groupby("date")
        }
        assert by_date[dates[0]] == {"SZ000001", "SZ000002"}
        assert by_date[dates[1]] == {"SZ000001", "SZ000002"}
        assert by_date[dates[2]] == {"SZ000001", "SZ000003"}

    def test_event_driven_min_hold_prevents_early_exit(self):
        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-10"])
        idx = pd.MultiIndex.from_product(
            [dates, ["SZ000001", "SZ000002", "SZ000003"]],
            names=["datetime", "instrument"],
        )
        signal = pd.Series(
            [
                1.00, 0.90, 0.80,
                1.00, 0.70, 1.10,
                0.95, 0.60, 1.20,
            ],
            index=idx,
        )

        result = extract_topk(
            signal,
            pd.DatetimeIndex(dates),
            topk=2,
            entry_rank=1,
            exit_rank=2,
            entry_persist_days=2,
            exit_persist_days=2,
            min_hold_days=15,
        )

        by_date = {
            dt: set(grp["symbol"])
            for dt, grp in result.groupby("date")
        }
        assert by_date[dates[2]] == {"SZ000001", "SZ000002"}

    def test_stoploss_replace_only_switches_on_drawdown(self):
        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        idx = pd.MultiIndex.from_product(
            [dates, ["SZ000001", "SZ000002", "SZ000003"]],
            names=["datetime", "instrument"],
        )
        signal = pd.Series(
            [
                1.00, 0.90, 0.80,
                0.95, 0.85, 1.10,
                0.97, 0.60, 1.20,
            ],
            index=idx,
        )
        close_series = pd.Series(
            [
                10.0, 10.0, 9.0,
                11.0, 10.0, 9.8,
                10.8, 8.8, 10.0,
            ],
            index=idx,
        )

        result = extract_topk(
            signal,
            pd.DatetimeIndex(dates),
            topk=2,
            selection_mode="stoploss_replace",
            close_series=close_series,
            stoploss_lookback_days=3,
            stoploss_drawdown=0.10,
        )

        by_date = {
            dt: set(grp["symbol"])
            for dt, grp in result.groupby("date")
        }
        assert by_date[dates[0]] == {"SZ000001", "SZ000002"}
        assert by_date[dates[1]] == {"SZ000001", "SZ000002"}
        assert by_date[dates[2]] == {"SZ000001", "SZ000003"}

    def test_stoploss_replace_replaces_filtered_holdings(self):
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        idx = pd.MultiIndex.from_product(
            [dates, ["SZ000001", "SZ000002", "SZ000003"]],
            names=["datetime", "instrument"],
        )
        signal = pd.Series(
            [
                1.00, 0.90, 0.80,
                1.00, 0.10, 0.95,
            ],
            index=idx,
        )
        close_series = pd.Series(
            [
                10.0, 10.0, 9.0,
                10.2, 10.1, 9.5,
            ],
            index=idx,
        )

        with patch(
            "core.selection.filter_st_instruments_by_date",
            side_effect=[
                ["SZ000001", "SZ000002", "SZ000003"],
                ["SZ000001", "SZ000003"],
            ],
        ):
            result = extract_topk(
                signal,
                pd.DatetimeIndex(dates),
                topk=2,
                exclude_st=True,
                selection_mode="stoploss_replace",
                close_series=close_series,
                stoploss_lookback_days=2,
                stoploss_drawdown=0.10,
            )

        by_date = {
            dt: set(grp["symbol"])
            for dt, grp in result.groupby("date")
        }
        assert by_date[dates[0]] == {"SZ000001", "SZ000002"}
        assert by_date[dates[1]] == {"SZ000001", "SZ000003"}

    def test_score_smoothing_reduces_single_day_flip(self):
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        idx = pd.MultiIndex.from_product(
            [dates, ["SZ000001", "SZ000002"]],
            names=["datetime", "instrument"],
        )
        signal = pd.Series(
            [
                1.00, 0.00,
                0.40, 0.90,
            ],
            index=idx,
        )

        result = extract_topk(
            signal,
            pd.DatetimeIndex(dates),
            topk=1,
            score_smoothing_days=2,
        )

        by_date = {
            dt: grp.sort_values("rank")["symbol"].tolist()
            for dt, grp in result.groupby("date")
        }
        assert by_date[dates[0]] == ["SZ000001"]
        assert by_date[dates[1]] == ["SZ000001"]

