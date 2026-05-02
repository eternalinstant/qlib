"""
测试 DataUpdater 整合功能

测试覆盖：
1. 检查是否需要更新（本地 vs 远程最新交易日）
2. 下载增量数据（mock Tushare API）
3. 合并数据到 parquet
4. 整合更新流程
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from types import SimpleNamespace


class TestDataUpdaterCheckUpdate:
    """测试更新检查逻辑"""

    def test_check_update_needed_when_stale(self, tmp_path):
        """本地数据落后于远程，需要更新"""
        from modules.data.updater import DataUpdater

        # 创建模拟的日历文件（本地最新是 2026-02-26）
        cal_dir = tmp_path / "calendars"
        cal_dir.mkdir(parents=True)
        cal_file = cal_dir / "day.txt"
        cal_file.write_text("2026-02-25\n2026-02-26\n")

        updater = DataUpdater(qlib_data_path=str(tmp_path))

        # Mock get_remote_latest_date 返回更晚的日期
        with patch.object(updater, 'get_remote_latest_date', return_value=datetime(2026, 3, 1)):
            assert updater.check_update_needed() is True

    def test_check_update_not_needed_when_current(self, tmp_path):
        """本地数据已是最新，不需要更新"""
        from modules.data.updater import DataUpdater

        cal_dir = tmp_path / "calendars"
        cal_dir.mkdir(parents=True)
        cal_file = cal_dir / "day.txt"
        # 本地最新是今天
        today = datetime.now().strftime("%Y-%m-%d")
        cal_file.write_text(f"2026-02-26\n{today}\n")

        updater = DataUpdater(qlib_data_path=str(tmp_path))

        with patch.object(updater, 'get_remote_latest_date', return_value=datetime.now()):
            assert updater.check_update_needed() is False


class TestDataUpdaterRemoteDate:
    """测试获取远程最新日期"""

    def test_get_remote_latest_date_success(self, tmp_path):
        """成功获取远程最新交易日"""
        from modules.data.updater import DataUpdater

        updater = DataUpdater(qlib_data_path=str(tmp_path))

        # Mock Tushare API
        mock_pro = Mock()
        mock_pro.trade_cal.return_value = pd.DataFrame({
            'cal_date': ['20260228', '20260301', '20260302'],
            'is_open': ['1', '1', '0']  # 3月2日非交易日
        })

        with patch('modules.data.updater.get_tushare_pro', return_value=mock_pro):
            result = updater.get_remote_latest_date()
            # 应返回最后一个开市日 2026-02-28（假设今天是3月1日）
            assert result is not None

    def test_get_remote_latest_date_no_api(self, tmp_path):
        """无 Tushare API 时返回 None"""
        from modules.data.updater import DataUpdater

        updater = DataUpdater(qlib_data_path=str(tmp_path))

        with patch('modules.data.updater.get_tushare_pro', return_value=None):
            result = updater.get_remote_latest_date()
            assert result is None

    def test_get_remote_latest_date_retries_transient_error(self, tmp_path):
        """trade_cal 短暂失败时应等待重试。"""
        from modules.data.updater import DataUpdater

        updater = DataUpdater(qlib_data_path=str(tmp_path))

        mock_pro = Mock()
        mock_pro.trade_cal.side_effect = [
            Exception("temporary network error"),
            pd.DataFrame({"cal_date": ["20260302"], "is_open": ["1"]}),
        ]

        with patch("modules.data.updater.get_tushare_pro", return_value=mock_pro), \
             patch("modules.data.updater.time.sleep", return_value=None):
            result = updater.get_remote_latest_date()

        assert result == datetime(2026, 3, 2)
        assert mock_pro.trade_cal.call_count == 2


class TestDataUpdaterDownload:
    """测试数据下载逻辑"""

    def test_download_daily_basic_incremental(self, tmp_path):
        """增量下载每日基础数据"""
        from modules.data.updater import DataUpdater

        # 创建已有数据
        existing_data = pd.DataFrame({
            'ts_code': ['000001.SZ', '000001.SZ'],
            'trade_date': ['20260226', '20260227'],
            'close': [10.0, 10.5],
            'pe_ttm': [8.0, 8.2]
        })
        output_path = tmp_path / "daily_basic.parquet"
        existing_data.to_parquet(output_path, index=False)

        updater = DataUpdater(qlib_data_path=str(tmp_path))
        updater.tushare_dir = tmp_path

        # Mock 新数据
        new_data = pd.DataFrame({
            'ts_code': ['000001.SZ', '000002.SZ'],
            'trade_date': ['20260228', '20260228'],
            'close': [11.0, 20.0],
            'pe_ttm': [8.5, 15.0]
        })

        mock_pro = Mock()
        mock_pro.daily_basic.return_value = new_data

        with patch('modules.data.updater.get_tushare_pro', return_value=mock_pro):
            result = updater.download_daily_basic()

        assert result is True

        # 验证合并后的数据
        combined = pd.read_parquet(output_path)
        assert len(combined) == 4  # 2 + 2
        assert '20260228' in combined['trade_date'].values

    def test_download_daily_basic_no_new_data(self, tmp_path):
        """无新数据时跳过"""
        from modules.data.updater import DataUpdater

        output_path = tmp_path / "daily_basic.parquet"
        existing_data = pd.DataFrame({
            'ts_code': ['000001.SZ'],
            'trade_date': ['20260228'],
            'close': [10.0]
        })
        existing_data.to_parquet(output_path, index=False)

        updater = DataUpdater(qlib_data_path=str(tmp_path))
        updater.tushare_dir = tmp_path

        mock_pro = Mock()
        mock_pro.daily_basic.return_value = pd.DataFrame()  # 空数据

        with patch('modules.data.updater.get_tushare_pro', return_value=mock_pro):
            result = updater.download_daily_basic()

        # 无新数据也算成功
        assert result is True

    def test_download_daily_basic_incremental_month_boundary(self, tmp_path):
        """跨月增量下载时起始日期应为下一个自然日。"""
        from modules.data.updater import DataUpdater

        output_path = tmp_path / "daily_basic.parquet"
        pd.DataFrame({
            "ts_code": ["000001.SZ"],
            "trade_date": ["20260131"],
            "close": [10.0],
        }).to_parquet(output_path, index=False)

        updater = DataUpdater(qlib_data_path=str(tmp_path))
        updater.tushare_dir = tmp_path

        mock_pro = Mock()
        mock_pro.daily_basic.return_value = pd.DataFrame()

        with patch("modules.data.updater.get_tushare_pro", return_value=mock_pro):
            result = updater.download_daily_basic()

        assert result is True
        assert mock_pro.daily_basic.call_args.kwargs["start_date"] == "20260201"

    def test_download_daily_basic_bootstrap_uses_full_history_start(self, tmp_path):
        """首次 bootstrap 时应从全量历史起点下载 daily_basic。"""
        from modules.data.updater import BOOTSTRAP_MARKET_START, DataUpdater

        updater = DataUpdater(qlib_data_path=str(tmp_path))
        updater.tushare_dir = tmp_path

        mock_pro = Mock()
        mock_pro.daily_basic.return_value = pd.DataFrame()

        with patch("modules.data.updater.get_tushare_pro", return_value=mock_pro):
            result = updater.download_daily_basic(start_date=BOOTSTRAP_MARKET_START)

        assert result is True
        assert mock_pro.daily_basic.call_args.kwargs["start_date"] == BOOTSTRAP_MARKET_START

    def test_download_daily_basic_bootstrap_resume_honors_explicit_start(self, tmp_path):
        """bootstrap 续跑时，显式 start_date 不应被已有 daily_basic 覆盖。"""
        from modules.data.updater import BOOTSTRAP_MARKET_START, DataUpdater

        pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "trade_date": ["20260228"],
                "close": [10.0],
            }
        ).to_parquet(tmp_path / "daily_basic.parquet", index=False)

        updater = DataUpdater(qlib_data_path=str(tmp_path))
        updater.tushare_dir = tmp_path

        mock_pro = Mock()
        mock_pro.daily_basic.return_value = pd.DataFrame()

        with patch("modules.data.updater.get_tushare_pro", return_value=mock_pro):
            result = updater.download_daily_basic(start_date=BOOTSTRAP_MARKET_START)

        assert result is True
        assert mock_pro.daily_basic.call_args.kwargs["start_date"] == BOOTSTRAP_MARKET_START

    def test_download_handles_api_error(self, tmp_path):
        """API 错误时优雅降级"""
        from modules.data.updater import DataUpdater

        updater = DataUpdater(qlib_data_path=str(tmp_path))
        updater.tushare_dir = tmp_path

        mock_pro = Mock()
        mock_pro.daily_basic.side_effect = Exception("API Error")

        with patch('modules.data.updater.get_tushare_pro', return_value=mock_pro):
            result = updater.download_daily_basic()

        assert result is False

    def test_download_daily_basic_retries_transient_error(self, tmp_path):
        """daily_basic 短暂失败时应等待重试，而不是直接留下缺口。"""
        from modules.data.updater import DataUpdater

        updater = DataUpdater(qlib_data_path=str(tmp_path))
        updater.tushare_dir = tmp_path

        mock_pro = Mock()
        mock_pro.daily_basic.side_effect = [
            Exception("temporary network error"),
            pd.DataFrame(
                {
                    "ts_code": ["000001.SZ"],
                    "trade_date": ["20260228"],
                    "close": [11.0],
                    "pe_ttm": [8.5],
                }
            ),
        ]

        with patch("modules.data.updater.get_tushare_pro", return_value=mock_pro), \
             patch("modules.data.updater.time.sleep", return_value=None):
            result = updater.download_daily_basic()

        assert result is True
        assert mock_pro.daily_basic.call_count == 2
        combined = pd.read_parquet(tmp_path / "daily_basic.parquet")
        assert combined["trade_date"].tolist() == ["20260228"]

    def test_download_adj_factor_rewinds_recent_window(self, tmp_path):
        """adj_factor 增量更新应回补最近窗口，避免尾部缺口永久保留。"""
        from modules.data.updater import DataUpdater

        pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "trade_date": ["20260228"],
                "adj_factor": [1.0],
            }
        ).to_parquet(tmp_path / "adj_factor.parquet", index=False)

        updater = DataUpdater(qlib_data_path=str(tmp_path))
        updater.tushare_dir = tmp_path

        mock_pro = Mock()
        mock_pro.adj_factor.return_value = pd.DataFrame()

        with patch("modules.data.updater.get_tushare_pro", return_value=mock_pro), \
             patch("modules.data.updater.time.sleep", return_value=None):
            result = updater.download_adj_factor()

        assert result is True
        assert mock_pro.adj_factor.call_args_list[0].kwargs["start_date"] == "20260128"

    def test_update_raw_data_quotes_incremental(self, tmp_path):
        """按交易日增量更新 raw_data。"""
        from modules.data.updater import DataUpdater

        updater = DataUpdater(qlib_data_path=str(tmp_path))
        updater.tushare_dir = tmp_path
        updater.raw_data_dir = tmp_path / "raw_data"
        updater.raw_data_dir.mkdir(parents=True)

        pd.DataFrame({"trade_date": ["20260227", "20260228"]}).to_parquet(
            tmp_path / "daily_basic.parquet",
            index=False,
        )
        pd.DataFrame(
            {
                "date": [pd.Timestamp("2026-02-27")],
                "open": [10.0],
                "high": [10.5],
                "low": [9.9],
                "close": [10.2],
                "volume": [1000.0],
                "amount": [10000.0],
                "symbol": ["600000.SH"],
            }
        ).to_parquet(updater.raw_data_dir / "sh600000.parquet", index=False)

        mock_pro = Mock()
        mock_pro.daily.return_value = pd.DataFrame(
            {
                "ts_code": ["600000.SH", "000001.SZ"],
                "trade_date": ["20260228", "20260228"],
                "open": [10.3, 12.0],
                "high": [10.6, 12.5],
                "low": [10.1, 11.8],
                "close": [10.4, 12.2],
                "vol": [1200.0, 800.0],
                "amount": [12000.0, 9500.0],
            }
        )

        with patch("modules.data.updater.get_tushare_pro", return_value=mock_pro), \
             patch("modules.data.updater.time.sleep", return_value=None):
            result = updater.update_raw_data_quotes()

        assert result is True
        sh = pd.read_parquet(updater.raw_data_dir / "sh600000.parquet")
        sz = pd.read_parquet(updater.raw_data_dir / "sz000001.parquet")
        assert pd.to_datetime(sh["date"]).max() == pd.Timestamp("2026-02-28")
        assert len(sz) == 1
        assert sz.iloc[0]["symbol"] == "000001.SZ"

    def test_update_raw_data_quotes_bootstrap_backfills_full_history(self, tmp_path):
        """首次 bootstrap 没有 raw_data 文件时，应按 daily_basic 全历史回补。"""
        from modules.data.updater import DataUpdater

        updater = DataUpdater(qlib_data_path=str(tmp_path))
        updater.tushare_dir = tmp_path
        updater.raw_data_dir = tmp_path / "raw_data"
        updater.raw_data_dir.mkdir(parents=True)

        pd.DataFrame({"trade_date": ["20260102", "20260331"]}).to_parquet(
            tmp_path / "daily_basic.parquet",
            index=False,
        )

        mock_pro = Mock()
        mock_pro.daily.side_effect = [
            pd.DataFrame(
                {
                    "ts_code": ["600000.SH"],
                    "trade_date": ["20260102"],
                    "open": [10.0],
                    "high": [10.5],
                    "low": [9.9],
                    "close": [10.2],
                    "vol": [1000.0],
                    "amount": [10000.0],
                }
            ),
            pd.DataFrame(
                {
                    "ts_code": ["600000.SH"],
                    "trade_date": ["20260331"],
                    "open": [11.0],
                    "high": [11.5],
                    "low": [10.8],
                    "close": [11.2],
                    "vol": [1200.0],
                    "amount": [12000.0],
                }
            ),
        ]

        with patch("modules.data.updater.get_tushare_pro", return_value=mock_pro), \
             patch("modules.data.updater.time.sleep", return_value=None):
            result = updater.update_raw_data_quotes()

        assert result is True
        assert mock_pro.daily.call_count == 2
        assert mock_pro.daily.call_args_list[0].kwargs["trade_date"] == "20260102"
        assert mock_pro.daily.call_args_list[1].kwargs["trade_date"] == "20260331"

    def test_download_index_daily_incremental_month_boundary(self, tmp_path):
        """跨月增量下载 index_daily 时起始日期应为下一个自然日。"""
        from modules.data.updater import DataUpdater

        output_path = tmp_path / "index_daily.parquet"
        pd.DataFrame({
            "ts_code": ["000300.SH"],
            "trade_date": ["20260228"],
            "close": [3900.0],
        }).to_parquet(output_path, index=False)

        updater = DataUpdater(qlib_data_path=str(tmp_path))
        updater.tushare_dir = tmp_path

        mock_pro = Mock()
        mock_pro.index_daily.return_value = pd.DataFrame()

        with patch("modules.data.updater.get_tushare_pro", return_value=mock_pro):
            result = updater.download_index_daily()

        assert result is True
        assert mock_pro.index_daily.call_args.kwargs["start_date"] == "20260301"

    def test_bootstrap_raw_data_for_instruments(self, tmp_path):
        """为缺失标的补齐历史 raw_data。"""
        from modules.data.updater import DataUpdater

        updater = DataUpdater(qlib_data_path=str(tmp_path))
        updater.raw_data_dir = tmp_path / "raw_data"
        updater.raw_data_dir.mkdir(parents=True)

        mock_pro = Mock()
        mock_pro.daily.return_value = pd.DataFrame(
            {
                "ts_code": ["688036.SH", "688036.SH"],
                "trade_date": ["20260305", "20260306"],
                "open": [30.0, 31.0],
                "high": [30.5, 31.5],
                "low": [29.8, 30.7],
                "close": [30.2, 31.2],
                "vol": [100.0, 120.0],
                "amount": [3000.0, 3600.0],
            }
        )

        with patch("modules.data.updater.get_tushare_pro", return_value=mock_pro), \
             patch("modules.data.updater.time.sleep", return_value=None):
            result = updater.bootstrap_raw_data_for_instruments(["SH688036"], start_date="20260301", end_date="20260310")

        assert result is True
        path = updater.raw_data_dir / "sh688036.parquet"
        assert path.exists()
        df = pd.read_parquet(path)
        assert len(df) == 2
        assert df.iloc[0]["symbol"] == "688036.SH"

    def test_ensure_provider_structure_uses_each_raw_span(self, tmp_path):
        """all.txt 应记录每个 instrument 自己的 raw_data 时间区间。"""
        from modules.data.updater import DataUpdater

        qlib_root = tmp_path / "qlib_data" / "cn_data"
        cal_dir = qlib_root / "calendars"
        cal_dir.mkdir(parents=True)
        (cal_dir / "day.txt").write_text("2026-01-02\n2026-01-05\n2026-01-06\n2026-01-07\n")

        updater = DataUpdater(qlib_data_path=str(qlib_root))
        updater.raw_data_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(
            {
                "date": [pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-06")],
                "close": [10.0, 11.0],
            }
        ).to_parquet(updater.raw_data_dir / "sh600000.parquet", index=False)
        pd.DataFrame(
            {
                "date": [pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-07")],
                "close": [20.0, 21.0],
            }
        ).to_parquet(updater.raw_data_dir / "sz000001.parquet", index=False)

        count = updater._ensure_provider_structure()

        all_txt = (updater.qlib_data_path / "instruments" / "all.txt").read_text().splitlines()
        assert count == 2
        assert "sh600000\t2026-01-02\t2026-01-06" in all_txt
        assert "sz000001\t2026-01-05\t2026-01-07" in all_txt


class TestDataUpdaterIntegration:
    """测试整合后的更新流程"""

    def test_update_daily_full_flow(self, tmp_path):
        """完整更新流程：检查 → 下载 → 选股"""
        from modules.data.updater import DataUpdater

        # 设置目录结构
        cal_dir = tmp_path / "calendars"
        cal_dir.mkdir(parents=True)
        cal_file = cal_dir / "day.txt"
        cal_file.write_text("2026-02-26\n")

        updater = DataUpdater(qlib_data_path=str(tmp_path))
        updater.tushare_dir = tmp_path

        # Mock 所有外部依赖
        ok_precheck = SimpleNamespace(ok=True, errors=[])
        with patch.object(updater, 'check_update_needed', return_value=True), \
             patch('modules.data.updater.run_data_precheck', return_value=ok_precheck), \
             patch.object(updater, 'download_daily_basic', return_value=True), \
             patch.object(updater, 'download_stock_basic', return_value=True), \
             patch.object(updater, 'download_financial_data', return_value=True), \
             patch.object(updater, 'update_raw_data_quotes', return_value=True), \
             patch.object(updater, 'download_index_daily', return_value=True), \
             patch.object(updater, 'download_index_weight', return_value=True), \
             patch.object(updater, 'download_namechange', return_value=True), \
             patch.object(updater, 'convert_to_qlib', return_value=True), \
             patch.object(updater, 'regenerate_selections', return_value=True):

            result = updater.update_daily()

        assert result['success'] is True
        assert result['data_updated'] is True
        assert result['raw_data_updated'] is True
        assert result['reference_updated'] is True
        assert result['converted'] is True
        assert result['precheck_ok'] is True
        assert result['selections_updated'] is True

    def test_update_daily_skips_when_not_needed(self, tmp_path):
        """不需要更新时跳过"""
        from modules.data.updater import DataUpdater

        cal_dir = tmp_path / "calendars"
        cal_dir.mkdir(parents=True)
        cal_file = cal_dir / "day.txt"
        cal_file.write_text("2026-02-28\n")

        updater = DataUpdater(qlib_data_path=str(tmp_path))

        ok_precheck = SimpleNamespace(ok=True, errors=[])
        with patch.object(updater, 'check_update_needed', return_value=False), \
             patch('modules.data.updater.run_data_precheck', return_value=ok_precheck):
            result = updater.update_daily()

        assert result['success'] is True
        assert result['message'] == "数据已是最新"

    def test_update_daily_partial_failure(self, tmp_path):
        """部分失败时的处理"""
        from modules.data.updater import DataUpdater

        cal_dir = tmp_path / "calendars"
        cal_dir.mkdir(parents=True)
        cal_file = cal_dir / "day.txt"
        cal_file.write_text("2026-02-26\n")

        updater = DataUpdater(qlib_data_path=str(tmp_path))
        updater.tushare_dir = tmp_path

        ok_precheck = SimpleNamespace(ok=True, errors=[])
        with patch.object(updater, 'check_update_needed', return_value=True), \
             patch('modules.data.updater.run_data_precheck', return_value=ok_precheck), \
             patch.object(updater, 'download_daily_basic', return_value=False), \
             patch.object(updater, 'download_stock_basic', return_value=True), \
             patch.object(updater, 'download_financial_data', return_value=False), \
             patch.object(updater, 'update_raw_data_quotes', return_value=False), \
             patch.object(updater, 'download_index_daily', return_value=False), \
             patch.object(updater, 'download_index_weight', return_value=False), \
             patch.object(updater, 'download_namechange', return_value=False), \
             patch.object(updater, 'convert_to_qlib', return_value=False), \
             patch.object(updater, 'regenerate_selections', return_value=True):

            result = updater.update_daily()

        assert result['success'] is True
        assert result['data_updated'] is False
        assert result['precheck_ok'] is True
        assert result['selections_updated'] is True


class TestDataUpdaterConvert:
    """测试 Tushare → Qlib 转换和日历更新"""

    def test_convert_to_qlib_calls_converter(self, tmp_path):
        """转换步骤调用 TushareToQlibConverter"""
        from modules.data.updater import DataUpdater

        updater = DataUpdater(qlib_data_path=str(tmp_path))
        updater.tushare_dir = tmp_path

        # 创建模拟的 daily_basic.parquet（用于日历更新）
        daily = pd.DataFrame({
            'ts_code': ['000001.SZ', '000001.SZ', '000002.SZ'],
            'trade_date': ['20260226', '20260227', '20260227'],
            'close': [10.0, 10.5, 20.0],
        })
        daily.to_parquet(tmp_path / "daily_basic.parquet", index=False)

        mock_converter = Mock()
        mock_df = pd.DataFrame({'instrument': ['000001sz'], 'datetime': [datetime(2026, 2, 27)]})
        mock_converter.convert.return_value = mock_df
        mock_converter.repair_price_provider.return_value = {}
        mock_converter.update_close_bins.return_value = 0
        mock_converter.update_ohlcv_bins.return_value = {}

        with patch('modules.data.updater.TushareToQlibConverter', return_value=mock_converter):
            result = updater.convert_to_qlib()

        assert result is True
        mock_converter.convert.assert_called_once()
        mock_converter.save.assert_called_once()

    def test_convert_to_qlib_updates_calendar(self, tmp_path):
        """转换后更新日历文件"""
        from modules.data.updater import DataUpdater

        # 创建已有日历
        cal_dir = tmp_path / "calendars"
        cal_dir.mkdir(parents=True)
        cal_file = cal_dir / "day.txt"
        cal_file.write_text("2026-02-25\n2026-02-26\n")

        updater = DataUpdater(qlib_data_path=str(tmp_path))
        updater.tushare_dir = tmp_path

        # 创建 daily_basic 包含新日期
        daily = pd.DataFrame({
            'ts_code': ['000001.SZ', '000001.SZ', '000001.SZ'],
            'trade_date': ['20260225', '20260226', '20260227'],
        })
        daily.to_parquet(tmp_path / "daily_basic.parquet", index=False)

        mock_converter = Mock()
        mock_converter.convert.return_value = pd.DataFrame({'a': [1]})
        mock_converter.repair_price_provider.return_value = {}
        mock_converter.update_close_bins.return_value = 0
        mock_converter.update_ohlcv_bins.return_value = {}

        with patch('modules.data.updater.TushareToQlibConverter', return_value=mock_converter):
            updater.convert_to_qlib()

        # 检查日历已更新
        updated_cal = cal_file.read_text().strip().split('\n')
        assert '2026-02-27' in updated_cal
        assert len(updated_cal) >= 3  # 25, 26, 27

    def test_convert_to_qlib_bootstraps_provider_structure(self, tmp_path):
        """首次转换时应自动创建 provider 目录骨架并初始化前复权 bin。"""
        from modules.data.updater import DataUpdater

        updater = DataUpdater(qlib_data_path=str(tmp_path / "cn_data"))
        updater.tushare_dir = tmp_path / "tushare"
        updater.tushare_dir.mkdir(parents=True)
        updater.raw_data_dir = tmp_path / "raw_data"
        updater.raw_data_dir.mkdir(parents=True)

        pd.DataFrame(
            {
                "trade_date": ["20260102", "20260105"],
                "ts_code": ["600000.SH", "600000.SH"],
            }
        ).to_parquet(updater.tushare_dir / "daily_basic.parquet", index=False)
        pd.DataFrame(
            {
                "date": [pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-05")],
                "open": [10.0, 10.2],
                "high": [10.5, 10.7],
                "low": [9.9, 10.1],
                "close": [10.3, 10.4],
                "volume": [1000.0, 1100.0],
                "amount": [10000.0, 11000.0],
                "symbol": ["600000.SH", "600000.SH"],
            }
        ).to_parquet(updater.raw_data_dir / "sh600000.parquet", index=False)

        adjusted = pd.DataFrame(
            {
                "date": [pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-05")],
                "open": [10.0, 10.2],
                "high": [10.5, 10.7],
                "low": [9.9, 10.1],
                "close": [10.3, 10.4],
                "volume": [1000.0, 1100.0],
                "amount": [10000.0, 11000.0],
            }
        )

        mock_converter = Mock()
        mock_converter.convert.return_value = pd.DataFrame({"instrument": ["sh600000"]})
        mock_converter.compute_forward_adjusted_prices.return_value = {"sh600000": adjusted}
        mock_converter.write_adjusted_bins.return_value = 1
        mock_converter.repair_price_provider.return_value = {}
        mock_converter.update_close_bins.return_value = 0
        mock_converter.update_ohlcv_bins.return_value = {}

        with patch("modules.data.updater.TushareToQlibConverter", return_value=mock_converter):
            result = updater.convert_to_qlib()

        assert result is True
        assert (updater.qlib_data_path / "features" / "sh600000").exists()
        assert (updater.qlib_data_path / "instruments" / "all.txt").exists()
        assert mock_converter.compute_forward_adjusted_prices.called
        assert mock_converter.write_adjusted_bins.called

    def test_convert_to_qlib_handles_converter_failure(self, tmp_path):
        """转换失败时返回 False"""
        from modules.data.updater import DataUpdater

        updater = DataUpdater(qlib_data_path=str(tmp_path))
        updater.tushare_dir = tmp_path

        mock_converter = Mock()
        mock_converter.convert.return_value = None

        with patch('modules.data.updater.TushareToQlibConverter', return_value=mock_converter):
            result = updater.convert_to_qlib()

        assert result is False

    def test_update_daily_includes_convert_step(self, tmp_path):
        """完整流程包含转换步骤"""
        from modules.data.updater import DataUpdater

        cal_dir = tmp_path / "calendars"
        cal_dir.mkdir(parents=True)
        (cal_dir / "day.txt").write_text("2026-02-26\n")

        updater = DataUpdater(qlib_data_path=str(tmp_path))
        updater.tushare_dir = tmp_path

        ok_precheck = SimpleNamespace(ok=True, errors=[])
        with patch.object(updater, 'check_update_needed', return_value=True), \
             patch('modules.data.updater.run_data_precheck', return_value=ok_precheck), \
             patch.object(updater, 'download_daily_basic', return_value=True), \
             patch.object(updater, 'download_stock_basic', return_value=True), \
             patch.object(updater, 'download_financial_data', return_value=True), \
             patch.object(updater, 'update_raw_data_quotes', return_value=True), \
             patch.object(updater, 'download_index_daily', return_value=True), \
             patch.object(updater, 'download_index_weight', return_value=True), \
             patch.object(updater, 'download_namechange', return_value=True), \
             patch.object(updater, 'convert_to_qlib', return_value=True) as mock_convert, \
             patch.object(updater, 'regenerate_selections', return_value=True):

            result = updater.update_daily()

        # 确认转换步骤被调用
        mock_convert.assert_called_once()
        assert result['success'] is True
        assert result['converted'] is True

    def test_update_daily_bootstrap_forces_full_history_download(self, tmp_path):
        """首次 bootstrap 时，即使 check_update_needed=False 也应走全量历史初始化。"""
        from modules.data.updater import BOOTSTRAP_MARKET_START, DataUpdater

        updater = DataUpdater(qlib_data_path=str(tmp_path))
        updater.tushare_dir = tmp_path

        initial_precheck = SimpleNamespace(ok=False, errors=["缺少文件: factor_data.parquet"])
        ok_precheck = SimpleNamespace(ok=True, errors=[])

        with patch.object(updater, "_needs_bootstrap", return_value=True), \
             patch.object(updater, "check_update_needed", return_value=False), \
             patch("modules.data.updater.run_data_precheck", side_effect=[initial_precheck, ok_precheck]), \
             patch.object(updater, "download_daily_basic", return_value=True) as mock_daily_basic, \
             patch.object(updater, "download_stock_basic", return_value=True), \
             patch.object(updater, "download_financial_data", return_value=True), \
             patch.object(updater, "update_raw_data_quotes", return_value=True) as mock_raw_data, \
             patch.object(updater, "download_adj_factor", return_value=True), \
             patch.object(updater, "download_index_daily", return_value=True), \
             patch.object(updater, "download_index_weight", return_value=True), \
             patch.object(updater, "download_namechange", return_value=True), \
             patch.object(updater, "convert_to_qlib", return_value=True), \
             patch.object(updater, "regenerate_selections", return_value=True):

            result = updater.update_daily()

        assert result["success"] is True
        assert mock_daily_basic.call_args.kwargs["start_date"] == BOOTSTRAP_MARKET_START
        assert mock_raw_data.call_args.kwargs["start_date"] == BOOTSTRAP_MARKET_START

    def test_update_daily_fails_fast_when_tushare_unavailable(self, tmp_path):
        """需要更新但 Tushare 不可用时，应直接报清晰错误。"""
        from modules.data.updater import DataUpdater

        updater = DataUpdater(qlib_data_path=str(tmp_path))
        updater.tushare_dir = tmp_path

        initial_precheck = SimpleNamespace(ok=False, errors=["缺少文件: factor_data.parquet"])

        with patch.object(updater, "_needs_bootstrap", return_value=True), \
             patch.object(updater, "check_update_needed", return_value=False), \
             patch("modules.data.updater.run_data_precheck", return_value=initial_precheck), \
             patch("modules.data.updater.get_tushare_pro", return_value=None):

            result = updater.update_daily()

        assert result["success"] is False
        assert "Tushare API 不可用" in result["message"]

    def test_update_daily_repairs_provider_when_market_data_is_current(self, tmp_path):
        """仅 provider 脏时也应触发转换修复。"""
        from modules.data.updater import DataUpdater

        cal_dir = tmp_path / "calendars"
        cal_dir.mkdir(parents=True)
        (cal_dir / "day.txt").write_text("2026-03-20\n")

        updater = DataUpdater(qlib_data_path=str(tmp_path))
        updater.tushare_dir = tmp_path

        provider_bad = SimpleNamespace(ok=False, errors=["Qlib provider 字段不一致"])
        ok_precheck = SimpleNamespace(ok=True, errors=[])

        with patch.object(updater, 'check_update_needed', return_value=False), \
             patch('modules.data.updater.run_data_precheck', side_effect=[provider_bad, ok_precheck]), \
             patch.object(updater, 'download_stock_basic', return_value=True), \
             patch.object(updater, 'download_index_weight', return_value=False), \
             patch.object(updater, 'download_namechange', return_value=False), \
             patch.object(updater, 'convert_to_qlib', return_value=True) as mock_convert, \
             patch.object(updater, 'regenerate_selections', return_value=True):

            result = updater.update_daily()

        mock_convert.assert_called_once()
        assert result['success'] is True
        assert result['converted'] is True

    def test_update_daily_repairs_missing_history_when_market_data_is_current(self, tmp_path):
        """行情已是最新，但缺历史成分/ST 数据时仍应补数据并通过预检。"""
        from modules.data.updater import DataUpdater

        cal_dir = tmp_path / "calendars"
        cal_dir.mkdir(parents=True)
        (cal_dir / "day.txt").write_text("2026-03-20\n")

        updater = DataUpdater(qlib_data_path=str(tmp_path))
        updater.tushare_dir = tmp_path

        missing_precheck = SimpleNamespace(
            ok=False,
            errors=["缺少历史指数成分文件 index_weight.parquet/csv（csi300 股票池必需）"],
        )
        ok_precheck = SimpleNamespace(ok=True, errors=[])

        with patch.object(updater, 'check_update_needed', return_value=False), \
             patch('modules.data.updater.run_data_precheck', side_effect=[missing_precheck, ok_precheck]), \
             patch.object(updater, 'download_stock_basic', return_value=True), \
             patch.object(updater, 'download_index_weight', return_value=True), \
             patch.object(updater, 'download_namechange', return_value=True), \
             patch.object(updater, 'regenerate_selections', return_value=True):
            result = updater.update_daily()

        assert result['success'] is True
        assert result['data_updated'] is False
        assert result['reference_updated'] is True
        assert result['precheck_ok'] is True
        assert result['selections_updated'] is True

    def test_repair_price_provider_rebuilds_aligned_bins(self, tmp_path):
        """provider 修复应按交易日历补 NaN 对齐字段。"""
        from modules.data.tushare_to_qlib import TushareToQlibConverter

        qlib_root = tmp_path / "cn_data"
        features_dir = qlib_root / "features" / "sz000001"
        features_dir.mkdir(parents=True)
        cal_dir = qlib_root / "calendars"
        cal_dir.mkdir(parents=True)
        (cal_dir / "day.txt").write_text("2026-01-02\n2026-01-05\n2026-01-06\n")

        raw_dir = tmp_path / "raw_data"
        raw_dir.mkdir(parents=True)
        pd.DataFrame(
            {
                "date": [pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-06")],
                "open": [9.0, 11.0],
                "high": [10.2, 12.2],
                "low": [8.8, 10.8],
                "close": [10.0, 12.0],
                "volume": [1000.0, 1200.0],
                "amount": [10000.0, 13000.0],
            }
        ).to_parquet(raw_dir / "sz000001.parquet", index=False)

        np.array([0.0, 10.0, 11.0, 12.0], dtype="<f4").tofile(features_dir / "close.day.bin")
        np.array([0.0, 9.0], dtype="<f4").tofile(features_dir / "open.day.bin")
        np.array([0.0, 10.2], dtype="<f4").tofile(features_dir / "high.day.bin")
        np.array([0.0, 8.8], dtype="<f4").tofile(features_dir / "low.day.bin")
        np.array([0.0, 1000.0], dtype="<f4").tofile(features_dir / "volume.day.bin")
        np.array([0.0, 10000.0], dtype="<f4").tofile(features_dir / "amount.day.bin")

        converter = TushareToQlibConverter(tushare_dir=str(tmp_path), qlib_dir=str(qlib_root))
        stats = converter.repair_price_provider()

        assert stats["repaired_instruments"] == 1

        open_bin = np.fromfile(features_dir / "open.day.bin", dtype="<f4")
        volume_bin = np.fromfile(features_dir / "volume.day.bin", dtype="<f4")
        assert int(open_bin[0]) == 0
        assert len(open_bin) == 4
        assert float(open_bin[1]) == pytest.approx(9.0)
        assert np.isnan(open_bin[2])
        assert float(open_bin[3]) == pytest.approx(11.0)
        assert int(volume_bin[0]) == 0
        assert np.isnan(volume_bin[2])
        assert float(volume_bin[3]) == pytest.approx(1200.0)

    def test_repair_price_provider_rebuilds_broken_close_and_missing_fields(self, tmp_path):
        """close 起始索引越界且 OHLCVA 缺失时，应从 raw_data 重建。"""
        from modules.data.tushare_to_qlib import TushareToQlibConverter

        qlib_root = tmp_path / "cn_data"
        features_dir = qlib_root / "features" / "sz000001"
        features_dir.mkdir(parents=True)
        cal_dir = qlib_root / "calendars"
        cal_dir.mkdir(parents=True)
        (cal_dir / "day.txt").write_text("2026-01-02\n2026-01-05\n2026-01-06\n")

        raw_dir = tmp_path / "raw_data"
        raw_dir.mkdir(parents=True)
        pd.DataFrame(
            {
                "date": [
                    pd.Timestamp("2026-01-02"),
                    pd.Timestamp("2026-01-05"),
                    pd.Timestamp("2026-01-06"),
                ],
                "open": [9.0, 10.0, 11.0],
                "high": [10.2, 11.2, 12.2],
                "low": [8.8, 9.8, 10.8],
                "close": [10.0, 11.0, 12.0],
                "volume": [1000.0, 1100.0, 1200.0],
                "amount": [10000.0, 11500.0, 13000.0],
            }
        ).to_parquet(raw_dir / "sz000001.parquet", index=False)

        np.array([5.0, 99.0], dtype="<f4").tofile(features_dir / "close.day.bin")

        converter = TushareToQlibConverter(tushare_dir=str(tmp_path), qlib_dir=str(qlib_root))
        stats = converter.repair_price_provider()

        close_bin = np.fromfile(features_dir / "close.day.bin", dtype="<f4")
        open_bin = np.fromfile(features_dir / "open.day.bin", dtype="<f4")
        volume_bin = np.fromfile(features_dir / "volume.day.bin", dtype="<f4")

        assert stats["close_rebuilt_files"] == 1
        assert stats["repaired_instruments"] == 1
        assert int(close_bin[0]) == 0
        assert close_bin[1:].tolist() == pytest.approx([10.0, 11.0, 12.0])
        assert int(open_bin[0]) == 0
        assert open_bin[1:].tolist() == pytest.approx([9.0, 10.0, 11.0])
        assert int(volume_bin[0]) == 0
        assert volume_bin[1:].tolist() == pytest.approx([1000.0, 1100.0, 1200.0])

    def test_repair_price_provider_rebuilds_compressed_close_span(self, tmp_path):
        """close.bin 若把停牌日压缩掉，应按 raw_data 跨度重建并补 NaN。"""
        from modules.data.tushare_to_qlib import TushareToQlibConverter

        qlib_root = tmp_path / "cn_data"
        features_dir = qlib_root / "features" / "sz000001"
        features_dir.mkdir(parents=True)
        cal_dir = qlib_root / "calendars"
        cal_dir.mkdir(parents=True)
        (cal_dir / "day.txt").write_text("2026-01-02\n2026-01-05\n2026-01-06\n")

        raw_dir = tmp_path / "raw_data"
        raw_dir.mkdir(parents=True)
        pd.DataFrame(
            {
                "date": [pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-06")],
                "open": [9.0, 11.0],
                "high": [10.2, 12.2],
                "low": [8.8, 10.8],
                "close": [10.0, 12.0],
                "volume": [1000.0, 1200.0],
                "amount": [10000.0, 13000.0],
            }
        ).to_parquet(raw_dir / "sz000001.parquet", index=False)

        np.array([0.0, 10.0, 12.0], dtype="<f4").tofile(features_dir / "close.day.bin")
        np.array([0.0, 9.0, 11.0], dtype="<f4").tofile(features_dir / "open.day.bin")
        np.array([0.0, 10.2, 12.2], dtype="<f4").tofile(features_dir / "high.day.bin")
        np.array([0.0, 8.8, 10.8], dtype="<f4").tofile(features_dir / "low.day.bin")
        np.array([0.0, 1000.0, 1200.0], dtype="<f4").tofile(features_dir / "volume.day.bin")
        np.array([0.0, 10000.0, 13000.0], dtype="<f4").tofile(features_dir / "amount.day.bin")

        converter = TushareToQlibConverter(tushare_dir=str(tmp_path), qlib_dir=str(qlib_root))
        stats = converter.repair_price_provider()

        assert stats["close_rebuilt_files"] == 1
        close_bin = np.fromfile(features_dir / "close.day.bin", dtype="<f4")
        open_bin = np.fromfile(features_dir / "open.day.bin", dtype="<f4")
        assert len(close_bin) == 4
        assert close_bin[1] == pytest.approx(10.0)
        assert np.isnan(close_bin[2])
        assert close_bin[3] == pytest.approx(12.0)
        assert np.isnan(open_bin[2])
        assert open_bin[3] == pytest.approx(11.0)


class TestGetTusharePro:
    """测试 Tushare API 获取"""

    def test_get_tushare_pro_with_token(self):
        """有 token 时成功获取"""
        from modules.data.updater import get_tushare_pro

        with patch.dict(os.environ, {'TUSHARE_TOKEN': 'test_token'}):
            with patch('tushare.pro_api') as mock_pro_api:
                result = get_tushare_pro()
                mock_pro_api.assert_called_once_with('test_token')

    def test_get_tushare_pro_on_exception(self):
        """API 异常时返回 None"""
        from modules.data.updater import get_tushare_pro

        with patch('tushare.pro_api', side_effect=Exception("API Error")):
            result = get_tushare_pro()
            assert result is None

