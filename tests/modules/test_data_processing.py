"""
数据处理模块测试
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import os


class TestTushareToQlibConverter:
    """TushareToQlibConverter 测试"""

    def test_converter_default_paths(self):
        """测试默认路径"""
        from modules.data.tushare_to_qlib import TushareToQlibConverter
        
        converter = TushareToQlibConverter()
        
        assert "tushare" in str(converter.tushare_dir)
        assert "qlib_data" in str(converter.qlib_dir)

    def test_converter_custom_paths(self):
        """测试自定义路径"""
        from modules.data.tushare_to_qlib import TushareToQlibConverter
        
        converter = TushareToQlibConverter(
            tushare_dir="/custom/tushare",
            qlib_dir="/custom/qlib"
        )
        
        assert str(converter.tushare_dir) == "/custom/tushare"
        assert str(converter.qlib_dir) == "/custom/qlib"

    def test_build_adjusted_bins_for_instruments_preserves_calendar_gaps(self, tmp_path):
        """前复权 bin 写入必须按交易日历补 NaN，不能压缩停牌日。"""
        from modules.data.tushare_to_qlib import TushareToQlibConverter

        qlib_dir = tmp_path / "cn_data"
        raw_dir = qlib_dir.parent / "raw_data"
        raw_dir.mkdir(parents=True)
        features_dir = qlib_dir / "features" / "sz000001"
        features_dir.mkdir(parents=True)
        cal_dir = qlib_dir / "calendars"
        cal_dir.mkdir(parents=True)
        (cal_dir / "day.txt").write_text("2026-01-02\n2026-01-05\n2026-01-06\n")

        # 写入 raw_data
        raw_df = pd.DataFrame(
            {
                "date": [pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-06")],
                "close": [10.0, 12.0],
                "open": [9.0, 11.0],
                "high": [10.2, 12.2],
                "low": [8.8, 10.8],
                "volume": [1000.0, 1200.0],
                "amount": [10000.0, 13000.0],
            }
        )
        raw_df.to_parquet(raw_dir / "sz000001.parquet", index=False)

        # 写入 adj_factor (ratio=1.0，价格不变)
        tushare_dir = tmp_path / "tushare"
        tushare_dir.mkdir(parents=True)
        adj_df = pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "000001.SZ"],
                "trade_date": ["20260102", "20260106"],
                "adj_factor": [1.0, 1.0],
            }
        )
        adj_df.to_parquet(tushare_dir / "adj_factor.parquet", index=False)

        converter = TushareToQlibConverter(
            tushare_dir=str(tushare_dir), qlib_dir=str(qlib_dir)
        )
        written = converter.build_adjusted_bins_for_instruments(["sz000001"])

        assert written == 1
        close_bin = np.fromfile(features_dir / "close.day.bin", dtype="<f4")
        volume_bin = np.fromfile(features_dir / "volume.day.bin", dtype="<f4")
        assert int(close_bin[0]) == 0
        assert close_bin[1] == pytest.approx(10.0)
        assert np.isnan(close_bin[2])
        assert close_bin[3] == pytest.approx(12.0)
        assert np.isnan(volume_bin[2])
        assert volume_bin[3] == pytest.approx(1200.0)


class TestTushareDownloader:
    """TushareDownloader 测试"""

    def test_downloader_env_token(self):
        """测试环境变量 Token"""
        from modules.data.tushare_downloader import TushareDownloader

        with patch.dict(os.environ, {"TUSHARE_TOKEN": "env_token"}):
            with patch("tushare.pro_api") as mock_pro_api:
                downloader = TushareDownloader()

        assert downloader.token == "env_token"
        mock_pro_api.assert_called_once_with("env_token")

    def test_downloader_custom_token(self):
        """测试自定义 Token"""
        from modules.data.tushare_downloader import TushareDownloader

        with patch("tushare.pro_api") as mock_pro_api:
            downloader = TushareDownloader(token="custom_token")

        assert downloader.token == "custom_token"
        mock_pro_api.assert_called_once_with("custom_token")

    def test_downloader_default_data_dir(self):
        """测试默认数据目录"""
        from modules.data.tushare_downloader import TushareDownloader

        with patch.dict(os.environ, {"TUSHARE_TOKEN": "env_token"}):
            with patch("tushare.pro_api"):
                downloader = TushareDownloader()

        assert "tushare" in str(downloader.data_dir)

    def test_downloader_max_workers(self):
        """测试并发线程数"""
        from modules.data.tushare_downloader import TushareDownloader

        with patch.dict(os.environ, {"TUSHARE_TOKEN": "env_token"}):
            with patch("tushare.pro_api"):
                downloader = TushareDownloader()

        assert downloader.MAX_WORKERS == 8

    def test_downloader_requires_token(self):
        """未提供 Token 时抛错"""
        from modules.data.tushare_downloader import TushareDownloader

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="TUSHARE_TOKEN"):
                TushareDownloader()


class TestDataUpdater:
    """数据更新器测试"""

    def test_updater_default_path(self):
        """测试默认路径"""
        from modules.data.updater import DataUpdater
        
        updater = DataUpdater()
        
        assert "qlib_data" in str(updater.qlib_data_path)

    def test_updater_custom_path(self, tmp_path):
        """测试自定义路径"""
        from modules.data.updater import DataUpdater

        custom_path = str(tmp_path / "custom_qlib_data")
        updater = DataUpdater(qlib_data_path=custom_path)

        assert str(updater.qlib_data_path) == custom_path


class TestDataFrameHelpers:
    """DataFrame 处理辅助函数测试"""

    def test_standardize_instrument_code(self):
        """测试股票代码标准化"""
        df = pd.DataFrame({
            'ts_code': ['000001.SZ', '600000.SH', '300001.SZ']
        })
        
        df['instrument'] = df['ts_code'].str.lower().str.replace('.', '', regex=False)
        
        expected = ['000001sz', '600000sh', '300001sz']
        assert df['instrument'].tolist() == expected

    def test_parse_trade_date(self):
        """测试交易日期解析"""
        dates = pd.to_datetime(['20240101', '20240102', '20240103'], format='%Y%m%d')
        
        expected = pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03'])
        assert all(dates == expected)

    def test_forward_fill_financial_data(self):
        """测试财务数据前向填充"""
        df = pd.DataFrame({
            'instrument': ['000001sz'] * 4,
            'datetime': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04']),
            'roe': [1.5, np.nan, 1.6, np.nan]
        })
        
        df['roe'] = df['roe'].ffill()
        
        assert df['roe'].tolist() == [1.5, 1.5, 1.6, 1.6]

    def test_merge_price_and_financial_data(self):
        """测试价格数据与财务数据合并"""
        price_df = pd.DataFrame({
            'instrument': ['000001sz', '000001sz'],
            'datetime': pd.to_datetime(['2024-01-01', '2024-01-02']),
            'close': [10.0, 10.5]
        })
        
        financial_df = pd.DataFrame({
            'instrument': ['000001sz', '000001sz'],
            'datetime': pd.to_datetime(['2024-01-01', '2024-01-02']),
            'roe': [1.5, 1.6]
        })
        
        merged = price_df.merge(
            financial_df, 
            on=['instrument', 'datetime'], 
            how='left'
        )
        
        assert len(merged) == 2
        assert 'roe' in merged.columns


class TestDataValidation:
    """数据验证测试"""

    def test_check_missing_data(self):
        """测试缺失数据检查"""
        df = pd.DataFrame({
            'a': [1, 2, np.nan, 4],
            'b': [np.nan, np.nan, np.nan, 4]
        })
        
        missing_ratio = df.isna().mean()
        
        assert missing_ratio['a'] == 0.25
        assert missing_ratio['b'] == 0.75

    def test_check_duplicate_records(self):
        """测试重复记录检查"""
        df = pd.DataFrame({
            'instrument': ['000001sz', '000001sz', '600000sh'],
            'datetime': pd.to_datetime(['2024-01-01', '2024-01-01', '2024-01-02'])
        })
        
        duplicates = df.duplicated(subset=['instrument', 'datetime']).sum()
        
        assert duplicates == 1

    def test_check_data_range(self):
        """测试数据范围检查"""
        df = pd.DataFrame({
            'close': [10.0, -1.0, 15.0],  # 负价格是异常
            'volume': [1000, 2000, 3000]
        })
        
        invalid_close = (df['close'] <= 0).sum()
        
        assert invalid_close == 1

    def test_check_price_range(self):
        """测试价格合理性检查"""
        df = pd.DataFrame({
            'open': [10.0, 1000.0, 5.0],  # 1000 可能是异常值
            'high': [10.5, 1500.0, 5.5],
            'low': [9.5, 800.0, 4.5],
            'close': [10.0, 1200.0, 5.0]
        })
        
        # high >= low 检查
        invalid = (df['high'] < df['low']).sum()
        assert invalid == 0
        
        # 价格不能为负
        invalid = (df[['open', 'high', 'low', 'close']] <= 0).any(axis=1).sum()
        assert invalid == 0


class TestDataTransformation:
    """数据转换测试"""

    def test_calculate_returns(self):
        """测试收益率计算"""
        df = pd.DataFrame({
            'close': [100, 105, 110, 115]
        })
        
        df['return'] = df['close'].pct_change()
        
        expected_returns = [np.nan, 0.05, 0.0476, 0.0455]
        assert np.isclose(df['return'].iloc[1], 0.05, rtol=0.01)
        assert np.isclose(df['return'].iloc[3], 0.045, rtol=0.1)

    def test_calculate_log_returns(self):
        """测试对数收益率计算"""
        df = pd.DataFrame({
            'close': [100, 105, 110]
        })
        
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        assert np.isclose(df['log_return'].iloc[1], np.log(1.05), rtol=0.01)

    def test_rolling_statistics(self):
        """测试滚动统计"""
        df = pd.DataFrame({
            'close': [100, 102, 101, 105, 107, 110, 112]
        })
        
        df['ma5'] = df['close'].rolling(5).mean()
        
        # (100 + 102 + 101 + 105 + 107) / 5 = 103
        assert df['ma5'].iloc[4] == pytest.approx(103.0, rel=0.01)
        # (102 + 101 + 105 + 107 + 110) / 5 = 105
        assert df['ma5'].iloc[5] == pytest.approx(105.0, rel=0.01)

    def test_standardize_factors(self):
        """测试因子标准化"""
        df = pd.DataFrame({
            'factor': [1, 2, 3, 4, 5]
        })
        
        mean = df['factor'].mean()
        std = df['factor'].std()
        df['factor_zscore'] = (df['factor'] - mean) / std
        
        assert df['factor_zscore'].mean() == pytest.approx(0, abs=0.01)
        assert df['factor_zscore'].std() == pytest.approx(1, abs=0.01)
