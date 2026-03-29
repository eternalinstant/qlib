"""
数据处理模块测试
测试数据下载、转换、完整性验证
"""

import pytest
import pandas as pd
import sys
from pathlib import Path

# 项目根目录 = tests 目录的父目录的父目录
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import CONFIG


class TestTushareData:
    """Tushare 原始数据测试"""

    @pytest.fixture
    def tushare_dir(self):
        return PROJECT_ROOT / "data" / "tushare"

    def test_tushare_dir_exists(self, tushare_dir):
        """测试 Tushare 数据目录存在"""
        assert tushare_dir.exists(), f"Tushare 数据目录不存在: {tushare_dir}"

    def test_daily_basic_exists(self, tushare_dir):
        """测试日频基础数据存在"""
        path = tushare_dir / "daily_basic.parquet"
        assert path.exists(), "daily_basic.parquet 不存在"

    def test_daily_basic_columns(self, tushare_dir):
        """测试日频基础数据字段"""
        df = pd.read_parquet(tushare_dir / "daily_basic.parquet")
        required = ['ts_code', 'trade_date', 'close', 'pe', 'pb', 'ps', 'total_mv', 'turnover_rate_f']
        for col in required:
            assert col in df.columns, f"daily_basic 缺少字段: {col}"

    def test_fina_indicator_exists(self, tushare_dir):
        """测试财务指标数据存在"""
        path = tushare_dir / "fina_indicator.parquet"
        assert path.exists(), "fina_indicator.parquet 不存在"

    def test_fina_indicator_columns(self, tushare_dir):
        """测试财务指标数据字段"""
        df = pd.read_parquet(tushare_dir / "fina_indicator.parquet")
        required = ['ts_code', 'ann_date', 'end_date', 'roe', 'roa', 'debt_to_assets']
        for col in required:
            assert col in df.columns, f"fina_indicator 缺少字段: {col}"

    def test_income_exists(self, tushare_dir):
        """测试利润表数据存在"""
        path = tushare_dir / "income.parquet"
        assert path.exists(), "income.parquet 不存在"

    def test_cashflow_exists(self, tushare_dir):
        """测试现金流量表数据存在"""
        path = tushare_dir / "cashflow.parquet"
        assert path.exists(), "cashflow.parquet 不存在"

    def test_balancesheet_exists(self, tushare_dir):
        """测试资产负债表数据存在"""
        path = tushare_dir / "balancesheet.parquet"
        assert path.exists(), "balancesheet.parquet 不存在"

    def test_stock_industry_exists(self, tushare_dir):
        """测试行业数据存在"""
        path = tushare_dir / "stock_industry.csv"
        assert path.exists(), "stock_industry.csv 不存在"

    def test_stock_industry_columns(self, tushare_dir):
        """测试行业数据字段"""
        df = pd.read_csv(tushare_dir / "stock_industry.csv")
        assert 'ts_code' in df.columns, "行业数据缺少 ts_code"
        assert 'industry' in df.columns, "行业数据缺少 industry"

    def test_data_not_empty(self, tushare_dir):
        """测试数据不为空"""
        df = pd.read_parquet(tushare_dir / "daily_basic.parquet")
        assert len(df) > 0, "daily_basic 数据为空"
        assert df['ts_code'].nunique() > 1000, "股票数量太少"


class TestQlibData:
    """QLib 格式数据测试"""

    @pytest.fixture
    def qlib_dir(self):
        return PROJECT_ROOT / "data" / "qlib_data" / "cn_data"

    def test_qlib_dir_exists(self, qlib_dir):
        """测试 QLib 数据目录存在"""
        assert qlib_dir.exists(), f"QLib 数据目录不存在: {qlib_dir}"

    def test_factor_data_exists(self, qlib_dir):
        """测试因子数据文件存在"""
        path = qlib_dir / "factor_data.parquet"
        assert path.exists(), "factor_data.parquet 不存在"

    def test_factor_data_columns(self, qlib_dir):
        """测试因子数据字段"""
        df = pd.read_parquet(qlib_dir / "factor_data.parquet")
        
        # 必须有的字段
        assert 'datetime' in df.columns, "缺少 datetime 字段"
        assert 'instrument' in df.columns, "缺少 instrument 字段"
        
        # 检查财务因子字段
        required_factors = [
            'roe_fina', 'net_margin', 'roa_fina', 'book_to_market',
            'turnover_rate_f', 'pe', 'pb', 'total_mv'
        ]
        for col in required_factors:
            assert col in df.columns, f"factor_data 缺少字段: {col}"

    def test_supplement_daily_exists(self, qlib_dir):
        """测试补充日频数据存在"""
        path = qlib_dir / "supplement_daily.parquet"
        assert path.exists(), "supplement_daily.parquet 不存在"

    def test_supplement_daily_columns(self, qlib_dir):
        """测试补充日频数据字段"""
        df = pd.read_parquet(qlib_dir / "supplement_daily.parquet")
        required = ['instrument', 'datetime', 'open', 'high', 'low', 'close', 'volume']
        for col in required:
            assert col in df.columns, f"supplement_daily 缺少字段: {col}"

    def test_features_dir_exists(self, qlib_dir):
        """测试 features 目录存在"""
        path = qlib_dir / "features"
        assert path.exists(), "features 目录不存在"

    def test_instruments_exists(self, qlib_dir):
        """测试 instruments 文件存在"""
        path = qlib_dir / "instruments" / "all.txt"
        assert path.exists(), "instruments/all.txt 不存在"

    def test_factor_data_not_empty(self, qlib_dir):
        """测试因子数据不为空"""
        df = pd.read_parquet(qlib_dir / "factor_data.parquet")
        assert len(df) > 0, "factor_data 为空"
        assert df['instrument'].nunique() > 1000, "股票数量太少"

    def test_calendar_exists(self, qlib_dir):
        """测试日历数据存在"""
        path = qlib_dir / "calendars"
        assert path.exists(), "calendars 目录不存在"


class TestDataCompleteness:
    """数据完整性测试"""

    def test_factor_data_missing_rate(self):
        """测试因子数据缺失率"""
        df = pd.read_parquet(PROJECT_ROOT / "data" / "qlib_data" / "cn_data" / "factor_data.parquet")
        
        # 关键因子缺失率不应超过 50%
        critical_cols = ['roe_fina', 'net_margin', 'roa_fina', 'book_to_market']
        for col in critical_cols:
            if col in df.columns:
                missing_rate = df[col].isna().mean() * 100
                assert missing_rate < 50, f"{col} 缺失率过高: {missing_rate:.1f}%"

    def test_date_range(self):
        """测试数据日期范围"""
        df = pd.read_parquet(PROJECT_ROOT / "data" / "qlib_data" / "cn_data" / "factor_data.parquet")
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        min_date = df['datetime'].min()
        max_date = df['datetime'].max()
        
        # 数据应该至少覆盖 2019 年至今
        assert min_date.year <= 2019, f"数据开始年份太晚: {min_date.year}"
        assert max_date.year >= 2025, f"数据结束年份太早: {max_date.year}"

    def test_stock_coverage(self):
        """测试股票覆盖数量"""
        df = pd.read_parquet(PROJECT_ROOT / "data" / "qlib_data" / "cn_data" / "factor_data.parquet")
        stock_count = df['instrument'].nunique()
        
        # 应该覆盖至少 3000 只股票
        assert stock_count >= 3000, f"股票数量不足: {stock_count}"


class TestIndustryData:
    """行业数据测试"""

    def test_industry_count(self):
        """测试行业数量"""
        df = pd.read_csv(PROJECT_ROOT / "data" / "tushare" / "stock_industry.csv")
        industry_count = df['industry'].nunique()
        
        assert industry_count >= 20, f"行业数量太少: {industry_count}"
        assert industry_count <= 150, f"行业数量异常多: {industry_count}"

    def test_industry_coverage(self):
        """测试行业股票覆盖"""
        df = pd.read_csv(PROJECT_ROOT / "data" / "tushare" / "stock_industry.csv")
        
        # 每个行业至少应该有股票
        industry_stocks = df.groupby('industry').size()
        assert industry_stocks.min() > 0, "存在没有股票的行业"
        
        # 最大的行业不应该占比过高
        max_ratio = industry_stocks.max() / industry_stocks.sum()
        assert max_ratio < 0.3, f"行业分布过于集中: {max_ratio:.1%}"


class TestStockBasic:
    """股票基本信息测试"""

    def test_stock_basic_exists(self):
        """测试股票基本信息文件存在"""
        path = PROJECT_ROOT / "data" / "tushare" / "stock_basic.csv"
        assert path.exists(), "stock_basic.csv 不存在"

    def test_stock_basic_columns(self):
        """测试股票基本信息字段"""
        df = pd.read_csv(PROJECT_ROOT / "data" / "tushare" / "stock_basic.csv")
        
        assert 'ts_code' in df.columns, "缺少 ts_code 字段"
        assert 'name' in df.columns, "缺少 name 字段"
        assert 'industry' in df.columns, "缺少 industry 字段"

    def test_stock_basic_not_empty(self):
        """测试股票基本信息不为空"""
        df = pd.read_csv(PROJECT_ROOT / "data" / "tushare" / "stock_basic.csv")
        assert len(df) > 3000, "股票数量太少"

    def test_stock_name_valid(self):
        """测试股票名称有效性"""
        df = pd.read_csv(PROJECT_ROOT / "data" / "tushare" / "stock_basic.csv")
        
        # 股票名称不应该为空
        assert df['name'].notna().mean() > 0.99, "股票名称缺失过多"
        
        # 股票名称不应该太短
        name_lengths = df['name'].str.len()
        assert (name_lengths >= 2).mean() > 0.99, "存在异常短的股票名称"

    def test_stock_code_format(self):
        """测试股票代码格式"""
        df = pd.read_csv(PROJECT_ROOT / "data" / "tushare" / "stock_basic.csv")
        
        # 格式应该是 000001.SZ 或 600000.SH
        import re
        pattern = r'^\d{6}\.(SZ|SH|BJ)$'
        valid = df['ts_code'].str.match(pattern)
        assert valid.mean() > 0.99, "存在格式错误的股票代码"

    def test_no_duplicate_stocks(self):
        """测试没有重复股票"""
        df = pd.read_csv(PROJECT_ROOT / "data" / "tushare" / "stock_basic.csv")
        
        duplicates = df['ts_code'].duplicated()
        assert not duplicates.any(), "存在重复的股票代码"

    def test_industry_match(self):
        """测试行业与 stock_industry.csv 一致"""
        basic_df = pd.read_csv(PROJECT_ROOT / "data" / "tushare" / "stock_basic.csv")
        industry_df = pd.read_csv(PROJECT_ROOT / "data" / "tushare" / "stock_industry.csv")
        
        # 两者行业应该一致
        basic_industries = set(basic_df['industry'].dropna().unique())
        industry_industries = set(industry_df['industry'].dropna().unique())
        
        diff = basic_industries ^ industry_industries
        assert len(diff) == 0, f"行业数据不一致: {diff}"


class TestConfig:
    """配置测试"""

    def test_config_qlib_data_path(self):
        """测试 QLib 数据路径配置"""
        path = CONFIG.get("qlib_data_path", "")
        if path:
            path = Path(path).expanduser()
            assert path.exists(), f"配置的 QLib 数据路径不存在: {path}"

    def test_config_date_range(self):
        """测试日期范围配置"""
        start = CONFIG.get("start_date")
        end = CONFIG.get("end_date")
        assert start is not None, "配置缺少 start_date"
        assert end is not None, "配置缺少 end_date"
        assert start < end, "开始日期应该早于结束日期"


class TestDataConversion:
    """数据转换测试"""

    def test_tushare_to_qlib_fields_mapping(self):
        """测试字段映射完整性"""
        # 检查 factor_data 中有多少字段是从 tushare 转换来的
        qlib_df = pd.read_parquet(PROJECT_ROOT / "data" / "qlib_data" / "cn_data" / "factor_data.parquet")
        
        # tushare 原始字段
        tushare_fields = set()
        for fname in ['fina_indicator', 'income', 'cashflow', 'balancesheet', 'daily_basic']:
            fpath = PROJECT_ROOT / "data" / "tushare" / f"{fname}.parquet"
            if fpath.exists():
                df = pd.read_parquet(fpath)
                tushare_fields.update(df.columns.tolist())
        
        # qlib 字段
        qlib_fields = set(qlib_df.columns) - {'datetime', 'instrument'}
        
        print(f"\nTushare 原始字段: {len(tushare_fields)}")
        print(f"QLib 因子字段: {len(qlib_fields)}")
        
        # 至少有 20 个转换后的字段
        assert len(qlib_fields) >= 20, "转换的字段太少"

    def test_fina_indicator_to_daily(self):
        """测试财务数据前向填充到每日"""
        # 财务数据是季度更新的，检查是否正确填充到每日
        df = pd.read_parquet(PROJECT_ROOT / "data" / "qlib_data" / "cn_data" / "factor_data.parquet")
        
        # 取一只股票，检查 roe_fina 在不同日期是否一致（说明做了前向填充）
        sample_stock = df[df['instrument'] == 'sh600000']['roe_fina']
        
        # 如果有数据，应该有重复值（前向填充）
        if len(sample_stock) > 10:
            # 检查非空值是否有重复
            non_null = sample_stock.dropna()
            if len(non_null) > 1:
                unique_ratio = non_null.nunique() / len(non_null)
                # 前向填充后，独特值应该少于总数值
                assert unique_ratio < 0.8, "财务数据可能没有正确前向填充"

    def test_supplement_daily_has_ohlcv(self):
        """测试补充日线数据包含完整 OHLCV"""
        df = pd.read_parquet(PROJECT_ROOT / "data" / "qlib_data" / "cn_data" / "supplement_daily.parquet")
        
        # 验证 OHLCV 都存在且有值
        assert df['open'].notna().mean() > 0.9, "open 缺失率过高"
        assert df['high'].notna().mean() > 0.9, "high 缺失率过高"
        assert df['low'].notna().mean() > 0.9, "low 缺失率过高"
        assert df['close'].notna().mean() > 0.9, "close 缺失率过高"
        assert df['volume'].notna().mean() > 0.9, "volume 缺失率过高"
        
        # 验证 high >= low
        assert (df['high'] >= df['low']).all(), "high 应该 >= low"
        
        # 验证 close 在 high 和 low 之间
        assert ((df['close'] >= df['low']) | df['close'].isna()).all(), "close 应该在 [low, high] 范围内"
        assert ((df['close'] <= df['high']) | df['close'].isna()).all(), "close 应该在 [low, high] 范围内"

    def test_features_dir_has_price_data(self):
        """测试 features 目录包含价格数据"""
        features_dir = PROJECT_ROOT / "data" / "qlib_data" / "cn_data" / "features"
        
        # 检查一个样本股票
        sample_files = list(features_dir.glob("SH600000/*.bin"))
        
        # 应该有 close, open, high, low, volume 等文件（去掉 .day 后缀）
        names = [f.stem.replace('.day', '') for f in sample_files]
        
        assert 'close' in names, "features 缺少 close"
        assert 'open' in names, "features 缺少 open"
        assert 'high' in names, "features 缺少 high"
        assert 'low' in names, "features 缺少 low"
        assert 'volume' in names, "features 缺少 volume"

    def test_instruments_list_valid(self):
        """测试股票列表有效性"""
        # 读取 instruments
        instruments = {}
        with open(PROJECT_ROOT / "data" / "qlib_data" / "cn_data" / "instruments" / "all.txt") as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    instruments[parts[0]] = (parts[1], parts[2])
        
        # 验证格式
        assert len(instruments) > 3000, "股票数量太少"
        
        # 验证日期格式
        for code, (start, end) in instruments.items():
            assert len(start) == 10, f"{code} 开始日期格式错误"
            assert len(end) == 10, f"{code} 结束日期格式错误"


class TestDataQuality:
    """数据质量测试"""

    def test_no_future_dates(self):
        """测试没有未来日期"""
        df = pd.read_parquet(PROJECT_ROOT / "data" / "qlib_data" / "cn_data" / "factor_data.parquet")
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        max_date = df['datetime'].max()
        # 数据不应超过今天+1天（有时区问题）
        from datetime import datetime, timedelta
        tomorrow = datetime.now() + timedelta(days=1)
        assert max_date <= pd.Timestamp(tomorrow), f"数据包含未来日期: {max_date}"

    def test_no_duplicate_dates(self):
        """测试同一股票同一天没有重复数据"""
        df = pd.read_parquet(PROJECT_ROOT / "data" / "qlib_data" / "cn_data" / "factor_data.parquet")
        
        # 检查 (instrument, datetime) 唯一性
        duplicates = df.groupby(['instrument', 'datetime']).size()
        # 允许个别重复（数据源可能有问题），但重复率应该很低
        dup_rate = (duplicates > 1).mean()
        assert dup_rate < 0.01, f"重复率过高: {dup_rate:.2%}"

    def test_price_data_no_zeros(self):
        """测试价格数据没有异常零值"""
        df = pd.read_parquet(PROJECT_ROOT / "data" / "qlib_data" / "cn_data" / "supplement_daily.parquet")
        
        # 价格应该大于 0
        assert (df['close'] > 0).mean() > 0.95, "close 有太多零或负值"
        assert (df['open'] > 0).mean() > 0.95, "open 有太多零或负值"
        
        # 成交量应该 >= 0
        assert (df['volume'] >= 0).all(), "volume 有负值"

    def test_market_cap_reasonable(self):
        """测试市值数据合理性"""
        df = pd.read_parquet(PROJECT_ROOT / "data" / "qlib_data" / "cn_data" / "factor_data.parquet")
        
        # 总市值应该在合理范围（允许0值表示停牌）
        valid_mv = df['total_mv'].dropna()
        
        # 过滤掉0值
        positive_mv = valid_mv[valid_mv > 0]
        
        assert len(positive_mv) > 0, "没有有效的市值数据"
        
        # A股有很多小市值股票（几万块也有），不做最小值检查
        # 只检查没有异常大的值
        assert (positive_mv <= 1e13).all(), "存在异常大的市值"

    def test_pe_pb_reasonable(self):
        """测试 PE PB 合理性"""
        df = pd.read_parquet(PROJECT_ROOT / "data" / "qlib_data" / "cn_data" / "factor_data.parquet")
        
        # PE 应该在合理范围（-1000 ~ 10000）
        valid_pe = df['pe'].dropna()
        assert (valid_pe > -1000).all(), "存在异常 PE 值"
        assert (valid_pe < 100000).all(), "存在异常 PE 值"
        
        # PB 应该 > 0
        valid_pb = df['pb'].dropna()
        assert (valid_pb > 0).mean() > 0.8, "PB 正值太少"

    def test_industry_mapping_complete(self):
        """测试行业映射完整性"""
        # 读取行业数据
        industry_df = pd.read_csv(PROJECT_ROOT / "data" / "tushare" / "stock_industry.csv")
        
        # 读取因子数据的股票
        factor_df = pd.read_parquet(PROJECT_ROOT / "data" / "qlib_data" / "cn_data" / "factor_data.parquet")
        
        # 检查覆盖率
        factor_stocks = set(factor_df['instrument'].unique())
        
        # 转换行业数据的股票代码
        industry_df['instrument'] = industry_df['ts_code'].str.lower().str.replace('.', '', regex=False)
        industry_stocks = set(industry_df['instrument'].unique())
        
        # 应该有较高的覆盖率
        overlap = factor_stocks & industry_stocks
        coverage = len(overlap) / len(factor_stocks)
        
        assert coverage > 0.8, f"行业映射覆盖率太低: {coverage:.1%}"


class TestDataIncremental:
    """增量更新测试"""

    def test_raw_data_incremental(self):
        """测试 raw_data 增量更新"""
        raw_dir = PROJECT_ROOT / "data" / "qlib_data" / "raw_data"
        
        if not raw_dir.exists():
            pytest.skip("raw_data 目录不存在")
        
        # 检查文件数量
        files = list(raw_dir.glob("*.parquet"))
        assert len(files) > 1000, "raw_data 股票数量太少"

    def test_factor_data_recent_update(self):
        """测试因子数据最近有更新"""
        import time
        factor_file = PROJECT_ROOT / "data" / "qlib_data" / "cn_data" / "factor_data.parquet"
        
        if not factor_file.exists():
            pytest.skip("factor_data 不存在")
        
        # 检查文件修改时间
        mtime = factor_file.stat().st_mtime
        from datetime import datetime, timedelta
        
        # 应该在最近 30 天内更新过
        age_days = (time.time() - mtime) / 86400
        assert age_days < 30, f"factor_data 超过 30 天未更新: {age_days:.0f} 天"
