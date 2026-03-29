"""
compute 模块测试
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.compute import compute_layer_score, neutralize_by_industry


class TestComputeLayerScore:
    """测试 compute_layer_score 函数"""

    @pytest.fixture
    def sample_multiindex_df(self):
        """创建测试用 MultiIndex DataFrame"""
        dates = pd.date_range("2024-01-01", "2024-01-10", freq="D")
        instruments = ["SZ000001", "SZ000002", "SZ000003"]
        
        data = {}
        for col in ["alpha_roa", "alpha_book_to_price", "risk_vol"]:
            np.random.seed(42)
            data[col] = np.random.randn(len(dates) * len(instruments))
        
        index = pd.MultiIndex.from_product(
            [dates, instruments], names=["datetime", "instrument"]
        )
        df = pd.DataFrame(data, index=index)
        return df

    def test_batch_rank_single_column(self, sample_multiindex_df):
        """测试单列排名"""
        cols = ["alpha_roa"]
        result = compute_layer_score(sample_multiindex_df, cols)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_multiindex_df)
        # 排名应在 0-1 之间
        assert result.min() >= 0
        assert result.max() <= 1

    def test_batch_rank_multiple_columns(self, sample_multiindex_df):
        """测试多列批量排名（优化后）"""
        cols = ["alpha_roa", "alpha_book_to_price", "risk_vol"]
        result = compute_layer_score(sample_multiindex_df, cols)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_multiindex_df)
        # 多列平均后仍在 0-1 之间
        assert result.min() >= 0
        assert result.max() <= 1

    def test_empty_columns(self, sample_multiindex_df):
        """测试空列名"""
        result = compute_layer_score(sample_multiindex_df, [])
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_multiindex_df)

    def test_non_existent_columns(self, sample_multiindex_df):
        """测试不存在的列"""
        result = compute_layer_score(sample_multiindex_df, ["nonexistent"])
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_multiindex_df)

    def test_prefix_mode(self, sample_multiindex_df):
        """测试 prefix 模式"""
        # prefix 模式需要 factor_cols 参数但可以为列表
        result = compute_layer_score(sample_multiindex_df, factor_cols=[], prefix="alpha")
        
        assert isinstance(result, pd.Series)
        # 应该只包含 alpha 开头的列
        assert len(result) == len(sample_multiindex_df)

    def test_batch_vs_loop_equivalence(self, sample_multiindex_df):
        """验证批量方法和逐列方法结果一致"""
        cols = ["alpha_roa", "alpha_book_to_price"]
        
        # 使用优化后的批量方法
        result_batch = compute_layer_score(sample_multiindex_df, cols)
        
        # 手动逐列计算
        scores = pd.DataFrame(index=sample_multiindex_df.index)
        for col in cols:
            scores[col] = sample_multiindex_df.groupby(level="datetime")[col].transform(
                lambda x: x.rank(pct=True)
            )
        result_loop = scores.mean(axis=1)
        
        # 结果应该非常接近（由于浮点精度可能有微小差异）
        np.testing.assert_allclose(
            result_batch.values, 
            result_loop.values, 
            rtol=1e-10
        )


class TestNeutralizeByIndustry:
    """测试行业中性化函数"""

    @pytest.fixture
    def df_with_industry(self):
        """创建带行业标签的测试数据"""
        dates = pd.date_range("2024-01-01", "2024-01-05", freq="D")
        instruments = ["SZ000001", "SZ000002", "SZ000003"]
        
        data = {
            "alpha_roa": [0.1, 0.2, 0.3, 0.4, 0.5] * 3,
            "risk_vol": [0.5, 0.4, 0.3, 0.2, 0.1] * 3,
        }
        
        index = pd.MultiIndex.from_product(
            [dates, instruments], names=["datetime", "instrument"]
        )
        df = pd.DataFrame(data, index=index)
        return df

    def test_neutralize_basic(self, df_with_industry):
        """基本中性化测试"""
        industry_map = {
            "SZ000001": "银行",
            "SZ000002": "银行",
            "SZ000003": "科技",
        }
        
        result = neutralize_by_industry(df_with_industry, industry_map)
        
        # 同一行业同一日期的因子均值应接近0
        for date in df_with_industry.index.get_level_values("datetime").unique()[:2]:
            bank_data = result.loc[date].loc[
                ["SZ000001", "SZ000002"], "alpha_roa"
            ]
            # 均值应接近0
            assert abs(bank_data.mean()) < 1e-10

    def test_neutralize_with_specific_cols(self, df_with_industry):
        """测试指定列的中性化"""
        industry_map = {
            "SZ000001": "银行",
            "SZ000002": "银行",
            "SZ000003": "科技",
        }
        
        result = neutralize_by_industry(
            df_with_industry, 
            industry_map, 
            factor_cols=["alpha_roa"]
        )
        
        # 只应中性化 alpha_roa，risk_vol 保持不变
        # (此处简化测试)
        assert isinstance(result, pd.DataFrame)

    def test_neutralize_empty_map(self, df_with_industry):
        """测试空行业映射"""
        result = neutralize_by_industry(df_with_industry, {})
        # 应返回原始数据副本
        pd.testing.assert_frame_equal(result, df_with_industry)

    def test_neutralize_empty_df(self):
        """测试空 DataFrame"""
        df = pd.DataFrame(columns=["alpha_roa"])
        df.index = pd.MultiIndex.from_arrays([[], []], names=["datetime", "instrument"])
        
        result = neutralize_by_industry(df, {"SZ000001": "银行"})
        assert len(result) == 0
