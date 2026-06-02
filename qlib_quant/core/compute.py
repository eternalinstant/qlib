"""
信号计算共享模块
提供分层因子得分计算的公共函数
"""

import pandas as pd
from typing import Dict, List, Optional


def compute_layer_score(
    df: pd.DataFrame,
    factor_cols: List[str],
    prefix: str = None,
    ir_weights: Optional[Dict[str, float]] = None,
) -> pd.Series:
    """
    计算某一层的综合得分（截面rank标准化）

    Parameters
    ----------
    df : pd.DataFrame
        MultiIndex (datetime, instrument) 或普通 DataFrame
    factor_cols : List[str]
        因子列名列表
    prefix : str, optional
        因子名前缀，如果提供则只选择匹配 prefix_xxx 的列
    ir_weights : dict, optional
        {列名: IR值}，提供时按 |IR| 加权平均，否则等权

    Returns
    -------
    pd.Series
        综合得分，index 同 df
    """
    if prefix:
        cols = [c for c in df.columns if c.startswith(f"{prefix}_")]
    else:
        cols = [c for c in factor_cols if c in df.columns]

    if not cols:
        return pd.Series(0.0, index=df.index)

    if isinstance(df.index, pd.MultiIndex):
        scores = df[cols].groupby(level="datetime").rank(pct=True)
    else:
        scores = df[cols].rank(pct=True)

    if ir_weights:
        weights = pd.Series({c: abs(ir_weights.get(c, 1.0)) for c in cols})
        total = weights.sum()
        if total > 0:
            return (scores * weights).sum(axis=1) / total

    return scores.mean(axis=1)


def neutralize_by_industry(
    df: pd.DataFrame,
    industry_map: Dict[str, str],
    factor_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    行业中性化：对每个因子按行业-日期去均值

    Parameters
    ----------
    df : pd.DataFrame
        MultiIndex (datetime, instrument) 的因子数据
    industry_map : Dict[str, str]
        股票代码→行业映射
    factor_cols : List[str], optional
        要中性化的列名，默认选择 alpha_/risk_/enhance_ 开头的列

    Returns
    -------
    pd.DataFrame
        中性化后的数据（副本）
    """
    if not industry_map:
        return df

    result = df.copy()
    instruments = result.index.get_level_values("instrument")
    industries = instruments.map(lambda x: industry_map.get(x, "unknown"))
    datetimes = result.index.get_level_values("datetime")

    if factor_cols is None:
        factor_cols = [c for c in result.columns
                       if c.startswith(("alpha_", "risk_", "enhance_"))]

    valid_cols = [col for col in factor_cols if col in result.columns]
    if not valid_cols:
        return result

    group_keys = [datetimes, industries]
    means = result[valid_cols].groupby(group_keys).transform("mean")
    result.loc[:, valid_cols] = result[valid_cols] - means

    return result
