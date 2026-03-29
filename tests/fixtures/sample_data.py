"""
测试数据生成器
提供各类测试用的样本数据
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def create_sample_factor_data(n_dates=5, n_stocks=10):
    """
    创建样本因子数据

    Parameters
    ----------
    n_dates : int
        日期数量
    n_stocks : int
        股票数量

    Returns
    -------
    pd.DataFrame
        MultiIndex (datetime, instrument) 的因子数据
    """
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="D")
    stocks = [f"SZ00000{i}" for i in range(1, n_stocks + 1)]

    index = pd.MultiIndex.from_product(
        [dates, stocks],
        names=["datetime", "instrument"]
    )
    index.names = ["datetime", "instrument"]

    np.random.seed(42)
    n_rows = len(index)

    data = {
        # Alpha 层（5个因子）
        "alpha_roa": np.random.uniform(0.02, 0.20, n_rows),
        "alpha_book_to_price": np.random.uniform(0.5, 2.0, n_rows),
        "alpha_ebit_to_mv": np.random.uniform(0.01, 0.15, n_rows),
        "alpha_ocf_to_ev": np.random.uniform(-0.05, 0.20, n_rows),
        "alpha_retained_earnings": np.random.uniform(1e8, 1e10, n_rows),
        # Risk 层（2个因子）
        "risk_vol_std_20d": np.random.uniform(-0.05, -0.01, n_rows),
        "risk_turnover_rate_f": np.random.uniform(-0.1, -0.01, n_rows),
        # Enhance 层（3个因子）
        "enhance_bbi_momentum": np.random.uniform(-0.05, 0.10, n_rows),
        "enhance_price_pos_52w": np.random.uniform(0.0, 1.0, n_rows),
        "enhance_mom_20d": np.random.uniform(-0.10, 0.15, n_rows),
    }

    return pd.DataFrame(data, index=index)


def create_sample_market_data(n_days=200):
    """
    创建样本市场数据（模拟沪深300）

    Parameters
    ----------
    n_days : int
        天数

    Returns
    -------
    pd.Series
        收盘价序列
    """
    dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
    np.random.seed(42)

    # 模拟价格走势
    returns = np.random.normal(0.0005, 0.015, n_days)
    price = 4000 * np.cumprod(1 + returns)

    return pd.Series(price, index=dates, name="close")


def create_sample_selections():
    """
    创建样本选股数据

    Returns
    -------
    pd.DataFrame
        选股结果
    """
    dates = pd.date_range("2024-01-31", periods=3, freq="ME")
    stocks = [f"SZ00000{i}" for i in range(1, 21)]

    rows = []
    for dt in dates:
        for rank, sym in enumerate(stocks, 1):
            rows.append({
                "date": dt,
                "rank": rank,
                "symbol": sym,
                "score": 0.8 - rank * 0.01
            })

    return pd.DataFrame(rows)
