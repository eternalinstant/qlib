#!/usr/bin/env python3
"""
测试迅投因子看板中因子的数据可用性
"""

import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

from config.config import CONFIG

def get_real_stock_codes(n=10):
    """从instruments文件中获取真实的股票代码"""
    instruments_file = Path(CONFIG.get("paths.qlib_data", "~/code/qlib/data/qlib_data/cn_data")).expanduser() / "instruments" / "all.txt"
    
    if not instruments_file.exists():
        print(f"[ERROR] instruments文件不存在: {instruments_file}")
        return []
    
    stocks = []
    with open(instruments_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 3:
                    stock_code = parts[0]
                    # 只取A股，排除BJ开头（北交所）和市场指数
                    if not stock_code.startswith('BJ') and not stock_code.startswith('market'):
                        stocks.append(stock_code)
                        if len(stocks) >= n:
                            break
    
    return stocks

def test_factor_fields():
    """测试迅投因子看板中的关键因子字段"""
    print("=" * 80)
    print("迅投因子看板 - 数据可用性测试")
    print("=" * 80)
    
    import qlib
    from qlib.config import REG_CN
    from qlib.data import D
    
    # 初始化QLib
    provider_uri = Path(CONFIG.get("paths.qlib_data", "~/code/qlib/data/qlib_data/cn_data")).expanduser()
    print(f"QLib数据路径: {provider_uri}")
    
    if not provider_uri.exists():
        print(f"[ERROR] Qlib 数据目录不存在: {provider_uri}")
        return False
    
    qlib.init(provider_uri=str(provider_uri), region=REG_CN)
    print("[OK] Qlib 初始化成功")
    
    # 获取真实股票代码
    stock_codes = get_real_stock_codes(5)
    if not stock_codes:
        print("[WARN] 无法获取股票代码，尝试使用默认代码")
        stock_codes = ['sh600000', 'sz000001', 'sz000002', 'sz000858', 'sh600036']
    
    print(f"测试股票: {stock_codes}")
    
    # 测试时间范围（最近1年，与迅投看板一致）
    end_date = "2024-12-31"  # 假设数据到2024年底
    start_date = "2024-01-01"  # 近1年
    
    # 因子分类测试
    factor_categories = {
        "价量技术因子": [
            "$close", "$open", "$high", "$low", "$volume", "$turnover_rate",
            "Ref($close, 5)", "Mean($close, 5)", "Mean($close, 20)", "Mean($close, 60)",
            "Std($close, 20)", "Max($close, 20)", "Min($close, 20)",
            "Slope($close, 6)", "Slope($close, 12)", "Slope($close, 24)",
        ],
        "基本面因子": [
            "$total_hldr_eqy_exc_min_int",  # 股东权益合计(不含少数股东权益)
            "$total_profit",  # 利润总额
            "$revenue",  # 营业收入
            "$net_profit",  # 净利润
            "$operating_cash_flow",  # 经营活动产生的现金流量净额
            "$total_mv",  # 总市值
            "$PE", "$PB", "$PS",  # 估值指标
            "$ROE", "$ROA", "$Net_Margin",  # 盈利能力
        ],
        "财务质量因子": [
            "$operating_cash_flow / $total_mv",  # 现金流市值比
            "$operating_cash_flow / $revenue",  # 现金流营收比
            "$net_profit / $total_hldr_eqy_exc_min_int",  # ROE(替代计算)
            "$total_profit / Ref($total_profit, 4) - 1",  # 利润总额增长率(季度)
            "$total_hldr_eqy_exc_min_int / Ref($total_hldr_eqy_exc_min_int, 4) - 1",  # 净资产增长率
        ],
        "技术指标表达式": [
            "($close / Mean($close, 5) - 1)",  # 5日乖离率
            "($close / Mean($close, 20) - 1)",  # 20日乖离率
            "($close / Ref($close, 6) - 1)",  # 6日变动速率
            "($close / Ref($close, 12) - 1)",  # 12日变动速率
            "Mean($turnover_rate, 5) / Mean($turnover_rate, 120)",  # 换手率比率
            "(Mean($close, 5) / Mean($close, 20) - 1)",  # 均线比率
        ]
    }
    
    results = {}
    
    for category, fields in factor_categories.items():
        print(f"\n{'='*60}")
        print(f"测试类别: {category}")
        print(f"{'='*60}")
        
        category_results = {}
        
        for field in fields:
            try:
                # 测试字段/表达式
                df = D.features(stock_codes, [field], start_date, end_date, freq="day")
                
                if df.empty:
                    category_results[field] = {"status": "empty", "shape": df.shape}
                    print(f"  {field:40s} → 数据为空")
                else:
                    # 计算缺失率
                    missing_rate = df.isna().mean().iloc[0]
                    category_results[field] = {
                        "status": "ok", 
                        "shape": df.shape,
                        "missing_rate": float(missing_rate)
                    }
                    print(f"  {field:40s} → 可用 (缺失率: {missing_rate:.1%}, 形状: {df.shape})")
                    
            except Exception as e:
                error_msg = str(e)
                if "not found" in error_msg or "not exist" in error_msg:
                    category_results[field] = {"status": "field_not_found", "error": error_msg}
                    print(f"  {field:40s} → 字段不存在")
                else:
                    category_results[field] = {"status": "error", "error": error_msg}
                    print(f"  {field:40s} → 错误: {error_msg[:50]}...")
        
        results[category] = category_results
    
    # 总结报告
    print(f"\n{'='*80}")
    print("数据可用性总结")
    print(f"{'='*80}")
    
    for category, field_results in results.items():
        total = len(field_results)
        available = sum(1 for r in field_results.values() if r["status"] == "ok")
        field_not_found = sum(1 for r in field_results.values() if r["status"] == "field_not_found")
        errors = sum(1 for r in field_results.values() if r["status"] == "error")
        empty = sum(1 for r in field_results.values() if r["status"] == "empty")
        
        print(f"\n{category}:")
        print(f"  总计: {total}, 可用: {available}, 字段不存在: {field_not_found}, 错误: {errors}, 空数据: {empty}")
        
        # 显示可用的关键因子
        if available > 0:
            print("  可用因子:")
            for field, result in field_results.items():
                if result["status"] == "ok":
                    missing_rate = result.get("missing_rate", 1.0)
                    if missing_rate < 0.8:  # 缺失率低于80%的认为可用
                        print(f"    - {field} (缺失率: {missing_rate:.1%})")
    
    # 检查具体股票的数据目录
    print(f"\n{'='*80}")
    print("数据目录检查")
    print(f"{'='*80}")
    
    features_dir = provider_uri / "features"
    if features_dir.exists():
        # 检查几个股票的数据文件
        sample_stocks = stock_codes[:3] if len(stock_codes) >= 3 else ['sh600000']
        for stock in sample_stocks:
            stock_dir = features_dir / stock
            if stock_dir.exists():
                files = list(stock_dir.glob("*.bin"))
                parquet_files = list(stock_dir.glob("*.parquet"))
                print(f"  {stock}: {len(files)}个.bin文件, {len(parquet_files)}个.parquet文件")
                
                # 检查关键文件
                key_files = ["close.day.bin", "open.day.bin", "volume.day.bin", 
                           "total_mv.day.bin", "net_profit.day.bin", "operating_cash_flow.day.bin"]
                existing = []
                missing = []
                for f in key_files:
                    if (stock_dir / f).exists():
                        existing.append(f.split(".")[0])
                    else:
                        missing.append(f.split(".")[0])
                
                print(f"    存在: {existing[:5]}{'...' if len(existing)>5 else ''}")
                if missing:
                    print(f"    缺失: {missing[:5]}{'...' if len(missing)>5 else ''}")
            else:
                print(f"  {stock}: 目录不存在")
    else:
        print(f"  features目录不存在: {features_dir}")
    
    return True

if __name__ == "__main__":
    test_factor_fields()