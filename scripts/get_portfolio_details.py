#!/usr/bin/env python3
"""
获取持仓股票详细信息（名称、市值等）
"""
import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
from pathlib import Path

from config.config import CONFIG

def init_qlib():
    """初始化 Qlib"""
    import qlib
    from qlib.config import REG_CN
    
    os.environ["JOBLIB_START_METHOD"] = "fork"
    provider_uri = Path(CONFIG.get("paths.qlib_data", "~/code/qlib/data/qlib_data/cn_data")).expanduser()
    
    if not provider_uri.exists():
        print(f"[ERROR] Qlib 数据目录不存在: {provider_uri}")
        sys.exit(1)
    
    qlib.init(provider_uri=str(provider_uri), region=REG_CN)
    
    try:
        from qlib.config import C
        C.n_jobs = 1
    except Exception:
        pass
    
    return True

def load_features_safe(instruments, fields, start_time, end_time, freq="day"):
    """安全加载因子数据"""
    from qlib.data import D
    
    if not isinstance(instruments, list):
        inst_list = D.list_instruments(instruments, start_time=start_time, end_time=end_time)
        inst_list = list(inst_list.keys())
    else:
        inst_list = list(instruments)
    
    return D.features(inst_list, fields, start_time, end_time, freq)

def get_stock_names(symbols, date):
    """获取股票名称"""
    try:
        # 尝试加载名称字段
        name_fields = ["$name", "$sec_name", "$stock_name"]
        
        for field in name_fields:
            try:
                df_names = load_features_safe(
                    symbols,
                    [field],
                    start_time=date.strftime("%Y-%m-%d"),
                    end_time=date.strftime("%Y-%m-%d"),
                    freq="day"
                )
                
                if not df_names.empty:
                    # 获取名称数据
                    name_data = {}
                    for symbol in symbols:
                        try:
                            name_value = df_names.xs(date, level="datetime").loc[symbol, field]
                            name_data[symbol] = name_value
                        except:
                            name_data[symbol] = symbol  # 如果获取失败，使用股票代码
                    
                    print(f"[OK] 成功加载股票名称，字段: {field}")
                    return name_data
            except Exception as e:
                continue
        
        print("[WARN] 无法加载股票名称，使用股票代码")
        return {symbol: symbol for symbol in symbols}
        
    except Exception as e:
        print(f"[WARN] 获取股票名称失败: {e}")
        return {symbol: symbol for symbol in symbols}

def get_portfolio_details(date="2026-02-06"):
    """获取指定日期的持仓详细信息"""
    # 初始化 Qlib
    init_qlib()
    
    # 加载选股列表
    selection_csv = Path(CONFIG.get("paths.selections", "~/code/qlib/data/monthly_selections.csv")).expanduser()
    
    if not selection_csv.exists():
        print(f"[ERROR] 选股列表不存在: {selection_csv}")
        return None
    
    df_selection = pd.read_csv(selection_csv, parse_dates=["date"])
    
    # 获取指定日期的持仓
    target_date = pd.Timestamp(date)
    holdings = df_selection[df_selection["date"] == target_date]
    
    if holdings.empty:
        print(f"[WARN] 未找到 {date} 的持仓数据")
        # 尝试找最近的日期
        dates = df_selection["date"].unique()
        target_date = pd.Timestamp(dates[-1])
        holdings = df_selection[df_selection["date"] == target_date]
        print(f"[INFO] 使用最近日期: {target_date.date()}")
    
    symbols = holdings["symbol"].tolist()
    
    print(f"\n{'='*60}")
    print(f"  持仓分析 - {target_date.date()}")
    print(f"{'='*60}")
    print(f"  持仓数量: {len(symbols)} 只股票")
    
    # 获取股票名称
    stock_names = get_stock_names(symbols, target_date)
    
    # 加载市值数据
    market_cap_field = "$total_mv"  # 总市值字段
    
    # 使用持仓日期的前一个交易日获取市值
    date_before = target_date - pd.Timedelta(days=5)  # 多取几天以确保有数据
    df_market_cap = load_features_safe(
        symbols, 
        [market_cap_field],
        start_time=date_before.strftime("%Y-%m-%d"),
        end_time=target_date.strftime("%Y-%m-%d"),
        freq="day"
    )
    
    # 加载收盘价（备用）
    df_close = load_features_safe(
        symbols,
        ["$close"],
        start_time=date_before.strftime("%Y-%m-%d"),
        end_time=target_date.strftime("%Y-%m-%d"),
        freq="day"
    )
    
    # 创建持仓DataFrame
    portfolio_data = []
    
    for symbol in symbols:
        rank = holdings[holdings["symbol"] == symbol]["rank"].values[0]
        score = holdings[holdings["symbol"] == symbol]["score"].values[0]
        stock_name = stock_names.get(symbol, symbol)
        
        # 获取市值
        market_cap = None
        try:
            # 尝试获取目标日期的市值
            cap_series = df_market_cap.xs(target_date, level="datetime")
            market_cap = cap_series.loc[symbol, market_cap_field]
        except:
            try:
                # 尝试获取最近可用日期的市值
                available_dates = df_market_cap.index.get_level_values("datetime").unique()
                if len(available_dates) > 0:
                    last_date = available_dates[-1]
                    cap_series = df_market_cap.xs(last_date, level="datetime")
                    market_cap = cap_series.loc[symbol, market_cap_field]
            except:
                market_cap = None
        
        # 获取收盘价
        close_price = None
        try:
            close_series = df_close.xs(target_date, level="datetime")
            close_price = close_series.loc[symbol, "$close"]
        except:
            try:
                available_dates = df_close.index.get_level_values("datetime").unique()
                if len(available_dates) > 0:
                    last_date = available_dates[-1]
                    close_series = df_close.xs(last_date, level="datetime")
                    close_price = close_series.loc[symbol, "$close"]
            except:
                close_price = None
        
        portfolio_data.append({
            "symbol": symbol,
            "name": stock_name,
            "rank": rank,
            "score": score,
            "market_cap": market_cap,
            "close_price": close_price
        })
    
    portfolio_df = pd.DataFrame(portfolio_data)
    
    # 计算市值（亿元）
    if 'market_cap' in portfolio_df.columns:
        portfolio_df["market_cap_billion"] = portfolio_df["market_cap"] / 1e8  # 转换为亿元
    else:
        portfolio_df["market_cap_billion"] = None
    
    # 计算权重（等权重）
    portfolio_df["weight_pct"] = 100.0 / len(portfolio_df)
    
    # 排序
    portfolio_df = portfolio_df.sort_values("rank")
    
    return portfolio_df, target_date

def print_portfolio_details(portfolio_df, date):
    """打印持仓详细信息"""
    print(f"\n{'='*80}")
    print(f"  持仓详情 - {date.date()}")
    print(f"{'='*80}")
    
    # 检查是否有市值数据
    has_market_cap = portfolio_df["market_cap_billion"].notna().any()
    has_close_price = portfolio_df["close_price"].notna().any()
    
    if has_market_cap and has_close_price:
        print(f"{'排名':<4} {'代码':<10} {'名称':<20} {'信号得分':<10} {'市值(亿元)':<12} {'股价(元)':<10} {'权重(%)':<8}")
        print("-" * 80)
        for _, row in portfolio_df.iterrows():
            market_cap = f"{row['market_cap_billion']:.2f}" if pd.notna(row['market_cap_billion']) else "N/A"
            close_price = f"{row['close_price']:.2f}" if pd.notna(row['close_price']) else "N/A"
            print(f"{row['rank']:<4} {row['symbol']:<10} {row['name'][:18]:<20} {row['score']:<10.4f} {market_cap:<12} {close_price:<10} {row['weight_pct']:<8.2f}")
    elif has_market_cap:
        print(f"{'排名':<4} {'代码':<10} {'名称':<20} {'信号得分':<10} {'市值(亿元)':<12} {'权重(%)':<8}")
        print("-" * 70)
        for _, row in portfolio_df.iterrows():
            market_cap = f"{row['market_cap_billion']:.2f}" if pd.notna(row['market_cap_billion']) else "N/A"
            print(f"{row['rank']:<4} {row['symbol']:<10} {row['name'][:18]:<20} {row['score']:<10.4f} {market_cap:<12} {row['weight_pct']:<8.2f}")
    elif has_close_price:
        print(f"{'排名':<4} {'代码':<10} {'名称':<20} {'信号得分':<10} {'股价(元)':<10} {'权重(%)':<8}")
        print("-" * 70)
        for _, row in portfolio_df.iterrows():
            close_price = f"{row['close_price']:.2f}" if pd.notna(row['close_price']) else "N/A"
            print(f"{row['rank']:<4} {row['symbol']:<10} {row['name'][:18]:<20} {row['score']:<10.4f} {close_price:<10} {row['weight_pct']:<8.2f}")
    else:
        print(f"{'排名':<4} {'代码':<10} {'名称':<20} {'信号得分':<10} {'权重(%)':<8}")
        print("-" * 60)
        for _, row in portfolio_df.iterrows():
            print(f"{row['rank']:<4} {row['symbol']:<10} {row['name'][:18]:<20} {row['score']:<10.4f} {row['weight_pct']:<8.2f}")
    
    # 统计信息
    print(f"\n{'='*80}")
    print("  持仓统计")
    print(f"{'='*80}")
    
    if has_market_cap:
        total_market_cap = portfolio_df["market_cap_billion"].sum()
        avg_market_cap = portfolio_df["market_cap_billion"].mean()
        min_market_cap = portfolio_df["market_cap_billion"].min()
        max_market_cap = portfolio_df["market_cap_billion"].max()
        
        print(f"  持仓总市值: {total_market_cap:.2f} 亿元")
        print(f"  平均个股市值: {avg_market_cap:.2f} 亿元")
        print(f"  最小个股市值: {min_market_cap:.2f} 亿元")
        print(f"  最大个股市值: {max_market_cap:.2f} 亿元")
        
        # 市值分布
        small_cap = (portfolio_df["market_cap_billion"] < 50).sum()
        mid_cap = ((portfolio_df["market_cap_billion"] >= 50) & (portfolio_df["market_cap_billion"] < 200)).sum()
        large_cap = (portfolio_df["market_cap_billion"] >= 200).sum()
        
        print(f"\n  市值分布:")
        print(f"    小盘股 (<50亿): {small_cap} 只 ({small_cap/len(portfolio_df)*100:.1f}%)")
        print(f"    中盘股 (50-200亿): {mid_cap} 只 ({mid_cap/len(portfolio_df)*100:.1f}%)")
        print(f"    大盘股 (≥200亿): {large_cap} 只 ({large_cap/len(portfolio_df)*100:.1f}%)")
    
    if has_close_price:
        avg_price = portfolio_df["close_price"].mean()
        min_price = portfolio_df["close_price"].min()
        max_price = portfolio_df["close_price"].max()
        
        print(f"\n  股价统计:")
        print(f"    平均股价: {avg_price:.2f} 元")
        print(f"    最低股价: {min_price:.2f} 元")
        print(f"    最高股价: {max_price:.2f} 元")
    
    print(f"\n  持仓总数: {len(portfolio_df)} 只股票")
    print(f"  持仓权重: 等权重，每只 {100/len(portfolio_df):.2f}%")
    print(f"  调仓频率: 月度调仓")
    
    # 显示策略配置
    print(f"\n  策略配置:")
    print(f"    因子权重: Alpha层65% + 风控层20% + 增强层15%")
    print(f"    选股数量: Top {len(portfolio_df)}")
    print(f"    仓位控制: 基于沪深300趋势的动态仓位")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="获取持仓股票详细信息")
    parser.add_argument("--date", "-d", default="2026-02-06", 
                       help="持仓日期 (格式: YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    portfolio_df, date = get_portfolio_details(args.date)
    
    if portfolio_df is not None:
        print_portfolio_details(portfolio_df, date)
        
        # 保存到CSV文件
        output_file = f"portfolio_details_{date.date()}.csv"
        portfolio_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n[OK] 持仓详情已保存: {output_file}")

if __name__ == "__main__":
    main()