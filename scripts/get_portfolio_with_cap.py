#!/usr/bin/env python3
"""
获取持仓股票市值信息
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
    
    print(f"[OK] Qlib 初始化成功")
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

def get_portfolio_with_market_cap(date="2026-02-06"):
    """获取指定日期的持仓股票及市值信息"""
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
    
    # 尝试不同的市值字段
    market_cap_fields = [
        "$market_cap",  # 常见市值字段
        "$total_mv",    # 总市值
        "$mkt_cap",     # 市场资本化
        "$mv",          # 市值缩写
    ]
    
    # 加载市值数据
    from qlib.data import D
    successful_field = None
    df_market_cap = None
    
    for field in market_cap_fields:
        try:
            print(f"\n[尝试加载市值字段: {field}]")
            # 获取持仓日期的市值数据
            # 使用持仓日期的前一天，因为持仓日当天的市值数据可能还没有
            date_before = target_date - pd.Timedelta(days=1)
            
            df_market_cap = load_features_safe(
                symbols, 
                [field],
                start_time=date_before.strftime("%Y-%m-%d"),
                end_time=target_date.strftime("%Y-%m-%d"),
                freq="day"
            )
            
            if df_market_cap.empty:
                print(f"  [WARN] {field} 返回空数据")
                continue
                
            # 获取持仓日期的市值（使用前一天的收盘市值）
            try:
                # 尝试获取目标日期的数据
                cap_data = df_market_cap.xs(target_date, level="datetime")
                successful_field = field
                print(f"  [OK] 成功加载市值数据，字段: {field}")
                break
            except KeyError:
                # 如果目标日期没有数据，尝试前一个交易日
                try:
                    available_dates = df_market_cap.index.get_level_values("datetime").unique()
                    last_available_date = available_dates[-1]
                    cap_data = df_market_cap.xs(last_available_date, level="datetime")
                    successful_field = field
                    print(f"  [OK] 使用最近可用日期: {last_available_date.date()}")
                    break
                except Exception as e:
                    print(f"  [WARN] 获取市值数据失败: {e}")
                    continue
                    
        except Exception as e:
            print(f"  [WARN] {field} 加载失败: {e}")
            continue
    
    if df_market_cap is None or successful_field is None:
        print("\n[ERROR] 无法加载市值数据，尝试加载收盘价计算近似市值")
        # 如果市值字段都不行，尝试用收盘价和总股本估算
        try:
            # 加载收盘价
            df_close = load_features_safe(
                symbols,
                ["$close"],
                start_time=target_date.strftime("%Y-%m-%d"),
                end_time=target_date.strftime("%Y-%m-%d"),
                freq="day"
            )
            
            # 加载总股本（如果可用）
            try:
                df_shares = load_features_safe(
                    symbols,
                    ["$total_share"],  # 总股本
                    start_time=target_date.strftime("%Y-%m-%d"),
                    end_time=target_date.strftime("%Y-%m-%d"),
                    freq="day"
                )
                
                # 计算市值：收盘价 * 总股本
                cap_data = pd.DataFrame(index=symbols)
                for symbol in symbols:
                    try:
                        close_price = df_close.xs(target_date, level="datetime").loc[symbol, "$close"]
                        total_shares = df_shares.xs(target_date, level="datetime").loc[symbol, "$total_share"]
                        cap_data.loc[symbol, "market_cap"] = close_price * total_shares
                    except:
                        cap_data.loc[symbol, "market_cap"] = None
                
                successful_field = "estimated_market_cap"
                print("  [OK] 使用收盘价和总股本估算市值")
                
            except:
                # 如果总股本也不可用，只显示收盘价
                cap_data = pd.DataFrame(index=symbols)
                for symbol in symbols:
                    try:
                        close_price = df_close.xs(target_date, level="datetime").loc[symbol, "$close"]
                        cap_data.loc[symbol, "close_price"] = close_price
                    except:
                        cap_data.loc[symbol, "close_price"] = None
                
                successful_field = "close_price_only"
                print("  [WARN] 只能获取收盘价，无法计算完整市值")
                
        except Exception as e:
            print(f"  [ERROR] 无法获取任何价格数据: {e}")
            cap_data = pd.DataFrame(index=symbols)
            successful_field = "unknown"
    
    # 合并持仓信息和市值数据
    portfolio_df = holdings.copy()
    portfolio_df = portfolio_df.set_index("symbol")
    
    # 添加市值信息
    if successful_field == "estimated_market_cap":
        portfolio_df["market_cap"] = cap_data["market_cap"]
        portfolio_df["market_cap_million"] = portfolio_df["market_cap"] / 1e6  # 转换为百万元
    elif successful_field == "close_price_only":
        portfolio_df["close_price"] = cap_data["close_price"]
    elif successful_field != "unknown":
        portfolio_df["market_cap"] = cap_data[successful_field]
        portfolio_df["market_cap_million"] = portfolio_df["market_cap"] / 1e6  # 转换为百万元
    
    # 计算权重（假设等权重）
    portfolio_df["weight_pct"] = 100.0 / len(portfolio_df)
    
    # 排序
    portfolio_df = portfolio_df.sort_values("rank")
    
    return portfolio_df, successful_field, target_date

def print_portfolio(portfolio_df, field_used, date):
    """打印持仓信息"""
    print(f"\n{'='*60}")
    print(f"  持仓详情 - {date.date()}")
    print(f"{'='*60}")
    
    if field_used == "close_price_only":
        print(f"{'排名':<4} {'股票代码':<10} {'信号得分':<10} {'收盘价(元)':<12} {'权重(%)':<8}")
        print("-" * 50)
        for idx, row in portfolio_df.iterrows():
            close_price = f"{row['close_price']:.2f}" if pd.notna(row['close_price']) else "N/A"
            print(f"{row['rank']:<4} {idx:<10} {row['score']:<10.4f} {close_price:<12} {row['weight_pct']:<8.2f}")
    
    elif field_used == "estimated_market_cap" or "market_cap" in field_used:
        print(f"{'排名':<4} {'股票代码':<10} {'信号得分':<10} {'市值(亿元)':<12} {'权重(%)':<8}")
        print("-" * 50)
        for idx, row in portfolio_df.iterrows():
            market_cap = f"{row['market_cap_million']/100:.2f}" if pd.notna(row.get('market_cap_million')) else "N/A"
            print(f"{row['rank']:<4} {idx:<10} {row['score']:<10.4f} {market_cap:<12} {row['weight_pct']:<8.2f}")
        
        # 计算总市值
        if 'market_cap_million' in portfolio_df.columns:
            total_market_cap = portfolio_df['market_cap_million'].sum() / 100  # 转换为亿元
            avg_market_cap = portfolio_df['market_cap_million'].mean() / 100  # 转换为亿元
            print(f"\n  持仓总市值: {total_market_cap:.2f} 亿元")
            print(f"  平均个股市值: {avg_market_cap:.2f} 亿元")
    
    else:
        print(f"{'排名':<4} {'股票代码':<10} {'信号得分':<10} {'权重(%)':<8}")
        print("-" * 50)
        for idx, row in portfolio_df.iterrows():
            print(f"{row['rank']:<4} {idx:<10} {row['score']:<10.4f} {row['weight_pct']:<8.2f}")
    
    print(f"\n  数据来源: {field_used}")
    print(f"  持仓日期: {date.date()}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="获取持仓股票市值信息")
    parser.add_argument("--date", "-d", default="2026-02-06", 
                       help="持仓日期 (格式: YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    portfolio_df, field_used, date = get_portfolio_with_market_cap(args.date)
    
    if portfolio_df is not None:
        print_portfolio(portfolio_df, field_used, date)
        
        # 保存到CSV文件
        output_file = f"portfolio_{date.date()}.csv"
        portfolio_df.to_csv(output_file)
        print(f"\n[OK] 持仓详情已保存: {output_file}")

if __name__ == "__main__":
    main()