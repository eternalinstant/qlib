"""
更新 raw_data 目录 (高效版)
使用 8 线程并发下载
"""

import os
import pandas as pd
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).parent.parent

from modules.data.tushare_downloader import TushareDownloader

RAW_DATA_DIR = PROJECT_ROOT / "data" / "qlib_data" / "raw_data"
TUSHARE_DIR = PROJECT_ROOT / "data" / "tushare"


def get_stocks_from_daily_basic():
    """从 daily_basic.parquet 获取股票列表 (只读 ts_code 列)"""
    # 只读取 ts_code 列，速度快很多
    df = pd.read_parquet(TUSHARE_DIR / "daily_basic.parquet", columns=['ts_code'])
    ts_codes = df['ts_code'].unique()
    # 转换: 000001.SZ -> sz000001
    stocks = {}
    for ts_code in ts_codes:
        code, exchange = ts_code.split('.')
        raw_code = exchange.lower() + code
        stocks[raw_code] = ts_code
    return stocks


def get_existing_stocks():
    """获取 raw_data 中已有的股票"""
    return set(f.stem for f in RAW_DATA_DIR.glob("*.parquet"))


def download_batch(pro, ts_codes, start_date, end_date, desc=""):
    """批量下载日线数据"""
    all_data = []
    total = len(ts_codes)

    def download_one(ts_code):
        try:
            df = pro.daily(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                fields='ts_code,trade_date,open,high,low,close,pre_close,vol,amount'
            )
            return df if df is not None and len(df) > 0 else None
        except:
            return None

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(download_one, code): code for code in ts_codes}
        done = 0
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                all_data.append(result)
            done += 1
            if done % 500 == 0:
                print(f"  {desc}: {done}/{total}")

    print(f"  {desc}: 完成 {done}/{total}")
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return None


def merge_to_files(df):
    """将数据合并到 raw_data 文件"""
    if df is None or len(df) == 0:
        return 0

    count = 0
    grouped = df.groupby('ts_code')

    for ts_code, group in grouped:
        code, exchange = ts_code.split('.')
        raw_code = exchange.lower() + code
        file_path = RAW_DATA_DIR / f"{raw_code}.parquet"

        # 格式转换
        group = group.copy()
        group['date'] = pd.to_datetime(group['trade_date'], format='%Y%m%d')
        group = group.rename(columns={'vol': 'volume'})
        group['symbol'] = ts_code
        if 'pre_close' not in group.columns:
            group['pre_close'] = pd.NA
        group = group[['date', 'open', 'high', 'low', 'close', 'pre_close', 'volume', 'amount', 'symbol']]
        group = group.sort_values('date')

        if file_path.exists():
            existing = pd.read_parquet(file_path)
            existing['date'] = pd.to_datetime(existing['date'])
            combined = pd.concat([existing, group], ignore_index=True)
            combined = combined.drop_duplicates(subset=['date'], keep='last')
            combined = combined.sort_values('date')
            combined.to_parquet(file_path, index=False)
        else:
            group.to_parquet(file_path, index=False)
        count += 1

    return count


def main():
    print("="*60)
    print("更新 raw_data (8线程)")
    print("="*60)

    # 获取股票列表
    print("加载股票列表...")
    all_stocks = get_stocks_from_daily_basic()  # {raw_code: ts_code}
    existing = get_existing_stocks()

    print(f"Tushare 股票: {len(all_stocks)}")
    print(f"raw_data 已有: {len(existing)}")

    missing = set(all_stocks.keys()) - existing
    print(f"缺失股票: {len(missing)}")

    # 初始化 Tushare
    import tushare as ts
    token = os.environ.get("TUSHARE_TOKEN")
    if not token:
        raise ValueError("缺少 TUSHARE_TOKEN 环境变量")
    pro = ts.pro_api(token)

    # 1. 更新现有股票 (最近25天)
    if existing:
        print(f"\n[1/2] 更新现有股票 (2026-02-04 ~ 今天)")
        ts_codes = [all_stocks[s] for s in existing if s in all_stocks]
        new_data = download_batch(pro, ts_codes, "20260204", None, "更新")
        if new_data is not None:
            count = merge_to_files(new_data)
            print(f"  更新了 {count} 只股票")

    # 2. 下载缺失股票 (2016至今)
    if missing:
        print(f"\n[2/2] 下载缺失股票 (2016至今)")
        ts_codes = [all_stocks[s] for s in missing]
        # 分批下载避免内存问题
        batch_size = 1000
        for i in range(0, len(ts_codes), batch_size):
            batch = ts_codes[i:i+batch_size]
            print(f"  批次 {i//batch_size + 1}/{(len(ts_codes)-1)//batch_size + 1}")
            new_data = download_batch(pro, batch, "20160101", None, f"批次{i//batch_size + 1}")
            if new_data is not None:
                merge_to_files(new_data)

    # 结果统计
    final = len(list(RAW_DATA_DIR.glob("*.parquet")))
    print(f"\n完成! raw_data: {len(existing)} -> {final} 只股票")


if __name__ == "__main__":
    main()
