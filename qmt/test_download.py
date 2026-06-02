"""测试下载功能"""
import qmt_data

qmt_data.init()

print("=== 下载日K线 ===")
qmt_data.download_data(["000001.SZ", "600519.SH"], period="1d", start_time="20250101")

print("\n=== 下载5分钟线 ===")
qmt_data.download_data(["000001.SZ"], period="5m", start_time="20260501")

print("\n=== 下载财务数据 ===")
qmt_data.download_financial(["000001.SZ"], table_list=["Balance", "Income"], start_time="20240101")

print("\n=== 验证日K线 ===")
daily = qmt_data.get_daily(["000001.SZ", "600519.SH"], count=5)
for code, df in daily.items():
    print(f"  {code}: {len(df)} 根K线")
    if len(df) > 0:
        print(df.tail(3))

print("\n=== 验证财务数据 ===")
fin = qmt_data.get_financial(["000001.SZ"], table_list=["Balance", "Income"])
for stock, tables in fin.items():
    for table_name, df in tables.items():
        rows = len(df) if df is not None else 0
        print(f"  {stock} {table_name}: {rows} 行")

print("\n完成")
