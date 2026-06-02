"""测试 MiniQMT 连接"""
import sys

try:
    from xtquant.xttrader import XtQuantTrader
    from xtquant.xttype import StockAccount
    print("[OK] xtquant.xttrader 模块导入成功")
except ImportError as e:
    print(f"[FAIL] xtquant.xttrader 导入失败: {e}")
    sys.exit(1)

try:
    from xtquant import xtdata
    print("[OK] xtquant.xtdata 模块导入成功")
except ImportError as e:
    print(f"[FAIL] xtquant.xtdata 导入失败: {e}")

# 测试行情连接 - 获取一只股票的最新行情
print("\n--- 测试行情连接 ---")
try:
    xtdata.subscribe_quote("000001.SZ")
    import time
    time.sleep(3)
    quote = xtdata.get_full_tick(["000001.SZ"])
    if quote and "000001.SZ" in quote:
        tick = quote["000001.SZ"]
        print(f"[OK] 行情连接成功 - 平安银行(000001.SZ)")
        print(f"     最新价: {tick.get('lastPrice', 'N/A')}")
        print(f"     时间: {tick.get('lastTime', 'N/A')}")
    else:
        print("[WARN] 行情数据返回为空，可能 MiniQMT 行情服务未启动")
        print(f"     返回内容: {quote}")
except Exception as e:
    print(f"[FAIL] 行情连接失败: {e}")
    import traceback
    traceback.print_exc()

# 测试交易连接
print("\n--- 测试交易连接 ---")
try:
    session_id = 123456
    path = "C:\\Users\\Administrator\\国金QMT交易端模拟"
    xt_trader = XtQuantTrader(path, session_id)
    print("[OK] XtQuantTrader 对象创建成功")

    xt_trader.start()
    print("[OK] 交易接口 start() 调用成功")

    print("\n[INFO] 交易连接测试完成")

except Exception as e:
    print(f"[FAIL] 交易连接失败: {e}")
    import traceback
    traceback.print_exc()

print("\n--- 测试完成 ---")
