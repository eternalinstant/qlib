"""演示脚本 — 逐项测试所有数据接口

运行: C:\\Python312\\python.exe demo.py
"""

import time


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def ok(msg):
    print(f"  [OK] {msg}")


def fail(msg):
    print(f"  [FAIL] {msg}")


def warn(msg):
    print(f"  [WARN] {msg}")


# ── 1. 初始化连接 ─────────────────────────────────────────

section("1. 初始化连接")

try:
    import qmt_data
    import qmt_trade

    qmt_data.init()
    ok("数据服务连接")
except Exception as e:
    fail(f"数据服务连接失败: {e}")

try:
    qmt_trade.init()
    ok("交易服务连接")
except Exception as e:
    warn(f"交易服务连接失败（脱机模式可能不可用）: {e}")

# ── 2. 实时行情 ────────────────────────────────────────────

section("2. 实时行情")

try:
    tick = qmt_data.get_tick("000001.SZ")
    if tick:
        ok(f"平安银行 最新价: {tick.get('lastPrice')}, 开盘: {tick.get('open')}, 最高: {tick.get('high')}, 最低: {tick.get('low')}")
    else:
        warn("无行情数据（非交易时段可能返回空）")
except Exception as e:
    fail(str(e))

try:
    ticks = qmt_data.get_ticks(["000001.SZ", "600519.SH", "000300.SH"])
    for code, t in ticks.items():
        if t:
            print(f"  {code}: 最新价 {t.get('lastPrice')}")
    ok(f"批量行情: 获取到 {len(ticks)} 只股票")
except Exception as e:
    fail(str(e))

# ── 3. 历史 K 线（日K）─────────────────────────────────────

section("3. 历史日K线")

try:
    daily = qmt_data.get_daily(["000001.SZ", "600519.SH"], count=10)
    for code, df in daily.items():
        if df is not None and len(df) > 0:
            ok(f"{code}: {len(df)} 根K线, 日期范围 {df.index[0]} ~ {df.index[-1]}")
            print(f"    最新收盘: {df['close'].iloc[-1]}")
        else:
            warn(f"{code}: 无数据")
except Exception as e:
    fail(str(e))

# ── 4. 历史 K 线（分钟）─────────────────────────────────────

section("4. 历史分钟K线")

try:
    min5 = qmt_data.get_minutes(["000001.SZ"], period="5m", count=20)
    for code, df in min5.items():
        if df is not None and len(df) > 0:
            ok(f"{code} 5分钟线: {len(df)} 根")
            print(f"    最近5根:\n{df.tail(5).to_string()}")
        else:
            warn(f"{code}: 无分钟数据（可能未下载）")
except Exception as e:
    fail(str(e))

# ── 5. 财务数据 ─────────────────────────────────────────────

section("5. 财务数据")

try:
    fin = qmt_data.get_financial(["000001.SZ"], table_list=["Balance", "Income"])
    for stock, tables in fin.items():
        for table_name, df in tables.items():
            if df is not None and len(df) > 0:
                ok(f"{stock} {table_name}: {len(df)} 行")
            else:
                warn(f"{stock} {table_name}: 无数据")
except Exception as e:
    fail(str(e))

# ── 6. 证券详细信息 ─────────────────────────────────────────

section("6. 证券详细信息")

try:
    detail = qmt_data.get_instrument_detail("000001.SZ")
    if detail:
        name = detail.get("InstrumentName", "N/A")
        pre_close = detail.get("PreClose", "N/A")
        up_stop = detail.get("UpStopPrice", "N/A")
        down_stop = detail.get("DownStopPrice", "N/A")
        ok(f"名称: {name}, 前收: {pre_close}, 涨停: {up_stop}, 跌停: {down_stop}")
    else:
        warn("无详细信息")
except Exception as e:
    fail(str(e))

# ── 7. 板块成分股 ───────────────────────────────────────────

section("7. 板块成分股")

try:
    stocks = qmt_data.get_stock_list("沪深A股")
    ok(f"沪深A股数量: {len(stocks)}")
    print(f"    前5只: {stocks[:5]}")
except Exception as e:
    fail(str(e))

try:
    hs300 = qmt_data.get_stock_list("沪深300")
    ok(f"沪深300成分股数量: {len(hs300)}")
except Exception as e:
    warn(f"沪深300获取失败: {e}")

# ── 8. 交易日历 ─────────────────────────────────────────────

section("8. 交易日历")

try:
    today = time.strftime("%Y%m%d")
    trading = qmt_data.is_trading_day("SH")
    ok(f"今天 ({today}) 是交易日: {trading}")

    dates = qmt_data.get_trading_dates_str("SH", "20260101", today)
    ok(f"2026年至今交易日数量: {len(dates)}")
    if dates:
        print(f"    最近5个交易日: {dates[-5:]}")
except Exception as e:
    fail(str(e))

# ── 9. 账户信息 ─────────────────────────────────────────────

section("9. 账户信息")

try:
    assets = qmt_trade.get_assets()
    if assets:
        ok(f"可用资金: {assets.get('cash')}, 总资产: {assets.get('total_asset')}, 市值: {assets.get('market_value')}")
    else:
        warn("资产查询返回空（脱机模式可能无数据）")
except Exception as e:
    warn(f"资产查询失败: {e}")

try:
    positions = qmt_trade.get_positions()
    ok(f"持仓数量: {len(positions)}")
    for p in positions[:5]:
        print(f"    {p.get('stock_code')} {p.get('instrument_name')} 持仓:{p.get('volume')} 盈亏:{p.get('profit_rate', 'N/A')}")
except Exception as e:
    warn(f"持仓查询失败: {e}")

try:
    orders = qmt_trade.get_orders()
    ok(f"委托数量: {len(orders)}")
except Exception as e:
    warn(f"委托查询失败: {e}")

try:
    trades = qmt_trade.get_trades()
    ok(f"成交数量: {len(trades)}")
except Exception as e:
    warn(f"成交查询失败: {e}")

# ── 10. 数据下载（默认注释掉）─────────────────────────────────

section("10. 数据下载（已跳过，取消注释后运行）")

# print("正在下载 000001.SZ 1分钟数据...")
# qmt_data.download_data(["000001.SZ"], period="1m", start_time="20250101")
# ok("下载完成")

print("\n  [跳过] 如需下载数据，请编辑 demo.py 取消第10部分注释")

# ── 完成 ────────────────────────────────────────────────────

section("测试完成")
print("所有接口测试完毕。")
