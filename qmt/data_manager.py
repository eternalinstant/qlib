"""本地数据仓库管理器 — 全 A 股日线/财务/日历的增量维护与导出

用法（MiniQMT 客户端须先启动）:
    C:\\Python312\\python.exe data_manager.py universe              # 解析全 A 股票池并快照
    C:\\Python312\\python.exe data_manager.py backfill              # 全历史回填（重活，可断点重跑）
    C:\\Python312\\python.exe data_manager.py backfill --limit 5    # 小样冒烟
    C:\\Python312\\python.exe data_manager.py update                # 每日增量（默认）
    C:\\Python312\\python.exe data_manager.py update --with-financial
    C:\\Python312\\python.exe data_manager.py status                # 覆盖率总览
    C:\\Python312\\python.exe data_manager.py verify                # 对齐日历查缺口
    C:\\Python312\\python.exe data_manager.py export                # 不下载，仅从 QMT 缓存重写 CSV

幂等：backfill/update 都是「调到目标状态」，重跑只补残差。
"""

import sys
import time
import json
import argparse

import data_config
import data_store
import target_portfolio


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def _today():
    return time.strftime("%Y%m%d")


def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _batches(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


# ── 预检 ──────────────────────────────────────────────────

def preflight():
    """连接数据服务并探活，失败则 [ABORT] 退出"""
    import qmt_data
    qmt_data.init()
    # 探活：取一段交易日历，连不上会返回空
    probe = qmt_data.get_trading_dates_str("SH", "20200101", _today())
    if not probe:
        _log("[ABORT] 数据服务未就绪（交易日历为空）。请确认 MiniQMT 客户端已启动并登录。")
        sys.exit(1)
    _log("[OK] 数据服务已连接")


# ── 股票池 ────────────────────────────────────────────────

def _is_a_share(code):
    """过滤出 A 股（排除指数/基金等）：6/688=SH，0/3=SZ"""
    if "." not in code:
        return False
    num, suffix = code.split(".", 1)
    if suffix == "SH":
        return num.startswith(("60", "68"))
    if suffix == "SZ":
        return num.startswith(("00", "30"))
    return False


def resolve_universe():
    """解析全 A 代码列表，并落 meta/universe.json"""
    import qmt_data
    sectors = qmt_data.get_sector_list() or []
    sector = data_config.UNIVERSE_SECTOR
    if sector not in sectors:
        _log(f"[WARN] 配置板块名 '{sector}' 不在可用板块中；可用候选: "
             f"{[s for s in sectors if 'A股' in s]}")
    codes = qmt_data.get_stock_list(sector) or []
    codes = sorted({target_portfolio.normalize_symbol(c) for c in codes})
    codes = [c for c in codes if _is_a_share(c)]

    data_config.ensure_dirs()
    with open(data_store.universe_path(), "w", encoding="utf-8") as f:
        json.dump({"sector": sector, "resolved_at": _now(),
                   "count": len(codes), "codes": codes},
                  f, ensure_ascii=False, indent=2)
    return codes


def load_universe():
    """读已快照的 universe.json，没有则返回 []"""
    import os
    path = data_store.universe_path()
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f).get("codes", [])


# ── 数据落地 ──────────────────────────────────────────────

def _ingest_daily(codes, start_time, manifest):
    """下载→读回→去重合并落盘→更新 manifest，返回成功落地的只数"""
    import qmt_data
    qmt_data.download_data(codes, period="1d", start_time=start_time)
    bars = qmt_data.get_daily(codes, start_time=start_time,
                              dividend_type=data_config.DIVIDEND_TYPE)
    ok = 0
    for code in codes:
        df = bars.get(code)
        rows = data_store.write_daily(code, df)
        if rows:
            data_store.update_daily_entry(
                manifest, code, data_store.first_date(code),
                data_store.last_date(code), rows, _now())
            ok += 1
    return ok


def _ingest_financial(codes, manifest):
    import qmt_data
    qmt_data.download_financial(codes, table_list=data_config.FINANCIAL_TABLES,
                                start_time=data_config.HISTORY_START)
    fin = qmt_data.get_financial(codes, table_list=data_config.FINANCIAL_TABLES)
    ok = 0
    for code in codes:
        tables = fin.get(code) or {}
        written = []
        for table, df in tables.items():
            if data_store.write_financial(code, table, df):
                written.append(table)
        if written:
            data_store.update_financial_entry(manifest, code, written, _now())
            ok += 1
    return ok


def _refresh_calendar():
    import qmt_data
    for market in data_config.CALENDAR_MARKETS:
        dates = qmt_data.get_trading_dates_str(market, "19900101", _today())
        if dates:
            data_store.write_calendar(market, dates)
            _log(f"[OK] 日历 {market}: {len(dates)} 个交易日")


def _lookback_start():
    """增量起点：最近 INCREMENTAL_LOOKBACK_DAYS 个交易日的起始日"""
    cal = data_store.read_calendar(data_config.CALENDAR_MARKETS[0])
    n = data_config.INCREMENTAL_LOOKBACK_DAYS
    if not cal:
        return ""
    return cal[-min(n, len(cal))]


# ── 子命令 ────────────────────────────────────────────────

def cmd_universe(args):
    preflight()
    codes = resolve_universe()
    _log(f"[OK] 股票池 {data_config.UNIVERSE_SECTOR}: {len(codes)} 只，已写 universe.json")


def cmd_backfill(args):
    preflight()
    _refresh_calendar()
    codes = resolve_universe()
    if args.limit:
        codes = codes[:args.limit]
    manifest = data_store.load_manifest()

    if not args.force:
        done = set(manifest.get("daily", {}).keys())
        todo = [c for c in codes if c not in done]
        _log(f"断点续传：跳过已落地 {len(codes) - len(todo)} 只，待处理 {len(todo)} 只")
    else:
        todo = codes
        _log(f"强制全量：{len(todo)} 只")

    total = 0
    for bi, batch in enumerate(_batches(todo, data_config.DOWNLOAD_BATCH), 1):
        _log(f"--- 日线批 {bi}（{len(batch)} 只）---")
        total += _ingest_daily(batch, data_config.HISTORY_START, manifest)
        manifest["generated_at"] = _now()
        manifest["dividend_type"] = data_config.DIVIDEND_TYPE
        manifest["universe_size"] = len(codes)
        data_store.save_manifest(manifest)   # 每批落盘，断点可续

    if args.with_financial:
        for bi, batch in enumerate(_batches(todo, data_config.DOWNLOAD_BATCH), 1):
            _log(f"--- 财务批 {bi}（{len(batch)} 只）---")
            _ingest_financial(batch, manifest)
            data_store.save_manifest(manifest)

    _log(f"[OK] backfill 完成，日线落地 {total} 只")


def cmd_update(args):
    preflight()
    _refresh_calendar()
    codes = load_universe() or resolve_universe()
    if args.limit:
        codes = codes[:args.limit]
    manifest = data_store.load_manifest()
    start = _lookback_start()
    _log(f"增量起点 {start or '(全history)'}，universe {len(codes)} 只")

    total = 0
    for bi, batch in enumerate(_batches(codes, data_config.DOWNLOAD_BATCH), 1):
        _log(f"--- 日线批 {bi}（{len(batch)} 只）---")
        total += _ingest_daily(batch, start, manifest)
        manifest["generated_at"] = _now()
        manifest["dividend_type"] = data_config.DIVIDEND_TYPE
        manifest["universe_size"] = len(codes)
        data_store.save_manifest(manifest)

    if args.with_financial:
        for bi, batch in enumerate(_batches(codes, data_config.DOWNLOAD_BATCH), 1):
            _log(f"--- 财务批 {bi}（{len(batch)} 只）---")
            _ingest_financial(batch, manifest)
            data_store.save_manifest(manifest)

    _log(f"[OK] update 完成，日线刷新 {total} 只")


def cmd_status(args):
    manifest = data_store.load_manifest()
    universe = load_universe()
    cal = data_store.read_calendar(data_config.CALENDAR_MARKETS[0])
    latest = cal[-1] if cal else None
    s = data_store.coverage_summary(manifest, universe, latest)
    print(f"仓库目录       : {data_config.DATA_DIR}")
    print(f"复权口径       : {manifest.get('dividend_type', data_config.DIVIDEND_TYPE)}")
    print(f"最近更新       : {s['generated_at']}")
    print(f"最近交易日(历) : {latest}")
    print(f"股票池规模     : {s['universe_size']}")
    print(f"已落地只数     : {s['stored']}（池内 {s['in_universe_stored']}）")
    print(f"缺失只数       : {s['missing_count']}")
    print(f"过期只数       : {s['stale_count']}  (last < {latest})")
    print(f"日线总行数     : {s['total_rows']}")
    if s["missing_count"]:
        print(f"  缺失样例: {s['missing'][:10]}")
    if s["stale_count"]:
        print(f"  过期样例: {s['stale'][:10]}")


def cmd_verify(args):
    manifest = data_store.load_manifest()
    cal = data_store.read_calendar(data_config.CALENDAR_MARKETS[0])
    if not cal:
        _log("[WARN] 无交易日历，先跑 update/backfill 刷新日历")
        return
    codes = sorted(manifest.get("daily", {}).keys())
    if args.limit:
        codes = codes[:args.limit]
    flagged = 0
    for code in codes:
        df = data_store.read_daily(code)
        if df is None or len(df) == 0:
            continue
        gaps = data_store.find_gaps(df["date"].astype(str).tolist(), cal)
        if gaps:
            flagged += 1
            _log(f"  {code}: {len(gaps)} 个缺口，样例 {gaps[:5]}")
    _log(f"[OK] verify 完成，{flagged}/{len(codes)} 只有缺口")


def cmd_export(args):
    """不下载，仅从 QMT 本地缓存重新读回并重写 CSV"""
    preflight()
    codes = load_universe() or resolve_universe()
    if args.limit:
        codes = codes[:args.limit]
    manifest = data_store.load_manifest()
    import qmt_data
    total = 0
    for bi, batch in enumerate(_batches(codes, data_config.DOWNLOAD_BATCH), 1):
        _log(f"--- 导出批 {bi}（{len(batch)} 只）---")
        bars = qmt_data.get_daily(batch, start_time=data_config.HISTORY_START,
                                  dividend_type=data_config.DIVIDEND_TYPE)
        for code in batch:
            df = bars.get(code)
            rows = data_store.write_daily(code, df)
            if rows:
                data_store.update_daily_entry(
                    manifest, code, data_store.first_date(code),
                    data_store.last_date(code), rows, _now())
                total += 1
        data_store.save_manifest(manifest)
    _log(f"[OK] export 完成，重写 {total} 只")


def main():
    parser = argparse.ArgumentParser(description="MiniQMT 本地数据仓库管理器")
    sub = parser.add_subparsers(dest="cmd")

    def add_common(p):
        p.add_argument("--limit", type=int, default=0, help="只处理前 N 只（调试/冒烟）")

    p = sub.add_parser("universe"); add_common(p)
    p = sub.add_parser("backfill"); add_common(p)
    p.add_argument("--force", action="store_true", help="忽略 manifest 强制全量重下")
    p.add_argument("--with-financial", action="store_true", help="同时回填财务数据")
    p = sub.add_parser("update"); add_common(p)
    p.add_argument("--with-financial", action="store_true", help="同时刷新财务数据")
    p = sub.add_parser("status"); add_common(p)
    p = sub.add_parser("verify"); add_common(p)
    p = sub.add_parser("export"); add_common(p)

    args = parser.parse_args()
    if args.cmd is None:                       # 无子命令默认 update
        args = parser.parse_args(["update"])
    cmd = args.cmd
    {
        "universe": cmd_universe, "backfill": cmd_backfill, "update": cmd_update,
        "status": cmd_status, "verify": cmd_verify, "export": cmd_export,
    }[cmd](args)


if __name__ == "__main__":
    main()
