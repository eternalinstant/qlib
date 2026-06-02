"""实盘调仓执行器 — 读今日目标持仓，对账下单

用法（模拟端必须已启动）:
    C:\\Python312\\python.exe live_trade.py            # 按 live_config.DRY_RUN
    C:\\Python312\\python.exe live_trade.py --live      # 强制真实下单
    C:\\Python312\\python.exe live_trade.py --dry-run   # 强制只算不下单

幂等：把账户「调到目标」。重跑只会补齐残差，目标已达成则零下单。
"""

import os
import sys
import json
import time

import live_config
import target_portfolio
import rebalancer


def _now():
    return time.strftime("%H:%M")


def _in_trading_window():
    now = _now()
    return any(s <= now <= e for s, e in live_config.TRADING_WINDOWS)


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def preflight(force_live):
    """连接并做安全检查，返回 (ok, dry_run)"""
    import qmt_data
    import qmt_trade

    dry_run = live_config.DRY_RUN
    if "--live" in sys.argv:
        dry_run = False
    if "--dry-run" in sys.argv:
        dry_run = True

    if os.path.exists(live_config.KILL_SWITCH_FILE):
        _log(f"[ABORT] kill-switch 文件存在: {live_config.KILL_SWITCH_FILE}")
        return False, dry_run

    qmt_data.init()
    if not qmt_data.is_trading_day("SH"):
        _log("[ABORT] 今天不是交易日")
        return False, dry_run

    if not dry_run and not _in_trading_window():
        _log(f"[ABORT] 当前 {_now()} 不在交易时段 {live_config.TRADING_WINDOWS}")
        return False, dry_run

    qmt_trade.init()
    if qmt_trade.get_account() is None:
        _log("[ABORT] 未检测到交易账户")
        return False, dry_run

    return True, dry_run


def _positions_map():
    import qmt_trade
    out = {}
    for p in qmt_trade.get_positions():
        out[p["stock_code"]] = {
            "volume": int(p.get("volume", 0)),
            "can_use_volume": int(p.get("can_use_volume", 0)),
        }
    return out


def _place(order):
    import qmt_trade
    if order["side"] == "sell":
        return qmt_trade.sell_limit(order["symbol"], order["shares"], order["limit_price"])
    return qmt_trade.buy_limit(order["symbol"], order["shares"], order["limit_price"])


def _wait_and_cancel(order_ids):
    """等待成交，超时撤掉仍可撤的单"""
    import qmt_trade
    deadline = time.time() + live_config.ORDER_TIMEOUT_SEC
    pending = set(order_ids)
    while time.time() < deadline and pending:
        time.sleep(2)
        cancelable = {o["order_id"] for o in qmt_trade.get_orders(cancelable_only=True)}
        pending &= cancelable
    for oid in pending:
        _log(f"  超时撤单 order_id={oid}")
        qmt_trade.cancel_order(oid)
    return len(pending)


def _execute_round(target, dry_run):
    """单轮：抓快照→出计划→下单→等成交。返回 (orders, blocked, warnings, placed)"""
    import qmt_trade

    assets = qmt_trade.get_assets() or {}
    total_asset = float(assets.get("total_asset") or 0)
    available_cash = float(assets.get("cash") or 0)
    positions = _positions_map()

    symbols = sorted({p["symbol"] for p in target["positions"]} | set(positions.keys()))
    quotes, details = rebalancer.gather_market_data(symbols)

    orders, blocked, warnings = rebalancer.plan_orders(
        target, total_asset, available_cash, positions, quotes, details,
    )

    turnover = rebalancer.summarize_turnover(orders, total_asset)
    if turnover > live_config.MAX_TURNOVER:
        warnings.append(
            f"[ABORT] 换手率 {turnover:.2%} 超上限 {live_config.MAX_TURNOVER:.0%}，本轮不下单")
        return orders, blocked, warnings, []

    placed = []
    if dry_run:
        return orders, blocked, warnings, []

    order_ids = []
    for o in orders:                       # orders 已先卖后买
        oid = _place(o)
        rec = {**o, "order_id": oid, "accepted": oid > 0}
        placed.append(rec)
        if oid > 0:
            order_ids.append(oid)
            _log(f"  下单 {o['side']} {o['symbol']} {o['shares']}@{o['limit_price']} -> {oid}")
        else:
            _log(f"  [FAIL] 下单失败 {o['side']} {o['symbol']}")
    if order_ids:
        _wait_and_cancel(order_ids)
    return orders, blocked, warnings, placed


def _write_report(target, dry_run, rounds, blocked, warnings):
    import qmt_trade
    live_config.ensure_dirs()
    final_pos = _positions_map()
    report = {
        "date": target["date"],
        "strategy": target["strategy"],
        "dry_run": dry_run,
        "invested_pct": target["invested_pct"],
        "target": target["positions"],
        "rounds": rounds,
        "blocked": blocked,
        "warnings": warnings,
        "final_positions": final_pos,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    path = os.path.join(live_config.REPORT_DIR, f"exec_{target['date']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    _log(f"报告已写入 {path}")


def main():
    force_live = "--live" in sys.argv
    ok, dry_run = preflight(force_live)
    if not ok:
        sys.exit(1)

    _log(f"模式: {'DRY_RUN（只算不下单）' if dry_run else '实盘下单'}")
    target = target_portfolio.load_today_target()
    _log(f"策略 {target['strategy']}  目标 {len(target['positions'])} 只  仓位 {target['invested_pct']:.0%}")

    rounds = []
    all_blocked, all_warnings = [], []
    for attempt in range(live_config.MAX_RETRIES + 1):
        _log(f"--- 第 {attempt + 1} 轮 ---")
        orders, blocked, warnings, placed = _execute_round(target, dry_run)
        rounds.append({
            "attempt": attempt + 1,
            "orders": orders,
            "placed": placed,
        })
        all_blocked = blocked
        all_warnings.extend(warnings)
        for o in orders:
            _log(f"  计划 {o['side']} {o['symbol']} {o['shares']}@{o['limit_price']}  ({o['reason']})")
        for b in blocked:
            _log(f"  跳过 {b['symbol']}: {b['reason']}")
        for w in warnings:
            _log(f"  注意 {w}")
        if not orders or dry_run:
            break                          # 干跑只跑一轮；无单则已达目标
        time.sleep(2)                      # 让上一轮成交回执落地再重算残差

    _write_report(target, dry_run, rounds, all_blocked, all_warnings)
    _log("调仓结束")


if __name__ == "__main__":
    main()
