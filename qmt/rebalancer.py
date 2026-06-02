"""调仓大脑 — 纯计算，无副作用

plan_orders() 接收已抓好的账户/行情快照，算出计划订单列表。
不直接调 QMT，便于离线干跑和核对。实盘数据抓取见 gather_market_data()。
"""

import math

import live_config


def _round_price(p):
    return round(float(p), 2)


def _floor_lot(shares, lot=None):
    lot = lot or live_config.LOT_SIZE
    return int(math.floor(shares / lot) * lot)


def _quote_price(tick, side):
    """取报价：买用卖一价，卖用买一价，缺失回落最新价"""
    if not tick:
        return 0.0
    last = float(tick.get("lastPrice") or 0)
    if side == "buy":
        asks = tick.get("askPrice") or []
        ask1 = float(asks[0]) if asks and asks[0] else 0.0
        return ask1 or last
    bids = tick.get("bidPrice") or []
    bid1 = float(bids[0]) if bids and bids[0] else 0.0
    return bid1 or last


def _buy_limit_price(tick, up_stop):
    base = _quote_price(tick, "buy")
    if base <= 0:
        return 0.0
    px = base * (1 + live_config.SLIPPAGE_BPS / 10000.0)
    if up_stop and up_stop > 0:
        px = min(px, up_stop)
    return _round_price(px)


def _sell_limit_price(tick, down_stop):
    base = _quote_price(tick, "sell")
    if base <= 0:
        return 0.0
    px = base * (1 - live_config.SLIPPAGE_BPS / 10000.0)
    if down_stop and down_stop > 0:
        px = max(px, down_stop)
    return _round_price(px)


def _is_tradable(tick):
    """无 tick 或最新价<=0 视为停牌/无行情"""
    return bool(tick) and float(tick.get("lastPrice") or 0) > 0


def plan_orders(target, total_asset, available_cash, positions, quotes, details):
    """计算计划订单

    target: load_today_target() 的返回
    total_asset: 账户总资产（元）
    available_cash: 可用资金（元）
    positions: {symbol: {"volume": int, "can_use_volume": int}}
    quotes:    {symbol: tick_dict}
    details:   {symbol: instrument_detail_dict}

    返回 (orders, blocked, warnings)
      orders:  [{symbol, side, shares, limit_price, reason}]  先卖后买已排序
      blocked: [{symbol, reason}]  涨跌停/停牌等被跳过
      warnings: [str]
    """
    blocked = []
    warnings = []

    target_w = {p["symbol"]: p["weight"] for p in target["positions"]}
    invested = target["invested_pct"]

    # 当前持仓 + 目标持仓的全集
    all_syms = set(positions.keys()) | set(target_w.keys())

    sells = []
    buys = []

    for sym in sorted(all_syms):
        tick = quotes.get(sym)
        detail = details.get(sym) or {}
        up_stop = float(detail.get("UpStopPrice") or 0)
        down_stop = float(detail.get("DownStopPrice") or 0)

        cur_vol = int(positions.get(sym, {}).get("volume", 0))
        can_sell = int(positions.get(sym, {}).get("can_use_volume", 0))

        if not _is_tradable(tick):
            if sym in target_w or cur_vol > 0:
                blocked.append({"symbol": sym, "reason": "停牌/无行情，跳过"})
            continue

        ref_price = float(tick.get("lastPrice"))
        weight = target_w.get(sym, 0.0)
        target_value = total_asset * invested * weight
        target_shares = _floor_lot(target_value / ref_price) if weight > 0 else 0

        diff = target_shares - cur_vol

        if diff < 0:                       # 需要卖
            want_sell = -diff
            if target_shares == 0:
                want_sell = cur_vol        # 清仓：含零股一起卖
            sell_shares = min(want_sell, can_sell)
            if sell_shares <= 0:
                if can_sell <= 0 and want_sell > 0:
                    blocked.append({"symbol": sym, "reason": "T+1 无可卖量"})
                continue
            px = _sell_limit_price(tick, down_stop)
            if down_stop and ref_price <= down_stop + 1e-9:
                blocked.append({"symbol": sym, "reason": "跌停，无法卖出"})
                continue
            sells.append({
                "symbol": sym, "side": "sell", "shares": sell_shares,
                "limit_price": px,
                "reason": f"目标{target_shares}/当前{cur_vol}，卖{sell_shares}",
            })

        elif diff > 0:                     # 需要买
            if up_stop and ref_price >= up_stop - 1e-9:
                blocked.append({"symbol": sym, "reason": "涨停，无法买入"})
                continue
            buy_shares = _floor_lot(diff)
            if buy_shares <= 0:
                continue
            px = _buy_limit_price(tick, up_stop)
            if px <= 0:
                blocked.append({"symbol": sym, "reason": "无有效买价"})
                continue
            buys.append({
                "symbol": sym, "side": "buy", "shares": buy_shares,
                "limit_price": px,
                "reason": f"目标{target_shares}/当前{cur_vol}，买{buy_shares}",
            })

    # 买入受可用现金约束：卖出回款当日可用于买入
    est_sell_proceeds = sum(o["shares"] * o["limit_price"] for o in sells)
    buying_power = available_cash + est_sell_proceeds

    accepted_buys = []
    for o in buys:                         # 按需求顺序贪心分配现金
        notional = o["shares"] * o["limit_price"]
        if notional > live_config.MAX_ORDER_NOTIONAL:
            warnings.append(
                f"{o['symbol']} 买单金额 {notional:.0f} 超单笔上限 "
                f"{live_config.MAX_ORDER_NOTIONAL}，缩量")
            max_shares = _floor_lot(live_config.MAX_ORDER_NOTIONAL / o["limit_price"])
            o["shares"] = max_shares
            notional = o["shares"] * o["limit_price"]
            if o["shares"] <= 0:
                continue
        if notional > buying_power:
            affordable = _floor_lot(buying_power / o["limit_price"])
            if affordable <= 0:
                warnings.append(f"{o['symbol']} 现金不足，跳过买入")
                continue
            warnings.append(
                f"{o['symbol']} 现金不足，缩量 {o['shares']}→{affordable}")
            o["shares"] = affordable
            notional = o["shares"] * o["limit_price"]
        buying_power -= notional
        accepted_buys.append(o)

    orders = sells + accepted_buys         # 先卖后买
    return orders, blocked, warnings


def gather_market_data(symbols):
    """实盘抓取行情快照，返回 (quotes, details)。需先 qmt_data.init()"""
    import qmt_data
    quotes = qmt_data.get_ticks(symbols)
    details = {}
    for sym in symbols:
        details[sym] = qmt_data.get_instrument_detail(sym) or {}
    return quotes, details


def summarize_turnover(orders, total_asset):
    """计划订单总成交额 / 总资产，用于换手率风控"""
    notional = sum(o["shares"] * o["limit_price"] for o in orders)
    return notional / total_asset if total_asset > 0 else 0.0
