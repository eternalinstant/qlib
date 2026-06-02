"""交易接口 — 封装 xtquant.xttrader"""

from xtquant.xttrader import XtQuantTrader, XtQuantTraderCallback
from xtquant.xttype import StockAccount
from xtquant import xtconstant as C
from config import QMT_PATH, SESSION_ID, ACCOUNT_ID

_trader = None
_account = None


def _to_dict(obj):
    if obj is None:
        return None
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    # XtAsset/XtPosition/XtOrder 等是 C 扩展对象，无 __dict__；
    # 从 dir() 提取公开、非可调用属性（同时保留干净属性名与 m_ 原始字段）
    out = {}
    for k in dir(obj):
        if k.startswith("_"):
            continue
        try:
            v = getattr(obj, k)
        except Exception:
            continue
        if callable(v):
            continue
        out[k] = v
    return out


class _DefaultCallback(XtQuantTraderCallback):
    def on_disconnected(self):
        print("[qmt_trade] 连接断开")

    def on_stock_order(self, order):
        print(f"[推送] 委托变更: {order.stock_code} {order.order_sysid}")

    def on_stock_trade(self, trade):
        print(f"[推送] 成交: {trade.stock_code} {trade.traded_volume}@{trade.traded_price}")

    def on_order_error(self, order_error):
        print(f"[推送] 委托错误: {order_error.order_id} {order_error.error_msg}")

    def on_cancel_error(self, cancel_error):
        print(f"[推送] 撤单错误: {cancel_error.order_id} {cancel_error.error_msg}")

    def on_account_status(self, status):
        print(f"[推送] 账户状态: {status.account_id} {status.status}")


def get_trader():
    """获取单例 XtQuantTrader 实例"""
    global _trader
    if _trader is None:
        _trader = XtQuantTrader(QMT_PATH, SESSION_ID)
    return _trader


def get_account():
    """获取账户对象，ACCOUNT_ID 为空时自动检测"""
    global _account
    if _account is not None:
        return _account

    trader = get_trader()
    aid = ACCOUNT_ID
    if not aid:
        infos = trader.query_account_infos()
        if infos:
            aid = str(infos[0].account_id)
            print(f"[qmt_trade] 自动检测到账户: {aid}")
        else:
            print("[ERROR] 未检测到账户，请在 config.py 中填写 ACCOUNT_ID")
            return None
    _account = StockAccount(aid)
    return _account


def init():
    """初始化交易连接"""
    trader = get_trader()
    trader.register_callback(_DefaultCallback())
    trader.start()
    connect_result = trader.connect()
    if connect_result == 0:
        print("[qmt_trade] 交易连接成功")
    else:
        print(f"[qmt_trade] 交易连接返回: {connect_result}")

    account = get_account()
    if account:
        trader.subscribe(account)


def disconnect():
    """断开交易连接"""
    global _trader, _account
    if _trader:
        _trader.stop()
        _trader = None
        _account = None
        print("[qmt_trade] 已断开")


# ── 账户查询 ──────────────────────────────────────────────

def get_assets():
    """查询账户资产，返回 dict 或 None"""
    try:
        trader = get_trader()
        account = get_account()
        if not account:
            return None
        asset = trader.query_stock_asset(account)
        return _to_dict(asset)
    except Exception as e:
        print(f"[ERROR] get_assets(): {e}")
        return None


def get_positions():
    """查询持仓列表（过滤零持仓），返回 [dict, ...]"""
    try:
        trader = get_trader()
        account = get_account()
        if not account:
            return []
        positions = trader.query_stock_positions(account)
        result = []
        for p in positions:
            d = _to_dict(p)
            if d.get("volume", 0) > 0:
                result.append(d)
        return result
    except Exception as e:
        print(f"[ERROR] get_positions(): {e}")
        return []


def get_orders(cancelable_only=False):
    """查询委托列表，返回 [dict, ...]"""
    try:
        trader = get_trader()
        account = get_account()
        if not account:
            return []
        orders = trader.query_stock_orders(account, cancelable_only)
        return [_to_dict(o) for o in orders]
    except Exception as e:
        print(f"[ERROR] get_orders(): {e}")
        return []


def get_trades():
    """查询成交列表，返回 [dict, ...]"""
    try:
        trader = get_trader()
        account = get_account()
        if not account:
            return []
        trades = trader.query_stock_trades(account)
        return [_to_dict(t) for t in trades]
    except Exception as e:
        print(f"[ERROR] get_trades(): {e}")
        return []


def get_position(stock_code):
    """查询单只股票持仓"""
    try:
        trader = get_trader()
        account = get_account()
        if not account:
            return None
        pos = trader.query_stock_position(account, stock_code)
        return _to_dict(pos)
    except Exception as e:
        print(f"[ERROR] get_position({stock_code}): {e}")
        return None


# ── 下单 & 撤单 ──────────────────────────────────────────

def _get_market_price_type(stock_code):
    suffix = stock_code.split(".")[-1]
    if suffix == "SZ":
        return C.MARKET_SZ_CONVERT_5_CANCEL
    return C.MARKET_SH_CONVERT_5_CANCEL


def buy(stock_code, volume, price=0, price_type=None):
    """买入股票，返回 order_id (>0 成功, -1 失败)"""
    try:
        trader = get_trader()
        account = get_account()
        if not account:
            return -1
        if price_type is None:
            price_type = C.FIX_PRICE if price > 0 else C.LATEST_PRICE
        return trader.order_stock(
            account, stock_code, C.STOCK_BUY,
            volume, price_type, price,
        )
    except Exception as e:
        print(f"[ERROR] buy({stock_code}): {e}")
        return -1


def sell(stock_code, volume, price=0, price_type=None):
    """卖出股票，返回 order_id (>0 成功, -1 失败)"""
    try:
        trader = get_trader()
        account = get_account()
        if not account:
            return -1
        if price_type is None:
            price_type = C.FIX_PRICE if price > 0 else C.LATEST_PRICE
        return trader.order_stock(
            account, stock_code, C.STOCK_SELL,
            volume, price_type, price,
        )
    except Exception as e:
        print(f"[ERROR] sell({stock_code}): {e}")
        return -1


def cancel_order(order_id):
    """撤单，返回 0 成功"""
    try:
        trader = get_trader()
        account = get_account()
        if not account:
            return -1
        return trader.cancel_order_stock(account, order_id)
    except Exception as e:
        print(f"[ERROR] cancel_order({order_id}): {e}")
        return -1


def buy_limit(stock_code, volume, price):
    """限价买入"""
    return buy(stock_code, volume, price, C.FIX_PRICE)


def sell_limit(stock_code, volume, price):
    """限价卖出"""
    return sell(stock_code, volume, price, C.FIX_PRICE)


def buy_market(stock_code, volume):
    """市价买入（五档即成剩撤）"""
    return buy(stock_code, volume, price_type=_get_market_price_type(stock_code))


def sell_market(stock_code, volume):
    """市价卖出（五档即成剩撤）"""
    return sell(stock_code, volume, price_type=_get_market_price_type(stock_code))
