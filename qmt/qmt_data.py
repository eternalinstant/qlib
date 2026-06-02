"""行情数据接口 — 封装 xtquant.xtdata"""

from xtquant import xtdata
from config import MINI_QMT_IP, MINI_QMT_PORT, DEFAULT_DIVIDEND_TYPE
import time

VALID_DIVIDEND_TYPES = {"none", "front", "back", "front_ratio", "back_ratio"}


def init():
    """连接 MiniQMT 数据服务"""
    try:
        xtdata.connect(MINI_QMT_IP, MINI_QMT_PORT)
        print("[qmt_data] 数据服务已连接")
    except Exception as e:
        print(f"[qmt_data] 连接失败: {e}")


# ── 实时行情 ──────────────────────────────────────────────

def get_tick(stock_code):
    """获取单只股票最新 tick"""
    try:
        result = xtdata.get_full_tick([stock_code])
        if result and stock_code in result:
            return result[stock_code]
        print(f"[WARN] {stock_code} 无行情数据")
        return None
    except Exception as e:
        print(f"[ERROR] get_tick({stock_code}): {e}")
        return None


def get_ticks(stock_list):
    """获取多只股票最新 tick，返回 {code: dict}"""
    try:
        return xtdata.get_full_tick(stock_list) or {}
    except Exception as e:
        print(f"[ERROR] get_ticks(): {e}")
        return {}


def subscribe_tick(stock_code, callback, period="tick"):
    """订阅实时行情推送，返回订阅 seq"""
    try:
        return xtdata.subscribe_quote(stock_code, period=period, callback=callback)
    except Exception as e:
        print(f"[ERROR] subscribe_tick({stock_code}): {e}")
        return -1


def subscribe_market(code_list, callback):
    """订阅全市场推送，code_list 可以是 ['SH','SZ'] 或具体股票代码"""
    try:
        return xtdata.subscribe_whole_quote(code_list, callback=callback)
    except Exception as e:
        print(f"[ERROR] subscribe_market(): {e}")
        return -1


def unsubscribe(seq):
    """取消订阅"""
    try:
        xtdata.unsubscribe_quote(seq)
    except Exception as e:
        print(f"[ERROR] unsubscribe({seq}): {e}")


# ── 历史 K 线 ─────────────────────────────────────────────

def get_bars(stock_list, period="1d", start_time="", end_time="",
            count=-1, dividend_type=None, fill_data=True):
    """获取历史 K 线

    period: tick, 1m, 5m, 15m, 30m, 60m, 1h, 1d, 1w, 1mon
    start_time / end_time: "YYYYMMDD" 或 "YYYYMMDDHHmmss"
    count: -1 全部, N 为从 end_time 往前取 N 根
    dividend_type: "none", "front", "back" 等
    返回 {stock_code: DataFrame}
    """
    try:
        dt = dividend_type if dividend_type in VALID_DIVIDEND_TYPES else DEFAULT_DIVIDEND_TYPE
        return xtdata.get_market_data_ex(
            field_list=[], stock_list=stock_list, period=period,
            start_time=start_time, end_time=end_time, count=count,
            dividend_type=dt, fill_data=fill_data,
        )
    except Exception as e:
        print(f"[ERROR] get_bars(): {e}")
        return {}


def get_daily(stock_list, start_time="", end_time="", count=-1, dividend_type=None):
    """日 K 线快捷方式"""
    return get_bars(stock_list, period="1d", start_time=start_time,
                   end_time=end_time, count=count, dividend_type=dividend_type)


def get_minutes(stock_list, period="5m", start_time="", end_time="",
                count=-1, dividend_type=None):
    """分钟 K 线快捷方式"""
    return get_bars(stock_list, period=period, start_time=start_time,
                   end_time=end_time, count=count, dividend_type=dividend_type)


# ── 财务数据 ──────────────────────────────────────────────

def get_financial(stock_list, table_list=None, start_time="", end_time=""):
    """获取财务报表数据

    table_list: "Balance", "Income", "CashFlow", "Capital",
                "HolderNum", "Top10Holder", "Top10FlowHolder", "PershareIndex"
    返回 {stock_code: {table_name: DataFrame}}
    """
    try:
        if table_list is None:
            table_list = ["Balance", "Income", "CashFlow"]
        return xtdata.get_financial_data(
            stock_list, table_list, start_time, end_time,
        )
    except Exception as e:
        print(f"[ERROR] get_financial(): {e}")
        return {}


def get_balance_sheet(stock_list, start_time="", end_time=""):
    return get_financial(stock_list, ["Balance"], start_time, end_time)


def get_income_statement(stock_list, start_time="", end_time=""):
    return get_financial(stock_list, ["Income"], start_time, end_time)


def get_cashflow_statement(stock_list, start_time="", end_time=""):
    return get_financial(stock_list, ["CashFlow"], start_time, end_time)


# ── 证券信息 & 板块 ──────────────────────────────────────

def get_instrument_detail(stock_code):
    """获取证券详细信息（名称、交易所、涨跌停价等）"""
    try:
        return xtdata.get_instrument_detail(stock_code)
    except Exception as e:
        print(f"[ERROR] get_instrument_detail({stock_code}): {e}")
        return None


def get_stock_list(sector_name):
    """获取板块成分股代码列表"""
    try:
        return xtdata.get_stock_list_in_sector(sector_name)
    except Exception as e:
        print(f"[ERROR] get_stock_list({sector_name}): {e}")
        return []


def get_sector_list():
    """获取所有可用板块名称"""
    try:
        return xtdata.get_sector_list()
    except Exception as e:
        print(f"[ERROR] get_sector_list(): {e}")
        return []


def get_index_weights(index_code):
    """获取指数成分权重"""
    try:
        return xtdata.get_index_weight(index_code)
    except Exception as e:
        print(f"[ERROR] get_index_weights({index_code}): {e}")
        return {}


# ── 交易日历 ─────────────────────────────────────────────

def get_trading_dates(market="SH", start_time="", end_time=""):
    """获取交易日列表（毫秒时间戳）"""
    try:
        return xtdata.get_trading_dates(market, start_time, end_time)
    except Exception as e:
        print(f"[ERROR] get_trading_dates(): {e}")
        return []


def get_trading_dates_str(market="SH", start_time="", end_time=""):
    """获取交易日列表，返回 ['YYYYMMDD', ...]"""
    dates = get_trading_dates(market, start_time, end_time)
    return [time.strftime("%Y%m%d", time.localtime(d / 1000)) for d in dates]


def is_trading_day(market="SH"):
    """判断今天是否为交易日"""
    try:
        today = time.strftime("%Y%m%d")
        dates = get_trading_dates_str(market, today, today)
        return len(dates) > 0
    except Exception:
        return False


# ── 数据下载 ──────────────────────────────────────────────

def _download_callback(data):
    """默认下载进度回调"""
    msg = data.get("message", "")
    finished = data.get("finished", 0)
    total = data.get("total", 0)
    print(f"  [{finished}/{total}] {msg}")


def download_data(stock_list, period="1d", start_time="", end_time=""):
    """批量下载历史数据到本地（阻塞，带进度回调）"""
    try:
        xtdata.download_history_data2(
            stock_list, period, start_time, end_time, callback=_download_callback,
        )
        print(f"[OK] 已下载 {len(stock_list)} 只股票 {period} 数据")
    except Exception as e:
        print(f"[ERROR] download_data(): {e}")


def download_financial(stock_list, table_list=None, start_time="", end_time=""):
    """下载财务数据到本地（阻塞，带进度回调）"""
    try:
        if table_list is None:
            table_list = ["Balance", "Income", "CashFlow"]
        for table in table_list:
            xtdata.download_history_data2(
                stock_list, table, start_time, end_time, callback=_download_callback,
            )
        print(f"[OK] 已下载 {len(stock_list)} 只股票财务数据")
    except Exception as e:
        print(f"[ERROR] download_financial(): {e}")
