"""
共享交易工具层
从 qlib_engine.py 提取，供 QlibBacktestEngine 和 RuleBasedEngine 共用
"""
import numpy as np
import pandas as pd
from decimal import Decimal, ROUND_HALF_UP
from functools import lru_cache
from pathlib import Path

from config.config import CONFIG
from utils.platform import resolve_path


# 涨跌停规则已下沉至 common/price_limit.py（五层迁移）。re-export 兼容旧 import 路径。
from common.price_limit import (  # noqa: E402,F401
    CHINEXT_REFORM_DATE,
    PRICE_LIMIT_TOL,
    round_limit_price,
    get_price_limit_pct,
    get_limit_prices,
    can_buy_at_open,
    can_sell_at_open,
)


# ── 路径 ──────────────────────────────────────────────────────────────────────
# 已下沉至 common/paths.py（五层迁移）。re-export 兼容旧 import 路径。
# 注：load_raw_trade_quotes（仍驻本模块）调用这两个 re-export 名，
# 故 patch("modules.backtest.common.raw_data_root") 仍能拦截。
from common.paths import raw_data_root, raw_data_path_for_instrument  # noqa: E402,F401


# ── 交易日历 ──────────────────────────────────────────────────────────────────
# 已下沉至 common/calendar.py（五层迁移）。re-export 兼容旧 import 路径。
from common.calendar import load_trade_calendar  # noqa: E402,F401


# ── 涨跌停 ────────────────────────────────────────────────────────────────────
# 已下沉至 common/price_limit.py（见文件顶部 re-export 块）。


# ── 交易成本 ──────────────────────────────────────────────────────────────────
# 已下沉至 common/costs.py（五层迁移）。此处 re-export 兼容旧 import 路径。
from common.costs import compute_trade_cost  # noqa: E402,F401


# ── 原始日线行情 ──────────────────────────────────────────────────────────────
# 已上移至 data/quotes.py（五层迁移，根治数据层→引擎层循环依赖）。re-export 兼容旧路径。
from data.quotes import load_raw_trade_quotes  # noqa: E402,F401
