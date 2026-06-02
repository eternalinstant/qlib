"""实盘调仓配置 — 与行情/交易连接配置 config.py 解耦"""

import os

# ── 目标持仓文件 ──────────────────────────────────────────
# qlib 模型每个交易日导出 target_YYYYMMDD.json 到此目录
TARGET_DIR = r"C:\Users\Administrator\Documents\stock\qmt\targets"

# ── 调仓时刻 ──────────────────────────────────────────────
# Windows 任务计划程序在此时刻拉起 live_trade.py（脚本本身不常驻）
REBALANCE_TIME = "14:30"

# ── 仓位与下单 ────────────────────────────────────────────
DEFAULT_INVESTED_PCT = 0.8     # 总仓位上限（目标文件可覆盖），留 20% 现金缓冲
SLIPPAGE_BPS = 30              # 限价滑点，30 = 0.3%；买高卖低以求成交
ORDER_TIMEOUT_SEC = 30         # 单批挂单等待成交秒数，超时撤单
MAX_RETRIES = 2                # 撤单后按新报价重报次数

# ── 风控 ─────────────────────────────────────────────────
DRY_RUN = True                 # True=只算计划订单不下单
MAX_ORDER_NOTIONAL = 200000    # 单笔订单金额上限（元），超过则中止报警
MAX_TURNOVER = 0.5             # 单次调仓换手率上限（买卖额/总资产），超过则中止报警
KILL_SWITCH_FILE = r"C:\Users\Administrator\Documents\stock\qmt\STOP"  # 此文件存在则拒绝下单

# ── 报告 ─────────────────────────────────────────────────
REPORT_DIR = r"C:\Users\Administrator\Documents\stock\qmt\reports"

# ── 交易时段（A 股，连续竞价）─────────────────────────────
TRADING_WINDOWS = [("09:30", "11:30"), ("13:00", "15:00")]

# ── 整手 ─────────────────────────────────────────────────
LOT_SIZE = 100                 # A 股 100 股整手；目标清零时允许卖零股


def ensure_dirs():
    """确保目标/报告目录存在"""
    for d in (TARGET_DIR, REPORT_DIR):
        os.makedirs(d, exist_ok=True)
